"""Hosted embedding provider interface and bounded batch orchestration."""

import logging
import math
import os
import random
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Base class for safe, provider-independent embedding failures."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        category: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.category = category or code
        self.retry_after_seconds = retry_after_seconds


class EmbeddingConfigurationError(EmbeddingError):
    """Raised for missing or invalid embedding configuration."""


class EmbeddingProviderError(EmbeddingError):
    """Raised for a non-retryable or exhausted hosted provider failure."""


class EmbeddingRetryableError(EmbeddingError):
    """Raised for a transient provider failure that the service may retry."""


class EmbeddingRateLimitError(EmbeddingRetryableError):
    """The hosted provider rate limited a request."""


class EmbeddingTimeoutError(EmbeddingRetryableError):
    """The hosted provider request timed out or could not connect."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when the provider output is incompatible with configured pgvector size."""


class EmbeddingSettings(BaseModel):
    """Backend-only embedding provider settings."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    api_key: SecretStr | None = None
    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", min_length=1, max_length=200
    )
    dimensions: int = Field(default=384, ge=1, le=3072)
    batch_size: int = Field(default=64, ge=1, le=64)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_retries: int = Field(default=3, ge=0, le=5)
    retry_base_delay_seconds: float = Field(default=0.5, ge=0, le=10)

    @model_validator(mode="after")
    def validate_provider_dimensions(self) -> "EmbeddingSettings":
        """Validate fixed local dimensions while keeping the hosted option compatible."""
        if self.provider == "sentence_transformers":
            if self.dimensions != 384:
                raise ValueError("sentence-transformers/all-MiniLM-L6-v2 requires 384 dimensions")
            if self.api_key is not None:
                raise ValueError("The local embedding provider does not accept API keys.")
        if self.provider == "openai" and self.api_key is None:
            raise ValueError("The selected embedding provider requires a key.")
        return self

    @property
    def model_version(self) -> str:
        """Stable profile key used to prevent mixing model or dimension versions."""
        return f"{self.provider}:{self.model}:dimensions={self.dimensions}"

    @property
    def storage_profile(self) -> Literal["local", "legacy"]:
        """Use an additive 384-dimension column for local vectors, preserving legacy data."""
        return "local" if self.provider == "sentence_transformers" else "legacy"


def _load_root_environment() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


def _format_seconds(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def _retry_after_seconds(error: RateLimitError) -> float | None:
    """Read only numeric timing information from standard provider response headers."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            value = float(retry_after_ms) / 1000
            return value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            return None
    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        value = float(retry_after)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(retry_after))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            value = (retry_at - datetime.now(timezone.utc)).total_seconds()
            return max(value, 0.0) if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            return None


def _classify_rate_limit(error: RateLimitError) -> tuple[str, float | None]:
    """Classify known 429 categories without retaining or logging provider messages."""
    body = getattr(error, "body", None)
    error_body = body.get("error") if isinstance(body, dict) else None
    code = error_body.get("code") if isinstance(error_body, dict) else None
    error_type = error_body.get("type") if isinstance(error_body, dict) else None
    code = code.lower() if isinstance(code, str) else ""
    error_type = error_type.lower() if isinstance(error_type, str) else ""
    if code in {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active"}:
        return "quota_exhausted", _retry_after_seconds(error)
    if code == "rate_limit_exceeded" or error_type in {"requests", "tokens", "rate_limit"}:
        return "transient_rate_limit", _retry_after_seconds(error)
    # A 429 without quota markers is treated as transient for backwards compatibility,
    # but its logs explicitly show that the subtype was not supplied by the provider.
    if not code and not error_type:
        return "rate_limit_unspecified", _retry_after_seconds(error)
    return "unknown_429", _retry_after_seconds(error)


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    """Load embedding settings; local inference does not require a provider key."""
    _load_root_environment()
    try:
        provider = os.getenv("PAGEGLANCE_EMBEDDING_PROVIDER", "sentence_transformers").strip()
        if provider == "sentence_transformers":
            model = os.getenv(
                "PAGEGLANCE_LOCAL_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ).strip()
            dimensions = 384
            api_key = None
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise EmbeddingConfigurationError(
                    "embedding_configuration_error",
                    "OPENAI_API_KEY is required when PAGEGLANCE_EMBEDDING_PROVIDER=openai.",
                )
            model = os.getenv("PAGEGLANCE_EMBEDDING_MODEL", "text-embedding-3-small").strip()
            dimensions = int(os.getenv("PAGEGLANCE_EMBEDDING_DIMENSIONS", "1536"))
        else:
            raise EmbeddingConfigurationError(
                "embedding_configuration_error",
                "PAGEGLANCE_EMBEDDING_PROVIDER must be sentence_transformers or openai.",
            )
        return EmbeddingSettings(
            provider=provider,
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            batch_size=int(os.getenv("PAGEGLANCE_EMBEDDING_BATCH_SIZE", "64")),
            timeout_seconds=float(os.getenv("PAGEGLANCE_EMBEDDING_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("PAGEGLANCE_EMBEDDING_MAX_RETRIES", "3")),
        )
    except EmbeddingConfigurationError:
        raise
    except (ValueError, ValidationError) as error:
        raise EmbeddingConfigurationError(
            "embedding_configuration_error",
            "Embedding settings are invalid; check the PAGEGLANCE_EMBEDDING_* values.",
        ) from error


class EmbeddingProvider(Protocol):
    """Provider-independent contract for embedding ordered batches of text."""

    @property
    def model_version(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def storage_profile(self) -> Literal["local", "legacy"]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    """OpenAI embeddings adapter using the official Python SDK."""

    def __init__(self, settings: EmbeddingSettings, client: OpenAI | None = None) -> None:
        self.settings = settings
        if settings.provider != "openai":
            raise EmbeddingConfigurationError(
                "embedding_configuration_error",
                "The OpenAI adapter requires PAGEGLANCE_EMBEDDING_PROVIDER=openai.",
            )
        if settings.api_key is None:
            raise EmbeddingConfigurationError(
                "embedding_configuration_error", "The selected embedding provider requires a key."
            )
        self.client = client or OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    @property
    def model_version(self) -> str:
        return self.settings.model_version

    @property
    def dimensions(self) -> int:
        return self.settings.dimensions

    @property
    def storage_profile(self) -> Literal["local", "legacy"]:
        return "legacy"

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.settings.model,
                input=list(texts),
                dimensions=self.settings.dimensions,
                encoding_format="float",
            )
        except RateLimitError as error:
            category, retry_after = _classify_rate_limit(error)
            logger.warning(
                "embedding_provider_error status_code=%d category=%s retry_after_seconds=%s",
                error.status_code,
                category,
                _format_seconds(retry_after),
            )
            if category == "quota_exhausted":
                raise EmbeddingProviderError(
                    "embedding_quota_exhausted",
                    "The embedding provider quota or billing limit is exhausted.",
                    status_code=error.status_code,
                    category=category,
                    retry_after_seconds=retry_after,
                ) from None
            if category == "unknown_429":
                raise EmbeddingProviderError(
                    "embedding_provider_429_unknown",
                    "The embedding provider returned an unclassified rate-limit response.",
                    status_code=error.status_code,
                    category=category,
                ) from None
            raise EmbeddingRateLimitError(
                "embedding_rate_limited",
                "The embedding provider rate limited the request.",
                status_code=error.status_code,
                category=category,
                retry_after_seconds=retry_after,
            ) from None
        except (APITimeoutError, APIConnectionError):
            raise EmbeddingTimeoutError(
                "embedding_timeout", "The embedding provider request timed out or disconnected."
            ) from None
        except APIStatusError as error:
            transient = error.status_code in {408, 409} or error.status_code >= 500
            category = "provider_transient_http" if transient else "provider_http_error"
            logger.warning(
                "embedding_provider_error status_code=%d category=%s retry_after_seconds=none",
                error.status_code,
                category,
            )
            if transient:
                raise EmbeddingRetryableError(
                    "embedding_provider_unavailable",
                    "The embedding provider is temporarily unavailable.",
                    status_code=error.status_code,
                    category="provider_transient_http",
                ) from None
            raise EmbeddingProviderError(
                "embedding_provider_error",
                "The embedding provider rejected the request.",
                status_code=error.status_code,
                category="provider_http_error",
            ) from None
        except Exception:
            raise EmbeddingProviderError(
                "embedding_provider_error", "The embedding provider request failed."
            ) from None

        rows = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in rows] != list(range(len(texts))):
            raise EmbeddingProviderError(
                "embedding_response_invalid",
                "The embedding provider returned invalid result order.",
            )
        return [list(item.embedding) for item in rows]


class SentenceTransformerEmbeddingProvider:
    """CPU-based local embeddings from a Sentence Transformers model."""

    def __init__(self, settings: EmbeddingSettings, model: object | None = None) -> None:
        if settings.provider != "sentence_transformers" or settings.dimensions != 384:
            raise EmbeddingConfigurationError(
                "embedding_configuration_error",
                "The local sentence-transformers provider requires a 384-dimensional profile.",
            )
        self.settings = settings
        self._model = model

    @property
    def model_version(self) -> str:
        return self.settings.model_version

    @property
    def dimensions(self) -> int:
        return 384

    @property
    def storage_profile(self) -> Literal["local", "legacy"]:
        return "local"

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.model, device="cpu")
            except Exception:
                raise EmbeddingConfigurationError(
                    "embedding_model_unavailable",
                    "The local embedding model is unavailable. Install dependencies and ensure "
                    "the model is present in the local cache or network access is available.",
                ) from None
        return self._model

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            encoded = self._get_model().encode(
                list(texts),
                batch_size=self.settings.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return encoded.tolist()
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingProviderError(
                "embedding_generation_failed", "The local embedding model failed to encode text."
            ) from None


class EmbeddingService:
    """Generate, retry, batch, and validate embeddings without persisting them."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 64,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 60.0,
        sleep=time.sleep,
        jitter=random.random,
    ) -> None:
        if batch_size < 1 or batch_size > 64:
            raise ValueError("batch_size must be between 1 and 64")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_delay_seconds < 0:
            raise ValueError("retry_base_delay_seconds cannot be negative")
        if retry_max_delay_seconds < 0:
            raise ValueError("retry_max_delay_seconds cannot be negative")
        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.sleep = sleep
        self.jitter = jitter

    @property
    def model_version(self) -> str:
        return self.provider.model_version

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    @property
    def storage_profile(self) -> Literal["local", "legacy"]:
        return getattr(self.provider, "storage_profile", "legacy")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed inputs in order, retry transient failures, and verify every vector."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            for attempt in range(self.max_retries + 1):
                try:
                    batch_vectors = self.provider.embed_batch(batch)
                    break
                except EmbeddingRetryableError as error:
                    if attempt >= self.max_retries:
                        raise EmbeddingProviderError(
                            error.code,
                            "The embedding provider request failed after retries.",
                            status_code=error.status_code,
                            category=error.category,
                            retry_after_seconds=error.retry_after_seconds,
                        ) from None
                    ceiling = min(self.retry_base_delay_seconds * (2**attempt), 8.0)
                    delay = ceiling * min(max(self.jitter(), 0.0), 1.0)
                    if error.retry_after_seconds is not None:
                        if error.retry_after_seconds > self.retry_max_delay_seconds:
                            raise EmbeddingProviderError(
                                error.code,
                                "The provider retry delay exceeds the configured retry window.",
                                status_code=error.status_code,
                                category=error.category,
                                retry_after_seconds=error.retry_after_seconds,
                            ) from None
                        delay = max(delay, error.retry_after_seconds)
                    logger.warning(
                        "embedding_retry status_code=%s category=%s "
                        "retry_after_seconds=%s attempt=%d delay_seconds=%.3f",
                        error.status_code if error.status_code is not None else "none",
                        error.category,
                        _format_seconds(error.retry_after_seconds),
                        attempt + 1,
                        delay,
                    )
                    self.sleep(delay)
                except EmbeddingError:
                    raise
                except Exception:
                    raise EmbeddingProviderError(
                        "embedding_provider_error", "The embedding provider request failed."
                    ) from None

            if len(batch_vectors) != len(batch):
                raise EmbeddingProviderError(
                    "embedding_response_invalid",
                    "The embedding provider returned an invalid count.",
                )
            vectors.extend(self._validate_vector(vector) for vector in batch_vectors)
        return vectors

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.dimensions:
            raise EmbeddingDimensionError(
                "embedding_dimension_mismatch",
                f"Expected {self.dimensions} embedding values; received {len(vector)}.",
            )
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError):
            raise EmbeddingProviderError(
                "embedding_response_invalid", "The embedding provider returned nonnumeric values."
            ) from None
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingProviderError(
                "embedding_response_invalid", "The embedding provider returned non-finite values."
            )
        return values


def create_embedding_service(settings: EmbeddingSettings | None = None) -> EmbeddingService:
    """Construct the configured provider and bounded embedding service."""
    config = settings or get_embedding_settings()
    if config.provider == "sentence_transformers":
        provider: EmbeddingProvider = SentenceTransformerEmbeddingProvider(config)
    else:
        provider = OpenAIEmbeddingProvider(config)
    return EmbeddingService(
        provider,
        batch_size=config.batch_size,
        max_retries=config.max_retries,
        retry_base_delay_seconds=config.retry_base_delay_seconds,
    )


@lru_cache(maxsize=1)
def get_default_embedding_service() -> EmbeddingService:
    """Share the configured provider and loaded local model across all app flows."""
    return create_embedding_service()
