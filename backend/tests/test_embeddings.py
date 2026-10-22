"""Mocked provider and persistence orchestration tests; no paid API calls."""

from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, RateLimitError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.chunking import ChunkingConfig
from app.documents import IngestionResult
from app.embedding_pipeline import embed_and_persist
from app.embeddings import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingRetryableError,
    EmbeddingService,
    EmbeddingSettings,
    EmbeddingTimeoutError,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_service,
    get_default_embedding_service,
    get_embedding_settings,
)
from app.ingestion import ingest_text
from app.models import Base, DocumentChunkRecord
from app.normalization import exact_text_hash
from app.repository import DocumentRepository


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as database_session:
        yield database_session
    engine.dispose()


class FakeProvider:
    def __init__(
        self,
        *,
        dimensions: int = 3,
        model_version: str = "test:embedding-v1",
        failure: Exception | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.model_version = model_version
        self.failure = failure
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.failure is not None:
            raise self.failure
        return [[float(len(text)), 0.25, -0.5][: self.dimensions] for text in texts]


def test_openai_adapter_sends_config_and_restores_response_order() -> None:
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[2.0, 2.0, 2.0]),
                    SimpleNamespace(index=0, embedding=[1.0, 1.0, 1.0]),
                ]
            )

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    settings = EmbeddingSettings(
        provider="openai",
        api_key="private-test-key",
        model="text-embedding-3-small",
        dimensions=3,
    )
    provider = OpenAIEmbeddingProvider(settings, client=client)

    vectors = provider.embed_batch(["first", "second"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert captured == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
        "dimensions": 3,
        "encoding_format": "float",
    }
    assert provider.model_version == "openai:text-embedding-3-small:dimensions=3"


@pytest.mark.parametrize(
    ("provider_error", "expected_type"),
    [
        (
            RateLimitError(
                "private rate detail",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.test")),
                body={"error": {"message": "private rate detail"}},
            ),
            EmbeddingRateLimitError,
        ),
        (
            APITimeoutError(request=httpx.Request("POST", "https://api.test")),
            EmbeddingTimeoutError,
        ),
    ],
)
def test_openai_adapter_maps_provider_errors_safely(
    provider_error: Exception, expected_type: type[Exception]
) -> None:
    class FailingEmbeddings:
        def create(self, **_kwargs):
            raise provider_error

    provider = OpenAIEmbeddingProvider(
        EmbeddingSettings(provider="openai", api_key="test-key"),
        client=SimpleNamespace(embeddings=FailingEmbeddings()),
    )

    with pytest.raises(expected_type) as error:
        provider.embed_batch(["private page text"])

    assert "private rate detail" not in str(error.value)
    assert "private page text" not in str(error.value)


def test_openai_429_diagnostics_distinguish_transient_and_quota(caplog) -> None:
    request = httpx.Request("POST", "https://api.test")
    transient = RateLimitError(
        "PRIVATE_PROVIDER_MESSAGE",
        response=httpx.Response(429, headers={"Retry-After": "2.5"}, request=request),
        body={"error": {"code": "rate_limit_exceeded", "message": "PRIVATE_BODY"}},
    )
    quota = RateLimitError(
        "PRIVATE_PROVIDER_MESSAGE",
        response=httpx.Response(429, headers={"Retry-After": "10"}, request=request),
        body={"error": {"code": "insufficient_quota", "message": "PRIVATE_BODY"}},
    )

    def adapter(error: Exception) -> OpenAIEmbeddingProvider:
        class FailingEmbeddings:
            def create(self, **_kwargs):
                raise error

        return OpenAIEmbeddingProvider(
            EmbeddingSettings(provider="openai", api_key="PRIVATE_API_KEY"),
            client=SimpleNamespace(embeddings=FailingEmbeddings()),
        )

    with pytest.raises(EmbeddingRateLimitError) as limited:
        adapter(transient).embed_batch(["PRIVATE_UPLOAD_TEXT"])
    assert limited.value.status_code == 429
    assert limited.value.category == "transient_rate_limit"
    assert limited.value.retry_after_seconds == 2.5

    with pytest.raises(EmbeddingProviderError) as exhausted:
        adapter(quota).embed_batch(["PRIVATE_UPLOAD_TEXT"])
    assert exhausted.value.code == "embedding_quota_exhausted"
    assert exhausted.value.category == "quota_exhausted"
    assert exhausted.value.retry_after_seconds == 10
    assert "status_code=429 category=transient_rate_limit retry_after_seconds=2.500" in caplog.text
    assert "status_code=429 category=quota_exhausted retry_after_seconds=10.000" in caplog.text
    for secret in (
        "PRIVATE_API_KEY",
        "PRIVATE_UPLOAD_TEXT",
        "PRIVATE_BODY",
        "PRIVATE_PROVIDER_MESSAGE",
    ):
        assert secret not in caplog.text


def test_openai_permanent_error_is_sanitized_and_non_retryable() -> None:
    from openai import BadRequestError

    failure = BadRequestError(
        "PRIVATE_BAD_REQUEST",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.test")),
        body={"error": {"message": "PRIVATE_BAD_REQUEST"}},
    )

    class FailingEmbeddings:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            raise failure

    client = SimpleNamespace(embeddings=FailingEmbeddings())
    provider = OpenAIEmbeddingProvider(
        EmbeddingSettings(provider="openai", api_key="test-key"), client=client
    )
    service = EmbeddingService(provider, max_retries=3, sleep=lambda _delay: None)

    with pytest.raises(EmbeddingProviderError) as error:
        service.embed_texts(["text"])

    assert error.value.code == "embedding_provider_error"
    assert error.value.status_code == 400
    assert client.embeddings.calls == 1
    assert "PRIVATE_BAD_REQUEST" not in str(error.value)


def test_embedding_settings_default_to_local_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.embeddings._load_root_environment", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PAGEGLANCE_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("PAGEGLANCE_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("PAGEGLANCE_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("PAGEGLANCE_EMBEDDING_MODEL", "text-embedding-3-small")
    get_embedding_settings.cache_clear()

    settings = get_embedding_settings()

    assert settings.api_key is None
    assert settings.provider == "sentence_transformers"
    assert settings.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.dimensions == 384
    assert settings.storage_profile == "local"
    get_embedding_settings.cache_clear()


def test_openai_provider_requires_key_only_when_explicitly_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.embeddings import EmbeddingConfigurationError

    monkeypatch.setattr("app.embeddings._load_root_environment", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAGEGLANCE_EMBEDDING_PROVIDER", "openai")
    get_embedding_settings.cache_clear()

    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY is required"):
        get_embedding_settings()
    get_embedding_settings.cache_clear()


def test_sentence_transformer_provider_uses_mocked_model_in_batches() -> None:
    class Encoded:
        def tolist(self):
            return [[0.5] * 384, [0.25] * 384]

    class MockModel:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return Encoded()

    model = MockModel()
    settings = EmbeddingSettings(batch_size=2)
    provider = SentenceTransformerEmbeddingProvider(settings, model=model)

    vectors = provider.embed_batch(["first text", "second text"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert provider.model_version == (
        "sentence_transformers:sentence-transformers/all-MiniLM-L6-v2:dimensions=384"
    )
    assert provider.storage_profile == "local"
    assert model.calls == [
        (
            ["first text", "second text"],
            {
                "batch_size": 2,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        )
    ]


def test_sentence_transformer_errors_are_safe() -> None:
    class BrokenModel:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("PRIVATE_INPUT_AND_MODEL_ERROR")

    provider = SentenceTransformerEmbeddingProvider(EmbeddingSettings(), model=BrokenModel())
    with pytest.raises(EmbeddingProviderError) as error:
        provider.embed_batch(["PRIVATE_INPUT"])
    assert error.value.code == "embedding_generation_failed"
    assert "PRIVATE_INPUT" not in str(error.value)
    assert "PRIVATE_MODEL_ERROR" not in str(error.value)


def test_default_factory_uses_local_provider_without_loading_weights() -> None:
    service = create_embedding_service(EmbeddingSettings())
    assert isinstance(service.provider, SentenceTransformerEmbeddingProvider)
    assert service.storage_profile == "local"


def test_page_and_query_paths_share_the_cached_embedding_service(monkeypatch) -> None:
    from app import page_indexing
    from app.retrieval import VectorRetriever

    service = object()
    monkeypatch.setattr("app.embeddings.create_embedding_service", lambda: service)
    get_default_embedding_service.cache_clear()
    page_indexing._embedding_service.cache_clear()

    assert get_default_embedding_service() is service
    assert page_indexing.get_ingestion_embedding_service() is service
    assert VectorRetriever(session=object()).embedding_service is service

    get_default_embedding_service.cache_clear()
    page_indexing._embedding_service.cache_clear()


def test_service_batches_texts_in_order() -> None:
    provider = FakeProvider()
    service = EmbeddingService(provider, batch_size=2)

    vectors = service.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(batch) for batch in provider.calls] == [2, 2, 1]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(len(vector) == 3 for vector in vectors)


def test_service_retries_rate_limits_with_bounded_backoff() -> None:
    class RetryProvider(FakeProvider):
        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            if len(self.calls) < 3:
                raise EmbeddingRateLimitError("embedding_rate_limited", "safe detail")
            return [[0.1, 0.2, 0.3] for _ in texts]

    provider = RetryProvider()
    delays: list[float] = []
    service = EmbeddingService(
        provider,
        max_retries=2,
        retry_base_delay_seconds=0.25,
        sleep=delays.append,
        jitter=lambda: 0.5,
    )

    assert service.embed_texts(["retry me"]) == [[0.1, 0.2, 0.3]]
    assert len(provider.calls) == 3
    assert delays == [0.125, 0.25]


def test_service_respects_retry_after_and_does_not_retry_quota(caplog) -> None:
    class RetryAfterProvider(FakeProvider):
        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            if len(self.calls) == 1:
                raise EmbeddingRateLimitError(
                    "embedding_rate_limited",
                    "safe message",
                    status_code=429,
                    category="transient_rate_limit",
                    retry_after_seconds=2.75,
                )
            return [[0.1, 0.2, 0.3] for _ in texts]

    delays: list[float] = []
    provider = RetryAfterProvider()
    service = EmbeddingService(
        provider,
        max_retries=2,
        retry_base_delay_seconds=0.5,
        sleep=delays.append,
        jitter=lambda: 0,
    )
    assert service.embed_texts(["text"]) == [[0.1, 0.2, 0.3]]
    assert delays == [2.75]
    assert "status_code=429" in caplog.text
    assert "retry_after_seconds=2.750" in caplog.text
    assert "delay_seconds=2.750" in caplog.text

    quota_provider = FakeProvider(
        failure=EmbeddingProviderError(
            "embedding_quota_exhausted",
            "safe quota error",
            status_code=429,
            category="quota_exhausted",
        )
    )
    quota_service = EmbeddingService(quota_provider, max_retries=3, sleep=delays.append)
    with pytest.raises(EmbeddingProviderError) as quota:
        quota_service.embed_texts(["text"])
    assert quota.value.code == "embedding_quota_exhausted"
    assert len(quota_provider.calls) == 1


def test_retry_after_beyond_bound_stops_without_early_retry() -> None:
    provider = FakeProvider(
        failure=EmbeddingRateLimitError(
            "embedding_rate_limited",
            "safe message",
            status_code=429,
            category="transient_rate_limit",
            retry_after_seconds=120,
        )
    )
    sleeps: list[float] = []
    service = EmbeddingService(
        provider,
        max_retries=3,
        retry_max_delay_seconds=30,
        sleep=sleeps.append,
    )
    with pytest.raises(EmbeddingProviderError) as error:
        service.embed_texts(["text"])
    assert error.value.retry_after_seconds == 120
    assert len(provider.calls) == 1
    assert sleeps == []


def test_service_exhausts_retries_without_exposing_provider_details() -> None:
    provider = FakeProvider(
        failure=EmbeddingRetryableError("embedding_timeout", "private provider response")
    )
    service = EmbeddingService(provider, max_retries=1, sleep=lambda _delay: None)

    with pytest.raises(EmbeddingProviderError, match="failed after retries") as error:
        service.embed_texts(["text"])

    assert error.value.code == "embedding_timeout"
    assert "private provider response" not in str(error.value)
    assert len(provider.calls) == 2


def test_service_rejects_dimension_mismatch_and_non_finite_values() -> None:
    class MismatchProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=4)

    with pytest.raises(EmbeddingDimensionError):
        EmbeddingService(MismatchProvider()).embed_texts(["text"])

    class NonFiniteProvider(FakeProvider):
        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            return [[float("nan"), 0.0, 0.0] for _ in texts]

    with pytest.raises(EmbeddingProviderError, match="non-finite"):
        EmbeddingService(NonFiniteProvider()).embed_texts(["text"])


def test_model_or_dimension_changes_create_a_distinct_profile() -> None:
    first = EmbeddingSettings(
        provider="openai", api_key="test-key", model="text-embedding-3-small", dimensions=1536
    )
    other_model = EmbeddingSettings(
        provider="openai", api_key="test-key", model="text-embedding-3-large", dimensions=1536
    )
    other_dimensions = EmbeddingSettings(
        provider="openai", api_key="test-key", model="text-embedding-3-small", dimensions=512
    )

    assert (
        len({first.model_version, other_model.model_version, other_dimensions.model_version}) == 3
    )


def test_embedding_pipeline_persists_vectors_and_skips_unchanged_chunks(session: Session) -> None:
    result = ingest_text(
        "First paragraph has enough words.\n\nSecond paragraph has more words.",
        source_type="text",
        filename="notes.txt",
        chunking=ChunkingConfig(chunk_size=24, overlap=4),
    )
    repository = DocumentRepository(session)
    provider = FakeProvider()
    service = EmbeddingService(provider, batch_size=2)

    first_saved = embed_and_persist(result, repository=repository, service=service)
    first_call_count = len(provider.calls)
    first_updated_at = repository.get_by_id(result.document.id).document.updated_at
    repeated_result = ingest_text(
        "First paragraph has enough words.\n\nSecond paragraph has more words.",
        source_type="text",
        filename="notes.txt",
        chunking=ChunkingConfig(chunk_size=24, overlap=4),
    )
    second_saved = embed_and_persist(repeated_result, repository=repository, service=service)
    second_updated_at = repository.get_by_id(result.document.id).document.updated_at
    stored = session.scalars(
        select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == result.document.id)
    ).all()

    assert first_saved.id == second_saved.id == result.document.id
    assert first_call_count > 0
    assert len(provider.calls) == first_call_count
    assert second_updated_at == first_updated_at
    assert all(chunk.embedding is not None for chunk in stored)
    assert all(chunk.embedding_model == service.model_version for chunk in stored)
    assert all(chunk.embedding_dimensions == 3 for chunk in stored)
    assert all(chunk.embedding_text_hash == exact_text_hash(chunk.text) for chunk in stored)


def test_new_model_reembeds_all_chunks_without_mixing_profiles(session: Session) -> None:
    result = ingest_text(
        "A document that gets re-embedded when the model changes.",
        source_type="text",
        filename="model-change.txt",
        chunking=ChunkingConfig(chunk_size=18, overlap=2),
    )
    repository = DocumentRepository(session)
    original = EmbeddingService(FakeProvider(model_version="test:model-a"), batch_size=10)
    revised_provider = FakeProvider(model_version="test:model-b")
    revised = EmbeddingService(revised_provider, batch_size=10)

    embed_and_persist(result, repository=repository, service=original)
    embed_and_persist(result, repository=repository, service=revised)
    stored = session.scalars(
        select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == result.document.id)
    ).all()

    assert revised_provider.calls
    assert {chunk.embedding_model for chunk in stored} == {"test:model-b"}


def test_local_profile_preserves_legacy_vectors_and_reuses_local_vectors(session: Session) -> None:
    class Legacy1536Provider:
        dimensions = 1536
        model_version = "openai:text-embedding-3-small:dimensions=1536"
        storage_profile = "legacy"

        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.125] * self.dimensions for _ in texts]

    class Encoded:
        def tolist(self):
            return [[0.25] * 384]

    class LocalMockModel:
        calls = 0

        def encode(self, texts, **_kwargs):
            self.calls += 1
            assert texts
            return Encoded()

    result = ingest_text(
        "Legacy vectors must remain available after local model indexing.",
        source_type="text",
        filename="preserve-legacy.txt",
    )
    repository = DocumentRepository(session)
    embed_and_persist(result, repository=repository, service=EmbeddingService(Legacy1536Provider()))
    old_record = session.scalar(
        select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == result.document.id)
    )
    assert old_record is not None
    old_vector = list(old_record.embedding)

    model = LocalMockModel()
    local_provider = SentenceTransformerEmbeddingProvider(EmbeddingSettings(), model=model)
    local_service = EmbeddingService(local_provider)
    embed_and_persist(result, repository=repository, service=local_service)
    embed_and_persist(result, repository=repository, service=local_service)
    refreshed = session.scalar(
        select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == result.document.id)
    )
    assert refreshed is not None

    assert refreshed.embedding == old_vector
    assert refreshed.embedding_model == "openai:text-embedding-3-small:dimensions=1536"
    assert refreshed.embedding_dimensions == 1536
    assert len(refreshed.local_embedding) == 384
    assert refreshed.local_embedding_model == local_service.model_version
    assert refreshed.local_embedding_dimensions == 384
    assert model.calls == 1


def test_embedding_failure_does_not_persist_partial_document(session: Session) -> None:
    result: IngestionResult = ingest_text(
        "Do not persist this if embeddings fail.", source_type="text", filename="failure.txt"
    )
    repository = DocumentRepository(session)
    service = EmbeddingService(
        FakeProvider(failure=EmbeddingProviderError("embedding_provider_error", "safe failure")),
        max_retries=2,
        sleep=lambda _delay: None,
    )

    with pytest.raises(EmbeddingProviderError):
        embed_and_persist(result, repository=repository, service=service)

    assert repository.get_by_id(result.document.id) is None


def test_failed_reindex_leaves_existing_document_and_embeddings_unchanged(session: Session) -> None:
    repository = DocumentRepository(session)
    original = ingest_text(
        "Original stable file content and its stored embedding.",
        source_type="text",
        filename="stable.txt",
    )
    embed_and_persist(
        original,
        repository=repository,
        service=EmbeddingService(FakeProvider(), batch_size=20),
    )
    before = repository.get_by_id(original.document.id)
    assert before is not None
    before_state = [
        (chunk.id, chunk.text, list(record.embedding), record.embedding_model)
        for chunk, record in zip(
            before.chunks,
            session.scalars(
                select(DocumentChunkRecord)
                .where(DocumentChunkRecord.document_id == original.document.id)
                .order_by(DocumentChunkRecord.index)
            ).all(),
            strict=True,
        )
    ]

    changed = ingest_text(
        "Completely changed replacement content fails during embedding.",
        source_type="text",
        filename="stable.txt",
    )
    failing = EmbeddingService(
        FakeProvider(failure=EmbeddingProviderError("embedding_provider_error", "safe")),
        max_retries=0,
        sleep=lambda _delay: None,
    )
    with pytest.raises(EmbeddingProviderError):
        embed_and_persist(changed, repository=repository, service=failing)

    after = repository.get_by_id(original.document.id)
    assert after is not None
    after_records = session.scalars(
        select(DocumentChunkRecord)
        .where(DocumentChunkRecord.document_id == original.document.id)
        .order_by(DocumentChunkRecord.index)
    ).all()
    after_state = [
        (chunk.id, chunk.text, list(record.embedding), record.embedding_model)
        for chunk, record in zip(after.chunks, after_records, strict=True)
    ]
    assert after.document.content_hash == before.document.content_hash
    assert after.document.text == before.document.text
    assert after_state == before_state
