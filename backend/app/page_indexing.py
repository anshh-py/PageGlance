"""Index an extension-supplied webpage using the shared ingestion pipeline."""

import logging
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.exc import SQLAlchemyError

from app.db import DatabaseConfigurationError, create_session_factory
from app.embedding_pipeline import embed_and_persist
from app.embeddings import (
    EmbeddingError,
    EmbeddingService,
    get_default_embedding_service,
)
from app.ingestion import ingest_text
from app.repository import DocumentRepository
from app.url_identity import normalize_page_url

logger = logging.getLogger(__name__)
MAX_PAGE_TEXT_CHARS = 100_000
MAX_PAGE_CHUNKS = 128


@dataclass(frozen=True)
class PageIndexResult:
    """Safe indexing status used to decide whether query-time retrieval can run."""

    indexed: bool
    document_id: str | None
    chunk_count: int
    fallback_code: str | None = None


@lru_cache(maxsize=1)
def _session_factory():
    return create_session_factory()


@lru_cache(maxsize=1)
def _embedding_service() -> EmbeddingService:
    return get_default_embedding_service()


def get_ingestion_session_factory():
    """Return the shared cached factory used by automatic document ingestion."""
    return _session_factory()


def get_ingestion_embedding_service() -> EmbeddingService:
    """Return the shared cached provider service used by all ingestion flows."""
    return _embedding_service()


def index_page_context(page: dict[str, str]) -> PageIndexResult:
    """Normalize, chunk, embed, and persist one active page with bounded work."""
    text = page.get("text", "")
    if len(text) > MAX_PAGE_TEXT_CHARS:
        return PageIndexResult(False, None, 0, "page_text_too_large")

    try:
        source_url = normalize_page_url(page.get("url", ""))
        result = ingest_text(
            text,
            source_type="webpage",
            title=page.get("title", ""),
            source_url=source_url,
        )
        if len(result.chunks) > MAX_PAGE_CHUNKS:
            return PageIndexResult(False, None, 0, "page_chunk_limit_exceeded")
        if not result.chunks:
            return PageIndexResult(False, None, 0, "empty_page_text")

        with get_ingestion_session_factory()() as session:
            saved = embed_and_persist(
                result,
                repository=DocumentRepository(session),
                service=get_ingestion_embedding_service(),
            )
        logger.info(
            "page_indexed chunks=%d page_text_chars=%d",
            len(result.chunks),
            len(text),
        )
        return PageIndexResult(True, saved.id, len(result.chunks))
    except (DatabaseConfigurationError, EmbeddingError, SQLAlchemyError, ValueError) as error:
        # Error details may contain provider or connection data; log only a safe code.
        code = getattr(error, "code", "page_index_failed")
        logger.warning("page_index_fallback code=%s page_text_chars=%d", code, len(text))
        return PageIndexResult(False, None, 0, code)
