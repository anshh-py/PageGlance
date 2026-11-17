"""Index an extension-supplied webpage using the shared ingestion pipeline."""

import logging
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.exc import SQLAlchemyError

from app.contracts import BlockReference, ContentSource, ExtractedDocument
from app.converters import ingest_extracted_document
from app.db import DatabaseConfigurationError, create_session_factory
from app.embeddings import (
    EmbeddingError,
    EmbeddingService,
    get_default_embedding_service,
)
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


def _structured_blocks(page: dict[str, object]) -> list[BlockReference]:
    """Build validated block references from caller data, dropping invalid ones.

    Offsets are validated against the normalized text (the same Pydantic
    validator the contract enforces). Malformed or out-of-range references are
    rejected rather than failing the whole page index.
    """
    raw = page.get("blocks") or page.get("structured_blocks") or []
    if not isinstance(raw, list):
        return []
    text_length = len(str(page.get("text", "")))
    blocks: list[BlockReference] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            block = BlockReference(**item)
        except (ValueError, TypeError):
            continue
        if block.end_char > text_length or block.start_char > block.end_char:
            continue
        blocks.append(block)
    return blocks


def index_page_context(
    page: dict[str, object], *, owner_id: str | None = None
) -> PageIndexResult:
    """Normalize, chunk, embed, and persist one active page with bounded work.

    The owner may be passed explicitly or embedded as ``page["_owner_id"]`` so
    callers that splice the owner into the page payload (e.g. HTTP routes) keep
    test fixtures that stub this function signature-compatible.
    """
    if owner_id is None:
        embedded = page.get("_owner_id")
        owner_id = embedded if isinstance(embedded, str) and embedded else None
    text = str(page.get("text", ""))
    if len(text) > MAX_PAGE_TEXT_CHARS:
        return PageIndexResult(False, None, 0, "page_text_too_large")

    try:
        source_url = normalize_page_url(str(page.get("url", "")))
        extracted = ExtractedDocument(
            title=str(page.get("title", "")),
            normalized_text=text,
            source=ContentSource(source_type="webpage", original_url=source_url),
            extraction_status="complete",
            structured_blocks=_structured_blocks(page),
        )
        converted = ingest_extracted_document(
            extracted,
            max_chunks=MAX_PAGE_CHUNKS,
            repository_factory=DocumentRepository,
            owner_id=owner_id,
        )
    except (DatabaseConfigurationError, EmbeddingError, SQLAlchemyError, ValueError) as error:
        # Error details may contain provider or connection data; log only a safe code.
        code = getattr(error, "code", "page_index_failed")
        logger.warning("page_index_fallback code=%s page_text_chars=%d", code, len(text))
        return PageIndexResult(False, None, 0, code)

    if not converted.indexed:
        fallback = converted.fallback_code or "page_index_failed"
        if fallback == "chunk_limit_exceeded":
            fallback = "page_chunk_limit_exceeded"
        logger.warning(
            "page_index_fallback code=%s page_text_chars=%d", fallback, len(text)
        )
        return PageIndexResult(False, None, 0, fallback)

    logger.info(
        "page_indexed chunks=%d page_text_chars=%d",
        converted.chunk_count,
        len(text),
    )
    return PageIndexResult(True, converted.document_id, converted.chunk_count)
