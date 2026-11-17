"""Thin adapter from universal content contracts into the shared ingestion pipeline.

This module only converts a validated :class:`~app.contracts.ExtractedDocument` into
the inputs already understood by ``ingest_text`` and ``embed_and_persist``. It does
not introduce a parallel ingestion pipeline, a new identifier scheme, or any
deduplication semantics beyond what the repository already defines.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError

from app.contracts import BlockReference, ExtractedDocument
from app.db import DatabaseConfigurationError
from app.embedding_pipeline import embed_and_persist
from app.embeddings import EmbeddingError, EmbeddingService
from app.ingestion import ingest_text
from app.repository import DocumentRepository

logger = logging.getLogger(__name__)

# Bounds on the structured references mapped into each chunk's stored metadata.
MAX_BLOCK_REFS_PER_CHUNK = 32
MAX_BLOCK_TEXT_CHARS_PER_REF = 400


@dataclass(frozen=True)
class ExtractedIngestResult:
    """Safe indexing status for a converted content contract."""

    indexed: bool
    document_id: str | None
    chunk_count: int
    fallback_code: str | None = None
    duplicate: bool = False


def _source_metadata(source) -> dict[str, object]:
    """Preserve caller metadata plus the stable source identifiers."""
    metadata = dict(source.metadata)
    if source.source_id:
        metadata["source_id"] = source.source_id
    if source.mime_type:
        metadata["mime_type"] = source.mime_type
    return metadata


def _bounded_block_ref(block: BlockReference) -> dict[str, object]:
    ref: dict[str, object] = {
        "block_type": block.block_type,
        "start_char": block.start_char,
        "end_char": block.end_char,
    }
    if block.text:
        ref["text"] = block.text[:MAX_BLOCK_TEXT_CHARS_PER_REF]
    if block.page is not None:
        ref["page"] = block.page
    if block.section is not None:
        ref["section"] = block.section
    if block.timestamp_sec is not None:
        ref["timestamp_sec"] = block.timestamp_sec
    return ref


def _add_block_references(extracted: ExtractedDocument, result) -> None:
    """Map validated block references into existing chunk metadata.

    Page numbers are folded into the same ``page_numbers``/``page_start``/
    ``page_end`` metadata fields the citation path already reads, so page-level
    citations genuinely propagate. Section and timestamp are stored under a
    ``structured_blocks`` key that has no citation consumer today.
    """
    if not extracted.structured_blocks:
        return
    for chunk in result.chunks:
        refs: list[dict[str, object]] = []
        pages: list[int] = []
        for block in extracted.structured_blocks:
            if block.start_char < chunk.end_char and block.end_char > chunk.start_char:
                refs.append(_bounded_block_ref(block))
                if block.page is not None:
                    pages.append(block.page)
                if len(refs) >= MAX_BLOCK_REFS_PER_CHUNK:
                    break
        if refs:
            chunk.metadata["structured_blocks"] = refs
        if pages:
            ordered = sorted(set(pages))
            chunk.metadata["page_numbers"] = ordered
            chunk.metadata["page_start"] = ordered[0]
            chunk.metadata["page_end"] = ordered[-1]


def ingest_extracted_document(
    extracted: ExtractedDocument,
    *,
    session_factory=None,
    embedding_service: EmbeddingService | None = None,
    repository_factory=DocumentRepository,
    max_chunks: int | None = None,
    annotate=None,
    owner_id: str | None = None,
) -> ExtractedIngestResult:
    """Convert a validated contract and persist it through the shared pipeline.

    A failed extraction is never turned into a successfully indexed empty document;
    it returns an ``indexed=False`` result instead. Identity and deduplication are
    delegated entirely to ``ingest_text`` and the repository, so webpage URL
    identity and non-webpage content-hash deduplication are preserved unchanged.

    ``repository_factory`` builds the repository around a fresh session (default
    ``DocumentRepository``). ``max_chunks`` caps stored chunk count before any
    persistence, and ``annotate`` lets a caller attach extra chunk metadata (such
    as PDF page references) just before the document is persisted.
    """
    from app import page_indexing  # lazy: avoids an import cycle with page_indexing

    if extracted.extraction_status == "failed":
        logger.info("content_conversion_skipped reason=extraction_failed")
        return ExtractedIngestResult(False, None, 0, "extraction_failed")

    if not extracted.normalized_text:
        return ExtractedIngestResult(False, None, 0, "empty_page_text")

    source = extracted.source
    source_url = source.original_url if source.source_type == "webpage" else None
    metadata = _source_metadata(source)
    metadata["extraction_status"] = extracted.extraction_status
    if extracted.extraction_warnings:
        metadata["extraction_warnings"] = list(extracted.extraction_warnings)
    try:
        result = ingest_text(
            extracted.normalized_text,
            source_type=source.source_type,
            title=extracted.title,
            source_url=source_url,
            filename=source.filename,
            metadata=metadata,
            owner_id=owner_id,
        )
    except ValueError:
        return ExtractedIngestResult(False, None, 0, "content_validation_failed")

    if not result.chunks:
        return ExtractedIngestResult(False, None, 0, "empty_page_text")

    if max_chunks is not None and len(result.chunks) > max_chunks:
        return ExtractedIngestResult(False, None, 0, "chunk_limit_exceeded")

    _add_block_references(extracted, result)
    if annotate is not None:
        annotate(result)

    try:
        create_session = session_factory or page_indexing.get_ingestion_session_factory()
        service = (
            embedding_service or page_indexing.get_ingestion_embedding_service()
        )
        with create_session() as session:
            repo = repository_factory(session)
            existing = (
                repo.get_by_id(result.document.id, owner_id=owner_id)
                if hasattr(repo, "get_by_id")
                else None
            )
            saved = embed_and_persist(
                result, repository=repo, service=service, owner_id=owner_id
            )
            duplicate = saved.id != result.document.id or (
                existing is not None
                and existing.document.content_hash == result.document.content_hash
            )
    except (DatabaseConfigurationError, EmbeddingError, SQLAlchemyError, ValueError) as error:
        if isinstance(error, DatabaseConfigurationError):
            code = "database_configuration_error"
        elif isinstance(error, SQLAlchemyError):
            code = "database_unavailable"
        elif isinstance(error, EmbeddingError):
            code = error.code
        else:
            code = getattr(error, "code", "persistence_unavailable")
        logger.warning("content_conversion_failed code=%s", code)
        return ExtractedIngestResult(False, None, 0, code)

    return ExtractedIngestResult(True, saved.id, len(result.chunks), duplicate=duplicate)
