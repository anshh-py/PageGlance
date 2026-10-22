"""Ingest caller-extracted text without coupling to parsers or persistence."""

import json
from uuid import NAMESPACE_URL, uuid5

from app.chunking import ChunkingConfig, chunk_document
from app.documents import Document, DocumentSourceType, IngestionResult
from app.normalization import content_hash, normalize_text
from app.url_identity import normalize_page_url


def ingest_text(
    text: str,
    *,
    source_type: DocumentSourceType,
    title: str = "",
    source_url: str | None = None,
    filename: str | None = None,
    metadata: dict[str, object] | None = None,
    chunking: ChunkingConfig | None = None,
) -> IngestionResult:
    """Normalize supplied text and return a document plus ordered chunks."""
    if source_type == "webpage" and source_url is not None:
        source_url = normalize_page_url(source_url)
    normalized = normalize_text(text)
    digest = content_hash(normalized)
    identity = source_url or filename or digest
    stable_key = json.dumps(["pageglance-document-v1", source_type, identity], ensure_ascii=False)
    document = Document(
        id=str(uuid5(NAMESPACE_URL, stable_key)),
        source_type=source_type,
        title=title,
        source_url=source_url,
        filename=filename,
        content_hash=digest,
        text=normalized,
        metadata=dict(metadata or {}),
    )
    return IngestionResult(document=document, chunks=chunk_document(document, chunking))
