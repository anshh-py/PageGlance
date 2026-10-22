"""Typed document and chunk records used by the ingestion foundation."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentSourceType = Literal["webpage", "pdf", "text", "markdown", "image"]


class Document(BaseModel):
    """Normalized source text and metadata, ready for later persistence or retrieval."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: DocumentSourceType
    title: str = ""
    source_url: str | None = None
    filename: str | None = None
    content_hash: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """A source-offset-preserving chunk belonging to a document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    index: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Result of ingesting already-extracted source text."""

    document: Document
    chunks: list[DocumentChunk]
