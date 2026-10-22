"""Deterministic character-based chunking with source offsets and overlap."""

import re
from dataclasses import dataclass

from app.documents import Document, DocumentChunk

_PARAGRAPH_BOUNDARY = re.compile(r"\n\n")
_SENTENCE_BOUNDARY = re.compile(r"[.!?\u3002\uff01\uff1f]+[\"'\u2019\u201d)}\]]*(?=\s|$)")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Character limits for deterministic document chunks."""

    chunk_size: int = 1200
    overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


def _preferred_end(text: str, start: int, hard_end: int, size: int) -> int:
    """Choose a nearby paragraph, sentence, or word boundary without exceeding size."""
    if hard_end == len(text):
        return hard_end

    minimum = start + max(1, size // 2)
    for pattern in (_PARAGRAPH_BOUNDARY, _SENTENCE_BOUNDARY, _WHITESPACE):
        candidates = list(pattern.finditer(text, start, hard_end))
        eligible = [match for match in candidates if start < match.end() <= hard_end]
        if eligible and eligible[-1].end() >= minimum:
            return eligible[-1].end()
    return hard_end


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[tuple[int, int]]:
    """Return ordered half-open spans into normalized text; empty input yields no spans."""
    settings = config or ChunkingConfig()
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + settings.chunk_size, len(text))
        end = _preferred_end(text, start, hard_end, settings.chunk_size)
        if end <= start:  # Defensive forward progress if a boundary rule changes later.
            end = min(start + settings.chunk_size, len(text))
        spans.append((start, end))
        if end == len(text):
            break

        next_start = max(start + 1, end - settings.overlap)
        # Keep overlap starts on a word boundary when one exists within the prior span.
        if (
            next_start > start
            and not text[next_start - 1].isspace()
            and not text[next_start].isspace()
        ):
            boundary = next(
                (match.end() for match in _WHITESPACE.finditer(text, next_start, end)), None
            )
            if boundary is not None:
                next_start = boundary
        start = next_start

    return spans


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[DocumentChunk]:
    """Chunk a document while preserving normalized-text offsets and metadata."""
    chunks: list[DocumentChunk] = []
    for index, (start, end) in enumerate(chunk_text(document.text, config)):
        chunks.append(
            DocumentChunk(
                id=f"{document.id}:chunk:{index}:{start}:{end}",
                document_id=document.id,
                index=index,
                text=document.text[start:end],
                start_char=start,
                end_char=end,
                metadata=document.metadata.copy(),
            )
        )
    return chunks
