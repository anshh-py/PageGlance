"""Universal content contracts for external content ingestion.

These models describe already-extracted document content that external callers may
hand to the backend. They are data contracts only: nothing here authenticates a
caller or asserts ownership. A ``ContentSource.metadata`` value such as an "owner"
key is untrusted input and is never used to gate access or enforce isolation.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ContentSourceType = Literal["webpage", "pdf", "text", "markdown", "image"]
ExtractionStatus = Literal["complete", "partial", "failed"]

# Hard bounds on the number and size of structured references retained in a single
# extracted document, so a hostile or malformed payload cannot exhaust memory.
MAX_STRUCTURED_BLOCKS = 100
MAX_EXTRACTION_WARNINGS = 20
MAX_WARNING_CHARS = 500


class ContentSource(BaseModel):
    """Provenance of a document, without any implied ownership semantics.

    ``metadata`` holds arbitrary untrusted fields. A caller-provided "owner" value
    is just data; it must never be treated as authentication or ownership, because
    the backend has no notion of document owners.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: ContentSourceType
    source_id: str = Field(default="")
    original_url: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BlockReference(BaseModel):
    """One structured fragment of a document, positioned within normalized text."""

    model_config = ConfigDict(extra="forbid")

    block_type: str = Field(min_length=1)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    page: int | None = None
    section: str | None = None
    timestamp_sec: float | None = None

    @model_validator(mode="after")
    def _validate_own_offsets(self) -> "BlockReference":
        if self.start_char > self.end_char:
            raise ValueError("BlockReference start_char must not exceed end_char.")
        return self


class ExtractedDocument(BaseModel):
    """A normalized, already-extracted document ready for shared ingestion."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=1000)
    normalized_text: str
    source: ContentSource
    structured_blocks: list[BlockReference] = Field(default_factory=list)
    extraction_status: ExtractionStatus = "complete"
    extraction_warnings: list[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("extraction_warnings")
    @classmethod
    def _bound_warnings(cls, warnings: list[str]) -> list[str]:
        bounded = [warning[:MAX_WARNING_CHARS] for warning in warnings]
        return bounded[-MAX_EXTRACTION_WARNINGS:]

    @model_validator(mode="after")
    def _validate_blocks(self) -> "ExtractedDocument":
        if len(self.structured_blocks) > MAX_STRUCTURED_BLOCKS:
            raise ValueError(
                f"At most {MAX_STRUCTURED_BLOCKS} structured blocks are allowed."
            )
        text_length = len(self.normalized_text)
        for index, block in enumerate(self.structured_blocks):
            if block.end_char > text_length:
                raise ValueError(
                    f"Structured block {index} ends past the normalized text."
                )
        return self
