"""Focused coverage for the universal content contracts."""

import pytest
from pydantic import ValidationError

from app.contracts import (
    MAX_EXTRACTION_WARNINGS,
    MAX_STRUCTURED_BLOCKS,
    MAX_WARNING_CHARS,
    BlockReference,
    ContentSource,
    ExtractedDocument,
)


def _source(**overrides):
    fields = {
        "source_type": "webpage",
        "source_id": "src-1",
        "original_url": "https://example.com/a",
    }
    fields.update(overrides)
    return ContentSource(**fields)


def _block(start: int = 0, end: int = 5, **overrides) -> BlockReference:
    fields = {
        "block_type": "paragraph",
        "text": "Hello",
        "start_char": start,
        "end_char": end,
    }
    fields.update(overrides)
    return BlockReference(**fields)


def test_valid_contract_construction_sets_defaults() -> None:
    doc = ExtractedDocument(
        title="Example",
        normalized_text="Hello world",
        source=_source(),
        structured_blocks=[_block(end=5)],
    )
    assert doc.title == "Example"
    assert doc.extraction_status == "complete"
    assert doc.extracted_at is not None
    assert doc.structured_blocks[0].block_type == "paragraph"
    assert doc.source.source_id == "src-1"


def test_content_source_defaults() -> None:
    src = _source()
    assert src.mime_type is None
    assert src.filename is None
    assert src.metadata == {}


def test_invalid_source_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ContentSource(source_type="docx")


@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
def test_accepted_extraction_statuses(status: str) -> None:
    doc = ExtractedDocument(normalized_text="x", source=_source(), extraction_status=status)
    assert doc.extraction_status == status


def test_invalid_extraction_status_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedDocument(normalized_text="x", source=_source(), extraction_status="pending")


def test_metadata_serialization_round_trip() -> None:
    doc = ExtractedDocument(
        normalized_text="Hello world",
        source=_source(metadata={"lang": "en"}),
        structured_blocks=[
            _block(end=5, page=1, section="intro", timestamp_sec=1.5)
        ],
        extraction_warnings=["ok"],
    )
    restored = ExtractedDocument.model_validate(doc.model_dump())
    assert restored == doc
    assert restored.source.metadata == {"lang": "en"}
    assert restored.structured_blocks[0].page == 1
    assert restored.structured_blocks[0].section == "intro"


def test_block_offsets_must_not_be_reversed() -> None:
    with pytest.raises(ValidationError):
        _block(start=5, end=2)


def test_negative_offset_rejected() -> None:
    with pytest.raises(ValidationError):
        _block(start=-1, end=2)


def test_block_bounds_validated_against_normalized_text() -> None:
    with pytest.raises(ValidationError):
        ExtractedDocument(
            normalized_text="abc",
            source=_source(),
            structured_blocks=[_block(start=0, end=4)],
        )


def test_too_many_blocks_rejected() -> None:
    blocks = [_block(end=1) for _ in range(MAX_STRUCTURED_BLOCKS + 1)]
    with pytest.raises(ValidationError):
        ExtractedDocument(
            normalized_text="x" * 10,
            source=_source(),
            structured_blocks=blocks,
        )


def test_warnings_are_bounded_by_count_and_length() -> None:
    long_warning = "a" * (MAX_WARNING_CHARS + 50)
    warnings = [long_warning] * (MAX_EXTRACTION_WARNINGS + 5)
    doc = ExtractedDocument(
        normalized_text="x", source=_source(), extraction_warnings=warnings
    )
    assert len(doc.extraction_warnings) == MAX_EXTRACTION_WARNINGS
    assert all(len(item) <= MAX_WARNING_CHARS for item in doc.extraction_warnings)
