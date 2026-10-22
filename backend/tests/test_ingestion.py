"""Tests for text normalization, stable identity, and deterministic chunking."""

import hashlib

import pytest

from app.chunking import ChunkingConfig, chunk_document, chunk_text
from app.ingestion import ingest_text
from app.normalization import content_hash, has_duplicate_content, normalize_text


def test_normalization_preserves_paragraphs_and_technical_spacing() -> None:
    source = "<tag>  value"
    assert normalize_text(source) == source
    assert normalize_text("  First\r\n\tcode  \r\n\r\n\r\nSecond\u00a0line  ") == (
        "First\n\tcode\n\nSecond line"
    )


@pytest.mark.parametrize("source", ["", " \t\r\n ", "\n\n\n"])
def test_normalization_of_empty_or_whitespace_input(source: str) -> None:
    assert normalize_text(source) == ""
    assert chunk_text(normalize_text(source)) == []


def test_hash_is_stable_and_duplicate_check_normalizes_first() -> None:
    expected = hashlib.sha256(b"Alpha\n\nBeta").hexdigest()
    assert content_hash("Alpha\r\n\r\nBeta") == expected
    assert has_duplicate_content(" Alpha\n\nBeta ", "Alpha\r\n\nBeta")
    assert not has_duplicate_content("Alpha", "alpha")


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_settings_are_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=chunk_size, overlap=overlap)


def test_short_document_stays_as_one_chunk() -> None:
    assert chunk_text("A short document.", ChunkingConfig(chunk_size=50, overlap=5)) == [(0, 17)]


def test_chunking_prefers_paragraph_and_sentence_boundaries() -> None:
    text = "First sentence. Second sentence.\n\nThird paragraph has enough text to continue."
    spans = chunk_text(text, ChunkingConfig(chunk_size=38, overlap=4))

    assert len(spans) >= 2
    assert all(0 < end - start <= 38 for start, end in spans)
    assert text[spans[0][0] : spans[0][1]].endswith("\n\n")
    assert spans[0][1] - spans[1][0] >= 0


def test_chunking_handles_unicode_and_text_without_sentence_punctuation() -> None:
    text = "你好世界 café naïve résumé 漢字 " * 9
    spans = chunk_text(text, ChunkingConfig(chunk_size=24, overlap=5))

    assert len(spans) > 1
    assert all(text[start:end] for start, end in spans)
    assert all(0 <= start < end <= len(text) for start, end in spans)
    assert all(spans[index][0] < spans[index + 1][0] for index in range(len(spans) - 1))


def test_chunking_recognizes_unicode_sentence_boundaries() -> None:
    text = (
        "\u7b2c\u4e00\u6587\u3002\u7b2c\u4e8c\u6587\u3002"
        "\u7b2c\u4e09\u6587\u3002\u7b2c\u56db\u6587\u3002"
    )
    spans = chunk_text(text, ChunkingConfig(chunk_size=8, overlap=1))

    assert len(spans) > 1
    assert any(text[start:end].endswith("\u3002") for start, end in spans[:-1])


def test_long_unbroken_token_is_split_with_bounded_chunks_and_progress() -> None:
    text = "x" * 57
    spans = chunk_text(text, ChunkingConfig(chunk_size=12, overlap=3))

    assert len(spans) > 1
    assert all(0 < end - start <= 12 for start, end in spans)
    assert all(spans[index][0] < spans[index + 1][0] for index in range(len(spans) - 1))
    assert len({(start, end) for start, end in spans}) == len(spans)


def test_chunk_offsets_metadata_and_order_are_preserved() -> None:
    result = ingest_text(
        "Alpha paragraph with some text.\n\nBeta paragraph with other text.",
        source_type="markdown",
        title="Notes",
        filename="notes.md",
        metadata={"workspace": "research", "labels": ["one"]},
        chunking=ChunkingConfig(chunk_size=25, overlap=5),
    )

    assert result.document.text == normalize_text(
        "Alpha paragraph with some text.\n\nBeta paragraph with other text."
    )
    assert result.document.source_type == "markdown"
    assert result.document.content_hash == content_hash(result.document.text)
    assert [chunk.index for chunk in result.chunks] == list(range(len(result.chunks)))
    assert all(chunk.document_id == result.document.id for chunk in result.chunks)
    assert all(
        result.document.text[chunk.start_char : chunk.end_char] == chunk.text
        for chunk in result.chunks
    )
    assert all(chunk.metadata == result.document.metadata for chunk in result.chunks)
    assert all(
        left.start_char < right.start_char and left.end_char >= right.start_char
        for left, right in zip(result.chunks, result.chunks[1:], strict=False)
    )


def test_document_identity_is_stable_for_source_and_fallbacks_to_content() -> None:
    first = ingest_text(
        "Original text", source_type="webpage", source_url="https://example.test/page"
    ).document
    updated = ingest_text(
        "Updated text", source_type="webpage", source_url="https://example.test/page"
    ).document
    same = ingest_text(
        "Original text", source_type="webpage", source_url="https://example.test/page"
    ).document
    content_addressed = ingest_text("Original text", source_type="text").document
    content_addressed_again = ingest_text("Original text", source_type="text").document

    assert first.id == same.id == updated.id
    assert first.content_hash != updated.content_hash
    assert content_addressed.id == content_addressed_again.id


def test_empty_document_has_hash_and_no_chunks() -> None:
    result = ingest_text("  \n", source_type="text")

    assert result.document.text == ""
    assert result.document.content_hash == content_hash("")
    assert result.chunks == []


def test_chunk_document_reuses_document_metadata() -> None:
    document = ingest_text("Some text for a chunk.", source_type="text", metadata={"tag": "x"})
    chunks = chunk_document(document.document, ChunkingConfig(chunk_size=8, overlap=2))

    assert chunks
    assert all(chunk.metadata == {"tag": "x"} for chunk in chunks)
