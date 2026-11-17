"""Focused regression tests for token-aware retrieval-context trimming.

These verify that oversized retrieved context is reduced by dropping whole,
least-relevant chunks (never mid-chunk truncation), that retained chunk
metadata/citation fields survive intact, and that the trim honours a budget
that accounts for the retrieved prompt, question, history, and output tokens.
"""

from app.main import _bounded_retrieved_sources


def _source(source_id: str, text: str) -> dict[str, object]:
    return {
        "id": source_id,
        "chunk_id": f"chunk-{source_id}",
        "document_id": "doc-1",
        "chunk_index": int(source_id[1:]) - 1,
        "title": "Title",
        "source_type": "webpage",
        "source_url": "https://example.com/doc",
        "filename": "doc.txt",
        "similarity": 0.9,
        "text": text,
        "chunk_metadata": {"page_numbers": [1]},
    }


def _budget(**overrides) -> dict[str, object]:
    params: dict[str, object] = {
        "context_limit": 20_000,
        "page_text_chars": 1_000,
        "question": "What is the fact?",
        "history_chars": 500,
        "max_completion_tokens": 512,
    }
    params.update(overrides)
    return params


def test_oversized_retrieved_context_drops_least_relevant_whole_chunks() -> None:
    # Two big chunks that cannot both fit the available budget.
    sources = [
        _source("S1", "A" * 6_000),
        _source("S2", "B" * 6_000),
        _source("S3", "C" * 6_000),
    ]
    kept = _bounded_retrieved_sources(sources, **_budget())
    assert isinstance(kept, list)
    assert len(kept) < len(sources)
    # Order preserved, most relevant (first) retained.
    assert kept[0]["id"] == "S1"
    # Every retained chunk is complete and untruncated.
    assert all(src["text"] in {s["text"] for s in sources} for src in kept)
    assert all(len(str(src["text"])) == 6_000 for src in kept)


def test_retained_chunks_preserve_metadata_and_citation_fields() -> None:
    sources = [
        _source("S1", "A" * 2_000),
        _source("S2", "B" * 2_000),
    ]
    kept = _bounded_retrieved_sources(sources, **_budget())
    assert isinstance(kept, list)
    for src in kept:
        # Full citation identity and metadata must survive trimming.
        assert src["id"] in {"S1", "S2"}
        assert src["chunk_id"] == f"chunk-{src['id']}"
        assert src["document_id"] == "doc-1"
        assert src["chunk_metadata"] == {"page_numbers": [1]}
        assert src["source_url"] == "https://example.com/doc"


def test_all_chunks_kept_when_budget_accommodates_them() -> None:
    sources = [_source("S1", "short one"), _source("S2", "short two")]
    kept = _bounded_retrieved_sources(sources, **_budget())
    assert kept == sources


def test_no_sources_returns_none() -> None:
    assert _bounded_retrieved_sources(None, **_budget()) is None
    assert _bounded_retrieved_sources([], **_budget()) == []


def test_budget_accounts_for_system_question_history_and_output_tokens() -> None:
    # A larger reserved output-token budget leaves less room for retrieved evidence.
    sources = [
        _source("S1", "A" * 1_000),
        _source("S2", "B" * 1_000),
        _source("S3", "C" * 1_000),
    ]
    common = dict(context_limit=9_000, page_text_chars=0, question="Q", history_chars=0)
    kept_small_reserve = _bounded_retrieved_sources(sources, **common, max_completion_tokens=256)
    kept_large_reserve = _bounded_retrieved_sources(sources, **common, max_completion_tokens=1024)
    assert isinstance(kept_small_reserve, list)
    assert isinstance(kept_large_reserve, list)
    assert len(kept_large_reserve) < len(kept_small_reserve)
    # Most relevant chunks are always retained first.
    assert kept_small_reserve[0]["id"] == "S1"
    assert kept_large_reserve[0]["id"] == "S1"
