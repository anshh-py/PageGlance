"""Focused regression tests for token-aware retrieval-context trimming.

These verify that oversized retrieved context is reduced by dropping whole,
least-relevant chunks (never mid-chunk truncation), that retained chunk
metadata/citation fields survive intact, and that the trim honours a shared
token budget accounting for the system prompt, selected page text, question,
history, reserved output tokens, and a safety margin.
"""

import pytest
from fastapi import HTTPException

from app import main
from app.main import _bounded_retrieved_sources, _budget_bounded_context


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
        "page_text": "A" * 1_000,
        "question": "What is the fact?",
        "history": [],
        "max_completion_tokens": 512,
        "model_input_tokens": 16_000,
    }
    params.update(overrides)
    return params


@pytest.fixture(autouse=True)
def _deterministic_tokens(monkeypatch):
    """Make token estimates deterministic for these tests.

    The fallback estimator uses 3 chars per token; we stub it so assertions are
    independent of whether tiktoken is installed.
    """

    def _estimate(text):
        if not text:
            return 0
        return max(1, (len(text) + 2) // 3)

    monkeypatch.setattr(main, "estimate_tokens", _estimate)


def test_oversized_retrieved_context_drops_least_relevant_whole_chunks() -> None:
    # Two big chunks that cannot both fit the available budget.
    sources = [
        _source("S1", "A" * 6_000),
        _source("S2", "B" * 6_000),
        _source("S3", "C" * 6_000),
    ]
    kept = _bounded_retrieved_sources(
        sources, page_text="", question="Q", history=[], max_completion_tokens=128,
        model_input_tokens=4_000,
    )
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
    kept = _bounded_retrieved_sources(sources, **_budget(model_input_tokens=1_000_000))
    assert kept == sources


def test_no_sources_returns_none() -> None:
    assert _bounded_retrieved_sources(None, **_budget()) is None
    assert _bounded_retrieved_sources([], **_budget()) == []


def test_budget_accounts_for_page_text_question_history_and_output_tokens() -> None:
    # A larger reserved output-token budget leaves less room for retrieved evidence.
    sources = [
        _source("S1", "A" * 1_200),
        _source("S2", "B" * 1_200),
        _source("S3", "C" * 1_200),
    ]
    common = dict(
        page_text="",
        question="Q",
        history=[],
        model_input_tokens=2_500,
    )
    kept_small_reserve = _bounded_retrieved_sources(sources, **common, max_completion_tokens=128)
    kept_large_reserve = _bounded_retrieved_sources(
        sources, **common, max_completion_tokens=4096
    )
    assert isinstance(kept_small_reserve, list)
    assert isinstance(kept_large_reserve, list)
    assert len(kept_large_reserve) < len(kept_small_reserve)
    # Most relevant chunks are always retained first.
    assert kept_small_reserve[0]["id"] == "S1"
    if kept_large_reserve:
        assert kept_large_reserve[0]["id"] == "S1"


def test_budget_accounts_for_page_text_volume() -> None:
    # A larger selected page eats into the budget available for retrieved evidence.
    sources = [
        _source("S1", "A" * 1_200),
        _source("S2", "B" * 1_200),
    ]
    small_page = _bounded_retrieved_sources(
        sources, page_text="", question="Q", history=[], max_completion_tokens=512,
        model_input_tokens=3_000,
    )
    large_page = _bounded_retrieved_sources(
        sources, page_text="P" * 6_000, question="Q", history=[],
        max_completion_tokens=512, model_input_tokens=3_000,
    )
    assert len(large_page) < len(small_page)
    assert small_page[0]["id"] == "S1"


def test_budget_gate_raises_recoverable_error_when_mandatory_exceeds(
    monkeypatch,
) -> None:
    """Mandatory content alone over budget -> clear recoverable 422, never an
    oversized silent provider request."""
    monkeypatch.setattr(main, "get_model_input_tokens", lambda: 4_000)
    with pytest.raises(HTTPException) as excinfo:
        _budget_bounded_context(
            None,
            page_text="P" * 50_000,
            question="Q",
            history=[],
            max_completion_tokens=512,
        )
    assert excinfo.value.status_code == 422
    assert "model's input limit" in excinfo.value.detail


def test_budget_gate_is_shared_with_trimming_mechanism(monkeypatch) -> None:
    """The shared gate delegates to the same token trimmer both routes use, so
    ask and ask_stream prepare retrieved context identically."""
    monkeypatch.setattr(main, "get_model_input_tokens", lambda: 4_000)
    sources = [
        _source("S1", "A" * 2_000),
        _source("S2", "B" * 2_000),
        _source("S3", "C" * 2_000),
    ]
    direct = _bounded_retrieved_sources(
        sources,
        page_text="",
        question="Q",
        history=[],
        max_completion_tokens=128,
        model_input_tokens=4_000,
    )
    gated = _budget_bounded_context(
        sources,
        page_text="",
        question="Q",
        history=[],
        max_completion_tokens=128,
    )
    assert isinstance(gated, list)
    assert gated == direct
    assert gated[0]["id"] == "S1"


def test_budget_gate_accounts_for_long_history_and_page(monkeypatch) -> None:
    """A very long history plus page triggers the recoverable error, proving the
    gate budgets the entire assembled request, not just retrieved chunks."""
    monkeypatch.setattr(main, "get_model_input_tokens", lambda: 3_000)
    history = [
        {"role": "user", "content": "H" * 4_000},
        {"role": "assistant", "content": "A" * 4_000},
    ]
    with pytest.raises(HTTPException) as excinfo:
        _budget_bounded_context(
            [],
            page_text="P" * 6_000,
            question="Q",
            history=history,
            max_completion_tokens=512,
        )
    assert excinfo.value.status_code == 422
