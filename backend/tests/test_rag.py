"""RAG orchestration tests use injected retrieval results and never call hosted APIs."""

from app.rag import MAX_ADAPTIVE_RETRY_QUERIES, RagOrchestrator
from app.retrieval import RetrievalDatabaseError, RetrievedChunk

PAGE = {
    "title": "Current page",
    "url": "https://example.test/current",
    "text": "Current page has a short summary.",
}


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc-1:chunk:0",
        document_id="doc-1",
        chunk_index=0,
        text="The document states an important fact.",
        start_char=0,
        end_char=39,
        title="Indexed source",
        source_type="webpage",
        source_url=PAGE["url"],
        filename=None,
        document_metadata={"owner": "test"},
        chunk_metadata={"section": "summary"},
        similarity=0.92,
    )


def test_rag_prepares_source_citations_and_restricts_to_active_scope() -> None:
    captured = {}

    def retrieve(question, *, document_id, document_ids, source_url):
        captured.update(
            question=question,
            document_id=document_id,
            document_ids=document_ids,
            source_url=source_url,
        )
        return [_chunk()]

    prepared = RagOrchestrator(retrieve).prepare(
        "What is the fact?", PAGE, context_limit=1000, document_id="doc-1"
    )

    assert captured == {
        "question": "What is the fact?",
        "document_id": "doc-1",
        "document_ids": None,
        "source_url": None,
    }
    assert prepared.retrieval_used is True
    assert prepared.page_context == PAGE
    assert prepared.retrieved_sources == [
        {
            "id": "S1",
            "chunk_id": "doc-1:chunk:0",
            "document_id": "doc-1",
            "chunk_index": 0,
            "title": "Indexed source",
            "source_type": "webpage",
            "source_url": PAGE["url"],
            "filename": None,
            "similarity": 0.92,
            "text": "The document states an important fact.",
            "chunk_metadata": {"section": "summary"},
        }
    ]


def test_page_url_scopes_retrieval_and_empty_index_uses_page_fallback() -> None:
    scopes = []

    def retrieve(_question, *, document_id, document_ids, source_url):
        scopes.append((document_id, document_ids, source_url))
        return []

    prepared = RagOrchestrator(retrieve).prepare("Ask", PAGE, context_limit=1000)

    assert scopes == [(None, None, PAGE["url"])]
    assert prepared.retrieval_used is False
    assert prepared.retrieved_sources is None
    assert prepared.fallback_code == "no_indexed_content"
    assert prepared.page_context == PAGE


def test_retrieval_database_failure_uses_selected_context_without_leaking_details() -> None:
    def fail(_question, *, document_id, document_ids, source_url):
        raise RetrievalDatabaseError("retrieval_database_error", "private database connection URL")

    long_page = {**PAGE, "text": "Intro paragraph.\n\n" + ("More page information.\n\n" * 200)}
    prepared = RagOrchestrator(fail).prepare("Ask about information", long_page, context_limit=1000)

    assert prepared.retrieval_used is False
    assert prepared.retrieved_sources is None
    assert prepared.fallback_code == "retrieval_database_error"
    assert len(prepared.page_context["text"]) <= 1000


def test_disabled_retrieval_uses_fallback_without_calling_retriever() -> None:
    def unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("retriever must not run without indexed chunks")

    prepared = RagOrchestrator(unexpected_retrieval).prepare(
        "Ask", PAGE, context_limit=1000, retrieval_enabled=False, fallback_code="index_failed"
    )

    assert prepared.retrieval_used is False
    assert prepared.retrieved_sources is None
    assert prepared.fallback_code == "index_failed"
    assert prepared.page_context == PAGE


def test_simple_question_uses_one_normal_retrieval_call() -> None:
    calls = []

    def retrieve(question, *, document_id, document_ids, source_url):
        calls.append((question, document_id, document_ids, source_url))
        return [_chunk()]

    prepared = RagOrchestrator(retrieve).prepare(
        "What is the important fact?", PAGE, context_limit=1000
    )

    assert prepared.retrieval_used
    assert len(calls) == 1


def test_multi_part_question_retries_only_missing_facets_in_same_scope() -> None:
    calls = []
    first = _chunk().model_copy(update={"text": "The valve is made from brass."})
    second = _chunk().model_copy(
        update={
            "chunk_id": "doc-1:chunk:1",
            "chunk_index": 1,
            "text": "The valve is inspected monthly.",
        }
    )

    def retrieve(question, *, document_id, document_ids, source_url):
        calls.append((question, document_id, document_ids, source_url))
        if question.startswith("What is it made"):
            return [first]
        return [second]

    prepared = RagOrchestrator(retrieve).prepare(
        "What is it made of and how often is it inspected?",
        PAGE,
        context_limit=1000,
        document_id="doc-1",
    )

    assert prepared.retrieval_used
    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:] == ("doc-1", None, None)
    assert {source["chunk_id"] for source in prepared.retrieved_sources or []} == {
        first.chunk_id,
        second.chunk_id,
    }


def test_adaptive_retry_round_is_bounded_to_three_queries() -> None:
    calls = []

    def retrieve(question, *, document_id, document_ids, source_url):
        calls.append(question)
        return []

    prepared = RagOrchestrator(retrieve).prepare(
        "What is A? What is B? What is C? What is D?",
        PAGE,
        context_limit=1000,
    )

    assert not prepared.retrieval_used
    assert len(calls) <= 1 + MAX_ADAPTIVE_RETRY_QUERIES


def test_explicit_empty_document_scope_never_falls_back_to_active_page() -> None:
    scopes = []

    def retrieve(_question, *, document_id, document_ids, source_url):
        scopes.append((document_id, document_ids, source_url))
        return []

    prepared = RagOrchestrator(retrieve).prepare("Ask", PAGE, context_limit=1000, document_ids=[])

    assert scopes == [(None, [], None)]
    assert not prepared.retrieval_used
    assert prepared.fallback_code == "no_indexed_content"


def test_explicit_document_list_is_forwarded_without_page_scope() -> None:
    scopes = []
    selected_documents = ["document-a", "document-b"]

    def retrieve(_question, *, document_id, document_ids, source_url):
        scopes.append((document_id, document_ids, source_url))
        return []

    RagOrchestrator(retrieve).prepare(
        "Compare the selected documents",
        PAGE,
        context_limit=1000,
        document_ids=selected_documents,
    )

    assert scopes == [(None, selected_documents, None)]
