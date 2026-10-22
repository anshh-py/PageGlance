"""Document-chat route tests use mocked storage, retrieval, and language models."""

from contextlib import nullcontext
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import main
from app.documents import Document, DocumentChunk, IngestionResult
from app.llm import LLMSettings
from app.rag import PreparedAnswerContext

client = TestClient(main.app)


def stored_document(document_id: str, source_type: str = "pdf") -> IngestionResult:
    now = datetime.now(timezone.utc)
    document = Document(
        id=document_id,
        source_type=source_type,
        title="Test source",
        filename="source.pdf" if source_type == "pdf" else "source.txt",
        content_hash="a" * 64,
        text="stored content",
        created_at=now,
        updated_at=now,
    )
    chunk = DocumentChunk(
        id=f"{document_id}:chunk:0",
        document_id=document_id,
        index=0,
        text="Stored source text.",
        start_char=0,
        end_char=19,
        metadata={"page_numbers": [3], "page_start": 3, "page_end": 3},
    )
    return IngestionResult(document=document, chunks=[chunk])


def configure_document(monkeypatch, document: IngestionResult) -> None:
    monkeypatch.setattr(main, "get_ingestion_session_factory", lambda: lambda: nullcontext())

    class FakeRepository:
        def __init__(self, _session):
            pass

        def get_by_id(self, document_id):
            return document if document.document.id == document_id else None

    monkeypatch.setattr(main, "DocumentRepository", FakeRepository)


def test_document_ask_uses_exact_validated_scope_and_returns_page_citation(monkeypatch) -> None:
    document_id = str(uuid4())
    configure_document(monkeypatch, stored_document(document_id))
    retrieval_calls = []
    citation = {
        "id": "S1",
        "document_id": document_id,
        "chunk_id": f"{document_id}:chunk:0",
        "chunk_index": 0,
        "title": "Test source",
        "source_type": "pdf",
        "filename": "source.pdf",
        "source_url": None,
        "chunk_metadata": {"page_numbers": [3], "page_start": 3, "page_end": 3},
        "text": "Stored source text.",
    }

    class FakeRag:
        def prepare(self, question, page, *, context_limit, document_id, retrieval_enabled):
            retrieval_calls.append((question, page, document_id, retrieval_enabled))
            return PreparedAnswerContext(page, [citation], True)

    class FakeAnswer:
        def answer(self, question, page, history, sources):
            assert page == {"title": "source.pdf", "url": "", "text": ""}
            assert sources == [citation]
            return "The cited evidence is on page 3. [S1]"

    monkeypatch.setattr(main, "rag_orchestrator", FakeRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    monkeypatch.setattr(main, "get_history_limits", lambda: (10, 12000))
    monkeypatch.setattr(main, "get_answer_service", lambda: FakeAnswer())

    response = client.post(
        "/ask", json={"question": "What does it say?", "document_id": document_id}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The cited evidence is on page 3. [S1]",
        "sources": [
            {
                "id": "S1",
                "document_id": document_id,
                "title": "Test source",
                "source_type": "pdf",
                "filename": "source.pdf",
                "page_start": 3,
                "page_end": 3,
                "page_numbers": [3],
            }
        ],
    }
    assert retrieval_calls == [
        (
            "What does it say?",
            {"title": "source.pdf", "url": "", "text": ""},
            document_id,
            True,
        )
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "Q"},
        {
            "question": "Q",
            "page": {"title": "p", "url": "https://a.test", "text": "x"},
            "document_id": str(uuid4()),
        },
        {"question": "Q", "document_id": "not-a-valid-document-id"},
    ],
)
def test_document_request_scope_is_validated(body) -> None:
    response = client.post("/ask", json=body)
    assert response.status_code == 422


def test_document_chat_rejects_missing_or_non_upload_document(monkeypatch) -> None:
    unknown_id = str(uuid4())
    configure_document(monkeypatch, stored_document(str(uuid4()), source_type="webpage"))
    response = client.post("/ask", json={"question": "Q", "document_id": unknown_id})

    assert response.status_code == 404
    assert response.json()["detail"] == "The selected document was not found."


def test_document_chat_never_falls_back_when_retrieval_has_no_evidence(monkeypatch) -> None:
    document_id = str(uuid4())
    configure_document(monkeypatch, stored_document(document_id))

    class EmptyRag:
        def prepare(self, _question, page, **_kwargs):
            return PreparedAnswerContext(page, None, False, "no_indexed_content")

    monkeypatch.setattr(main, "rag_orchestrator", EmptyRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    response = client.post("/ask", json={"question": "Q", "document_id": document_id})

    assert response.status_code == 422
    assert (
        response.json()["detail"]
        == "No matching indexed content was found in the selected document."
    )


def test_document_chat_rejects_cross_document_retrieval_results(monkeypatch) -> None:
    document_id = str(uuid4())
    configure_document(monkeypatch, stored_document(document_id))
    unrelated_id = str(uuid4())

    class CrossScopeRag:
        def prepare(self, _question, page, **_kwargs):
            return PreparedAnswerContext(
                page,
                [{"document_id": unrelated_id, "id": "S1"}],
                True,
            )

    monkeypatch.setattr(main, "rag_orchestrator", CrossScopeRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    response = client.post("/ask", json={"question": "Q", "document_id": document_id})

    assert response.status_code == 503
    assert "scope" in response.json()["detail"]


def test_multi_document_chat_validates_and_preserves_explicit_scope(monkeypatch) -> None:
    document_ids = [str(uuid4()), str(uuid4())]
    documents = {item: stored_document(item) for item in document_ids}
    monkeypatch.setattr(main, "get_ingestion_session_factory", lambda: lambda: nullcontext())

    class FakeRepository:
        def __init__(self, _session):
            pass

        def get_by_id(self, document_id):
            return documents.get(document_id)

    monkeypatch.setattr(main, "DocumentRepository", FakeRepository)
    captured = {}
    sources = [
        {
            "id": f"S{index}",
            "document_id": document_id,
            "title": f"Source {index}",
            "source_type": "pdf",
            "filename": f"source-{index}.pdf",
            "source_url": None,
            "chunk_metadata": {"page_numbers": [index], "page_start": index, "page_end": index},
            "text": f"Evidence {index}",
        }
        for index, document_id in enumerate(document_ids, start=1)
    ]

    class FakeRag:
        def prepare(self, _question, page, *, context_limit, document_ids, retrieval_enabled):
            captured.update(
                page=page,
                context_limit=context_limit,
                document_ids=document_ids,
                retrieval_enabled=retrieval_enabled,
            )
            return PreparedAnswerContext(page, sources, True)

    class FakeAnswer:
        def answer(self, *_args):
            return "Evidence from both documents. [S1] [S2]"

    monkeypatch.setattr(main, "rag_orchestrator", FakeRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    monkeypatch.setattr(main, "get_history_limits", lambda: (10, 12000))
    monkeypatch.setattr(main, "get_answer_service", lambda: FakeAnswer())
    response = client.post("/ask", json={"question": "Compare these", "document_ids": document_ids})

    assert response.status_code == 200
    assert response.json()["answer"] == "Evidence from both documents. [S1] [S2]"
    assert [source["document_id"] for source in response.json()["sources"]] == document_ids
    assert [source["page_numbers"] for source in response.json()["sources"]] == [[1], [2]]
    assert captured["document_ids"] == document_ids
    assert captured["retrieval_enabled"] is True


def test_multi_document_chat_rejects_invalid_or_unselected_source(monkeypatch) -> None:
    first, second, unrelated = (str(uuid4()) for _ in range(3))
    documents = {first: stored_document(first), second: stored_document(second)}
    monkeypatch.setattr(main, "get_ingestion_session_factory", lambda: lambda: nullcontext())

    class FakeRepository:
        def __init__(self, _session):
            pass

        def get_by_id(self, document_id):
            return documents.get(document_id)

    class CrossScopeRag:
        def prepare(self, _question, page, **kwargs):
            return PreparedAnswerContext(
                page,
                [{"id": "S1", "document_id": unrelated}],
                True,
            )

    monkeypatch.setattr(main, "DocumentRepository", FakeRepository)
    monkeypatch.setattr(main, "rag_orchestrator", CrossScopeRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)

    missing = client.post("/ask", json={"question": "Q", "document_ids": [first, unrelated]})
    duplicate = client.post("/ask", json={"question": "Q", "document_ids": [first, first]})
    cross_scope = client.post("/ask", json={"question": "Q", "document_ids": [first, second]})

    assert missing.status_code == 404
    assert duplicate.status_code == 422
    assert cross_scope.status_code == 503


def test_document_stream_sends_citations_then_answer_tokens(monkeypatch) -> None:
    document_id = str(uuid4())
    configure_document(monkeypatch, stored_document(document_id))
    citations = [
        {
            "id": "S1",
            "document_id": document_id,
            "title": "Source",
            "source_type": "text",
            "filename": "source.txt",
            "source_url": None,
            "chunk_metadata": {"page_numbers": []},
            "text": "Evidence",
        }
    ]

    class FakeRag:
        def prepare(self, _question, page, **_kwargs):
            return PreparedAnswerContext(page, citations, True)

    async def fake_stream(*_args):
        yield "Grounded"

    monkeypatch.setattr(main, "rag_orchestrator", FakeRag())
    monkeypatch.setattr(main, "get_settings", lambda: LLMSettings("test-key", "test-model", 2, 64))
    monkeypatch.setattr(main, "stream_answer", fake_stream)
    response = client.post("/ask/stream", json={"question": "Q", "document_id": document_id})

    assert response.status_code == 200
    assert response.text.index("event: sources") < response.text.index("event: token")
    assert '"page_numbers": []' in response.text
    assert 'event: token\ndata: {"text": "Grounded"}' in response.text
    assert "event: done" in response.text
