"""Document-chat route tests use mocked storage, retrieval, and language models."""

from contextlib import nullcontext
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.documents import Document, DocumentChunk, IngestionResult
from app.llm import LLMSettings
from app.models import Base
from app.rag import PreparedAnswerContext, RagOrchestrator
from app.repository import DocumentRepository
from app.retrieval import RetrievedChunk

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


def test_document_stats_returns_persisted_chunk_and_page_metadata(monkeypatch) -> None:
    document_id = str(uuid4())
    configure_document(monkeypatch, stored_document(document_id))

    response = client.get(f"/documents/{document_id}/stats")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "filename": "source.pdf",
        "source_type": "pdf",
        "indexed_chunk_count": 1,
        "chunks": [
            {
                "chunk_id": f"{document_id}:chunk:0",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 19,
                "page_numbers": [3],
                "page_start": 3,
                "page_end": 3,
            }
        ],
    }


def test_document_stats_hides_missing_and_webpage_records(monkeypatch) -> None:
    document_id = str(uuid4())
    webpage_id = str(uuid4())
    configure_document(monkeypatch, stored_document(webpage_id, source_type="webpage"))

    missing = client.get(f"/documents/{document_id}/stats")
    webpage = client.get(f"/documents/{webpage_id}/stats")

    assert missing.status_code == 404
    assert webpage.status_code == 404


def test_document_diagnostic_ask_uses_database_metadata_without_llm(monkeypatch) -> None:
    document_id = str(uuid4())
    document = stored_document(document_id)
    configure_document(monkeypatch, document)
    citation = {
        "id": "S1",
        "document_id": document_id,
        "chunk_id": document.chunks[0].id,
        "chunk_index": 0,
        "chunk_metadata": document.chunks[0].metadata,
        "text": "Stored source text.",
    }

    class FakeRag:
        def prepare(self, _question, page, **_kwargs):
            return PreparedAnswerContext(page, [citation], True)

    def llm_must_not_run():
        pytest.fail("Document diagnostics must be answered from persisted metadata.")

    monkeypatch.setattr(main, "rag_orchestrator", FakeRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    monkeypatch.setattr(main, "get_answer_service", llm_must_not_run)
    response = client.post(
        "/ask",
        json={
            "question": "How many chunks are indexed? List their chunk IDs and page ranges.",
            "document_id": document_id,
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Indexed chunks in this document: 1." in answer
    assert "Chunks retrieved for this question: 1." in answer
    assert document.chunks[0].id in answer
    assert "pages 3" in answer
    assert response.json().get("sources") is None


def test_streamed_document_diagnostic_uses_metadata_without_groq(monkeypatch) -> None:
    document_id = str(uuid4())
    document = stored_document(document_id)
    configure_document(monkeypatch, document)
    chunk = document.chunks[0]
    source = {
        "id": "S1",
        "document_id": document_id,
        "chunk_id": chunk.id,
        "chunk_index": chunk.index,
        "chunk_metadata": chunk.metadata,
        "text": chunk.text,
    }

    class FakeRag:
        def prepare(self, _question, page, **_kwargs):
            return PreparedAnswerContext(page, [source], True)

    async def groq_must_not_run(*_args, **_kwargs):
        pytest.fail("Document diagnostics must not be sent to Groq.")

    monkeypatch.setattr(main, "rag_orchestrator", FakeRag())
    monkeypatch.setattr(main, "get_settings", lambda: LLMSettings("test-key", "test-model", 2, 64))
    monkeypatch.setattr(main, "stream_answer", groq_must_not_run)
    response = client.post(
        "/ask/stream",
        json={"question": "What are the chunk IDs and page ranges?", "document_id": document_id},
    )

    assert response.status_code == 200
    assert 'event: token\ndata: {"text": "Indexed chunks in this document: 1.' in response.text
    assert chunk.id in response.text
    assert "pages 3" in response.text
    assert "event: done" in response.text


def test_unique_fact_upload_extraction_retrieval_and_grounding(monkeypatch, caplog) -> None:
    """Exercise upload persistence through document-scoped RAG and citation validation."""
    from app import document_ingestion

    caplog.set_level("INFO")
    fact = "The calibration key is ORCHID-7319 and the safe temperature is 17 C."
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    class FakeEmbeddingService:
        model_version = "test:document-chat:dimensions=3"
        dimensions = 3

        @staticmethod
        def embed_texts(texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(document_ingestion, "get_ingestion_session_factory", lambda: factory)
    monkeypatch.setattr(
        document_ingestion,
        "get_ingestion_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    monkeypatch.setattr(main, "get_ingestion_session_factory", lambda: factory)
    upload = client.post(
        "/documents/upload",
        files={"file": ("calibration.txt", fact.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 200
    document_id = upload.json()["document_id"]
    scopes = []

    def retrieve(_question, *, document_id, document_ids, source_url):
        scopes.append((document_id, document_ids, source_url))
        if document_ids is not None or source_url is not None:
            return []
        with factory() as session:
            stored = DocumentRepository(session).get_by_id(document_id)
        if stored is None:
            return []
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document_id,
                chunk_index=chunk.index,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                title=stored.document.title,
                source_type=stored.document.source_type,
                source_url=None,
                filename=stored.document.filename,
                document_metadata=stored.document.metadata,
                chunk_metadata=chunk.metadata,
                similarity=0.91,
            )
            for chunk in stored.chunks
            if document_id == upload.json()["document_id"]
        ]

    class FakeAnswer:
        def answer(self, question, page, history, sources):
            assert "ORCHID-7319" in sources[0]["text"]
            assert page["text"] == ""
            return "The calibration key is ORCHID-7319, and the safe temperature is 17 C. [S1]"

    monkeypatch.setattr(main, "rag_orchestrator", RagOrchestrator(retrieve))
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    monkeypatch.setattr(main, "get_history_limits", lambda: (10, 12000))
    monkeypatch.setattr(main, "get_answer_service", lambda: FakeAnswer())
    response = client.post(
        "/ask", json={"question": "What is the calibration key?", "document_id": document_id}
    )

    assert response.status_code == 200
    assert "ORCHID-7319" in response.json()["answer"]
    assert response.json()["sources"][0]["document_id"] == document_id
    assert scopes == [(document_id, None, None)]
    assert "extracted_chars=68 chunk_count=1" in caplog.text
    assert f"document_indexed document_id={document_id} indexing_status=indexed" in caplog.text
    assert "scope_type': 'document'" in caplog.text
    assert upload.json()["document_id"] in caplog.text
    assert "chunk_id': '" in caplog.text
    assert "source_id': 'S1'" in caplog.text
    assert "similarity': 0.91" in caplog.text
    assert "evidence_context_chars=" in caplog.text
    assert "reason=valid_citations" in caplog.text
    assert "ORCHID-7319" not in caplog.text
    assert fact not in caplog.text


def test_uncited_document_answer_reports_validation_reason_and_abstains(
    monkeypatch, caplog
) -> None:
    caplog.set_level("INFO")
    document_id = str(uuid4())
    configure_document(monkeypatch, stored_document(document_id))
    citation = {
        "id": "S1",
        "document_id": document_id,
        "chunk_id": f"{document_id}:chunk:0",
        "title": "Test source",
        "source_type": "text",
        "filename": "source.txt",
        "chunk_metadata": {},
        "text": "An unlogged private fact.",
    }

    class FakeRag:
        def prepare(self, _question, page, **_kwargs):
            return PreparedAnswerContext(page, [citation], True)

    class FakeAnswer:
        def answer(self, *_args):
            return "The answer is 42."

    monkeypatch.setattr(main, "rag_orchestrator", FakeRag())
    monkeypatch.setattr(main, "get_context_max_chars", lambda: 16000)
    monkeypatch.setattr(main, "get_history_limits", lambda: (10, 12000))
    monkeypatch.setattr(main, "get_answer_service", lambda: FakeAnswer())
    response = client.post("/ask", json={"question": "What is it?", "document_id": document_id})

    assert response.status_code == 200
    assert response.json()["answer"] == main._SAFE_GROUNDED_ABSTENTION
    assert "reason=missing_citations" in caplog.text
    assert "An unlogged private fact" not in caplog.text


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
