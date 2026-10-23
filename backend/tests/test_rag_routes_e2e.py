"""Route-to-index-to-retrieval tests with local persistence and mock providers."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import llm, main, page_indexing
from app.embeddings import EmbeddingProviderError
from app.models import Base
from app.rag import RagOrchestrator
from app.repository import DocumentRepository
from app.retrieval import RetrievalDatabaseError, RetrievedChunk

ACTIVE_URL = "https://active.example/article"
UNRELATED_URL = "https://unrelated.example/article"


class MockEmbeddingService:
    model_version = "pageglance-test:route:dimensions=3"
    dimensions = 3

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.events.append("embedding")
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


class MockGroqCompletions:
    def __init__(self, events: list[str], captured: list[list[dict[str, str]]]) -> None:
        self.events = events
        self.captured = captured

    def create(self, *, messages, **_kwargs):
        self.events.append("llm")
        self.captured.append(messages)
        answer = (
            "Supported answer [S1]."
            if messages[0]["content"] == llm.RAG_SYSTEM_INSTRUCTIONS
            else "The supplied page does not provide enough evidence."
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])


def install_route_dependencies(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    events: list[str] = []
    embedding = MockEmbeddingService(events)
    monkeypatch.setattr(page_indexing, "_session_factory", lambda: factory)
    monkeypatch.setattr(page_indexing, "_embedding_service", lambda: embedding)

    retrieved_documents: list[tuple[str, str]] = []

    def retrieve(_question, *, document_id, document_ids, source_url):
        events.append("retrieval")
        assert document_id is not None
        assert document_ids is None
        assert source_url is None
        with factory() as session:
            indexed = DocumentRepository(session).get_by_id(document_id)
            assert indexed is not None, "retrieval must run after persistence commits"
            document = indexed.document
            retrieved_documents.append((document.id, document.source_url))
            return [
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    title=document.title,
                    source_type=document.source_type,
                    source_url=document.source_url,
                    filename=document.filename,
                    document_metadata=document.metadata,
                    chunk_metadata=chunk.metadata,
                    similarity=0.93,
                )
                for chunk in indexed.chunks
            ]

    monkeypatch.setattr(main, "rag_orchestrator", RagOrchestrator(retrieve))
    llm_messages: list[list[dict[str, str]]] = []
    completions = MockGroqCompletions(events, llm_messages)
    answer_service = llm.GroqAnswerService(
        llm.LLMSettings("test-key", "mock-model", 2, 64),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        ),
    )
    monkeypatch.setattr(main, "get_answer_service", lambda: answer_service)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: llm.LLMSettings("test-key", "mock-model", 2, 64),
    )

    async def mock_stream(question, page_context, _settings, history, retrieved_sources):
        events.append("llm")
        messages = llm._build_messages(question, page_context, history, retrieved_sources)
        llm_messages.append(messages)
        yield (
            "Supported answer [S1]."
            if messages[0]["content"] == llm.RAG_SYSTEM_INSTRUCTIONS
            else "The supplied page does not provide enough evidence."
        )

    monkeypatch.setattr(main, "stream_answer", mock_stream)
    return TestClient(main.app), factory, embedding, events, retrieved_documents, llm_messages


def _ask(client: TestClient, endpoint: str, text: str, url: str = ACTIVE_URL):
    return client.post(
        endpoint,
        json={
            "question": "What does the article say about PageGlance?",
            "page": {"title": "Active article", "url": url, "text": text},
        },
    )


def _payload(messages: list[dict[str, str]]) -> dict[str, object]:
    return json.loads(messages[-1]["content"].split("\n", 1)[1])


def test_ask_and_stream_index_before_scoped_retrieval_and_reuse_then_reindex(monkeypatch):
    client, factory, embedding, events, retrieved, llm_messages = install_route_dependencies(
        monkeypatch
    )
    active_text = "The active article says PageGlance stores webpage chunks for search."
    unrelated_text = "The unrelated article discusses mountain climbing and weather."

    unrelated = page_indexing.index_page_context(
        {"title": "Other article", "url": UNRELATED_URL, "text": unrelated_text}
    )
    assert unrelated.indexed
    events.clear()

    first = _ask(client, "/ask", active_text)
    assert first.status_code == 200
    assert first.json()["answer"] == "Supported answer [S1]."
    assert first.json()["sources"][0]["source_url"] == ACTIVE_URL
    assert events.index("embedding") < events.index("retrieval") < events.index("llm")
    first_embedding_calls = len(embedding.calls)
    first_document_id = retrieved[-1][0]
    assert retrieved[-1][1] == ACTIVE_URL
    grounded_payload = _payload(llm_messages[-1])
    assert llm_messages[-1][0]["content"] == llm.RAG_SYSTEM_INSTRUCTIONS
    assert grounded_payload["retrieved_sources"][0]["text"] == active_text
    assert UNRELATED_URL not in json.dumps(grounded_payload)

    repeated = _ask(client, "/ask/stream", active_text)
    assert repeated.status_code == 200
    assert repeated.headers["content-type"].startswith("text/event-stream")
    assert 'event: token\ndata: {"text": "Supported answer [S1]."}\n\n' in repeated.text
    assert "event: done" in repeated.text
    assert len(embedding.calls) == first_embedding_calls
    assert retrieved[-1] == (first_document_id, ACTIVE_URL)

    changed_text = "The active article now says PageGlance also refreshes changed page chunks."
    changed = _ask(client, "/ask/stream", changed_text)
    assert changed.status_code == 200
    assert len(embedding.calls) > first_embedding_calls
    assert retrieved[-1] == (first_document_id, ACTIVE_URL)
    changed_payload = _payload(llm_messages[-1])
    assert changed_payload["retrieved_sources"][0]["text"] == changed_text

    confirmed = _ask(client, "/ask", changed_text)
    assert confirmed.status_code == 200
    assert len(embedding.calls) == first_embedding_calls + 1
    assert retrieved[-1] == (first_document_id, ACTIVE_URL)

    with factory() as session:
        stored = DocumentRepository(session).get_by_id(first_document_id)
        assert stored is not None
        assert stored.document.text == changed_text
        assert stored.document.source_url == ACTIVE_URL


@pytest.mark.parametrize("failure", ["index", "retrieval"])
def test_ask_and_stream_fallback_prompts_do_not_claim_rag_when_unavailable(
    monkeypatch, failure: str
) -> None:
    client, _factory, embedding, _events, _retrieved, llm_messages = install_route_dependencies(
        monkeypatch
    )
    if failure == "index":

        class FailingEmbeddingService:
            model_version = MockEmbeddingService.model_version
            dimensions = 3

            def embed_texts(self, _texts):
                raise EmbeddingProviderError("embedding_provider_error", "private error")

        monkeypatch.setattr(page_indexing, "_embedding_service", lambda: FailingEmbeddingService())
    else:
        monkeypatch.setattr(
            main,
            "rag_orchestrator",
            RagOrchestrator(
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RetrievalDatabaseError("retrieval_database_error", "private error")
                )
            ),
        )

    page_text = "Page content remains available as a safe fallback."
    answer = _ask(client, "/ask", page_text)
    stream = _ask(client, "/ask/stream", page_text)

    assert answer.status_code == 200
    assert answer.json() == {"answer": "The supplied page does not provide enough evidence."}
    assert stream.status_code == 200
    assert "[S1]" not in stream.text
    assert "event: done" in stream.text
    for messages in llm_messages[-2:]:
        assert messages[0]["content"] == llm.SYSTEM_INSTRUCTIONS
        payload = _payload(messages)
        assert "retrieved_sources" not in payload
    if failure == "index":
        assert embedding.calls == []
