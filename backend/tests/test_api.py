from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import AsyncGroq

from app import llm, main
from app.page_indexing import PageIndexResult
from app.rag import RagOrchestrator
from app.retrieval import RetrievedChunk

client = TestClient(main.app)
VALID_PAGE = {"title": "Example", "url": "https://example.com", "text": "Page text"}


class FakeAnswerService:
    def answer(
        self,
        question: str,
        page_context: dict[str, str],
        history: list[dict[str, str]] | None = None,
        retrieved_sources=None,
    ) -> str:
        assert question == "What is this page about?"
        assert page_context == {**VALID_PAGE, "url": "https://example.com/"}
        assert history == []
        assert retrieved_sources is None
        return "The page is an example."


def no_live_retrieval(monkeypatch) -> None:
    """Keep existing endpoint tests independent of hosted embedding credentials."""
    monkeypatch.setattr(main, "rag_orchestrator", RagOrchestrator(lambda *_args, **_kwargs: []))
    monkeypatch.setattr(
        main,
        "index_page_context",
        lambda page: PageIndexResult(True, page.get("document_id"), 1),
    )


@pytest.fixture(autouse=True)
def prevent_live_embedding_requests(monkeypatch) -> None:
    no_live_retrieval(monkeypatch)


def test_ask_returns_predictable_answer_schema(monkeypatch) -> None:
    no_live_retrieval(monkeypatch)
    monkeypatch.setattr(main, "get_answer_service", lambda: FakeAnswerService())

    response = client.post(
        "/ask",
        json={"question": " What is this page about? ", "page": VALID_PAGE},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "The page is an example."}


def test_ask_indexes_reuses_embeddings_and_keeps_url_scopes_separate(monkeypatch) -> None:
    state: dict[str, str] = {}
    embedded_hashes: set[str] = set()
    embed_calls: list[str] = []
    retrieval_scopes: list[str | None] = []

    def index_page(page):
        normalized_url = page["url"]
        page_hash = sha256(page["text"].encode()).hexdigest()
        if page_hash not in embedded_hashes:
            embedded_hashes.add(page_hash)
            embed_calls.append(page_hash)
        document_id = state.setdefault(
            normalized_url,
            str(uuid5(NAMESPACE_URL, f"pageglance-test:{normalized_url}")),
        )
        return PageIndexResult(True, document_id, 1)

    def retrieve(_question, *, document_id, source_url):
        assert source_url is None
        retrieval_scopes.append(document_id)
        return []

    monkeypatch.setattr(main, "index_page_context", index_page)
    monkeypatch.setattr(main, "rag_orchestrator", RagOrchestrator(retrieve))

    class AnswerService:
        def answer(self, *_args):
            return "Fallback answer from active page."

    monkeypatch.setattr(main, "get_answer_service", lambda: AnswerService())

    def ask_page(url: str, text: str) -> None:
        response = client.post(
            "/ask",
            json={
                "question": "What does this say?",
                "page": {"title": "Page", "url": url, "text": text},
            },
        )
        assert response.status_code == 200
        assert response.json() == {"answer": "Fallback answer from active page."}

    ask_page("HTTPS://EXAMPLE.COM:443/article#summary", "same page text")
    ask_page("https://example.com/article#summary", "same page text")
    ask_page("https://example.com/article#summary", "updated page text")
    ask_page("https://another.example/article#summary", "updated page text")

    assert len(embed_calls) == 2
    assert len(state) == 2
    assert len(set(retrieval_scopes)) == 2
    assert retrieval_scopes[0] == retrieval_scopes[1] == retrieval_scopes[2]
    assert retrieval_scopes[3] != retrieval_scopes[2]


def test_retrieval_scope_uses_indexed_active_page_not_request_document_id(monkeypatch) -> None:
    seen_page_ids = []
    scopes = []

    def index_page(page):
        seen_page_ids.append(page.get("document_id"))
        return PageIndexResult(True, "indexed-active-page", 1)

    def retrieve(_question, *, document_id, source_url):
        scopes.append((document_id, source_url))
        return []

    monkeypatch.setattr(main, "index_page_context", index_page)
    monkeypatch.setattr(main, "rag_orchestrator", RagOrchestrator(retrieve))

    class AnswerService:
        def answer(self, *_args):
            return "Answer from page context."

    monkeypatch.setattr(main, "get_answer_service", lambda: AnswerService())
    response = client.post(
        "/ask",
        json={
            "question": "Summarize",
            "page": {**VALID_PAGE, "document_id": "unrelated-document"},
        },
    )

    assert response.status_code == 200
    assert seen_page_ids == ["unrelated-document"]
    assert scopes == [("indexed-active-page", None)]


def test_indexing_failure_skips_retrieval_and_keeps_fallback_for_both_routes(monkeypatch) -> None:
    from app.retrieval import RetrievalDatabaseError

    retrieval_calls = []
    monkeypatch.setattr(
        main,
        "index_page_context",
        lambda _page: PageIndexResult(False, None, 0, "embedding_configuration_error"),
    )
    monkeypatch.setattr(
        main,
        "rag_orchestrator",
        RagOrchestrator(
            lambda *_args, **_kwargs: (
                retrieval_calls.append(True)
                or (_ for _ in ()).throw(RetrievalDatabaseError("bad", "private error"))
            )
        ),
    )

    class AnswerService:
        def answer(self, _question, _page, _history, sources):
            assert sources is None
            return "Page context fallback."

    monkeypatch.setattr(main, "get_answer_service", lambda: AnswerService())
    response = client.post("/ask", json={"question": "Summarize", "page": VALID_PAGE})
    assert response.status_code == 200
    assert response.json() == {"answer": "Page context fallback."}
    assert retrieval_calls == []

    async def fake_stream(*_args, **_kwargs):
        yield "Page context fallback."

    monkeypatch.setattr(main, "get_settings", lambda: llm.LLMSettings("test", "model", 2, 64))
    monkeypatch.setattr(main, "stream_answer", fake_stream)
    streamed = client.post("/ask/stream", json={"question": "Summarize", "page": VALID_PAGE})
    assert streamed.status_code == 200
    assert 'event: token\ndata: {"text": "Page context fallback."}\n\n' in streamed.text
    assert "event: done" in streamed.text
    assert retrieval_calls == []


def test_retrieval_failure_after_indexing_keeps_safe_page_fallback(monkeypatch) -> None:
    from app.retrieval import RetrievalDatabaseError

    monkeypatch.setattr(
        main, "index_page_context", lambda _page: PageIndexResult(True, "active-doc", 1)
    )

    def fail_retrieval(*_args, **_kwargs):
        raise RetrievalDatabaseError("retrieval_database_error", "private database detail")

    monkeypatch.setattr(main, "rag_orchestrator", RagOrchestrator(fail_retrieval))

    class AnswerService:
        def answer(self, _question, _page, _history, sources):
            assert sources is None
            return "Fallback answer."

    monkeypatch.setattr(main, "get_answer_service", lambda: AnswerService())
    response = client.post("/ask", json={"question": "Summarize", "page": VALID_PAGE})

    assert response.status_code == 200
    assert response.json() == {"answer": "Fallback answer."}


def test_ask_passes_scoped_retrieved_sources_and_history_to_answer_service(monkeypatch) -> None:
    chunk = RetrievedChunk(
        chunk_id="doc-1:chunk:0",
        document_id="doc-1",
        chunk_index=0,
        text="The indexed page states a grounded fact.",
        start_char=0,
        end_char=40,
        title="Indexed page",
        source_type="webpage",
        source_url=VALID_PAGE["url"],
        filename=None,
        document_metadata={},
        chunk_metadata={},
        similarity=0.9,
    )
    scopes = []
    monkeypatch.setattr(
        main,
        "rag_orchestrator",
        RagOrchestrator(
            lambda question, *, document_id, source_url: (
                scopes.append((question, document_id, source_url)) or [chunk]
            )
        ),
    )
    captured = {}

    class CapturingAnswerService:
        def answer(self, question, page_context, history, retrieved_sources):
            captured.update(
                question=question,
                page_context=page_context,
                history=history,
                retrieved_sources=retrieved_sources,
            )
            return "The fact is supported. [S1]"

    monkeypatch.setattr(main, "get_answer_service", lambda: CapturingAnswerService())
    monkeypatch.setattr(main, "get_history_limits", lambda: (10, 12_000))
    response = client.post(
        "/ask",
        json={
            "question": "What is the fact?",
            "page": {**VALID_PAGE, "document_id": "doc-1"},
            "history": [{"role": "user", "content": "What did I ask?"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "The fact is supported. [S1]"
    assert response.json()["sources"][0]["id"] == "S1"
    assert response.json()["sources"][0]["source_url"] == VALID_PAGE["url"]
    assert scopes == [("What is the fact?", "doc-1", None)]
    assert captured["history"] == [{"role": "user", "content": "What did I ask?"}]
    assert captured["retrieved_sources"][0]["id"] == "S1"
    assert captured["retrieved_sources"][0]["source_url"] == VALID_PAGE["url"]


@pytest.mark.parametrize("answer", ["The fact is supported.", "The fact is supported. [S99]"])
def test_ask_abstains_when_rag_answer_citations_are_missing_or_unknown(monkeypatch, answer):
    chunk = RetrievedChunk(
        chunk_id="doc-1:chunk:0",
        document_id="doc-1",
        chunk_index=0,
        text="Evidence for a grounded answer.",
        start_char=0,
        end_char=32,
        title="Evidence",
        source_type="webpage",
        source_url="https://example.com/",
        filename=None,
        document_metadata={},
        chunk_metadata={},
        similarity=0.9,
    )
    monkeypatch.setattr(
        main,
        "rag_orchestrator",
        RagOrchestrator(lambda *_args, **_kwargs: [chunk]),
    )

    class AnswerService:
        def answer(self, *_args):
            return answer

    monkeypatch.setattr(main, "get_answer_service", lambda: AnswerService())
    response = client.post("/ask", json={"question": "What is the fact?", "page": VALID_PAGE})

    assert response.status_code == 200
    assert response.json()["answer"] == main._SAFE_GROUNDED_ABSTENTION
    assert response.json()["sources"][0]["id"] == "S1"


def test_ask_preserves_explicit_abstention_for_unsupported_question(monkeypatch):
    chunk = RetrievedChunk(
        chunk_id="doc-1:chunk:0",
        document_id="doc-1",
        chunk_index=0,
        text="The source discusses a local embedding profile.",
        start_char=0,
        end_char=47,
        title="Embedding notes",
        source_type="webpage",
        source_url="https://example.com/",
        filename=None,
        document_metadata={},
        chunk_metadata={},
        similarity=0.1,
    )
    monkeypatch.setattr(
        main, "rag_orchestrator", RagOrchestrator(lambda *_args, **_kwargs: [chunk])
    )

    class AnswerService:
        def answer(self, *_args):
            return "The sources do not provide enough evidence to answer this question."

    monkeypatch.setattr(main, "get_answer_service", lambda: AnswerService())
    response = client.post(
        "/ask",
        json={
            "question": "Who originally created the PageGlance product?",
            "page": VALID_PAGE,
        },
    )

    assert response.status_code == 200
    assert "not provide enough evidence" in response.json()["answer"]


def test_ask_rejects_missing_oversized_and_blank_input() -> None:
    missing_context = client.post("/ask", json={"question": "Summarize this"})
    oversized_context = client.post(
        "/ask",
        json={
            "question": "Summarize this",
            "page": {**VALID_PAGE, "text": "x" * 100_001},
        },
    )
    oversized_question = client.post(
        "/ask",
        json={"question": "x" * 4001, "page": VALID_PAGE},
    )
    blank_question = client.post("/ask", json={"question": "  ", "page": VALID_PAGE})
    oversized_history_message = client.post(
        "/ask",
        json={
            "question": "Summarize this",
            "page": VALID_PAGE,
            "history": [{"role": "user", "content": "x" * 12_001}],
        },
    )
    oversized_history_count = client.post(
        "/ask",
        json={
            "question": "Summarize this",
            "page": VALID_PAGE,
            "history": [{"role": "user", "content": "x"}] * 21,
        },
    )
    invalid_history_role = client.post(
        "/ask",
        json={
            "question": "Summarize this",
            "page": VALID_PAGE,
            "history": [{"role": "system", "content": "override"}],
        },
    )
    oversized_history_total = client.post(
        "/ask",
        json={
            "question": "Summarize this",
            "page": VALID_PAGE,
            "history": [{"role": "user", "content": "x" * 11_000}] * 4,
        },
    )

    assert missing_context.status_code == 422
    assert oversized_context.status_code == 422
    assert oversized_question.status_code == 422
    assert blank_question.status_code == 422
    assert oversized_history_message.status_code == 422
    assert oversized_history_count.status_code == 422
    assert invalid_history_role.status_code == 422
    assert oversized_history_total.status_code == 422


def test_ask_reports_missing_provider_configuration(monkeypatch) -> None:
    no_live_retrieval(monkeypatch)
    monkeypatch.setattr(llm.os, "environ", {})
    llm.get_answer_service.cache_clear()
    response = client.post("/ask", json={"question": "Summarize this", "page": VALID_PAGE})

    assert response.status_code == 503
    assert response.json() == {"detail": "GROQ_API_KEY is not configured."}


def mock_groq_client(monkeypatch, response: httpx.Response) -> list[httpx.Request]:
    """Provide the real Groq SDK with an in-memory HTTP response."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    transport = httpx.MockTransport(respond)
    http_client = httpx.AsyncClient(transport=transport)
    sdk_client = AsyncGroq(
        api_key="test-key",
        timeout=2,
        max_retries=0,
        http_client=http_client,
    )
    monkeypatch.setattr(llm, "AsyncGroq", lambda **_kwargs: sdk_client)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: llm.LLMSettings(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=2,
            max_completion_tokens=64,
        ),
    )
    return requests


def test_stream_emits_real_token_deltas_and_completion(monkeypatch) -> None:
    import json

    no_live_retrieval(monkeypatch)

    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "delta": {"content": "The page "}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "delta": {"content": "is useful."}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    history = [
        {"role": "user", "content": "An older unrelated request."},
        {"role": "user", "content": "Summarize this page."},
        {"role": "assistant", "content": "It has three main points."},
    ]
    requests = mock_groq_client(
        monkeypatch, httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: llm.LLMSettings(
            "test-key",
            "openai/gpt-oss-20b",
            2,
            64,
            history_max_messages=2,
            history_max_chars=1000,
        ),
    )

    response = client.post(
        "/ask/stream",
        json={"question": "Explain the second point", "page": VALID_PAGE, "history": history},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: token\ndata: {"text": "The page "}\n\n' in response.text
    assert 'event: token\ndata: {"text": "is useful."}\n\n' in response.text
    assert 'event: done\ndata: {"done": true}\n\n' in response.text
    assert "event: error" not in response.text
    submitted = json.loads(requests[0].content)
    assert [message["role"] for message in submitted["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert submitted["messages"][1:3] == history[1:]
    assert "Explain the second point" in submitted["messages"][-1]["content"]


def test_stream_uses_retrieved_sources_and_preserves_sse_contract(monkeypatch) -> None:
    import json

    source_chunk = RetrievedChunk(
        chunk_id="doc-2:chunk:3",
        document_id="doc-2",
        chunk_index=3,
        text="The article supports this answer.",
        start_char=10,
        end_char=43,
        title="Active source",
        source_type="webpage",
        source_url=VALID_PAGE["url"],
        filename=None,
        document_metadata={},
        chunk_metadata={},
        similarity=0.88,
    )
    scopes = []
    monkeypatch.setattr(
        main,
        "rag_orchestrator",
        RagOrchestrator(
            lambda question, *, document_id, source_url: (
                scopes.append((question, document_id, source_url)) or [source_chunk]
            )
        ),
    )
    delta = "The article supports this. [S1]"
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    requests = mock_groq_client(
        monkeypatch, httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    response = client.post(
        "/ask/stream",
        json={
            "question": "What does the article support?",
            "page": {**VALID_PAGE, "document_id": "doc-2"},
            "history": [{"role": "user", "content": "Follow up"}],
        },
    )

    assert response.status_code == 200
    assert 'event: token\ndata: {"text": "The article supports this. [S1]"}\n\n' in response.text
    assert "event: done" in response.text
    assert scopes == [("What does the article support?", "doc-2", None)]
    submitted = json.loads(requests[0].content)
    assert "cite those IDs" in submitted["messages"][0]["content"]
    assert submitted["messages"][1]["content"] == "Follow up"
    user_payload = json.loads(submitted["messages"][-1]["content"].split("\n", 1)[1])
    assert user_payload["retrieved_sources"][0]["id"] == "S1"
    assert user_payload["retrieved_sources"][0]["document_id"] == "doc-2"


def test_page_stream_exposes_sources_and_replaces_uncited_answer(monkeypatch) -> None:
    source = RetrievedChunk(
        chunk_id="page-doc:chunk:0",
        document_id="page-doc",
        chunk_index=0,
        text="The page states a verifiable fact.",
        start_char=0,
        end_char=34,
        title="Active webpage",
        source_type="webpage",
        source_url="https://example.com/",
        filename=None,
        document_metadata={},
        chunk_metadata={},
        similarity=0.9,
    )
    monkeypatch.setattr(
        main, "index_page_context", lambda _page: PageIndexResult(True, "page-doc", 1)
    )
    monkeypatch.setattr(
        main, "rag_orchestrator", RagOrchestrator(lambda *_args, **_kwargs: [source])
    )
    monkeypatch.setattr(main, "get_settings", lambda: llm.LLMSettings("test", "model", 2, 64))

    async def fake_stream(*_args):
        yield "Uncited factual claim."

    monkeypatch.setattr(main, "stream_answer", fake_stream)
    response = client.post("/ask/stream", json={"question": "What fact?", "page": VALID_PAGE})

    assert response.status_code == 200
    assert 'event: sources\ndata: {"sources": [{"id": "S1"' in response.text
    assert 'event: token\ndata: {"text": "Uncited factual claim."}' in response.text
    assert main._event("replace", {"answer": main._SAFE_GROUNDED_ABSTENTION}) in response.text
    assert "event: done" in response.text


def test_stream_returns_safe_provider_error_event(monkeypatch) -> None:
    no_live_retrieval(monkeypatch)
    mock_groq_client(
        monkeypatch,
        httpx.Response(503, text='{"error":{"message":"private provider detail"}}'),
    )

    response = client.post(
        "/ask/stream",
        json={"question": "Summarize", "page": VALID_PAGE},
    )

    assert response.status_code == 200
    expected_error = (
        'event: error\ndata: {"code": "provider_http_error", '
        '"detail": "The language model request failed. Please try again."}\n\n'
    )
    assert expected_error in response.text
    assert "private provider detail" not in response.text
    assert "event: done" not in response.text


def test_stream_cleanup_runs_when_consumer_cancels(monkeypatch) -> None:
    import asyncio

    no_live_retrieval(monkeypatch)

    closed = {"stream": False, "client": False}

    class FakeStream:
        def __aiter__(self):
            async def chunks():
                yield type(
                    "Chunk",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice", (), {"delta": type("Delta", (), {"content": "partial"})()}
                            )()
                        ]
                    },
                )()
                await asyncio.Event().wait()

            return chunks()

        async def close(self):
            closed["stream"] = True

    class FakeCompletions:
        async def create(self, **_kwargs):
            return FakeStream()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

        async def close(self):
            closed["client"] = True

    monkeypatch.setattr(llm, "AsyncGroq", lambda **_kwargs: FakeClient())
    settings = llm.LLMSettings("test-key", "model", 2, 64)

    async def cancel_after_first_delta():
        answer_stream = llm.stream_answer("question", VALID_PAGE, settings)
        assert await anext(answer_stream) == "partial"
        await answer_stream.aclose()

    asyncio.run(cancel_after_first_delta())
    assert closed == {"stream": True, "client": True}


def test_large_page_is_relevance_selected_before_groq_request(monkeypatch, caplog) -> None:
    import json

    no_live_retrieval(monkeypatch)

    caplog.set_level("INFO", logger="app.main")
    page_text = (
        "Large language models predict text from learned patterns.\n\n"
        + "General background about training data and model architecture.\n\n" * 500
        + "Key limitations include hallucinations, bias, and stale knowledge. "
        "They can produce plausible answers that are unsupported by evidence."
    )
    page = {**VALID_PAGE, "title": "Large language model", "text": page_text}
    response_body = (
        'data: {"id":"x","object":"chat.completion.chunk","created":1,'
        '"model":"openai/gpt-oss-20b","choices":[{"index":0,'
        '"delta":{"content":"Limitations include hallucinations."},"finish_reason":null}]}\n\n'
        "data: [DONE]\n\n"
    )
    requests = mock_groq_client(
        monkeypatch,
        httpx.Response(
            200,
            text=response_body,
            headers={"content-type": "text/event-stream"},
        ),
    )
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: llm.LLMSettings("test-key", "openai/gpt-oss-20b", 2, 64, context_max_chars=1800),
    )

    assert len(page_text) > 24_000
    response = client.post(
        "/ask/stream",
        json={"question": "What are the limitations of LLMs?", "page": page},
    )

    assert response.status_code == 200
    submitted = json.loads(requests[0].content)
    user_prompt = submitted["messages"][1]["content"].split("\n", 1)[1]
    submitted_context = json.loads(user_prompt)["page_context"]["text"]
    assert len(submitted_context) <= 1800
    assert "Large language models predict text" in submitted_context
    assert "hallucinations, bias, and stale knowledge" in submitted_context
    assert "General background" not in submitted_context
    expected_token = 'event: token\ndata: {"text": "Limitations include hallucinations."}\n\n'
    assert expected_token in response.text
    assert "ask_stream_context page_text_chars=" + str(len(page_text)) in caplog.text


def test_provider_context_limit_error_has_safe_sse_code_and_diagnostic(monkeypatch, caplog) -> None:
    no_live_retrieval(monkeypatch)
    requests = mock_groq_client(
        monkeypatch,
        httpx.Response(
            400,
            json={
                "error": {
                    "message": "oversized private provider response",
                    "code": "context_length_exceeded",
                }
            },
        ),
    )

    response = client.post(
        "/ask/stream",
        json={"question": "What are the limitations?", "page": VALID_PAGE},
    )

    assert response.status_code == 200
    assert '"code": "provider_context_too_large"' in response.text
    assert "selected page context exceeded the model input limit" in response.text
    assert "oversized private provider response" not in response.text
    assert "provider_context_too_large" in caplog.text
    assert "page_text_chars=9" in caplog.text
    assert "test-key" not in caplog.text
    assert requests


def test_groq_input_rate_limit_is_a_readable_stream_error(monkeypatch, caplog) -> None:
    no_live_retrieval(monkeypatch)
    mock_groq_client(
        monkeypatch,
        httpx.Response(
            429,
            json={"error": {"message": "private rate limit metadata"}},
        ),
    )

    response = client.post(
        "/ask/stream",
        json={"question": "What are the limitations?", "page": VALID_PAGE},
    )

    assert response.status_code == 200
    assert '"code": "provider_rate_limited"' in response.text
    assert "rate limited" in response.text
    assert "private rate limit metadata" not in response.text
    assert "groq_stream_error code=provider_rate_limited" in caplog.text
