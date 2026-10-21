import httpx
from fastapi.testclient import TestClient
from groq import AsyncGroq

from app import llm, main

client = TestClient(main.app)
VALID_PAGE = {"title": "Example", "url": "https://example.com", "text": "Page text"}


class FakeAnswerService:
    def answer(
        self,
        question: str,
        page_context: dict[str, str],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        assert question == "What is this page about?"
        assert page_context == VALID_PAGE
        assert history == []
        return "The page is an example."


def test_ask_returns_predictable_answer_schema(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_answer_service", lambda: FakeAnswerService())

    response = client.post(
        "/ask",
        json={"question": " What is this page about? ", "page": VALID_PAGE},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "The page is an example."}


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


def test_stream_returns_safe_provider_error_event(monkeypatch) -> None:
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
