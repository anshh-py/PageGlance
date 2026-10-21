"""FastAPI application entry point."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import APIConnectionError, APIStatusError, APITimeoutError, GroqError, RateLimitError
from pydantic import BaseModel, Field, field_validator, model_validator

from app.context import select_page_context
from app.llm import (
    LLMConfigurationError,
    LLMResponseError,
    LLMSettings,
    get_answer_service,
    get_context_max_chars,
    get_history_limits,
    get_settings,
    stream_answer,
)

app = FastAPI(title="PageGlance API", version="0.1.0")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PageContext(BaseModel):
    """Bounded page data sent alongside a question."""

    title: str = Field(max_length=1000)
    url: str = Field(max_length=4000)
    text: str = Field(max_length=100_000)


class ConversationMessage(BaseModel):
    """One bounded user or assistant message from the current page conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation messages must contain text.")
        return value


class AskRequest(BaseModel):
    """Validated input for a page question request."""

    question: str = Field(min_length=1, max_length=4000)
    page: PageContext
    history: list[ConversationMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        """Reject whitespace-only questions and remove surrounding whitespace."""

        value = value.strip()
        if not value:
            raise ValueError("Question must contain non-whitespace text.")
        return value

    @model_validator(mode="after")
    def bound_history_chars(self) -> "AskRequest":
        if sum(len(message.content) for message in self.history) > 40_000:
            raise ValueError("Conversation history exceeds the maximum combined size.")
        return self


class AskResponse(BaseModel):
    """Stable answer schema returned to the Chrome extension."""

    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return basic service health."""

    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a page question using the configured hosted language model."""

    try:
        context_limit = get_context_max_chars()
        page_context = select_page_context(
            request.question,
            request.page.model_dump(),
            context_limit,
        )
        logger.info(
            "ask_context page_text_chars=%d selected_chars=%d context_limit_chars=%d",
            len(request.page.text),
            len(page_context["text"]),
            context_limit,
        )
        history_max_messages, history_max_chars = get_history_limits()
        answer = get_answer_service().answer(
            request.question,
            page_context,
            _bounded_history(request.history, history_max_messages, history_max_chars),
        )
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except APITimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="The language model request timed out. Please try again.",
        ) from error
    except RateLimitError as error:
        raise HTTPException(
            status_code=503,
            detail="The language model is temporarily rate limited. Please try again shortly.",
        ) from error
    except (APIConnectionError, APIStatusError, GroqError, LLMResponseError) as error:
        raise HTTPException(
            status_code=502,
            detail="The language model request failed. Please try again.",
        ) from error

    return AskResponse(answer=answer)


def _event(event: str, payload: dict[str, str | bool]) -> str:
    """Serialize one safe Server-Sent Event frame."""

    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _bounded_history(
    history: list[ConversationMessage],
    max_messages: int,
    max_chars: int,
) -> list[dict[str, str]]:
    """Keep the most recent complete messages under both configured limits."""
    chosen: list[ConversationMessage] = []
    remaining = max_chars
    recent = history[-max_messages:] if max_messages else []
    for message in reversed(recent):
        if len(message.content) <= remaining:
            chosen.append(message)
            remaining -= len(message.content)
    chosen.reverse()
    return [message.model_dump() for message in chosen]


def _provider_error(error: Exception) -> tuple[str, str]:
    """Map provider exceptions to an allowlisted code and safe user-facing detail."""
    if isinstance(error, APITimeoutError):
        return "provider_timeout", "The language model request timed out."
    if isinstance(error, RateLimitError):
        return (
            "provider_rate_limited",
            "The language model is rate limited. Please try again shortly.",
        )

    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    provider_code = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        candidate = body["error"].get("code")
        if isinstance(candidate, str):
            provider_code = candidate.lower()
    if status == 413 or provider_code in {
        "context_length_exceeded",
        "input_too_large",
        "token_limit_exceeded",
    }:
        return (
            "provider_context_too_large",
            "The selected page context exceeded the model input limit.",
        )
    if status == 429:
        return (
            "provider_rate_limited",
            "The language model is rate limited. Please try again shortly.",
        )
    if status == 400:
        return "provider_bad_request", "The language model rejected the request."
    if isinstance(error, APIConnectionError):
        return "provider_connection_error", "Could not connect to the language model provider."
    if isinstance(error, APIStatusError):
        return "provider_http_error", "The language model request failed. Please try again."
    return "provider_error", "The language model request failed. Please try again."


async def _answer_events(
    request: AskRequest,
    settings: LLMSettings,
    page_context: dict[str, str],
    raw_page_text_chars: int,
    history: list[dict[str, str]],
) -> AsyncIterator[str]:
    answer_stream = stream_answer(
        request.question,
        page_context,
        settings,
        history,
    )
    try:
        async for text in answer_stream:
            yield _event("token", {"text": text})
        yield _event("done", {"done": True})
    except Exception as error:
        # Log lengths and an allowlisted code only; never provider bodies or page data.
        code, detail = _provider_error(error)
        logger.warning(
            "groq_stream_error code=%s page_text_chars=%d selected_chars=%d",
            code,
            raw_page_text_chars,
            len(page_context["text"]),
        )
        yield _event("error", {"code": code, "detail": detail})
    finally:
        await answer_stream.aclose()


@app.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream generated answer deltas as Server-Sent Events."""

    try:
        settings = get_settings()
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    page_context = select_page_context(
        request.question,
        request.page.model_dump(),
        settings.context_max_chars,
    )
    logger.info(
        "ask_stream_context page_text_chars=%d selected_chars=%d context_limit_chars=%d",
        len(request.page.text),
        len(page_context["text"]),
        settings.context_max_chars,
    )
    history = _bounded_history(
        request.history,
        settings.history_max_messages,
        settings.history_max_chars,
    )
    return StreamingResponse(
        _answer_events(request, settings, page_context, len(request.page.text), history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
