"""FastAPI application entry point."""

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import APIConnectionError, APIStatusError, APITimeoutError, GroqError, RateLimitError
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.db import DatabaseConfigurationError
from app.document_ingestion import (
    UploadProcessingError,
    UploadValidationError,
    ingest_uploaded_file,
)
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
from app.page_indexing import get_ingestion_session_factory, index_page_context
from app.rag import rag_orchestrator
from app.repository import DocumentRepository
from app.upload_limits import UploadBodyLimitMiddleware, get_max_upload_bytes, get_max_upload_mb
from app.url_identity import normalize_page_url

app = FastAPI(title="PageGlance API", version="0.1.0")
logger = logging.getLogger(__name__)
_UPLOAD_ERROR_DETAILS = {
    "database_configuration_error": (
        "The backend database is not configured. "
        "Set DATABASE_URL in the repository root .env and restart the backend."
    ),
    "database_unavailable": (
        "The backend database is unavailable. Check the local PostgreSQL service and try again."
    ),
    "embedding_configuration_error": (
        "The embedding provider configuration is invalid. "
        "Check PAGEGLANCE_EMBEDDING_PROVIDER and local model settings in the repository root .env."
    ),
    "embedding_model_unavailable": (
        "The local embedding model is unavailable. Install backend dependencies and download "
        "or cache the configured model before retrying."
    ),
    "embedding_generation_failed": (
        "The local embedding model could not process this upload. Check backend logs and try again."
    ),
    "pdf_parser_unavailable": (
        "PDF uploads are unavailable because the backend PDF parser is not installed. "
        "Install backend dependencies and restart the backend."
    ),
    "pdf_renderer_unavailable": (
        "Scanned PDF processing is unavailable because PyMuPDF could not open or render the file."
    ),
    "pytesseract_unavailable": (
        "Scanned PDF OCR requires the pytesseract Python package. Install backend dependencies "
        "and restart the backend."
    ),
    "tesseract_executable_unavailable": (
        "Tesseract OCR could not be started. Set PAGEGLANCE_TESSERACT_CMD to the full path of "
        "tesseract.exe or add Tesseract to PATH, then restart the backend."
    ),
    "ocr_engine_unavailable": (
        "Local OCR is unavailable. Check the pytesseract package and Tesseract executable "
        "configuration."
    ),
    "ocr_timeout": "Local OCR took too long to process this PDF. Try a smaller file.",
    "pdf_ocr_failed": "Local OCR could not read a scanned PDF page. Check backend logs and retry.",
    "embedding_rate_limited": (
        "The embedding provider is rate limiting uploads. Please try again shortly."
    ),
    "embedding_quota_exhausted": (
        "The embedding provider quota or billing limit is exhausted. "
        "Check the provider account configuration before retrying."
    ),
    "embedding_provider_429_unknown": (
        "The embedding provider rejected the upload with an unclassified rate-limit response. "
        "Check the backend logs before retrying."
    ),
    "embedding_timeout": (
        "The embedding provider timed out while processing this upload. Please try again."
    ),
    "embedding_provider_unavailable": (
        "The embedding provider is temporarily unavailable. Please try again."
    ),
    "embedding_provider_error": (
        "The embedding provider could not process this upload. "
        "Check its model configuration and try again."
    ),
    "embedding_dimension_mismatch": (
        "The embedding provider returned an incompatible result. "
        "Check its model and dimension configuration."
    ),
    "embedding_response_invalid": (
        "The embedding provider returned an invalid result. Please try again."
    ),
}
_GENERIC_UPLOAD_ERROR = "Document ingestion is temporarily unavailable. Please try again."

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.add_middleware(UploadBodyLimitMiddleware)


class PageContext(BaseModel):
    """Bounded page data sent alongside a question."""

    title: str = Field(max_length=1000)
    url: str = Field(max_length=4000)
    text: str = Field(max_length=100_000)
    document_id: str | None = Field(default=None, min_length=1, max_length=36)

    @field_validator("url")
    @classmethod
    def validate_and_normalize_url(cls, value: str) -> str:
        return normalize_page_url(value)


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
    page: PageContext | None = None
    document_id: str | None = Field(default=None, min_length=36, max_length=36)
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=5)
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
        supplied_scopes = sum(
            value is not None for value in (self.page, self.document_id, self.document_ids)
        )
        if supplied_scopes != 1:
            raise ValueError("Provide exactly one of page, document_id, or document_ids.")
        if self.document_id is not None:
            try:
                if str(UUID(self.document_id)) != self.document_id.lower():
                    raise ValueError
            except ValueError:
                raise ValueError("document_id must be a valid document identifier.") from None
        if self.document_ids is not None:
            if len(set(self.document_ids)) != len(self.document_ids):
                raise ValueError("document_ids must not contain duplicates.")
            for document_id in self.document_ids:
                try:
                    if str(UUID(document_id)) != document_id.lower():
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        "document_ids must contain valid document identifiers."
                    ) from None
        if sum(len(message.content) for message in self.history) > 40_000:
            raise ValueError("Conversation history exceeds the maximum combined size.")
        return self


class AnswerSource(BaseModel):
    """Citation metadata for a retrieved chunk, without exposing its full text."""

    id: str
    document_id: str
    title: str
    source_type: str
    filename: str | None = None
    source_url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    page_numbers: list[int] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Stable answer schema returned to the Chrome extension."""

    answer: str
    sources: list[AnswerSource] | None = None


class DocumentChunkStats(BaseModel):
    """Index metadata for one stored chunk, with no document text exposed."""

    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    page_numbers: list[int] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class DocumentStatsResponse(BaseModel):
    """Persisted indexing facts for one selected uploaded document."""

    document_id: str
    filename: str | None
    source_type: Literal["text", "pdf"]
    indexed_chunk_count: int
    chunks: list[DocumentChunkStats]


def _document_stats(document) -> DocumentStatsResponse:
    chunks = []
    for chunk in document.chunks:
        metadata = chunk.metadata
        pages = metadata.get("page_numbers", [])
        chunks.append(
            DocumentChunkStats(
                chunk_id=chunk.id,
                chunk_index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                page_numbers=[page for page in pages if isinstance(page, int)]
                if isinstance(pages, list)
                else [],
                page_start=metadata.get("page_start")
                if isinstance(metadata.get("page_start"), int)
                else None,
                page_end=metadata.get("page_end")
                if isinstance(metadata.get("page_end"), int)
                else None,
            )
        )
    return DocumentStatsResponse(
        document_id=document.document.id,
        filename=document.document.filename,
        source_type=document.document.source_type,
        indexed_chunk_count=len(chunks),
        chunks=chunks,
    )


def _document_diagnostics_question(question: str) -> bool:
    pattern = (
        r"\b(?:how many|total|count|number of)\b.{0,40}\bchunks?\b"
        r"|\bchunk count\b|\bchunk ids?\b|\bpage ranges?\b|\bpage numbers?\b"
        r"|\bchunks?\b.{0,40}\b(?:retrieved|indexed|count|total)\b"
    )
    return re.search(pattern, question, re.IGNORECASE) is not None


def _page_label(numbers: object, start: object, end: object) -> str:
    if isinstance(start, int) and isinstance(end, int):
        return str(start) if start == end else f"{start}-{end}"
    if isinstance(numbers, list) and numbers:
        return ", ".join(str(page) for page in numbers if isinstance(page, int))
    return "not available"


def _format_document_diagnostics(
    question: str,
    stats: DocumentStatsResponse,
    retrieved_sources: list[dict[str, object]] | None,
) -> str:
    """Answer index diagnostics directly from persisted metadata, not the LLM."""
    lines = [
        f"Indexed chunks in this document: {stats.indexed_chunk_count}.",
        f"Chunks retrieved for this question: {len(retrieved_sources or [])}.",
    ]
    show_chunk_details = (
        re.search(
            r"\b(?:id|ids|range|ranges|page|pages|metadata|list|show)\b",
            question,
            re.IGNORECASE,
        )
        is not None
    )
    if show_chunk_details and stats.chunks:
        lines.append("Indexed chunk metadata:")
        for chunk in stats.chunks:
            page_label = _page_label(chunk.page_numbers, chunk.page_start, chunk.page_end)
            lines.append(
                f"- index {chunk.chunk_index}: {chunk.chunk_id}; pages {page_label}; "
                f"characters {chunk.start_char}-{chunk.end_char}"
            )
    if show_chunk_details and retrieved_sources:
        lines.append("Retrieved chunks:")
        for source in retrieved_sources:
            metadata = source.get("chunk_metadata")
            chunk_metadata = metadata if isinstance(metadata, dict) else {}
            page_label = _page_label(
                chunk_metadata.get("page_numbers"),
                chunk_metadata.get("page_start"),
                chunk_metadata.get("page_end"),
            )
            lines.append(
                f"- {source.get('id')}: {source.get('chunk_id')} "
                f"(index {source.get('chunk_index')}; pages {page_label})"
            )
    return "\n".join(lines)


def _document_diagnostic_answer(request: AskRequest, prepared) -> str | None:
    if not _document_diagnostics_question(request.question):
        return None
    if request.document_ids is not None:
        raise HTTPException(
            status_code=422,
            detail="Document diagnostics require one selected document at a time.",
        )
    if request.document_id is None:
        return None
    stats = _load_document_stats(request.document_id)
    return _format_document_diagnostics(request.question, stats, prepared.retrieved_sources)


def _prepare_page_answer_context(request: AskRequest, context_limit: int):
    """Index the active page, then retrieve only from that successfully indexed page."""
    if request.page is None:
        raise ValueError("A webpage context is required for page retrieval.")
    request_page = request.page.model_dump()
    page = request.page.model_dump(exclude={"document_id"})
    indexed = index_page_context(request_page)
    if not indexed.indexed:
        logger.info(
            "ask_index_fallback code=%s page_text_chars=%d",
            indexed.fallback_code or "page_index_unavailable",
            len(request.page.text) if request.page else 0,
        )
    return rag_orchestrator.prepare(
        request.question,
        page,
        context_limit=context_limit,
        document_id=indexed.document_id if indexed.indexed else None,
        retrieval_enabled=indexed.indexed,
        fallback_code=indexed.fallback_code,
    )


def _prepare_document_answer_context(request: AskRequest, context_limit: int):
    """Validate and retrieve only the selected uploaded TXT/PDF document."""
    if request.document_id is None:
        raise ValueError("A document identifier is required for document retrieval.")
    try:
        with get_ingestion_session_factory()() as session:
            document = DocumentRepository(session).get_by_id(request.document_id)
    except (DatabaseConfigurationError, SQLAlchemyError):
        raise HTTPException(
            status_code=503,
            detail="The selected document is temporarily unavailable.",
        ) from None

    if document is None or document.document.source_type not in {"text", "pdf"}:
        raise HTTPException(status_code=404, detail="The selected document was not found.")
    if not document.chunks:
        raise HTTPException(status_code=422, detail="The selected document has no indexed text.")

    page = {
        "title": document.document.filename or document.document.title,
        "url": "",
        "text": "",
    }
    prepared = rag_orchestrator.prepare(
        request.question,
        page,
        context_limit=context_limit,
        document_id=request.document_id,
        retrieval_enabled=True,
    )
    if not prepared.retrieval_used:
        if prepared.fallback_code == "no_indexed_content":
            raise HTTPException(
                status_code=422,
                detail="No matching indexed content was found in the selected document.",
            )
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve evidence from the selected document.",
        )
    if not prepared.retrieved_sources or any(
        source.get("document_id") != request.document_id for source in prepared.retrieved_sources
    ):
        raise HTTPException(
            status_code=503,
            detail="The selected document returned invalid retrieval scope.",
        )
    return prepared


def _prepare_multi_document_context(request: AskRequest, context_limit: int):
    """Validate a small explicit document set and retrieve strictly within that set."""
    document_ids = request.document_ids
    if not document_ids:
        raise ValueError("A document identifier list is required for document retrieval.")
    try:
        with get_ingestion_session_factory()() as session:
            documents = [DocumentRepository(session).get_by_id(item) for item in document_ids]
    except (DatabaseConfigurationError, SQLAlchemyError):
        raise HTTPException(
            status_code=503,
            detail="The selected documents are temporarily unavailable.",
        ) from None
    if any(item is None or item.document.source_type not in {"text", "pdf"} for item in documents):
        raise HTTPException(
            status_code=404, detail="One or more selected documents were not found."
        )

    page = {"title": "Selected documents", "url": "", "text": ""}
    prepared = rag_orchestrator.prepare(
        request.question,
        page,
        context_limit=context_limit,
        document_ids=document_ids,
        retrieval_enabled=True,
    )
    allowed = set(document_ids)
    if not prepared.retrieval_used or not prepared.retrieved_sources:
        raise HTTPException(
            status_code=422,
            detail="No matching indexed content was found in the selected documents.",
        )
    if any(source.get("document_id") not in allowed for source in prepared.retrieved_sources):
        raise HTTPException(
            status_code=503, detail="The selected documents returned invalid scope."
        )
    return prepared


def _prepare_answer_context(request: AskRequest, context_limit: int):
    if request.document_ids is not None:
        return _prepare_multi_document_context(request, context_limit)
    if request.document_id is not None:
        return _prepare_document_answer_context(request, context_limit)
    return _prepare_page_answer_context(request, context_limit)


_CITATION_REFERENCE = re.compile(r"\[([A-Z][0-9]*)\]")
_ABSTENTION_PHRASES = (
    "not enough verified evidence",
    "not enough evidence",
    "insufficient evidence",
    "cannot determine from the provided",
    "does not provide enough information",
    "do not provide enough evidence",
    "don't provide enough evidence",
    "don't have enough",
    "do not have enough",
)
_SAFE_GROUNDED_ABSTENTION = (
    "I don't have enough verified evidence in the retrieved sources to answer that."
)


def _validate_grounded_answer(
    answer: str,
    retrieved_sources: list[dict[str, object]] | None,
    page_context: dict[str, str],
) -> str:
    """Reject missing or unknown source markers when RAG evidence was supplied."""
    if not retrieved_sources:
        return answer
    if any(phrase in answer.casefold() for phrase in _ABSTENTION_PHRASES):
        logger.info("rag_answer_validation outcome=accepted_abstention reason=model_abstained")
        return answer
    valid_ids = {
        str(source.get("id")) for source in retrieved_sources if isinstance(source.get("id"), str)
    }
    if page_context.get("text", "").strip():
        valid_ids.add("P")
    references = _CITATION_REFERENCE.findall(answer)
    if not references:
        logger.info(
            "rag_answer_validation outcome=rejected reason=missing_citations source_count=%d",
            len(valid_ids),
        )
        return _SAFE_GROUNDED_ABSTENTION
    invalid_reference_count = sum(reference not in valid_ids for reference in references)
    if invalid_reference_count:
        logger.info(
            "rag_answer_validation outcome=rejected reason=unknown_citations "
            "source_count=%d invalid_reference_count=%d",
            len(valid_ids),
            invalid_reference_count,
        )
        return _SAFE_GROUNDED_ABSTENTION
    logger.info(
        "rag_answer_validation outcome=accepted reason=valid_citations source_count=%d "
        "citation_count=%d",
        len(valid_ids),
        len(references),
    )
    return answer


def _evidence_context_chars(
    page_context: dict[str, str], sources: list[dict[str, object]] | None
) -> int:
    """Count evidence text handed to generation without recording its contents."""
    page_chars = sum(len(value) for value in page_context.values())
    source_chars = sum(
        len(source.get("text", ""))
        for source in sources or []
        if isinstance(source.get("text", ""), str)
    )
    return page_chars + source_chars


def _citation_sources(sources: list[dict[str, object]] | None) -> list[AnswerSource] | None:
    if not sources:
        return None
    citations: list[AnswerSource] = []
    for source in sources:
        metadata = source.get("chunk_metadata")
        chunk_metadata = metadata if isinstance(metadata, dict) else {}
        pages = chunk_metadata.get("page_numbers", [])
        filename = source.get("filename")
        source_url = source.get("source_url")
        citations.append(
            AnswerSource(
                id=str(source.get("id", "")),
                document_id=str(source.get("document_id", "")),
                title=str(source.get("title", "")),
                source_type=str(source.get("source_type", "")),
                filename=filename if isinstance(filename, str) else None,
                source_url=source_url if isinstance(source_url, str) else None,
                page_start=chunk_metadata.get("page_start")
                if isinstance(chunk_metadata.get("page_start"), int)
                else None,
                page_end=chunk_metadata.get("page_end")
                if isinstance(chunk_metadata.get("page_end"), int)
                else None,
                page_numbers=[page for page in pages if isinstance(page, int)]
                if isinstance(pages, list)
                else [],
            )
        )
    return citations


@app.get("/health")
def health() -> dict[str, str]:
    """Return basic service health."""

    return {"status": "ok"}


def _load_document_stats(document_id: str) -> DocumentStatsResponse:
    try:
        with get_ingestion_session_factory()() as session:
            document = DocumentRepository(session).get_by_id(document_id)
    except (DatabaseConfigurationError, SQLAlchemyError):
        raise HTTPException(
            status_code=503, detail="Document statistics are temporarily unavailable."
        ) from None
    if document is None or document.document.source_type not in {"text", "pdf"}:
        raise HTTPException(status_code=404, detail="The selected document was not found.")
    return _document_stats(document)


@app.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
def get_document_stats(document_id: UUID) -> DocumentStatsResponse:
    """Return persisted chunk/page metadata for one uploaded document only."""
    return _load_document_stats(str(document_id))


@app.post("/documents/upload")
async def upload_document(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    """Index a bounded TXT or text-based PDF without changing chat contracts."""
    try:
        max_upload_bytes = get_max_upload_bytes()
        file.file.seek(0, 2)
        upload_size = file.file.tell()
        file.file.seek(0)
        if upload_size > max_upload_bytes:
            raise UploadValidationError(
                "file_too_large",
                f"Files must be {get_max_upload_mb()} MiB or smaller.",
                413,
            )
        saved = await run_in_threadpool(ingest_uploaded_file, file.filename, file.file)
    except UploadValidationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None
    except UploadProcessingError as error:
        code = (
            error.code if error.code in _UPLOAD_ERROR_DETAILS else "document_ingestion_unavailable"
        )
        logger.warning("document_upload_failed code=%s", code)
        raise HTTPException(
            status_code=503,
            detail=_UPLOAD_ERROR_DETAILS.get(code, _GENERIC_UPLOAD_ERROR),
        ) from None
    finally:
        await file.close()

    return {
        "document_id": saved.document.id,
        "filename": saved.document.filename,
        "source_type": saved.document.source_type,
        "content_hash": saved.document.content_hash,
        "chunk_count": saved.chunk_count,
        "duplicate": saved.duplicate,
        "page_count": saved.page_count,
    }


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(request: AskRequest) -> AskResponse:
    """Answer a page question using the configured hosted language model."""

    try:
        context_limit = get_context_max_chars()
        prepared = _prepare_answer_context(request, context_limit)
        diagnostic_answer = _document_diagnostic_answer(request, prepared)
        page_text_chars = len(request.page.text) if request.page else 0
        logger.info(
            "ask_context page_text_chars=%d selected_chars=%d context_limit_chars=%d "
            "retrieval_used=%s retrieved_chunks=%d evidence_context_chars=%d",
            page_text_chars,
            len(prepared.page_context["text"]),
            context_limit,
            prepared.retrieval_used,
            len(prepared.retrieved_sources or []),
            _evidence_context_chars(prepared.page_context, prepared.retrieved_sources),
        )
        if diagnostic_answer is not None:
            answer = diagnostic_answer
        else:
            history_max_messages, history_max_chars = get_history_limits()
            answer = get_answer_service().answer(
                request.question,
                prepared.page_context,
                _bounded_history(request.history, history_max_messages, history_max_chars),
                prepared.retrieved_sources,
            )
            answer = _validate_grounded_answer(
                answer, prepared.retrieved_sources, prepared.page_context
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

    citations = (
        None if diagnostic_answer is not None else _citation_sources(prepared.retrieved_sources)
    )
    return AskResponse(answer=answer, sources=citations)


def _event(event: str, payload: dict[str, object]) -> str:
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
    retrieved_sources: list[dict[str, object]] | None,
    citation_sources: list[AnswerSource] | None,
    diagnostic_answer: str | None = None,
) -> AsyncIterator[str]:
    if diagnostic_answer is not None:
        yield _event("token", {"text": diagnostic_answer})
        yield _event("done", {"done": True})
        return
    answer_stream = stream_answer(
        request.question,
        page_context,
        settings,
        history,
        retrieved_sources,
    )
    try:
        if citation_sources:
            yield _event(
                "sources",
                {"sources": [source.model_dump() for source in citation_sources]},
            )
        answer_parts: list[str] = []
        async for text in answer_stream:
            answer_parts.append(text)
            yield _event("token", {"text": text})
        validated_answer = _validate_grounded_answer(
            "".join(answer_parts), retrieved_sources, page_context
        )
        if validated_answer != "".join(answer_parts):
            yield _event("replace", {"answer": validated_answer})
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

    prepared = await run_in_threadpool(_prepare_answer_context, request, settings.context_max_chars)
    diagnostic_answer = await run_in_threadpool(_document_diagnostic_answer, request, prepared)
    page_text_chars = len(request.page.text) if request.page else 0
    logger.info(
        "ask_stream_context page_text_chars=%d selected_chars=%d context_limit_chars=%d "
        "retrieval_used=%s retrieved_chunks=%d evidence_context_chars=%d",
        page_text_chars,
        len(prepared.page_context["text"]),
        settings.context_max_chars,
        prepared.retrieval_used,
        len(prepared.retrieved_sources or []),
        _evidence_context_chars(prepared.page_context, prepared.retrieved_sources),
    )
    history = _bounded_history(
        request.history,
        settings.history_max_messages,
        settings.history_max_chars,
    )
    return StreamingResponse(
        _answer_events(
            request,
            settings,
            prepared.page_context,
            len(request.page.text) if request.page else 0,
            history,
            prepared.retrieved_sources,
            None
            if diagnostic_answer is not None
            else _citation_sources(prepared.retrieved_sources),
            diagnostic_answer,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
