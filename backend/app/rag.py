"""Coordinate scoped retrieval and grounded context preparation for chat routes."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.context import select_page_context
from app.db import DatabaseConfigurationError, create_session_factory
from app.embeddings import EmbeddingError
from app.retrieval import RetrievalError, RetrievedChunk, VectorRetriever

logger = logging.getLogger(__name__)
_ADAPTIVE_TERMS = frozenset(
    "a an and are as at be by for from how i in is it of on or the this to was what when "
    "where which who why with".split()
)
_QUERY_BOUNDARY = re.compile(
    r"[?;]+\s*|\s+and\s+(?=(?:what|which|how|why|who|when|where|also|list|explain|describe)\b)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MAX_ADAPTIVE_RETRY_QUERIES = 3
MAX_PREPARED_SOURCES = 5


@dataclass(frozen=True)
class PreparedAnswerContext:
    """LLM inputs plus whether retrieval actually supplied evidence."""

    page_context: dict[str, str]
    retrieved_sources: list[dict[str, Any]] | None
    retrieval_used: bool
    fallback_code: str | None = None


def _retrieve_scoped_chunks(
    question: str,
    *,
    document_id: str | None,
    document_ids: list[str] | None,
    source_url: str | None,
) -> list[RetrievedChunk]:
    """Open a request-scoped database session and retrieve within the active source."""
    session_factory = create_session_factory()
    with session_factory() as session:
        retriever = VectorRetriever(session)
        scope: dict[str, object] = {
            "source_url": None if document_id or document_ids else source_url
        }
        if document_id:
            scope["document_id"] = document_id
        if document_ids:
            scope["document_ids"] = document_ids
        return retriever.retrieve(question, **scope)


def _adaptive_queries(question: str, page: dict[str, str]) -> list[str]:
    """Deterministically split multi-part asks or clarify a pronoun from source title."""
    parts = [part.strip(" ,\n\t") for part in _QUERY_BOUNDARY.split(question) if part.strip()]
    if len(parts) > 1:
        return parts[:MAX_ADAPTIVE_RETRY_QUERIES]
    if re.match(r"^(?:it|this|that|they|these|those)\b", question, re.IGNORECASE):
        title = page.get("title", "").strip()[:200]
        if title:
            return [f"{question} {title}"]
    return []


def _covered_query_queries(queries: list[str], chunks: list[RetrievedChunk]) -> set[str]:
    """Estimate which query facets have lexical evidence in retrieved chunks."""
    combined = " ".join(chunk.text.casefold() for chunk in chunks)
    covered: set[str] = set()
    for query in queries:
        terms = {
            token.casefold()
            for token in _WORD.findall(query)
            if len(token) >= 3 and token.casefold() not in _ADAPTIVE_TERMS
        }
        if terms and any(term in combined for term in terms):
            covered.add(query)
    return covered


def _merge_retry_results(rankings: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    candidates: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (60 + rank)
            candidates.setdefault(chunk.chunk_id, chunk)
    return sorted(
        candidates.values(), key=lambda chunk: (-scores[chunk.chunk_id], -chunk.similarity)
    )[:MAX_PREPARED_SOURCES]


class RagOrchestrator:
    """Build RAG context, falling back to selected active-page context safely."""

    def __init__(
        self,
        retrieve: Callable[..., list[RetrievedChunk]] = _retrieve_scoped_chunks,
    ) -> None:
        self.retrieve = retrieve

    def prepare(
        self,
        question: str,
        page: dict[str, str],
        *,
        context_limit: int,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        retrieval_enabled: bool = True,
        fallback_code: str | None = None,
    ) -> PreparedAnswerContext:
        """Retrieve only from the page/document scope and preserve the page fallback."""
        selected_page = select_page_context(question, page, context_limit)
        source_url = page.get("url", "").strip() or None
        if not retrieval_enabled:
            code = fallback_code or "no_indexed_content"
            logger.info("rag_fallback code=%s page_text_chars=%d", code, len(page.get("text", "")))
            return PreparedAnswerContext(selected_page, None, False, code)
        try:
            scope: dict[str, object] = {
                "document_id": document_id,
                "source_url": None if document_id or document_ids else source_url,
            }
            if document_ids is not None:
                scope["document_ids"] = document_ids
            chunks = self.retrieve(question, **scope)
        except (
            DatabaseConfigurationError,
            EmbeddingError,
            RetrievalError,
            SQLAlchemyError,
        ) as error:
            # Do not expose database/provider exception details in fallback logs or responses.
            code = getattr(error, "code", "retrieval_unavailable")
            logger.info("rag_fallback code=%s page_text_chars=%d", code, len(page.get("text", "")))
            return PreparedAnswerContext(selected_page, None, False, code)

        if not chunks:
            adaptive_queries = _adaptive_queries(question, page)
            retry_results: list[list[RetrievedChunk]] = []
            if adaptive_queries:
                try:
                    retry_results = [self.retrieve(query, **scope) for query in adaptive_queries]
                    chunks = _merge_retry_results(retry_results)
                    logger.info(
                        "rag_adaptive_retry queries=%d results=%d",
                        len(adaptive_queries),
                        len(chunks),
                    )
                except (
                    DatabaseConfigurationError,
                    EmbeddingError,
                    RetrievalError,
                    SQLAlchemyError,
                ):
                    logger.info("rag_adaptive_retry_failed code=retry_unavailable")
            if not chunks:
                logger.info(
                    "rag_fallback code=no_indexed_content page_text_chars=%d",
                    len(page.get("text", "")),
                )
                return PreparedAnswerContext(selected_page, None, False, "no_indexed_content")
        else:
            adaptive_queries = _adaptive_queries(question, page)
            covered = _covered_query_queries(adaptive_queries, chunks)
            missing = [query for query in adaptive_queries if query not in covered]
            if missing:
                try:
                    retry_results = [self.retrieve(query, **scope) for query in missing]
                    chunks = _merge_retry_results([chunks, *retry_results])
                    logger.info(
                        "rag_adaptive_retry queries=%d results=%d", len(missing), len(chunks)
                    )
                except (
                    DatabaseConfigurationError,
                    EmbeddingError,
                    RetrievalError,
                    SQLAlchemyError,
                ):
                    logger.info("rag_adaptive_retry_failed code=retry_unavailable")

        sources = [
            {
                "id": f"S{index}",
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "title": chunk.title,
                "source_type": chunk.source_type,
                "source_url": chunk.source_url,
                "filename": chunk.filename,
                "similarity": chunk.similarity,
                "text": chunk.text,
                "chunk_metadata": chunk.chunk_metadata,
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        logger.info(
            "rag_retrieval_used sources=%d page_text_chars=%d",
            len(sources),
            len(page.get("text", "")),
        )
        return PreparedAnswerContext(selected_page, sources, True)


rag_orchestrator = RagOrchestrator()
