"""Query-time vector retrieval, independent of HTTP and generation services."""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import Select, func, literal_column, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingError, EmbeddingService, get_default_embedding_service
from app.models import DocumentChunkRecord, DocumentRecord

logger = logging.getLogger(__name__)

_LEXICAL_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_LEXICAL_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i "
    "if in into is it its may might of on or our should than that the their them there these "
    "they this those to was what when where which who why will with would you your".split()
)


def _lexical_query_string(query: str) -> str:
    """Build a safe OR query from content terms so natural questions can match passages."""
    terms: list[str] = []
    for raw_term in _LEXICAL_TOKEN.findall(query.casefold()):
        if len(raw_term) < 2 or raw_term in _LEXICAL_STOP_WORDS or raw_term in terms:
            continue
        terms.append(raw_term)
        if len(terms) >= 24:
            break
    return " | ".join(terms)


class RetrievalError(RuntimeError):
    """Base class for safe retrieval errors exposed to internal callers."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RetrievalEmbeddingError(RetrievalError):
    """Raised when query embedding could not be produced."""


class RetrievalDatabaseError(RetrievalError):
    """Raised when the vector query could not be completed."""


class RetrievalConfigurationError(RetrievalError):
    """Raised for invalid retrieval limits."""


class RetrievalSettings(BaseModel):
    """Bounded retrieval result-count settings."""

    model_config = ConfigDict(frozen=True)

    default_top_k: int = Field(default=5, ge=1, le=50)
    max_top_k: int = Field(default=50, ge=1, le=100)
    candidate_top_k: int = Field(default=20, ge=1, le=200)
    max_candidate_top_k: int = Field(default=200, ge=1, le=500)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    reranker_enabled: bool = False
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=200)
    reranker_candidate_k: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def default_does_not_exceed_maximum(self) -> "RetrievalSettings":
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k cannot exceed max_top_k")
        if self.candidate_top_k > self.max_candidate_top_k:
            raise ValueError("candidate_top_k cannot exceed max_candidate_top_k")
        return self


def _load_root_environment() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


@lru_cache(maxsize=1)
def get_retrieval_settings() -> RetrievalSettings:
    """Read validated result limits from the repository environment."""
    _load_root_environment()
    try:
        return RetrievalSettings(
            default_top_k=int(os.getenv("PAGEGLANCE_RETRIEVAL_TOP_K", "5")),
            max_top_k=int(os.getenv("PAGEGLANCE_RETRIEVAL_MAX_TOP_K", "50")),
            candidate_top_k=int(os.getenv("PAGEGLANCE_RETRIEVAL_CANDIDATE_K", "20")),
            max_candidate_top_k=int(os.getenv("PAGEGLANCE_RETRIEVAL_MAX_CANDIDATE_K", "200")),
            rrf_constant=int(os.getenv("PAGEGLANCE_RETRIEVAL_RRF_K", "60")),
            reranker_enabled=os.getenv("PAGEGLANCE_RERANKER_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            reranker_model=os.getenv(
                "PAGEGLANCE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            reranker_candidate_k=int(os.getenv("PAGEGLANCE_RERANKER_CANDIDATE_K", "20")),
        )
    except (ValueError, ValidationError):
        raise RetrievalConfigurationError(
            "retrieval_configuration_error",
            "Retrieval limits are invalid; check PAGEGLANCE_RETRIEVAL_* settings.",
        ) from None


class RetrievedChunk(BaseModel):
    """A ranked chunk with its source document and citation metadata."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    title: str
    source_type: str
    source_url: str | None
    filename: str | None
    document_metadata: dict[str, Any]
    chunk_metadata: dict[str, Any]
    similarity: float


def build_similarity_query(
    query_vector: list[float],
    *,
    model_version: str,
    dimensions: int,
    top_k: int,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    source_type: str | None = None,
    source_url: str | None = None,
    storage_profile: str = "legacy",
) -> Select[tuple[DocumentChunkRecord, DocumentRecord, Any]]:
    """Build a PostgreSQL pgvector cosine query with scope filters applied first."""
    if storage_profile == "local":
        vector_column = DocumentChunkRecord.local_embedding
        model_column = DocumentChunkRecord.local_embedding_model
        dimensions_column = DocumentChunkRecord.local_embedding_dimensions
    elif storage_profile == "legacy":
        vector_column = DocumentChunkRecord.embedding
        model_column = DocumentChunkRecord.embedding_model
        dimensions_column = DocumentChunkRecord.embedding_dimensions
    else:
        raise RetrievalConfigurationError(
            "retrieval_embedding_profile_invalid", "The embedding storage profile is invalid."
        )
    distance = vector_column.cosine_distance(query_vector)
    statement = (
        select(DocumentChunkRecord, DocumentRecord, (1 - distance).label("similarity"))
        .join(DocumentRecord, DocumentRecord.id == DocumentChunkRecord.document_id)
        .where(
            vector_column.is_not(None),
            model_column == model_version,
            dimensions_column == dimensions,
        )
    )
    if document_id is not None:
        statement = statement.where(DocumentChunkRecord.document_id == document_id)
    if document_ids is not None:
        statement = statement.where(DocumentChunkRecord.document_id.in_(document_ids))
    if source_type is not None:
        statement = statement.where(DocumentRecord.source_type == source_type)
    if source_url is not None:
        statement = statement.where(DocumentRecord.source_url == source_url)
    return statement.order_by(distance.asc()).limit(top_k)


def _full_text_expression():
    """Use a fixed, language-neutral config matching the expression GIN index."""
    return func.to_tsvector(literal_column("'simple'::regconfig"), DocumentChunkRecord.text)


def build_lexical_query(
    query: str,
    *,
    top_k: int,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    source_type: str | None = None,
    source_url: str | None = None,
) -> Select[tuple[DocumentChunkRecord, DocumentRecord, Any]]:
    """Build a PostgreSQL full-text candidate query with scope filters applied first."""
    document_tsv = _full_text_expression()
    lexical_query = _lexical_query_string(query)
    if not lexical_query:
        raise RetrievalConfigurationError(
            "retrieval_query_has_no_terms", "The search question has no searchable terms."
        )
    query_tsv = func.to_tsquery(literal_column("'simple'::regconfig"), lexical_query)
    lexical_score = func.ts_rank_cd(document_tsv, query_tsv).label("lexical_score")
    statement = (
        select(DocumentChunkRecord, DocumentRecord, lexical_score)
        .join(DocumentRecord, DocumentRecord.id == DocumentChunkRecord.document_id)
        .where(document_tsv.bool_op("@@")(query_tsv))
    )
    if document_id is not None:
        statement = statement.where(DocumentChunkRecord.document_id == document_id)
    if document_ids is not None:
        statement = statement.where(DocumentChunkRecord.document_id.in_(document_ids))
    if source_type is not None:
        statement = statement.where(DocumentRecord.source_type == source_type)
    if source_url is not None:
        statement = statement.where(DocumentRecord.source_url == source_url)
    return statement.order_by(lexical_score.desc()).limit(top_k)


def fuse_rankings(
    vector_results: list["RetrievedChunk"],
    lexical_results: list["RetrievedChunk"],
    *,
    rrf_constant: int,
) -> list["RetrievedChunk"]:
    """Merge ranked candidate lists using Reciprocal Rank Fusion and chunk identity."""
    scores: dict[str, float] = {}
    candidates: dict[str, RetrievedChunk] = {}
    for ranking in (vector_results, lexical_results):
        for rank, candidate in enumerate(ranking, start=1):
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1 / (
                rrf_constant + rank
            )
            candidates.setdefault(candidate.chunk_id, candidate)
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -scores[candidate.chunk_id],
            -candidate.similarity,
            candidate.chunk_index,
            candidate.chunk_id,
        ),
    )


class VectorRetriever:
    """Search with local vectors and PostgreSQL lexical ranking, then fuse candidates."""

    def __init__(
        self,
        session: Session,
        embedding_service: EmbeddingService | None = None,
        settings: RetrievalSettings | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.session = session
        self.embedding_service = embedding_service or get_default_embedding_service()
        self.settings = settings or get_retrieval_settings()
        self.reranker = reranker

    def _validate_limit(self, top_k: int | None) -> int:
        result_limit = self.settings.default_top_k if top_k is None else top_k
        if result_limit < 1 or result_limit > self.settings.max_top_k:
            raise RetrievalConfigurationError(
                "retrieval_limit_invalid",
                f"top_k must be between 1 and {self.settings.max_top_k}.",
            )
        return result_limit

    @staticmethod
    def _validate_scope(document_id: str | None, document_ids: list[str] | None) -> None:
        if document_id is not None and document_ids is not None:
            raise RetrievalConfigurationError(
                "retrieval_scope_invalid", "Choose one document ID or a document ID list."
            )
        if document_ids is not None and (
            len(document_ids) > 5 or len(set(document_ids)) != len(document_ids)
        ):
            raise RetrievalConfigurationError(
                "retrieval_scope_invalid", "At most five unique document IDs may be selected."
            )

    def _embed_query(self, query: str) -> list[float]:
        try:
            return self.embedding_service.embed_texts([query.strip()])[0]
        except EmbeddingError as error:
            raise RetrievalEmbeddingError(
                error.code, "Could not create an embedding for the search query."
            ) from None

    def _execute_vector_search(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None,
        document_ids: list[str] | None,
        source_type: str | None,
        source_url: str | None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embed_query(query)
        statement = build_similarity_query(
            query_vector,
            model_version=self.embedding_service.model_version,
            dimensions=self.embedding_service.dimensions,
            top_k=top_k,
            document_id=document_id,
            document_ids=document_ids,
            source_type=source_type,
            source_url=source_url,
            storage_profile=getattr(self.embedding_service, "storage_profile", "legacy"),
        )
        try:
            rows = self.session.execute(statement).all()
        except SQLAlchemyError:
            raise RetrievalDatabaseError(
                "retrieval_database_error", "Vector search is temporarily unavailable."
            ) from None
        return [
            self._to_retrieved_chunk((chunk, document), float(similarity))
            for chunk, document, similarity in rows
        ]

    def _execute_lexical_search(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None,
        document_ids: list[str] | None,
        source_type: str | None,
        source_url: str | None,
    ) -> list[RetrievedChunk]:
        statement = build_lexical_query(
            query.strip(),
            top_k=top_k,
            document_id=document_id,
            document_ids=document_ids,
            source_type=source_type,
            source_url=source_url,
        )
        try:
            rows = self.session.execute(statement).all()
        except SQLAlchemyError:
            raise RetrievalDatabaseError(
                "retrieval_database_error", "Lexical search is temporarily unavailable."
            ) from None
        # Keep lexical rank separate from vector similarity; fused ordering is internal.
        return [
            self._to_retrieved_chunk((chunk, document), 0.0) for chunk, document, _score in rows
        ]

    @staticmethod
    def _to_retrieved_chunk(row, similarity: float) -> RetrievedChunk:
        chunk, document = row
        return RetrievedChunk(
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
            document_metadata=dict(document.document_metadata),
            chunk_metadata=dict(chunk.chunk_metadata),
            similarity=similarity,
        )

    def retrieve_vector_only(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        source_type: str | None = None,
        source_url: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return cosine-ranked candidates only (evaluation and comparison path)."""
        if not query.strip():
            return []
        self._validate_scope(document_id, document_ids)
        result_limit = self._validate_limit(top_k)
        return self._execute_vector_search(
            query,
            top_k=result_limit,
            document_id=document_id,
            document_ids=document_ids,
            source_type=source_type,
            source_url=source_url,
        )

    def retrieve_lexical_only(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        source_type: str | None = None,
        source_url: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return PostgreSQL full-text-ranked candidates without embedding the query."""
        if not query.strip() or not _lexical_query_string(query):
            return []
        self._validate_scope(document_id, document_ids)
        result_limit = self._validate_limit(top_k)
        return self._execute_lexical_search(
            query,
            top_k=result_limit,
            document_id=document_id,
            document_ids=document_ids,
            source_type=source_type,
            source_url=source_url,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        source_type: str | None = None,
        source_url: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return RRF-fused vector and lexical candidates within the requested scope."""
        if not query.strip():
            return []
        self._validate_scope(document_id, document_ids)
        result_limit = self._validate_limit(top_k)
        candidate_limit = min(
            self.settings.max_candidate_top_k,
            max(self.settings.candidate_top_k, result_limit),
        )
        vector_results = self._execute_vector_search(
            query,
            top_k=candidate_limit,
            document_id=document_id,
            document_ids=document_ids,
            source_type=source_type,
            source_url=source_url,
        )
        lexical_results = (
            self._execute_lexical_search(
                query,
                top_k=candidate_limit,
                document_id=document_id,
                document_ids=document_ids,
                source_type=source_type,
                source_url=source_url,
            )
            if _lexical_query_string(query)
            else []
        )
        fused = fuse_rankings(
            vector_results,
            lexical_results,
            rrf_constant=self.settings.rrf_constant,
        )
        if self.settings.reranker_enabled and fused:
            candidate_limit = min(
                self.settings.reranker_candidate_k, self.settings.max_candidate_top_k, 20
            )
            candidates = fused[:candidate_limit]
            try:
                reranker = self.reranker
                if reranker is None:
                    from app.reranking import get_default_reranker

                    reranker = get_default_reranker(self.settings.reranker_model)
                reranked = reranker.rerank(query.strip(), candidates)
                candidate_ids = {item.chunk_id for item in candidates}
                ordered_ids = list(
                    dict.fromkeys(
                        item.chunk_id for item in reranked if item.chunk_id in candidate_ids
                    )
                )
                by_id = {item.chunk_id: item for item in candidates}
                ordered = [by_id[item_id] for item_id in ordered_ids]
                ordered.extend(item for item in candidates if item.chunk_id not in ordered_ids)
                return (ordered + fused[candidate_limit:])[:result_limit]
            except Exception as error:
                logger.warning(
                    "retrieval_rerank_failed code=%s",
                    type(error).__name__,
                )
        return fused[:result_limit]
