"""Internal evaluation-only pipeline interface for capturing RAG execution artifacts.

This module provides a non-invasive harness to capture full retrieved context text,
chunk IDs, generated answers, and grounding validation results for evaluation frameworks
(such as Ragas and custom retrieval metrics) without altering public API contracts.
"""

from dataclasses import dataclass, field
import time
from typing import Any

from app.llm import GroqAnswerService, get_answer_service
from app.main import _ABSTENTION_PHRASES, _validate_grounded_answer
from app.rag import PreparedAnswerContext, RagOrchestrator, rag_orchestrator


@dataclass
class EvaluationSample:
    """Captured inputs, outputs, and metadata for one evaluation query."""

    question: str
    generated_answer: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_source_ids: list[str]
    reference_answer: str | None
    relevant_chunk_ids: list[str]
    expected_abstention: bool = False
    latency_ms: float = 0.0
    retrieval_used: bool = True
    fallback_code: str | None = None
    model_abstained: bool = False
    valid_citations: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to JSON-serializable dictionary."""
        return {
            "question": self.question,
            "generated_answer": self.generated_answer,
            "retrieved_contexts": self.retrieved_contexts,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_source_ids": self.retrieved_source_ids,
            "reference_answer": self.reference_answer,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "expected_abstention": self.expected_abstention,
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_used": self.retrieval_used,
            "fallback_code": self.fallback_code,
            "model_abstained": self.model_abstained,
            "valid_citations": self.valid_citations,
            "metadata": self.metadata,
        }


def capture_rag_sample(
    question: str,
    *,
    page: dict[str, str] | None = None,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    source_type: str | None = None,
    reference_answer: str | None = None,
    relevant_chunk_ids: list[str] | None = None,
    expected_abstention: bool = False,
    orchestrator: RagOrchestrator | None = None,
    answer_service: GroqAnswerService | None = None,
    generate_answer: bool = True,
    context_limit: int = 16_000,
    metadata: dict[str, Any] | None = None,
) -> EvaluationSample:
    """Execute retrieval and generation through the existing pipeline and capture evaluation artifacts.

    Does not modify API schemas or expose retrieved chunk texts to public HTTP clients.
    """
    t0 = time.perf_counter()
    active_orchestrator = orchestrator or rag_orchestrator
    page_context = page or {"title": "", "url": "", "text": ""}
    prepared: PreparedAnswerContext = active_orchestrator.prepare(
        question,
        page_context,
        context_limit=context_limit,
        document_id=document_id,
        document_ids=document_ids,
        source_type=source_type,
        retrieval_enabled=True,
    )

    sources = prepared.retrieved_sources or []
    retrieved_contexts = [
        source["text"] for source in sources if isinstance(source.get("text"), str)
    ]
    retrieved_chunk_ids = [
        str(source["chunk_id"]) for source in sources if source.get("chunk_id") is not None
    ]
    retrieved_source_ids = [
        str(source["id"]) for source in sources if source.get("id") is not None
    ]

    generated_answer = ""
    model_abstained = False
    valid_citations = True

    if generate_answer:
        service = answer_service or get_answer_service()
        raw_answer = service.answer(
            question,
            prepared.page_context,
            history=[],
            retrieved_sources=prepared.retrieved_sources,
        )
        validated_answer = _validate_grounded_answer(
            raw_answer, prepared.retrieved_sources, prepared.page_context
        )
        model_abstained = any(phrase in validated_answer.casefold() for phrase in _ABSTENTION_PHRASES)
        valid_citations = validated_answer == raw_answer
        generated_answer = validated_answer

    latency_ms = (time.perf_counter() - t0) * 1000

    return EvaluationSample(
        question=question,
        generated_answer=generated_answer,
        retrieved_contexts=retrieved_contexts,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_source_ids=retrieved_source_ids,
        reference_answer=reference_answer,
        relevant_chunk_ids=relevant_chunk_ids or [],
        expected_abstention=expected_abstention,
        latency_ms=latency_ms,
        retrieval_used=prepared.retrieval_used,
        fallback_code=prepared.fallback_code,
        model_abstained=model_abstained,
        valid_citations=valid_citations,
        metadata=metadata or {},
    )

