"""Optional, bounded local cross-encoder reranking for retrieval candidates."""

from functools import lru_cache
from typing import Protocol

from app.retrieval import RetrievedChunk


class Reranker(Protocol):
    """Rerank only the candidates supplied by the already-scoped retriever."""

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class LocalCrossEncoderReranker:
    """Use a local Sentence Transformers cross-encoder with fixed CPU/resource bounds."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError("reranker_dependency_unavailable") from None
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        bounded = candidates[:50]
        if not bounded:
            return []
        pairs = [(query[:2000], candidate.text[:4000]) for candidate in bounded]
        scores = self.model.predict(
            pairs,
            batch_size=min(8, len(pairs)),
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(bounded, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [candidate for candidate, _score in ranked]


@lru_cache(maxsize=2)
def get_default_reranker(model_name: str) -> LocalCrossEncoderReranker:
    """Load at most two configured local reranker models per backend process."""
    return LocalCrossEncoderReranker(model_name)
