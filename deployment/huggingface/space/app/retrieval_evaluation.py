"""Small, dependency-free retrieval metrics used by the versioned evaluation suite."""

import math
from collections.abc import Sequence


def rank_metrics(
    relevant_ids: set[str] | Sequence[str],
    ranked_ids: Sequence[str],
    *,
    k: int = 5,
) -> dict[str, float | None]:
    """Compute binary Recall@K, MRR@K, and nDCG@K for one judged query."""
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_ids)
    if not relevant:
        return {"recall": None, "mrr": None, "ndcg": None}

    top = list(ranked_ids[:k])
    hits = [index for index, chunk_id in enumerate(top, start=1) if chunk_id in relevant]
    recall = len({chunk_id for chunk_id in top if chunk_id in relevant}) / len(relevant)
    mrr = 1 / hits[0] if hits else 0.0

    def dcg(gains: Sequence[float]) -> float:
        return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))

    dcg_value = dcg([1.0 if chunk_id in relevant else 0.0 for chunk_id in top])
    ideal = dcg([1.0] * min(len(relevant), k))
    return {"recall": recall, "mrr": mrr, "ndcg": dcg_value / ideal if ideal else 0.0}


def aggregate_metrics(
    cases: Sequence[dict[str, float | None]],
) -> dict[str, float | int | None]:
    """Average supported-query metrics while reporting how many queries were judged."""
    values: dict[str, list[float]] = {"recall": [], "mrr": [], "ndcg": []}
    for case in cases:
        for metric in values:
            value = case.get(metric)
            if value is not None:
                values[metric].append(value)
    return {
        "queries": sum(case.get("recall") is not None for case in cases),
        **{
            metric: sum(scores) / len(scores) if scores else None
            for metric, scores in values.items()
        },
    }
