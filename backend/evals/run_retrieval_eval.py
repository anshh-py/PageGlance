"""Run the versioned local-vector, PostgreSQL full-text, and hybrid RAG evaluation.

Run from backend/ with PAGEGLANCE_TEST_DATABASE_URL pointing only at localhost's
pageglance_test database. This script uses the configured local embedding model and
never calls Groq or a hosted embedding provider.
"""

import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.db import DatabaseConfigurationError
from app.embedding_pipeline import embed_and_persist
from app.embeddings import get_default_embedding_service
from app.ingestion import ingest_text
from app.models import DocumentRecord
from app.repository import DocumentRepository
from app.retrieval import VectorRetriever, get_retrieval_settings
from app.retrieval_evaluation import aggregate_metrics, rank_metrics

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "rag_eval_v1.json"
REPORT_PATH = EVALUATION_DIR / "reports" / "rag_eval_v1.md"
TOP_K = 5
REPEATS = 3


def _test_database_url() -> str:
    value = os.getenv("PAGEGLANCE_TEST_DATABASE_URL", "").strip()
    if not value:
        raise DatabaseConfigurationError(
            "Set PAGEGLANCE_TEST_DATABASE_URL to a disposable localhost pageglance_test database."
        )
    url = make_url(value)
    if (
        url.drivername != "postgresql+psycopg"
        or url.host not in {"localhost", "127.0.0.1", "::1"}
        or url.database != "pageglance_test"
    ):
        raise DatabaseConfigurationError(
            "Evaluation accepts only localhost PostgreSQL database pageglance_test."
        )
    return value


def _load_dataset() -> dict[str, object]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if dataset.get("version") != "rag-eval-v1":
        raise ValueError("Unsupported retrieval evaluation dataset version.")
    return dataset


def _scope_arguments(scope: dict[str, str]) -> dict[str, str]:
    allowed = {"document_id", "source_type", "source_url"}
    if set(scope) - allowed:
        raise ValueError("Evaluation scope contains unsupported filters.")
    return scope


def _index_corpus(
    session: Session, dataset: dict[str, object], service, fixture_ids: set[str]
) -> None:
    repository = DocumentRepository(session)
    for entry in dataset["corpus"]:
        document = ingest_text(
            entry["text"],
            source_type=entry["source_type"],
            title=entry["title"],
            filename=entry.get("filename"),
            source_url=entry.get("source_url"),
            metadata={"eval_dataset": dataset["version"], "eval_ref": entry["ref"]},
        )
        if document.document.id != entry["document_id"]:
            raise ValueError(f"Dataset document identity changed for {entry['ref']}.")
        if len(document.chunks) != 1 or document.chunks[0].id != entry["chunk_id"]:
            raise ValueError(f"Dataset chunk identity changed for {entry['ref']}.")
        existing = repository.get_by_id(document.document.id)
        if existing is not None:
            raise ValueError(
                f"Evaluation fixture {entry['ref']} already exists; refusing to overwrite "
                "or delete it."
            )
        fixture_ids.add(document.document.id)
        embed_and_persist(document, repository=repository, service=service)


def _scope_violation(item, scope: dict[str, str]) -> bool:
    return (
        ("document_id" in scope and item.document_id != scope["document_id"])
        or ("source_type" in scope and item.source_type != scope["source_type"])
        or ("source_url" in scope and item.source_url != scope["source_url"])
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) < 2:
        return ordered[0] if ordered else 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _plan_lines(engine) -> tuple[list[str], list[str]]:
    statement = text(
        "EXPLAIN (COSTS OFF) SELECT id FROM document_chunks "
        "WHERE to_tsvector('simple'::regconfig, text) @@ "
        "websearch_to_tsquery('simple'::regconfig, :query) LIMIT 20"
    )
    with engine.connect() as connection:
        connection.execute(text("ANALYZE document_chunks"))
        actual = [
            str(line)
            for line in connection.execute(
                statement, {"query": "PageGlance local embeddings"}
            ).scalars()
        ]
        connection.commit()
        with connection.begin():
            connection.execute(text("SET LOCAL enable_seqscan = off"))
            indexed = [
                str(line)
                for line in connection.execute(
                    statement, {"query": "PageGlance local embeddings"}
                ).scalars()
            ]
    return actual, indexed


def _render_report(
    dataset: dict[str, object],
    summaries: dict[str, dict[str, object]],
    latencies: dict[str, list[float]],
    unsupported_hits: dict[str, int],
    scope_violations: dict[str, int],
    actual_plan: list[str],
    indexed_plan: list[str],
) -> str:
    lines = [
        f"# PageGlance retrieval evaluation — {dataset['version']}",
        "",
        f"- Embedding profile: `{dataset['embedding_profile']}`",
        f"- Queries: {len(dataset['cases'])}; retrieval cutoff: top-{TOP_K}; repeats: {REPEATS}",
        "- Scope isolation is checked against each case's declared document/source filter.",
        "- Unsupported cases are excluded from relevance averages; non-empty retrieval is "
        "reported separately.",
        "- This report evaluates retrieval only. It does not call Groq or score generated answers.",
        "",
        "| Method | Supported queries | Recall@5 | MRR@5 | nDCG@5 | Median ms | p95 ms | "
        "Unsupported with candidates | Scope violations | Citation recall@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_methods = ["vector", "lexical", "rrf"]
    if "reranked" in summaries:
        ordered_methods.append("reranked")
    for method in ordered_methods:
        summary = summaries[method]
        latency = latencies[method]
        lines.append(
            "| {method} | {queries} | {recall} | {mrr} | {ndcg} | {median:.2f} | {p95:.2f} | "
            "{unsupported} | {violations} | {citation} |".format(
                method=method,
                queries=summary["queries"],
                recall=_metric(summary["recall"]),
                mrr=_metric(summary["mrr"]),
                ndcg=_metric(summary["ndcg"]),
                median=statistics.median(latency),
                p95=_percentile(latency, 0.95),
                unsupported=unsupported_hits[method],
                violations=scope_violations[method],
                citation=_metric(summary["citation_recall"]),
            )
        )
    lines.extend(["", "## PostgreSQL query plan", "", "Default planner:", "", "```text"])
    lines.extend(actual_plan)
    lines.extend(
        ["```", "", "GIN-index eligibility probe with sequential scans disabled:", "", "```text"]
    )
    lines.extend(indexed_plan)
    lines.extend(["```", ""])
    return "\n".join(lines)


def _metric(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run() -> str:
    dataset = _load_dataset()
    database_url = _test_database_url()
    service = get_default_embedding_service()
    settings = get_retrieval_settings()
    evaluation_methods = ["vector", "lexical", "rrf"]
    if settings.reranker_enabled:
        evaluation_methods.append("reranked")
    if service.model_version != dataset["embedding_profile"]:
        raise ValueError(
            "Configure the dataset's local MiniLM embedding profile before evaluation."
        )

    engine = create_engine(database_url, pool_pre_ping=True)
    fixture_ids: set[str] = set()
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        with engine.connect() as connection:
            if connection.scalar(text("select current_database()")) != "pageglance_test":
                raise DatabaseConfigurationError("Evaluation database safety check failed.")
            revision = connection.scalar(text("select version_num from alembic_version"))
            if revision != "20261024_01":
                raise DatabaseConfigurationError(
                    "Apply Alembic migrations to head before running retrieval evaluation."
                )
        # Warm the local model outside measured query times.
        service.embed_texts(["retrieval evaluation warmup"])
        with factory() as session:
            _index_corpus(session, dataset, service, fixture_ids)
        summaries: dict[str, dict[str, object]] = {}
        latencies: dict[str, list[float]] = {method: [] for method in evaluation_methods}
        unsupported_hits = dict.fromkeys(latencies, 0)
        scope_violations = dict.fromkeys(latencies, 0)
        metric_rows: dict[str, list[dict[str, float | None]]] = {method: [] for method in latencies}
        citation_hits: dict[str, list[float]] = {method: [] for method in latencies}
        for case in dataset["cases"]:
            scope = _scope_arguments(case["scope"])
            with factory() as session:
                baseline_settings = settings.model_copy(update={"reranker_enabled": False})
                retriever = VectorRetriever(session, service, baseline_settings)
                methods = {
                    "vector": partial(retriever.retrieve_vector_only, case["question"], **scope),
                    "lexical": partial(retriever.retrieve_lexical_only, case["question"], **scope),
                    "rrf": partial(retriever.retrieve, case["question"], **scope),
                }
                if settings.reranker_enabled:
                    reranked_retriever = VectorRetriever(session, service, settings)
                    methods["reranked"] = partial(
                        reranked_retriever.retrieve, case["question"], **scope
                    )
                for method, invoke in methods.items():
                    results = []
                    durations = []
                    for _ in range(REPEATS):
                        started = time.perf_counter()
                        results = invoke()
                        durations.append((time.perf_counter() - started) * 1000)
                    latencies[method].append(statistics.median(durations))
                    ids = [item.chunk_id for item in results]
                    relevant = set(case["relevant_chunk_ids"])
                    metric = rank_metrics(relevant, ids, k=TOP_K)
                    if relevant:
                        metric_rows[method].append(metric)
                        expected = set(case["expected_citations"])
                        citation_hits[method].append(
                            len(expected.intersection(ids[:TOP_K])) / len(expected)
                            if expected
                            else 1.0
                        )
                    elif ids:
                        unsupported_hits[method] += 1
                    scope_violations[method] += sum(
                        _scope_violation(item, scope) for item in results
                    )
        for method in latencies:
            summaries[method] = {
                **aggregate_metrics(metric_rows[method]),
                "citation_recall": (
                    sum(citation_hits[method]) / len(citation_hits[method])
                    if citation_hits[method]
                    else None
                ),
            }
        actual_plan, indexed_plan = _plan_lines(engine)
        report = _render_report(
            dataset,
            summaries,
            latencies,
            unsupported_hits,
            scope_violations,
            actual_plan,
            indexed_plan,
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        return report
    finally:
        if fixture_ids:
            with factory() as session:
                records = session.scalars(
                    select(DocumentRecord).where(DocumentRecord.id.in_(fixture_ids))
                ).all()
                for record in records:
                    if record.document_metadata.get("eval_dataset") == dataset["version"]:
                        session.delete(record)
                session.commit()
        engine.dispose()


if __name__ == "__main__":
    print(run())
