
"""Run RAGAS evaluation on the PageGlance RAG pipeline."""

import json
import math
import os
import traceback
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import instructor
from dotenv import load_dotenv
from openai import OpenAI

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms.base import InstructorLLM
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.evaluation_interface import capture_rag_sample, EvaluationSample
from app.ragas_embeddings import PageGlanceRagasEmbedding


EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "rag_eval_v2.json"
REPORTS_DIR = EVALUATION_DIR / "reports"

JSON_REPORT_PATH = REPORTS_DIR / "ragas_eval.json"
MD_REPORT_PATH = REPORTS_DIR / "ragas_eval.md"


def _set_report_paths(prefix: str) -> None:
    """Set output paths for a named evaluation run."""
    global JSON_REPORT_PATH, MD_REPORT_PATH

    JSON_REPORT_PATH = REPORTS_DIR / f"{prefix}.json"
    MD_REPORT_PATH = REPORTS_DIR / f"{prefix}.md"


def _json_safe(value: Any) -> Any:
    """Convert values into strict JSON-compatible representations."""

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    # Handle NumPy scalar values and similar scalar objects.
    if not isinstance(value, (str, int, float, bool, type(None))):
        item_method = getattr(value, "item", None)

        if callable(item_method):
            try:
                converted = item_method()

                if converted is not value:
                    return _json_safe(converted)
            except (TypeError, ValueError):
                pass

    return value


def _load_dataset() -> dict[str, Any]:
    """Load and validate the evaluation dataset."""

    dataset_path = Path(DATASET_PATH)

    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    if dataset.get("version") not in {
        "rag-eval-v1",
        "rag-eval-v2",
    }:
        raise ValueError("Unsupported evaluation dataset version.")

    _validate_dataset_schema(dataset)

    return dataset


def _validate_dataset_schema(dataset: dict[str, Any]) -> None:
    """Validate dataset structure, unique IDs, required fields, and corpus references."""

    if "corpus" not in dataset or not isinstance(dataset["corpus"], list):
        raise ValueError("Dataset must contain a 'corpus' list")

    if "cases" not in dataset or not isinstance(dataset["cases"], list):
        raise ValueError("Dataset must contain a 'cases' list")

    corpus_entries = dataset["corpus"]
    cases = dataset["cases"]

    corpus_chunk_ids = set()
    corpus_document_ids = set()

    for entry in corpus_entries:
        if "chunk_id" not in entry:
            raise ValueError(
                "Corpus entry missing required 'chunk_id'"
            )

        if "document_id" not in entry:
            raise ValueError(
                "Corpus entry missing required 'document_id'"
            )

        corpus_chunk_ids.add(entry["chunk_id"])
        corpus_document_ids.add(entry["document_id"])

    case_ids = [case.get("id") for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs found in dataset")

    for case in cases:
        if "id" not in case or not case["id"]:
            raise ValueError("Case missing required 'id'")

        if "question" not in case or not case["question"]:
            raise ValueError(
                f"Case {case.get('id', 'unknown')} "
                "missing required 'question'"
            )

        if "scope" not in case or not isinstance(case["scope"], dict):
            raise ValueError(
                f"Case {case['id']} missing required 'scope' dict"
            )

        relevant_chunk_ids = case.get("relevant_chunk_ids", [])

        for chunk_id in relevant_chunk_ids:
            if chunk_id not in corpus_chunk_ids:
                raise ValueError(
                    f"Case {case['id']} references unknown "
                    f"chunk_id: {chunk_id}"
                )

        scope = case.get("scope", {})

        if (
            "document_id" in scope
            and scope["document_id"] not in corpus_document_ids
        ):
            raise ValueError(
                f"Case {case['id']} scope references unknown "
                f"document_id: {scope['document_id']}"
            )

        if "source_url" in scope:
            matching = [
                entry
                for entry in corpus_entries
                if entry.get("source_url") == scope["source_url"]
            ]

            if not matching:
                raise ValueError(
                    f"Case {case['id']} scope references unknown "
                    f"source_url: {scope['source_url']}"
                )


def _build_page_context_for_case(
    case: dict[str, Any],
    corpus: list[dict[str, Any]],
) -> dict[str, str]:
    """Build the page context corresponding to an evaluation case."""

    scope = case.get("scope", {})

    if "source_url" in scope:
        for entry in corpus:
            if entry.get("source_url") == scope["source_url"]:
                return {
                    "title": entry.get("title", ""),
                    "url": entry.get("source_url", ""),
                    "text": entry.get("text", ""),
                }

    if "document_id" in scope:
        for entry in corpus:
            if entry.get("document_id") == scope["document_id"]:
                return {
                    "title": entry.get("title", ""),
                    "url": entry.get("source_url", "") or "",
                    "text": entry.get("text", ""),
                }

    return {
        "title": "",
        "url": "",
        "text": "",
    }


def _setup_llm():
    """Initialize the Groq-backed RAGAS LLM."""

    from app import llm as app_llm

    app_llm.get_answer_service.cache_clear()

    if hasattr(app_llm.get_settings, "cache_clear"):
        app_llm.get_settings.cache_clear()

    load_dotenv(
        dotenv_path=Path(__file__).resolve().parents[2] / ".env",
        override=True,
    )

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY not configured")

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    ).strip()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    try:
        patched = instructor.from_openai(
            client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
    except Exception:
        try:
            patched = instructor.from_openai(
                client,
                mode=instructor.Mode.MD_JSON,
            )
        except Exception:
            patched = instructor.from_openai(
                client,
                mode=instructor.Mode.JSON,
            )

    return InstructorLLM(
        client=patched,
        model=model,
        provider="openai",
    )


def _setup_embeddings():
    """Initialize PageGlance RAGAS embeddings."""

    return PageGlanceRagasEmbedding()


def _dataset_to_ragas_samples(
    dataset: dict[str, Any],
    samples: list[Any],
) -> EvaluationDataset:
    """Convert captured PageGlance samples into a RAGAS dataset."""

    ragas_samples = []

    for sample in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=sample.question,
                response=sample.generated_answer,
                retrieved_contexts=sample.retrieved_contexts,
                reference=(
                    sample.reference_answer
                    if sample.reference_answer
                    else None
                ),
            )
        )

    return EvaluationDataset(samples=ragas_samples)


def _generate_reports(
    result,
    dataset: dict[str, Any],
    samples: list[EvaluationSample],
    failed_cases: list[dict[str, Any]],
    ragas_dataset: EvaluationDataset,
) -> tuple[dict, str]:
    """Generate JSON and Markdown reports, including failed evaluations."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    scores = {}
    score_means = {}

    if result is not None:
        try:
            df = result.to_pandas()

            excluded_columns = {
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
            }

            for column in df.columns:
                if column not in excluded_columns:
                    scores[column] = df[column].tolist()

            for metric_name, values in scores.items():
                valid_values = []

                for value in values:
                    if value is None:
                        continue

                    if isinstance(value, Real) and not isinstance(value, bool):
                        numeric_value = float(value)

                        if math.isfinite(numeric_value):
                            valid_values.append(numeric_value)

                score_means[metric_name] = (
                    sum(valid_values) / len(valid_values)
                    if valid_values
                    else None
                )

        except Exception as exc:
            score_means = {
                "error": str(exc),
            }
            scores = {}

    timestamp = (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    report_json = {
        "timestamp": timestamp,
        "dataset_version": dataset.get("version"),
        "evaluation_status": (
            "completed" if result is not None else "failed_or_not_run"
        ),
        "num_samples": len(samples),
        "num_failed": len(failed_cases),
        "metrics": score_means,
        "detailed_scores": scores,
        "samples": [
            {
                "question": sample.question,
                "answer": sample.generated_answer,
                "num_contexts": len(sample.retrieved_contexts),
                "reference": sample.reference_answer,
                "expected_abstention": sample.expected_abstention,
            }
            for sample in samples
        ],
        "failed_cases": failed_cases,
    }

    safe_report = _json_safe(report_json)

    JSON_REPORT_PATH.write_text(
        json.dumps(
            safe_report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# PageGlance RAGAS Evaluation Report",
        "",
        f"- Dataset: {dataset.get('version')}",
        f"- Timestamp: {timestamp}",
        f"- Evaluation status: {report_json['evaluation_status']}",
        f"- Successful samples: {len(samples)}",
        f"- Failed cases: {len(failed_cases)}",
        "",
        "## Metric Scores (Mean)",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]

    if not score_means:
        lines.append("| No metrics available | N/A |")

    for metric, score in score_means.items():
        if metric == "error":
            lines.append(f"| Error | {score} |")
        elif score is None:
            lines.append(f"| {metric} | N/A |")
        else:
            lines.append(f"| {metric} | {score:.4f} |")

    lines.extend([
        "",
        "## Sample Details",
        "",
    ])

    for index, sample in enumerate(samples):
        question_preview = sample.question[:80]
        question_suffix = (
            "..." if len(sample.question) > 80 else ""
        )

        lines.append(
            f"### Sample {index + 1}: "
            f"{question_preview}{question_suffix}"
        )

        lines.append(
            f"- Retrieved contexts: {len(sample.retrieved_contexts)}"
        )

        lines.append(
            f"- Expected abstention: {sample.expected_abstention}"
        )

        if sample.reference_answer:
            reference_preview = sample.reference_answer[:100]
            reference_suffix = (
                "..." if len(sample.reference_answer) > 100 else ""
            )

            lines.append(
                f"- Reference: {reference_preview}{reference_suffix}"
            )

        lines.append("")

    if failed_cases:
        lines.extend([
            "## Failed Cases",
            "",
        ])

        for failed_case in failed_cases:
            case_id = failed_case.get("case_id", "unknown")
            category = failed_case.get("category", "unknown")
            question = failed_case.get("question", "")
            stage = failed_case.get("stage", "unknown")
            error = failed_case.get("error", "Unknown error")

            lines.append(
                f"### {case_id} ({category})"
            )

            lines.append(
                f"- Question: {question[:80]}"
                f"{'...' if len(question) > 80 else ''}"
            )

            lines.append(f"- Failure stage: {stage}")
            lines.append(f"- Error: {error}")
            lines.append("")

    md_content = "\n".join(lines)

    MD_REPORT_PATH.write_text(
        md_content,
        encoding="utf-8",
    )

    return safe_report, md_content


def run(
    small_run: bool = False,
    use_mock_llm: bool = False,
    max_cases: int | None = None,
    exclude_unsupported: bool = True,
    report_prefix: str | None = None,
) -> tuple[dict, str]:
    """Capture RAG samples, run RAGAS evaluation, and generate reports."""

    if report_prefix:
        _set_report_paths(report_prefix)

    dataset = _load_dataset()

    corpus = dataset["corpus"]
    cases = dataset["cases"]

    if exclude_unsupported:
        cases = [
            case
            for case in cases
            if (
                case.get("category") != "unsupported"
                and not case.get("expected_abstention")
                and case.get("relevant_chunk_ids")
                and len(case.get("relevant_chunk_ids")) > 0
            )
        ]

    if small_run:
        cases = cases[:3]

    if max_cases is not None:
        if max_cases < 0:
            raise ValueError("max_cases cannot be negative")

        cases = cases[:max_cases]

    samples: list[EvaluationSample] = []
    failed_cases: list[dict[str, Any]] = []

    # Capture RAG samples independently so one failed case
    # does not prevent the remaining cases from running.
    for case in cases:
        case_id = case["id"]
        category = case.get("category", "unknown")
        question = case["question"]

        try:
            page_context = _build_page_context_for_case(
                case,
                corpus,
            )

            scope = case.get("scope", {})

            sample = capture_rag_sample(
                question,
                page=page_context,
                document_id=scope.get("document_id"),
                document_ids=scope.get("document_ids"),
                source_type=scope.get("source_type"),
                reference_answer=case.get("reference"),
                relevant_chunk_ids=case.get(
                    "relevant_chunk_ids",
                    [],
                ),
                expected_abstention=case.get(
                    "expected_abstention",
                    False,
                ),
                generate_answer=True,
                metadata={
                    "case_id": case_id,
                    "category": category,
                    "scope": scope,
                },
            )

            samples.append(sample)

        except Exception as exc:
            failed_cases.append({
                "case_id": case_id,
                "category": category,
                "question": question,
                "stage": "capture_rag_sample",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    # No captured samples means there is nothing to evaluate.
    if not samples:
        ragas_dataset = EvaluationDataset(samples=[])

        return _generate_reports(
            None,
            dataset,
            samples,
            failed_cases,
            ragas_dataset,
        )

    # Conversion, setup and metric initialization are all protected.
    result = None
    stage = "ragas_dataset_conversion"

    try:
        ragas_dataset = _dataset_to_ragas_samples(
            dataset,
            samples,
        )

        stage = "ragas_setup"

        llm = _setup_llm()
        embeddings = _setup_embeddings()

        faithfulness = Faithfulness(llm=llm)

        answer_relevancy = AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        )

        context_precision = ContextPrecision(llm=llm)
        context_recall = ContextRecall(llm=llm)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        stage = "ragas_evaluate"

        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )

    except Exception as exc:
        failed_cases.append({
            "case_id": "ragas-batch",
            "category": "ragas_evaluation",
            "question": "RAGAS dataset conversion, initialization or evaluation",
            "stage": stage,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

        result = None

    # Reports are generated even if initialization or evaluation failed.
    return _generate_reports(
        result,
        dataset,
        samples,
        failed_cases,
        ragas_dataset if "ragas_dataset" in locals()
        else EvaluationDataset(samples=[]),
    )


if __name__ == "__main__":
    try:
        report_json, report_md = run()
        print(report_md)

        if report_json.get("evaluation_status") != "completed":
            raise SystemExit(
                "RAGAS evaluation did not complete successfully. "
                "Inspect the generated reports for failure details."
            )

    except Exception as exc:
        print(f"Error: {exc}")
        raise
