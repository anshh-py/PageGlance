"""Tests for Ragas evaluation integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.evaluation_interface import EvaluationSample
from app.ragas_embeddings import PageGlanceRagasEmbedding
from evals import run_ragas_eval


class FakeEmbeddingService:
    model_version = (
        "sentence_transformers:"
        "sentence-transformers/all-MiniLM-L6-v2:dimensions=384"
    )
    dimensions = 384

    def embed_texts(self, texts):
        return [[0.1] * 384 for _ in texts]


def make_sample(
    question="What is X?",
    answer="X is Y.",
    contexts=None,
):
    """Create a reusable evaluation sample."""
    if contexts is None:
        contexts = ["context"]

    return EvaluationSample(
        question=question,
        generated_answer=answer,
        retrieved_contexts=contexts,
        retrieved_chunk_ids=["chunk-1"] if contexts else [],
        retrieved_source_ids=["S1"] if contexts else [],
        reference_answer="X is Y.",
        relevant_chunk_ids=["chunk-1"],
    )


def configure_run_mocks(
    monkeypatch,
    capture_side_effect=None,
    evaluate_side_effect=None,
    metric_rows=None,
):
    """Mock external Ragas dependencies for run() tests."""

    mock_setup_llm = MagicMock(return_value=MagicMock())
    mock_setup_embeddings = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(
        run_ragas_eval,
        "_setup_llm",
        mock_setup_llm,
    )
    monkeypatch.setattr(
        run_ragas_eval,
        "_setup_embeddings",
        mock_setup_embeddings,
    )

    metric_mocks = {}

    for metric_name in (
        "Faithfulness",
        "AnswerRelevancy",
        "ContextPrecision",
        "ContextRecall",
    ):
        metric_mock = MagicMock(return_value=MagicMock())
        metric_mocks[metric_name] = metric_mock
        monkeypatch.setattr(
            run_ragas_eval,
            metric_name,
            metric_mock,
        )

    mock_capture = MagicMock()

    if capture_side_effect is not None:
        mock_capture.side_effect = capture_side_effect

    monkeypatch.setattr(
        run_ragas_eval,
        "capture_rag_sample",
        mock_capture,
    )

    mock_evaluate = MagicMock()

    if evaluate_side_effect is not None:
        mock_evaluate.side_effect = evaluate_side_effect
    else:
        if metric_rows is None:
            metric_rows = {
                "faithfulness": [0.9],
                "answer_relevancy": [0.8],
                "context_precision": [0.7],
                "context_recall": [0.6],
            }

        mock_result = MagicMock()
        mock_result.to_pandas.return_value = pd.DataFrame(metric_rows)
        mock_evaluate.return_value = mock_result

    monkeypatch.setattr(
        run_ragas_eval,
        "evaluate",
        mock_evaluate,
    )

    return {
        "capture": mock_capture,
        "evaluate": mock_evaluate,
        "metrics": metric_mocks,
        "setup_llm": mock_setup_llm,
        "setup_embeddings": mock_setup_embeddings,
    }


def test_pageglance_ragas_embedding_wraps_service():
    service = FakeEmbeddingService()
    emb = PageGlanceRagasEmbedding(service)

    result = emb.embed_text("hello")

    assert len(result) == 384

    results = emb.embed_texts(["hello", "world"])

    assert len(results) == 2
    assert len(results[0]) == 384


def test_dataset_to_ragas_samples_conversion():
    dataset = run_ragas_eval._load_dataset()

    sample = EvaluationSample(
        question="What is X?",
        generated_answer="X is Y.",
        retrieved_contexts=["context about X"],
        retrieved_chunk_ids=["chunk-1"],
        retrieved_source_ids=["S1"],
        reference_answer="X is Y.",
        relevant_chunk_ids=["chunk-1"],
    )

    ragas_ds = run_ragas_eval._dataset_to_ragas_samples(
        dataset,
        [sample],
    )

    assert len(ragas_ds) == 1


@patch("evals.run_ragas_eval.OpenAI")
@patch("evals.run_ragas_eval.instructor")
def test_setup_llm_requires_groq_key(mock_instructor, mock_openai):
    import os

    old = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = "test-key"

    try:
        llm = run_ragas_eval._setup_llm()

        mock_openai.assert_called_once()
        mock_instructor.from_openai.assert_called_once()
    finally:
        if old is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = old


@patch("evals.run_ragas_eval.PageGlanceRagasEmbedding")
def test_setup_embeddings_returns_embedding(mock_emb):
    emb = run_ragas_eval._setup_embeddings()

    mock_emb.assert_called_once()


def test_evaluation_construction_metric_execution_and_reporting(
    monkeypatch,
):
    sample = make_sample(
        question=(
            "What is the exact identifier of the local embedding model?"
        ),
        answer=(
            "The model is sentence-transformers/all-MiniLM-L6-v2."
        ),
    )

    mocks = configure_run_mocks(
        monkeypatch,
        capture_side_effect=lambda question, **kwargs: sample,
    )

    report_json, report_md = run_ragas_eval.run(small_run=True)

    for metric_name, metric_mock in mocks["metrics"].items():
        metric_mock.assert_called()

    mocks["evaluate"].assert_called_once()

    assert report_json["num_samples"] >= 1
    assert "faithfulness" in report_json["metrics"]
    assert report_md is not None


def test_empty_retrieval_context_handled(monkeypatch):
    """Empty retrieved contexts should be captured without raising."""

    sample = make_sample(
        question="What is X?",
        answer="X is Y.",
        contexts=[],
    )

    mocks = configure_run_mocks(
        monkeypatch,
        capture_side_effect=lambda question, **kwargs: sample,
    )

    report_json, report_md = run_ragas_eval.run(
        small_run=True,
        max_cases=1,
    )

    assert report_json["num_samples"] == 1
    assert report_json["samples"][0]["num_contexts"] == 0
    assert report_json["samples"][0]["answer"] == "X is Y."

    mocks["evaluate"].assert_called_once()


def test_retrieval_exception_isolated(monkeypatch):
    """A retrieval failure in one case must not stop later cases."""

    call_count = [0]

    def capture_side_effect(question, **kwargs):
        call_count[0] += 1

        if call_count[0] == 1:
            raise RuntimeError("retrieval failed")

        return make_sample(
            question=question,
            answer="OK",
        )

    mocks = configure_run_mocks(
        monkeypatch,
        capture_side_effect=capture_side_effect,
    )

    report_json, report_md = run_ragas_eval.run(small_run=True)

    assert call_count[0] == 3
    assert report_json["num_samples"] == 2
    assert report_json["num_failed"] == 1

    failed_case = report_json["failed_cases"][0]

    assert failed_case["stage"] == "capture_rag_sample"
    assert "retrieval failed" in failed_case["error"]

    assert len(report_json["samples"]) == 2

    mocks["evaluate"].assert_called_once()


def test_generation_exception_isolated(monkeypatch):
    """One generation failure must not stop subsequent cases."""

    call_count = [0]

    def capture_side_effect(question, **kwargs):
        call_count[0] += 1

        if call_count[0] == 1:
            raise RuntimeError("LLM generation failed")

        return make_sample(
            question=question,
            answer="OK",
        )

    mocks = configure_run_mocks(
        monkeypatch,
        capture_side_effect=capture_side_effect,
    )

    report_json, report_md = run_ragas_eval.run(small_run=True)

    assert call_count[0] == 3
    assert report_json["num_samples"] == 2
    assert report_json["num_failed"] == 1

    failed_case = report_json["failed_cases"][0]

    assert failed_case["stage"] == "capture_rag_sample"
    assert "LLM generation failed" in failed_case["error"]

    assert all(
        sample["answer"] == "OK"
        for sample in report_json["samples"]
    )

    mocks["evaluate"].assert_called_once()


def test_successful_case_after_failed_case(monkeypatch):
    """Successful cases after a failure must remain in the report."""

    call_count = [0]

    def capture_side_effect(question, **kwargs):
        call_count[0] += 1

        if call_count[0] == 1:
            raise RuntimeError("first case failed")

        return make_sample(
            question=question,
            answer=f"Answer for {question}",
        )

    mocks = configure_run_mocks(
        monkeypatch,
        capture_side_effect=capture_side_effect,
        metric_rows={
            "faithfulness": [0.9, 0.8],
            "answer_relevancy": [0.8, 0.7],
            "context_precision": [0.7, 0.6],
            "context_recall": [0.6, 0.5],
        },
    )

    report_json, report_md = run_ragas_eval.run(small_run=True)

    assert report_json["num_samples"] == 2
    assert report_json["num_failed"] == 1

    assert report_json["failed_cases"][0]["stage"] == (
        "capture_rag_sample"
    )

    for sample in report_json["samples"]:
        assert sample["answer"] == (
            f"Answer for {sample['question']}"
        )

    mocks["evaluate"].assert_called_once()


def test_invalid_dataset_entry_rejected():
    """Dataset with missing required fields should raise ValueError."""

    invalid_dataset = {
        "version": "rag-eval-v2",
        "embedding_profile": "test",
        "corpus": [],
        "cases": [{"id": "case1"}],
    }

    original_path = run_ragas_eval.DATASET_PATH

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as file:
        json.dump(invalid_dataset, file)
        temp_path = file.name

    try:
        run_ragas_eval.DATASET_PATH = temp_path

        with pytest.raises(ValueError, match="missing required"):
            run_ragas_eval._load_dataset()
    finally:
        run_ragas_eval.DATASET_PATH = original_path
        Path(temp_path).unlink(missing_ok=True)


def test_invalid_relevant_chunk_id_rejected():
    """A case referencing an unknown chunk_id should raise ValueError."""

    invalid_dataset = {
        "version": "rag-eval-v2",
        "embedding_profile": "test",
        "corpus": [
            {
                "chunk_id": "known-chunk",
                "document_id": "doc-1",
                "source_type": "text",
                "title": "Test",
                "text": "test",
            }
        ],
        "cases": [
            {
                "id": "case1",
                "question": "What?",
                "scope": {"source_type": "text"},
                "relevant_chunk_ids": ["unknown-chunk"],
            }
        ],
    }

    original_path = run_ragas_eval.DATASET_PATH

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as file:
        json.dump(invalid_dataset, file)
        temp_path = file.name

    try:
        run_ragas_eval.DATASET_PATH = temp_path

        with pytest.raises(ValueError, match="unknown chunk_id"):
            run_ragas_eval._load_dataset()
    finally:
        run_ragas_eval.DATASET_PATH = original_path
        Path(temp_path).unlink(missing_ok=True)


def test_duplicate_case_ids_rejected():
    """Duplicate case IDs should raise ValueError."""

    invalid_dataset = {
        "version": "rag-eval-v2",
        "embedding_profile": "test",
        "corpus": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_type": "text",
                "title": "Test",
                "text": "test",
            }
        ],
        "cases": [
            {
                "id": "same-id",
                "question": "Q1",
                "scope": {"source_type": "text"},
            },
            {
                "id": "same-id",
                "question": "Q2",
                "scope": {"source_type": "text"},
            },
        ],
    }

    original_path = run_ragas_eval.DATASET_PATH

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as file:
        json.dump(invalid_dataset, file)
        temp_path = file.name

    try:
        run_ragas_eval.DATASET_PATH = temp_path

        with pytest.raises(ValueError, match="Duplicate case IDs"):
            run_ragas_eval._load_dataset()
    finally:
        run_ragas_eval.DATASET_PATH = original_path
        Path(temp_path).unlink(missing_ok=True)


def test_metric_evaluation_failure_isolated(monkeypatch):
    """A Ragas evaluate() failure must be recorded without losing samples."""

    sample = make_sample()

    mocks = configure_run_mocks(
        monkeypatch,
        capture_side_effect=lambda question, **kwargs: sample,
        evaluate_side_effect=RuntimeError(
            "metric computation failed"
        ),
    )

    report_json, report_md = run_ragas_eval.run(small_run=True)

    # Captured samples must survive a batch-level metric failure.
    assert report_json["num_samples"] == 3
    assert report_json["num_failed"] == 1

    failed_case = report_json["failed_cases"][0]

    assert failed_case["case_id"] == "ragas-batch"
    assert failed_case["stage"] == "ragas_evaluate"
    assert "metric computation failed" in failed_case["error"]

    mocks["evaluate"].assert_called_once()


def test_unsupported_samples_excluded_from_metrics_but_retained_for_abstention():
    """
    Unsupported and expected-abstention cases must not enter
    the supported metric-evaluation dataset.
    """

    dataset = run_ragas_eval._load_dataset()

    unsupported_cases = [
        case
        for case in dataset["cases"]
        if case.get("category") == "unsupported"
    ]

    abstention_cases = [
        case
        for case in dataset["cases"]
        if case.get("expected_abstention")
    ]

    assert len(unsupported_cases) > 0
    assert len(abstention_cases) > 0

    supported_cases = [
        case
        for case in dataset["cases"]
        if case.get("category") != "unsupported"
        and not case.get("expected_abstention")
        and case.get("relevant_chunk_ids")
        and len(case.get("relevant_chunk_ids")) > 0
    ]

    assert len(supported_cases) > 0

    for case in supported_cases:
        assert case.get("reference") is not None
        assert case.get("relevant_chunk_ids")
        assert len(case["relevant_chunk_ids"]) > 0
        assert not case.get("expected_abstention", False)
        assert case.get("category") != "unsupported"