"""Versioned retrieval evaluation fixtures and metrics remain reproducible."""

import json
from pathlib import Path

from app.ingestion import ingest_text


def test_versioned_dataset_has_stable_chunk_ids_scopes_and_required_case_types() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "evals" / "rag_eval_v1.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["version"] == "rag-eval-v1"
    assert dataset["embedding_profile"].endswith("dimensions=384")

    corpus = {entry["ref"]: entry for entry in dataset["corpus"]}
    for entry in dataset["corpus"]:
        result = ingest_text(
            entry["text"],
            source_type=entry["source_type"],
            title=entry["title"],
            filename=entry.get("filename"),
            source_url=entry.get("source_url"),
        )
        assert result.document.id == entry["document_id"]
        assert [chunk.id for chunk in result.chunks] == [entry["chunk_id"]]

    cases = dataset["cases"]
    categories = {case["category"] for case in cases}
    assert {
        "exact_identifier",
        "numeric_specification",
        "paraphrase",
        "unsupported",
        "source_isolation",
    }.issubset(categories)
    for case in cases:
        assert case["question"].strip()
        assert isinstance(case["scope"], dict) and case["scope"]
        assert isinstance(case["reference_facts"], list)
        assert case["expected_citations"] == case["relevant_chunk_ids"]
        assert all(
            any(chunk_id == entry["chunk_id"] for entry in corpus.values())
            for chunk_id in case["relevant_chunk_ids"]
        )
