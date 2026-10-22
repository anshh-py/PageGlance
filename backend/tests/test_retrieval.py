"""Vector retrieval tests use mocked sessions and PostgreSQL SQL compilation."""

from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from app.embeddings import EmbeddingProviderError
from app.retrieval import (
    RetrievalConfigurationError,
    RetrievalDatabaseError,
    RetrievalEmbeddingError,
    RetrievalSettings,
    RetrievedChunk,
    VectorRetriever,
    build_lexical_query,
    build_similarity_query,
    fuse_rankings,
)
from app.retrieval_evaluation import aggregate_metrics, rank_metrics


class FakeEmbeddingService:
    model_version = "test:model-v1:dimensions=3"
    dimensions = 3

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.failure is not None:
            raise self.failure
        return [[0.1, 0.2, 0.3]]


def _row(*, document_id: str = "doc-1"):
    chunk = SimpleNamespace(
        id=f"{document_id}:chunk:0",
        document_id=document_id,
        index=0,
        text="A relevant paragraph.",
        start_char=0,
        end_char=22,
        chunk_metadata={"section": "intro"},
    )
    document = SimpleNamespace(
        id=document_id,
        title="Example",
        source_type="webpage",
        source_url="https://example.test/page",
        filename=None,
        document_metadata={"language": "en"},
    )
    return chunk, document, 0.87


def test_retrieval_embeds_query_and_returns_ranked_citation_metadata() -> None:
    session = SimpleNamespace(execute=lambda _statement: SimpleNamespace(all=lambda: [_row()]))
    embeddings = FakeEmbeddingService()
    retriever = VectorRetriever(
        session, embeddings, RetrievalSettings(default_top_k=4, max_top_k=20)
    )

    results = retriever.retrieve("  what is relevant?  ")

    assert embeddings.calls == [["what is relevant?"]]
    assert len(results) == 1
    assert results[0].model_dump() == {
        "chunk_id": "doc-1:chunk:0",
        "document_id": "doc-1",
        "chunk_index": 0,
        "text": "A relevant paragraph.",
        "start_char": 0,
        "end_char": 22,
        "title": "Example",
        "source_type": "webpage",
        "source_url": "https://example.test/page",
        "filename": None,
        "document_metadata": {"language": "en"},
        "chunk_metadata": {"section": "intro"},
        "similarity": 0.87,
    }


def test_empty_query_and_no_matches_return_empty_without_extra_work() -> None:
    execution = SimpleNamespace(all=lambda: [])
    session = SimpleNamespace(execute=lambda _statement: execution)
    embeddings = FakeEmbeddingService()
    retriever = VectorRetriever(session, embeddings)

    assert retriever.retrieve(" \n ") == []
    assert embeddings.calls == []
    assert retriever.retrieve("query") == []


def test_hybrid_query_without_content_terms_skips_lexical_search() -> None:
    statements = []
    session = SimpleNamespace(
        execute=lambda statement: statements.append(statement) or SimpleNamespace(all=lambda: [])
    )
    embeddings = FakeEmbeddingService()
    retriever = VectorRetriever(session, embeddings)

    assert retriever.retrieve("what is the") == []

    assert len(statements) == 1
    assert embeddings.calls == [["what is the"]]


def test_postgresql_query_uses_cosine_order_and_applies_all_scope_filters() -> None:
    statement = build_similarity_query(
        [0.1, 0.2, 0.3],
        model_version="test:model-v1:dimensions=3",
        dimensions=3,
        top_k=7,
        document_id="doc-locked",
        source_type="webpage",
        source_url="https://example.test/locked",
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert " <=> " in sql
    assert "ORDER BY (document_chunks.embedding <=> " in sql
    assert "document_chunks.document_id = " in sql
    assert "documents.source_type = " in sql
    assert "documents.source_url = " in sql
    assert "document_chunks.embedding_model = " in sql
    assert "document_chunks.embedding_dimensions = " in sql
    assert "LIMIT" in sql
    parameters = list(compiled.params.values())
    assert [0.1, 0.2, 0.3] in parameters
    assert all(
        value in parameters
        for value in (
            "test:model-v1:dimensions=3",
            3,
            "doc-locked",
            "webpage",
            "https://example.test/locked",
            7,
        )
    )


def test_local_query_uses_only_fixed_dimension_local_embedding_column() -> None:
    statement = build_similarity_query(
        [0.0] * 384,
        model_version="sentence_transformers:sentence-transformers/all-MiniLM-L6-v2:dimensions=384",
        dimensions=384,
        top_k=5,
        storage_profile="local",
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ORDER BY (document_chunks.local_embedding <=> " in sql
    assert "document_chunks.local_embedding_model = " in sql
    assert "document_chunks.local_embedding_dimensions = " in sql
    assert "document_chunks.local_embedding IS NOT NULL" in sql
    assert "document_chunks.embedding <=> " not in sql


def test_retrieval_never_relaxes_requested_scope() -> None:
    statements = []
    session = SimpleNamespace(
        execute=lambda statement: statements.append(statement) or SimpleNamespace(all=lambda: [])
    )
    retriever = VectorRetriever(session, FakeEmbeddingService())

    assert retriever.retrieve("query", document_id="only-this-doc") == []

    assert len(statements) == 2
    for statement in statements:
        sql = str(statement.compile(dialect=postgresql.dialect()))
        assert "document_chunks.document_id = " in sql


def test_retrieval_applies_multi_document_filter_in_both_paths() -> None:
    statements = []
    session = SimpleNamespace(
        execute=lambda statement: statements.append(statement) or SimpleNamespace(all=lambda: [])
    )
    retriever = VectorRetriever(session, FakeEmbeddingService())

    assert retriever.retrieve("model identifier", document_ids=["doc-a", "doc-b"]) == []

    assert len(statements) == 2
    for statement in statements:
        compiled = statement.compile(dialect=postgresql.dialect())
        assert "document_chunks.document_id IN" in str(compiled)
        assert ["doc-a", "doc-b"] in compiled.params.values()


@pytest.mark.parametrize("fails", [False, True])
def test_local_reranking_is_bounded_and_falls_back_to_rrf_on_failure(fails: bool) -> None:
    def row(chunk_id: str, text: str, similarity: float):
        chunk, document, _ = _row()
        chunk.id = chunk_id
        chunk.text = text
        chunk.end_char = len(text)
        return chunk, document, similarity

    rows = [row("first", "First candidate", 0.9), row("second", "Second candidate", 0.8)]
    session = SimpleNamespace(execute=lambda _statement: SimpleNamespace(all=lambda: rows))

    class FakeReranker:
        def rerank(self, query, candidates):
            assert query == "which candidate"
            assert len(candidates) == 2
            if fails:
                raise RuntimeError("private model failure")
            return list(reversed(candidates))

    retriever = VectorRetriever(
        session,
        FakeEmbeddingService(),
        RetrievalSettings(
            default_top_k=2,
            max_top_k=5,
            candidate_top_k=2,
            max_candidate_top_k=3,
            reranker_enabled=True,
            reranker_candidate_k=2,
        ),
        reranker=FakeReranker(),
    )

    results = retriever.retrieve("which candidate")

    assert [item.chunk_id for item in results] == (
        ["first", "second"] if fails else ["second", "first"]
    )


def test_embedding_and_database_failures_are_safe() -> None:
    provider_failure = EmbeddingProviderError(
        "embedding_timeout", "private provider response including page text"
    )
    retriever = VectorRetriever(SimpleNamespace(), FakeEmbeddingService(provider_failure))
    with pytest.raises(RetrievalEmbeddingError) as embedding_error:
        retriever.retrieve("query")
    assert embedding_error.value.code == "embedding_timeout"
    assert "private provider response" not in str(embedding_error.value)

    session = SimpleNamespace(
        execute=lambda _statement: (_ for _ in ()).throw(SQLAlchemyError("secret db url"))
    )
    retriever = VectorRetriever(session, FakeEmbeddingService())
    with pytest.raises(RetrievalDatabaseError) as database_error:
        retriever.retrieve("query")
    assert database_error.value.code == "retrieval_database_error"
    assert "secret db url" not in str(database_error.value)


def test_top_k_is_bounded() -> None:
    retriever = VectorRetriever(
        SimpleNamespace(), FakeEmbeddingService(), RetrievalSettings(default_top_k=3, max_top_k=4)
    )

    with pytest.raises(RetrievalConfigurationError, match="top_k must be between 1 and 4"):
        retriever.retrieve("query", top_k=5)


def test_lexical_query_uses_index_expression_and_applies_all_scope_filters() -> None:
    statement = build_lexical_query(
        'MiniLM "384 dimensions"',
        top_k=12,
        document_id="doc-locked",
        source_type="text",
        source_url="https://example.test/locked",
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "to_tsvector('simple'::regconfig, document_chunks.text)" in sql
    assert "to_tsquery('simple'::regconfig" in sql
    assert "ts_rank_cd" in sql
    assert "document_chunks.document_id = " in sql
    assert "documents.source_type = " in sql
    assert "documents.source_url = " in sql
    assert "LIMIT" in sql
    assert "minilm | 384 | dimensions" in compiled.params.values()


def test_rrf_fuses_duplicate_candidates_without_duplicate_results() -> None:
    def candidate(chunk_id: str, similarity: float) -> RetrievedChunk:
        chunk, document, _ = _row()
        chunk.id = chunk_id
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document.id,
            chunk_index=chunk.index,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            title=document.title,
            source_type=document.source_type,
            source_url=document.source_url,
            filename=document.filename,
            document_metadata=document.document_metadata,
            chunk_metadata=chunk.chunk_metadata,
            similarity=similarity,
        )

    vector = [candidate("a", 0.9), candidate("b", 0.8)]
    lexical = [candidate("b", 0.0), candidate("c", 0.0)]
    fused = fuse_rankings(vector, lexical, rrf_constant=1)

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert len({item.chunk_id for item in fused}) == len(fused)
    assert fused[0].similarity == 0.8


def test_retrieval_metrics_cover_recall_mrr_ndcg_and_unsupported_cases() -> None:
    metrics = rank_metrics({"relevant-a", "relevant-b"}, ["other", "relevant-a", "relevant-b"], k=3)
    assert metrics == {"recall": 1.0, "mrr": 0.5, "ndcg": pytest.approx(0.6934264036172708)}
    unsupported = rank_metrics(set(), ["one", "two"], k=5)
    assert unsupported == {"recall": None, "mrr": None, "ndcg": None}
    assert aggregate_metrics([metrics, unsupported])["queries"] == 1
