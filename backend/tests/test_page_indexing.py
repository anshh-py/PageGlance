"""Automatic webpage indexing tests with a mock persistence boundary/provider."""

from collections.abc import Iterator
from contextlib import contextmanager

from app import page_indexing
from app.documents import Document
from app.embeddings import EmbeddingProviderError
from app.normalization import exact_text_hash


class FakeEmbeddingService:
    model_version = "test:page-model:dimensions=3"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeRepository:
    def __init__(self) -> None:
        self.vectors_by_hash: dict[str, list[float]] = {}
        self.documents_by_url: dict[str, Document] = {}

    def get_reusable_embeddings(
        self, result, *, embedding_model, embedding_dimensions, storage_profile="legacy"
    ):
        assert embedding_model == FakeEmbeddingService.model_version
        assert embedding_dimensions == 3
        assert storage_profile == "legacy"
        return {
            chunk.id: self.vectors_by_hash[exact_text_hash(chunk.text)]
            for chunk in result.chunks
            if exact_text_hash(chunk.text) in self.vectors_by_hash
        }

    def persist(
        self,
        result,
        *,
        embedding_vectors,
        embedding_model,
        embedding_dimensions,
        storage_profile="legacy",
    ):
        assert embedding_model == FakeEmbeddingService.model_version
        assert embedding_dimensions == 3
        assert storage_profile == "legacy"
        self.vectors_by_hash.update(
            {exact_text_hash(chunk.text): embedding_vectors[chunk.id] for chunk in result.chunks}
        )
        self.documents_by_url[result.document.source_url] = result.document
        return result.document


def test_index_page_reuses_vectors_and_preserves_each_url(monkeypatch) -> None:
    repository = FakeRepository()
    service = FakeEmbeddingService()

    @contextmanager
    def session_scope() -> Iterator[object]:
        yield object()

    monkeypatch.setattr(page_indexing, "_session_factory", lambda: lambda: session_scope())
    monkeypatch.setattr(page_indexing, "_embedding_service", lambda: service)
    monkeypatch.setattr(page_indexing, "DocumentRepository", lambda _session: repository)
    page = {"title": "Title", "url": "HTTPS://EXAMPLE.TEST:443/article", "text": "Page text."}

    first = page_indexing.index_page_context(page)
    repeated = page_indexing.index_page_context({**page, "url": "https://example.test/article"})
    other_url = page_indexing.index_page_context({**page, "url": "https://other.example/article"})
    changed = page_indexing.index_page_context({**page, "text": "Updated page text."})

    assert first.indexed and repeated.indexed and other_url.indexed and changed.indexed
    assert first.document_id == repeated.document_id == changed.document_id
    assert other_url.document_id != first.document_id
    assert len(service.calls) == 2
    assert set(repository.documents_by_url) == {
        "https://example.test/article",
        "https://other.example/article",
    }
    assert repository.documents_by_url["https://example.test/article"].text == "Updated page text."


def test_page_indexing_fails_safely_when_embedding_provider_fails(monkeypatch) -> None:
    class FailingEmbeddingService:
        model_version = FakeEmbeddingService.model_version
        dimensions = 3

        def embed_texts(self, _texts):
            raise EmbeddingProviderError("embedding_provider_error", "private provider detail")

    @contextmanager
    def session_scope() -> Iterator[object]:
        yield object()

    monkeypatch.setattr(page_indexing, "_session_factory", lambda: lambda: session_scope())
    monkeypatch.setattr(page_indexing, "_embedding_service", lambda: FailingEmbeddingService())
    monkeypatch.setattr(page_indexing, "DocumentRepository", lambda _session: FakeRepository())

    result = page_indexing.index_page_context(
        {"title": "Title", "url": "https://example.test", "text": "Page text."}
    )

    assert result.indexed is False
    assert result.document_id is None
    assert result.fallback_code == "embedding_provider_error"


def test_page_chunk_limit_skips_persistence(monkeypatch) -> None:
    monkeypatch.setattr(page_indexing, "MAX_PAGE_CHUNKS", 0)
    monkeypatch.setattr(
        page_indexing,
        "_session_factory",
        lambda: (_ for _ in ()).throw(AssertionError("database must not be opened")),
    )

    result = page_indexing.index_page_context(
        {"title": "Title", "url": "https://example.test", "text": "Page text."}
    )

    assert result.indexed is False
    assert result.fallback_code == "page_chunk_limit_exceeded"
