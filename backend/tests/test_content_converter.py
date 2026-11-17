"""Coverage for the universal-content compatibility adapter (converters.py)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import main, page_indexing
from app.contracts import ContentSource, ExtractedDocument
from app.converters import ingest_extracted_document
from app.models import Base, DocumentRecord
from app.repository import DocumentRepository

client = TestClient(main.app)


class DeterministicEmbeddingService:
    """Stable vectors; no hosted provider or model weights are touched."""

    calls = 0
    model_version = "pageglance-test:semantic-fixture:dimensions=3"
    dimensions = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        type(self).calls += 1
        return [[1.0, 0.5, 0.0] for _ in texts]


@pytest.fixture
def sqlite_session(monkeypatch) -> Iterator[Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    # Recording spies around the *mocked* providers. They only fire when the
    # converter resolves these at call time from the module (never via an
    # import-time capture). If the production factories were used instead, the
    # document would not land in this SQLite session.
    provider_calls = {"session_factory": 0, "embedding_service": 0}
    embedding_service = DeterministicEmbeddingService()

    def rec_session_factory():
        provider_calls["session_factory"] += 1
        return session_factory

    def rec_embedding_service():
        provider_calls["embedding_service"] += 1
        return embedding_service

    monkeypatch.setattr(page_indexing, "get_ingestion_session_factory", rec_session_factory)
    monkeypatch.setattr(
        page_indexing, "get_ingestion_embedding_service", rec_embedding_service
    )
    with session_factory() as session:
        session.provider_calls = provider_calls
        yield session
    engine.dispose()


def _extracted(
    *,
    source_type: str,
    url: str | None = None,
    filename: str | None = None,
    mime: str | None = None,
    text: str = "body text",
    blocks=(),
    status: str = "complete",
    warnings=(),
) -> ExtractedDocument:
    source = ContentSource(
        source_type=source_type,
        source_id=f"src:{source_type}",
        original_url=url,
        filename=filename,
        mime_type=mime,
        metadata={},
    )
    return ExtractedDocument(
        title="Title",
        normalized_text=text,
        source=source,
        structured_blocks=list(blocks),
        extraction_status=status,
        extraction_warnings=list(warnings),
    )


def test_conversion_preserves_source_metadata_and_block_refs(sqlite_session) -> None:
    doc = _extracted(
        source_type="webpage",
        url="https://example.com/page",
        filename="page.html",
        mime="text/html",
        text="Intro paragraph. Second part.",
        warnings=["missing table of contents"],
        blocks=[
            {
                "block_type": "paragraph",
                "text": "Intro paragraph",
                "start_char": 0,
                "end_char": 14,
                "page": 1,
                "section": "s1",
                "timestamp_sec": 12.5,
            }
        ],
    )
    result = ingest_extracted_document(doc)
    assert result.indexed is True
    assert result.chunk_count >= 1

    restored = DocumentRepository(sqlite_session).get_by_id(result.document_id)
    assert restored is not None
    assert restored.document.source_type == "webpage"
    assert restored.document.title == "Title"
    assert restored.document.source_url == "https://example.com/page"
    assert restored.document.filename == "page.html"
    assert restored.document.metadata.get("mime_type") == "text/html"
    assert restored.document.metadata.get("source_id") == "src:webpage"
    assert restored.document.metadata.get("extraction_status") == "complete"
    assert restored.document.metadata["extraction_warnings"] == ["missing table of contents"]
    assert any("structured_blocks" in chunk.metadata for chunk in restored.chunks)
    assert any("page_numbers" in chunk.metadata for chunk in restored.chunks)

    # The mocked (test) providers are the ones used, so the production database
    # and real embedding model were never invoked.
    assert sqlite_session.provider_calls["session_factory"] >= 1
    assert sqlite_session.provider_calls["embedding_service"] >= 1
    assert DeterministicEmbeddingService.calls >= 1


def test_webpage_url_identity_preserved(sqlite_session) -> None:
    first = _extracted(
        source_type="webpage", url="https://example.com/doc", text="Same body"
    )
    second = _extracted(
        source_type="webpage", url="https://example.com/doc", text="Different body"
    )
    r1 = ingest_extracted_document(first)
    r2 = ingest_extracted_document(second)
    assert r1.indexed and r2.indexed
    assert r1.document_id == r2.document_id
    assert sqlite_session.scalar(select(func.count()).select_from(DocumentRecord)) == 1


def test_non_webpage_content_deduplicated(sqlite_session) -> None:
    r1 = ingest_extracted_document(
        _extracted(source_type="text", filename="a.txt", text="shared content")
    )
    r2 = ingest_extracted_document(
        _extracted(source_type="text", filename="b.txt", text="shared content")
    )
    assert r1.indexed and r2.indexed
    assert r1.document_id == r2.document_id
    assert sqlite_session.scalar(select(func.count()).select_from(DocumentRecord)) == 1


def test_failed_extraction_is_not_indexed(sqlite_session) -> None:
    doc = _extracted(source_type="text", text="some text", status="failed")
    result = ingest_extracted_document(doc)
    assert result.indexed is False
    assert result.fallback_code == "extraction_failed"
    assert sqlite_session.scalar(select(func.count()).select_from(DocumentRecord)) == 0


def test_empty_extraction_is_not_indexed(sqlite_session) -> None:
    result = ingest_extracted_document(_extracted(source_type="text", text=""))
    assert result.indexed is False
    assert result.fallback_code == "empty_page_text"
    assert sqlite_session.scalar(select(func.count()).select_from(DocumentRecord)) == 0


def test_ingest_url_response_contract_unaffected(monkeypatch) -> None:
    from app.page_indexing import PageIndexResult

    class FakeResponse:
        url = "https://example.com/article"
        status_code = 200
        content = b"<html><title>Example</title><body><p>Hello</p></body></html>"

    async def fake_fetch(url, **kw):
        return FakeResponse()

    monkeypatch.setattr(main, "fetch_url", fake_fetch)
    monkeypatch.setattr(
        main, "index_page_context", lambda page: PageIndexResult(True, "doc-url-1", 1)
    )
    response = client.post("/ingest/url", json={"url": "https://example.com/article"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "document_id",
        "chunk_count",
        "title",
        "source_url",
        "indexed",
        "fallback_code",
    }
    assert body["source_url"] == "https://example.com/article"


def test_text_upload_ingestion_path_unaffected(sqlite_session) -> None:
    from app.document_ingestion import ingest_uploaded_file

    uploaded = ingest_uploaded_file("notes.txt", b"A short text file body.")
    assert uploaded.document.source_type == "text"
    assert uploaded.chunk_count >= 1
