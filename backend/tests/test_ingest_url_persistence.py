"""End-to-end persistence for /ingest/url without a live PostgreSQL server.

Runs the real URL-ingestion pipeline (fetch mocked, embedding deterministic) against
an in-memory SQLite database so persistence, deduplication, and empty-extraction
reporting are verified through the actual code path used by the endpoint.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import main, page_indexing
from app.models import Base, DocumentRecord
from app.repository import DocumentRepository

client = TestClient(main.app)


class DeterministicEmbeddingService:
    """Stable vectors; no hosted provider or model weights are touched."""

    model_version = "pageglance-test:semantic-fixture:dimensions=3"
    dimensions = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.5, 0.0] for _ in texts]


class FakeResponse:
    url = "https://example.com/article#section"
    status_code = 200
    content = b"<html><title>Example</title><body><p>Hello world</p></body></html>"


@pytest.fixture
def sqlite_session(monkeypatch) -> Iterator[Session]:
    # StaticPool + check_same_thread=False let the app (which runs in the
    # TestClient's event-loop thread) reuse this isolated in-memory database.
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    # Point the real ingestion session factory and embedding service at the
    # isolated SQLite engine so the route persists through the real pipeline.
    monkeypatch.setattr(page_indexing, "get_ingestion_session_factory", lambda: session_factory)
    monkeypatch.setattr(
        page_indexing,
        "get_ingestion_embedding_service",
        lambda: DeterministicEmbeddingService(),
    )

    async def fake_fetch(url, **kw):
        return FakeResponse()

    monkeypatch.setattr(main, "fetch_url", fake_fetch)
    with session_factory() as session:
        yield session
    engine.dispose()


def test_ingest_url_persists_via_real_pipeline(sqlite_session: Session) -> None:
    response = client.post("/ingest/url", json={"url": "https://example.com/article#section"})

    assert response.status_code == 200
    body = response.json()
    assert body["indexed"] is True
    assert body["title"] == "Example"
    assert body["chunk_count"] >= 1

    restored = DocumentRepository(sqlite_session).get_by_id(body["document_id"])
    assert restored is not None
    assert restored.document.source_url == "https://example.com/article#section"
    assert restored.document.source_type == "webpage"
    assert len(restored.chunks) == body["chunk_count"]
    assert "Hello world" in restored.document.text


def test_ingest_url_same_url_deduplicates_and_returns_same_document(
    sqlite_session: Session,
) -> None:
    first = client.post("/ingest/url", json={"url": "https://example.com/article#section"})
    second = client.post("/ingest/url", json={"url": "https://example.com/article#section"})

    assert first.status_code == second.status_code == 200
    assert first.json()["document_id"] == second.json()["document_id"]
    assert (
        sqlite_session.scalar(select(func.count()).select_from(DocumentRecord)) == 1
    )


def test_ingest_url_different_url_same_content_stays_separate(sqlite_session: Session) -> None:
    a = client.post("/ingest/url", json={"url": "https://example.com/a"})
    b = client.post("/ingest/url", json={"url": "https://example.com/b"})

    assert a.json()["document_id"] != b.json()["document_id"]
    assert (
        sqlite_session.scalar(select(func.count()).select_from(DocumentRecord)) == 2
    )


def test_ingest_url_empty_body_text_is_reported_not_indexed(sqlite_session: Session) -> None:
    class EmptyResponse:
        url = "https://example.com/empty"
        status_code = 200
        content = b"<html><head></head><body></body></html>"

    import app.main as main_module

    async def fake_fetch(url, **kw):
        return EmptyResponse()

    original = main_module.fetch_url
    main_module.fetch_url = fake_fetch
    try:
        response = client.post("/ingest/url", json={"url": "https://example.com/empty"})
    finally:
        main_module.fetch_url = original

    assert response.status_code == 200
    body = response.json()
    assert body["indexed"] is False
    assert body["fallback_code"] in {"empty_page_text", "page_index_unavailable"}
