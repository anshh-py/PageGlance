"""Focused tests for M9.6 owner-scoped document deletion."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import auth, main, page_indexing, ratelimit
from app.ingestion import ingest_text
from app.models import Base, DocumentChunkRecord, DocumentRecord
from app.repository import DocumentRepository


def _fresh_app_and_factory():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _install(monkeypatch, factory):
    main.app.dependency_overrides.pop(main.require_installation, None)
    monkeypatch.setattr(auth, "get_session_factory", lambda: factory)
    monkeypatch.setattr(page_indexing, "_session_factory", lambda: factory)
    # Registration calls the IP rate limiter; point it at the SQLite DB so the
    # test never touches the production PostgreSQL store.
    limiter = ratelimit.RateLimiter(session_factory=lambda: factory)
    monkeypatch.setattr(main, "rate_limiter", limiter)
    return TestClient(main.app)


def _register(client, token):
    resp = client.post("/auth/register", json={"token": token})
    assert resp.status_code == 200, resp.text
    return resp.json()["installation_id"]


def _persist(factory, *, owner_id, text="secret", filename="s.txt"):
    with factory() as session:
        result = ingest_text(text, source_type="text", title=filename, filename=filename, owner_id=owner_id)
        return DocumentRepository(session).persist(result, owner_id=owner_id)


def _deleted_cleanup(factory, doc_id):
    with factory() as session:
        doc = session.get(DocumentRecord, doc_id)
        chunks = session.scalars(
            select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == doc_id)
        ).all()
        return doc, chunks


def test_authorized_owner_deletion_removes_data(monkeypatch):
    factory = _fresh_app_and_factory()
    client = _install(monkeypatch, factory)
    token = "x" * 40
    owner = _register(client, token)
    doc = _persist(factory, owner_id=owner)
    r = client.delete(f"/documents/{doc.id}", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 204
    doc_row, chunks = _deleted_cleanup(factory, doc.id)
    assert doc_row is None
    assert chunks == []
    # Repeated deletion is safe and predictable.
    r2 = client.delete(f"/documents/{doc.id}", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 404


def test_another_installation_cannot_delete(monkeypatch):
    factory = _fresh_app_and_factory()
    client = _install(monkeypatch, factory)
    owner_token = "a" * 40
    attacker_token = "b" * 40
    owner = _register(client, owner_token)
    _register(client, attacker_token)
    doc = _persist(factory, owner_id=owner)
    r = client.delete(
        f"/documents/{doc.id}", headers={"Authorization": f"Bearer {attacker_token}"}
    )
    assert r.status_code == 404
    doc_row, chunks = _deleted_cleanup(factory, doc.id)
    assert doc_row is not None
    assert len(chunks) > 0


def test_missing_document_returns_404(monkeypatch):
    factory = _fresh_app_and_factory()
    client = _install(monkeypatch, factory)
    token = "x" * 40
    _register(client, token)
    r = client.delete(
        "/documents/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404


def test_missing_credentials_rejected(monkeypatch):
    factory = _fresh_app_and_factory()
    client = _install(monkeypatch, factory)
    r = client.delete("/documents/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 401
