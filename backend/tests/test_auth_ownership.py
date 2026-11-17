"""Focused tests for installation authentication and document ownership isolation.

These exercise the real enforcement paths:
- token hashing / idempotent registration,
- the ``require_installation`` dependency (401 / 403 / 200),
- repository-level owner scoping (cross-owner and legacy ownerless access),
- owner-scoped deduplication,
- retrieval query owner filters,
- HTTP route rejection without a token.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import auth, main
from app.ingestion import ingest_text
from app.models import Base
from app.repository import DocumentRepository

client = TestClient(main.app)


@pytest.fixture
def sqlite_factory(monkeypatch):
    """A SQLite session factory used by auth and repository tests."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(auth, "get_session_factory", lambda: factory)
    return factory


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def test_token_hash_is_domain_separated_and_not_reversible():
    token = auth.generate_token()
    digest = auth.hash_token(token)
    assert digest.startswith("pga1:")
    assert digest != token
    assert len(token) >= 32
    assert auth.hash_token(token) == digest
    assert auth.hash_token(token + "x") != digest


def test_register_is_idempotent(sqlite_factory):
    token = auth.generate_token()
    first = auth.register_installation(token)
    second = auth.register_installation(token)
    assert first.id == second.id
    fresh = auth.register_installation(auth.generate_token())
    assert fresh.id != first.id


def test_register_rejects_short_token(sqlite_factory):
    with pytest.raises(Exception) as excinfo:
        auth.register_installation("short")
    assert excinfo.value.status_code == 422


# ---------------------------------------------------------------------------
# Dependency enforcement
# ---------------------------------------------------------------------------


def test_require_installation_rejects_missing_token(sqlite_factory):
    with pytest.raises(Exception) as excinfo:
        auth.require_installation(None)
    assert excinfo.value.status_code == 401


def test_require_installation_rejects_unknown_token(sqlite_factory):
    with pytest.raises(Exception) as excinfo:
        auth.require_installation("Bearer some-unknown-token")
    assert excinfo.value.status_code == 403


def test_require_installation_accepts_registered_token(sqlite_factory):
    token = auth.generate_token()
    installation = auth.register_installation(token)
    resolved = auth.require_installation(f"Bearer {token}")
    assert resolved.id == installation.id


def test_stats_route_rejects_missing_token():
    main.app.dependency_overrides.pop(main.require_installation, None)
    try:
        response = client.get("/documents/00000000-0000-0000-0000-000000000000/stats")
        assert response.status_code == 401
    finally:
        main.app.dependency_overrides[main.require_installation] = (
            lambda: auth.Installation(id="test-install", token_hash="pga1:test")
        )


# ---------------------------------------------------------------------------
# Repository ownership isolation
# ---------------------------------------------------------------------------


def _persist(session, owner_id, text, filename):
    result = ingest_text(text, source_type="text", title=filename, filename=filename, owner_id=owner_id)
    return DocumentRepository(session).persist(result, owner_id=owner_id)


def test_get_by_id_blocks_cross_owner_access(sqlite_factory):
    with sqlite_factory() as session:
        doc_a = _persist(session, "owner-a", "alpha text", "a.txt")
        repo = DocumentRepository(session)
        assert repo.get_by_id(doc_a.id, owner_id="owner-a") is not None
        assert repo.get_by_id(doc_a.id, owner_id="owner-b") is None
        # Ownerless (legacy) documents are unreachable through owner scoped calls.
        legacy = _persist(session, None, "legacy text", "legacy.txt")
        assert repo.get_by_id(legacy.id, owner_id="owner-a") is None
        assert repo.get_by_id(legacy.id, owner_id="owner-b") is None
        assert repo.get_by_id(legacy.id) is not None


def test_dedup_is_owner_scoped(sqlite_factory):
    with sqlite_factory() as session:
        repo = DocumentRepository(session)
        first_a = _persist(session, "owner-a", "identical content", "x.txt")
        # Same owner re-ingesting identical content returns the same document.
        dup_a = _persist(session, "owner-a", "identical content", "x.txt")
        assert dup_a.id == first_a.id
        # A different owner gets its own separate document.
        first_b = _persist(session, "owner-b", "identical content", "x.txt")
        assert first_b.id != first_a.id
        assert repo.get_by_id(first_a.id, owner_id="owner-a") is not None
        assert repo.get_by_id(first_a.id, owner_id="owner-b") is None
        assert repo.get_by_id(first_b.id, owner_id="owner-b") is not None


def test_webpage_identity_is_owner_scoped(sqlite_factory):
    with sqlite_factory() as session:
        page_a = ingest_text("page text", source_type="webpage", source_url="https://x.com/", owner_id="owner-a")
        page_b = ingest_text("page text", source_type="webpage", source_url="https://x.com/", owner_id="owner-b")
        assert page_a.document.id != page_b.document.id


# ---------------------------------------------------------------------------
# Retrieval owner filters
# ---------------------------------------------------------------------------


def test_similarity_query_includes_owner_filter():
    from app.retrieval import build_similarity_query

    statement = build_similarity_query(
        [0.0] * 384,
        model_version="m",
        dimensions=384,
        top_k=5,
        document_id="doc-1",
        owner_id="owner-a",
    )
    assert "owner_id" in str(statement.compile())


def test_lexical_query_includes_owner_filter():
    from app.retrieval import build_lexical_query

    statement = build_lexical_query(
        "relevant terms",
        top_k=5,
        source_url="https://x.com/",
        owner_id="owner-a",
    )
    assert "owner_id" in str(statement.compile())
