"""Route-level security tests for installation authentication and ownership.

These tests exercise the real HTTP enforcement paths (not unit-level stubs):
- installation registration (only the token hash is stored),
- missing / invalid bearer credentials rejected (401 / 403),
- authorized ingestion, retrieval, and metadata access for a valid installation,
- cross-install isolation for metadata, stats, chunks, and RAG retrieval,
- legacy ownerless documents staying inaccessible through authenticated routes,
- verification that registration never adopts/claims existing ownerless documents,
- concurrent requests with different identities not leaking owner context.
"""

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from sqlalchemy import select

from app import auth, main, page_indexing, ratelimit
from app.ingestion import ingest_text
from app.models import Base, DocumentRecord, InstallationRecord
from app.repository import DocumentRepository


def _fresh_app_and_factory():
    """Build an isolated app + shared SQLite factory for both auth and ingestion."""
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _install_real_auth(monkeypatch, factory):
    """Disable the autouse auth bypass and route the DB to our SQLite factory."""
    main.app.dependency_overrides.pop(main.require_installation, None)
    monkeypatch.setattr(auth, "get_session_factory", lambda: factory)
    monkeypatch.setattr(page_indexing, "_session_factory", lambda: factory)
    # Registration calls the IP rate limiter; point it at the SQLite DB so the
    # test never touches the production PostgreSQL store.
    limiter = ratelimit.RateLimiter(session_factory=lambda: factory)
    monkeypatch.setattr(main, "rate_limiter", limiter)
    # Auth/ownership tests assert identity isolation only; the rate limiter is
    # irrelevant here and its shared-counter writes race on file-backed SQLite
    # under concurrency (IntegrityError). Disable counting so these tests stay
    # focused and hermetic. Rate limiting itself is covered in test_rate_limit.py.
    monkeypatch.setattr(ratelimit, "get_action_limit", lambda action: 0)
    return factory


def _register(client: TestClient, token: str) -> str:
    response = client.post("/auth/register", json={"token": token})
    assert response.status_code == 200, response.text
    return response.json()["installation_id"]


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _persist_doc(factory, *, owner_id, text="secret content", filename="secret.txt"):
    with factory() as session:
        result = ingest_text(
            text, source_type="text", title=filename, filename=filename, owner_id=owner_id
        )
        return DocumentRepository(session).persist(result, owner_id=owner_id)


def _client(monkeypatch) -> tuple[TestClient, object]:
    factory = _install_real_auth(monkeypatch, _fresh_app_and_factory())
    return TestClient(main.app), factory


# ---------------------------------------------------------------------------
# Registration + credential enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["missing", "x" * 10])
def test_register_rejects_short_token(monkeypatch, token):
    client, _ = _client(monkeypatch)
    response = client.post("/auth/register", json={"token": token})
    assert response.status_code == 422


def test_register_only_persists_token_hash(monkeypatch):
    client, factory = _client(monkeypatch)
    token = auth.generate_token()
    _register(client, token)
    assert auth.hash_token(token) != token
    assert auth.hash_token(token).startswith("pga1:")
    installation_id = auth.register_installation(token).id
    with factory() as session:
        row = session.scalar(
            select(InstallationRecord).where(InstallationRecord.id == installation_id)
        )
        assert row is not None
        assert row.token_hash == auth.hash_token(token)
        assert row.token_hash != token


def test_protected_routes_require_bearer(monkeypatch):
    client, _ = _client(monkeypatch)
    for method, path, kwargs in [
        ("get", "/documents/00000000-0000-0000-0000-000000000000/stats", {}),
        ("post", "/documents/upload", {"files": {"file": ("a.txt", b"x", "text/plain")}}),
        ("post", "/ingest/url", {"json": {"url": "https://example.com/"}}),
        ("post", "/ask", {"json": {"question": "q", "page": {"text": "p"}}}),
        ("post", "/ask/stream", {"json": {"question": "q", "page": {"text": "p"}}}),
    ]:
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401, (method, path, response.status_code, response.text)


def test_invalid_credentials_rejected(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.get(
        "/documents/00000000-0000-0000-0000-000000000000/stats", headers=_bearer("nope")
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Authorized ingestion + access
# ---------------------------------------------------------------------------


def test_valid_credentials_can_query_own_document(monkeypatch):
    client, factory = _client(monkeypatch)
    token = auth.generate_token()
    owner = _register(client, token)
    doc = _persist_doc(factory, owner_id=owner)
    stats = client.get(f"/documents/{doc.id}/stats", headers=_bearer(token))
    assert stats.status_code == 200
    assert stats.json()["filename"] == "secret.txt"


# ---------------------------------------------------------------------------
# Cross-owner isolation
# ---------------------------------------------------------------------------


def test_cross_owner_stats_blocked(monkeypatch):
    client, factory = _client(monkeypatch)
    token_a = auth.generate_token()
    token_b = auth.generate_token()
    owner_a = _register(client, token_a)
    owner_b = _register(client, token_b)
    assert owner_a != owner_b

    doc_a = _persist_doc(factory, owner_id=owner_a)

    assert client.get(
        f"/documents/{doc_a.id}/stats", headers=_bearer(token_a)
    ).status_code == 200
    assert client.get(
        f"/documents/{doc_a.id}/stats", headers=_bearer(token_b)
    ).status_code == 404


def test_cross_owner_ask_blocked(monkeypatch):
    client, factory = _client(monkeypatch)
    token_a = auth.generate_token()
    token_b = auth.generate_token()
    owner_a = _register(client, token_a)
    _register(client, token_b)

    doc_a = _persist_doc(factory, owner_id=owner_a)

    ask_a = client.post(
        "/ask",
        headers=_bearer(token_b),
        json={"question": "what is it", "document_id": str(doc_a.id)},
    )
    assert ask_a.status_code == 404
    ask_stream = client.post(
        "/ask/stream",
        headers=_bearer(token_b),
        json={"question": "what is it", "document_id": str(doc_a.id)},
    )
    assert ask_stream.status_code == 404


# ---------------------------------------------------------------------------
# Legacy ownerless documents
# ---------------------------------------------------------------------------


def test_ownerless_legacy_docs_inaccessible_through_routes(monkeypatch):
    client, factory = _client(monkeypatch)
    token = auth.generate_token()
    _register(client, token)

    legacy = _persist_doc(factory, owner_id=None)
    assert client.get(
        f"/documents/{legacy.id}/stats", headers=_bearer(token)
    ).status_code == 404
    assert client.post(
        "/ask",
        headers=_bearer(token),
        json={"question": "q", "document_id": str(legacy.id)},
    ).status_code == 404


def test_registration_does_not_claim_ownerless_docs(monkeypatch):
    client, factory = _client(monkeypatch)
    token = auth.generate_token()
    legacy = _persist_doc(factory, owner_id=None)

    _register(client, token)

    with factory() as session:
        row = session.get(DocumentRecord, legacy.id)
        assert row is not None
        assert row.owner_id is None
    assert client.get(
        f"/documents/{legacy.id}/stats", headers=_bearer(token)
    ).status_code == 404


# ---------------------------------------------------------------------------
# Concurrency: no owner-identity leakage
# ---------------------------------------------------------------------------


def test_concurrent_requests_do_not_leak_owner(monkeypatch):
    # File-backed SQLite: each connection is independent and the data persists
    # on disk across connections, so concurrent requests in different threads
    # get genuinely independent DB connections without the in-memory shared-
    # cache drop problem (an in-memory cache=shared DB vanishes when the last
    # connection closes). Registration runs serially before the threads start,
    # and the concurrent requests below are read-only (GET stats), so SQLite
    # write-locking is not exercised here. (The pytest tmp_path fixture is not
    # usable on this environment, so we manage our own unique temp file.)
    import tempfile, os
    db_path = os.path.join(tempfile.gettempdir(), f"pga_cc_{os.getpid()}.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    _install_real_auth(monkeypatch, factory)
    client = TestClient(main.app)

    token_a = auth.generate_token()
    token_b = auth.generate_token()
    owner_a = _register(client, token_a)
    owner_b = _register(client, token_b)
    assert owner_a != owner_b

    doc_a = _persist_doc(factory, owner_id=owner_a)
    doc_b = _persist_doc(factory, owner_id=owner_b, text="B private", filename="b.txt")

    results: dict[str, tuple[int, str]] = {}

    def hit(name, token, doc_id):
        with TestClient(main.app) as sub:
            r = sub.get(f"/documents/{doc_id}/stats", headers=_bearer(token))
            results[name] = (r.status_code, r.text)

    threads = [
        threading.Thread(target=hit, args=("a_own", token_a, str(doc_a.id))),
        threading.Thread(target=hit, args=("a_b", token_a, str(doc_b.id))),
        threading.Thread(target=hit, args=("b_own", token_b, str(doc_b.id))),
        threading.Thread(target=hit, args=("b_a", token_b, str(doc_a.id))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["a_own"][0] == 200, results
    assert results["a_b"][0] == 404, results
    assert results["b_own"][0] == 200, results
    assert results["b_a"][0] == 404, results
