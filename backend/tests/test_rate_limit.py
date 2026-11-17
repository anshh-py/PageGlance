"""Focused tests for M9.3 rate limiting.

These tests exercise the real HTTP paths with installation-authenticated clients
against a SQLite-backed app (mirroring test_auth_routes.py). The rate limiter
is configured with tiny limits so requests within/exceeding the limit are
observable without flooding.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import auth, main, page_indexing, ratelimit
from app.models import Base


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
    limiter = ratelimit.RateLimiter(session_factory=lambda: factory)
    monkeypatch.setattr(main, "rate_limiter", limiter)
    # Keep installation quotas tiny (2) while leaving the shared IP backstop high
    # so per-install independent quotas are observable in isolation.
    monkeypatch.setattr(
        ratelimit,
        "get_action_limit",
        lambda action: 1000 if action == "ip" else 2,
    )
    return TestClient(main.app)


def _register(client, token="x" * 40):
    resp = client.post("/auth/register", json={"token": token})
    assert resp.status_code == 200, resp.text
    return resp.json()["installation_id"]


def _bearer(token="x" * 40):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def rate_client(monkeypatch):
    factory = _fresh_app_and_factory()
    owner = _register(_install(monkeypatch, factory))
    return _install(monkeypatch, factory), owner


def test_requests_within_limit_allowed(rate_client):
    client, owner = rate_client
    for _ in range(2):
        r = client.get(f"/documents/{owner}/stats", headers=_bearer())
        # 404 (unknown doc) is fine; the point is it is not 429.
        assert r.status_code != 429


def test_requests_exceeding_limit_rejected(rate_client):
    client, owner = rate_client
    for _ in range(2):
        client.get(f"/documents/{owner}/stats", headers=_bearer())
    r = client.get(f"/documents/{owner}/stats", headers=_bearer())
    assert r.status_code == 429
    assert r.headers.get("retry-after") is not None


def test_separate_installations_have_independent_quotas(monkeypatch):
    factory = _fresh_app_and_factory()
    client = _install(monkeypatch, factory)
    token_a = "a" * 40
    token_b = "b" * 40
    _register(client, token_a)
    _register(client, token_b)
    for _ in range(2):
        client.get("/documents/00000000-0000-0000-0000-000000000000/stats", headers=_bearer(token_a))
    # A is over its quota; B is still within its own.
    assert client.get(
        "/documents/00000000-0000-0000-0000-000000000000/stats", headers=_bearer(token_a)
    ).status_code == 429
    assert client.get(
        "/documents/00000000-0000-0000-0000-000000000000/stats", headers=_bearer(token_b)
    ).status_code != 429


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
def test_streaming_endpoint_protected(monkeypatch, path):
    factory = _fresh_app_and_factory()
    client = _install(monkeypatch, factory)
    token = "x" * 40
    _register(client, token)
    body = {"question": "q", "page": {"title": "t", "url": "https://x.com", "text": "p"}}
    # Avoid the live LLM provider and the real embedding pipeline for the
    # allowed (in-limit) requests: we are testing rate-limit enforcement, not
    # generation or retrieval. The 429 rejection is raised by
    # _enforce_rate_limit before any provider call, and mocking the answer
    # prep keeps the allowed requests fast and hermetic.
    from types import SimpleNamespace

    class _FakeAnswerService:
        def answer(self, *_a, **_k):
            return "ok"

    async def _fake_stream(*_args, **_kwargs):
        yield "ok"

    def _fake_prepare(*_args, **_kwargs):
        return SimpleNamespace(
            page_context={"title": "t", "url": "", "text": "p"},
            retrieval_used=False,
            retrieved_sources=None,
        )

    monkeypatch.setattr(main, "get_answer_service", lambda: _FakeAnswerService())
    monkeypatch.setattr(main, "stream_answer", _fake_stream)
    monkeypatch.setattr(main, "_prepare_answer_context", _fake_prepare)
    monkeypatch.setattr(main, "_document_diagnostic_answer", lambda *a, **k: None)
    for _ in range(2):
        client.post(path, json=body, headers=_bearer(token))
    r = client.post(path, json=body, headers=_bearer(token))
    assert r.status_code == 429


def test_rate_limit_store_unavailable_fails_closed(monkeypatch, rate_client):
    client, owner = rate_client

    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(main.rate_limiter, "_session_factory", boom)
    r = client.get(f"/documents/{owner}/stats", headers=_bearer())
    # The limiter must never fail open: when the counter store is unreachable we
    # refuse the request (503) rather than allow an unlimited expensive one.
    assert r.status_code == 503
