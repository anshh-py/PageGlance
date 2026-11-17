"""Tests for Neon-compatible database engine configuration."""

import pytest

from app.db import DatabaseConfigurationError, create_database_engine, get_database_url


@pytest.fixture(autouse=True)
def _env_isolated(monkeypatch):
    """Isolate DATABASE_* env vars used by create_database_engine."""
    for key in (
        "DATABASE_URL",
        "DATABASE_SSLMODE",
        "DATABASE_POOL_SIZE",
        "DATABASE_MAX_OVERFLOW",
        "DATABASE_POOL_RECYCLE",
    ):
        monkeypatch.delenv(key, raising=False)


def _capture_engine_kwargs(monkeypatch, url, **env):
    """Monkeypatch app.db.create_engine to capture kwargs (independent of pool internals)."""
    import app.db as db_module

    captured = {}

    def fake_create_engine(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise RuntimeError("stop-before-real-engine")

    monkeypatch.setattr(db_module, "create_engine", fake_create_engine)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="stop-before-real-engine"):
        create_database_engine(url)
    return captured


def test_get_database_url_requires_config(monkeypatch):
    # Empty string means "unset"; .env reload cannot reintroduce it (override=False).
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(DatabaseConfigurationError):
        get_database_url()


def test_engine_sslmode_passed_as_connect_args(monkeypatch):
    captured = _capture_engine_kwargs(
        monkeypatch, "postgresql://u:p@host/db", DATABASE_SSLMODE="require"
    )
    assert captured["kwargs"].get("connect_args") == {"sslmode": "require"}


def test_engine_no_sslmode_by_default(monkeypatch):
    captured = _capture_engine_kwargs(monkeypatch, "postgresql://u:p@host/db")
    assert captured["kwargs"].get("connect_args") in (None, {})


def test_engine_pool_env_tuning(monkeypatch):
    captured = _capture_engine_kwargs(
        monkeypatch,
        "postgresql://u:p@host/db",
        DATABASE_POOL_SIZE="2",
        DATABASE_MAX_OVERFLOW="4",
        DATABASE_POOL_RECYCLE="60",
    )
    assert captured["kwargs"]["pool_size"] == 2
    assert captured["kwargs"]["max_overflow"] == 4
    assert captured["kwargs"]["pool_recycle"] == 60


def test_engine_default_pool_when_unset(monkeypatch):
    captured = _capture_engine_kwargs(monkeypatch, "postgresql://u:p@host/db")
    assert captured["kwargs"]["pool_size"] == 5
    assert captured["kwargs"]["max_overflow"] == 10
    assert captured["kwargs"]["pool_recycle"] == 300


def test_engine_invalid_env_falls_back_to_defaults(monkeypatch):
    captured = _capture_engine_kwargs(
        monkeypatch,
        "postgresql://u:p@host/db",
        DATABASE_POOL_SIZE="not-a-number",
        DATABASE_MAX_OVERFLOW="also-bad",
    )
    assert captured["kwargs"]["pool_size"] == 5
    assert captured["kwargs"]["max_overflow"] == 10


def test_engine_pool_pre_ping_always_on(monkeypatch):
    captured = _capture_engine_kwargs(monkeypatch, "postgresql://u:p@host/db")
    assert captured["kwargs"]["pool_pre_ping"] is True


def test_engine_url_credentials_never_in_renderable_kwargs(monkeypatch):
    captured = _capture_engine_kwargs(
        monkeypatch, "postgresql://secrettuser:supersecret@host/db", DATABASE_SSLMODE="require"
    )
    # The URL is passed positionally; credentials live only in the URL object, never in kwargs.
    url_arg = captured["args"][0]
    assert "supersecret" not in repr(captured["kwargs"])
    assert url_arg.startswith("postgresql://")


def test_database_url_placeholder_not_required_for_explicit_engine(monkeypatch):
    captured = _capture_engine_kwargs(monkeypatch, "postgresql://u:p@host/db")
    assert captured["args"][0] == "postgresql://u:p@host/db"
