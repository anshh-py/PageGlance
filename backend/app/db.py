"""Database configuration and SQLAlchemy engine/session factories."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    """Raised when the persistence layer lacks its required database URL."""


def load_environment() -> None:
    """Load the repository-root .env independent of the current directory."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


@lru_cache(maxsize=1)
def get_database_url() -> str:
    """Return the configured SQLAlchemy URL without logging its credentials."""
    load_environment()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured.")
    return database_url


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create an engine lazily so the API can run without a database in this phase.

    Connection pooling and SSL are configurable through environment variables so the
    same code works against a local PostgreSQL instance and a Neon cloud database:

    - ``DATABASE_POOL_SIZE``: pool size (default 5, SQLAlchemy default).
    - ``DATABASE_MAX_OVERFLOW``: max overflow connections (default 10).
    - ``DATABASE_POOL_RECYCLE``: connection recycle seconds (default 300).
    - ``DATABASE_SSLMODE``: when set (e.g. ``require``), passed as ``connect_args``
      to the driver. Neon requires SSL; local PostgreSQL typically does not, so this
      stays unset by default.
    """
    url = database_url or get_database_url()
    engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
    # Pool sizing and SSL are PostgreSQL-specific; SQLite (tests) and other dialects
    # reject these kwargs, so only apply them for PostgreSQL URLs.
    if url.startswith("postgresql"):
        connect_args: dict[str, object] = {}
        sslmode = os.getenv("DATABASE_SSLMODE", "").strip()
        if sslmode:
            connect_args["sslmode"] = sslmode
        try:
            engine_kwargs["pool_size"] = int(os.getenv("DATABASE_POOL_SIZE", "5"))
        except ValueError:
            engine_kwargs["pool_size"] = 5
        try:
            engine_kwargs["max_overflow"] = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        except ValueError:
            engine_kwargs["max_overflow"] = 10
        try:
            engine_kwargs["pool_recycle"] = int(os.getenv("DATABASE_POOL_RECYCLE", "300"))
        except ValueError:
            engine_kwargs["pool_recycle"] = 300
        if connect_args:
            engine_kwargs["connect_args"] = connect_args
    return create_engine(url, **engine_kwargs)


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Build a session factory for an injected or environment-configured engine."""
    database_engine = engine or create_database_engine()
    return sessionmaker(bind=database_engine, expire_on_commit=False)
