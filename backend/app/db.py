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
    """Create an engine lazily so the API can run without a database in this phase."""
    return create_engine(database_url or get_database_url(), pool_pre_ping=True)


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Build a session factory for an injected or environment-configured engine."""
    database_engine = engine or create_database_engine()
    return sessionmaker(bind=database_engine, expire_on_commit=False)
