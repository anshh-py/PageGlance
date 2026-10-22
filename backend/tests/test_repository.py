"""Isolated SQLite tests for repository persistence and database configuration."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app import db
from app.chunking import ChunkingConfig
from app.db import DatabaseConfigurationError
from app.ingestion import ingest_text
from app.models import Base, DocumentChunkRecord, DocumentRecord
from app.repository import DocumentRepository


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as database_session:
        yield database_session
    engine.dispose()


def test_database_url_requires_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "load_environment", lambda: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.get_database_url.cache_clear()

    with pytest.raises(DatabaseConfigurationError, match="DATABASE_URL is not configured"):
        db.get_database_url()


def test_database_environment_file_path_is_root_anchored(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def capture_dotenv(*, dotenv_path: Path, override: bool) -> bool:
        captured["path"] = dotenv_path
        captured["override"] = override
        return True

    monkeypatch.setattr(db, "load_dotenv", capture_dotenv)
    monkeypatch.chdir(Path(__file__).resolve().parent)
    db.load_environment()

    expected = Path(__file__).resolve().parents[2] / ".env"
    assert captured == {"path": expected, "override": False}


def test_database_url_is_read_from_environment_without_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_url = "postgresql+psycopg://user:private-password@localhost/pageglance"
    monkeypatch.setattr(db, "load_environment", lambda: None)
    monkeypatch.setenv("DATABASE_URL", configured_url)
    db.get_database_url.cache_clear()

    assert db.get_database_url() == configured_url
    db.get_database_url.cache_clear()


def test_engine_and_session_factory_accept_isolated_database() -> None:
    engine = db.create_database_engine("sqlite+pysqlite:///:memory:")
    factory = db.create_session_factory(engine)

    with factory() as session:
        assert session.bind is engine
    engine.dispose()


def test_persist_and_retrieve_document_chunks_in_order(session: Session) -> None:
    result = ingest_text(
        "First paragraph has words.\n\nSecond paragraph has more words.",
        source_type="text",
        title="Test document",
        metadata={"origin": "test", "tags": ["a"]},
        chunking=ChunkingConfig(chunk_size=24, overlap=4),
    )
    repository = DocumentRepository(session)

    saved = repository.persist(result)
    restored = repository.get_by_id(saved.id)

    assert restored is not None
    assert restored.document.id == result.document.id
    assert restored.document.text == result.document.text
    assert restored.document.content_hash == result.document.content_hash
    assert restored.document.metadata == {"origin": "test", "tags": ["a"]}
    assert [chunk.index for chunk in restored.chunks] == list(range(len(result.chunks)))
    assert [chunk.text for chunk in restored.chunks] == [chunk.text for chunk in result.chunks]
    assert all(chunk.metadata == restored.document.metadata for chunk in restored.chunks)
    assert all(
        restored.document.text[chunk.start_char : chunk.end_char] == chunk.text
        for chunk in restored.chunks
    )


def test_duplicate_normalized_content_reuses_existing_document(session: Session) -> None:
    repository = DocumentRepository(session)
    first = ingest_text(
        "Shared content\r\nwith normalization.",
        source_type="text",
        filename="first.txt",
    )
    duplicate = ingest_text(
        "Shared content\nwith normalization.",
        source_type="markdown",
        filename="second.md",
    )

    original_saved = repository.persist(first)
    duplicate_saved = repository.persist(duplicate)

    assert original_saved.id == duplicate_saved.id == first.document.id
    assert session.scalar(select(func.count()).select_from(DocumentRecord)) == 1
    assert session.scalar(select(func.count()).select_from(DocumentChunkRecord)) == len(
        first.chunks
    )


def test_identical_webpage_content_remains_available_under_each_source_url(
    session: Session,
) -> None:
    repository = DocumentRepository(session)
    first = ingest_text(
        "Shared article text.", source_type="webpage", source_url="https://one.test/article"
    )
    second = ingest_text(
        "Shared article text.", source_type="webpage", source_url="https://two.test/article"
    )

    saved_first = repository.persist(first)
    saved_second = repository.persist(second)

    assert saved_first.id != saved_second.id
    assert repository.get_by_id(saved_first.id).document.source_url == "https://one.test/article"
    assert repository.get_by_id(saved_second.id).document.source_url == "https://two.test/article"
    assert session.scalar(select(func.count()).select_from(DocumentRecord)) == 2


def test_reingest_at_same_source_updates_document_and_replaces_chunks(session: Session) -> None:
    repository = DocumentRepository(session)
    first = ingest_text(
        "Old body with enough words to make chunks.",
        source_type="webpage",
        source_url="https://example.test/article",
        chunking=ChunkingConfig(chunk_size=10, overlap=2),
    )
    updated = ingest_text(
        "New body contains revised material and additional text.",
        source_type="webpage",
        source_url="https://example.test/article",
        chunking=ChunkingConfig(chunk_size=12, overlap=3),
    )

    repository.persist(first)
    repository.persist(updated)
    restored = repository.get_by_id(first.document.id)

    assert restored is not None
    assert restored.document.text == updated.document.text
    assert [chunk.text for chunk in restored.chunks] == [chunk.text for chunk in updated.chunks]
    assert session.scalar(select(func.count()).select_from(DocumentRecord)) == 1
    assert session.scalar(select(func.count()).select_from(DocumentChunkRecord)) == len(
        updated.chunks
    )


def test_retrieval_returns_none_for_unknown_document(session: Session) -> None:
    assert DocumentRepository(session).get_by_id("missing") is None


def test_embedding_columns_are_nullable_for_future_population() -> None:
    assert DocumentRecord.__table__.c.embedding.nullable
    assert DocumentChunkRecord.__table__.c.embedding.nullable
    assert DocumentRecord.__table__.c.embedding.type.dim is None
    assert DocumentChunkRecord.__table__.c.embedding.type.dim is None
