"""Opt-in integration tests for migrations and the PostgreSQL/pgvector pipeline."""

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, delete, inspect, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.chunking import ChunkingConfig
from app.db import get_database_url
from app.document_ingestion import ingest_uploaded_file
from app.embedding_pipeline import embed_and_persist
from app.ingestion import ingest_text
from app.models import DocumentChunkRecord, DocumentRecord
from app.repository import DocumentRepository
from app.retrieval import VectorRetriever

pytestmark = pytest.mark.integration
TEST_DATABASE_NAME = "pageglance_test"
TEST_EMBEDDING_MODEL = "pageglance-test:semantic-fixture:dimensions=3"
LOCAL_EMBEDDING_MODEL = (
    "sentence_transformers:sentence-transformers/all-MiniLM-L6-v2:dimensions=384"
)


class MockEmbeddingService:
    """Deterministic local vectors; no hosted provider is constructed or called."""

    model_version = TEST_EMBEDDING_MODEL
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for value in texts:
            text_value = value.casefold()
            if "postgresql" in text_value:
                vectors.append([1.0, 0.0, 0.0])
            elif "database" in text_value or "vector" in text_value:
                vectors.append([0.5, 0.5, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class MockLocalEmbeddingService:
    """Mock local model output for integration coverage without downloading weights."""

    model_version = LOCAL_EMBEDDING_MODEL
    dimensions = 384
    storage_profile = "local"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] + [0.0] * 383 for _ in texts]


def _integration_database_url() -> str:
    value = os.getenv("PAGEGLANCE_TEST_DATABASE_URL", "").strip()
    if not value:
        pytest.skip(
            "Set PAGEGLANCE_TEST_DATABASE_URL to run live PostgreSQL + pgvector tests "
            f"against a disposable local database named {TEST_DATABASE_NAME}."
        )
    try:
        url = make_url(value)
    except Exception:
        pytest.fail(
            "PAGEGLANCE_TEST_DATABASE_URL is invalid; value was not displayed.", pytrace=False
        )
    if (
        url.drivername != "postgresql+psycopg"
        or url.host not in {"localhost", "127.0.0.1", "::1"}
        or url.database != TEST_DATABASE_NAME
    ):
        pytest.fail(
            "Integration tests only accept postgresql+psycopg on localhost and the "
            f"dedicated {TEST_DATABASE_NAME} database. URL was not displayed.",
            pytrace=False,
        )
    return value


@pytest.fixture(scope="module")
def postgres_engine() -> Iterator[Engine]:
    """Migrate only the explicitly named test database and provide a local engine."""
    test_url = _integration_database_url()
    engine = create_engine(test_url, pool_pre_ping=True)

    # Check the server and pgvector installation before running the migration chain.
    try:
        with engine.connect() as connection:
            extension_available = connection.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')")
            )
    except Exception as error:
        engine.dispose()
        pytest.fail(
            "Could not connect to the configured local PostgreSQL test database "
            f"({type(error).__name__}); URL and driver details were not displayed.",
            pytrace=False,
        )
    if not extension_available:
        engine.dispose()
        pytest.skip(
            "The PostgreSQL server does not have pgvector installed. Install the server-side "
            "extension for this PostgreSQL version, then rerun the integration tests."
        )

    backend_directory = Path(__file__).resolve().parents[1]
    config = Config(str(backend_directory / "alembic.ini"))
    config.set_main_option("script_location", str(backend_directory / "migrations"))
    environment = pytest.MonkeyPatch()
    environment.setenv("DATABASE_URL", test_url)
    get_database_url.cache_clear()
    try:
        command.upgrade(config, "head")
    except Exception as error:
        engine.dispose()
        pytest.fail(
            "Alembic migration failed for the dedicated PostgreSQL integration database "
            f"({type(error).__name__}); connection details were not displayed.",
            pytrace=False,
        )
    finally:
        environment.undo()
        get_database_url.cache_clear()

    yield engine
    engine.dispose()


@pytest.fixture
def postgres_session(postgres_engine: Engine) -> Iterator[Session]:
    factory: sessionmaker[Session] = sessionmaker(bind=postgres_engine, expire_on_commit=False)
    with factory() as session:
        yield session


def test_postgresql_migrations_create_pgvector_schema(postgres_engine: Engine) -> None:
    with postgres_engine.connect() as connection:
        revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        extension_version = connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
    inspector = inspect(postgres_engine)

    assert revision == "20261024_01"
    assert extension_version
    assert {"documents", "document_chunks"}.issubset(inspector.get_table_names())
    columns = {column["name"] for column in inspector.get_columns("document_chunks")}
    assert {"embedding_model", "embedding_dimensions", "embedding_text_hash"}.issubset(columns)
    assert {
        "local_embedding",
        "local_embedding_model",
        "local_embedding_dimensions",
        "local_embedding_text_hash",
    }.issubset(columns)
    assert "ix_document_chunks_embedding_cache" in {
        index["name"] for index in inspector.get_indexes("document_chunks")
    }
    assert "uq_documents_non_webpage_content_hash" in {
        index["name"] for index in inspector.get_indexes("documents")
    }
    assert isinstance(DocumentChunkRecord.__table__.c.embedding.type, Vector)
    assert DocumentChunkRecord.__table__.c.local_embedding.type.dim == 384
    with postgres_engine.connect() as connection:
        lexical_index = connection.scalar(
            text(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() "
                "AND tablename = 'document_chunks' AND indexname = 'ix_document_chunks_text_fts'"
            )
        )
    assert lexical_index == "ix_document_chunks_text_fts"


def test_postgres_local_384_vectors_store_and_retrieve_separately_from_legacy(
    postgres_session: Session,
) -> None:
    marker = uuid4().hex
    source_url = f"https://local-integration.invalid/{marker}"
    document = ingest_text(
        f"The local MiniLM vector has 384 dimensions. PostgreSQL stores this test {marker}.",
        source_type="webpage",
        title="Local embedding integration fixture",
        source_url=source_url,
    )
    repository = DocumentRepository(postgres_session)
    service = MockLocalEmbeddingService()
    try:
        saved = embed_and_persist(document, repository=repository, service=service)
        stored = postgres_session.scalars(
            select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == saved.id)
        ).all()
        assert stored
        assert all(chunk.embedding is None for chunk in stored)
        assert all(len(chunk.local_embedding) == 384 for chunk in stored)
        assert all(chunk.local_embedding_model == LOCAL_EMBEDDING_MODEL for chunk in stored)
        assert all(chunk.local_embedding_dimensions == 384 for chunk in stored)

        results = VectorRetriever(
            postgres_session, embedding_service=MockLocalEmbeddingService()
        ).retrieve("What is the local vector size?", source_url=source_url)
        assert results
        assert results[0].document_id == saved.id
        assert results[0].source_url == source_url
    finally:
        postgres_session.rollback()
        postgres_session.execute(
            delete(DocumentRecord).where(DocumentRecord.id == document.document.id)
        )
        postgres_session.commit()


def test_postgres_ingestion_embedding_storage_and_vector_retrieval(
    postgres_session: Session,
) -> None:
    marker = uuid4().hex
    source_url = f"https://integration.invalid/{marker}"
    document = ingest_text(
        "PostgreSQL stores vectors for semantic search. The database ranks similar chunks.\n\n"
        f"The moon orbits Earth and influences tides. Integration marker {marker}.",
        source_type="webpage",
        title="PostgreSQL integration fixture",
        source_url=source_url,
        metadata={"integration_test": True},
        chunking=ChunkingConfig(chunk_size=58, overlap=0),
    )
    assert len(document.chunks) >= 2
    repository = DocumentRepository(postgres_session)

    try:
        saved = embed_and_persist(
            document,
            repository=repository,
            service=MockEmbeddingService(),
        )
        stored = postgres_session.scalars(
            select(DocumentChunkRecord)
            .where(DocumentChunkRecord.document_id == saved.id)
            .order_by(DocumentChunkRecord.index)
        ).all()

        assert saved.id == document.document.id
        assert len(stored) == len(document.chunks)
        assert all(chunk.embedding is not None for chunk in stored)
        assert all(chunk.embedding_model == TEST_EMBEDDING_MODEL for chunk in stored)
        assert all(chunk.embedding_dimensions == 3 for chunk in stored)
        assert all(len(chunk.embedding) == 3 for chunk in stored)

        results = VectorRetriever(
            postgres_session,
            embedding_service=MockEmbeddingService(),
        ).retrieve("What does PostgreSQL store?", source_url=source_url, top_k=3)

        assert results
        assert results[0].document_id == saved.id
        assert results[0].source_url == source_url
        assert "PostgreSQL" in results[0].text
        assert results[0].similarity == pytest.approx(1.0)
    finally:
        postgres_session.rollback()
        postgres_session.execute(
            delete(DocumentRecord).where(DocumentRecord.id == document.document.id)
        )
        postgres_session.commit()


def test_postgres_page_indexing_reuses_vectors_and_retrieves_only_requested_url(
    postgres_session: Session,
) -> None:
    marker = uuid4().hex
    page_text = (
        "PostgreSQL stores page chunks for semantic search. "
        f"PageGlance integration fixture {marker}."
    )
    first = ingest_text(
        page_text,
        source_type="webpage",
        title="First URL",
        source_url=f"HTTPS://FIRST.INTEGRATION.INVALID:443/{marker}#page",
    )
    second = ingest_text(
        page_text,
        source_type="webpage",
        title="Second URL",
        source_url=f"https://second.integration.invalid/{marker}#page",
    )
    assert first.document.source_url == f"https://first.integration.invalid/{marker}#page"
    assert first.document.id != second.document.id

    service = MockEmbeddingService()
    repository = DocumentRepository(postgres_session)
    try:
        saved_first = embed_and_persist(first, repository=repository, service=service)
        after_first_index = len(service.calls)
        saved_second = embed_and_persist(second, repository=repository, service=service)
        saved_first_again = embed_and_persist(
            ingest_text(
                page_text,
                source_type="webpage",
                title="First URL",
                source_url=first.document.source_url,
            ),
            repository=repository,
            service=service,
        )

        assert saved_first.id == saved_first_again.id == first.document.id
        assert saved_second.id == second.document.id
        assert saved_first.id != saved_second.id
        assert len(service.calls) == after_first_index

        retriever = VectorRetriever(postgres_session, embedding_service=service)
        first_results = retriever.retrieve(
            "What does PostgreSQL store?", source_url=first.document.source_url
        )
        second_results = retriever.retrieve(
            "What does PostgreSQL store?", source_url=second.document.source_url
        )
        assert first_results
        assert second_results
        assert {chunk.document_id for chunk in first_results} == {saved_first.id}
        assert {chunk.source_url for chunk in first_results} == {first.document.source_url}
        assert {chunk.document_id for chunk in second_results} == {saved_second.id}
        assert {chunk.source_url for chunk in second_results} == {second.document.source_url}
        assert all("PostgreSQL" in chunk.text for chunk in first_results)
    finally:
        postgres_session.rollback()
        postgres_session.execute(
            delete(DocumentRecord).where(
                DocumentRecord.id.in_([first.document.id, second.document.id])
            )
        )
        postgres_session.commit()


def test_postgres_uploaded_document_isolated_from_webpage_url_retrieval(
    postgres_session: Session,
) -> None:
    marker = uuid4().hex
    text_body = f"PageGlance uploaded document isolation fixture {marker}."
    page_url = f"https://upload-isolation.integration.invalid/{marker}"
    service = MockEmbeddingService()
    factory: sessionmaker[Session] = sessionmaker(
        bind=postgres_session.get_bind(), expire_on_commit=False
    )
    uploaded = ingest_uploaded_file(
        "fixture.txt",
        text_body.encode(),
        session_factory=factory,
        embedding_service=service,
    )
    page = ingest_text(
        text_body,
        source_type="webpage",
        title="Page fixture",
        source_url=page_url,
    )
    repository = DocumentRepository(postgres_session)
    try:
        embed_and_persist(page, repository=repository, service=service)
        retriever = VectorRetriever(postgres_session, embedding_service=service)
        page_results = retriever.retrieve("isolation fixture", source_url=page_url)
        upload_results = retriever.retrieve("isolation fixture", document_id=uploaded.document.id)

        assert page_results
        assert {item.source_type for item in page_results} == {"webpage"}
        assert {item.document_id for item in page_results} == {page.document.id}
        assert upload_results
        assert {item.source_type for item in upload_results} == {"text"}
        assert {item.document_id for item in upload_results} == {uploaded.document.id}
        assert all(item.source_url is None for item in upload_results)
    finally:
        postgres_session.rollback()
        postgres_session.execute(
            delete(DocumentRecord).where(
                DocumentRecord.id.in_([uploaded.document.id, page.document.id])
            )
        )
        postgres_session.commit()


def test_postgres_hybrid_lexical_vector_rrf_and_url_scope(postgres_session: Session) -> None:
    marker = uuid4().hex
    active_url = f"https://hybrid.integration.invalid/active/{marker}"
    unrelated_url = f"https://hybrid.integration.invalid/unrelated/{marker}"
    active_text = f"Active record valve uses cobalt-alloy-{marker} for pressure control."
    unrelated_text = f"Unrelated record valve uses nickel-alloy-{marker} for pressure control."
    embedding = MockLocalEmbeddingService()
    repository = DocumentRepository(postgres_session)
    active = ingest_text(
        active_text,
        source_type="webpage",
        title="Active hybrid fixture",
        source_url=active_url,
    )
    unrelated = ingest_text(
        unrelated_text,
        source_type="webpage",
        title="Unrelated hybrid fixture",
        source_url=unrelated_url,
    )
    saved_active = embed_and_persist(active, repository=repository, service=embedding)
    saved_unrelated = embed_and_persist(unrelated, repository=repository, service=embedding)

    try:
        retriever = VectorRetriever(postgres_session, embedding)
        question = f"What alloy is used in the active valve cobalt-alloy-{marker}?"
        lexical = retriever.retrieve_lexical_only(question, source_url=active_url)
        vector = retriever.retrieve_vector_only(question, source_url=active_url)
        fused = retriever.retrieve(question, source_url=active_url)

        assert lexical and lexical[0].document_id == saved_active.id
        assert vector and {item.document_id for item in vector} == {saved_active.id}
        assert fused and {item.document_id for item in fused} == {saved_active.id}
        assert all(item.document_id != saved_unrelated.id for item in fused)

        with postgres_session.get_bind().connect() as connection:
            with connection.begin():
                connection.execute(text("SET LOCAL enable_seqscan = off"))
                plan = (
                    connection.execute(
                        text(
                            "EXPLAIN (COSTS OFF) SELECT id FROM document_chunks "
                            "WHERE to_tsvector('simple'::regconfig, text) @@ "
                            "websearch_to_tsquery('simple'::regconfig, :query)"
                        ),
                        {"query": f"cobalt-alloy-{marker}"},
                    )
                    .scalars()
                    .all()
                )
        assert any("ix_document_chunks_text_fts" in str(line) for line in plan)
    finally:
        postgres_session.rollback()
        postgres_session.execute(
            delete(DocumentRecord).where(
                DocumentRecord.id.in_([saved_active.id, saved_unrelated.id])
            )
        )
        postgres_session.commit()
