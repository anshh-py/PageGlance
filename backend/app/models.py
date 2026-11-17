"""SQLAlchemy persistence models for normalized documents and text chunks."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

_JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Declarative metadata shared by application models and Alembic."""


class DocumentRecord(Base):
    """Persistent normalized document with source metadata and stable content hash."""

    __tablename__ = "documents"
    __table_args__ = (
        Index(
            "uq_documents_owner_content_hash",
            "owner_id",
            "content_hash",
            unique=True,
            postgresql_where=text("source_type <> 'webpage'"),
            sqlite_where=text("source_type <> 'webpage'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Installation-level owner for the document. NULL = legacy/ownerless record
    # that predates auth and is never reachable through authenticated routes.
    owner_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    document_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", _JSON_TYPE, default=dict, nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunkRecord.index",
        passive_deletes=True,
    )


class InstallationRecord(Base):
    """An authenticated browser installation and its token hash (never the token)."""

    __tablename__ = "installations"
    __table_args__ = (Index("ix_installations_token_hash", "token_hash"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RateLimitCounter(Base):
    """Fixed-window rate-limit counter shared across workers."""

    __tablename__ = "rate_limit_counters"
    __table_args__ = (
        Index(
            "uq_rate_limit_action_key_window",
            "action",
            "key",
            "window_start",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    window_start: Mapped[int] = mapped_column(Integer, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)


class DocumentChunkRecord(Base):
    """Persistent chunk with order, exact normalized-text offsets, and future vector slot."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
        CheckConstraint("chunk_index >= 0", name="ck_document_chunks_nonnegative_index"),
        CheckConstraint("start_char >= 0 AND end_char >= start_char", name="ck_chunk_offsets"),
        Index("ix_document_chunks_document_id", "document_id"),
        Index(
            "ix_document_chunks_text_fts",
            text("to_tsvector('simple'::regconfig, text)"),
            postgresql_using="gin",
        ).ddl_if(dialect="postgresql"),
        Index(
            "ix_document_chunks_embedding_cache",
            "embedding_model",
            "embedding_dimensions",
            "embedding_text_hash",
        ),
        Index(
            "ix_document_chunks_local_embedding_cache",
            "local_embedding_model",
            "local_embedding_dimensions",
            "local_embedding_text_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column("chunk_index", Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", _JSON_TYPE, default=dict, nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer)
    embedding_text_hash: Mapped[str | None] = mapped_column(String(64))
    # Additive fixed-size slot for local MiniLM vectors; legacy vectors remain untouched.
    local_embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    local_embedding_model: Mapped[str | None] = mapped_column(String(255))
    local_embedding_dimensions: Mapped[int | None] = mapped_column(Integer)
    local_embedding_text_hash: Mapped[str | None] = mapped_column(String(64))
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")
