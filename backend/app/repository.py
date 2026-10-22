"""Persistence operations kept separate from HTTP routes and LLM services."""

from datetime import datetime, timezone
from math import isfinite

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.documents import Document, DocumentChunk, IngestionResult
from app.models import DocumentChunkRecord, DocumentRecord
from app.normalization import exact_text_hash


def _aware(value: datetime) -> datetime:
    """Give SQLite-returned naive timestamps the UTC timezone used by domain models."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _to_document(record: DocumentRecord) -> Document:
    return Document(
        id=record.id,
        source_type=record.source_type,
        title=record.title,
        source_url=record.source_url,
        filename=record.filename,
        content_hash=record.content_hash,
        text=record.text,
        metadata=dict(record.document_metadata),
        created_at=_aware(record.created_at),
        updated_at=_aware(record.updated_at),
    )


def _to_chunk(record: DocumentChunkRecord) -> DocumentChunk:
    return DocumentChunk(
        id=record.id,
        document_id=record.document_id,
        index=record.index,
        text=record.text,
        start_char=record.start_char,
        end_char=record.end_char,
        metadata=dict(record.chunk_metadata),
    )


class DocumentRepository:
    """Save and retrieve ingestion results using a caller-owned SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def _matches_persisted_page(
        self,
        record: DocumentRecord,
        result: IngestionResult,
        embedding_vectors: dict[str, list[float]] | None,
        embedding_model: str | None,
        embedding_dimensions: int | None,
        storage_profile: str,
    ) -> bool:
        """Check whether an indexed page already has the complete requested state."""
        if embedding_vectors is None or not (
            record.source_type == result.document.source_type
            and record.title == result.document.title
            and record.source_url == result.document.source_url
            and record.filename == result.document.filename
            and record.content_hash == result.document.content_hash
            and record.text == result.document.text
            and dict(record.document_metadata) == result.document.metadata
            and len(record.chunks) == len(result.chunks)
        ):
            return False

        for stored, chunk in zip(record.chunks, result.chunks, strict=True):
            vector = embedding_vectors.get(chunk.id)
            if storage_profile == "local":
                stored_vector = stored.local_embedding
                stored_model = stored.local_embedding_model
                stored_dimensions = stored.local_embedding_dimensions
                stored_hash = stored.local_embedding_text_hash
            else:
                stored_vector = stored.embedding
                stored_model = stored.embedding_model
                stored_dimensions = stored.embedding_dimensions
                stored_hash = stored.embedding_text_hash
            if not (
                stored.index == chunk.index
                and stored.text == chunk.text
                and stored.start_char == chunk.start_char
                and stored.end_char == chunk.end_char
                and dict(stored.chunk_metadata) == chunk.metadata
                and vector is not None
                and list(stored_vector or []) == vector
                and stored_model == embedding_model
                and stored_dimensions == embedding_dimensions
                and stored_hash == exact_text_hash(chunk.text)
            ):
                return False
        return True

    def _chunk_records(
        self,
        result: IngestionResult,
        document_id: str,
        embedding_vectors: dict[str, list[float]] | None,
        embedding_model: str | None,
        embedding_dimensions: int | None,
        storage_profile: str,
        existing_chunks: dict[tuple[int, str, int, int], DocumentChunkRecord],
    ) -> list[DocumentChunkRecord]:
        records: list[DocumentChunkRecord] = []
        for chunk in result.chunks:
            vector = embedding_vectors.get(chunk.id) if embedding_vectors is not None else None
            prior = existing_chunks.get((chunk.index, chunk.text, chunk.start_char, chunk.end_char))
            chunk_id = (
                chunk.id
                if document_id == chunk.document_id
                else f"{document_id}:chunk:{chunk.index}:{chunk.start_char}:{chunk.end_char}"
            )
            fields: dict[str, object] = {
                "id": chunk_id,
                "document_id": document_id,
                "index": chunk.index,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_metadata": chunk.metadata,
            }
            if storage_profile == "local":
                fields.update(
                    embedding=prior.embedding if prior is not None else None,
                    embedding_model=prior.embedding_model if prior is not None else None,
                    embedding_dimensions=prior.embedding_dimensions if prior is not None else None,
                    embedding_text_hash=prior.embedding_text_hash if prior is not None else None,
                    local_embedding=vector,
                    local_embedding_model=embedding_model if vector is not None else None,
                    local_embedding_dimensions=embedding_dimensions if vector is not None else None,
                    local_embedding_text_hash=exact_text_hash(chunk.text)
                    if vector is not None
                    else None,
                )
            else:
                fields.update(
                    embedding=vector,
                    embedding_model=embedding_model if vector is not None else None,
                    embedding_dimensions=embedding_dimensions if vector is not None else None,
                    embedding_text_hash=exact_text_hash(chunk.text) if vector is not None else None,
                    local_embedding=prior.local_embedding if prior is not None else None,
                    local_embedding_model=prior.local_embedding_model
                    if prior is not None
                    else None,
                    local_embedding_dimensions=prior.local_embedding_dimensions
                    if prior is not None
                    else None,
                    local_embedding_text_hash=prior.local_embedding_text_hash
                    if prior is not None
                    else None,
                )
            records.append(DocumentChunkRecord(**fields))
        return records

    def _existing_chunk_map(
        self, document_id: str
    ) -> dict[tuple[int, str, int, int], DocumentChunkRecord]:
        records = self.session.scalars(
            select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == document_id)
        ).all()
        return {
            (record.index, record.text, record.start_char, record.end_char): record
            for record in records
        }

    def persist(
        self,
        result: IngestionResult,
        *,
        embedding_vectors: dict[str, list[float]] | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        storage_profile: str = "legacy",
    ) -> Document:
        """Save a document and its chunks, returning the canonical copy on duplicates.

        Non-webpage content-hash duplicates share one stored document. Webpages retain
        separate records per normalized URL. Re-ingesting changed content at a stable
        source ID updates that document atomically.
        """
        if embedding_vectors is not None:
            expected_ids = {chunk.id for chunk in result.chunks}
            if set(embedding_vectors) != expected_ids:
                raise ValueError("An embedding vector is required for each document chunk.")
            if not embedding_model or not embedding_dimensions:
                raise ValueError("Embedding model and dimensions are required with vectors.")
            if any(
                len(vector) != embedding_dimensions or not all(isfinite(value) for value in vector)
                for vector in embedding_vectors.values()
            ):
                raise ValueError("An embedding vector does not match its configured dimensions.")
        if storage_profile not in {"legacy", "local"}:
            raise ValueError("Unsupported embedding storage profile.")

        duplicate_query = select(DocumentRecord).where(
            DocumentRecord.content_hash == result.document.content_hash,
            DocumentRecord.source_type != "webpage",
        )
        duplicate = (
            None
            if result.document.source_type == "webpage"
            else self.session.scalar(duplicate_query)
        )
        if duplicate is not None:
            if embedding_vectors is not None:
                existing_chunks = self._existing_chunk_map(duplicate.id)
                duplicate.chunks = self._chunk_records(
                    result,
                    duplicate.id,
                    embedding_vectors,
                    embedding_model,
                    embedding_dimensions,
                    storage_profile,
                    existing_chunks,
                )
                self.session.commit()
            return _to_document(duplicate)

        record = self.session.get(DocumentRecord, result.document.id)
        existing_chunks = self._existing_chunk_map(result.document.id)
        if record is not None and self._matches_persisted_page(
            record,
            result,
            embedding_vectors,
            embedding_model,
            embedding_dimensions,
            storage_profile,
        ):
            return _to_document(record)

        if record is None:
            record = DocumentRecord(id=result.document.id)
            self.session.add(record)
        else:
            self.session.execute(
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.document_id == result.document.id
                )
            )

        record.source_type = result.document.source_type
        record.title = result.document.title
        record.source_url = result.document.source_url
        record.filename = result.document.filename
        record.content_hash = result.document.content_hash
        record.text = result.document.text
        record.document_metadata = result.document.metadata
        if record.created_at is None:
            record.created_at = result.document.created_at
        record.updated_at = result.document.updated_at
        record.chunks = self._chunk_records(
            result,
            record.id,
            embedding_vectors,
            embedding_model,
            embedding_dimensions,
            storage_profile,
            existing_chunks,
        )

        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            duplicate = (
                self.session.get(DocumentRecord, result.document.id)
                if result.document.source_type == "webpage"
                else self.session.scalar(duplicate_query)
            )
            if duplicate is None:
                raise
            if result.document.source_type == "webpage":
                duplicate.source_type = result.document.source_type
                duplicate.title = result.document.title
                duplicate.source_url = result.document.source_url
                duplicate.filename = result.document.filename
                duplicate.content_hash = result.document.content_hash
                duplicate.text = result.document.text
                duplicate.document_metadata = result.document.metadata
                duplicate.updated_at = result.document.updated_at
            if embedding_vectors is not None:
                duplicate.chunks = self._chunk_records(
                    result,
                    duplicate.id,
                    embedding_vectors,
                    embedding_model,
                    embedding_dimensions,
                    storage_profile,
                    self._existing_chunk_map(duplicate.id),
                )
                self.session.commit()
            return _to_document(duplicate)

        self.session.refresh(record)
        return _to_document(record)

    def get_reusable_embeddings(
        self,
        result: IngestionResult,
        *,
        embedding_model: str,
        embedding_dimensions: int,
        storage_profile: str = "legacy",
    ) -> dict[str, list[float]]:
        """Find vectors reusable by exact chunk-text hash and matching model profile."""
        hashes = {exact_text_hash(chunk.text) for chunk in result.chunks}
        if not hashes:
            return {}
        if storage_profile == "local":
            model_column = DocumentChunkRecord.local_embedding_model
            dimensions_column = DocumentChunkRecord.local_embedding_dimensions
            hash_column = DocumentChunkRecord.local_embedding_text_hash
            vector_column = DocumentChunkRecord.local_embedding
        elif storage_profile == "legacy":
            model_column = DocumentChunkRecord.embedding_model
            dimensions_column = DocumentChunkRecord.embedding_dimensions
            hash_column = DocumentChunkRecord.embedding_text_hash
            vector_column = DocumentChunkRecord.embedding
        else:
            raise ValueError("Unsupported embedding storage profile.")
        records = self.session.scalars(
            select(DocumentChunkRecord).where(
                model_column == embedding_model,
                dimensions_column == embedding_dimensions,
                hash_column.in_(hashes),
                vector_column.is_not(None),
            )
        ).all()
        vectors_by_hash: dict[str, list[float]] = {}
        for record in records:
            if storage_profile == "local":
                text_hash, vector = record.local_embedding_text_hash, record.local_embedding
            else:
                text_hash, vector = record.embedding_text_hash, record.embedding
            if text_hash is not None and vector is not None:
                vectors_by_hash[text_hash] = list(vector)
        return {
            chunk.id: vectors_by_hash[exact_text_hash(chunk.text)]
            for chunk in result.chunks
            if exact_text_hash(chunk.text) in vectors_by_hash
        }

    def get_by_id(self, document_id: str) -> IngestionResult | None:
        """Return a document and its chunks ordered by chunk index, or None."""
        record = self.session.get(DocumentRecord, document_id)
        if record is None:
            return None
        chunk_records = self.session.scalars(
            select(DocumentChunkRecord)
            .where(DocumentChunkRecord.document_id == document_id)
            .order_by(DocumentChunkRecord.index)
        ).all()
        return IngestionResult(
            document=_to_document(record),
            chunks=[_to_chunk(chunk) for chunk in chunk_records],
        )
