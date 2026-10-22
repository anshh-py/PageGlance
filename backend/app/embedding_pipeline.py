"""Coordinate chunk embedding with persistence while keeping each layer independent."""

from app.documents import Document, IngestionResult
from app.embeddings import EmbeddingService
from app.normalization import exact_text_hash
from app.repository import DocumentRepository


def embed_and_persist(
    result: IngestionResult,
    *,
    repository: DocumentRepository,
    service: EmbeddingService,
) -> Document:
    """Embed only uncached chunk text, then atomically persist content and vectors."""
    model_version = service.model_version
    dimensions = service.dimensions
    storage_profile = getattr(service, "storage_profile", "legacy")
    vectors = repository.get_reusable_embeddings(
        result,
        embedding_model=model_version,
        embedding_dimensions=dimensions,
        storage_profile=storage_profile,
    )

    pending_by_hash: dict[str, str] = {}
    for chunk in result.chunks:
        if chunk.id not in vectors:
            pending_by_hash.setdefault(exact_text_hash(chunk.text), chunk.text)

    if pending_by_hash:
        generated = service.embed_texts(list(pending_by_hash.values()))
        if len(generated) != len(pending_by_hash):
            raise ValueError("Embedding service returned an invalid number of vectors.")
        generated_by_hash = dict(zip(pending_by_hash, generated, strict=True))
        for chunk in result.chunks:
            if chunk.id not in vectors:
                vectors[chunk.id] = generated_by_hash[exact_text_hash(chunk.text)]

    return repository.persist(
        result,
        embedding_vectors=vectors,
        embedding_model=model_version,
        embedding_dimensions=dimensions,
        storage_profile=storage_profile,
    )
