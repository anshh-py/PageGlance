"""Ragas embedding adapter that uses PageGlance's local embedding service."""

from typing import List

from ragas.embeddings.base import BaseRagasEmbedding

from app.embeddings import EmbeddingService, get_default_embedding_service


class PageGlanceRagasEmbedding(BaseRagasEmbedding):
    """Adapter to make PageGlance embeddings compatible with Ragas."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        super().__init__()
        self.embedding_service = embedding_service or get_default_embedding_service()

    def embed_text(self, text: str) -> List[float]:
        result = self.embedding_service.embed_texts([text])
        return result[0] if result else []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_service.embed_texts(texts)

    async def aembed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.aembed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.aembed_texts(texts)
