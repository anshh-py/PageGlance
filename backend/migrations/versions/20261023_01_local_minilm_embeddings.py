"""Add a dedicated 384-dimensional local embedding slot without changing legacy vectors."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "20261023_01"
down_revision = "20261022_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document_chunks", sa.Column("local_embedding", Vector(384), nullable=True))
    op.add_column("document_chunks", sa.Column("local_embedding_model", sa.String(length=255)))
    op.add_column("document_chunks", sa.Column("local_embedding_dimensions", sa.Integer()))
    op.add_column("document_chunks", sa.Column("local_embedding_text_hash", sa.String(length=64)))
    op.create_index(
        "ix_document_chunks_local_embedding_cache",
        "document_chunks",
        ["local_embedding_model", "local_embedding_dimensions", "local_embedding_text_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_local_embedding_cache", table_name="document_chunks")
    op.drop_column("document_chunks", "local_embedding_text_hash")
    op.drop_column("document_chunks", "local_embedding_dimensions")
    op.drop_column("document_chunks", "local_embedding_model")
    op.drop_column("document_chunks", "local_embedding")
