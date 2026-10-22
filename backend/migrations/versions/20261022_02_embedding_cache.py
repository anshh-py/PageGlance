"""Track embedding model profiles and chunk content hashes."""

import sqlalchemy as sa
from alembic import op

revision = "20261022_02"
down_revision = "20261022_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document_chunks", sa.Column("embedding_model", sa.String(length=255)))
    op.add_column("document_chunks", sa.Column("embedding_dimensions", sa.Integer()))
    op.add_column("document_chunks", sa.Column("embedding_text_hash", sa.String(length=64)))
    op.create_index(
        "ix_document_chunks_embedding_cache",
        "document_chunks",
        ["embedding_model", "embedding_dimensions", "embedding_text_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding_cache", table_name="document_chunks")
    op.drop_column("document_chunks", "embedding_text_hash")
    op.drop_column("document_chunks", "embedding_dimensions")
    op.drop_column("document_chunks", "embedding_model")
