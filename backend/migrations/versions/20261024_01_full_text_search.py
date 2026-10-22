"""Add a GIN expression index for PostgreSQL chunk full-text search."""

from alembic import op

revision = "20261024_01"
down_revision = "20261023_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_document_chunks_text_fts "
        "ON document_chunks USING gin (to_tsvector('simple'::regconfig, text))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX ix_document_chunks_text_fts")
