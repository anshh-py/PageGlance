"""Replace the global content-hash uniqueness with owner-scoped uniqueness.

Why: the previous global unique index on ``content_hash`` (for non-webpage
documents) made it impossible for two different installations to ingest the
same file -- the second owner's insert was rejected by the database with a
UNIQUE violation, even though the application layer already implemented
per-owner deduplication.

This migration is additive and non-destructive:
- It drops the global partial unique index ``uq_documents_non_webpage_content_hash``.
- It creates a partial unique index on ``(owner_id, content_hash)`` for
  non-webpage documents. Because PostgreSQL treats NULLs as distinct in a
  unique index, legacy ownerless rows (``owner_id`` NULL) remain untouched and
  readable; no production data is rewritten, reassigned, or lost.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260401_02"
down_revision = "20260401_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("uq_documents_non_webpage_content_hash", table_name="documents")
    op.create_index(
        "uq_documents_owner_content_hash",
        "documents",
        ["owner_id", "content_hash"],
        unique=True,
        postgresql_where=sa.text("source_type <> 'webpage'"),
    )


def downgrade() -> None:
    op.drop_index("uq_documents_owner_content_hash", table_name="documents")
    op.create_index(
        "uq_documents_non_webpage_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
        postgresql_where=sa.text("source_type <> 'webpage'"),
    )
