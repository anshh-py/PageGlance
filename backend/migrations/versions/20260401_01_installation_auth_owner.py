"""Add installation auth table and owner scoping on documents.

This migration is additive and non-destructive:
- It creates a new ``installations`` table (tokens are stored only as hashes).
- It adds a nullable ``owner_id`` column to ``documents``. Existing rows keep
  ``owner_id`` NULL (ownerless legacy records), which are deliberately not
  reachable through authenticated routes, so no production data is lost or
  rewritten.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260401_01"
down_revision = "20261024_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "installations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("token_hash", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_installations_token_hash", "installations", ["token_hash"])
    op.add_column("documents", sa.Column("owner_id", sa.String(length=64), nullable=True))
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_column("documents", "owner_id")
    op.drop_index("ix_installations_token_hash", table_name="installations")
    op.drop_table("installations")
