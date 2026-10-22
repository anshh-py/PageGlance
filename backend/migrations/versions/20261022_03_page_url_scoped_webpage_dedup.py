"""Allow identical webpage text to be stored under distinct source URLs."""

import sqlalchemy as sa
from alembic import op

revision = "20261022_03"
down_revision = "20261022_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_documents_content_hash", "documents", type_="unique")
    op.create_index(
        "uq_documents_non_webpage_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
        postgresql_where=sa.text("source_type <> 'webpage'"),
    )


def downgrade() -> None:
    op.drop_index("uq_documents_non_webpage_content_hash", table_name="documents")
    op.create_unique_constraint("uq_documents_content_hash", "documents", ["content_hash"])
