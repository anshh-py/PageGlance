"""Add the shared fixed-window rate-limit counter table.

The Hugging Face deployment has no Redis, so rate limits are enforced with a
counter stored in the same application database (shared across workers and
across restarts). This table is additive and non-destructive.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260401_03"
down_revision = "20260401_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_counters",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("window_start", sa.Integer(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
    )
    op.create_index(
        "uq_rate_limit_action_key_window",
        "rate_limit_counters",
        ["action", "key", "window_start"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_rate_limit_action_key_window", table_name="rate_limit_counters")
    op.drop_table("rate_limit_counters")
