from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "002_add_assistant_audit"
down_revision = "001_create_careops_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("assistant_audit_logs"):
        op.create_table(
            "assistant_audit_logs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("run_id", sa.String(length=64), nullable=True),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("provider", sa.String(length=64), nullable=False),
            sa.Column("model", sa.String(length=128), nullable=False),
            sa.Column("response_status", sa.String(length=32), nullable=False),
            sa.Column("sources", sa.JSON(), nullable=False),
            sa.Column("latency_ms", sa.Integer(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("assistant_audit_logs"):
        op.drop_table("assistant_audit_logs")
