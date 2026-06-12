"""Add belief_states table.

Revision ID: 009_belief_states
Revises: 008_ingestion_hardening
Create Date: 2026-06-13 00:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]

revision: str = "009_belief_states"
down_revision: str | None = "008_ingestion_hardening"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "belief_states",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state", postgresql.JSONB(), nullable=False),
        sa.Column("rebuild_type", sa.Text(), nullable=False),
        sa.Column(
            "last_summary_created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
        ),
        sa.Column("summary_count_covered", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rebuild_type IN ('incremental', 'full')",
            name="ck_belief_states_rebuild_type",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "version", name="uq_belief_states_project_version"),
    )
    op.create_index(
        "ix_belief_states_project_version",
        "belief_states",
        ["project_id", sa.text("version DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_belief_states_project_version", table_name="belief_states")
    op.drop_table("belief_states")
