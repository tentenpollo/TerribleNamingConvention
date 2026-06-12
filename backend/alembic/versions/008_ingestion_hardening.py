"""Harden Phase 2 ingestion storage and idempotency.

Revision ID: 008_ingestion_hardening
Revises: 007_ingestion_jobs
Create Date: 2026-06-11 00:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]

revision: str = "008_ingestion_hardening"
down_revision: str | None = "007_ingestion_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("raw_bytes", postgresql.BYTEA(), nullable=True))
    op.execute("UPDATE documents SET raw_bytes = convert_to(raw_content, 'UTF8')")
    op.alter_column("documents", "raw_bytes", nullable=False)
    op.drop_column("documents", "raw_content")

    op.create_unique_constraint(
        "uq_document_summaries_document_id",
        "document_summaries",
        ["document_id"],
    )
    op.create_index(
        "ix_document_summaries_project_created",
        "document_summaries",
        ["project_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_summaries_project_created", table_name="document_summaries")
    op.drop_constraint(
        "uq_document_summaries_document_id",
        "document_summaries",
        type_="unique",
    )

    op.add_column("documents", sa.Column("raw_content", sa.Text(), nullable=True))
    connection = op.get_bind()
    rows = connection.execute(sa.text("SELECT id, raw_bytes FROM documents")).mappings()
    for row in rows:
        raw_bytes = bytes(row["raw_bytes"])
        raw_content = raw_bytes.decode("utf-8", errors="replace")
        connection.execute(
            sa.text("UPDATE documents SET raw_content = :raw_content WHERE id = :id"),
            {"raw_content": raw_content, "id": row["id"]},
        )
    op.alter_column("documents", "raw_content", nullable=False)
    op.drop_column("documents", "raw_bytes")
