from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal
import uuid

from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BeliefStateNotFoundError,
    BeliefStateVersionConflictError,
    InvalidBeliefStateError,
)
from app.models.belief_state import BeliefState
from app.models.document_summary import DocumentSummary
from app.schemas.belief_state import BeliefStateContent, BeliefStateRecord, BeliefStateVersionMeta

SynthesizeFn = Callable[
    [BeliefStateContent | None, list[DocumentSummary]],
    Awaitable[BeliefStateContent],
]


class CAGService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_latest(self, project_id: uuid.UUID) -> BeliefStateRecord | None:
        result = await self.session.execute(
            select(BeliefState)
            .where(BeliefState.project_id == project_id)
            .order_by(BeliefState.version.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return _to_record(row)

    async def list_versions(self, project_id: uuid.UUID) -> list[BeliefStateVersionMeta]:
        result = await self.session.execute(
            select(
                BeliefState.version,
                BeliefState.rebuild_type,
                BeliefState.created_at,
                BeliefState.summary_count_covered,
            )
            .where(BeliefState.project_id == project_id)
            .order_by(BeliefState.version.desc())
        )
        return [BeliefStateVersionMeta.model_validate(row._mapping) for row in result.all()]

    async def get_version(self, project_id: uuid.UUID, version: int) -> BeliefStateRecord:
        result = await self.session.execute(
            select(BeliefState).where(
                BeliefState.project_id == project_id, BeliefState.version == version
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise BeliefStateNotFoundError(
                f"Belief state version {version} not found for project {project_id}"
            )
        return _to_record(row)

    async def get_window_since(
        self,
        project_id: uuid.UUID,
        watermark: datetime | None,
    ) -> list[DocumentSummary]:
        # Exclude rows where summary->>'raw_text_fallback' is the JSON boolean true.
        # Rows that lack the key must be INCLUDED — COALESCE handles that.
        not_fallback = func.coalesce(
            DocumentSummary.summary["raw_text_fallback"].as_boolean(),
            False,
        ).is_(False)

        stmt = (
            select(DocumentSummary)
            .where(DocumentSummary.project_id == project_id)
            .where(not_fallback)
            .order_by(DocumentSummary.created_at.asc())
        )
        if watermark is not None:
            stmt = stmt.where(DocumentSummary.created_at > watermark)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def synthesize_and_insert(
        self,
        project_id: uuid.UUID,
        synthesize_fn: SynthesizeFn,
        batch_size: int = 40,
    ) -> tuple[BeliefStateRecord | None, int]:
        """Read the current belief state, synthesize a new version, and insert it.

        Holds a transaction-scoped advisory lock across the read-synthesize-insert
        sequence to serialize belief-state writes per project. The LLM call runs
        inside that lock — a deliberate v1 trade-off that makes concurrent updates
        for the same project single-flight; enqueue debounce keeps contention rare.

        Returns the new record and the number of summaries remaining in the window
        after the consumed batch (0 if the whole window was consumed).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        async with self.session.begin():
            await self.session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                {"k": f"cag:{project_id}"},
            )

            latest = await self.get_latest(project_id)
            watermark = latest.last_summary_created_at if latest else None
            previous_count = latest.summary_count_covered if latest else 0

            window = await self.get_window_since(project_id, watermark)
            if not window:
                return None, 0

            remaining = 0
            if len(window) > batch_size:
                remaining = len(window) - batch_size
                window = window[:batch_size]

            current_state = latest.state if latest else None
            content = await synthesize_fn(current_state, window)
            if not isinstance(content, BeliefStateContent):
                raise TypeError("synthesize_fn must return a BeliefStateContent instance")

            max_version_result = await self.session.execute(
                select(func.max(BeliefState.version)).where(BeliefState.project_id == project_id)
            )
            current_max = max_version_result.scalar_one_or_none()
            next_version = (current_max or 0) + 1

            last_summary_created_at = max(summary.created_at for summary in window)
            summary_count_covered = previous_count + len(window)

            row = BeliefState(
                project_id=project_id,
                version=next_version,
                state=content.model_dump(),
                rebuild_type="incremental",
                last_summary_created_at=last_summary_created_at,
                summary_count_covered=summary_count_covered,
            )
            self.session.add(row)

        await self.session.refresh(row)
        return _to_record(row), remaining

    async def count_pending(
        self,
        project_id: uuid.UUID,
        watermark: datetime | None,
    ) -> int:
        """Count non-fallback summaries for project newer than the watermark."""
        not_fallback = func.coalesce(
            DocumentSummary.summary["raw_text_fallback"].as_boolean(),
            False,
        ).is_(False)

        stmt = (
            select(func.count())
            .select_from(DocumentSummary)
            .where(DocumentSummary.project_id == project_id)
            .where(not_fallback)
        )
        if watermark is not None:
            stmt = stmt.where(DocumentSummary.created_at > watermark)

        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def insert_version(
        self,
        project_id: uuid.UUID,
        content: BeliefStateContent,
        rebuild_type: Literal["incremental", "full"],
        last_summary_created_at: datetime,
        summary_count_covered: int,
    ) -> BeliefStateRecord:
        if not isinstance(content, BeliefStateContent):
            raise TypeError("content must be a BeliefStateContent instance")

        try:
            async with self.session.begin():
                await self.session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                    {"k": f"cag:{project_id}"},
                )

                max_version_result = await self.session.execute(
                    select(func.max(BeliefState.version)).where(
                        BeliefState.project_id == project_id
                    )
                )
                current_max = max_version_result.scalar_one_or_none()
                next_version = (current_max or 0) + 1

                row = BeliefState(
                    project_id=project_id,
                    version=next_version,
                    state=content.model_dump(),
                    rebuild_type=rebuild_type,
                    last_summary_created_at=last_summary_created_at,
                    summary_count_covered=summary_count_covered,
                )
                self.session.add(row)
        except IntegrityError as exc:
            if "uq_belief_states_project_version" in str(exc.orig):
                raise BeliefStateVersionConflictError(
                    f"Version conflict writing belief state for project {project_id}"
                ) from exc
            raise

        await self.session.refresh(row)
        return _to_record(row)


def _to_record(row: BeliefState) -> BeliefStateRecord:
    try:
        return BeliefStateRecord.model_validate(row)
    except ValidationError as exc:
        raise InvalidBeliefStateError(
            f"Stored belief state for project {row.project_id} version {row.version} "
            f"failed schema validation: {exc}"
        ) from exc
