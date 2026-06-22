from __future__ import annotations

import asyncio
import functools
import json
import math
import random
from typing import Any, cast
import uuid

from arq.connections import ArqRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.exceptions import LLMError
from app.core.llm import llm_call
from app.core.logging import logger
from app.ingestion.embedder import Embedder
from app.models.document_summary import DocumentSummary
from app.models.project import Project
from app.schemas.belief_state import BeliefStateContent, BeliefStateRecord
from app.services.cag import CAGService
from app.services.cag_prompts import (
    format_batch_digest_prompt,
    format_digest_merge_prompt,
    format_incremental_prompt,
    format_initial_prompt,
)

_REBUILD_BATCH_SIZE = 40


async def cag_update(ctx: dict[str, object], project_id: uuid.UUID) -> None:
    """ARQ job: synthesize and insert the next belief-state version.

    Reads the current belief state under a per-project advisory lock, fetches the
    window of summaries not yet covered, calls the LLM to synthesize a new version,
    validates the response, and inserts it. If the window is larger than the batch
    size, consumes one batch and re-enqueues the job to drain the remainder.
    """
    arq_pool = cast(ArqRedis, ctx["redis"])

    async with AsyncSessionLocal() as session:
        project = await _fetch_project(session, project_id)
        context_model = project.config.get("context_model", settings.litellm_context_model)
        # The select above autobegins a transaction. synthesize_and_insert needs to
        # start its own explicit transaction for the advisory lock, so close this one.
        await session.rollback()

        service = CAGService(session)

        async def synthesize_fn(
            current_state: BeliefStateContent | None,
            window: list[Any],
        ) -> BeliefStateContent:
            base_prompt = (
                format_initial_prompt(window)
                if current_state is None
                else format_incremental_prompt(current_state.model_dump(), window)
            )
            return await _synthesize_prompt(
                prompt=base_prompt,
                project_id=project_id,
                context_model=context_model,
            )

        record, remaining = await service.synthesize_and_insert(
            project_id=project_id,
            synthesize_fn=synthesize_fn,
            batch_size=_REBUILD_BATCH_SIZE,
        )

    if record is None:
        logger.info("No new summaries to synthesize", project_id=str(project_id))
        return

    logger.info(
        "Belief state updated incrementally",
        project_id=str(project_id),
        version=record.version,
        summary_count_covered=record.summary_count_covered,
        remaining=remaining,
    )

    if remaining > 0:
        enqueue_result = await arq_pool.enqueue_job(
            "cag_update",
            project_id,
            _job_id=f"cag-update-{project_id}",
        )
        if enqueue_result is None:
            logger.info(
                "CAG update already queued, skipping duplicate enqueue",
                project_id=str(project_id),
            )


async def cag_rebuild(
    ctx: dict[str, object],
    project_id: uuid.UUID,
    mode: str,
) -> None:
    """ARQ job: rebuild a project's belief state from the event log.

    Supports two modes:
      - compaction: latest full state + summaries since its watermark (cheap,
        intended for the weekly cron). Degrades to genesis if no full state exists.
      - genesis: full non-fallback event log from the beginning, ignoring all prior
        states (expensive, admin-only).

    Uses hierarchical Map-Reduce so unbounded logs never exceed a single LLM context
    window. Intermediate digests live in memory only (v1 trade-off; persistence is a
    v2 option for resumability). Shares the same per-project advisory lock key as
    cag_update so rebuilds and updates cannot interleave.
    """
    if mode not in {"compaction", "genesis"}:
        raise ValueError(f"Invalid rebuild mode '{mode}'; must be one of compaction, genesis")

    async with AsyncSessionLocal() as session:
        project = await _fetch_project(session, project_id)
        context_model = project.config.get("context_model", settings.litellm_context_model)
        await session.rollback()

        service = CAGService(session)
        rebuild_fn = functools.partial(
            _hierarchical_reduce,
            project_id=project_id,
            context_model=context_model,
            batch_size=_REBUILD_BATCH_SIZE,
        )

        result = await service.rebuild(
            project_id=project_id,
            rebuild_fn=rebuild_fn,
            mode=mode,  # type: ignore[arg-type]
            batch_size=_REBUILD_BATCH_SIZE,
        )
        if result is None:
            logger.info(
                "CAG rebuild no-op: no non-fallback summaries",
                project_id=str(project_id),
                mode=mode,
            )
            return
        record, previous_record = result

    logger.info(
        "Belief state rebuilt",
        project_id=str(project_id),
        mode=mode,
        version=record.version,
        summary_count_covered=record.summary_count_covered,
    )

    await _emit_rebuild_drift(
        project_id=project_id,
        previous=previous_record,
        current=record,
        embedder=cast(Embedder, ctx.get("embedder")),
    )


async def cag_weekly_rebuild(ctx: dict[str, object]) -> None:
    """ARQ cron: enqueue compaction rebuilds for every project with summaries."""
    arq_pool = cast(ArqRedis, ctx["redis"])

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DocumentSummary.project_id).distinct())
        project_ids = list(result.scalars().all())

    logger.info(
        "Weekly CAG compaction cron starting",
        project_count=len(project_ids),
    )

    for project_id in project_ids:
        enqueue_result = await arq_pool.enqueue_job(
            "cag_rebuild",
            project_id,
            "compaction",
            _job_id=f"cag-rebuild-{project_id}",
        )
        if enqueue_result is None:
            logger.info(
                "CAG rebuild already queued, skipping duplicate enqueue",
                project_id=str(project_id),
            )
        # Jitter to avoid a thundering herd against the LLM provider.
        await asyncio.sleep(random.uniform(0, 5))  # noqa: S311


async def _fetch_project(session: AsyncSession, project_id: uuid.UUID) -> Project:
    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    return project


async def _hierarchical_reduce(
    accumulator: BeliefStateContent | None,
    summaries: list[DocumentSummary],
    project_id: uuid.UUID,
    context_model: str,
    batch_size: int = _REBUILD_BATCH_SIZE,
) -> BeliefStateContent:
    """Map-reduce summaries into a single BeliefStateContent."""
    if not summaries:
        if accumulator is None:
            raise ValueError("No summaries and no accumulator for hierarchical reduce")
        return accumulator

    # Level 0: summaries -> digests. Intermediate digests stay in memory only.
    current_level: list[BeliefStateContent] = []
    for i in range(0, len(summaries), batch_size):
        summary_batch = summaries[i : i + batch_size]
        prompt = format_batch_digest_prompt(summary_batch)
        digest = await _synthesize_prompt(
            prompt=prompt,
            project_id=project_id,
            context_model=context_model,
        )
        current_level.append(digest)

    # The accumulator represents the state of all summaries up to its watermark,
    # so it is the chronologically earliest digest.
    if accumulator is not None:
        current_level.insert(0, accumulator)

    # Level N: merge digests until one remains.
    while len(current_level) > 1:
        next_level: list[BeliefStateContent] = []
        for i in range(0, len(current_level), batch_size):
            digest_batch = current_level[i : i + batch_size]
            prompt = format_digest_merge_prompt(digest_batch)
            merged = await _synthesize_prompt(
                prompt=prompt,
                project_id=project_id,
                context_model=context_model,
            )
            next_level.append(merged)
        current_level = next_level

    return current_level[0]


async def _synthesize_prompt(
    prompt: str,
    project_id: uuid.UUID,
    context_model: str,
) -> BeliefStateContent:
    """Call the LLM to synthesize a belief state, with one retry on validation failure."""
    estimated_tokens = len(prompt) // 4
    if estimated_tokens > settings.cag_max_prompt_tokens:
        raise LLMError(
            f"CAG prompt for project {project_id} exceeds token budget: "
            f"estimated {estimated_tokens} tokens > {settings.cag_max_prompt_tokens}"
        )

    errors: list[str] = []

    for _attempt in range(2):
        current_prompt = prompt
        if errors:
            current_prompt += "\n\nYour previous output violated:\n" + "\n".join(errors)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a structured project memory synthesis system. "
                    "Return only valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": current_prompt},
        ]

        try:
            llm_result = await llm_call(
                messages=messages,
                model=context_model,
                max_tokens=4000,
                response_format={"type": "json_object"},
                temperature=0,
            )
        except LLMError:
            raise

        try:
            parsed = json.loads(llm_result.text)
            return BeliefStateContent.model_validate(parsed)
        except json.JSONDecodeError as exc:
            errors.append(f"JSON parse error: {exc}")
        except ValueError as exc:
            errors.append(f"Schema validation error: {exc}")

    logger.error(
        "Belief state synthesis failed after retry",
        project_id=str(project_id),
        errors=errors,
    )
    raise LLMError(f"Belief state synthesis failed for project {project_id}: {'; '.join(errors)}")


async def _emit_rebuild_drift(
    project_id: uuid.UUID,
    previous: BeliefStateRecord | None,
    current: BeliefStateRecord,
    embedder: Embedder | None,
) -> None:
    """Best-effort drift metrics after a full rebuild. Failures are logged, not raised."""
    try:
        if previous is None:
            decisions_added, decisions_dropped = len(current.state.decisions), 0
            open_items_added, open_items_dropped = len(current.state.open_items), 0
            similarity = None
        else:
            decisions_added, decisions_dropped = _delta_entries(
                previous.state.decisions,
                current.state.decisions,
            )
            open_items_added, open_items_dropped = _delta_entries(
                previous.state.open_items,
                current.state.open_items,
            )
            similarity = await _project_summary_similarity(
                previous.state.project_summary,
                current.state.project_summary,
                embedder,
            )

        logger.info(
            "cag_rebuild_drift",
            project_id=str(project_id),
            previous_version=previous.version if previous else None,
            current_version=current.version,
            decisions_added=decisions_added,
            decisions_dropped=decisions_dropped,
            open_items_added=open_items_added,
            open_items_dropped=open_items_dropped,
            project_summary_similarity=similarity,
        )
    except Exception as exc:
        logger.warning(
            "CAG rebuild drift instrumentation failed",
            project_id=str(project_id),
            error=str(exc),
        )


def _delta_entries(
    previous: list[Any],
    current: list[Any],
) -> tuple[int, int]:
    """Return (added, dropped) counts, matching by ref id or description."""

    def _key(item: Any) -> tuple[object, str]:
        ref = getattr(item, "summary_id_ref", None) or getattr(item, "first_seen_summary_id", None)
        return (ref, getattr(item, "description", "").lower())

    prev_keys = {_key(p) for p in previous}
    curr_keys = {_key(c) for c in current}
    added = sum(1 for c in current if _key(c) not in prev_keys)
    dropped = sum(1 for p in previous if _key(p) not in curr_keys)
    return added, dropped


async def _project_summary_similarity(
    previous_summary: str,
    current_summary: str,
    embedder: Embedder | None,
) -> float | None:
    """Cosine similarity between two project summary strings, or None if no embedder."""
    if embedder is None:
        return None

    previous_vector = embedder.embed_query(previous_summary)
    current_vector = embedder.embed_query(current_summary)

    previous_norm = math.sqrt(sum(v * v for v in previous_vector))
    current_norm = math.sqrt(sum(v * v for v in current_vector))

    if previous_norm == 0 or current_norm == 0:
        return 0.0

    dot = sum(a * b for a, b in zip(previous_vector, current_vector, strict=True))
    return dot / (previous_norm * current_norm)
