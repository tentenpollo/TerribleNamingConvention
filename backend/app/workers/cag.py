from __future__ import annotations

import json
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
from app.models.project import Project
from app.schemas.belief_state import BeliefStateContent
from app.services.cag import CAGService
from app.services.cag_prompts import format_incremental_prompt, format_initial_prompt


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
            return await _synthesize(
                current_state=current_state,
                window=window,
                project_id=project_id,
                context_model=context_model,
            )

        record, remaining = await service.synthesize_and_insert(
            project_id=project_id,
            synthesize_fn=synthesize_fn,
            batch_size=40,
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


async def _fetch_project(session: AsyncSession, project_id: uuid.UUID) -> Project:
    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    return project


async def _synthesize(
    current_state: BeliefStateContent | None,
    window: list[Any],
    project_id: uuid.UUID,
    context_model: str,
) -> BeliefStateContent:
    """Call the LLM to synthesize a belief state, with one retry on validation failure."""
    base_prompt = (
        format_initial_prompt(window)
        if current_state is None
        else format_incremental_prompt(current_state.model_dump(), window)
    )
    errors: list[str] = []

    for _attempt in range(2):
        prompt = base_prompt
        if errors:
            prompt += "\n\nYour previous output violated:\n" + "\n".join(errors)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a structured project memory synthesis system. "
                    "Return only valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = await llm_call(
                messages=messages,
                model=context_model,
                max_tokens=4000,
                response_format={"type": "json_object"},
                temperature=0,
            )
        except LLMError:
            raise

        try:
            parsed = json.loads(response_text)
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
