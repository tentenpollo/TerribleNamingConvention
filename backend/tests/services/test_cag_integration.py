from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text

from app.core import database
from app.core.exceptions import BeliefStateVersionConflictError, InvalidBeliefStateError
from app.schemas.belief_state import BeliefStateContent
from app.services.cag import CAGService


def _sample_content() -> BeliefStateContent:
    return BeliefStateContent(
        project_summary="Integration test project summary.",
        decisions=[],
        open_items=[],
        key_people=[],
        recurring_themes=[],
    )


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def cag_project() -> dict:
    """Creates a team + project for CAG integration tests, cleans up after."""
    team_id = uuid.uuid4()
    project_id = uuid.uuid4()

    async with database.AsyncSessionLocal() as session:
        await session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "CAG Test Team"},
        )
        await session.execute(
            text(
                "INSERT INTO projects (id, name, team_id, config) "
                "VALUES (:id, :name, :team_id, CAST(:config AS jsonb))"
            ),
            {
                "id": project_id,
                "name": "CAG Test Project",
                "team_id": team_id,
                "config": "{}",
            },
        )
        await session.commit()

    yield {"project_id": project_id, "team_id": team_id}

    async with database.AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("DELETE FROM belief_states WHERE project_id = :pid"),
                {"pid": project_id},
            )
            await session.execute(
                text("DELETE FROM projects WHERE id = :id"),
                {"id": project_id},
            )
            await session.execute(
                text("DELETE FROM teams WHERE id = :id"),
                {"id": team_id},
            )
            await session.commit()
        finally:
            pass


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_insert_version_no_gaps_no_duplicates(cag_project) -> None:
    """20 concurrent tasks write version N+1 via separate sessions.
    All succeed; final versions are exactly 1..20 with no gaps/duplicates."""
    project_id = cag_project["project_id"]
    content = _sample_content()
    watermark = datetime.now(UTC)
    conflict_errors: list[BeliefStateVersionConflictError] = []

    async def write_one() -> int | None:
        async with database.AsyncSessionLocal() as session:
            service = CAGService(session)
            try:
                record = await service.insert_version(
                    project_id=project_id,
                    content=content,
                    rebuild_type="incremental",
                    last_summary_created_at=watermark,
                    summary_count_covered=1,
                )
                return record.version
            except BeliefStateVersionConflictError as exc:
                conflict_errors.append(exc)
                return None

    versions = await asyncio.gather(*[write_one() for _ in range(20)])

    assert not conflict_errors, f"Got {len(conflict_errors)} conflict errors"
    successful = sorted(v for v in versions if v is not None)
    assert successful == list(range(1, 21)), f"Expected 1..20, got {successful}"


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_get_latest_corrupted_state_raises_invalid(cag_project) -> None:
    """A row with an invalid state JSONB must raise InvalidBeliefStateError on get_latest."""
    project_id = cag_project["project_id"]

    async with database.AsyncSessionLocal() as session:
        await session.execute(
            text(
                "INSERT INTO belief_states "
                "(project_id, version, state, rebuild_type, "
                "last_summary_created_at, summary_count_covered) "
                "VALUES (:pid, 1, CAST(:state AS jsonb), 'full', now(), 0)"
            ),
            {"pid": project_id, "state": '{"not_valid": "shape"}'},
        )
        await session.commit()

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        with pytest.raises(InvalidBeliefStateError):
            await service.get_latest(project_id)


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_insert_and_read_roundtrip(cag_project) -> None:
    project_id = cag_project["project_id"]
    content = _sample_content()
    watermark = datetime.now(UTC)

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        record = await service.insert_version(
            project_id=project_id,
            content=content,
            rebuild_type="full",
            last_summary_created_at=watermark,
            summary_count_covered=5,
        )

    assert record.version == 1
    assert record.rebuild_type == "full"
    assert record.summary_count_covered == 5
    assert record.state.project_summary == content.project_summary

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        latest = await service.get_latest(project_id)

    assert latest is not None
    assert latest.version == 1


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_get_window_since_excludes_raw_text_fallback(cag_project) -> None:
    """get_window_since must exclude rows where summary->>'raw_text_fallback' is true,
    but include rows where the key is absent or the value is false."""
    project_id = cag_project["project_id"]

    # Three document_summaries rows:
    #   normal        — no raw_text_fallback key  → INCLUDED
    #   raw_fallback  — raw_text_fallback: true    → EXCLUDED
    #   explicit_false — raw_text_fallback: false  → INCLUDED
    async with database.AsyncSessionLocal() as session:
        for summary_json in [
            '{"text": "normal"}',
            '{"text": "raw", "raw_text_fallback": true}',
            '{"text": "explicit_false", "raw_text_fallback": false}',
        ]:
            doc_id = uuid.uuid4()
            await session.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, filename, file_type, raw_bytes) "
                    "VALUES (:id, :pid, :fn, 'txt', '')"
                ),
                {"id": doc_id, "pid": project_id, "fn": f"doc_{doc_id}.txt"},
            )
            await session.execute(
                text(
                    "INSERT INTO document_summaries "
                    "(id, document_id, project_id, summary) "
                    "VALUES (:id, :doc_id, :pid, CAST(:summary AS jsonb))"
                ),
                {
                    "id": uuid.uuid4(),
                    "doc_id": doc_id,
                    "pid": project_id,
                    "summary": summary_json,
                },
            )
        await session.commit()

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        rows = await service.get_window_since(project_id, watermark=None)

    texts = [r.summary["text"] for r in rows]
    assert "raw" not in texts, "raw_text_fallback:true row must be excluded"
    assert "normal" in texts
    assert "explicit_false" in texts
    # order is created_at ASC — just confirm both are present in any order
    assert len(texts) == 2
