from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.exceptions import LLMError
from app.models.document_summary import DocumentSummary
from app.models.project import Project
from app.workers.cag import cag_update


def _valid_belief_state_json() -> str:
    return (
        '{"project_summary": "A project summary.", '
        '"decisions": [], "open_items": [], '
        '"key_people": [], "recurring_themes": []}'
    )


def _make_fake_project() -> Project:
    project = MagicMock(spec=Project)
    project.id = uuid.uuid4()
    project.config = {}
    return project


def _make_fake_cag_service_class(
    inserted_record: MagicMock | None,
    remaining: int = 0,
) -> type:
    class FakeCAGService:
        def __init__(self, session: object) -> None:
            self.session = session
            self.synthesize_fn: object = None
            self.call_project_id: uuid.UUID | None = None
            self.call_batch_size: int | None = None

        async def synthesize_and_insert(
            self,
            project_id: uuid.UUID,
            synthesize_fn: object,
            batch_size: int = 40,
        ) -> tuple[MagicMock | None, int]:
            self.synthesize_fn = synthesize_fn
            self.call_project_id = project_id
            self.call_batch_size = batch_size

            window = [
                MagicMock(spec=DocumentSummary),
                MagicMock(spec=DocumentSummary),
            ]
            await synthesize_fn(None, window)
            return inserted_record, remaining

    return FakeCAGService


@pytest.mark.unit
async def test_cag_update_valid_json_inserts_version() -> None:
    project = _make_fake_project()
    record = MagicMock()
    record.version = 1
    record.summary_count_covered = 2
    record.last_summary_created_at = datetime.now(UTC)

    fake_service_class = _make_fake_cag_service_class(record, remaining=0)
    mock_arq = AsyncMock()
    mock_arq.enqueue_job.return_value = object()
    ctx: dict[str, object] = {"redis": mock_arq}

    with patch("app.workers.cag._fetch_project", return_value=project):
        with patch("app.workers.cag.AsyncSessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session_local.return_value = mock_session
            with patch("app.workers.cag.CAGService", fake_service_class):
                with patch("app.workers.cag.llm_call", return_value=_valid_belief_state_json()):
                    await cag_update(ctx, project.id)

    mock_arq.enqueue_job.assert_not_awaited()


@pytest.mark.unit
async def test_cag_update_invalid_json_twice_raises_no_insert() -> None:
    project = _make_fake_project()
    fake_service_class = _make_fake_cag_service_class(None, remaining=0)
    mock_arq = AsyncMock()
    ctx: dict[str, object] = {"redis": mock_arq}

    with patch("app.workers.cag._fetch_project", return_value=project):
        with patch("app.workers.cag.AsyncSessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session_local.return_value = mock_session
            with patch("app.workers.cag.CAGService", fake_service_class):
                with patch(
                    "app.workers.cag.llm_call",
                    return_value="not valid json {{{",
                ):
                    with pytest.raises(LLMError):
                        await cag_update(ctx, project.id)

    mock_arq.enqueue_job.assert_not_awaited()


@pytest.mark.unit
async def test_cag_update_invalid_then_valid_succeeds_with_retry() -> None:
    project = _make_fake_project()
    record = MagicMock()
    record.version = 1
    record.summary_count_covered = 2
    record.last_summary_created_at = datetime.now(UTC)

    fake_service_class = _make_fake_cag_service_class(record, remaining=0)
    mock_arq = AsyncMock()
    ctx: dict[str, object] = {"redis": mock_arq}

    call_count = 0

    async def fake_llm_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "not valid json {{{"
        return _valid_belief_state_json()

    with patch("app.workers.cag._fetch_project", return_value=project):
        with patch("app.workers.cag.AsyncSessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session_local.return_value = mock_session
            with patch("app.workers.cag.CAGService", fake_service_class):
                with patch("app.workers.cag.llm_call", side_effect=fake_llm_call):
                    await cag_update(ctx, project.id)

    assert call_count == 2


@pytest.mark.unit
async def test_cag_update_remaining_summaries_re_enqueues_with_fixed_job_id() -> None:
    project = _make_fake_project()
    record = MagicMock()
    record.version = 1
    record.summary_count_covered = 40
    record.last_summary_created_at = datetime.now(UTC)

    fake_service_class = _make_fake_cag_service_class(record, remaining=15)
    mock_arq = AsyncMock()
    mock_arq.enqueue_job.return_value = object()
    ctx: dict[str, object] = {"redis": mock_arq}

    with patch("app.workers.cag._fetch_project", return_value=project):
        with patch("app.workers.cag.AsyncSessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session_local.return_value = mock_session
            with patch("app.workers.cag.CAGService", fake_service_class):
                with patch("app.workers.cag.llm_call", return_value=_valid_belief_state_json()):
                    await cag_update(ctx, project.id)

    mock_arq.enqueue_job.assert_awaited_once()
    call_args = mock_arq.enqueue_job.call_args
    assert call_args.args[0] == "cag_update"
    assert call_args.args[1] == project.id
    assert call_args.kwargs.get("_job_id") == f"cag-update-{project.id}"
