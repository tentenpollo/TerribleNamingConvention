from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.exceptions import LLMError
from app.core.llm import LLMResult
from app.models.document_summary import DocumentSummary
from app.models.project import Project
from app.schemas.belief_state import BeliefStateContent, Decision, OpenItem
from app.workers.cag import (
    _delta_entries,
    _emit_rebuild_drift,
    _hierarchical_reduce,
    _project_summary_similarity,
    cag_rebuild,
    cag_update,
)


def _valid_belief_state_json() -> str:
    return (
        '{"project_summary": "A project summary.", '
        '"decisions": [], "open_items": [], '
        '"key_people": [], "recurring_themes": []}'
    )


def _llm_result(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        prompt_tokens=10,
        completion_tokens=5,
        model="gpt-4o-mini",
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
                with patch(
                    "app.workers.cag.llm_call",
                    return_value=_llm_result(_valid_belief_state_json()),
                ):
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
                    return_value=_llm_result("not valid json {{{"),
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

    async def fake_llm_call(*args: object, **kwargs: object) -> LLMResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _llm_result("not valid json {{{")
        return _llm_result(_valid_belief_state_json())

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
                with patch(
                    "app.workers.cag.llm_call",
                    return_value=_llm_result(_valid_belief_state_json()),
                ):
                    await cag_update(ctx, project.id)

    mock_arq.enqueue_job.assert_awaited_once()
    call_args = mock_arq.enqueue_job.call_args
    assert call_args.args[0] == "cag_update"
    assert call_args.args[1] == project.id
    assert call_args.kwargs.get("_job_id") == f"cag-update-{project.id}"


# ---------------------------------------------------------------------------
# Helpers for rebuild tests
# ---------------------------------------------------------------------------


def _make_summaries(count: int) -> list[MagicMock]:
    base_time = datetime.now(UTC)
    summaries: list[MagicMock] = []
    for i in range(count):
        summary = MagicMock(spec=DocumentSummary)
        summary.id = uuid.uuid4()
        summary.created_at = base_time + timedelta(seconds=i)
        summary.summary = {"text": f"summary {i}"}
        summaries.append(summary)
    return summaries


# ---------------------------------------------------------------------------
# Hierarchical reduce arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hierarchical_reduce_500_summaries_makes_14_calls() -> None:
    summaries = _make_summaries(500)
    call_count = 0

    async def fake_llm(*args: object, **kwargs: object) -> LLMResult:
        nonlocal call_count
        call_count += 1
        return _llm_result(_valid_belief_state_json())

    with patch("app.workers.cag.llm_call", side_effect=fake_llm):
        result = await _hierarchical_reduce(
            accumulator=None,
            summaries=summaries,
            project_id=uuid.uuid4(),
            context_model="gpt-4o-mini",
            batch_size=40,
        )

    assert call_count == 14
    assert isinstance(result, BeliefStateContent)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hierarchical_reduce_95_summaries_makes_4_calls() -> None:
    summaries = _make_summaries(95)
    call_count = 0

    async def fake_llm(*args: object, **kwargs: object) -> LLMResult:
        nonlocal call_count
        call_count += 1
        return _llm_result(_valid_belief_state_json())

    with patch("app.workers.cag.llm_call", side_effect=fake_llm):
        result = await _hierarchical_reduce(
            accumulator=None,
            summaries=summaries,
            project_id=uuid.uuid4(),
            context_model="gpt-4o-mini",
            batch_size=40,
        )

    assert call_count == 4
    assert isinstance(result, BeliefStateContent)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hierarchical_reduce_30_summaries_makes_1_call() -> None:
    summaries = _make_summaries(30)
    call_count = 0

    async def fake_llm(*args: object, **kwargs: object) -> LLMResult:
        nonlocal call_count
        call_count += 1
        return _llm_result(_valid_belief_state_json())

    with patch("app.workers.cag.llm_call", side_effect=fake_llm):
        result = await _hierarchical_reduce(
            accumulator=None,
            summaries=summaries,
            project_id=uuid.uuid4(),
            context_model="gpt-4o-mini",
            batch_size=40,
        )

    assert call_count == 1
    assert isinstance(result, BeliefStateContent)


# ---------------------------------------------------------------------------
# Compaction mode
# ---------------------------------------------------------------------------


def _fake_cag_service_class_for_compaction(
    latest_full_record: MagicMock | None,
    summaries_after_watermark: list[MagicMock],
    all_summaries: list[MagicMock],
) -> type:
    class FakeCAGService:
        def __init__(self, session: object) -> None:
            self.session = session
            self.rebuild_call: dict[str, object] | None = None

        async def get_latest_full(self, project_id: uuid.UUID) -> MagicMock | None:
            return latest_full_record

        async def get_window_since(
            self,
            project_id: uuid.UUID,
            watermark: datetime | None,
        ) -> list[MagicMock]:
            return summaries_after_watermark

        async def get_all_summaries(self, project_id: uuid.UUID) -> list[MagicMock]:
            return all_summaries

        async def rebuild(
            self,
            project_id: uuid.UUID,
            rebuild_fn: object,
            mode: str,
            batch_size: int = 40,
        ) -> tuple[MagicMock, MagicMock | None]:
            self.rebuild_call = {
                "project_id": project_id,
                "mode": mode,
                "batch_size": batch_size,
            }
            if mode == "compaction" and latest_full_record is not None:
                accumulator = latest_full_record.state
                summaries = summaries_after_watermark
            else:
                accumulator = None
                summaries = all_summaries
            content = await rebuild_fn(accumulator, summaries)
            record = MagicMock()
            record.version = 2
            record.summary_count_covered = len(summaries)
            record.state = content
            return record, None

    return FakeCAGService


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cag_rebuild_compaction_with_full_state_uses_accumulator() -> None:
    project = _make_fake_project()
    accumulator = BeliefStateContent(project_summary="existing full state")
    latest_full = MagicMock()
    latest_full.state = accumulator
    latest_full.last_summary_created_at = datetime.now(UTC)
    latest_full.summary_count_covered = 60

    after_watermark = _make_summaries(30)
    all_summaries = _make_summaries(90)

    fake_service_class = _fake_cag_service_class_for_compaction(
        latest_full_record=latest_full,
        summaries_after_watermark=after_watermark,
        all_summaries=all_summaries,
    )
    mock_arq = AsyncMock()
    ctx: dict[str, object] = {"redis": mock_arq}

    received: dict[str, object] = {}

    async def capture_rebuild_fn(
        accumulator: BeliefStateContent | None,
        summaries: list[MagicMock],
        **kwargs: object,
    ) -> BeliefStateContent:
        received["accumulator"] = accumulator
        received["summaries"] = summaries
        return BeliefStateContent(project_summary="rebuilt")

    with patch("app.workers.cag._fetch_project", return_value=project):
        with patch("app.workers.cag.AsyncSessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session_local.return_value = mock_session
            with patch("app.workers.cag.CAGService", fake_service_class):
                with patch(
                    "app.workers.cag._hierarchical_reduce",
                    side_effect=capture_rebuild_fn,
                ):
                    await cag_rebuild(ctx, project.id, "compaction")

    assert received["accumulator"] is accumulator
    assert received["summaries"] == after_watermark


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cag_rebuild_compaction_without_full_state_degrades_to_genesis() -> None:
    project = _make_fake_project()
    after_watermark = _make_summaries(30)
    all_summaries = _make_summaries(90)

    fake_service_class = _fake_cag_service_class_for_compaction(
        latest_full_record=None,
        summaries_after_watermark=after_watermark,
        all_summaries=all_summaries,
    )
    mock_arq = AsyncMock()
    ctx: dict[str, object] = {"redis": mock_arq}

    received: dict[str, object] = {}

    async def capture_rebuild_fn(
        accumulator: BeliefStateContent | None,
        summaries: list[MagicMock],
        **kwargs: object,
    ) -> BeliefStateContent:
        received["accumulator"] = accumulator
        received["summaries"] = summaries
        return BeliefStateContent(project_summary="rebuilt")

    with patch("app.workers.cag._fetch_project", return_value=project):
        with patch("app.workers.cag.AsyncSessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session_local.return_value = mock_session
            with patch("app.workers.cag.CAGService", fake_service_class):
                with patch(
                    "app.workers.cag._hierarchical_reduce",
                    side_effect=capture_rebuild_fn,
                ):
                    await cag_rebuild(ctx, project.id, "compaction")

    assert received["accumulator"] is None
    assert received["summaries"] == all_summaries


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cag_rebuild_invalid_mode_raises() -> None:
    ctx: dict[str, object] = {"redis": AsyncMock()}
    with pytest.raises(ValueError, match="Invalid rebuild mode"):
        await cag_rebuild(ctx, uuid.uuid4(), "invalid")


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delta_entries_matches_by_reference_then_description() -> None:
    ref_a = uuid.uuid4()
    ref_b = uuid.uuid4()
    ref_c = uuid.uuid4()

    previous = [
        Decision(description="Decision A", summary_id_ref=ref_a),
        Decision(description="Decision B", summary_id_ref=ref_b),
        Decision(description="Decision C", summary_id_ref=None),
    ]
    current = [
        Decision(description="decision a", summary_id_ref=ref_a),  # same ref, changed desc
        Decision(description="Decision D", summary_id_ref=ref_c),
        Decision(description="decision c", summary_id_ref=None),  # same desc
    ]

    added, dropped = _delta_entries(previous, current)
    assert added == 1
    assert dropped == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_project_summary_similarity_normalized() -> None:
    class FakeEmbedder:
        def embed_query(self, text: str) -> list[float]:
            if text == "summary one":
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

    similarity = await _project_summary_similarity(
        "summary one",
        "summary two",
        FakeEmbedder(),
    )
    assert similarity == 0.0

    similarity = await _project_summary_similarity(
        "summary one",
        "summary one",
        FakeEmbedder(),
    )
    assert similarity == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_emit_rebuild_drift_logs_metrics() -> None:
    previous = MagicMock()
    previous.version = 1
    previous.state = BeliefStateContent(
        project_summary="previous summary",
        decisions=[Decision(description="Old decision", summary_id_ref=uuid.uuid4())],
        open_items=[OpenItem(description="Old item", first_seen_summary_id=None)],
    )

    current = MagicMock()
    current.version = 2
    current.state = BeliefStateContent(
        project_summary="current summary",
        decisions=[Decision(description="New decision", summary_id_ref=uuid.uuid4())],
        open_items=[OpenItem(description="Old item", first_seen_summary_id=None)],
    )

    class FakeEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    with patch("app.workers.cag.logger") as mock_logger:
        await _emit_rebuild_drift(
            project_id=uuid.uuid4(),
            previous=previous,
            current=current,
            embedder=FakeEmbedder(),
        )

    mock_logger.info.assert_called_once()
    call_kwargs = mock_logger.info.call_args.kwargs
    assert call_kwargs["decisions_added"] == 1
    assert call_kwargs["decisions_dropped"] == 1
    assert call_kwargs["open_items_added"] == 0
    assert call_kwargs["open_items_dropped"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_emit_rebuild_drift_failure_does_not_raise() -> None:
    previous = MagicMock()
    previous.version = 1
    previous.state = BeliefStateContent(project_summary="previous summary")

    current = MagicMock()
    current.version = 2
    current.state = BeliefStateContent(project_summary="current summary")

    class BrokenEmbedder:
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("embedder failed")

    with patch("app.workers.cag.logger") as mock_logger:
        await _emit_rebuild_drift(
            project_id=uuid.uuid4(),
            previous=previous,
            current=current,
            embedder=BrokenEmbedder(),
        )

    mock_logger.warning.assert_called_once()
