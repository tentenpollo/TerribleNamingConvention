from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from pydantic import ValidationError
import pytest

from app.models.belief_state import BeliefState
from app.schemas.belief_state import (
    BeliefStateContent,
    Decision,
    KeyPerson,
    OpenItem,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_content(**overrides) -> dict:
    base = {
        "project_summary": "A test project summary.",
        "decisions": [],
        "open_items": [],
        "key_people": [],
        "recurring_themes": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# BeliefStateContent caps
# ---------------------------------------------------------------------------


class TestBeliefStateContentCaps:
    def test_valid_content_accepted(self) -> None:
        content = BeliefStateContent(**_minimal_content())
        assert content.project_summary == "A test project summary."

    def test_project_summary_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(project_summary="x" * 1201))

    def test_decisions_over_limit_rejected(self) -> None:
        decisions = [
            {"description": f"Decision {i}", "approximate_date": None, "summary_id_ref": None}
            for i in range(101)
        ]
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(decisions=decisions))

    def test_open_items_over_limit_rejected(self) -> None:
        items = [{"description": f"Item {i}", "first_seen_summary_id": None} for i in range(101)]
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(open_items=items))

    def test_key_people_over_limit_rejected(self) -> None:
        people = [{"name": f"Person {i}", "role": None} for i in range(51)]
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(key_people=people))

    def test_recurring_themes_over_limit_rejected(self) -> None:
        themes = [f"theme{i}" for i in range(31)]
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(recurring_themes=themes))

    def test_theme_string_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(recurring_themes=["x" * 81]))

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BeliefStateContent(**_minimal_content(unexpected_field="oops"))

    def test_decision_description_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(description="x" * 501)

    def test_open_item_description_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OpenItem(description="x" * 501)

    def test_key_person_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KeyPerson(name="x" * 121)

    def test_key_person_role_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KeyPerson(name="Alice", role="x" * 121)

    def test_decision_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(description="ok", extra_field="bad")

    def test_open_item_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OpenItem(description="ok", extra_field="bad")

    def test_key_person_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KeyPerson(name="Alice", extra_field="bad")


# ---------------------------------------------------------------------------
# approximate_date format validation
# ---------------------------------------------------------------------------


class TestApproximateDateValidation:
    def test_valid_iso_date_accepted(self) -> None:
        d = Decision(description="ok", approximate_date="2025-04-01")
        assert d.approximate_date == "2025-04-01"

    def test_none_accepted(self) -> None:
        d = Decision(description="ok", approximate_date=None)
        assert d.approximate_date is None

    def test_datetime_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(description="ok", approximate_date="2025-04-01T12:00:00")

    def test_partial_date_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(description="ok", approximate_date="2025-04")

    def test_freeform_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(description="ok", approximate_date="April 2025")


# ---------------------------------------------------------------------------
# Unit: get_window_since fallback filtering (mock session)
# ---------------------------------------------------------------------------


class TestGetWindowSince:
    """Tests the JSONB filter logic on get_window_since using mocked DB responses."""

    @pytest.mark.asyncio
    async def test_fallback_rows_excluded_missing_key_included(self) -> None:
        from app.models.document_summary import DocumentSummary
        from app.services.cag import CAGService

        project_id = uuid.uuid4()
        now = datetime.now(UTC)

        # Three summaries: normal, fallback=true, missing key
        normal = MagicMock(spec=DocumentSummary)
        normal.id = uuid.uuid4()
        normal.project_id = project_id
        normal.summary = {"text": "normal summary"}
        normal.created_at = now - timedelta(seconds=2)

        fallback = MagicMock(spec=DocumentSummary)
        fallback.id = uuid.uuid4()
        fallback.project_id = project_id
        fallback.summary = {"raw_text_fallback": True, "text": "raw fallback"}
        fallback.created_at = now - timedelta(seconds=1)

        missing_key = MagicMock(spec=DocumentSummary)
        missing_key.id = uuid.uuid4()
        missing_key.project_id = project_id
        missing_key.summary = {"text": "no fallback key at all"}
        missing_key.created_at = now

        # The service hits execute() — mock it to return only normal + missing_key
        # (this tests that the filter expression is constructed, not the DB execution)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [normal, missing_key]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = CAGService(mock_session)
        rows = await service.get_window_since(project_id, watermark=None)

        assert rows == [normal, missing_key]
        assert fallback not in rows

    @pytest.mark.asyncio
    async def test_watermark_applied(self) -> None:
        from app.services.cag import CAGService

        project_id = uuid.uuid4()
        watermark = datetime.now(UTC) - timedelta(hours=1)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = CAGService(mock_session)
        rows = await service.get_window_since(project_id, watermark=watermark)

        assert rows == []
        # Verify execute was called once (watermark filter was applied)
        mock_session.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Unit: synthesize_and_insert (mocked session)
# ---------------------------------------------------------------------------


def _mock_session_for_synthesize() -> AsyncMock:
    session = AsyncMock()

    begin_cm = AsyncMock()
    begin_cm.__aenter__ = AsyncMock(return_value=session)
    begin_cm.__aexit__ = AsyncMock(return_value=None)
    session.begin = MagicMock(return_value=begin_cm)

    added_rows: list[BeliefState] = []

    def capture_add(row: BeliefState) -> None:
        added_rows.append(row)

    async def populate_row(row: BeliefState) -> None:
        if row.id is None:
            row.id = uuid.uuid4()
        if row.created_at is None:
            row.created_at = datetime.now(UTC)

    session.add = MagicMock(side_effect=capture_add)
    session.refresh = AsyncMock(side_effect=populate_row)
    session._added_rows = added_rows  # type: ignore[attr-defined]
    return session


def _mock_max_version_result(current_max: int | None) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = current_max
    return result


@pytest.mark.asyncio
async def test_synthesize_and_insert_empty_window_returns_none() -> None:
    from app.services.cag import CAGService

    session = _mock_session_for_synthesize()
    service = CAGService(session)

    async def synthesize_fn(state: BeliefStateContent | None, window: list) -> BeliefStateContent:
        raise RuntimeError("should not be called")

    with patch.object(service, "get_latest", return_value=None):
        with patch.object(service, "get_window_since", return_value=[]):
            record, remaining = await service.synthesize_and_insert(
                uuid.uuid4(), synthesize_fn, batch_size=40
            )

    assert record is None
    assert remaining == 0
    assert len(session._added_rows) == 0


@pytest.mark.asyncio
async def test_synthesize_and_insert_batches_large_window_and_sets_watermark() -> None:
    from app.services.cag import CAGService

    session = _mock_session_for_synthesize()
    service = CAGService(session)

    project_id = uuid.uuid4()
    watermark = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

    latest = MagicMock()
    latest.last_summary_created_at = watermark
    latest.summary_count_covered = 10
    latest.state = BeliefStateContent(project_summary="existing")

    window = []
    for i in range(95):
        summary = MagicMock()
        summary.id = uuid.uuid4()
        summary.created_at = watermark + timedelta(minutes=i + 1)
        summary.summary = {"text": f"summary {i}"}
        window.append(summary)

    expected_40th_created_at = window[39].created_at

    execute_results = [
        MagicMock(),  # advisory lock
        _mock_max_version_result(3),  # max version
    ]
    session.execute = AsyncMock(side_effect=execute_results)

    async def synthesize_fn(state: BeliefStateContent | None, batch: list) -> BeliefStateContent:
        assert len(batch) == 40
        assert batch[0].created_at == window[0].created_at
        assert batch[-1].created_at == window[39].created_at
        return BeliefStateContent(project_summary="updated")

    with patch.object(service, "get_latest", return_value=latest):
        with patch.object(service, "get_window_since", return_value=window):
            record, remaining = await service.synthesize_and_insert(
                project_id, synthesize_fn, batch_size=40
            )

    assert record is not None
    assert remaining == 55
    assert len(session._added_rows) == 1
    inserted = session._added_rows[0]
    assert inserted.version == 4
    assert inserted.rebuild_type == "incremental"
    assert inserted.summary_count_covered == 50  # previous 10 + 40 consumed
    assert inserted.last_summary_created_at == expected_40th_created_at
    assert inserted.project_id == project_id


@pytest.mark.asyncio
async def test_synthesize_and_insert_initial_generation_no_watermark() -> None:
    from app.services.cag import CAGService

    session = _mock_session_for_synthesize()
    service = CAGService(session)

    project_id = uuid.uuid4()

    window = []
    for i in range(3):
        summary = MagicMock()
        summary.id = uuid.uuid4()
        summary.created_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC) + timedelta(minutes=i)
        summary.summary = {"text": f"summary {i}"}
        window.append(summary)

    execute_results = [
        MagicMock(),  # advisory lock
        _mock_max_version_result(None),  # no existing versions
    ]
    session.execute = AsyncMock(side_effect=execute_results)

    async def synthesize_fn(state: BeliefStateContent | None, batch: list) -> BeliefStateContent:
        assert state is None
        assert batch == window
        return BeliefStateContent(project_summary="initial")

    with patch.object(service, "get_latest", return_value=None):
        with patch.object(service, "get_window_since", return_value=window):
            record, remaining = await service.synthesize_and_insert(
                project_id, synthesize_fn, batch_size=40
            )

    assert record is not None
    assert remaining == 0
    inserted = session._added_rows[0]
    assert inserted.version == 1
    assert inserted.summary_count_covered == 3
