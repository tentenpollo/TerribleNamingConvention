from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
import uuid

import pytest

from app.models.document_summary import DocumentSummary
from app.services.cag_prompts import (
    format_incremental_prompt,
    format_initial_prompt,
)


def _make_summary(summary_id: uuid.UUID, filename: str, text: str) -> DocumentSummary:
    summary = MagicMock(spec=DocumentSummary)
    summary.id = summary_id
    summary.created_at = datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC)
    summary.summary = {"text": text, "filename": filename}
    return summary


@pytest.mark.unit
def test_initial_prompt_contains_schema_and_cap_instructions() -> None:
    summary = _make_summary(uuid.uuid4(), "meeting-2025-04-01.md", "We decided to use Qdrant.")
    prompt = format_initial_prompt([summary])

    assert "decisions" in prompt
    assert "open_items" in prompt
    assert "key_people" in prompt
    assert "recurring_themes" in prompt
    assert "max 100 items" in prompt
    assert "max 50 items" in prompt
    assert "max 30 items" in prompt
    assert "summary_id_ref" in prompt
    assert str(summary.id) in prompt


@pytest.mark.unit
def test_initial_prompt_requires_non_empty_summary_list() -> None:
    with pytest.raises(ValueError):
        format_initial_prompt([])


@pytest.mark.unit
def test_incremental_prompt_contains_conflict_resolution_instructions() -> None:
    current_state = {
        "project_summary": "A project.",
        "decisions": [],
        "open_items": [],
        "key_people": [],
        "recurring_themes": [],
    }
    summary = _make_summary(uuid.uuid4(), "update.md", "New decision.")
    prompt = format_incremental_prompt(current_state, [summary])

    assert "Current belief state JSON" in prompt
    assert '"project_summary": "A project."' in prompt
    assert "LATER in-content date" in prompt
    assert "DROP the superseded one" in prompt
    assert "Preserve untouched entries verbatim" in prompt
    assert "do not re-paraphrase" in prompt.lower()


@pytest.mark.unit
def test_incremental_prompt_requires_non_empty_window() -> None:
    with pytest.raises(ValueError):
        format_incremental_prompt({"project_summary": "x"}, [])
