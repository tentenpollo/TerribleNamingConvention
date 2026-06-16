from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
import uuid

import pytest

from app.models.document_summary import DocumentSummary
from app.schemas.belief_state import BeliefStateContent
from app.services.cag_prompts import (
    format_batch_digest_prompt,
    format_digest_merge_prompt,
    format_incremental_prompt,
    format_initial_prompt,
)


def _make_summary(summary_id: uuid.UUID, filename: str, text: str) -> DocumentSummary:
    summary = MagicMock(spec=DocumentSummary)
    summary.id = summary_id
    summary.created_at = datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC)
    summary.summary = {"text": text}
    document = MagicMock()
    document.filename = filename
    summary.document = document
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
    assert "filename: meeting-2025-04-01.md" in prompt


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


@pytest.mark.unit
def test_batch_digest_prompt_contains_close_rule_and_conflict_resolution() -> None:
    summary = _make_summary(uuid.uuid4(), "batch-doc.md", "We resolved the open item.")
    prompt = format_batch_digest_prompt([summary])

    assert "Document summaries (in chronological order)" in prompt
    assert "Close (remove) open_items" in prompt
    assert "batch explicitly shows as resolved" in prompt
    assert "LATER in-content date" in prompt
    assert "document filename dates" in prompt
    assert str(summary.id) in prompt
    assert "filename: batch-doc.md" in prompt


@pytest.mark.unit
def test_batch_digest_prompt_requires_non_empty_summary_list() -> None:
    with pytest.raises(ValueError):
        format_batch_digest_prompt([])


@pytest.mark.unit
def test_digest_merge_prompt_contains_conservative_matching_and_close_rule() -> None:
    digest = BeliefStateContent(
        project_summary="A project.",
        decisions=[],
        open_items=[],
        key_people=[],
        recurring_themes=[],
    )
    prompt = format_digest_merge_prompt([digest])

    assert "Intermediate digests (in chronological order)" in prompt
    assert "Match entries by summary_id_ref" in prompt
    assert "case-insensitive description AND approximate_date" in prompt
    assert "Treat entries with different dates" in prompt
    assert "Close (remove) open_items" in prompt
    assert "later digests explicitly show as resolved" in prompt
    assert "LATER digests supersede EARLIER digests" in prompt


@pytest.mark.unit
def test_digest_merge_prompt_requires_non_empty_digest_list() -> None:
    with pytest.raises(ValueError):
        format_digest_merge_prompt([])
