from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.exceptions import LLMError
from app.core.llm import llm_call
from app.ingestion.summarizer import summarize_document


@pytest.mark.unit
async def test_llm_call_success_returns_content() -> None:
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Hello, world!"
    mock_response.usage = AsyncMock()
    mock_response.usage.total_tokens = 10

    with patch("app.core.llm.litellm.acompletion", return_value=mock_response):
        result = await llm_call(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )

    assert result == "Hello, world!"


@pytest.mark.unit
async def test_llm_call_failure_raises_llm_error() -> None:
    with patch("app.core.llm.litellm.acompletion", side_effect=Exception("connection refused")):
        with pytest.raises(LLMError) as exc_info:
            await llm_call(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
            )

    assert "LLM call failed" in str(exc_info.value)


@pytest.mark.unit
async def test_summarize_document_valid_json_returns_parsed_dict() -> None:
    expected = {
        "key_points": ["First point", "Second point"],
        "decisions": ["Chose Postgres"],
        "action_items": ["Schedule review"],
        "people_mentioned": ["Alice"],
        "topics": ["architecture", "database"],
    }

    with patch("app.ingestion.summarizer.llm_call", return_value=json.dumps(expected)):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result == expected


@pytest.mark.unit
async def test_summarize_document_invalid_json_returns_fallback() -> None:
    with patch("app.ingestion.summarizer.llm_call", return_value="not valid json {{{"):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result == {"key_points": ["test.md"], "raw_text_fallback": True}


@pytest.mark.unit
async def test_summarize_document_llm_error_returns_fallback() -> None:
    with patch("app.ingestion.summarizer.llm_call", side_effect=LLMError("LLM failed")):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result == {"key_points": ["test.md"], "raw_text_fallback": True}
