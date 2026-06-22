from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.exceptions import LLMError
from app.core.llm import LLMResult, llm_call
from app.ingestion.summarizer import summarize_document


@pytest.mark.unit
async def test_llm_call_success_returns_content() -> None:
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Hello, world!"
    mock_response.usage = AsyncMock()
    mock_response.usage.prompt_tokens = 7
    mock_response.usage.completion_tokens = 3

    with patch("app.core.llm.litellm.acompletion", return_value=mock_response):
        result = await llm_call(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )

    assert isinstance(result, LLMResult)
    assert result.text == "Hello, world!"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert result.model == "gpt-4o-mini"


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
async def test_llm_call_passes_response_format_and_temperature() -> None:
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "{}"
    mock_response.usage = AsyncMock()
    mock_response.usage.prompt_tokens = 3
    mock_response.usage.completion_tokens = 2

    with patch("app.core.llm.litellm.acompletion", return_value=mock_response) as mock_completion:
        result = await llm_call(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0,
        )

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["temperature"] == 0
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2


@pytest.mark.unit
async def test_llm_call_omits_optional_kwargs_when_none() -> None:
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "hi"
    mock_response.usage = AsyncMock()
    mock_response.usage.prompt_tokens = 1
    mock_response.usage.completion_tokens = 1

    with patch("app.core.llm.litellm.acompletion", return_value=mock_response) as mock_completion:
        result = await llm_call(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )

    call_kwargs = mock_completion.call_args.kwargs
    assert "response_format" not in call_kwargs
    assert "temperature" not in call_kwargs
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 1


@pytest.mark.unit
async def test_summarize_document_valid_json_returns_parsed_dict() -> None:
    expected = {
        "summary": "A technical design document describing the ingestion pipeline.",
        "key_points": [
            "Ingestion uses ARQ workers for async processing",
            "Documents are chunked, embedded, and stored in Qdrant",
            "Summaries are stored in Postgres as an event log",
        ],
        "technical_concepts": ["vector embeddings", "chunking", "async workers"],
        "architectural_components": ["Qdrant", "Postgres", "ARQ"],
        "decisions": [
            {"decision": "Use Qdrant", "reasoning": "Self-hostable and performant"},
        ],
        "action_items": [
            {"task": "Implement CAG rebuild", "owner": "Alice", "status": "open"},
        ],
        "entities": {
            "people": ["Alice"],
            "organizations": [],
            "technologies": ["Qdrant", "Postgres", "ARQ"],
            "repositories": [],
            "services": [],
        },
        "topics": ["ingestion", "rag", "architecture"],
        "important_relationships": [
            {
                "source": "IngestJob",
                "relationship": "writes to",
                "target": "Qdrant",
            },
        ],
        "document_type": "architecture",
        "confidence": 0.9,
    }

    with patch(
        "app.ingestion.summarizer.llm_call",
        return_value=LLMResult(
            text=json.dumps(expected),
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        ),
    ):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result == expected


@pytest.mark.unit
async def test_summarize_document_invalid_json_returns_fallback() -> None:
    with patch(
        "app.ingestion.summarizer.llm_call",
        return_value=LLMResult(
            text="not valid json {{{",
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        ),
    ):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result["raw_text_fallback"] is True
    assert result["key_points"] == []
    assert result["document_type"] == "other"
    assert result["confidence"] == 0.0


@pytest.mark.unit
async def test_summarize_document_llm_error_returns_fallback() -> None:
    with patch("app.ingestion.summarizer.llm_call", side_effect=LLMError("LLM failed")):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result["raw_text_fallback"] is True
    assert result["key_points"] == []
    assert result["document_type"] == "other"
    assert result["confidence"] == 0.0


@pytest.mark.unit
async def test_summarize_document_non_dict_json_returns_fallback() -> None:
    with patch(
        "app.ingestion.summarizer.llm_call",
        return_value=LLMResult(
            text='["not", "a", "dict"]',
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        ),
    ):
        result = await summarize_document(
            text="Some document content",
            filename="test.md",
            model="gpt-4o-mini",
        )

    assert result["raw_text_fallback"] is True
    assert result["key_points"] == []
    assert result["document_type"] == "other"
    assert result["confidence"] == 0.0
