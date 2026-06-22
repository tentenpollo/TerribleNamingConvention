from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core.llm import LLMResult, llm_call


@pytest.mark.unit
async def test_llm_call_populates_result_from_usage_block() -> None:
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "answer"
    mock_response.usage = AsyncMock()
    mock_response.usage.prompt_tokens = 12
    mock_response.usage.completion_tokens = 3

    with patch("app.core.llm.litellm.acompletion", return_value=mock_response):
        result = await llm_call(messages=[{"role": "user", "content": "hi"}], model="m")

    assert isinstance(result, LLMResult)
    assert result.text == "answer"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 3
    assert result.model == "m"


@pytest.mark.unit
async def test_llm_call_handles_missing_usage_block() -> None:
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "answer"
    mock_response.usage = None

    with patch("app.core.llm.litellm.acompletion", return_value=mock_response):
        result = await llm_call(messages=[{"role": "user", "content": "hi"}], model="m")

    assert isinstance(result, LLMResult)
    assert result.text == "answer"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
