from __future__ import annotations

from dataclasses import dataclass

import litellm

from app.core.exceptions import LLMError
from app.core.logging import logger


@dataclass
class LLMResult:
    """Result of a single LLM completion call."""

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    model: str


async def llm_call(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1000,
    response_format: dict[str, str] | None = None,
    temperature: float | None = None,
) -> LLMResult:
    completion_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        completion_kwargs["response_format"] = response_format
    if temperature is not None:
        completion_kwargs["temperature"] = temperature

    try:
        response = await litellm.acompletion(**completion_kwargs)
        content = str(response.choices[0].message.content)
        prompt_tokens = None
        completion_tokens = None
        if response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        logger.info(
            "LLM call completed",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_format="json_object" if response_format else None,
        )
        return LLMResult(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )
    except Exception as exc:
        raise LLMError(f"LLM call failed for model {model}: {exc}") from exc
