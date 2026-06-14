from __future__ import annotations

import litellm

from app.core.exceptions import LLMError
from app.core.logging import logger


async def llm_call(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1000,
    response_format: dict[str, str] | None = None,
    temperature: float | None = None,
) -> str:
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
        token_count = response.usage.total_tokens if response.usage else None
        logger.info(
            "LLM call completed",
            model=model,
            token_count=token_count,
            response_format="json_object" if response_format else None,
        )
        return content
    except Exception as exc:
        raise LLMError(f"LLM call failed for model {model}: {exc}") from exc
