from __future__ import annotations

import litellm

from app.core.exceptions import LLMError
from app.core.logging import logger


async def llm_call(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1000,
) -> str:
    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        content = str(response.choices[0].message.content)
        token_count = response.usage.total_tokens if response.usage else None
        logger.info(
            "LLM call completed",
            model=model,
            token_count=token_count,
        )
        return content
    except Exception as exc:
        raise LLMError(f"LLM call failed for model {model}: {exc}") from exc
