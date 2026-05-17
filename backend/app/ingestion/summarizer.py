from __future__ import annotations

import json

from app.core.exceptions import LLMError
from app.core.llm import llm_call
from app.core.logging import logger

_SUMMARY_PROMPT = (
    "You are a document summarization assistant. Analyze the following document "
    "and return a JSON object with these fields:\n\n"
    "- key_points: list of up to 5 main points\n"
    "- decisions: list of any decisions mentioned (may be empty)\n"
    "- action_items: list of any action items or tasks (may be empty)\n"
    "- people_mentioned: list of names of people mentioned (may be empty)\n"
    "- topics: list of up to 5 main topics covered (short tags)\n\n"
    "Return ONLY valid JSON. No markdown, no explanation, no code fences.\n\n"
    "Document filename: {filename}\n\n"
    "Document content:\n{text}\n"
)


async def summarize_document(
    text: str,
    filename: str,
    model: str,
) -> dict[str, object]:
    fallback: dict[str, object] = {
        "key_points": [filename],
        "raw_text_fallback": True,
    }

    try:
        messages = [
            {
                "role": "system",
                "content": ("You are a document summarization assistant. Return only valid JSON."),
            },
            {
                "role": "user",
                "content": _SUMMARY_PROMPT.format(filename=filename, text=text),
            },
        ]
        response_text = await llm_call(messages=messages, model=model, max_tokens=1000)
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return parsed
        logger.error(
            "LLM response was not a JSON object",
            filename=filename,
            response_type=type(parsed).__name__,
        )
        return fallback
    except LLMError as exc:
        logger.error(
            "LLM call failed during summarization, returning fallback",
            filename=filename,
            error=str(exc),
        )
        return fallback
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse LLM response as JSON, returning fallback",
            filename=filename,
            error=str(exc),
        )
        return fallback
