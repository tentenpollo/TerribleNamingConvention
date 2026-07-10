from __future__ import annotations

import json

from app.core.exceptions import LLMError
from app.core.llm import llm_call
from app.core.logging import logger

_SUMMARY_PROMPT = (
    "You are an expert software project memory extraction system.\n\n"
    "Your task is to analyze a document and extract durable, retrieval-optimized "
    "knowledge for a long-term AI memory engine.\n\n"
    "The output will be stored inside a Context-Augmented Generation (CAG) system "
    "that combines vector retrieval (RAG), evolving project belief states, semantic "
    "memory, and architectural context tracking.\n\n"
    "Focus on preserving:\n"
    "- technical intent\n"
    "- architectural decisions\n"
    "- implementation details\n"
    "- APIs, models, systems, workflows\n"
    "- constraints and assumptions\n"
    "- important entities and concepts\n\n"
    "Avoid:\n"
    "- fluff, repetition, vague summaries\n"
    "- generic phrasing\n"
    "- speculative information\n\n"
    "Rules:\n"
    "- Return ONLY valid JSON\n"
    "- No markdown, no explanations, no code fences\n"
    "- Every field must exist — use empty arrays when no data exists\n"
    "- Do not invent information not present in the document\n"
    "- Keep summaries dense and information-rich\n\n"
    "Required JSON schema:\n"
    "{{\n"
    '  "summary": "Concise high-signal overview of the document",\n'
    '  "key_points": ["..."],\n'
    '  "technical_concepts": ["..."],\n'
    '  "architectural_components": ["..."],\n'
    '  "decisions": [{{"decision": "...", "reasoning": "..."}}],\n'
    '  "action_items": [\n'
    '    {{"task": "...", "owner": "...", "status": "open|in_progress|done|unknown"}}\n'
    "  ],\n"
    '  "entities": {{\n'
    '    "people": ["..."], "organizations": ["..."], "technologies": ["..."],\n'
    '    "repositories": ["..."], "services": ["..."]\n'
    "  }},\n"
    '  "topics": ["..."],\n'
    '  "important_relationships": '
    '[{{"source": "...", "relationship": "...", "target": "..."}}],\n'
    '  "document_type": "meeting_notes|architecture|specification|code|'
    'design_doc|research|other",\n'
    '  "confidence": 0.0\n'
    "}}\n\n"
    "Field requirements:\n"
    "- key_points: max 8 items\n"
    "- technical_concepts: libraries, algorithms, protocols, infrastructure, patterns\n"
    "- architectural_components: services, databases, APIs, pipelines, agents, subsystems\n"
    "- topics: short retrieval-friendly tags\n"
    "- confidence: float between 0 and 1 representing extraction confidence\n\n"
    "Document filename: {filename}\n\n"
    "Document content:\n{text}\n"
)

_FALLBACK: dict[str, object] = {
    "summary": "Failed to summarize document",
    "key_points": [],
    "technical_concepts": [],
    "architectural_components": [],
    "decisions": [],
    "action_items": [],
    "entities": {
        "people": [],
        "organizations": [],
        "technologies": [],
        "repositories": [],
        "services": [],
    },
    "topics": [],
    "important_relationships": [],
    "document_type": "other",
    "confidence": 0.0,
    "raw_text_fallback": True,
}


async def summarize_document(
    text: str,
    filename: str,
    model: str,
) -> dict[str, object]:
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a project memory extraction system. "
                    "Return only valid JSON matching the requested schema."
                ),
            },
            {
                "role": "user",
                "content": _SUMMARY_PROMPT.format(filename=filename, text=text),
            },
        ]
        llm_result = await llm_call(messages=messages, model=model, max_tokens=8000)
        parsed = json.loads(llm_result.text)
        if isinstance(parsed, dict):
            return parsed
        logger.error(
            "LLM response was not a JSON object",
            filename=filename,
            response_type=type(parsed).__name__,
        )
        return _FALLBACK
    except LLMError as exc:
        logger.error(
            "LLM call failed during summarization, returning fallback",
            filename=filename,
            error=str(exc),
        )
        return _FALLBACK
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse LLM response as JSON, returning fallback",
            filename=filename,
            error=str(exc),
        )
        return _FALLBACK
