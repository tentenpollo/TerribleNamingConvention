from __future__ import annotations

import json
from typing import Any

from app.models.document_summary import DocumentSummary
from app.schemas.belief_state import BeliefStateContent

_CAP_INSTRUCTIONS = (
    "Strict schema caps (enforce these exactly):\n"
    "- project_summary: required string, max 5000 characters\n"
    "- decisions: required list, max 100 items; each item has:\n"
    "    - description: required string, max 500 characters\n"
    "    - approximate_date: ISO date string (YYYY-MM-DD) or null\n"
    "    - summary_id_ref: the UUID of the summary that introduced this decision, "
    "or null\n"
    "- open_items: required list, max 100 items; each item has:\n"
    "    - description: required string, max 500 characters\n"
    "    - first_seen_summary_id: the UUID of the first summary mentioning this item, "
    "or null\n"
    "- key_people: required list, max 50 items; each item has:\n"
    "    - name: required string, max 120 characters\n"
    "    - role: string or null, max 120 characters\n"
    "- recurring_themes: required list of strings, max 30 items; each string max 80 "
    "characters\n"
    "Do not include any fields not listed above.\n"
)


_INITIAL_HEADER = (
    "You are updating a structured project belief state for a hybrid RAG/CAG "
    "knowledge system.\n\n"
    "Given the following document summaries, produce a single JSON object matching "
    "the exact schema below.\n\n"
)

_INITIAL_FOOTER = (
    "\nEach summary is prefixed with its summary id, document filename, and "
    "created_at timestamp.\n"
    "Use the summary id as summary_id_ref for decisions and "
    "first_seen_summary_id for open_items when applicable.\n"
    "Do not invent information that is not present in the summaries.\n"
    "If the summaries contain no decisions, open items, key people, or recurring "
    "themes, return empty lists for those fields.\n"
    "Return ONLY valid JSON. No markdown, no explanations, no code fences.\n\n"
    "Document summaries:\n"
    "{summaries}\n\n"
    "Output JSON:"
)

INITIAL_GENERATION_PROMPT = _INITIAL_HEADER + _CAP_INSTRUCTIONS + _INITIAL_FOOTER


_INCREMENTAL_HEADER = (
    "You are incrementally updating a structured project belief state for a hybrid "
    "RAG/CAG knowledge system.\n\n"
    "Current belief state JSON:\n"
    "{current_state}\n\n"
    "New document summaries to integrate:\n"
    "{window}\n\n"
    "Each summary is prefixed with its summary id, document filename, and "
    "created_at timestamp.\n\n"
    "Produce a single updated JSON object matching the exact same schema as the "
    "current state.\n\n"
)

_INCREMENTAL_CONFLICT_RULES = (
    "Conflict-resolution rules (apply in this order):\n"
    "1. When new information contradicts an existing decision or open item, prefer "
    "the claim with the LATER in-content date.\n"
    "   Use decision approximate_date or document filename dates when present.\n"
    "   If no in-content date exists, prefer the later event-log entry "
    "(later created_at / newer summary).\n"
    "2. Record the superseding decision and DROP the superseded one; do not keep "
    "both.\n"
    "3. Close (remove) open_items that the new window explicitly shows as resolved.\n"
    "4. Preserve untouched entries verbatim. Do NOT re-paraphrase entries the new "
    "window does not mention.\n"
    "   This is the primary drift-control instruction.\n\n"
    "Use the summary id as summary_id_ref for decisions and "
    "first_seen_summary_id for open_items when applicable.\n"
    "Do not invent information that is not present in the current state or the new "
    "window.\n"
    "Return ONLY valid JSON. No markdown, no explanations, no code fences.\n\n"
    "Output JSON:"
)

INCREMENTAL_UPDATE_PROMPT = (
    _INCREMENTAL_HEADER + _CAP_INSTRUCTIONS + "\n" + _INCREMENTAL_CONFLICT_RULES
)


def _format_summary(summary: DocumentSummary) -> str:
    document = getattr(summary, "document", None)
    filename = document.filename if document is not None else "unknown"
    header = (
        f"--- summary id: {summary.id} | "
        f"filename: {filename} | "
        f"created_at: {summary.created_at.isoformat()} ---"
    )
    body = json.dumps(summary.summary, indent=2, default=str)
    return f"{header}\n{body}"


def format_initial_prompt(summaries: list[DocumentSummary]) -> str:
    if not summaries:
        raise ValueError("At least one summary is required for initial generation")
    formatted = "\n\n".join(_format_summary(s) for s in summaries)
    return INITIAL_GENERATION_PROMPT.format(summaries=formatted)


def format_incremental_prompt(
    current_state: dict[str, Any],
    window: list[DocumentSummary],
) -> str:
    if not window:
        raise ValueError("At least one summary is required for incremental update")
    formatted_window = "\n\n".join(_format_summary(s) for s in window)
    return INCREMENTAL_UPDATE_PROMPT.format(
        current_state=json.dumps(current_state, indent=2, default=str),
        window=formatted_window,
    )


_BATCH_DIGEST_BODY = (
    "You are producing an intermediate digest for a hierarchical project belief-state "
    "rebuild.\n\n"
    "Given the following batch of document summaries, produce a single JSON object matching "
    "the exact schema below. This digest will later be merged with other digests, so it must "
    "be dense and self-contained.\n\n"
    "Rules for this digest:\n"
    "1. Capture all decisions, open items, key people, and recurring themes from the batch.\n"
    "2. Do not invent information not present in the summaries.\n"
    "3. Carry the summary id of the source summary as summary_id_ref for decisions and "
    "first_seen_summary_id for open_items whenever applicable.\n"
    "4. Resolve contradictions inside this batch by preferring the LATER in-content date; "
    "use document filename dates or decision approximate_date when present. "
    "If no in-content date exists, prefer the later event-log entry (later created_at / "
    "newer summary). Drop superseded entries; do not keep both.\n"
    "5. Close (remove) open_items that the batch explicitly shows as resolved.\n"
    "6. Return ONLY valid JSON. No markdown, no explanations, no code fences.\n\n"
    "Document summaries (in chronological order):\n"
    "{summaries}\n\n"
    "Output JSON:"
)

BATCH_DIGEST_PROMPT = _BATCH_DIGEST_BODY + "\n" + _CAP_INSTRUCTIONS


_DIGEST_MERGE_BODY = (
    "You are merging intermediate digests during a hierarchical project belief-state rebuild.\n\n"
    "Given the following chronologically-ordered intermediate digests, produce a single merged "
    "JSON object matching the exact schema below.\n\n"
    "Rules for merging:\n"
    "1. Combine all decisions, open items, key people, and recurring themes from the digests.\n"
    "2. The digests are provided in chronological order (earliest first, latest last). "
    "LATER digests supersede EARLIER digests on conflict.\n"
    "3. Match entries by summary_id_ref when present. Otherwise, match only when both "
    "case-insensitive description AND approximate_date are equal. Treat entries with "
    "different dates or no matching ref as distinct.\n"
    "4. Close (remove) open_items from earlier digests that later digests explicitly show "
    "as resolved.\n"
    "5. Do not invent information not present in the digests.\n"
    "6. Return ONLY valid JSON. No markdown, no explanations, no code fences.\n\n"
    "Intermediate digests (in chronological order):\n"
    "{digests}\n\n"
    "Output JSON:"
)

DIGEST_MERGE_PROMPT = _DIGEST_MERGE_BODY + "\n" + _CAP_INSTRUCTIONS


def format_batch_digest_prompt(summaries: list[DocumentSummary]) -> str:
    if not summaries:
        raise ValueError("At least one summary is required for batch digest")
    formatted = "\n\n".join(_format_summary(s) for s in summaries)
    return BATCH_DIGEST_PROMPT.format(summaries=formatted)


def format_digest_merge_prompt(digests: list[BeliefStateContent]) -> str:
    if not digests:
        raise ValueError("At least one digest is required for merge")
    formatted = "\n\n".join(json.dumps(d.model_dump(mode="json"), indent=2) for d in digests)
    return DIGEST_MERGE_PROMPT.format(digests=formatted)
