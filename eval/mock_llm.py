from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from typing import Any
from unittest.mock import patch

# A deterministic, key-free stand-in for ``litellm.acompletion``.
#
# The eval MEASURES real model judgment, so a mock cannot produce a meaningful
# eval RESULT. This mock exists only to prove the harness plumbing end-to-end
# (ingest -> two arms -> judge -> report) without spending tokens or requiring a
# key.  The mock returns DIFFERENTIATED scores keyed on answer content so the
# dry-run report shows a non-degenerate distribution and proves the aggregation
# math.  Numbers produced under the mock are still meaningless — only the
# distribution shape is real.

_SUMMARY_JSON = (
    '{"summary": "mock summary", "key_points": [], "technical_concepts": [], '
    '"architectural_components": [], "decisions": [], "action_items": [], '
    '"entities": {"people": [], "organizations": [], "technologies": [], '
    '"repositories": [], "services": []}, "topics": [], '
    '"important_relationships": [], "document_type": "other", "confidence": 0.5}'
)

_BELIEF_STATE_JSON = (
    '{"project_summary": "Mock belief state for the Drift project.", '
    '"decisions": [], "open_items": [], "key_people": [], "recurring_themes": []}'
)

# The CAG arm receives a belief state → the mock returns a more detailed answer.
# The RAG arm does not → the mock returns a shorter answer.  The mock judge
# later scores the detailed answer higher (by detecting "Detailed" in the text),
# which gives the CAG arm a systematic mock advantage that produces non-trivial
# per-axis means and per-question deltas.
_QUERY_ANSWER_CAG = (
    "[mock] Detailed CAG-style answer. It references multiple sources [S1][S2] "
    "and synthesises a coherent response spanning several sentences. This answer "
    "is deliberately longer and more grounded so the mock judge can distinguish "
    "it from the RAG-style answer by content alone — no arm label is present."
)

_QUERY_ANSWER_RAG = "[mock] Short RAG-style answer."


def _judge_json_for(score_better: int, score_worse: int) -> str:
    return json.dumps(
        {
            "answer1": {
                "factual_correctness": {
                    "score": score_better,
                    "justification": "mock: more detailed answer",
                },
                "grounding": {
                    "score": score_better,
                    "justification": "mock: more detailed answer",
                },
                "conflict_handling": {
                    "score": 3,
                    "justification": "mock: neutral",
                },
            },
            "answer2": {
                "factual_correctness": {
                    "score": score_worse,
                    "justification": "mock: briefer answer",
                },
                "grounding": {
                    "score": score_worse,
                    "justification": "mock: briefer answer",
                },
                "conflict_handling": {
                    "score": 3,
                    "justification": "mock: neutral",
                },
            },
        },
        separators=(",", ":"),
    )


def _extract_answer_texts(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Extract the text of Answer 1 and Answer 2 from the judge user prompt."""
    user = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    a1 = _between(user, "Answer 1:", "Answer 2:")
    a2 = _between(user, "Answer 2:", "Rubric")
    if not a2:
        a2 = user.split("Answer 2:")[-1] if "Answer 2:" in user else ""
    return a1.strip(), a2.strip()


def _between(text: str, start: str, end: str) -> str:
    """Return the substring between start and end markers, or empty."""
    _, _, after_start = text.partition(start)
    if end in after_start:
        return after_start.partition(end)[0]
    return after_start


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage
    model: str = "mock-llm"


@dataclass
class MockCallLog:
    """Records every mock completion call so tests can inspect routing."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.calls.clear()


CALL_LOG = MockCallLog()


def make_mock_acompletion() -> Any:
    """Return an async stand-in for ``litellm.acompletion``.

    The returned callable inspects the message content and returns a canned,
    valid response for each known call site.  Query answers and judge scores
    are DIFFERENTIATED so the dry-run report shows a non-degenerate distribution.
    """

    async def _mock_acompletion(**kwargs: Any) -> _Response:
        messages = kwargs.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        blob = " ".join(m.get("content", "") for m in messages).lower()

        # ---- judge ----
        if "blind evaluator" in blob:
            a1_text, a2_text = _extract_answer_texts(messages)
            a1_detailed = "detailed" in a1_text.lower()
            a2_detailed = "detailed" in a2_text.lower()
            if a1_detailed and not a2_detailed:
                content = _judge_json_for(score_better=4, score_worse=2)
            elif a2_detailed and not a1_detailed:
                content = _judge_json_for(score_better=2, score_worse=4)
            else:
                # Both or neither match; pick a fixed asymmetry
                content = _judge_json_for(score_better=4, score_worse=2)
            kind = "judge"
        # ---- cag synthesis ----
        elif "memory synthesis system" in blob:
            content, kind = _BELIEF_STATE_JSON, "cag_synthesis"
        # ---- summarizer ----
        elif "memory extraction system" in blob:
            content, kind = _SUMMARY_JSON, "summarize"
        # ---- query (differentiate CAG vs RAG by belief-state presence in
        #      the USER prompt only — the system prompt mentions the markers
        #      in its instructions so scanning the full blob is ambiguous.) ----
        elif "you answer questions about a team project" in blob:
            user_blob = " ".join(
                m.get("content", "") for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            ).lower()
            if "<project_state>" in user_blob:
                content, kind = _QUERY_ANSWER_CAG, "query_cag"
            else:
                content, kind = _QUERY_ANSWER_RAG, "query_rag"
        else:
            content, kind = "", "unknown"

        model = str(kwargs.get("model", "mock-llm"))
        joined = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        CALL_LOG.calls.append(
            {
                "kind": kind,
                "model": model,
                "prompt_chars": len(joined),
                "response_chars": len(content),
            }
        )
        return _Response(
            choices=[_Choice(message=_Message(content=content))],
            usage=_Usage(
                prompt_tokens=len(joined) // 4,
                completion_tokens=len(content) // 4,
            ),
            model=model,
        )

    return _mock_acompletion


@contextmanager
def mock_llm_enabled() -> Iterator[None]:
    """Context manager that swaps in the deterministic mock LLM for its scope."""
    CALL_LOG.reset()
    with patch("litellm.acompletion", new=make_mock_acompletion()):
        yield
