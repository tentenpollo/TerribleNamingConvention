from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
import random

from app.core.llm import LLMResult, llm_call
from app.core.logging import logger
from eval.types import (
    ArmKind,
    ExpectedBehavior,
    JudgeOutcome,
    JudgeScores,
    Question,
)

_JUDGE_SYSTEM = (
    "You are an impartial blind evaluator comparing two answers to the same "
    "question. You are NOT told how either answer was produced. Judge each answer "
    "on its own merits against the reference answer and the provided source "
    "documents. Return strict JSON only."
)

_RUBRIC_INTRO = (
    "Score EACH answer independently on three axes, 1-5, with a one-line "
    "justification per score:\n"
    "- factual_correctness: agreement with the reference answer and the sources; "
    "5 = fully correct, 1 = mostly wrong.\n"
    "- grounding: are claims attributable to the provided sources and cited; does "
    "it avoid inventing facts not in the sources; 5 = fully grounded, 1 = "
    "fabricated.\n"
    "- conflict_handling: see the question-specific rule below."
)

_CONFLICT_RULES: dict[ExpectedBehavior, str] = {
    "factual": ("conflict_handling is N/A for this question type: score 3 for both answers."),
    "supersession_resolved": (
        "conflict_handling: 5 if the answer reflects the LATER decision and does "
        "not assert the superseded one as current; lower if it asserts the "
        "superseded decision or treats both as equally current."
    ),
    "conflict_surfaced": (
        "conflict_handling: 5 if the answer surfaces BOTH sides of the "
        "disagreement and cites both, without fabricating consensus; lower if it "
        "silently picks one side or asserts a single value as settled."
    ),
}

_JSON_SHAPE = (
    "Return JSON shaped exactly as: "
    '{"answer1": {"factual_correctness": {"score": <1-5>, "justification": "..."}, '
    '"grounding": {"score": <1-5>, "justification": "..."}, '
    '"conflict_handling": {"score": <1-5>, "justification": "..."}}, '
    '"answer2": {"factual_correctness": {...}, "grounding": {...}, '
    '"conflict_handling": {...}}}.'
)

LLMFn = Callable[..., Awaitable[LLMResult]]


def build_judge_messages(
    question: Question,
    answer1: str,
    answer2: str,
    source_texts: list[str],
) -> list[dict[str, str]]:
    """Build the blind judge payload.

    The payload labels the two answers only as "Answer 1" and "Answer 2". It must
    not contain the arm labels "CAG" or "RAG", nor any indication of how an
    answer was produced.
    """
    doc_blocks = "\n\n".join(f"[Doc {i}] {text}" for i, text in enumerate(source_texts, start=1))
    user = "\n\n".join(
        [
            f"Question: {question.question_text}",
            f"Reference answer (ground truth): {question.reference_answer}",
            f"Source documents:\n{doc_blocks}",
            f"Answer 1:\n{answer1}",
            f"Answer 2:\n{answer2}",
            _RUBRIC_INTRO,
            _CONFLICT_RULES[question.expected_behavior],
            _JSON_SHAPE,
        ]
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_judge_json(text: str) -> JudgeScores:
    """Validate judge output against the JudgeScores schema; raise on failure."""
    parsed = json.loads(text)
    return JudgeScores.model_validate(parsed)


async def _call_and_parse(
    messages: list[dict[str, str]],
    judge_model: str,
    llm: LLMFn,
) -> JudgeScores:
    """Call the judge LLM and parse JSON, retrying once on a parse failure."""
    last_error = ""
    for attempt in range(2):
        current = list(messages)
        if attempt == 1:
            current = [
                *current,
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON matching the "
                        f"required shape. Error: {last_error}. Return ONLY the "
                        "JSON object, no prose, no code fences."
                    ),
                },
            ]
        result = await llm(
            messages=current,
            model=judge_model,
            max_tokens=8000,
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            return parse_judge_json(result.text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(
                "Judge JSON parse failed; retrying",
                attempt=attempt,
                error=last_error,
            )
    raise ValueError(f"Judge output failed validation after retry: {last_error}")


async def judge_question(
    question: Question,
    cag_answer: str,
    rag_answer: str,
    source_texts: list[str],
    judge_model: str,
    rng: random.Random,
    llm: LLMFn = llm_call,
) -> JudgeOutcome:
    """Blind-judge the two arm answers for one question.

    Arm assignment to "Answer 1"/"Answer 2" is randomized per question using
    ``rng``; the mapping is returned in the JudgeOutcome so results can be
    unblinded later. The judge payload never references CAG or RAG.
    """
    if rng.random() < 0.5:
        answer1_arm: ArmKind = "cag"
        answer2_arm: ArmKind = "rag"
        answer1, answer2 = cag_answer, rag_answer
    else:
        answer1_arm = "rag"
        answer2_arm = "cag"
        answer1, answer2 = rag_answer, cag_answer

    messages = build_judge_messages(question, answer1, answer2, source_texts)
    scores = await _call_and_parse(messages, judge_model, llm)
    return JudgeOutcome(
        answer1_arm=answer1_arm,
        answer2_arm=answer2_arm,
        scores=scores,
    )
