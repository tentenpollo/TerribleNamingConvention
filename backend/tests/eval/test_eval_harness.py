from __future__ import annotations

from datetime import UTC, datetime
import json
import random
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.llm import LLMResult
from app.models.project import Project
from app.retrieval.retriever import RetrievedChunk
from app.schemas.belief_state import BeliefStateContent, BeliefStateRecord
from app.services.query import QueryService
from eval.calibrate import compute_calibration
from eval.judge import build_judge_messages, judge_question, parse_judge_json
from eval.report import aggregate
from eval.run_eval import run_eval, run_two_arms, teardown
from eval.types import (
    ArmJudgeScores,
    ArmResult,
    AxisScore,
    HumanGrade,
    JudgeGrade,
    JudgeOutcome,
    JudgeScores,
    Question,
    QuestionResult,
)


def _question(behavior: str = "factual") -> Question:
    return Question(
        id="q1",
        question_text="What is Drift's sync interval?",
        reference_answer="ref",
        expected_behavior=behavior,  # type: ignore[arg-type]
        grounding=["05_sync-interval-5min", "06_sync-interval-60sec"],
    )


_VALID_JUDGE_JSON = json.dumps(
    {
        "answer1": {
            "factual_correctness": {"score": 4, "justification": "j1"},
            "grounding": {"score": 3, "justification": "j1"},
            "conflict_handling": {"score": 5, "justification": "j1"},
        },
        "answer2": {
            "factual_correctness": {"score": 3, "justification": "j2"},
            "grounding": {"score": 3, "justification": "j2"},
            "conflict_handling": {"score": 2, "justification": "j2"},
        },
    }
)


async def _fake_llm_valid(**kwargs: object) -> LLMResult:
    return LLMResult(text=_VALID_JUDGE_JSON, prompt_tokens=7, completion_tokens=9, model="mock")


# ---------------------------------------------------------------------------
# 1. Two arms: identical retrieved chunk ids + belief-state presence differs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_two_arms_identical_chunks_and_belief_versions() -> None:
    project_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunks = [
        RetrievedChunk(
            document_id=doc_id,
            chunk_index=0,
            text="t0",
            filename="f.md",
            score=1.0,
            project_id=project_id,
        ),
        RetrievedChunk(
            document_id=doc_id,
            chunk_index=1,
            text="t1",
            filename="f.md",
            score=0.9,
            project_id=project_id,
        ),
    ]

    project = Project(name="eval", team_id=uuid.uuid4(), config={})
    project.id = project_id

    record = BeliefStateRecord(
        id=uuid.uuid4(),
        project_id=project_id,
        version=1,
        rebuild_type="full",
        last_summary_created_at=datetime.now(UTC),
        summary_count_covered=1,
        created_at=datetime.now(UTC),
        state=BeliefStateContent(
            project_summary="s",
            decisions=[],
            open_items=[],
            key_people=[],
            recurring_themes=[],
        ),
    )

    session = MagicMock()
    session.get = AsyncMock(return_value=project)
    cag_service = MagicMock()
    cag_service.get_latest = AsyncMock(return_value=record)

    qs = QueryService(session, MagicMock(), MagicMock(), MagicMock(), cag_service)

    with patch("app.services.query.retrieve", new=AsyncMock(return_value=chunks)):
        llm_result = LLMResult(text="A", prompt_tokens=1, completion_tokens=2, model="m")
        with patch("app.services.query.llm_call", new=AsyncMock(return_value=llm_result)):
            cag_arm, rag_arm, match = await run_two_arms(
                qs, _question(), project_id, [project_id], uuid.uuid4(), 8
            )

    assert match is True
    assert cag_arm.belief_state_version == 1
    assert rag_arm.belief_state_version is None
    cag_keys = [(c.document_id, c.chunk_index) for c in cag_arm.chunks]
    rag_keys = [(c.document_id, c.chunk_index) for c in rag_arm.chunks]
    assert cag_keys == rag_keys
    assert len(cag_keys) == 2


@pytest.mark.unit
async def test_prompt_diff_only_belief_state_block() -> None:
    """The CAG and RAG arms' prompts must differ ONLY in the belief-state region.

    The CAG arm gets ``<project_state>...</project_state>``; the RAG arm gets
    ``(no project state available)``.  Every other character in the user prompt
    (sources block, question text, blank lines) must be byte-identical.  This
    is the actual controlled-variable guarantee.
    """
    from app.retrieval.prompting import build_user_prompt as _real_build

    project_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunks = [
        RetrievedChunk(
            document_id=doc_id, chunk_index=0, text="chunk text 0",
            filename="f.md", score=1.0, project_id=project_id,
        ),
        RetrievedChunk(
            document_id=doc_id, chunk_index=1, text="chunk text 1",
            filename="f.md", score=0.9, project_id=project_id,
        ),
    ]

    project = Project(name="eval", team_id=uuid.uuid4(), config={})
    project.id = project_id
    record = BeliefStateRecord(
        id=uuid.uuid4(), project_id=project_id, version=1, rebuild_type="full",
        last_summary_created_at=datetime.now(UTC), summary_count_covered=1,
        created_at=datetime.now(UTC),
        state=BeliefStateContent(
            project_summary="s", decisions=[], open_items=[],
            key_people=[], recurring_themes=[],
        ),
    )

    captured: list[dict[str, object]] = []

    def _capture_build_user_prompt(
        belief_state: object, ch: object, question: object,
        **kw: object,
    ) -> str:
        prompt = _real_build(belief_state, ch, question, **kw)  # type: ignore[arg-type]
        captured.append({
            "belief_state": belief_state,
            "chunks": ch,
            "question": question,
            "prompt": prompt,
        })
        return prompt

    session = MagicMock()
    session.get = AsyncMock(return_value=project)
    cag_service = MagicMock()
    cag_service.get_latest = AsyncMock(return_value=record)
    qs = QueryService(session, MagicMock(), MagicMock(), MagicMock(), cag_service)
    question = _question()

    with patch("app.services.query.retrieve", new=AsyncMock(return_value=chunks)):
        llm_result = LLMResult(text="A", prompt_tokens=1, completion_tokens=2, model="m")
        with patch("app.services.query.llm_call", new=AsyncMock(return_value=llm_result)):
            with patch(
                "app.services.query.build_user_prompt",
                new=_capture_build_user_prompt,
            ):
                _cag_arm, _rag_arm, match = await run_two_arms(
                    qs, question, project_id, [project_id], uuid.uuid4(), 8
                )

    assert match is True
    assert len(captured) == 2
    cag_call, rag_call = captured

    # 1. CAG got belief_state; RAG got None
    assert cag_call["belief_state"] is not None
    assert rag_call["belief_state"] is None

    # 2. Same chunks object and same question
    assert cag_call["chunks"] is rag_call["chunks"]  # same retrieval
    assert cag_call["question"] == rag_call["question"]

    cag_prompt: str = str(cag_call["prompt"])
    rag_prompt: str = str(rag_call["prompt"])

    # 3. The non-belief-state portions must be byte-identical.
    #    CAG has <project_state>...</project_state>; RAG has (no project state
    #    available).  After excising those header regions the rest must match.
    if "<project_state>" in cag_prompt and "</project_state>" in cag_prompt:
        _, _, after_cag = cag_prompt.partition("</project_state>")
    else:
        raise AssertionError("CAG prompt missing </project_state> marker")

    rag_tag = "(no project state available)"
    assert rag_tag in rag_prompt, f"RAG prompt missing '{rag_tag}'"
    _, _, after_rag = rag_prompt.partition(rag_tag)

    assert after_cag == after_rag, (
        f"Prompts differ beyond the belief-state block.\n"
        f"---CAG rest---\n{after_cag!r}\n---RAG rest---\n{after_rag!r}"
    )


# ---------------------------------------------------------------------------
# 2. Judge blinding: no arm labels in the payload; mapping randomized + recoverable.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_judge_payload_is_blind() -> None:
    messages = build_judge_messages(
        _question("conflict_surfaced"), "answer-A", "answer-B", ["src one", "src two"]
    )
    blob = json.dumps(messages)
    assert "CAG" not in blob
    assert "RAG" not in blob
    assert "Answer 1" in blob
    assert "Answer 2" in blob
    # the conflict rule is keyed on behavior, not on arm identity
    assert "conflict_surfaced" not in blob.lower() or "both sides" in blob.lower()


@pytest.mark.unit
async def test_judge_mapping_randomized_and_recoverable() -> None:
    seen_answer1: set[str] = set()
    for seed in range(20):
        outcome = await judge_question(
            _question(),
            "cag-answer",
            "rag-answer",
            ["src"],
            "mock",
            random.Random(seed),
            llm=_fake_llm_valid,
        )
        # mapping is a valid, complete assignment of the two arms
        assert {outcome.answer1_arm, outcome.answer2_arm} == {"cag", "rag"}
        assert outcome.answer1_arm != outcome.answer2_arm
        seen_answer1.add(outcome.answer1_arm)
    # randomization actually flips across seeds (both orders observed)
    assert seen_answer1 == {"cag", "rag"}
    # recoverability: which real answer was "Answer 1" is recorded
    outcome = await judge_question(
        _question(),
        "cag-answer",
        "rag-answer",
        ["src"],
        "mock",
        random.Random(0),
        llm=_fake_llm_valid,
    )
    expected_a1 = "cag-answer" if outcome.answer1_arm == "cag" else "rag-answer"
    assert expected_a1 in ("cag-answer", "rag-answer")


# ---------------------------------------------------------------------------
# 3. Judge JSON validation: Pydantic parse, one retry, then raise.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_judge_json_valid_and_invalid() -> None:
    scores = parse_judge_json(_VALID_JUDGE_JSON)
    assert scores.answer1.factual_correctness.score == 4
    assert scores.answer2.conflict_handling.score == 2
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_judge_json("not json")
    with pytest.raises(ValueError):
        parse_judge_json('{"answer1": {}}')


@pytest.mark.unit
async def test_judge_retries_once_then_succeeds() -> None:
    state = {"n": 0}

    async def flaky(**kwargs: object) -> LLMResult:
        state["n"] += 1
        text = "not json" if state["n"] == 1 else _VALID_JUDGE_JSON
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="mock")

    outcome = await judge_question(
        _question(), "a1", "a2", ["src"], "mock", random.Random(0), llm=flaky
    )
    assert state["n"] == 2
    assert outcome.scores.answer1.factual_correctness.score == 4


@pytest.mark.unit
async def test_judge_raises_after_retry_failure() -> None:
    async def always_bad(**kwargs: object) -> LLMResult:
        return LLMResult(text="still not json", prompt_tokens=1, completion_tokens=1, model="mock")

    with pytest.raises(ValueError, match="failed validation"):
        await judge_question(
            _question(), "a1", "a2", ["src"], "mock", random.Random(0), llm=always_bad
        )


# ---------------------------------------------------------------------------
# 4. Report aggregation math on synthetic scored results.
# ---------------------------------------------------------------------------


def _axis(score: int) -> AxisScore:
    return AxisScore(score=score, justification="j")


def _arm_scores(fc: int, g: int, ch: int) -> ArmJudgeScores:
    return ArmJudgeScores(
        factual_correctness=_axis(fc), grounding=_axis(g), conflict_handling=_axis(ch)
    )


def _arm(answer_text: str = "x") -> ArmResult:
    return ArmResult(answer=answer_text, chunks=[], belief_state_version=None, latency_ms=1)


def _qr(
    qid: str,
    answer1_arm: str,
    cag_fc: int,
    rag_fc: int,
    behavior: str = "factual",
) -> QuestionResult:
    if answer1_arm == "cag":
        a1, a2 = _arm_scores(cag_fc, 3, 3), _arm_scores(rag_fc, 3, 3)
    else:
        a1, a2 = _arm_scores(rag_fc, 3, 3), _arm_scores(cag_fc, 3, 3)
    judge = JudgeOutcome(
        answer1_arm=answer1_arm,  # type: ignore[arg-type]
        answer2_arm="rag" if answer1_arm == "cag" else "cag",
        scores=JudgeScores(answer1=a1, answer2=a2),
    )
    return QuestionResult(
        question_id=qid,
        question_text="question " + qid,
        reference_answer="r",
        expected_behavior=behavior,  # type: ignore[arg-type]
        grounding=[],
        cag=_arm("CAG answer for " + qid),
        rag=_arm("RAG answer for " + qid),
        retrieval_match=True,
        judge=judge,
    )


def _qr_unjudged(qid: str, behavior: str = "factual") -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question_text="question " + qid,
        reference_answer="r",
        expected_behavior=behavior,  # type: ignore[arg-type]
        grounding=[],
        cag=_arm("CAG answer for " + qid),
        rag=_arm("RAG answer for " + qid),
        retrieval_match=True,
        judge=None,
    )


@pytest.mark.unit
def test_report_aggregation_known_means_and_deltas() -> None:
    results = [
        _qr("q1", "cag", 5, 1),           # CAG wins big (+4), CAG=answer1
        _qr("q2", "rag", 2, 5),            # RAG wins big (-3), RAG=answer1
        _qr("q3", "cag", 4, 3),            # CAG wins small (+1)
        _qr("q5", "cag", 3, 4, "conflict_surfaced"),  # RAG wins small (-1), conflict
        _qr_unjudged("q4"),                # unjudged — excluded
    ]
    data = aggregate(results)

    # ---- per-axis means and deltas ----
    fc = data.axis_aggs["factual_correctness"]
    assert fc.n == 4  # unjudged excluded
    # CAG: 5,2,4,3 → 14/4=3.5   RAG: 1,5,3,4 → 13/4=3.25
    assert fc.cag_mean == 3.5
    assert fc.rag_mean == 3.25
    assert fc.delta_mean == 0.25
    assert fc.per_question_deltas == [4.0, -3.0, 1.0, -1.0]
    assert fc.cag_better == 2   # q1(+4), q3(+1)
    assert fc.rag_better == 2   # q2(-3), q5(-1)
    assert fc.tie == 0

    # Grounding constant 3 both arms → zero deltas, all ties
    gr = data.axis_aggs["grounding"]
    assert gr.delta_mean == 0.0
    assert gr.cag_better == 0 and gr.rag_better == 0 and gr.tie == 4

    # ---- total questions ----
    assert data.n_questions == 5  # includes unjudged
    assert data.n_judged == 4

    # ---- most_divergent (by abs delta) ----
    assert len(data.most_divergent) == 3
    assert data.most_divergent[0].question_id == "q1"  # |+4|
    assert data.most_divergent[1].question_id == "q2"  # |-3|
    # q3=+1, q5=-1 both abs=1 → tie, stable order matters less
    assert abs(data.most_divergent[2].delta_total) == 1.0

    # ---- conflict_views ----
    assert len(data.conflict_views) == 1
    cv = data.conflict_views[0]
    assert cv.question_id == "q5"
    assert "CAG answer for q5" in cv.cag_answer
    assert "RAG answer for q5" in cv.rag_answer
    # RAG won this one → rag_scores.factual_correctness.score == 4
    assert cv.rag_scores.factual_correctness.score == 4
    assert cv.cag_scores.factual_correctness.score == 3


# ---------------------------------------------------------------------------
# 5. Calibration: warning fires when diverged, silent when agreed.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_calibration_warns_when_diverged() -> None:
    judge = [
        JudgeGrade(
            question_id="q1",
            arm="cag",
            factual_correctness=5,
            grounding=5,
            conflict_handling=5,
        ),
        JudgeGrade(
            question_id="q1",
            arm="rag",
            factual_correctness=1,
            grounding=5,
            conflict_handling=5,
        ),
    ]
    human = [
        HumanGrade(
            question_id="q1",
            arm="cag",
            factual_correctness=1,
            grounding=5,
            conflict_handling=5,
        ),
        HumanGrade(
            question_id="q1",
            arm="rag",
            factual_correctness=1,
            grounding=5,
            conflict_handling=5,
        ),
    ]
    result = compute_calibration(judge, human)
    assert result.trustworthy is False
    assert result.warning is not None
    # factual_correctness diffs: |5-1|=4, |1-1|=0 -> mean 2.0 > 1.0
    assert result.per_axis["factual_correctness"].mean_abs_diff == 2.0


@pytest.mark.unit
def test_calibration_silent_when_agreed() -> None:
    judge = [
        JudgeGrade(
            question_id="q1",
            arm="cag",
            factual_correctness=4,
            grounding=3,
            conflict_handling=5,
        ),
        JudgeGrade(
            question_id="q1",
            arm="rag",
            factual_correctness=3,
            grounding=4,
            conflict_handling=3,
        ),
    ]
    human = [
        HumanGrade(
            question_id="q1",
            arm="cag",
            factual_correctness=4,
            grounding=3,
            conflict_handling=5,
        ),
        HumanGrade(
            question_id="q1",
            arm="rag",
            factual_correctness=3,
            grounding=4,
            conflict_handling=3,
        ),
    ]
    result = compute_calibration(judge, human)
    assert result.trustworthy is True
    assert result.warning is None
    assert result.per_axis["factual_correctness"].exact_agreement == 1.0
    assert result.per_axis["factual_correctness"].mean_abs_diff == 0.0


# ---------------------------------------------------------------------------
# 6. Teardown: deletes Qdrant + DB rows, and runs in finally even mid-run failure.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_teardown_deletes_qdrant_and_db_rows() -> None:
    session = AsyncMock()
    vector_store = AsyncMock()
    pid = uuid.uuid4()
    tid = uuid.uuid4()
    await teardown(session, vector_store, pid, tid)
    vector_store.delete_collection.assert_awaited_once_with(pid)
    # 6 explicit DELETE statements (belief_state, summary, job, document, project, team)
    assert session.execute.await_count == 6
    session.commit.assert_awaited_once()


class _FakeSessionCtx:
    def __init__(self, session: AsyncMock) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncMock:
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSessionFactory:
    def __init__(self, session: AsyncMock) -> None:
        self._session = session

    def __call__(self) -> _FakeSessionCtx:
        return _FakeSessionCtx(self._session)


@pytest.mark.unit
async def test_run_eval_tears_down_when_ingest_raises_midway() -> None:
    pid = uuid.uuid4()
    tid = uuid.uuid4()
    fake_project = MagicMock()
    fake_project.id = pid
    fake_team = MagicMock()
    fake_team.id = tid

    session = AsyncMock()
    factory = _FakeSessionFactory(session)
    vs = AsyncMock()
    embedder = AsyncMock()
    sparse = AsyncMock()

    teardown_mock = AsyncMock()

    with patch(
        "eval.run_eval.setup_project",
        new=AsyncMock(return_value=(fake_project, fake_team)),
    ):
        with patch(
            "eval.run_eval.ingest_fixtures",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            with patch("eval.run_eval.run_cag_rebuild", new=AsyncMock()):
                with patch("eval.run_eval.run_arms_and_judge", new=AsyncMock(return_value=[])):
                    with patch("eval.run_eval.teardown", new=teardown_mock):
                        with pytest.raises(RuntimeError, match="boom"):
                            await run_eval(
                                questions=[],
                                corpus=[],
                                session_factory=factory,  # type: ignore[arg-type]
                                vector_store=vs,
                                embedder=embedder,  # type: ignore[arg-type]
                                sparse_embedder=sparse,  # type: ignore[arg-type]
                                generation_model="m",
                                context_model="m",
                                judge_model="m",
                            )

    teardown_mock.assert_awaited_once()
    # project_id is the 3rd positional arg of teardown(session, vector_store, project_id, team_id)
    assert teardown_mock.call_args.args[2] == pid
