from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArmKind = Literal["cag", "rag"]
ExpectedBehavior = Literal["factual", "supersession_resolved", "conflict_surfaced"]
Axis = Literal["factual_correctness", "grounding", "conflict_handling"]

AXES: tuple[Axis, ...] = ("factual_correctness", "grounding", "conflict_handling")


class EvalInvariantError(Exception):
    """Raised when a controlled-variable invariant of the eval is violated."""


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question_text: str
    reference_answer: str
    expected_behavior: ExpectedBehavior
    grounding: list[str]


class FixtureDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    text: str


class AxisScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    justification: str


class ArmJudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factual_correctness: AxisScore
    grounding: AxisScore
    conflict_handling: AxisScore


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer1: ArmJudgeScores
    answer2: ArmJudgeScores


class JudgeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer1_arm: ArmKind
    answer2_arm: ArmKind
    scores: JudgeScores
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChunkRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_index: int
    filename: str


class ArmResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    chunks: list[ChunkRef]
    belief_state_version: int | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int


class QuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    reference_answer: str
    expected_behavior: ExpectedBehavior
    grounding: list[str]
    cag: ArmResult
    rag: ArmResult
    retrieval_match: bool
    judge: JudgeOutcome | None = None


class AxisCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_abs_diff: float
    exact_agreement: float
    within_one_agreement: float
    n: int


class CalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trustworthy: bool
    warning: str | None
    per_axis: dict[str, AxisCalibration]
    n: int


class RunMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project_id: str
    created_at: str
    generation_model: str
    judge_model: str
    mock_llm: bool


class RunResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMeta
    questions: list[QuestionResult]
    calibration: CalibrationResult | None = None


class HumanGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    arm: ArmKind
    factual_correctness: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    conflict_handling: int = Field(ge=1, le=5)


class JudgeGrade(BaseModel):
    """Unblinded judge score for one (question, arm) pair, used by calibration."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    arm: ArmKind
    factual_correctness: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    conflict_handling: int = Field(ge=1, le=5)
