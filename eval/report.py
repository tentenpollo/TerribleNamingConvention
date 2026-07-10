from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from eval.types import (
    AXES,
    ArmJudgeScores,
    CalibrationResult,
    EvalInvariantError,
    QuestionResult,
    RunMeta,
)


class AxisAgg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cag_mean: float
    rag_mean: float
    delta_mean: float
    per_question_deltas: list[float]
    cag_better: int
    rag_better: int
    tie: int
    n: int


class QuestionDivergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    expected_behavior: str
    cag_total: float
    rag_total: float
    delta_total: float


class ConflictQuestionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    cag_answer: str
    rag_answer: str
    cag_scores: ArmJudgeScores
    rag_scores: ArmJudgeScores


class ReportData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis_aggs: dict[str, AxisAgg]
    most_divergent: list[QuestionDivergence]
    conflict_views: list[ConflictQuestionView]
    n_questions: int
    n_judged: int


def _unblind_axis(qr: QuestionResult, axis: str) -> tuple[int, int]:
    """Return (cag_score, rag_score) for one axis on a judged question."""
    if qr.judge is None:
        raise EvalInvariantError(f"Question {qr.question_id} has no judge outcome")
    j = qr.judge
    a1 = getattr(j.scores.answer1, axis).score
    a2 = getattr(j.scores.answer2, axis).score
    if j.answer1_arm == "cag":
        return a1, a2
    return a2, a1


def _unblind_arm_scores(qr: QuestionResult, arm: Literal["cag", "rag"]) -> ArmJudgeScores:
    if qr.judge is None:
        raise EvalInvariantError(f"Question {qr.question_id} has no judge outcome")
    j = qr.judge
    if j.answer1_arm == arm:
        return j.scores.answer1
    return j.scores.answer2


def aggregate(results: list[QuestionResult]) -> ReportData:
    """Unblind judge scores and aggregate per-axis means and per-question deltas."""
    judged = [r for r in results if r.judge is not None]

    axis_aggs: dict[str, AxisAgg] = {}
    for axis in AXES:
        pairs: list[tuple[int, int]] = [_unblind_axis(qr, axis) for qr in judged]
        deltas: list[float] = [float(c - r) for c, r in pairs]
        n = len(pairs)
        cag_mean = sum(c for c, _ in pairs) / n if n else 0.0
        rag_mean = sum(r for _, r in pairs) / n if n else 0.0
        axis_aggs[axis] = AxisAgg(
            cag_mean=cag_mean,
            rag_mean=rag_mean,
            delta_mean=(sum(deltas) / n) if n else 0.0,
            per_question_deltas=deltas,
            cag_better=sum(1 for d in deltas if d > 0),
            rag_better=sum(1 for d in deltas if d < 0),
            tie=sum(1 for d in deltas if d == 0),
            n=n,
        )

    divergences: list[QuestionDivergence] = []
    for qr in judged:
        cag_total = sum(_unblind_axis(qr, axis)[0] for axis in AXES)
        rag_total = sum(_unblind_axis(qr, axis)[1] for axis in AXES)
        divergences.append(
            QuestionDivergence(
                question_id=qr.question_id,
                expected_behavior=qr.expected_behavior,
                cag_total=cag_total,
                rag_total=rag_total,
                delta_total=cag_total - rag_total,
            )
        )
    divergences.sort(key=lambda d: abs(d.delta_total), reverse=True)

    conflict_views: list[ConflictQuestionView] = []
    for qr in judged:
        if qr.expected_behavior != "conflict_surfaced":
            continue
        conflict_views.append(
            ConflictQuestionView(
                question_id=qr.question_id,
                question_text=qr.question_text,
                cag_answer=qr.cag.answer,
                rag_answer=qr.rag.answer,
                cag_scores=_unblind_arm_scores(qr, "cag"),
                rag_scores=_unblind_arm_scores(qr, "rag"),
            )
        )

    return ReportData(
        axis_aggs=axis_aggs,
        most_divergent=divergences[:3],
        conflict_views=conflict_views,
        n_questions=len(results),
        n_judged=len(judged),
    )


def _fmt_delta(d: float) -> str:
    return f"{d:+.2f}"


def render_report(
    data: ReportData,
    calibration: CalibrationResult | None,
    meta: RunMeta,
) -> str:
    """Render the aggregated results as a markdown report answering Q1/Q2/Q3."""
    lines: list[str] = []
    lines.append(f"# CAG vs RAG Evaluation Report — {meta.run_id}")
    lines.append("")
    lines.append(f"- Generated: {meta.created_at}")
    lines.append(f"- Generation model: `{meta.generation_model}`")
    lines.append(f"- Judge model: `{meta.judge_model}`")
    lines.append(f"- Mock LLM: {meta.mock_llm}")
    lines.append(f"- Questions: {data.n_questions} (judged: {data.n_judged})")
    lines.append("")

    lines.append("## Calibration")
    if calibration is None:
        lines.append(
            "> No calibration run. Judge trustworthiness is NOT established. "
            "Run `python -m eval.calibrate` with human grades before interpreting."
        )
    elif calibration.n == 0:
        lines.append(
            "> Calibration scaffold not yet filled in (n=0). Judge trustworthiness "
            "is NOT established. Fill in eval/fixtures/human_grades.yaml for ~5 "
            "questions before interpreting the full run."
        )
    elif calibration.trustworthy:
        lines.append(f"> Judge scores look trustworthy (n={calibration.n}).")
        for axis in AXES:
            c = calibration.per_axis.get(axis)
            if c and c.n:
                lines.append(
                    f"> - {axis}: mean_abs_diff={c.mean_abs_diff:.2f} "
                    f"exact={c.exact_agreement:.2f} within_one={c.within_one_agreement:.2f}"
                )
    else:
        lines.append(f"> ⚠️ **{calibration.warning}**")
    lines.append("")

    lines.append("## Q1 — Does belief-state injection beat plain RAG on answer quality?")
    lines.append("")
    if data.n_judged == 0:
        lines.append("> No judged questions; cannot compare arms.")
    else:
        lines.append("| Axis | CAG mean | RAG mean | Δ (CAG-RAG) | CAG better | RAG better | tie |")
        lines.append("|---|---|---|---|---|---|---|")
        for axis in AXES:
            a = data.axis_aggs[axis]
            lines.append(
                f"| {axis} | {a.cag_mean:.2f} | {a.rag_mean:.2f} | {_fmt_delta(a.delta_mean)} "
                f"| {a.cag_better} | {a.rag_better} | {a.tie} |"
            )
        lines.append("")
        lines.append("Per-question Δ distribution (CAG-RAG, factual_correctness):")
        fc = data.axis_aggs["factual_correctness"]
        lines.append("  " + ", ".join(_fmt_delta(d) for d in fc.per_question_deltas))
        lines.append("")
        lines.append(
            "> Note: this is **descriptive, not a significance test**. n is small "
            f"({data.n_judged}); a mean hides whether CAG helps some questions and "
            "hurts others — inspect the per-question Δ distribution above."
        )
    lines.append("")

    lines.append("## Q2 — Does injection change response style/consistency?")
    lines.append("")
    if not data.most_divergent:
        lines.append("> No judged questions to compare.")
    else:
        lines.append("Questions where the two arms diverged most in total score:")
        lines.append("")
        for d in data.most_divergent:
            lines.append(
                f"- `{d.question_id}` ({d.expected_behavior}): "
                f"CAG total {d.cag_total:.0f}, RAG total {d.rag_total:.0f}, "
                f"Δ {_fmt_delta(d.delta_total)} — review both answers for style drift."
            )
        lines.append("")
        lines.append(
            "> Inspect these answers by hand for stylistic or consistency "
            "differences between the CAG and RAG arms."
        )
    lines.append("")

    lines.append("## Q3 — Does the system fabricate false consensus, or hedge correctly?")
    lines.append("")
    if not data.conflict_views:
        lines.append("> No conflict_surfaced questions in this run.")
    else:
        for cv in data.conflict_views:
            lines.append(f"### {cv.question_id}")
            lines.append(f"Question: {cv.question_text}")
            lines.append("")
            lines.append(
                f"- CAG conflict_handling: {cv.cag_scores.conflict_handling.score}/5 "
                f"— {cv.cag_scores.conflict_handling.justification}"
            )
            lines.append(
                f"- RAG conflict_handling: {cv.rag_scores.conflict_handling.score}/5 "
                f"— {cv.rag_scores.conflict_handling.justification}"
            )
            lines.append("")
            lines.append("**CAG answer (verbatim):**")
            lines.append("")
            lines.append(f"> {cv.cag_answer}")
            lines.append("")
            lines.append("**RAG answer (verbatim):**")
            lines.append("")
            lines.append(f"> {cv.rag_answer}")
            lines.append("")
        lines.append(
            "> Did either arm surface the disagreement (cite both sides) rather "
            "than fabricate consensus? Judge conflict_handling and the verbatim "
            "answers above answer Q3 directly."
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "**POC-grade:** these results are proof-of-concept grade. n is small, the "
        "corpus is synthetic, and (under the mock) the numbers are meaningless. "
        "Do not treat this as a benchmark."
    )
    lines.append("")
    return "\n".join(lines)
