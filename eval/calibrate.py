from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eval.types import (
    AXES,
    AxisCalibration,
    CalibrationResult,
    HumanGrade,
    JudgeGrade,
    RunResults,
)

DEFAULT_WARN_THRESHOLD = 1.0


def compute_calibration(
    judge_grades: list[JudgeGrade],
    human_grades: list[HumanGrade],
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
) -> CalibrationResult:
    """Compare unblinded judge scores to human scores on the shared rubric.

    Pairs are matched by (question_id, arm). When the mean absolute difference
    on any axis exceeds ``warn_threshold``, the result is marked untrustworthy
    and carries a loud warning string; otherwise it is silent.
    """
    human_by_key: dict[tuple[str, str], HumanGrade] = {
        (g.question_id, g.arm): g for g in human_grades
    }

    diffs: dict[str, list[int]] = {axis: [] for axis in AXES}
    for jg in judge_grades:
        human = human_by_key.get((jg.question_id, jg.arm))
        if human is None:
            continue
        for axis in AXES:
            diffs[axis].append(abs(getattr(jg, axis) - getattr(human, axis)))

    per_axis: dict[str, AxisCalibration] = {}
    worst_axis: str | None = None
    worst_diff = 0.0
    for axis in AXES:
        values = diffs[axis]
        n = len(values)
        if n == 0:
            per_axis[axis] = AxisCalibration(
                mean_abs_diff=0.0, exact_agreement=0.0, within_one_agreement=0.0, n=0
            )
            continue
        mean_abs = sum(values) / n
        exact = sum(1 for d in values if d == 0) / n
        within_one = sum(1 for d in values if d <= 1) / n
        per_axis[axis] = AxisCalibration(
            mean_abs_diff=mean_abs,
            exact_agreement=exact,
            within_one_agreement=within_one,
            n=n,
        )
        if mean_abs > worst_diff:
            worst_diff = mean_abs
            worst_axis = axis

    total_n = max((per_axis[a].n for a in AXES), default=0)
    if total_n == 0:
        return CalibrationResult(
            trustworthy=True,
            warning=None,
            per_axis=per_axis,
            n=0,
        )

    if worst_axis is not None and worst_diff > warn_threshold:
        warning = (
            f"JUDGE SCORES UNTRUSTWORTHY: mean abs diff on axis "
            f"'{worst_axis}' is {worst_diff:.2f} > {warn_threshold:.2f}. "
            f"The judge rubric needs revision before interpreting the full run."
        )
        return CalibrationResult(
            trustworthy=False,
            warning=warning,
            per_axis=per_axis,
            n=total_n,
        )

    return CalibrationResult(
        trustworthy=True,
        warning=None,
        per_axis=per_axis,
        n=total_n,
    )


def extract_judge_grades(results: RunResults) -> list[JudgeGrade]:
    """Unblind a run's judge outcomes into per-(question, arm) judge grades."""
    out: list[JudgeGrade] = []
    for qr in results.questions:
        if qr.judge is None:
            continue
        j = qr.judge
        s1, s2 = j.scores.answer1, j.scores.answer2
        out.append(
            JudgeGrade(
                question_id=qr.question_id,
                arm=j.answer1_arm,
                factual_correctness=s1.factual_correctness.score,
                grounding=s1.grounding.score,
                conflict_handling=s1.conflict_handling.score,
            )
        )
        out.append(
            JudgeGrade(
                question_id=qr.question_id,
                arm=j.answer2_arm,
                factual_correctness=s2.factual_correctness.score,
                grounding=s2.grounding.score,
                conflict_handling=s2.conflict_handling.score,
            )
        )
    return out


def print_calibration(result: CalibrationResult) -> None:
    """Print the calibration result; emit a LOUD warning to stderr when untrustworthy."""
    if result.warning:
        print(f"\n*** {result.warning} ***\n", file=sys.stderr)
    print(f"Calibration: trustworthy={result.trustworthy} n={result.n}")
    for axis in AXES:
        c = result.per_axis.get(axis)
        if c is None or c.n == 0:
            continue
        print(
            f"  {axis}: mean_abs_diff={c.mean_abs_diff:.2f} "
            f"exact={c.exact_agreement:.2f} within_one={c.within_one_agreement:.2f} n={c.n}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare judge scores to human grades.")
    parser.add_argument("--run", required=True, help="Path to eval/results/run_<ts>.json")
    parser.add_argument(
        "--human",
        default="eval/fixtures/human_grades.yaml",
        help="Path to human_grades.yaml",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_WARN_THRESHOLD)
    args = parser.parse_args()

    from eval.fixtures import load_human_grades

    data = json.loads(Path(args.run).read_text(encoding="utf-8"))
    results = RunResults.model_validate(data)
    human = load_human_grades(Path(args.human))
    result = compute_calibration(extract_judge_grades(results), human, args.threshold)
    print_calibration(result)


if __name__ == "__main__":
    main()
