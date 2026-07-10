from __future__ import annotations

from pathlib import Path

import yaml

from eval.types import FixtureDoc, HumanGrade, Question

QUESTION_SCHEMA_MSG = (
    "question must have id, question_text, reference_answer, expected_behavior, grounding"
)


def load_corpus(corpus_dir: Path) -> list[FixtureDoc]:
    """Load every .md file in a fixture corpus directory as a FixtureDoc.

    The document id is the filename stem (e.g. ``03_datastore-blorbledb``).
    """
    docs: list[FixtureDoc] = []
    for path in sorted(corpus_dir.glob("*.md")):
        docs.append(
            FixtureDoc(
                id=path.stem,
                filename=path.name,
                text=path.read_text(encoding="utf-8"),
            )
        )
    if not docs:
        raise ValueError(f"No .md fixture documents found in {corpus_dir}")
    return docs


def load_questions(path: Path) -> list[Question]:
    """Load the eval question set from a YAML file and validate each entry."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a top-level YAML list of questions")
    questions: list[Question] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{QUESTION_SCHEMA_MSG}; got {type(entry).__name__}")
        try:
            questions.append(
                Question(
                    id=str(entry["id"]),
                    question_text=str(entry["question_text"]),
                    reference_answer=str(entry["reference_answer"]),
                    expected_behavior=str(entry["expected_behavior"]),  # type: ignore[arg-type]
                    grounding=[str(g) for g in entry["grounding"]],
                )
            )
        except KeyError as exc:
            raise ValueError(f"{QUESTION_SCHEMA_MSG}; missing key {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid question entry: {exc}") from exc
    if not questions:
        raise ValueError(f"No questions loaded from {path}")
    return questions


def load_human_grades(path: Path) -> list[HumanGrade]:
    """Load human calibration grades from a YAML file.

    Returns an empty list when the scaffold has not been filled in yet.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        return []
    grades_raw = raw.get("human_grades") if isinstance(raw, dict) else raw
    if not grades_raw:
        return []
    grades: list[HumanGrade] = []
    for entry in grades_raw:
        if not isinstance(entry, dict):
            continue
        grades.append(
            HumanGrade(
                question_id=str(entry["question_id"]),
                arm=str(entry["arm"]),  # type: ignore[arg-type]
                factual_correctness=int(entry["factual_correctness"]),
                grounding=int(entry["grounding"]),
                conflict_handling=int(entry["conflict_handling"]),
            )
        )
    return grades


def corpus_text_for(corpus: list[FixtureDoc], doc_ids: list[str]) -> list[str]:
    """Return the full text of the named fixture documents, in the order given."""
    by_id = {doc.id: doc.text for doc in corpus}
    missing = [doc_id for doc_id in doc_ids if doc_id not in by_id]
    if missing:
        raise ValueError(f"Unknown grounding document ids: {missing}")
    return [by_id[doc_id] for doc_id in doc_ids]
