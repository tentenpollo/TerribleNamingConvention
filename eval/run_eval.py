from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import AsyncMock, patch
import uuid

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.llm import LLMResult, llm_call
from app.core.logging import logger
from app.core.qdrant import get_qdrant_client
from app.ingestion.chunker import ChunkingStrategy, chunk_text
from app.ingestion.embedder import Embedder, SparseEmbedder, get_embedder, get_sparse_embedder
from app.ingestion.parser import parse_file
from app.ingestion.summarizer import summarize_document
from app.ingestion.vector_store import VectorStore
from app.models.belief_state import BeliefState
from app.models.document import Document, FileType
from app.models.document_summary import DocumentSummary
from app.models.ingestion_job import IngestionJob
from app.models.project import Project
from app.models.team import Team
from app.retrieval.retriever import RetrievedChunk
from app.retrieval.retriever import retrieve as _real_retrieve
from app.schemas.query import QueryResponse
from app.services.cag import CAGService
from app.services.query import QueryService
from app.workers.cag import cag_rebuild
from eval.fixtures import corpus_text_for, load_corpus, load_questions
from eval.judge import judge_question
from eval.mock_llm import mock_llm_enabled
from eval.report import aggregate, render_report
from eval.types import (
    ArmResult,
    CalibrationResult,
    ChunkRef,
    EvalInvariantError,
    FixtureDoc,
    JudgeOutcome,
    Question,
    QuestionResult,
    RunMeta,
    RunResults,
)

# Harness instrumentation: inject response_format=json_object into summarizer
# LLM calls so models that ignore prompt-only JSON instructions (e.g. DeepSeek)
# still produce parseable summaries.  Applied only during the ingestion step
# via a runtime patch; does not modify app/ code.
_real_summarizer_llm = llm_call  # captured at import before any patch


async def _summarizer_llm_with_json(**kwargs: object) -> LLMResult:
    kwargs.setdefault("response_format", {"type": "json_object"})
    return await _real_summarizer_llm(**kwargs)  # type: ignore[arg-type]


EVAL_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000000")
DEFAULT_QUESTIONS = Path("eval/fixtures/questions.yaml")
DEFAULT_CORPUS = Path("eval/fixtures/corpus")
DEFAULT_OUT = Path("eval/results")

# The undated-contradiction pair must be temporally symmetric so CAG synthesis
# does not silently resolve the contradiction via event-log ordering. The
# batch-digest prompt falls back to created_at when no in-content date exists.
# Assign both the same explicit timestamp so neither is "later".
_CONTRADICTION_PAIR_IDS = {"05_sync-interval-5min", "06_sync-interval-60sec"}
_CONTRADICTION_TS = datetime(2025, 3, 15, 12, 0, 0, tzinfo=UTC)


@dataclass
class LLMSpy:
    """Transparent wrapper around llm_call that records the last LLMResult."""

    real: Callable[..., Awaitable[LLMResult]]
    last: LLMResult | None = None

    async def __call__(self, **kwargs: object) -> LLMResult:
        result = await self.real(**kwargs)
        self.last = result
        return result


def _spy_tokens(spy: LLMSpy | None) -> tuple[int | None, int | None]:
    if spy is not None and spy.last is not None:
        return spy.last.prompt_tokens, spy.last.completion_tokens
    return None, None


def _chunk_keys(resp: QueryResponse) -> list[tuple[str, int]]:
    return [(str(s.document_id), s.chunk_index) for s in resp.sources]


def _to_arm_result(
    resp: QueryResponse,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: int,
) -> ArmResult:
    return ArmResult(
        answer=resp.answer,
        chunks=[
            ChunkRef(
                document_id=str(s.document_id),
                chunk_index=s.chunk_index,
                filename=s.filename,
            )
            for s in resp.sources
        ],
        belief_state_version=resp.belief_state_version,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


_RETRIEVE_CACHE: dict[tuple[uuid.UUID, str, int], list[RetrievedChunk]] = {}


def _clear_retrieve_cache() -> None:
    _RETRIEVE_CACHE.clear()


async def _deterministic_retrieve(
    project_id: uuid.UUID,
    query_text: str,
    accessible_ids: list[uuid.UUID],
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve once per (project, question, top_k) and reuse for both arms.

    Qdrant RRF does not guarantee a stable order or a stable top-k SET among
    equal-scored points: two consecutive identical queries can return the same
    chunks in a different order, or a different chunk at the cutoff when ties
    straddle top_k. Both would confound the CAG-vs-RAG comparison, since chunk
    order flows into the [S1..Sn] prompt labels and a different 8th chunk changes
    the evidence. To make belief-state presence the ONLY variable, the first
    retrieval call per question is cached (after a stable tie-break by
    -score, document_id, chunk_index) and replayed for the second arm. Both arms
    still call the real ``QueryService.query()``; only the retrieval
    nondeterminism is removed. This is harness instrumentation; it does not
    modify app/ code.
    """
    key = (project_id, query_text, top_k)
    cached = _RETRIEVE_CACHE.get(key)
    if cached is not None:
        return list(cached)
    chunks = await _real_retrieve(
        project_id=project_id,
        query_text=query_text,
        accessible_ids=accessible_ids,
        vector_store=vector_store,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        top_k=top_k,
    )
    chunks.sort(key=lambda c: (-c.score, str(c.document_id), c.chunk_index))
    _RETRIEVE_CACHE[key] = list(chunks)
    return chunks


async def run_two_arms(
    query_service: QueryService,
    question: Question,
    project_id: uuid.UUID,
    accessible_ids: list[uuid.UUID],
    user_id: uuid.UUID,
    top_k: int,
    spy: LLMSpy | None = None,
) -> tuple[ArmResult, ArmResult, bool]:
    """Run the CAG arm and the RAG arm for one question.

    The CAG arm is the real ``QueryService.query()`` as-is. The RAG arm is the
    SAME path with ``cag_service.get_latest`` forced to return None, so the only
    variable is belief-state presence. Retrieval does not depend on the belief
    state, so the retrieved chunk ids MUST be identical across arms; divergent
    retrieval means the arms are not comparable and the eval is invalid.
    """
    t0 = time.perf_counter()
    cag_resp = await query_service.query(
        question.question_text, project_id, accessible_ids, user_id, top_k
    )
    cag_lat = int((time.perf_counter() - t0) * 1000)
    cag_tokens = _spy_tokens(spy)

    with patch.object(query_service.cag_service, "get_latest", new=AsyncMock(return_value=None)):
        t1 = time.perf_counter()
        rag_resp = await query_service.query(
            question.question_text, project_id, accessible_ids, user_id, top_k
        )
        rag_lat = int((time.perf_counter() - t1) * 1000)
    rag_tokens = _spy_tokens(spy)

    cag_keys = _chunk_keys(cag_resp)
    rag_keys = _chunk_keys(rag_resp)
    # GUARD-RAIL: both arms share the pinned-retrieval cache, so these are
    # identical by construction today.  The assertion is a structural invariant:
    # if retrieval ever becomes belief-state-aware or the cache were removed,
    # THIS assertion fires and the run is invalid — the arms would no longer
    # be comparable.  It is tautological under the current design, not
    # decorative.
    if cag_keys != rag_keys:
        raise EvalInvariantError(
            f"Retrieved chunks differ across arms for question {question.id}: "
            f"CAG={cag_keys} RAG={rag_keys}"
        )

    return (
        _to_arm_result(cag_resp, *cag_tokens, cag_lat),
        _to_arm_result(rag_resp, *rag_tokens, rag_lat),
        True,
    )


async def setup_project(
    session: AsyncSession,
    name: str,
    config: dict[str, object],
) -> tuple[Project, Team]:
    team = Team(name=f"{name}-team")
    session.add(team)
    await session.flush()
    project = Project(
        name=name,
        description="CAG eval throwaway project — delete after the run",
        team_id=team.id,
        config=config,
    )
    session.add(project)
    await session.flush()
    await session.commit()
    return project, team


async def ingest_fixtures(
    session: AsyncSession,
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    corpus: list[FixtureDoc],
    project_id: uuid.UUID,
    context_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Ingest the fixture corpus through the real pipeline (parse/chunk/embed/upsert/summarize)."""
    for doc in corpus:
        document = Document(
            project_id=project_id,
            filename=doc.filename,
            file_type=FileType.MARKDOWN.value,
            raw_bytes=doc.text.encode("utf-8"),
        )
        session.add(document)
        await session.flush()

        parsed = parse_file(content=document.raw_bytes, file_type=FileType.MARKDOWN.value)
        chunks = chunk_text(
            text=parsed,
            strategy=ChunkingStrategy.NAIVE,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        for chunk in chunks:
            chunk.metadata["filename"] = doc.filename

        dense, sparse = await asyncio.gather(
            asyncio.to_thread(embedder.embed, chunks),
            asyncio.to_thread(sparse_embedder.embed, chunks),
        )
        await vector_store.upsert(project_id, dense, document.id, sparse)

        summary = await summarize_document(text=parsed, filename=doc.filename, model=context_model)
        summary_row = DocumentSummary(
            document_id=document.id,
            project_id=project_id,
            summary=summary,
        )
        if doc.id in _CONTRADICTION_PAIR_IDS:
            summary_row.created_at = _CONTRADICTION_TS  # pin both to same ts
        session.add(summary_row)
        await session.commit()
        logger.info(
            "Ingested fixture",
            filename=doc.filename,
            project_id=str(project_id),
            chunks=len(chunks),
        )


async def run_cag_rebuild(project_id: uuid.UUID, embedder: Embedder) -> None:
    """Produce a GENUINE belief state via the real CAG genesis rebuild path."""
    await cag_rebuild({"embedder": embedder}, project_id, "genesis")


async def run_arms_and_judge(
    query_service: QueryService,
    questions: list[Question],
    corpus: list[FixtureDoc],
    project_id: uuid.UUID,
    judge_model: str,
    spy: LLMSpy | None,
    top_k: int,
    judge_rng_seed: int,
) -> list[QuestionResult]:
    rng = random.Random(judge_rng_seed)  # noqa: S311
    _clear_retrieve_cache()
    accessible = [project_id]
    results: list[QuestionResult] = []
    for question in questions:
        cag_arm, rag_arm, retrieval_match = await run_two_arms(
            query_service, question, project_id, accessible, EVAL_USER_ID, top_k, spy
        )
        source_texts = corpus_text_for(corpus, question.grounding)
        outcome: JudgeOutcome = await judge_question(
            question,
            cag_arm.answer,
            rag_arm.answer,
            source_texts,
            judge_model,
            rng,
        )
        results.append(
            QuestionResult(
                question_id=question.id,
                question_text=question.question_text,
                reference_answer=question.reference_answer,
                expected_behavior=question.expected_behavior,
                grounding=question.grounding,
                cag=cag_arm,
                rag=rag_arm,
                retrieval_match=retrieval_match,
                judge=outcome,
            )
        )
    return results


async def teardown(
    session: AsyncSession,
    vector_store: VectorStore,
    project_id: uuid.UUID,
    team_id: uuid.UUID,
) -> None:
    """Remove the eval project's Qdrant collection and DB rows.

    DB cleanup uses explicit deletes by project_id so it does not rely solely on
    cascade behaviour. Qdrant errors are logged but do not block DB cleanup.
    """
    try:
        await vector_store.delete_collection(project_id)
    except Exception as exc:
        logger.error("Teardown: Qdrant delete failed", project_id=str(project_id), error=str(exc))

    await session.execute(delete(BeliefState).where(BeliefState.project_id == project_id))
    await session.execute(delete(DocumentSummary).where(DocumentSummary.project_id == project_id))
    await session.execute(delete(IngestionJob).where(IngestionJob.project_id == project_id))
    await session.execute(delete(Document).where(Document.project_id == project_id))
    await session.execute(delete(Project).where(Project.id == project_id))
    await session.execute(delete(Team).where(Team.id == team_id))
    await session.commit()


def _write_results(results: RunResults, out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{run_id}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {path}")
    path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_report(
    results: RunResults,
    out_dir: Path,
    run_id: str,
    calibration: CalibrationResult | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{run_id}.md"
    data = aggregate(results.questions)
    path.write_text(render_report(data, calibration, results.meta), encoding="utf-8")
    return path


async def run_eval(
    questions: list[Question],
    corpus: list[FixtureDoc],
    session_factory: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    generation_model: str,
    context_model: str,
    judge_model: str,
    mock_llm: bool = False,
    top_k: int = 8,
    out_dir: Path = DEFAULT_OUT,
    judge_rng_seed: int = 0,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> RunResults:
    """Run the full eval: ingest fixtures, rebuild CAG, two arms per question, judge, write results.

    Teardown runs in ``finally`` so the throwaway project is removed even if the
    run raises midway.
    """
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    config: dict[str, object] = {
        "query_model": generation_model,
        "context_model": context_model,
        "cag_rebuild_threshold": 1,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": "naive",
    }
    project_id: uuid.UUID | None = None
    team_id: uuid.UUID | None = None
    try:
        async with session_factory() as session:
            project, team = await setup_project(session, "cag-eval", config)
            project_id, team_id = project.id, team.id
            with patch(
                "app.ingestion.summarizer.llm_call",
                new=_summarizer_llm_with_json,
            ):
                await ingest_fixtures(
                    session,
                    vector_store,
                    embedder,
                    sparse_embedder,
                    corpus,
                    project.id,
                    context_model,
                    chunk_size,
                    chunk_overlap,
                )

        await run_cag_rebuild(project.id, embedder)

        spy = LLMSpy(llm_call)
        with (
            patch("app.services.query.llm_call", spy),
            patch("app.services.query.retrieve", new=_deterministic_retrieve),
        ):
            async with session_factory() as qsession:
                query_service = QueryService(
                    qsession,
                    vector_store,
                    embedder,
                    sparse_embedder,
                    CAGService(qsession),
                )
                question_results = await run_arms_and_judge(
                    query_service,
                    questions,
                    corpus,
                    project.id,
                    judge_model,
                    spy,
                    top_k,
                    judge_rng_seed,
                )

        results = RunResults(
            meta=RunMeta(
                run_id=run_id,
                project_id=str(project_id),
                created_at=datetime.now(UTC).isoformat(),
                generation_model=generation_model,
                judge_model=judge_model,
                mock_llm=mock_llm,
            ),
            questions=question_results,
        )
        _write_results(results, out_dir, run_id)
        return results
    finally:
        if project_id is not None and team_id is not None:
            try:
                async with session_factory() as tsession:
                    await teardown(tsession, vector_store, project_id, team_id)
            except Exception as exc:
                logger.error("Teardown failed", error=str(exc))


def _build_real_infra() -> tuple[VectorStore, Embedder, SparseEmbedder]:
    return VectorStore(client=get_qdrant_client()), get_embedder(), get_sparse_embedder()


async def _amain(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Run the CAG vs RAG evaluation harness.")
    parser.add_argument(
        "--mock", action="store_true", help="Use the deterministic mock LLM (no key)."
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("EVAL_JUDGE_MODEL", settings.litellm_query_model),
        help="Judge model (defaults to $EVAL_JUDGE_MODEL, then LITELLM_QUERY_MODEL).",
    )
    parser.add_argument("--generation-model", default=settings.litellm_query_model)
    parser.add_argument("--context-model", default=settings.litellm_context_model)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--judge-seed", type=int, default=0)
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    corpus = load_corpus(args.corpus)
    vector_store, embedder, sparse_embedder = _build_real_infra()

    cm = mock_llm_enabled() if args.mock else nullcontext()
    with cm:
        results = await run_eval(
            questions=questions,
            corpus=corpus,
            session_factory=AsyncSessionLocal,
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            generation_model=args.generation_model,
            context_model=args.context_model,
            judge_model=args.judge_model,
            mock_llm=args.mock,
            top_k=args.top_k,
            out_dir=args.out,
            judge_rng_seed=args.judge_seed,
        )

    from eval.calibrate import compute_calibration, extract_judge_grades
    from eval.fixtures import load_human_grades

    human = load_human_grades(Path("eval/fixtures/human_grades.yaml"))
    calibration = compute_calibration(extract_judge_grades(results), human)
    report_path = write_report(results, args.out, results.meta.run_id, calibration)
    print(f"Wrote {args.out / f'run_{results.meta.run_id}.json'}")
    print(f"Wrote {report_path}")
    if args.mock:
        print(
            "NOTE: mock LLM was used. The numbers in the report are meaningless; "
            "this run proves plumbing only.",
            file=sys.stderr,
        )


def main() -> None:
    asyncio.run(_amain(None))


if __name__ == "__main__":
    main()
