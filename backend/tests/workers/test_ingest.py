from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.exceptions import QdrantError, UnsupportedFileTypeError
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import EmbeddingResult
from app.models.document import Document, FileType
from app.models.ingestion_job import IngestionJob, JobStatus
from app.models.project import Project
from app.workers.ingest import _maybe_enqueue_cag_update, ingest_document

_SUMMARY_MOCK_RETURN = {
    "summary": "A test document for ingestion pipeline verification.",
    "key_points": ["test summary"],
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
    "topics": ["test"],
    "important_relationships": [],
    "document_type": "other",
    "confidence": 0.9,
}


def _make_mock_job(status: str = JobStatus.PENDING.value) -> IngestionJob:
    job = MagicMock(spec=IngestionJob)
    job.id = uuid.uuid4()
    job.status = status
    job.error_message = None
    job.completed_at = None
    return job


def _make_mock_project() -> Project:
    project = MagicMock(spec=Project)
    project.id = uuid.uuid4()
    project.config = {}
    return project


def _make_mock_document() -> Document:
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.project_id = uuid.uuid4()
    doc.filename = "test.md"
    doc.file_type = FileType.MARKDOWN.value
    doc.raw_bytes = b"# Hello\n\nThis is test content for ingestion."
    return doc


def _make_mock_session(
    job: IngestionJob | None = None,
    doc: Document | None = None,
    project: Project | None = None,
    commit_calls: list | None = None,
) -> AsyncMock:
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = None

    if commit_calls is not None:
        session.commit = AsyncMock(side_effect=lambda: commit_calls.append("commit"))

    def mock_execute(statement):
        # INSERT statements (e.g. pg_insert for idempotent summary) — no entity
        if not hasattr(statement, "column_descriptions"):
            return MagicMock()

        entity = statement.column_descriptions[0]["entity"]
        if entity is IngestionJob and job is not None:
            mock_result = MagicMock()
            mock_result.scalar_one_or_none = lambda: job
            return mock_result
        if entity is Document and doc is not None:
            mock_result = MagicMock()
            mock_result.scalar_one_or_none = lambda: doc
            return mock_result
        if entity is Project and project is not None:
            mock_result = MagicMock()
            mock_result.scalar_one_or_none = lambda: project
            return mock_result
        if entity is None:
            mock_result = MagicMock()
            mock_result.scalar = lambda: 1
            return mock_result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = lambda: None
        return mock_result

    session.execute = AsyncMock(side_effect=mock_execute)

    return session


def _make_mock_embedder() -> MagicMock:
    embedder = MagicMock()
    chunk = Chunk(text="hello world", index=0, metadata={"filename": "test.md"})
    embedder.embed.return_value = [
        EmbeddingResult(chunk=chunk, vector=[0.1] * 384),
    ]
    return embedder


def _make_mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.mark.unit
async def test_ingest_document_happy_path_sets_complete() -> None:
    job = _make_mock_job()
    doc = _make_mock_document()
    project = _make_mock_project()
    commit_calls: list[str] = []
    session = _make_mock_session(job=job, doc=doc, project=project, commit_calls=commit_calls)
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()

    ctx: dict[str, object] = {
        "embedder": embedder,
        "vector_store": vector_store,
    }

    job_id = uuid.uuid4()
    document_id = doc.id
    project_id = doc.project_id

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest.summarize_document", return_value=_SUMMARY_MOCK_RETURN):
            with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
                async with session:
                    await ingest_document(ctx, job_id, document_id, project_id)

    assert job.status == JobStatus.COMPLETE.value
    assert job.completed_at is not None
    assert len(commit_calls) >= 2
    embedder.embed.assert_called_once()
    vector_store.upsert.assert_awaited_once()


@pytest.mark.unit
async def test_ingest_document_parse_failure_sets_failed() -> None:
    job = _make_mock_job()
    doc = _make_mock_document()
    doc.file_type = "csv"
    project = _make_mock_project()
    commit_calls: list[str] = []
    session = _make_mock_session(job=job, doc=doc, project=project, commit_calls=commit_calls)
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()

    ctx: dict[str, object] = {
        "embedder": embedder,
        "vector_store": vector_store,
    }

    job_id = uuid.uuid4()
    document_id = doc.id
    project_id = doc.project_id

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
            with pytest.raises(UnsupportedFileTypeError):
                async with session:
                    await ingest_document(ctx, job_id, document_id, project_id)

    assert job.status == JobStatus.FAILED.value
    assert job.error_message is not None
    assert "Unsupported file type" in job.error_message


@pytest.mark.unit
async def test_ingest_document_qdrant_upsert_failure_sets_failed() -> None:
    job = _make_mock_job()
    doc = _make_mock_document()
    project = _make_mock_project()
    commit_calls: list[str] = []
    session = _make_mock_session(job=job, doc=doc, project=project, commit_calls=commit_calls)
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()
    vector_store.upsert.side_effect = QdrantError("Qdrant connection refused")

    ctx: dict[str, object] = {
        "embedder": embedder,
        "vector_store": vector_store,
    }

    job_id = uuid.uuid4()
    document_id = doc.id
    project_id = doc.project_id

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
            with pytest.raises(QdrantError):
                async with session:
                    await ingest_document(ctx, job_id, document_id, project_id)

    assert job.status == JobStatus.FAILED.value
    assert job.error_message is not None
    assert "Qdrant connection refused" in job.error_message


@pytest.mark.unit
async def test_ingest_document_status_commits_in_correct_order() -> None:
    job = _make_mock_job()
    doc = _make_mock_document()
    project = _make_mock_project()
    statuses_after_each_commit: list[str | None] = []
    commit_calls: list[str] = []

    async def tracking_commit():
        statuses_after_each_commit.append(job.status)
        commit_calls.append("commit")

    session = _make_mock_session(job=job, doc=doc, project=project, commit_calls=commit_calls)
    session.commit = tracking_commit
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()

    ctx: dict[str, object] = {
        "embedder": embedder,
        "vector_store": vector_store,
    }

    job_id = uuid.uuid4()
    document_id = doc.id
    project_id = doc.project_id

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest.summarize_document", return_value=_SUMMARY_MOCK_RETURN):
            with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
                async with session:
                    await ingest_document(ctx, job_id, document_id, project_id)

    assert JobStatus.RUNNING.value in statuses_after_each_commit
    assert JobStatus.COMPLETE.value in statuses_after_each_commit
    running_index = statuses_after_each_commit.index(JobStatus.RUNNING.value)
    complete_index = statuses_after_each_commit.index(JobStatus.COMPLETE.value)
    assert running_index < complete_index


@pytest.mark.unit
async def test_ingest_document_uses_project_chunk_config() -> None:
    job = _make_mock_job()
    doc = _make_mock_document()
    project = _make_mock_project()
    project.config = {"chunk_size": 256, "chunk_overlap": 32}
    commit_calls: list[str] = []
    session = _make_mock_session(job=job, doc=doc, project=project, commit_calls=commit_calls)
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()

    ctx: dict[str, object] = {
        "embedder": embedder,
        "vector_store": vector_store,
    }

    job_id = uuid.uuid4()
    document_id = doc.id
    project_id = doc.project_id

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest.summarize_document", return_value=_SUMMARY_MOCK_RETURN):
            with patch("app.workers.ingest.chunk_text") as chunk_text_mock:
                with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
                    chunk_text_mock.return_value = [
                        Chunk(text="hello", index=0, metadata={"filename": "test.md"}),
                    ]
                    async with session:
                        await ingest_document(ctx, job_id, document_id, project_id)

    assert chunk_text_mock.call_args.kwargs["chunk_size"] == 256
    assert chunk_text_mock.call_args.kwargs["overlap"] == 32


@pytest.mark.unit
async def test_ingest_document_binary_pdf_stored_and_parsed() -> None:
    """Binary PDF bytes must survive upload without UnicodeDecodeError and parse cleanly."""
    import fitz  # PyMuPDF

    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((72, 100), "Phase 3.0 PDF hardening test content.")
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    job = _make_mock_job()
    doc = _make_mock_document()
    doc.raw_bytes = pdf_bytes
    doc.file_type = FileType.PDF.value
    doc.filename = "test.pdf"
    project = _make_mock_project()
    session = _make_mock_session(job=job, doc=doc, project=project)
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()

    ctx: dict[str, object] = {"embedder": embedder, "vector_store": vector_store}

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest.summarize_document", return_value=_SUMMARY_MOCK_RETURN):
            with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
                async with session:
                    await ingest_document(ctx, job.id, doc.id, doc.project_id)

    assert job.status == JobStatus.COMPLETE.value
    vector_store.upsert.assert_awaited_once()


@pytest.mark.unit
async def test_ingest_document_contextual_strategy_raises_not_implemented() -> None:
    job = _make_mock_job()
    doc = _make_mock_document()
    project = _make_mock_project()
    project.config = {"chunking_strategy": "contextual"}
    session = _make_mock_session(job=job, doc=doc, project=project)
    embedder = _make_mock_embedder()
    vector_store = _make_mock_vector_store()

    ctx: dict[str, object] = {"embedder": embedder, "vector_store": vector_store}

    with patch("app.workers.ingest.AsyncSessionLocal", return_value=session):
        with patch("app.workers.ingest._maybe_enqueue_cag_update", return_value=None):
            with pytest.raises(NotImplementedError):
                async with session:
                    await ingest_document(ctx, job.id, doc.id, doc.project_id)

    assert job.status == JobStatus.FAILED.value
    assert job.error_message is not None
    assert "implement" in job.error_message.lower()


# ---------------------------------------------------------------------------
# CAG watermark trigger unit tests
# ---------------------------------------------------------------------------


def _make_mock_cag_service(
    latest: MagicMock | None,
    pending_count: int,
) -> MagicMock:
    service = MagicMock()
    service.get_latest = AsyncMock(return_value=latest)
    service.count_pending = AsyncMock(return_value=pending_count)
    return service


@pytest.mark.unit
async def test_maybe_enqueue_cag_update_initial_state_enqueues_on_first_summary() -> None:
    project = _make_mock_project()
    mock_arq = AsyncMock()
    mock_arq.enqueue_job.return_value = object()
    ctx: dict[str, object] = {"redis": mock_arq}
    session = _make_mock_session(project=project)

    service = _make_mock_cag_service(latest=None, pending_count=1)

    with patch("app.workers.ingest.CAGService", return_value=service):
        await _maybe_enqueue_cag_update(ctx, session, project)

    mock_arq.enqueue_job.assert_awaited_once()
    call_args = mock_arq.enqueue_job.call_args
    assert call_args.args[0] == "cag_update"
    assert call_args.args[1] == project.id
    assert call_args.kwargs.get("_job_id") == f"cag-update-{project.id}"


@pytest.mark.unit
async def test_maybe_enqueue_cag_update_below_threshold_does_not_enqueue() -> None:
    project = _make_mock_project()
    project.config = {"cag_rebuild_threshold": 5}
    mock_arq = AsyncMock()
    ctx: dict[str, object] = {"redis": mock_arq}
    session = _make_mock_session(project=project)

    latest = MagicMock()
    latest.last_summary_created_at = None
    service = _make_mock_cag_service(latest=latest, pending_count=3)

    with patch("app.workers.ingest.CAGService", return_value=service):
        await _maybe_enqueue_cag_update(ctx, session, project)

    mock_arq.enqueue_job.assert_not_awaited()


@pytest.mark.unit
async def test_maybe_enqueue_cag_update_at_threshold_enqueues() -> None:
    project = _make_mock_project()
    project.config = {"cag_rebuild_threshold": 5}
    mock_arq = AsyncMock()
    mock_arq.enqueue_job.return_value = object()
    ctx: dict[str, object] = {"redis": mock_arq}
    session = _make_mock_session(project=project)

    latest = MagicMock()
    latest.last_summary_created_at = None
    service = _make_mock_cag_service(latest=latest, pending_count=5)

    with patch("app.workers.ingest.CAGService", return_value=service):
        await _maybe_enqueue_cag_update(ctx, session, project)

    mock_arq.enqueue_job.assert_awaited_once()


@pytest.mark.unit
async def test_maybe_enqueue_cag_update_already_queued_logs_and_skips() -> None:
    project = _make_mock_project()
    project.config = {"cag_rebuild_threshold": 5}
    mock_arq = AsyncMock()
    mock_arq.enqueue_job.return_value = None  # ARQ dedup: already queued
    ctx: dict[str, object] = {"redis": mock_arq}
    session = _make_mock_session(project=project)

    service = _make_mock_cag_service(latest=None, pending_count=5)

    with patch("app.workers.ingest.CAGService", return_value=service):
        await _maybe_enqueue_cag_update(ctx, session, project)

    mock_arq.enqueue_job.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration tests — require real Postgres, Redis, Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ingest_document_full_pipeline() -> None:
    from unittest.mock import patch

    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import delete

    from app.core import database
    from app.core.config import settings
    from app.ingestion.embedder import Embedder
    from app.ingestion.vector_store import VectorStore
    from app.models.document_summary import DocumentSummary
    from app.models.project import Project
    from app.models.team import Team

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
        with patch(
            "app.workers.ingest.summarize_document",
            return_value={
                "summary": "Integration test document.",
                "key_points": ["test"],
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
                "confidence": 0.9,
            },
        ):
            async with database.AsyncSessionLocal() as session:
                team = Team(name="Ingest Worker Integration Team")
                session.add(team)
                await session.flush()

                project = Project(
                    name="Ingest Worker Integration Project",
                    description="Integration test project",
                    team_id=team.id,
                    config={},
                )
                session.add(project)
                await session.flush()

                doc = Document(
                    project_id=project.id,
                    filename="integration-test.md",
                    file_type=FileType.MARKDOWN.value,
                    raw_bytes=(
                        b"# Integration Test\n\nThis document verifies the full ingestion pipeline."
                    ),
                )
                session.add(doc)
                await session.flush()

                job = IngestionJob(
                    project_id=project.id,
                    document_id=doc.id,
                    status=JobStatus.PENDING.value,
                )
                session.add(job)
                await session.commit()

                project_id = project.id
                document_id = doc.id
                job_id = job.id
                team_id = team.id

            embedder = Embedder()
            vector_store = VectorStore(client=qdrant_client)

            mock_arq = AsyncMock()
            ctx: dict[str, object] = {
                "embedder": embedder,
                "vector_store": vector_store,
                "redis": mock_arq,
            }

            await ingest_document(ctx, job_id, document_id, project_id)

        async with database.AsyncSessionLocal() as session:
            from sqlalchemy import select as sa_select

            result = await session.execute(
                sa_select(IngestionJob).where(IngestionJob.id == job_id),
            )
            refreshed_job = result.scalar_one()
            assert refreshed_job.status == JobStatus.COMPLETE.value
            assert refreshed_job.completed_at is not None

            summary_result = await session.execute(
                sa_select(DocumentSummary).where(DocumentSummary.document_id == document_id),
            )
            summary = summary_result.scalar_one()
            assert summary.summary
            assert isinstance(summary.summary, dict)

    hits = await VectorStore(client=qdrant_client).search(
        project_id, query_vector=[0.0] * 384, top_k=10
    )
    assert len(hits) > 0

    first_payload = hits[0].payload
    assert first_payload is not None
    assert first_payload["document_id"] == str(document_id)
    assert first_payload["filename"] == "integration-test.md"

    await VectorStore(client=qdrant_client).delete_collection(project_id)

    async with database.AsyncSessionLocal() as session:
        await session.execute(delete(IngestionJob).where(IngestionJob.project_id == project_id))
        await session.execute(
            delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
        )
        await session.execute(delete(Document).where(Document.project_id == project_id))
        await session.execute(delete(Project).where(Project.id == project_id))
        await session.execute(delete(Team).where(Team.id == team_id))
        await session.commit()


@pytest.mark.integration
async def test_ingest_document_idempotent_on_retry() -> None:
    """Simulate an ARQ retry: failure after summary commit but before COMPLETE commit.

    After two runs the Qdrant point count must equal the chunk count exactly (no
    duplicates), and there must be exactly one document_summaries row.
    """
    from unittest.mock import patch

    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import delete
    from sqlalchemy import select as sa_select

    from app.core import database
    from app.core.config import settings
    from app.ingestion.embedder import Embedder
    from app.ingestion.vector_store import VectorStore
    from app.models.document_summary import DocumentSummary
    from app.models.project import Project
    from app.models.team import Team
    from app.workers.ingest import ingest_document

    project_id = None
    team_id = None

    try:
        async with database.AsyncSessionLocal() as session:
            team = Team(name="Idempotency Integration Team")
            session.add(team)
            await session.flush()

            project = Project(
                name="Idempotency Integration Project",
                team_id=team.id,
                config={},
            )
            session.add(project)
            await session.flush()

            doc = Document(
                project_id=project.id,
                filename="idempotency-test.md",
                file_type=FileType.MARKDOWN.value,
                raw_bytes=b"# Idempotency\n\nThis document tests retry safety.",
            )
            session.add(doc)
            await session.flush()

            job = IngestionJob(
                project_id=project.id,
                document_id=doc.id,
                status=JobStatus.PENDING.value,
            )
            session.add(job)
            await session.commit()

            project_id = project.id
            document_id = doc.id
            job_id = job.id
            team_id = team.id

        embedder = Embedder()
        qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
        vector_store = VectorStore(client=qdrant_client)
        mock_arq = AsyncMock()
        ctx: dict[str, object] = {
            "embedder": embedder,
            "vector_store": vector_store,
            "redis": mock_arq,
        }

        summary_mock = {
            "summary": "Idempotency test.",
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
            "confidence": 0.9,
        }

        # First run: raise after summary commit to simulate mid-job crash
        call_count = 0

        async def fail_once() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Injected failure after summary commit")

        with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.ingest.summarize_document", return_value=summary_mock):
                with patch("app.workers.ingest._after_summary_commit", fail_once):
                    with pytest.raises(RuntimeError, match="Injected failure"):
                        await ingest_document(ctx, job_id, document_id, project_id)

        assert call_count == 1, "Failure injection must have fired (not a vacuous pass)"

        # Reset job to PENDING so the second run proceeds
        async with database.AsyncSessionLocal() as session:
            result = await session.execute(sa_select(IngestionJob).where(IngestionJob.id == job_id))
            job_row = result.scalar_one()
            job_row.status = JobStatus.PENDING.value
            job_row.error_message = None
            await session.commit()

        # Second run: should complete cleanly
        with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.ingest.summarize_document", return_value=summary_mock):
                await ingest_document(ctx, job_id, document_id, project_id)

        # Count vectors in Qdrant — must equal chunk count, not double
        async with database.AsyncSessionLocal() as session:
            from app.ingestion.chunker import ChunkingStrategy, chunk_text
            from app.ingestion.parser import parse_file

            result = await session.execute(sa_select(Document).where(Document.id == document_id))
            doc_row = result.scalar_one()
            parsed = parse_file(content=doc_row.raw_bytes, file_type=doc_row.file_type)
            expected_chunks = chunk_text(
                text=parsed,
                strategy=ChunkingStrategy.NAIVE,
                chunk_size=512,
                overlap=64,
            )

        expected_count = len(expected_chunks)
        hits = await vector_store.search(project_id, query_vector=[0.0] * 384, top_k=100)
        assert len(hits) == expected_count, (
            f"Expected exactly {expected_count} Qdrant points, got {len(hits)}"
        )

        async with database.AsyncSessionLocal() as session:
            summary_result = await session.execute(
                sa_select(DocumentSummary).where(DocumentSummary.document_id == document_id)
            )
            summaries = summary_result.scalars().all()
        assert len(summaries) == 1, (
            f"Expected exactly 1 document_summaries row, got {len(summaries)}"
        )

    finally:
        if project_id is not None:
            try:
                await vector_store.delete_collection(project_id)
            except Exception:
                pass
        if project_id is not None:
            async with database.AsyncSessionLocal() as session:
                await session.execute(
                    delete(IngestionJob).where(IngestionJob.project_id == project_id)
                )
                await session.execute(
                    delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
                )
                await session.execute(delete(Document).where(Document.project_id == project_id))
                await session.execute(delete(Project).where(Project.id == project_id))
                if team_id is not None:
                    await session.execute(delete(Team).where(Team.id == team_id))
                await session.commit()


@pytest.mark.integration
async def test_sweep_pending_jobs_re_enqueues_stale_leaves_fresh() -> None:
    """Stale PENDING jobs (> 10 min) must be re-enqueued; fresh ones must be left alone."""
    from unittest.mock import AsyncMock, patch

    from sqlalchemy import delete, text

    from app.core import database
    from app.models.project import Project
    from app.models.team import Team
    from app.workers.ingest import sweep_pending_jobs

    project_id = None
    team_id = None
    stale_job_id = None
    fresh_job_id = None

    try:
        async with database.AsyncSessionLocal() as session:
            team = Team(name="Sweep Integration Team")
            session.add(team)
            await session.flush()

            project = Project(name="Sweep Integration Project", team_id=team.id, config={})
            session.add(project)
            await session.flush()
            project_id = project.id
            team_id = team.id

            stale_job = IngestionJob(
                project_id=project.id,
                status=JobStatus.PENDING.value,
            )
            session.add(stale_job)

            fresh_job = IngestionJob(
                project_id=project.id,
                status=JobStatus.PENDING.value,
            )
            session.add(fresh_job)
            await session.flush()
            stale_job_id = stale_job.id
            fresh_job_id = fresh_job.id
            await session.commit()

        # Back-date the stale job to 15 minutes ago
        async with database.AsyncSessionLocal() as session:
            await session.execute(
                text(
                    "UPDATE ingestion_jobs SET created_at = now() - interval '15 minutes'"
                    " WHERE id = :id"
                ),
                {"id": stale_job_id},
            )
            await session.commit()

        mock_arq = AsyncMock()
        mock_arq.enqueue_job.return_value = object()  # non-None = successfully enqueued

        ctx: dict[str, object] = {"redis": mock_arq}

        with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
            await sweep_pending_jobs(ctx)

        enqueued_job_ids = {
            call.kwargs.get("job_id") for call in mock_arq.enqueue_job.call_args_list
        }

        assert stale_job_id in enqueued_job_ids, "Stale job must be re-enqueued"
        assert fresh_job_id not in enqueued_job_ids, "Fresh job must not be re-enqueued"

        # Verify the fixed _job_id format to prevent ARQ duplicate enqueue
        for call in mock_arq.enqueue_job.call_args_list:
            assert call.kwargs.get("_job_id") == f"ingest-{call.kwargs.get('job_id')}"

    finally:
        if project_id is not None:
            async with database.AsyncSessionLocal() as session:
                await session.execute(
                    delete(IngestionJob).where(IngestionJob.project_id == project_id)
                )
                await session.execute(delete(Project).where(Project.id == project_id))
                if team_id is not None:
                    await session.execute(delete(Team).where(Team.id == team_id))
                await session.commit()
