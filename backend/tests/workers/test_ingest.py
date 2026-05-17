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
from app.workers.ingest import ingest_document

_SUMMARY_MOCK_RETURN = {
    "key_points": ["test summary"],
    "decisions": [],
    "action_items": [],
    "people_mentioned": [],
    "topics": ["test"],
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
    doc.raw_content = "# Hello\n\nThis is test content for ingestion."
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
            async with session:
                await ingest_document(ctx, job_id, document_id, project_id)

    assert JobStatus.RUNNING.value in statuses_after_each_commit
    assert JobStatus.COMPLETE.value in statuses_after_each_commit
    running_index = statuses_after_each_commit.index(JobStatus.RUNNING.value)
    complete_index = statuses_after_each_commit.index(JobStatus.COMPLETE.value)
    assert running_index < complete_index


# ---------------------------------------------------------------------------
# Integration tests — require real Postgres, Redis, Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ingest_document_full_pipeline() -> None:
    from unittest.mock import patch

    from sqlalchemy import delete

    from app.core import database
    from app.core.qdrant import get_qdrant_client
    from app.ingestion.embedder import Embedder
    from app.ingestion.vector_store import VectorStore
    from app.models.document_summary import DocumentSummary
    from app.models.project import Project
    from app.models.team import Team

    with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
        with patch(
            "app.workers.ingest.summarize_document",
            return_value={"key_points": ["test"], "raw_text_fallback": False},
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
                    raw_content=(
                        "# Integration Test\n\nThis document verifies the full ingestion pipeline."
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
            vector_store = VectorStore(client=get_qdrant_client())

            ctx: dict[str, object] = {
                "embedder": embedder,
                "vector_store": vector_store,
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

    vector_store = VectorStore(client=get_qdrant_client())
    hits = await vector_store.search(project_id, query_vector=[0.0] * 384, top_k=10)
    assert len(hits) > 0

    first_payload = hits[0].payload
    assert first_payload is not None
    assert first_payload["document_id"] == str(document_id)
    assert first_payload["filename"] == "integration-test.md"

    await vector_store.delete_collection(project_id)

    async with database.AsyncSessionLocal() as session:
        await session.execute(delete(IngestionJob).where(IngestionJob.project_id == project_id))
        await session.execute(
            delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
        )
        await session.execute(delete(Document).where(Document.project_id == project_id))
        await session.execute(delete(Project).where(Project.id == project_id))
        await session.execute(delete(Team).where(Team.id == team_id))
        await session.commit()
