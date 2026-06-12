from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models.document import Document, FileType
from app.models.document_summary import DocumentSummary
from app.models.ingestion_job import IngestionJob, JobStatus


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_retrieve_document(db_session, test_project) -> None:
    doc = Document(
        project_id=test_project.id,
        filename="report.pdf",
        file_type=FileType.PDF.value,
        raw_bytes=b"PDF content",
    )
    db_session.add(doc)
    await db_session.flush()

    result = await db_session.execute(select(Document).where(Document.id == doc.id))
    retrieved = result.scalar_one()

    assert retrieved.id == doc.id
    assert retrieved.filename == "report.pdf"
    assert retrieved.file_type == FileType.PDF.value
    assert retrieved.project_id == test_project.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_retrieve_document_summary(db_session, test_document) -> None:
    summary = DocumentSummary(
        document_id=test_document.id,
        project_id=test_document.project_id,
        summary={"key_points": ["point one"], "topics": ["testing"]},
    )
    db_session.add(summary)
    await db_session.flush()

    result = await db_session.execute(
        select(DocumentSummary).where(DocumentSummary.id == summary.id),
    )
    retrieved = result.scalar_one()

    assert retrieved.id == summary.id
    assert retrieved.document_id == test_document.id
    assert retrieved.summary["key_points"] == ["point one"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_retrieve_ingestion_job(db_session, test_project) -> None:
    job = IngestionJob(
        project_id=test_project.id,
        document_id=None,
    )
    db_session.add(job)
    await db_session.flush()

    result = await db_session.execute(select(IngestionJob).where(IngestionJob.id == job.id))
    retrieved = result.scalar_one()

    assert retrieved.id == job.id
    assert retrieved.project_id == test_project.id
    assert retrieved.status == JobStatus.PENDING.value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_summary_linked_to_document(db_session, test_document) -> None:
    summary = DocumentSummary(
        document_id=test_document.id,
        project_id=test_document.project_id,
        summary={"key_points": []},
    )
    db_session.add(summary)
    await db_session.flush()

    result = await db_session.execute(
        select(DocumentSummary).where(DocumentSummary.document_id == test_document.id),
    )
    summaries = result.scalars().all()

    assert len(summaries) == 1
    assert summaries[0].id == summary.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingestion_job_status_defaults_to_pending(db_session, test_project) -> None:
    job = IngestionJob(
        project_id=test_project.id,
    )
    db_session.add(job)
    await db_session.flush()

    result = await db_session.execute(select(IngestionJob).where(IngestionJob.id == job.id))
    retrieved = result.scalar_one()

    assert retrieved.status == JobStatus.PENDING.value
