from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.core.exceptions import (
    AccessDeniedError,
    IngestionJobNotFoundError,
    UnsupportedFileTypeError,
)
from app.models.document import Document, FileType
from app.models.ingestion_job import IngestionJob, JobStatus
from app.services.document import DocumentService


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_arq_pool() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def document_service(
    mock_session: AsyncMock,
    mock_arq_pool: AsyncMock,
) -> DocumentService:
    return DocumentService(mock_session, mock_arq_pool)


@pytest.mark.asyncio
async def test_upload_creates_document_and_job_and_enqueues(
    document_service: DocumentService,
    mock_session: AsyncMock,
    mock_arq_pool: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    accessible_ids = [project_id]
    content = b"# Hello\n\nWorld"

    def capture_job(obj: Document | IngestionJob) -> None:
        obj.id = uuid.uuid4()
        if isinstance(obj, IngestionJob):
            obj.created_at = datetime.now(UTC)

    mock_session.add.side_effect = capture_job

    result = await document_service.upload(
        project_id=project_id,
        filename="test.md",
        file_type=FileType.MARKDOWN,
        content=content,
        accessible_ids=accessible_ids,
    )

    assert result.status == JobStatus.PENDING.value
    assert mock_session.flush.await_count == 1
    assert mock_session.commit.await_count == 1
    mock_arq_pool.enqueue_job.assert_awaited_once()
    call_kwargs = mock_arq_pool.enqueue_job.call_args.kwargs
    assert call_kwargs["job_id"] is not None
    assert call_kwargs["document_id"] is not None
    assert call_kwargs["project_id"] == project_id


@pytest.mark.asyncio
async def test_upload_access_denied_when_out_of_scope(
    document_service: DocumentService,
) -> None:
    project_id = uuid.uuid4()
    accessible_ids: list[uuid.UUID] = []

    with pytest.raises(AccessDeniedError):
        await document_service.upload(
            project_id=project_id,
            filename="test.md",
            file_type=FileType.MARKDOWN,
            content=b"content",
            accessible_ids=accessible_ids,
        )


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(
    document_service: DocumentService,
) -> None:
    project_id = uuid.uuid4()
    accessible_ids = [project_id]

    with pytest.raises(UnsupportedFileTypeError):
        await document_service.upload(
            project_id=project_id,
            filename="test.exe",
            file_type=FileType.MARKDOWN,
            content=b"content",
            accessible_ids=accessible_ids,
        )


@pytest.mark.asyncio
async def test_get_job_returns_job(
    document_service: DocumentService,
    mock_session: AsyncMock,
) -> None:
    job_id = uuid.uuid4()
    job = _make_job(job_id)
    mock_session.get.return_value = job

    result = await document_service.get_job(job_id)

    assert result.id == job_id
    assert result.status == JobStatus.PENDING.value


@pytest.mark.asyncio
async def test_get_job_not_found_raises(
    document_service: DocumentService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None

    with pytest.raises(IngestionJobNotFoundError):
        await document_service.get_job(uuid.uuid4())


@pytest.mark.asyncio
async def test_list_documents_returns_documents(
    document_service: DocumentService,
    mock_session: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    accessible_ids = [project_id]
    docs = [_make_document(project_id, "a.md"), _make_document(project_id, "b.txt")]
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = docs
    mock_session.execute.return_value = result_mock

    returned = await document_service.list_documents(project_id, accessible_ids)

    assert len(returned) == 2
    assert returned[0].filename == "a.md"
    assert returned[1].filename == "b.txt"


@pytest.mark.asyncio
async def test_list_documents_access_denied_when_out_of_scope(
    document_service: DocumentService,
) -> None:
    project_id = uuid.uuid4()
    accessible_ids: list[uuid.UUID] = []

    with pytest.raises(AccessDeniedError):
        await document_service.list_documents(project_id, accessible_ids)


def _make_job(job_id: uuid.UUID) -> IngestionJob:
    return IngestionJob(
        id=job_id,
        project_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        status=JobStatus.PENDING.value,
        error_message=None,
        created_at=datetime.now(UTC),
        completed_at=None,
    )


def _make_document(project_id: uuid.UUID, filename: str) -> Document:
    return Document(
        id=uuid.uuid4(),
        project_id=project_id,
        filename=filename,
        file_type=FileType.MARKDOWN.value,
        raw_content="content",
        created_at=datetime.now(UTC),
    )
