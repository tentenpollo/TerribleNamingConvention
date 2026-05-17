from __future__ import annotations

from uuid import UUID

from arq.connections import ArqRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AccessDeniedError,
    IngestionJobNotFoundError,
    UnsupportedFileTypeError,
)
from app.models.document import Document, FileType
from app.models.ingestion_job import IngestionJob, JobStatus
from app.schemas.document import DocumentResponse, IngestionJobResponse

SUPPORTED_EXTENSIONS = {".md": FileType.MARKDOWN, ".txt": FileType.TXT, ".pdf": FileType.PDF}


class DocumentService:
    def __init__(self, session: AsyncSession, arq_pool: ArqRedis) -> None:
        self.session = session
        self.arq_pool = arq_pool

    async def upload(
        self,
        project_id: UUID,
        filename: str,
        file_type: FileType,
        content: bytes,
        accessible_ids: list[UUID],
    ) -> IngestionJobResponse:
        if project_id not in accessible_ids:
            raise AccessDeniedError(f"Project {project_id} is outside the user's access scope")

        extension = _get_extension(filename)
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"File type '{extension}' is not supported")

        raw_content = content.decode("utf-8")

        doc = Document(
            project_id=project_id,
            filename=filename,
            file_type=file_type.value,
            raw_content=raw_content,
        )
        self.session.add(doc)
        await self.session.flush()

        job = IngestionJob(
            project_id=project_id,
            document_id=doc.id,
            status=JobStatus.PENDING.value,
        )
        self.session.add(job)
        await self.session.commit()

        await self.arq_pool.enqueue_job(
            "ingest_document",
            job_id=job.id,
            document_id=doc.id,
            project_id=project_id,
        )

        return IngestionJobResponse.model_validate(job)

    async def get_job(self, job_id: UUID) -> IngestionJobResponse:
        job = await self.session.get(IngestionJob, job_id)
        if job is None:
            raise IngestionJobNotFoundError(f"Ingestion job {job_id} not found")
        return IngestionJobResponse.model_validate(job)

    async def list_documents(
        self,
        project_id: UUID,
        accessible_ids: list[UUID],
    ) -> list[DocumentResponse]:
        if project_id not in accessible_ids:
            raise AccessDeniedError(f"Project {project_id} is outside the user's access scope")

        result = await self.session.execute(
            select(Document)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at.desc()),
        )
        documents = list(result.scalars().all())
        return [DocumentResponse.model_validate(doc) for doc in documents]


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[1].lower()
