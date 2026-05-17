from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.logging import logger
from app.ingestion.chunker import ChunkingStrategy, chunk_text
from app.ingestion.embedder import Embedder
from app.ingestion.parser import parse_file
from app.ingestion.vector_store import VectorStore
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, JobStatus


async def ingest_document(
    ctx: dict[str, object],
    job_id: uuid.UUID,
    document_id: uuid.UUID,
    project_id: uuid.UUID,
) -> None:
    embedder = cast(Embedder, ctx["embedder"])
    vector_store = cast(VectorStore, ctx["vector_store"])

    async with AsyncSessionLocal() as session:
        try:
            job = await _fetch_job(session, job_id)
            job.status = JobStatus.RUNNING.value
            await session.commit()

            doc = await _fetch_document(session, document_id)

            parsed_text = parse_file(
                content=doc.raw_content.encode(),
                file_type=doc.file_type,
            )

            chunks = chunk_text(
                text=parsed_text,
                strategy=ChunkingStrategy.NAIVE,
                chunk_size=512,
                overlap=64,
            )
            for chunk in chunks:
                chunk.metadata["filename"] = doc.filename

            embedding_results = embedder.embed(chunks)

            await vector_store.upsert(project_id, embedding_results, document_id)

            job.status = JobStatus.COMPLETE.value
            job.completed_at = datetime.now(UTC)
            await session.commit()

            logger.info(
                "Ingestion job completed",
                job_id=str(job_id),
                document_id=str(document_id),
                project_id=str(project_id),
                chunk_count=len(chunks),
            )

        except Exception as exc:
            await _mark_job_failed(session, job_id, str(exc))
            logger.error(
                "Ingestion job failed",
                job_id=str(job_id),
                document_id=str(document_id),
                project_id=str(project_id),
                error=str(exc),
            )
            raise


async def _fetch_job(session: AsyncSession, job_id: uuid.UUID) -> IngestionJob:
    result = await session.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise ValueError(f"IngestionJob {job_id} not found")
    return job


async def _fetch_document(session: AsyncSession, document_id: uuid.UUID) -> Document:
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise ValueError(f"Document {document_id} not found")
    return doc


async def _mark_job_failed(session: AsyncSession, job_id: uuid.UUID, error_message: str) -> None:
    job = await _fetch_job(session, job_id)
    job.status = JobStatus.FAILED.value
    job.error_message = error_message
    await session.commit()
