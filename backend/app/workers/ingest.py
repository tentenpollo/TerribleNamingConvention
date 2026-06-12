from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast
import uuid

from arq.connections import ArqRedis
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import logger
from app.ingestion.chunker import ChunkingStrategy, chunk_text
from app.ingestion.embedder import Embedder
from app.ingestion.parser import parse_file
from app.ingestion.summarizer import summarize_document
from app.ingestion.vector_store import VectorStore
from app.models.document import Document
from app.models.document_summary import DocumentSummary
from app.models.ingestion_job import IngestionJob, JobStatus
from app.models.project import Project


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
            project = await _fetch_project(session, project_id)

            parsed_text = parse_file(
                content=doc.raw_bytes,
                file_type=doc.file_type,
            )

            chunk_size = project.config.get("chunk_size", settings.default_chunk_size)
            chunk_overlap = project.config.get("chunk_overlap", settings.default_chunk_overlap)
            strategy = ChunkingStrategy(
                project.config.get("chunking_strategy", settings.default_chunking_strategy)
            )
            chunks = chunk_text(
                text=parsed_text,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
            for chunk in chunks:
                chunk.metadata["filename"] = doc.filename

            embedding_results = await asyncio.to_thread(embedder.embed, chunks)

            await vector_store.upsert(project_id, embedding_results, document_id)

            context_model = project.config.get("context_model", settings.litellm_context_model)
            summary_dict = await summarize_document(
                text=parsed_text,
                filename=doc.filename,
                model=context_model,
            )

            await session.execute(
                pg_insert(DocumentSummary)
                .values(
                    document_id=doc.id,
                    project_id=project_id,
                    summary=summary_dict,
                )
                .on_conflict_do_nothing(constraint="uq_document_summaries_document_id")
            )
            await session.commit()
            await _after_summary_commit()

            summary_count_result = await session.execute(
                select(func.count())
                .select_from(DocumentSummary)
                .where(
                    DocumentSummary.project_id == project_id,
                ),
            )
            summary_count = summary_count_result.scalar() or 0
            cag_threshold = project.config.get(
                "cag_rebuild_threshold", settings.default_cag_rebuild_threshold
            )
            if summary_count > 0 and summary_count % cag_threshold == 0:
                logger.info(
                    "CAG threshold reached, rebuild not yet implemented",
                    project_id=str(project_id),
                    summary_count=summary_count,
                    cag_threshold=cag_threshold,
                )

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


async def sweep_pending_jobs(ctx: dict[str, object]) -> None:
    arq_pool = cast(ArqRedis, ctx["redis"])
    cutoff = datetime.now(UTC) - timedelta(minutes=10)
    scanned = 0
    enqueued = 0

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(IngestionJob).where(
                IngestionJob.status == JobStatus.PENDING.value,
                IngestionJob.created_at < cutoff,
            )
        )
        jobs = list(result.scalars().all())

    for job in jobs:
        scanned += 1
        enqueue_result = await arq_pool.enqueue_job(
            "ingest_document",
            job_id=job.id,
            document_id=job.document_id,
            project_id=job.project_id,
            _job_id=f"ingest-{job.id}",
        )
        if enqueue_result is not None:
            enqueued += 1

    logger.info(
        "Pending ingestion job sweep completed",
        scanned=scanned,
        enqueued=enqueued,
    )


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


async def _fetch_project(session: AsyncSession, project_id: uuid.UUID) -> Project:
    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    return project


async def _mark_job_failed(session: AsyncSession, job_id: uuid.UUID, error_message: str) -> None:
    job = await _fetch_job(session, job_id)
    job.status = JobStatus.FAILED.value
    job.error_message = error_message
    await session.commit()


async def _after_summary_commit() -> None:
    return None
