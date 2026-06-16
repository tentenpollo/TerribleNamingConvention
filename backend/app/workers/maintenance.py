from __future__ import annotations

from typing import cast
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.logging import logger
from app.ingestion.vector_store import VectorStore
from app.models.document import Document
from app.workers.ingest import _fetch_project, _index_document


async def reindex_project(ctx: dict[str, object], project_id: uuid.UUID) -> None:
    """ARQ maintenance job: rebuild a project's Qdrant collection from scratch.

    Deletes the existing collection, recreates it with the current schema (named
    dense vector + sparse bm25), and re-indexes every document in the project.
    Document summaries are intentionally not touched — the event log is already
    correct.

    This job exists so pre-v2 or pre-sparse collections can be upgraded without
    re-uploading files. It is enqueueable from a shell for now; use the fixed
    job ID to deduplicate concurrent reindexes:

        arq app.workers.settings.WorkerSettings
        # enqueue with _job_id=f"reindex-{project_id}"
    """
    vector_store = cast(VectorStore, ctx["vector_store"])

    async with AsyncSessionLocal() as session:
        project = await _fetch_project(session, project_id)
        documents = await _fetch_project_documents(session, project_id)

    await vector_store.delete_collection(project_id)
    await vector_store.ensure_collection(project_id)

    for doc in documents:
        embedding_results, sparse_embedding_results = await _index_document(
            ctx,
            doc,
            project,
        )
        await vector_store.upsert(
            project_id,
            embedding_results,
            doc.id,
            sparse_embedding_results,
        )

    logger.info(
        "Project reindex completed",
        project_id=str(project_id),
        document_count=len(documents),
    )


async def _fetch_project_documents(
    session: AsyncSession,
    project_id: uuid.UUID,
) -> list[Document]:
    result = await session.execute(select(Document).where(Document.project_id == project_id))
    return list(result.scalars().all())
