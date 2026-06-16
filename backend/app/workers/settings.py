from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.embedder import Embedder, SparseEmbedder
from app.ingestion.vector_store import VectorStore
from app.workers.cag import cag_rebuild, cag_update, cag_weekly_rebuild
from app.workers.ingest import ingest_document, sweep_pending_jobs
from app.workers.maintenance import reindex_project


async def on_startup(ctx: dict[str, Any]) -> None:
    logger.info("ARQ worker starting")
    ctx["embedder"] = Embedder()
    ctx["sparse_embedder"] = SparseEmbedder()
    ctx["vector_store"] = VectorStore(client=AsyncQdrantClient(url=settings.qdrant_url))


async def on_shutdown(ctx: dict[str, Any]) -> None:
    logger.info("ARQ worker shutting down")
    vector_store = ctx.get("vector_store")
    if vector_store is not None and isinstance(vector_store, VectorStore):
        await vector_store._client.close()


class WorkerSettings:
    redis_settings: ClassVar[RedisSettings] = RedisSettings.from_dsn(settings.redis_url)
    functions: ClassVar[list[Callable[..., Any]]] = [
        ingest_document,
        sweep_pending_jobs,
        reindex_project,
        cag_update,
        cag_rebuild,
        cag_weekly_rebuild,
    ]
    cron_jobs: ClassVar[list[Any]] = [
        cron(sweep_pending_jobs, minute=set(range(0, 60, 5))),
        cron(cag_weekly_rebuild, weekday=6, hour=3, minute=0),
    ]
    max_jobs: ClassVar[int] = settings.arq_max_jobs
    on_startup: ClassVar[Callable[..., Any]] = on_startup
    on_shutdown: ClassVar[Callable[..., Any]] = on_shutdown
