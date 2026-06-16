from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from qdrant_client.http.models import ScoredPoint

from app.core.exceptions import AccessDeniedError
from app.ingestion.embedder import Embedder, SparseEmbedder
from app.ingestion.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    document_id: UUID
    chunk_index: int
    text: str
    filename: str
    score: float
    project_id: UUID


async def retrieve(
    project_id: UUID,
    query_text: str,
    accessible_ids: list[UUID],
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks from a single project using hybrid dense + BM25."""
    if project_id not in accessible_ids:
        raise AccessDeniedError(f"Project {project_id} is not accessible")

    dense_vec, sparse_vec = await asyncio.gather(
        asyncio.to_thread(embedder.embed_query, query_text),
        asyncio.to_thread(sparse_embedder.embed_query, query_text),
    )

    points = await vector_store.hybrid_search(
        project_id=project_id,
        dense_query=dense_vec,
        sparse_query=sparse_vec,
        top_k=top_k,
    )

    return [_to_chunk(point, project_id) for point in points]


async def retrieve_multi(
    project_ids: list[UUID],
    query_text: str,
    accessible_ids: list[UUID],
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks across multiple projects.

    All requested project IDs must be in the accessible set; the call is all-or-
    nothing. Results from each project are merged and re-sorted by score. RRF
    scores from different collections are comparable only approximately; this is
    acceptable for v1 admin cross-project search.
    """
    for project_id in project_ids:
        if project_id not in accessible_ids:
            raise AccessDeniedError(f"Project {project_id} is not accessible")

    per_project_results = await asyncio.gather(
        *(
            retrieve(
                project_id=project_id,
                query_text=query_text,
                accessible_ids=accessible_ids,
                vector_store=vector_store,
                embedder=embedder,
                sparse_embedder=sparse_embedder,
                top_k=top_k,
            )
            for project_id in project_ids
        )
    )

    merged: list[RetrievedChunk] = []
    for results in per_project_results:
        merged.extend(results)

    merged.sort(key=lambda chunk: chunk.score, reverse=True)
    return merged[:top_k]


def _to_chunk(point: ScoredPoint, project_id: UUID) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        document_id=UUID(payload["document_id"]),
        chunk_index=payload["chunk_index"],
        text=payload["text"],
        filename=payload["filename"],
        score=point.score,
        project_id=project_id,
    )
