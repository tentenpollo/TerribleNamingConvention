from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    QueryResponse,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.exceptions import QdrantError
from app.ingestion.embedder import EmbeddingResult, SparseEmbeddingResult

CHUNK_NAMESPACE = uuid.UUID("7f6ae9bc-2d42-4e2c-9d98-5f79a76c8394")


def collection_name(project_id: uuid.UUID) -> str:
    return f"project_{project_id}"


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, vector_size: int = 384) -> None:
        self._client = client
        self._vector_size = vector_size

    async def ensure_collection(self, project_id: uuid.UUID) -> None:
        name = collection_name(project_id)
        try:
            exists = await self._client.collection_exists(name)
            if not exists:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self._vector_size,
                            distance=Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(modifier=Modifier.IDF),
                    },
                )
        except Exception as exc:
            raise QdrantError(
                f"Failed to ensure collection for project {project_id}: {exc}"
            ) from exc

    async def upsert(
        self,
        project_id: uuid.UUID,
        results: list[EmbeddingResult],
        document_id: uuid.UUID,
        sparse_results: list[SparseEmbeddingResult] | None = None,
    ) -> None:
        await self.ensure_collection(project_id)

        now = datetime.now(UTC).isoformat()
        sparse_by_index = {}
        if sparse_results:
            sparse_by_index = {r.chunk.index: r.vector for r in sparse_results}

        points: list[PointStruct] = []
        for result in results:
            vector: dict[str, Any] = {"dense": result.vector}
            sparse_vector = sparse_by_index.get(result.chunk.index)
            if sparse_vector is not None:
                vector["bm25"] = sparse_vector

            points.append(
                PointStruct(
                    id=str(uuid.uuid5(CHUNK_NAMESPACE, f"{document_id}:{result.chunk.index}")),
                    vector=vector,
                    payload={
                        "document_id": str(document_id),
                        "project_id": str(project_id),
                        "chunk_index": result.chunk.index,
                        "text": result.chunk.text,
                        "filename": result.chunk.metadata.get("filename", ""),
                        "created_at": now,
                    },
                )
            )

        name = collection_name(project_id)
        try:
            await self._client.upsert(collection_name=name, points=points)
        except Exception as exc:
            raise QdrantError(f"Failed to upsert vectors for project {project_id}: {exc}") from exc

    async def search(
        self,
        project_id: uuid.UUID,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[ScoredPoint]:
        name = collection_name(project_id)
        try:
            exists = await self._client.collection_exists(name)
            if not exists:
                return []
            response: QueryResponse = await self._client.query_points(
                collection_name=name,
                query=query_vector,
                using="dense",
                limit=top_k,
            )
            return response.points
        except QdrantError:
            raise
        except Exception as exc:
            raise QdrantError(
                f"Failed to search collection for project {project_id}: {exc}"
            ) from exc

    async def hybrid_search(
        self,
        project_id: uuid.UUID,
        dense_query: list[float],
        sparse_query: SparseVector,
        top_k: int = 8,
    ) -> list[ScoredPoint]:
        name = collection_name(project_id)
        try:
            exists = await self._client.collection_exists(name)
            if not exists:
                return []

            response: QueryResponse = await self._client.query_points(
                collection_name=name,
                prefetch=[
                    Prefetch(query=dense_query, using="dense", limit=top_k * 3),
                    Prefetch(query=sparse_query, using="bm25", limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return response.points
        except QdrantError:
            raise
        except Exception as exc:
            raise QdrantError(
                f"Failed to hybrid search collection for project {project_id}: {exc}"
            ) from exc

    async def delete_collection(self, project_id: uuid.UUID) -> None:
        name = collection_name(project_id)
        try:
            exists = await self._client.collection_exists(name)
            if exists:
                await self._client.delete_collection(collection_name=name)
        except Exception as exc:
            raise QdrantError(
                f"Failed to delete collection for project {project_id}: {exc}"
            ) from exc
