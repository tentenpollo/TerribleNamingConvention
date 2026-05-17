from __future__ import annotations

from datetime import UTC, datetime
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    QueryResponse,
    ScoredPoint,
    VectorParams,
)

from app.core.exceptions import QdrantError
from app.ingestion.embedder import EmbeddingResult


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
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
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
    ) -> None:
        await self.ensure_collection(project_id)

        now = datetime.now(UTC).isoformat()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=result.vector,
                payload={
                    "document_id": str(document_id),
                    "project_id": str(project_id),
                    "chunk_index": result.chunk.index,
                    "text": result.chunk.text,
                    "filename": result.chunk.metadata.get("filename", ""),
                    "created_at": now,
                },
            )
            for result in results
        ]

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
                limit=top_k,
            )
            return response.points
        except QdrantError:
            raise
        except Exception as exc:
            raise QdrantError(
                f"Failed to search collection for project {project_id}: {exc}"
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
