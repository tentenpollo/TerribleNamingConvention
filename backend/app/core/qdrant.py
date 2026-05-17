from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.core.config import settings

_client = AsyncQdrantClient(url=settings.qdrant_url)


def get_qdrant_client() -> AsyncQdrantClient:
    return _client
