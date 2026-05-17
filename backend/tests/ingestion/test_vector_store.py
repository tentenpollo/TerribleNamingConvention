from __future__ import annotations

import uuid

import pytest

from app.core.qdrant import get_qdrant_client
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import EmbeddingResult
from app.ingestion.vector_store import VectorStore


@pytest.fixture
def vector_store() -> VectorStore:
    return VectorStore(client=get_qdrant_client())


@pytest.fixture
def project_a() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def project_b() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def document_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def sample_embedding_results() -> list[EmbeddingResult]:
    return [
        EmbeddingResult(
            chunk=Chunk(text="hello world", index=0, metadata={"filename": "test.md"}),
            vector=[0.1] * 384,
        ),
        EmbeddingResult(
            chunk=Chunk(text="goodbye world", index=1, metadata={"filename": "test.md"}),
            vector=[0.2] * 384,
        ),
    ]


@pytest.fixture(autouse=True)
async def cleanup(vector_store: VectorStore, project_a: uuid.UUID, project_b: uuid.UUID) -> None:
    yield
    await vector_store.delete_collection(project_a)
    await vector_store.delete_collection(project_b)


@pytest.mark.integration
async def test_ensure_collection_creates_new_collection(
    vector_store: VectorStore,
    project_a: uuid.UUID,
) -> None:
    await vector_store.ensure_collection(project_a)

    exists = await vector_store._client.collection_exists(f"project_{project_a}")
    assert exists is True


@pytest.mark.integration
async def test_ensure_collection_is_idempotent(
    vector_store: VectorStore,
    project_a: uuid.UUID,
) -> None:
    await vector_store.ensure_collection(project_a)
    await vector_store.ensure_collection(project_a)

    exists = await vector_store._client.collection_exists(f"project_{project_a}")
    assert exists is True


@pytest.mark.integration
async def test_upsert_and_search_roundtrip(
    vector_store: VectorStore,
    project_a: uuid.UUID,
    document_id: uuid.UUID,
    sample_embedding_results: list[EmbeddingResult],
) -> None:
    await vector_store.upsert(project_a, sample_embedding_results, document_id)

    hits = await vector_store.search(
        project_a,
        query_vector=[0.1] * 384,
        top_k=5,
    )

    assert len(hits) == 2
    assert hits[0].payload is not None
    assert hits[0].payload["document_id"] == str(document_id)
    assert hits[0].payload["project_id"] == str(project_a)
    assert "text" in hits[0].payload
    assert "filename" in hits[0].payload
    assert "chunk_index" in hits[0].payload
    assert "created_at" in hits[0].payload


@pytest.mark.integration
async def test_search_returns_correct_payload(
    vector_store: VectorStore,
    project_a: uuid.UUID,
    document_id: uuid.UUID,
    sample_embedding_results: list[EmbeddingResult],
) -> None:
    await vector_store.upsert(project_a, sample_embedding_results, document_id)

    hits = await vector_store.search(
        project_a,
        query_vector=[0.1] * 384,
        top_k=1,
    )

    assert len(hits) == 1
    payload = hits[0].payload
    assert payload is not None
    assert payload["text"] in ("hello world", "goodbye world")
    assert payload["filename"] == "test.md"
    assert payload["chunk_index"] in (0, 1)


@pytest.mark.integration
async def test_delete_collection_removes_collection(
    vector_store: VectorStore,
    project_a: uuid.UUID,
    document_id: uuid.UUID,
    sample_embedding_results: list[EmbeddingResult],
) -> None:
    await vector_store.upsert(project_a, sample_embedding_results, document_id)
    await vector_store.delete_collection(project_a)

    exists = await vector_store._client.collection_exists(f"project_{project_a}")
    assert exists is False


@pytest.mark.integration
async def test_project_isolation(
    vector_store: VectorStore,
    project_a: uuid.UUID,
    project_b: uuid.UUID,
    document_id: uuid.UUID,
    sample_embedding_results: list[EmbeddingResult],
) -> None:
    await vector_store.upsert(project_a, sample_embedding_results, document_id)

    hits = await vector_store.search(
        project_b,
        query_vector=[0.1] * 384,
        top_k=5,
    )

    assert len(hits) == 0
