from __future__ import annotations

from unittest.mock import AsyncMock
import uuid

import pytest

from app.core.exceptions import QdrantError
from app.core.qdrant import get_qdrant_client
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import EmbeddingResult
from app.ingestion.vector_store import VectorStore, collection_name


@pytest.fixture
def vector_store() -> VectorStore:
    return VectorStore(client=get_qdrant_client())


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def store(mock_client: AsyncMock) -> VectorStore:
    return VectorStore(client=mock_client)


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


def test_collection_name_format(project_a: uuid.UUID) -> None:
    assert collection_name(project_a) == f"project_{project_a}"


@pytest.mark.unit
async def test_ensure_collection_creates_when_not_exists(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.return_value = False

    await store.ensure_collection(project_a)

    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == f"project_{project_a}"


@pytest.mark.unit
async def test_ensure_collection_skips_when_exists(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.return_value = True

    await store.ensure_collection(project_a)

    mock_client.create_collection.assert_not_called()


@pytest.mark.unit
async def test_ensure_collection_raises_qdrant_error_on_failure(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.side_effect = Exception("connection failed")

    with pytest.raises(QdrantError, match="Failed to ensure collection"):
        await store.ensure_collection(project_a)


@pytest.mark.unit
async def test_upsert_ensures_collection_and_inserts_points(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
    document_id: uuid.UUID,
    sample_embedding_results: list[EmbeddingResult],
) -> None:
    mock_client.collection_exists.return_value = False

    await store.upsert(project_a, sample_embedding_results, document_id)

    mock_client.create_collection.assert_called_once()
    mock_client.upsert.assert_called_once()
    call_kwargs = mock_client.upsert.call_args[1]
    assert call_kwargs["collection_name"] == f"project_{project_a}"
    assert len(call_kwargs["points"]) == 2
    assert call_kwargs["points"][0].payload["document_id"] == str(document_id)
    assert call_kwargs["points"][0].payload["text"] == "hello world"


@pytest.mark.unit
async def test_upsert_raises_qdrant_error_on_failure(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
    document_id: uuid.UUID,
    sample_embedding_results: list[EmbeddingResult],
) -> None:
    mock_client.collection_exists.return_value = True
    mock_client.upsert.side_effect = Exception("upsert failed")

    with pytest.raises(QdrantError, match="Failed to upsert"):
        await store.upsert(project_a, sample_embedding_results, document_id)


@pytest.mark.unit
async def test_search_returns_empty_when_collection_missing(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.return_value = False

    result = await store.search(project_a, query_vector=[0.1] * 384)

    assert result == []


@pytest.mark.unit
async def test_search_returns_points_when_collection_exists(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.return_value = True
    mock_response = AsyncMock()
    mock_response.points = [AsyncMock(score=0.95), AsyncMock(score=0.85)]
    mock_client.query_points.return_value = mock_response

    result = await store.search(project_a, query_vector=[0.1] * 384, top_k=2)

    assert len(result) == 2
    mock_client.query_points.assert_called_once()


@pytest.mark.unit
async def test_search_raises_qdrant_error_on_failure(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.side_effect = Exception("search failed")

    with pytest.raises(QdrantError, match="Failed to search"):
        await store.search(project_a, query_vector=[0.1] * 384)


@pytest.mark.unit
async def test_delete_collection_removes_when_exists(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.return_value = True

    await store.delete_collection(project_a)

    mock_client.delete_collection.assert_called_once()


@pytest.mark.unit
async def test_delete_collection_skips_when_not_exists(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.return_value = False

    await store.delete_collection(project_a)

    mock_client.delete_collection.assert_not_called()


@pytest.mark.unit
async def test_delete_collection_raises_qdrant_error_on_failure(
    store: VectorStore,
    mock_client: AsyncMock,
    project_a: uuid.UUID,
) -> None:
    mock_client.collection_exists.side_effect = Exception("delete failed")

    with pytest.raises(QdrantError, match="Failed to delete"):
        await store.delete_collection(project_a)


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
