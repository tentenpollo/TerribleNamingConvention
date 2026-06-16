from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.core.exceptions import AccessDeniedError
from app.ingestion.embedder import Embedder, SparseEmbedder
from app.ingestion.vector_store import VectorStore
from app.retrieval.retriever import RetrievedChunk, retrieve, retrieve_multi


@pytest.fixture
def embedder() -> MagicMock:
    mock = MagicMock(spec=Embedder)
    mock.embed_query.return_value = [0.1] * 384
    return mock


@pytest.fixture
def sparse_embedder() -> MagicMock:
    mock = MagicMock(spec=SparseEmbedder)
    from qdrant_client.http.models import SparseVector

    mock.embed_query.return_value = SparseVector(indices=[1], values=[0.5])
    return mock


@pytest.fixture
def vector_store() -> AsyncMock:
    return AsyncMock(spec=VectorStore)


@pytest.mark.unit
async def test_retrieve_raises_access_denied_without_qdrant_call(
    embedder: MagicMock,
    sparse_embedder: MagicMock,
    vector_store: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    accessible_ids = [uuid.uuid4()]

    with pytest.raises(AccessDeniedError):
        await retrieve(
            project_id=project_id,
            query_text="test query",
            accessible_ids=accessible_ids,
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
        )

    vector_store.hybrid_search.assert_not_awaited()
    embedder.embed_query.assert_not_called()
    sparse_embedder.embed_query.assert_not_called()


@pytest.mark.unit
async def test_retrieve_multi_rejects_when_any_project_out_of_scope(
    embedder: MagicMock,
    sparse_embedder: MagicMock,
    vector_store: AsyncMock,
) -> None:
    in_scope = uuid.uuid4()
    out_of_scope = uuid.uuid4()

    with pytest.raises(AccessDeniedError):
        await retrieve_multi(
            project_ids=[in_scope, out_of_scope],
            query_text="test query",
            accessible_ids=[in_scope],
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
        )

    vector_store.hybrid_search.assert_not_awaited()


@pytest.mark.unit
async def test_retrieve_returns_chunks_from_scored_points(
    embedder: MagicMock,
    sparse_embedder: MagicMock,
    vector_store: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    vector_store.hybrid_search.return_value = [
        MagicMock(
            score=0.95,
            payload={
                "document_id": str(document_id),
                "chunk_index": 3,
                "text": "matched chunk",
                "filename": "notes.md",
            },
        ),
    ]

    results = await retrieve(
        project_id=project_id,
        query_text="test query",
        accessible_ids=[project_id],
        vector_store=vector_store,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
    )

    assert len(results) == 1
    assert results[0] == RetrievedChunk(
        document_id=document_id,
        chunk_index=3,
        text="matched chunk",
        filename="notes.md",
        score=0.95,
        project_id=project_id,
    )


@pytest.mark.unit
async def test_retrieve_multi_merges_and_truncates_to_top_k(
    embedder: MagicMock,
    sparse_embedder: MagicMock,
    vector_store: AsyncMock,
) -> None:
    project_a = uuid.uuid4()
    project_b = uuid.uuid4()

    def make_point(project_id: uuid.UUID, score: float, index: int) -> MagicMock:
        point = MagicMock()
        point.score = score
        point.payload = {
            "document_id": str(uuid.uuid4()),
            "chunk_index": index,
            "text": f"chunk {project_id}-{index}",
            "filename": "notes.md",
        }
        return point

    vector_store.hybrid_search.side_effect = [
        [make_point(project_a, 0.9, 0), make_point(project_a, 0.7, 1)],
        [make_point(project_b, 0.8, 0)],
    ]

    results = await retrieve_multi(
        project_ids=[project_a, project_b],
        query_text="test query",
        accessible_ids=[project_a, project_b],
        vector_store=vector_store,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert {results[0].project_id, results[1].project_id} == {project_a, project_b}
