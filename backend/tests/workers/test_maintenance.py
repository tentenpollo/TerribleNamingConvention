from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from qdrant_client.http.models import SparseVector

from app.ingestion.embedder import EmbeddingResult, SparseEmbedder, SparseEmbeddingResult
from app.ingestion.vector_store import VectorStore
from app.models.document import Document, FileType
from app.models.project import Project
from app.workers.maintenance import reindex_project


def _make_mock_document(doc_id: uuid.UUID, project_id: uuid.UUID) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = doc_id
    doc.project_id = project_id
    doc.filename = "test.md"
    doc.file_type = FileType.MARKDOWN.value
    doc.raw_bytes = b"# Test\n\ncontent"
    return doc


def _make_mock_project(project_id: uuid.UUID) -> Project:
    project = MagicMock(spec=Project)
    project.id = project_id
    project.config = {}
    return project


@pytest.mark.unit
async def test_reindex_project_deletes_recreate_and_upserts_all_documents() -> None:
    project_id = uuid.uuid4()
    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    doc_a = _make_mock_document(doc_a_id, project_id)
    doc_b = _make_mock_document(doc_b_id, project_id)
    project = _make_mock_project(project_id)

    vector_store = AsyncMock(spec=VectorStore)

    embedder = MagicMock()
    sparse_embedder = MagicMock(spec=SparseEmbedder)
    ctx: dict[str, object] = {
        "embedder": embedder,
        "sparse_embedder": sparse_embedder,
        "vector_store": vector_store,
    }

    chunk_a = MagicMock()
    chunk_a.index = 0
    chunk_a.text = "content"
    chunk_a.metadata = {"filename": "test.md"}

    embedding_result = EmbeddingResult(chunk=chunk_a, vector=[0.1] * 384)
    sparse_result = SparseEmbeddingResult(
        chunk=chunk_a,
        vector=SparseVector(indices=[1], values=[0.5]),
    )

    with patch("app.workers.maintenance.AsyncSessionLocal") as mock_session_cls:
        session = AsyncMock()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        project_result = MagicMock()
        project_result.scalar_one_or_none.return_value = project
        document_result = MagicMock()
        document_result.scalars.return_value.all.return_value = [doc_a, doc_b]

        def side_effect(statement):
            entity = getattr(statement, "column_descriptions", None)
            if entity is not None and entity[0]["entity"] is Project:
                return project_result
            return document_result

        session.execute = AsyncMock(side_effect=side_effect)

        with patch(
            "app.workers.maintenance._index_document",
            return_value=([embedding_result], [sparse_result]),
        ):
            await reindex_project(ctx, project_id)

    vector_store.delete_collection.assert_awaited_once_with(project_id)
    vector_store.ensure_collection.assert_awaited_once_with(project_id)
    assert vector_store.upsert.await_count == 2
