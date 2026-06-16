from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.embedder import SparseEmbedder, SparseEmbeddingResult


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(text="first chunk", index=0),
        Chunk(text="second chunk", index=1),
    ]


@pytest.mark.unit
def test_sparse_embed_returns_one_result_per_chunk(sample_chunks: list[Chunk]) -> None:
    with patch("app.ingestion.embedder.SparseTextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_sparse = MagicMock()
        mock_sparse.indices.tolist.return_value = [1, 2]
        mock_sparse.values.tolist.return_value = [0.5, 0.5]
        mock_model.embed.return_value = [mock_sparse, mock_sparse]

        embedder = SparseEmbedder()
        results = embedder.embed(sample_chunks)

        assert len(results) == 2
        assert all(isinstance(r, SparseEmbeddingResult) for r in results)


@pytest.mark.unit
def test_sparse_embed_indices_and_values_equal_length(sample_chunks: list[Chunk]) -> None:
    with patch("app.ingestion.embedder.SparseTextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_sparse = MagicMock()
        mock_sparse.indices.tolist.return_value = [10, 20, 30]
        mock_sparse.values.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = [mock_sparse]

        embedder = SparseEmbedder()
        results = embedder.embed(sample_chunks[:1])

        assert len(results[0].vector.indices) == len(results[0].vector.values)
        assert results[0].vector.indices == [10, 20, 30]
        assert results[0].vector.values == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_sparse_embed_query_returns_sparse_vector() -> None:
    with patch("app.ingestion.embedder.SparseTextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_sparse = MagicMock()
        mock_sparse.indices.tolist.return_value = [1]
        mock_sparse.values.tolist.return_value = [0.75]
        mock_model.query_embed.return_value = iter([mock_sparse])

        embedder = SparseEmbedder()
        result = embedder.embed_query("test query")

        assert result.indices == [1]
        assert result.values == [0.75]


@pytest.mark.unit
def test_sparse_embed_empty_string_handled() -> None:
    with patch("app.ingestion.embedder.SparseTextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_sparse = MagicMock()
        mock_sparse.indices.tolist.return_value = []
        mock_sparse.values.tolist.return_value = []
        mock_model.embed.return_value = [mock_sparse]

        embedder = SparseEmbedder()
        results = embedder.embed([Chunk(text="", index=0)])

        assert len(results) == 1
        assert results[0].vector.indices == []
        assert results[0].vector.values == []
