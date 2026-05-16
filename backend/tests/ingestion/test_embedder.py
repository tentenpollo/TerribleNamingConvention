from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.embedder import Embedder, EmbeddingResult


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(text="first chunk", index=0),
        Chunk(text="second chunk", index=1),
    ]


@pytest.mark.unit
def test_embed_returns_one_result_per_chunk(sample_chunks: list[Chunk]) -> None:
    with patch("app.ingestion.embedder.TextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_model.embed.return_value = [
            np.array([0.1] * 384),
            np.array([0.2] * 384),
        ]

        embedder = Embedder()
        results = embedder.embed(sample_chunks)

        assert len(results) == 2
        assert all(isinstance(r, EmbeddingResult) for r in results)


@pytest.mark.unit
def test_embed_vector_is_list_of_float(sample_chunks: list[Chunk]) -> None:
    with patch("app.ingestion.embedder.TextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_model.embed.return_value = [np.array([0.1, 0.2, 0.3])]

        embedder = Embedder()
        results = embedder.embed(sample_chunks[:1])

        assert isinstance(results[0].vector, list)
        assert all(isinstance(v, float) for v in results[0].vector)


@pytest.mark.unit
def test_embed_correct_dimensionality(sample_chunks: list[Chunk]) -> None:
    with patch("app.ingestion.embedder.TextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_model.embed.return_value = [np.array([0.1] * 384)]

        embedder = Embedder()
        results = embedder.embed(sample_chunks[:1])

        assert len(results[0].vector) == 384


@pytest.mark.unit
def test_embed_query_returns_list_of_float() -> None:
    with patch("app.ingestion.embedder.TextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_model.query_embed.return_value = iter([np.array([0.1] * 384)])

        embedder = Embedder()
        result = embedder.embed_query("test query")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        assert len(result) == 384


@pytest.mark.unit
def test_embed_preserves_chunk_reference(sample_chunks: list[Chunk]) -> None:
    with patch("app.ingestion.embedder.TextEmbedding") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_model.embed.return_value = [
            np.array([0.1] * 384),
            np.array([0.2] * 384),
        ]

        embedder = Embedder()
        results = embedder.embed(sample_chunks)

        assert results[0].chunk is sample_chunks[0]
        assert results[1].chunk is sample_chunks[1]


@pytest.mark.slow
@pytest.mark.integration
def test_embedder_real_model(sample_chunks: list[Chunk]) -> None:
    embedder = Embedder()
    results = embedder.embed(sample_chunks)

    assert len(results) == 2
    for result in results:
        assert isinstance(result.vector, list)
        assert len(result.vector) == 384
        assert all(isinstance(v, float) for v in result.vector)


@pytest.mark.slow
@pytest.mark.integration
def test_embed_query_real_model() -> None:
    embedder = Embedder()
    result = embedder.embed_query("test query")

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)
