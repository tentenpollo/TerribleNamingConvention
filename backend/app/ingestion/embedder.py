from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastembed import TextEmbedding
import numpy as np

from app.ingestion.chunker import Chunk


@dataclass
class EmbeddingResult:
    chunk: Chunk
    vector: list[float]


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        if cache_dir is not None:
            self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        else:
            self._model = TextEmbedding(model_name=model_name)

    def embed(self, chunks: list[Chunk]) -> list[EmbeddingResult]:
        texts = [chunk.text for chunk in chunks]
        vectors = self._model.embed(texts)
        return [
            EmbeddingResult(chunk=chunk, vector=vector.tolist())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_query(self, query: str) -> list[float]:
        vectors = self._model.query_embed(query)
        first_vector: np.ndarray = next(iter(vectors))
        result: list[float] = first_vector.tolist()
        return result


@lru_cache(maxsize=1)
def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str | None = None,
) -> Embedder:
    return Embedder(model_name=model_name, cache_dir=cache_dir)
