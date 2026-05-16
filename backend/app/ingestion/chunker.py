from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ChunkingStrategy(StrEnum):
    NAIVE = "naive"
    CONTEXTUAL = "contextual"
    LATE = "late"


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)


def chunk_text(
    text: str,
    strategy: ChunkingStrategy,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    if strategy == ChunkingStrategy.NAIVE:
        return _naive_chunk(text, chunk_size, overlap)
    if strategy == ChunkingStrategy.CONTEXTUAL:
        raise NotImplementedError("contextual chunking not yet implemented")
    if strategy == ChunkingStrategy.LATE:
        raise NotImplementedError("late chunking not yet implemented")
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _naive_chunk(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [Chunk(text=text, index=0)]

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    index = 0

    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(text=" ".join(window), index=index))
        start += step
        index += 1

    return chunks
