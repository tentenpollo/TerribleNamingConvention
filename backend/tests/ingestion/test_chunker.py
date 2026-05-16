from __future__ import annotations

import pytest

from app.ingestion.chunker import Chunk, ChunkingStrategy, chunk_text


@pytest.mark.unit
def test_naive_multiple_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, ChunkingStrategy.NAIVE, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)


@pytest.mark.unit
def test_naive_overlap_window_positions() -> None:
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, ChunkingStrategy.NAIVE, chunk_size=10, overlap=5)
    assert len(chunks) == 4
    assert chunks[0].text == " ".join(words[0:10])
    assert chunks[1].text == " ".join(words[5:15])
    assert chunks[2].text == " ".join(words[10:20])
    assert chunks[3].text == " ".join(words[15:20])


@pytest.mark.unit
def test_naive_short_text_single_chunk() -> None:
    text = "only five words here"
    chunks = chunk_text(text, ChunkingStrategy.NAIVE, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == text


@pytest.mark.unit
def test_naive_empty_string() -> None:
    assert chunk_text("", ChunkingStrategy.NAIVE) == []


@pytest.mark.unit
def test_naive_whitespace_only() -> None:
    assert chunk_text("   \n  ", ChunkingStrategy.NAIVE) == []


@pytest.mark.unit
def test_naive_correct_indices() -> None:
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(text, ChunkingStrategy.NAIVE, chunk_size=10, overlap=0)
    assert [c.index for c in chunks] == [0, 1, 2]


@pytest.mark.unit
def test_contextual_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        chunk_text("text", ChunkingStrategy.CONTEXTUAL)
    assert "contextual chunking not yet implemented" in str(exc_info.value)


@pytest.mark.unit
def test_late_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        chunk_text("text", ChunkingStrategy.LATE)
    assert "late chunking not yet implemented" in str(exc_info.value)
