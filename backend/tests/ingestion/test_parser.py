from __future__ import annotations

import fitz
import pytest

from app.core.exceptions import UnsupportedFileTypeError
from app.ingestion.parser import parse_file
from app.models.document import FileType


@pytest.mark.unit
def test_parse_markdown() -> None:
    content = b"# Hello\n\nThis is **markdown** content.\n"
    result = parse_file(content, FileType.MARKDOWN)
    assert result == "# Hello\n\nThis is **markdown** content."


@pytest.mark.unit
def test_parse_txt() -> None:
    content = b"Plain text file.\nNothing special here.\n"
    result = parse_file(content, FileType.TXT)
    assert result == "Plain text file.\nNothing special here."


@pytest.mark.unit
def test_parse_pdf() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from PDF")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = parse_file(pdf_bytes, FileType.PDF)
    assert "Hello from PDF" in result


@pytest.mark.unit
def test_parse_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        parse_file(b"some content", "csv")  # type: ignore[arg-type]

    assert "Unsupported file type" in str(exc_info.value)


@pytest.mark.unit
def test_whitespace_collapsed() -> None:
    content = b"First paragraph.\n\n\n\n\nSecond paragraph.\n\n\n\nThird."
    result = parse_file(content, FileType.TXT)
    assert result == "First paragraph.\n\nSecond paragraph.\n\nThird."
