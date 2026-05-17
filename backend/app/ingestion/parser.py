from __future__ import annotations

import re

import fitz

from app.core.exceptions import UnsupportedFileTypeError
from app.models.document import FileType


def parse_file(content: bytes, file_type: FileType | str) -> str:
    """Extract plain text from raw file bytes.

    Args:
        content: Raw file bytes.
        file_type: The type of the file being parsed.

    Returns:
        Extracted text with excessive whitespace collapsed.

    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
    """
    if file_type == FileType.MARKDOWN:
        text = content.decode("utf-8")
    elif file_type == FileType.TXT:
        text = content.decode("utf-8")
    elif file_type == FileType.PDF:
        text = _extract_pdf_text(content)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_type}")

    return _normalize_whitespace(text)


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(stream=content, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _normalize_whitespace(text: str) -> str:
    """Collapse three or more consecutive newlines into two, and strip edges."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
