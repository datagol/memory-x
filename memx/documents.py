"""Turn an uploaded file into plain text for a ``Text`` ingest.

Plain-text files are decoded; PDFs go through pypdf, page by page. Nothing here calls a model: the result is the
same text a person would paste, so the normal extract step runs on it unchanged.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from memx.types import Text

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".log"}
SUFFIXES = TEXT_SUFFIXES | {".pdf"}


class DocumentError(ValueError):
    """The file can't be turned into text: wrong type, encrypted, or a PDF with no text layer."""


@dataclass(frozen=True)
class Document:
    text: str
    kind: str  # "pdf" | "text"
    filename: str | None = None
    pages: int | None = None
    empty_pages: int = 0


def read_document(data: bytes, filename: str | None = None) -> Document:
    suffix = Path(filename).suffix.lower() if filename else ""
    if data.startswith(b"%PDF-") or suffix == ".pdf":
        return _read_pdf(data, filename)
    if suffix and suffix not in TEXT_SUFFIXES:
        raise DocumentError(f"{filename}: only PDF and plain-text files ({', '.join(sorted(SUFFIXES))}) are supported")
    return Document(text=_tidy(_decode(data, filename)), kind="text", filename=filename)


def load_text(path: str | Path, *, source: str | None = None, trust: float | None = None) -> Text:
    """Read a file from disk into a ``Text`` input; the source label defaults to the file name."""
    path = Path(path)
    doc = read_document(path.read_bytes(), path.name)
    return Text(text=doc.text, source=source or path.name, trust=trust)


def _decode(data: bytes, filename: str | None) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:4096]:
        raise DocumentError(f"{filename or 'the file'} looks binary, not text")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _read_pdf(data: bytes, filename: str | None) -> Document:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise DocumentError("reading PDFs needs pypdf: pip install 'memx[pdf]'") from exc

    name = filename or "the PDF"
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError(f"{name} is password-protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise DocumentError(f"{name} could not be read as a PDF: {exc}") from exc
    texts = [_tidy(p) for p in pages]
    if not any(texts):
        raise DocumentError(f"{name} has no text layer (it may be a scanned image); run OCR first")
    return Document(text="\n\n".join(t for t in texts if t), kind="pdf", filename=filename,
                    pages=len(pages), empty_pages=sum(1 for t in texts if not t))


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
