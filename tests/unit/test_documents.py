from __future__ import annotations

import pytest

from memx.documents import DocumentError, load_text, read_document


def make_pdf(pages: list[str]) -> bytes:
    """A minimal PDF with one line of Helvetica text per page (empty string = a page with no text)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {len(objs)} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_pages_are_read_in_order():
    doc = read_document(make_pdf(["Priya leads the data platform team.", "", "We never deploy on Fridays."]),
                        "handbook.pdf")
    assert doc.kind == "pdf" and doc.pages == 3 and doc.empty_pages == 1
    assert doc.text == "Priya leads the data platform team.\n\nWe never deploy on Fridays."


def test_pdf_is_detected_by_content_without_a_name():
    assert read_document(make_pdf(["Hello"])).kind == "pdf"


def test_pdf_without_text_layer_is_refused():
    with pytest.raises(DocumentError, match="no text layer"):
        read_document(make_pdf(["", ""]), "scan.pdf")


def test_broken_pdf_is_refused():
    with pytest.raises(DocumentError, match="could not be read"):
        read_document(b"%PDF-1.4\nnot really a pdf", "broken.pdf")


def test_text_files_decode_and_tidy():
    doc = read_document("﻿Line one   \r\n\r\n\r\n\r\nLine two — café\r\n".encode(), "notes.md")
    assert doc.kind == "text" and doc.text == "Line one\n\nLine two — café"
    assert read_document("naïve".encode("cp1252"), "old.txt").text == "naïve"
    assert read_document("hi".encode("utf-16"), "wide.txt").text == "hi"


def test_unsupported_and_binary_files_are_refused():
    with pytest.raises(DocumentError, match="only PDF and plain-text"):
        read_document(b"PK\x03\x04", "deck.docx")
    with pytest.raises(DocumentError, match="looks binary"):
        read_document(b"\x89PNG\r\n\x1a\n\x00\x00", None)


def test_load_text_uses_the_file_name_as_source(tmp_path):
    path = tmp_path / "minutes.pdf"
    path.write_bytes(make_pdf(["The launch moved to March."]))
    inp = load_text(path, trust=0.7)
    assert inp.kind == "text" and inp.source == "minutes.pdf" and inp.trust == 0.7
    assert inp.text == "The launch moved to March."
