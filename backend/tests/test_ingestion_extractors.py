"""Tests for format-specific text extractors."""

from pathlib import Path

import pytest

from app.ingestion.extractors.docx import extract_from_docx
from app.ingestion.extractors.pdf import extract_from_pdf
from app.ingestion.extractors.text import extract_from_text
from app.ingestion.schemas import ExtractionStatus
from tests.conftest import (
    make_docx_bytes,
    make_docx_with_table_bytes,
    make_empty_docx_bytes,
    make_empty_pdf,
    make_malformed_docx_bytes,
    make_pdf_bytes,
    make_text_bytes,
)


class TestPDFExtractor:
    def test_single_page(self, tmp_path: Path) -> None:
        pdf_bytes = make_pdf_bytes("John Doe")
        path = tmp_path / "test.pdf"
        path.write_bytes(pdf_bytes)
        result = extract_from_pdf(path)
        assert result.status == ExtractionStatus.SUCCESS
        assert result.page_count == 1
        assert "John Doe" in result.text

    def test_multi_page(self, tmp_path: Path) -> None:
        pdf_bytes = make_pdf_bytes("Page 1", "Page 2", "Page 3")
        path = tmp_path / "multi.pdf"
        path.write_bytes(pdf_bytes)
        result = extract_from_pdf(path)
        assert result.page_count == 3
        assert result.status == ExtractionStatus.SUCCESS

    def test_empty_pdf_ocr_required(self, tmp_path: Path) -> None:
        pdf_bytes = make_empty_pdf()
        path = tmp_path / "empty.pdf"
        path.write_bytes(pdf_bytes)
        result = extract_from_pdf(path)
        assert result.status == ExtractionStatus.OCR_REQUIRED
        assert result.text == ""
        assert any("scanned" in w.lower() for w in result.warnings)

    def test_malformed_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"This is not a PDF")
        with pytest.raises(Exception, match="Failed to read PDF"):
            extract_from_pdf(path)


class TestDOCXExtractor:
    def test_paragraphs(self, tmp_path: Path) -> None:
        docx_bytes = make_docx_bytes("Header", "Body text")
        path = tmp_path / "test.docx"
        path.write_bytes(docx_bytes)
        result = extract_from_docx(path)
        assert result.status == ExtractionStatus.SUCCESS
        assert "Header" in result.text
        assert "Body text" in result.text

    def test_with_table(self, tmp_path: Path) -> None:
        docx_bytes = make_docx_with_table_bytes()
        path = tmp_path / "table.docx"
        path.write_bytes(docx_bytes)
        result = extract_from_docx(path)
        assert result.status == ExtractionStatus.SUCCESS
        assert "Header text" in result.text
        assert "Col A" in result.text
        assert "Val 1" in result.text
        assert "Footer text" in result.text

    def test_empty_docx(self, tmp_path: Path) -> None:
        docx_bytes = make_empty_docx_bytes()
        path = tmp_path / "empty.docx"
        path.write_bytes(docx_bytes)
        result = extract_from_docx(path)
        assert result.status == ExtractionStatus.EMPTY_CONTENT
        assert result.text == ""

    def test_malformed_docx(self, tmp_path: Path) -> None:
        docx_bytes = make_malformed_docx_bytes()
        path = tmp_path / "bad.docx"
        path.write_bytes(docx_bytes)
        with pytest.raises(Exception, match="Failed to read DOCX"):
            extract_from_docx(path)


class TestTextExtractor:
    def test_utf8(self) -> None:
        content = make_text_bytes("Hello World")
        result = extract_from_text(content)
        assert result.status == ExtractionStatus.SUCCESS
        assert result.text == "Hello World"

    def test_preserves_line_endings(self) -> None:
        content = make_text_bytes("line1\r\nline2\rline3")
        result = extract_from_text(content)
        assert result.status == ExtractionStatus.SUCCESS
        assert "line1" in result.text
        assert "line2" in result.text
        assert "line3" in result.text

    def test_latin1_fallback(self) -> None:
        content = "café".encode("latin-1")
        result = extract_from_text(content)
        assert result.status == ExtractionStatus.SUCCESS
        assert "café" in result.text
        assert any("Latin-1" in w for w in result.warnings)

    def test_empty_content(self) -> None:
        content = make_text_bytes("")
        result = extract_from_text(content)
        assert result.status == ExtractionStatus.EMPTY_CONTENT
        assert result.text == ""

    def test_whitespace_only(self) -> None:
        content = make_text_bytes("   \n  ")
        result = extract_from_text(content)
        assert result.status == ExtractionStatus.EMPTY_CONTENT
