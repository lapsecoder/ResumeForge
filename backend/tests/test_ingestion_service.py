"""Tests for the ingestion service orchestrator."""

from pathlib import Path

import pytest

from app.ingestion.schemas import ExtractionResult, ExtractionStatus, FileTypeEnum
from app.ingestion.service import IngestionError, extract_resume_text
from app.ingestion.validators import MAX_FILE_SIZE_BYTES, ValidationError
from tests.conftest import (
    install_tempfile_tracker,
    make_docx_bytes,
    make_pdf_bytes,
    make_text_bytes,
)


class TestServiceExtraction:
    def test_extract_pdf(self) -> None:
        content = make_pdf_bytes("John Doe", "Software Engineer")
        result = extract_resume_text(
            filename="resume.pdf",
            content_type="application/pdf",
            size=len(content),
            content=content,
        )
        assert isinstance(result, ExtractionResult)
        assert result.file_type == FileTypeEnum.PDF
        assert result.page_count == 2
        assert result.status == ExtractionStatus.SUCCESS
        assert "John Doe" in result.extracted_text
        assert result.word_count > 0
        assert result.character_count == len(result.extracted_text)

    def test_extract_docx(self) -> None:
        content = make_docx_bytes("John Doe", "Software Engineer")
        result = extract_resume_text(
            filename="resume.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size=len(content),
            content=content,
        )
        assert result.file_type == FileTypeEnum.DOCX
        assert result.page_count is None
        assert result.status == ExtractionStatus.SUCCESS
        assert "John Doe" in result.extracted_text

    def test_extract_txt(self) -> None:
        content = make_text_bytes("John Doe\nSoftware Engineer")
        result = extract_resume_text(
            filename="resume.txt",
            content_type="text/plain",
            size=len(content),
            content=content,
        )
        assert result.file_type == FileTypeEnum.TXT
        assert result.page_count is None
        assert result.status == ExtractionStatus.SUCCESS
        assert "John Doe" in result.extracted_text

    def test_unsupported_extension(self) -> None:
        content = b"blah"
        with pytest.raises(ValidationError):
            extract_resume_text(
                filename="resume.exe",
                content_type="application/octet-stream",
                size=len(content),
                content=content,
            )


class TestServiceCleanup:
    def test_temp_file_removed_after_pdf_success(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)

        content = make_pdf_bytes("Text")
        extract_resume_text(
            filename="resume.pdf",
            content_type="application/pdf",
            size=len(content),
            content=content,
        )
        assert created
        for p in created:
            assert not p.exists(), f"Temp file {p} should have been removed"

    def test_temp_file_removed_after_docx_success(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)

        content = make_docx_bytes("hello")
        extract_resume_text(
            filename="resume.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size=len(content),
            content=content,
        )
        assert created
        for p in created:
            assert not p.exists(), f"Temp file {p} should have been removed"

    def test_temp_file_removed_during_parser_failure(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)

        content = b"This is not a PDF at all"
        with pytest.raises(IngestionError):
            extract_resume_text(
                filename="resume.pdf",
                content_type="application/pdf",
                size=len(content),
                content=content,
            )
        assert created
        for p in created:
            assert not p.exists(), (
                f"Temp file {p} should have been removed after failure"
            )

    def test_no_temp_file_for_txt(self, monkeypatch) -> None:
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)

        content = make_text_bytes("just text")
        extract_resume_text(
            filename="resume.txt",
            content_type="text/plain",
            size=len(content),
            content=content,
        )
        assert created == []


class TestServiceValidation:
    def test_oversized_file(self) -> None:
        content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(ValidationError):
            extract_resume_text(
                filename="resume.pdf",
                content_type="application/pdf",
                size=len(content),
                content=content,
            )

    def test_empty_file(self) -> None:
        with pytest.raises(ValidationError):
            extract_resume_text(
                filename="resume.pdf",
                content_type="application/pdf",
                size=0,
                content=b"",
            )
