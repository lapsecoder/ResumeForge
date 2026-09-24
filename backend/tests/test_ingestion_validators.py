"""Tests for file validation."""

import pytest

from app.ingestion.validators import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    ValidationError,
    validate_upload,
)


def test_valid_pdf() -> None:
    validate_upload(
        filename="resume.pdf",
        content_type="application/pdf",
        size=1000,
        content=b"%PDF-1.4 fake",
    )


def test_valid_docx() -> None:
    validate_upload(
        filename="resume.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size=1000,
        content=b"fake docx",
    )


def test_valid_txt() -> None:
    validate_upload(
        filename="resume.txt",
        content_type="text/plain",
        size=1000,
        content=b"plain text",
    )


def test_valid_with_no_content_type() -> None:
    validate_upload(
        filename="resume.pdf",
        content_type=None,
        size=1000,
        content=b"fake pdf",
    )


def test_unsupported_extension() -> None:
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload(
            filename="resume.exe",
            content_type="application/octet-stream",
            size=1000,
            content=b"data",
        )


def test_unsupported_extension_image() -> None:
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload(
            filename="photo.jpg",
            content_type="image/jpeg",
            size=1000,
            content=b"data",
        )


def test_unsupported_content_type() -> None:
    with pytest.raises(ValidationError, match="Unsupported content type"):
        validate_upload(
            filename="resume.pdf",
            content_type="application/javascript",
            size=1000,
            content=b"data",
        )


def test_empty_file() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_upload(
            filename="resume.pdf",
            content_type="application/pdf",
            size=0,
            content=b"",
        )


def test_file_too_large() -> None:
    with pytest.raises(ValidationError, match="maximum size"):
        validate_upload(
            filename="resume.pdf",
            content_type="application/pdf",
            size=MAX_FILE_SIZE_BYTES + 1,
            content=b"x" * (MAX_FILE_SIZE_BYTES + 1),
        )


def test_no_filename() -> None:
    with pytest.raises(ValidationError, match="Filename is required"):
        validate_upload(
            filename=None,
            content_type="application/pdf",
            size=100,
            content=b"data",
        )


def test_empty_content_bytes() -> None:
    with pytest.raises(ValidationError, match="no usable content"):
        validate_upload(
            filename="resume.pdf",
            content_type="application/pdf",
            size=5,
            content=b"     ",
        )


def test_extension_case_insensitive() -> None:
    validate_upload(
        filename="Resume.PDF",
        content_type="application/pdf",
        size=100,
        content=b"data",
    )


def test_extension_in_allowed_list() -> None:
    assert ALLOWED_EXTENSIONS == frozenset({".pdf", ".docx", ".txt"})


def test_max_file_size_is_10mb() -> None:
    assert MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024
