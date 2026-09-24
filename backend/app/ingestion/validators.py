"""File validation for resume uploads.

Validates extension, MIME type, size, and empty-content checks. Does not
validate internal file structure — that is the extractor's responsibility.
"""

from __future__ import annotations

from pathlib import PurePosixPath

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})
ALLOWED_MIMES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
)
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


class ValidationError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def validate_upload(
    *,
    filename: str | None,
    content_type: str | None,
    size: int,
    content: bytes,
) -> None:
    if not filename:
        raise ValidationError("Filename is required.", "missing_filename")

    ext = PurePosixPath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(
            f"Unsupported file type: {ext}. Allowed: {allowed}",
            "unsupported_file_type",
        )

    if content_type and content_type not in ALLOWED_MIMES:
        raise ValidationError(
            f"Unsupported content type: {content_type}.",
            "unsupported_content_type",
        )

    if size <= 0:
        raise ValidationError("Uploaded file is empty.", "empty_file")

    if size > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES} bytes.",
            "file_too_large",
        )

    if not content or not content.strip():
        raise ValidationError(
            "Uploaded file contains no usable content.",
            "empty_content",
        )
