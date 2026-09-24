"""Transient Pydantic schemas for resume ingestion results.

These models exist only during request processing. They are not database
tables, are not persisted to disk, and are garbage-collected after the
response is sent.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    OCR_REQUIRED = "ocr_required"
    EMPTY_CONTENT = "empty_content"
    UNSUPPORTED_FORMAT = "unsupported_format"
    MALFORMED_FILE = "malformed_file"
    EXTRACTION_FAILED = "extraction_failed"


class FileTypeEnum(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ExtractionResult(BaseModel):
    extracted_text: str = Field(..., description="Normalised extracted text.")
    file_type: FileTypeEnum
    page_count: int | None = Field(
        None, description="Number of pages (PDF only)."
    )
    character_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    warnings: list[str] = Field(default_factory=list)
