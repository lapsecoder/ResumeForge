"""Ingestion service: orchestrates validation, extraction, and normalisation.

Everything is transient. No user data is persisted.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from app.ingestion.extractors.docx import (
    DOCXExtractionError,
    extract_from_docx,
)
from app.ingestion.extractors.pdf import (
    PDFExtractionError,
    extract_from_pdf,
)
from app.ingestion.extractors.text import extract_from_text
from app.ingestion.normalizer import normalize_text
from app.ingestion.schemas import ExtractionResult, FileTypeEnum
from app.ingestion.validators import validate_upload


class IngestionError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def extract_resume_text(
    *,
    filename: str | None,
    content_type: str | None,
    size: int,
    content: bytes,
) -> ExtractionResult:
    """Validate, extract, and normalise resume text. Entirely transient."""
    validate_upload(
        filename=filename,
        content_type=content_type,
        size=size,
        content=content,
    )

    ext = Path(filename).suffix.lower()  # type: ignore[arg-type]

    if ext == ".txt":
        return _extract_text(content)
    if ext in (".pdf", ".docx"):
        return _extract_from_file(ext, content)

    raise IngestionError("Unsupported file type.", "unsupported_file_type")


def _extract_text(content: bytes) -> ExtractionResult:
    result = extract_from_text(content)
    normalised = normalize_text(result.text)

    return ExtractionResult(
        extracted_text=normalised,
        file_type=FileTypeEnum.TXT,
        page_count=None,
        character_count=len(normalised),
        word_count=_word_count(normalised),
        status=result.status,
        warnings=result.warnings,
    )


def _extract_from_file(ext: str, content: bytes) -> ExtractionResult:
    with TemporaryDirectory(prefix="resumeforge_") as tmp_dir:
        try:
            tmp_path = Path(tmp_dir) / f"document{ext}"
            tmp_path.write_bytes(content)

            if ext == ".pdf":
                pdf_result = extract_from_pdf(tmp_path)
                normalised = normalize_text(pdf_result.text)
                return ExtractionResult(
                    extracted_text=normalised,
                    file_type=FileTypeEnum.PDF,
                    page_count=pdf_result.page_count,
                    character_count=len(normalised),
                    word_count=_word_count(normalised),
                    status=pdf_result.status,
                    warnings=pdf_result.warnings,
                )

            docx_result = extract_from_docx(tmp_path)
            normalised = normalize_text(docx_result.text)
            return ExtractionResult(
                extracted_text=normalised,
                file_type=FileTypeEnum.DOCX,
                page_count=None,
                character_count=len(normalised),
                word_count=_word_count(normalised),
                status=docx_result.status,
                warnings=docx_result.warnings,
            )
        except (PDFExtractionError, DOCXExtractionError) as exc:
            raise IngestionError(str(exc), "malformed_file") from exc
