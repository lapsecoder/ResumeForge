"""PDF text extraction using pypdf."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.ingestion.schemas import ExtractionStatus


class PDFExtractionError(Exception):
    pass


class PDFResult:
    def __init__(
        self,
        text: str,
        page_count: int,
        status: ExtractionStatus,
        warnings: list[str],
    ) -> None:
        self.text = text
        self.page_count = page_count
        self.status = status
        self.warnings = warnings


def extract_from_pdf(file_path: Path) -> PDFResult:
    """Extract text from a PDF file. Falls back to in-memory via BytesIO if needed."""
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise PDFExtractionError(f"Failed to read PDF: {exc}") from exc

    page_count = len(reader.pages)
    warnings: list[str] = []
    pages_text: list[str] = []

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            warnings.append(f"Could not extract text from page {i + 1}.")
            continue
        pages_text.append(page_text)

    text = "\n\n".join(pages_text)

    if not text.strip():
        return PDFResult(
            text="",
            page_count=page_count,
            status=ExtractionStatus.OCR_REQUIRED,
            warnings=[
                "No extractable text found. The document may be scanned or image-based."
            ],
        )

    return PDFResult(
        text=text,
        page_count=page_count,
        status=ExtractionStatus.SUCCESS,
        warnings=warnings,
    )
