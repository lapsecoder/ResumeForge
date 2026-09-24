"""Plain text extraction."""

from __future__ import annotations

from app.ingestion.schemas import ExtractionStatus


class TextExtractionResult:
    def __init__(
        self,
        text: str,
        status: ExtractionStatus,
        warnings: list[str],
    ) -> None:
        self.text = text
        self.status = status
        self.warnings = warnings


def extract_from_text(content: bytes) -> TextExtractionResult:
    """Decode UTF-8 text bytes into a string."""
    warnings: list[str] = []

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
            warnings.append("File was not valid UTF-8; decoded as Latin-1.")
        except Exception:
            return TextExtractionResult(
                text="",
                status=ExtractionStatus.MALFORMED_FILE,
                warnings=["Could not decode file content as text."],
            )

    if not text.strip():
        return TextExtractionResult(
            text="",
            status=ExtractionStatus.EMPTY_CONTENT,
            warnings=["File contains no usable text content."],
        )

    return TextExtractionResult(
        text=text,
        status=ExtractionStatus.SUCCESS,
        warnings=warnings,
    )
