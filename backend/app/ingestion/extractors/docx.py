"""DOCX text extraction using python-docx."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from docx.api import Document as _DocumentFactory
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.schemas import ExtractionStatus


class DOCXExtractionError(Exception):
    pass


class DOCXResult:
    def __init__(
        self,
        text: str,
        status: ExtractionStatus,
        warnings: list[str],
    ) -> None:
        self.text = text
        self.status = status
        self.warnings = warnings


def _table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append("\t".join(cells))
    return "\n".join(rows)


def _iter_block_items(doc: _Document) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables from the document body in order."""
    parent = doc.element.body
    for child in parent.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def extract_from_docx(file_path: Path) -> DOCXResult:
    """Extract text from a DOCX file, preserving paragraph and table order."""
    try:
        doc: _Document = _DocumentFactory(str(file_path))
    except Exception as exc:
        raise DOCXExtractionError(f"Failed to read DOCX: {exc}") from exc

    parts: list[str] = []
    warnings: list[str] = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(text)
        elif isinstance(block, Table):
            table_text = _table_to_text(block)
            if table_text.strip():
                parts.append(table_text)

    text = "\n\n".join(parts)

    if not text.strip():
        return DOCXResult(
            text="",
            status=ExtractionStatus.EMPTY_CONTENT,
            warnings=["Document contains no extractable text."],
        )

    return DOCXResult(
        text=text,
        status=ExtractionStatus.SUCCESS,
        warnings=warnings,
    )
