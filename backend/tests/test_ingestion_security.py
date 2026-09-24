"""Security-focused tests: cleanup guarantees, path traversal, and logging."""

import logging
from pathlib import Path

from app.ingestion.service import extract_resume_text
from tests.conftest import install_tempfile_tracker, make_pdf_bytes, make_text_bytes


class TestPathTraversal:
    def test_traversal_filename_uses_only_extension(self, monkeypatch) -> None:
        """A malicious filename must not influence filesystem paths used."""
        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)

        content = make_pdf_bytes("Text")
        extract_resume_text(
            filename="../../etc/passwd.pdf",
            content_type="application/pdf",
            size=len(content),
            content=content,
        )
        # Temp file must be created from a random path, not the user filename.
        assert created
        for p in created:
            assert p.name.startswith("resumeforge_"), f"Unexpected temp name: {p}"
            assert not p.exists(), f"Temp file {p} should have been removed"


class TestNoContentLogging:
    def test_extracted_content_never_logged(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.ingestion"):
            content = make_text_bytes("SECRET-PII-1234 very private")
            extract_resume_text(
                filename="resume.txt",
                content_type="text/plain",
                size=len(content),
                content=content,
            )
        # Ensure no log record contains resume contents.
        for record in caplog.records:
            message = record.getMessage()
            assert "SECRET-PII-1234" not in message
            assert "very private" not in message

    def test_filename_not_logged(self, caplog, monkeypatch) -> None:
        with caplog.at_level(logging.DEBUG):
            content = make_pdf_bytes("Good text")
            extract_resume_text(
                filename="SensitiveCandidateName_resume.pdf",
                content_type="application/pdf",
                size=len(content),
                content=content,
            )
        for record in caplog.records:
            assert "SensitiveCandidateName" not in record.getMessage()


class TestCleanupOnUnexpectedException:
    def test_unexpected_exception_cleans_up(self, monkeypatch) -> None:
        """If extraction raises an unexpected exception, temp files are removed."""

        import app.ingestion.service as service

        created: list[Path] = []
        install_tempfile_tracker(monkeypatch, created)

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(service, "extract_from_pdf", boom)

        content = make_pdf_bytes("text")
        try:
            extract_resume_text(
                filename="resume.pdf",
                content_type="application/pdf",
                size=len(content),
                content=content,
            )
        except Exception:
            pass

        assert created
        for p in created:
            assert not p.exists(), f"Temp file {p} should have been removed"
