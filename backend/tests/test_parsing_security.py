"""Security/privacy tests for the deterministic resume parser.

Proves the parser touches no filesystem resources, creates no temp files,
never uses user filenames as paths, never persists data, and never logs
resume content or PII.
"""

from __future__ import annotations

import logging
import tempfile

from app.parsing.parser import parse_resume


def _forbid_tempfile(monkeypatch) -> None:
    """Make any tempfile/filesystem helper fail if the parser touches it."""

    def boom(*args, **kwargs):
        raise AssertionError("Parser must not create any files")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", boom)
    monkeypatch.setattr(tempfile, "mkstemp", boom)
    monkeypatch.setattr(tempfile, "mkdtemp", boom)
    monkeypatch.setattr(tempfile, "mktemp", boom)


class TestNoFilesystemUse:
    def test_parser_creates_no_temp_files(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        resume = parse_resume(
            "Jane Doe\njane@example.com\n\nSKILLS\nPython, Java\n"
        )
        assert resume.contact.email == "jane@example.com"

    def test_parser_creates_no_temp_files_for_pdf_text(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        text = "John Smith\nSKILLS\nGo, Rust\n"
        resume = parse_resume(text)
        assert resume.contact.name == "John Smith"


class TestNoFilenameAsPath:
    def test_traversal_string_never_becomes_a_name(self) -> None:
        resume = parse_resume("../../etc/passwd\n\nSKILLS\nPython\n")
        assert resume.contact.name is None

    def test_path_like_content_is_not_used_as_path(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        resume = parse_resume("../outside/resume.pdf contents\n\nSKILLS\nPython\n")
        assert resume.contact.name is None


class TestNoLogging:
    def test_parser_never_logs_resume_content(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.parsing"):
            parse_resume(
                "SECRET-PII-1234\njane@example.com\n\nSKILLS\nPython\n"
            )
        for record in caplog.records:
            assert "SECRET-PII-1234" not in record.getMessage()
            assert "jane@example.com" not in record.getMessage()

    def test_parser_never_logs_contact(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.parsing"):
            parse_resume("Jane Doe\n+1 555 123 4567\n")
        for record in caplog.records:
            assert "555" not in record.getMessage()


class TestNoPersistence:
    def test_parser_source_never_imports_database(self) -> None:
        import pathlib

        from app.parsing import parser as parser_module

        package_dir = pathlib.Path(parser_module.__file__).parent
        assert package_dir.is_dir()
        for module_file in package_dir.glob("*.py"):
            source = module_file.read_text(encoding="utf-8")
            assert "app.db" not in source, f"{module_file} references the DB layer"
            assert "create_engine" not in source, f"{module_file} creates engines"

    def test_parse_result_exists_only_in_memory(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        resume = parse_resume("Jane Doe\nEDUCATION\nB.Sc Computer Science\n")
        assert resume.education
        assert resume.contact.name == "Jane Doe"
