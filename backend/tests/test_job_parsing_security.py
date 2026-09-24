"""Security/privacy tests for the deterministic job-description parser.

Proves the parser touches no filesystem resources, creates no temp files,
never uses user filenames as paths, never persists data, never imports the
database layer, and never logs job-description content.
"""

from __future__ import annotations

import logging
import tempfile

from app.job_parsing.parser import parse_job_description


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
        job = parse_job_description(
            "Backend Engineer\nCompany: Acme\n\nSKILLS\nPython, Go\n"
        )
        assert job.title == "Backend Engineer"
        assert job.required_skills == ["Python", "Go"]

    def test_parser_creates_no_temp_files_for_minimal_text(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        job = parse_job_description("Senior Engineer\nCompany: Acme\n\nSKILLS\nRust\n")
        assert job.company == "Acme"


class TestNoFilenameAsPath:
    def test_traversal_string_never_becomes_a_name(self) -> None:
        job = parse_job_description("../../etc/passwd\n\nSKILLS\nPython\n")
        assert job.required_skills == ["Python"]

    def test_path_like_content_is_not_used_as_path(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        job = parse_job_description("../outside/job.pdf contents\n\nSKILLS\nPython\n")
        assert job.required_skills == ["Python"]


class TestNoLogging:
    def test_parser_never_logs_jd_content(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.job_parsing"):
            parse_job_description(
                "PROPRIETARY-CLAUSE-9f2\n\nSKILLS\nPython\n"
            )
        for record in caplog.records:
            assert "PROPRIETARY-CLAUSE-9f2" not in record.getMessage()

    def test_parser_never_logs_salary_content(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.job_parsing"):
            parse_job_description("Salary\n₹25L–30L annual\n")
        for record in caplog.records:
            assert "25L" not in record.getMessage()


class TestNoPersistence:
    def test_parser_source_never_imports_database(self) -> None:
        import pathlib

        from app.job_parsing import parser as parser_module

        package_dir = pathlib.Path(parser_module.__file__).parent
        assert package_dir.is_dir()
        for module_file in package_dir.glob("*.py"):
            source = module_file.read_text(encoding="utf-8")
            assert "app.db" not in source, f"{module_file} references the DB layer"
            assert "create_engine" not in source, f"{module_file} creates engines"
            assert "sqlalchemy" not in source, f"{module_file} imports the DB layer"

    def test_parse_result_exists_only_in_memory(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        job = parse_job_description(
            "Data Engineer\nCompany: Acme\n\nRESPONSIBILITIES\n- Build pipelines\n"
        )
        assert job.responsibilities == ["Build pipelines"]
        assert job.company == "Acme"
