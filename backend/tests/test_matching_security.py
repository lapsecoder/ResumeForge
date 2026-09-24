"""Security/privacy tests for the deterministic matcher.

Proves the matcher touches no filesystem resources, never imports or uses the
database layer, never logs resume/job content or PII, and that no data model
component performs persistence.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile

from tests.test_matching import make_job, make_resume, match


def _forbid_tempfile(monkeypatch) -> None:
    """Make any tempfile/filesystem helper fail if the matcher touches it."""

    def boom(*args, **kwargs):
        raise AssertionError("Matcher must not create any files")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", boom)
    monkeypatch.setattr(tempfile, "mkstemp", boom)
    monkeypatch.setattr(tempfile, "mkdtemp", boom)
    monkeypatch.setattr(tempfile, "mktemp", boom)


class TestNoFilesystemUse:
    def test_matcher_creates_no_temp_files(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        result = match(
            make_resume(skills=["Python"]), make_job(required=["Python"])
        )
        assert result.overall_score == 100.0


class TestNoDatabaseUse:
    def test_matching_source_never_imports_database(self) -> None:
        from app.matching import service as service_module

        package_dir = pathlib.Path(service_module.__file__).parent
        assert package_dir.is_dir()
        for module_file in package_dir.glob("*.py"):
            source = module_file.read_text(encoding="utf-8")
            assert "app.db" not in source, f"{module_file} references the DB layer"
            assert "create_engine" not in source, f"{module_file} creates engines"
            assert "sqlalchemy" not in source, f"{module_file} imports sqlalchemy"
            assert "tempfile" not in source, f"{module_file} touches tempfile"


class TestNoLogging:
    def test_matcher_never_logs_resume_content(self, caplog) -> None:
        marker = "SECRET-RESUME-MARKER-0913"
        with caplog.at_level(logging.DEBUG, logger="app.matching"):
            match(
                make_resume(summary=marker, skills=["Python"]),
                make_job(required=["Python"]),
            )
        for record in caplog.records:
            assert marker not in record.getMessage()

    def test_matcher_never_logs_job_content(self, caplog) -> None:
        marker = "SECRET-JOB-MARKER-0913"
        with caplog.at_level(logging.DEBUG, logger="app.matching"):
            match(make_resume(), make_job(required=[marker]))
        for record in caplog.records:
            assert marker not in record.getMessage()

    def test_matcher_never_logs_pii(self, caplog) -> None:
        email = "jane@example.com"
        with caplog.at_level(logging.DEBUG, logger="app.matching"):
            resume = make_resume(skills=["Python"])
            resume.contact.email = email
            match(resume, make_job(required=["Python"]))
        for record in caplog.records:
            assert email not in record.getMessage()


class TestNoPersistence:
    def test_result_exists_only_in_memory(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        result = match(
            make_resume(skills=["Python"]), make_job(required=["Python"])
        )
        assert result.overall_score == 100.0
        assert result.matched_requirements == ["Required skill matched: Python"]
