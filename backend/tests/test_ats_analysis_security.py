"""Security/privacy tests for the ATS readiness engine.

Proves the engine touches no filesystem resources, never imports or uses the
database or network layers, never logs resume content or PII, and provides no
persistence path. It also confirms no embeddings / semantic / LLM dependency
is involved.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile

from app.ats_analysis.service import analyze_resume
from tests.test_ats_analysis import contact_info, exp, make_resume, meta, skill_set


def _forbid_tempfile(monkeypatch) -> None:
    """Make any tempfile/filesystem helper fail if the engine touches it."""

    def boom(*args, **kwargs):
        raise AssertionError("ATS analysis must not create any files")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", boom)
    monkeypatch.setattr(tempfile, "mkstemp", boom)
    monkeypatch.setattr(tempfile, "mkdtemp", boom)
    monkeypatch.setattr(tempfile, "mktemp", boom)


def _resume_with_marker(marker: str):
    return make_resume(
        contact=contact_info(
            name=marker,
            email=f"{marker}@example.com",
            phone="123-456-7890",
            linkedin=None,
        ),
        summary=f"Summary with {marker}.",
        experience=[exp(achievements=[f"Built a thing for {marker}."])],
        skills=skill_set(technical=["Python"]),
        metadata=meta(300),
    )


class TestNoFilesystemUse:
    def test_analysis_creates_no_temp_files(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        result = analyze_resume(_resume_with_marker("Jane"))
        assert result.overall_score is not None


class TestNoDatabaseUse:
    def test_ats_analysis_source_never_imports_database(self) -> None:
        from app.ats_analysis import service as service_module

        package_dir = pathlib.Path(service_module.__file__).parent
        assert package_dir.is_dir()
        for module_file in package_dir.glob("*.py"):
            source = module_file.read_text(encoding="utf-8")
            assert "app.db" not in source, f"{module_file} references the DB layer"
            assert "create_engine" not in source, f"{module_file} creates engines"
            assert "sqlalchemy" not in source, f"{module_file} imports sqlalchemy"
            assert "tempfile" not in source, f"{module_file} touches tempfile"
            assert "open(" not in source, f"{module_file} opens files"


class TestNoExternalAccess:
    def test_ats_analysis_source_has_no_network_semantic_or_llm(self) -> None:
        import re as _re

        from app.ats_analysis import service as service_module

        package_dir = pathlib.Path(service_module.__file__).parent
        # Scan for imports/signals, not bare words (e.g. "requests?" is a
        # counted unit inside a regex).
        bad_tokens = (
            "urllib",
            "httpx",
            "import socket",
            "socket.",
            "sentence_transformers",
            "torch",
            "transformers",
            "ollama",
            "openai",
            "open(",
        )
        import_re = _re.compile(r"(?m)^\s*(?:import|from)\s+(\w+)")
        for module_file in package_dir.glob("*.py"):
            source = module_file.read_text(encoding="utf-8")
            for token in bad_tokens:
                assert token not in source, (
                    f"{module_file} leaks external-access token '{token}'"
                )
            for imported in import_re.findall(source):
                assert imported not in ("requests", "http", "socket"), (
                    f"{module_file} imports external library '{imported}'"
                )


class TestNoLogging:
    def test_analysis_never_logs_resume_content(self, caplog) -> None:
        marker = "SECRET-RESUME-MARKER-6A"
        with caplog.at_level(logging.DEBUG, logger="app.ats_analysis"):
            analyze_resume(_resume_with_marker(marker))
        for record in caplog.records:
            assert marker not in record.getMessage()

    def test_analysis_never_logs_pii(self, caplog) -> None:
        email = "jane-secret@example.com"
        resume = _resume_with_marker("Jane")
        resume.contact.email = email
        with caplog.at_level(logging.DEBUG, logger="app.ats_analysis"):
            analyze_resume(resume)
        for record in caplog.records:
            assert email not in record.getMessage()

    def test_api_endpoint_never_logs_resume_content(self, caplog) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        marker = "SECRET-API-MARKER-6A"
        resume = _resume_with_marker(marker)
        with caplog.at_level(logging.DEBUG, logger="app.api.v1.resumes"):
            resp = TestClient(app).post(
                "/api/v1/resumes/ats-analysis",
                json={"resume": resume.model_dump(mode="json")},
            )
        assert resp.status_code == 200
        for record in caplog.records:
            assert marker not in record.getMessage()


class TestNoPersistence:
    def test_result_exists_only_in_memory(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        result = analyze_resume(_resume_with_marker("Jane"))
        assert result.overall_score is not None
        assert result.metadata.method == "ats-readiness-heuristic"

    def test_no_shared_state_between_calls(self) -> None:
        a = analyze_resume(_resume_with_marker("Alice"))
        b = analyze_resume(_resume_with_marker("Bob"))
        assert a is not b
        assert a == b  # identical resumes -> identical deterministic results
