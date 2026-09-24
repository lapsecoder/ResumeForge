"""Security/privacy tests for the Job-Specific ATS Coverage engine.

Proves the engine touches no filesystem resources, never imports or uses the
database or network layers, never logs resume/job content or PII, and provides
no persistence path. It also confirms no embeddings / semantic / LLM
dependency is involved and that the analysis is fully offline.
"""

from __future__ import annotations

import logging
import pathlib
import re as _re
import tempfile

from app.ats_analysis.job_specific.service import analyze_job_specific_ats
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.parsing.schemas import ConfidenceLevel
from tests.test_ats_analysis import make_resume, skill_set


def _forbid_tempfile(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("Job-specific ATS analysis must not create files")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", boom)
    monkeypatch.setattr(tempfile, "mkstemp", boom)
    monkeypatch.setattr(tempfile, "mkdtemp", boom)
    monkeypatch.setattr(tempfile, "mktemp", boom)


def _job() -> JobDescription:
    return JobDescription(
        required_skills=["Python", "Kafka"],
        responsibilities=["Build Python services."],
        metadata=JobMetadata(
            word_count=0, overall_confidence=ConfidenceLevel.HIGH
        ),
    )


def _resume_with_marker(marker: str):
    return make_resume(
        summary=f"Summary with {marker}.",
        experience=[],
        skills=skill_set(technical=["Python"]),
    )


class TestNoFilesystemUse:
    def test_analysis_creates_no_temp_files(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        result = analyze_job_specific_ats(
            _resume_with_marker("Jane"), _job()
        )
        assert result.overall_score is not None


class TestNoDatabaseUse:
    def test_source_never_imports_database(self) -> None:
        from app.ats_analysis.job_specific import service as service_module

        package_dir = pathlib.Path(service_module.__file__).parent
        assert package_dir.is_dir()
        for module_file in package_dir.glob("*.py"):
            source = module_file.read_text(encoding="utf-8")
            assert "app.db" not in source, f"{module_file} references DB layer"
            assert "create_engine" not in source
            assert "sqlalchemy" not in source
            assert "tempfile" not in source
            assert "open(" not in source, f"{module_file} opens files"


class TestNoExternalAccess:
    def test_source_has_no_network_semantic_or_llm(self) -> None:
        from app.ats_analysis.job_specific import service as service_module

        package_dir = pathlib.Path(service_module.__file__).parent
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
            "requests",
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
        marker = "SECRET-RESUME-MARKER-6B"
        with caplog.at_level(logging.DEBUG, logger="app.ats_analysis.job_specific"):
            analyze_job_specific_ats(_resume_with_marker(marker), _job())
        for record in caplog.records:
            assert marker not in record.getMessage()

    def test_analysis_never_logs_job_content(self, caplog) -> None:
        marker = "SECRET-JOB-MARKER-6B"
        job = JobDescription(
            required_skills=[marker],
            metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
        )
        with caplog.at_level(logging.DEBUG, logger="app.ats_analysis.job_specific"):
            analyze_job_specific_ats(_resume_with_marker("Jane"), job)
        for record in caplog.records:
            assert marker not in record.getMessage()

    def test_api_endpoint_never_logs_content(self, caplog) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        marker = "SECRET-API-MARKER-6B"
        resume_payload = _resume_with_marker(marker).model_dump(mode="json")
        job_payload = _job().model_dump(mode="json")
        with caplog.at_level(logging.DEBUG, logger="app.api.v1.resumes"):
            resp = TestClient(app).post(
                "/api/v1/resumes/job-specific-ats",
                json={"resume": resume_payload, "job_description": job_payload},
            )
        assert resp.status_code == 200
        for record in caplog.records:
            assert marker not in record.getMessage()


class TestNoPersistence:
    def test_result_exists_only_in_memory(self, monkeypatch) -> None:
        _forbid_tempfile(monkeypatch)
        result = analyze_job_specific_ats(_resume_with_marker("Jane"), _job())
        assert result.overall_score is not None
        assert result.metadata.method == "job-specific-ats-coverage-heuristic"

    def test_no_shared_state_between_calls(self) -> None:
        a = analyze_job_specific_ats(_resume_with_marker("Alice"), _job())
        b = analyze_job_specific_ats(_resume_with_marker("Bob"), _job())
        assert a is not b
        assert a == b
