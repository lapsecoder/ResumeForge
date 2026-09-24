"""Security and privacy guarantees of the semantic layer.

Everything here must hold while running offline with a fake provider: no user
content is logged, persisted, or sent anywhere, and the model stays lazy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.job_parsing.schemas import JobDescription, JobMetadata
from app.main import app
from app.parsing.schemas import ConfidenceLevel, Resume, ResumeMetadata, SkillSet
from app.semantic_matching.model import ModelMetadata, reset_embedding_provider
from app.semantic_matching.service import compute_semantic_match
from app.semantic_matching.text_builder import build_job_units, build_resume_units

client = TestClient(app)

RESUME_SECRET = "TOP-SECRET-PROJECT-NEPTUNE"
JOB_SECRET = "CONFIDENTIAL-ACQUISITION-2026"


class InMemoryProvider:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.prompts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.prompts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_name="fake", model_dimension=2, device="cpu")


def _resume() -> Resume:
    return Resume(
        summary=f"Worked on {RESUME_SECRET} integration.",
        skills=SkillSet(all=["Python"]),
        metadata=ResumeMetadata(
            word_count=0, file_type="pdf", overall_confidence=ConfidenceLevel.HIGH
        ),
    )


def _job() -> JobDescription:
    return JobDescription(
        summary=f"Role touching {JOB_SECRET} data.",
        required_skills=["Python"],
        metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
    )


class TestNoPersistence:
    def test_service_leaves_no_files(self, tmp_path: Path) -> None:
        before = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        compute_semantic_match(_resume(), _job(), provider=InMemoryProvider())
        after = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        assert after == before


class TestNoExternalContact:
    def test_package_has_no_network_or_storage_imports(self) -> None:
        package_dir = Path(__file__).resolve().parents[1] / "app" / "semantic_matching"
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in package_dir.glob("*.py")
        )
        for forbidden in (
            "import sqlalchemy",
            "from sqlalchemy",
            "app.db",
            "import redis",
            "import httpx",
            "import requests",
            "import aiohttp",
            "import urllib",
            "import tempfile",
            "import pickle",
            "import open",
            "open(",
            "shutil",
            "os.environ",
            "psycopg",
            "s3",
            "minio",
        ):
            assert forbidden not in source, f"found forbidden token: {forbidden}"

    def test_provider_never_imports_customer_data_paths(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "semantic_matching"
            / "model.py"
        ).read_text(encoding="utf-8")
        assert "huggingface_hub" not in source  # no eager network download here


class TestNoLoggingOfContent:
    def test_service_does_not_log_input_text(self, caplog) -> None:
        import logging  # noqa: PLC0415

        from app.semantic_matching import embedder, model, service  # noqa: PLC0415

        for target in (model, service, embedder):
            caplog.set_level(logging.DEBUG, logger=target.__name__)
        compute_semantic_match(_resume(), _job(), provider=InMemoryProvider())
        full = caplog.text
        assert RESUME_SECRET not in full
        assert JOB_SECRET not in full


class TestLazyLoadingIsNotTriggeredByImports:
    def test_units_require_no_model(self) -> None:
        reset_embedding_provider()
        units_resume = build_resume_units(_resume())
        units_job = build_job_units(_job())
        assert units_resume and units_job
        assert all(unit.text for unit in units_resume + units_job)


class TestApiSurface:
    def test_api_calls_are_transient_and_never_log_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        from app.semantic_matching import service as semantic_service  # noqa: PLC0415

        provider = InMemoryProvider()
        monkeypatch.setattr(
            semantic_service, "get_embedding_provider", lambda: provider
        )
        body = {
            "resume": {
                "summary": f"Worked on {RESUME_SECRET}.",
                "metadata": {
                    "word_count": 0,
                    "file_type": "pdf",
                    "overall_confidence": "high",
                },
            },
            "job": {
                "summary": f"Needs {JOB_SECRET} skills.",
                "required_skills": ["None"],
                "metadata": {
                    "word_count": 0,
                    "file_type": "txt",
                    "overall_confidence": "high",
                },
            },
        }
        before = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        resp = client.post("/api/v1/matching/semantic", json=body)
        assert resp.status_code == 200, resp.text
        after = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        assert after == before
        assert RESUME_SECRET not in caplog.text
        assert JOB_SECRET not in caplog.text
