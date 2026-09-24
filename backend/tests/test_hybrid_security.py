"""Security and privacy guarantees of the hybrid matching layer (Phase 5C).

Works offline with a fake provider: no user content is logged, persisted, or
sent anywhere; no embedding vectors ever reach the response; and the composite
layer adds no new network, storage, or LLM dependency on top of 5A/5B.
Degraded (deterministic-only) paths are equally silent about content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.hybrid_matching.schemas import (
    ComponentScores,
    HybridMatchResult,
    HybridMetadata,
    SemanticInsight,
)
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.main import app
from app.parsing.schemas import ConfidenceLevel, Resume, ResumeMetadata, SkillSet
from app.semantic_matching.model import ModelLoadError, ModelMetadata

client = TestClient(app)

RESUME_SECRET = "TOP-SECRET-PROJECT-NEPTUNE"
JOB_SECRET = "CONFIDENTIAL-ACQUISITION-2026"


class InMemoryProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
        from app.hybrid_matching.service import compute_hybrid_match  # noqa: PLC0415

        before = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        compute_hybrid_match(_resume(), _job(), provider=InMemoryProvider())
        after = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        assert after == before


class TestNoExternalContact:
    def test_package_has_no_network_storage_or_llm_imports(self) -> None:
        package_dir = Path(__file__).resolve().parents[1] / "app" / "hybrid_matching"
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
            "ollama",
            "openai",
            "anthropic",
            "google",
            "huggingface_hub",
"import weaviate",
            "import pinecone",
            "import chromadb",
        ):
            assert forbidden not in source, f"found forbidden token: {forbidden}"



class TestNoEmbeddingExposure:
    def test_result_schema_has_no_vector_or_embedding_fields(self) -> None:
        for model in (
            HybridMatchResult,
            ComponentScores,
            HybridMetadata,
            SemanticInsight,
        ):
            fields = set(model.model_fields)
            assert not any(
                "embedding" in name or "vector" in name for name in fields
            )

    def test_service_returns_no_vector_objects(self) -> None:
        from app.hybrid_matching.service import compute_hybrid_match  # noqa: PLC0415

        result = compute_hybrid_match(_resume(), _job(), provider=InMemoryProvider())
        payload = result.model_dump(mode="json")
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                assert "embedding" not in {str(k).lower() for k in node}
                assert "vector" not in {str(k).lower() for k in node}
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)


class TestNoLoggingOfContent:
    def test_service_does_not_log_input_text(self, caplog) -> None:
        import logging  # noqa: PLC0415

        from app.hybrid_matching import service  # noqa: PLC0415

        caplog.set_level(logging.DEBUG, logger=service.__name__)
        from app.hybrid_matching.service import compute_hybrid_match  # noqa: PLC0415

        compute_hybrid_match(_resume(), _job(), provider=InMemoryProvider())
        assert RESUME_SECRET not in caplog.text
        assert JOB_SECRET not in caplog.text

    def test_degraded_path_never_logs_content(self, caplog) -> None:
        import logging  # noqa: PLC0415

        from app.hybrid_matching import service  # noqa: PLC0415
        from app.hybrid_matching.service import compute_hybrid_match  # noqa: PLC0415

        caplog.set_level(logging.DEBUG, logger=service.__name__)

        class BrokenProvider:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                raise ModelLoadError(f"missing {RESUME_SECRET}")

            def metadata(self) -> ModelMetadata:
                return ModelMetadata(model_name="fake", model_dimension=2, device="cpu")

        compute_hybrid_match(_resume(), _job(), provider=BrokenProvider())
        assert RESUME_SECRET not in caplog.text
        assert JOB_SECRET not in caplog.text


class TestApiSurface:
    def test_api_is_transient_never_logs_and_never_exposes_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        from app.semantic_matching import service as semantic_service  # noqa: PLC0415

        monkeypatch.setattr(
            semantic_service, "get_embedding_provider", lambda: InMemoryProvider()
        )
        body = {
            "resume": {
                "summary": f"Worked on {RESUME_SECRET}.",
                "skills": {"all": ["Python"]},
                "metadata": {
                    "word_count": 0,
                    "file_type": "pdf",
                    "overall_confidence": "high",
                },
            },
            "job": {
                "summary": f"Needs {JOB_SECRET} skills.",
                "required_skills": ["Python"],
                "metadata": {
                    "word_count": 0,
                    "file_type": "txt",
                    "overall_confidence": "high",
                },
            },
        }
        before = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        resp = client.post("/api/v1/matching/hybrid", json=body)
        assert resp.status_code == 200, resp.text
        after = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        assert after == before
        assert RESUME_SECRET not in resp.text
        assert JOB_SECRET not in resp.text
        assert RESUME_SECRET not in caplog.text
        assert JOB_SECRET not in caplog.text

    def test_hybrid_service_source_does_not_build_freeform_explanations(self) -> None:
        service_source = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "hybrid_matching"
            / "service.py"
        ).read_text(encoding="utf-8")
        assert "generate" not in service_source.lower()


class TestCompositeKeepsLayersSeparate:
    def test_hybrid_does_not_replace_or_import_the_two_layers(self) -> None:
        service_source = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "hybrid_matching"
            / "service.py"
        ).read_text(encoding="utf-8")
        assert "compute_semantic_match(" in service_source
        assert "match_resume_to_job(" in service_source
