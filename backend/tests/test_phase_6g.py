"""Focused tests for Phase 6G ML integration into the production matching flow.

Offline only (no model download). Verifies that the integrated MiniLM
semantic layer:

1. reports model versioning/config metadata (§6G-14) without breaking the
   backward-compatible contract;
2. discloses the model in the hybrid response (§6G-13);
3. never converts an absent skill into possession (guardrail);
4. keeps the 70/30 deterministic/semantic blend intact;
5. stays transient and never exposes embedding vectors.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.hybrid_matching.schemas import HYBRID_LABEL, HYBRID_MODEL_DISCLOSURE
from app.main import app
from app.semantic_matching.config import SEMANTIC_IMPLEMENTATION_VERSION
from app.semantic_matching import service as semantic_service
from app.semantic_matching.model import (
    ModelLoadError,
    ModelMetadata,
)
from app.semantic_matching.schemas import (
    RESULT_NOTE,
    SemanticMatchMetadata,
)

client = TestClient(app)

RESUME_BODY = {
    "contact": {"name": "Ada Lovelace", "email": "ada@example.com"},
    "summary": "First programmer who built algorithms.",
    "experience": [{"company": "Analytical Engines", "title": "Engineer"}],
    "education": [{"institution": "Cambridge", "degree": "B.A."}],
    "skills": {"all": ["Python", "PostgreSQL"], "languages": []},
    "projects": [],
    "certifications": [],
    "custom_sections": [],
    "metadata": {"word_count": 0, "file_type": "pdf", "overall_confidence": "high"},
}

JOB_BODY = {
    "title": "Backend Engineer",
    "summary": "Design and scale APIs.",
    "required_skills": ["Python", "Kubernetes"],
    "preferred_skills": [],
    "responsibilities": ["Ship reliable software."],
    "experience_requirements": ["3+ years"],
    "education_requirements": ["Bachelor's"],
    "certifications": [],
    "qualifications": [],
    "metadata": {"word_count": 0, "file_type": "txt", "overall_confidence": "high"},
}


class ApiFakeProvider:
    """Deterministic fake: returns [[1.0, 0.0]] for any text."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_dimension=2,
            device="cpu",
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> ApiFakeProvider:
    provider = ApiFakeProvider()
    monkeypatch.setattr(semantic_service, "get_embedding_provider", lambda: provider)
    return provider


def _post_hybrid(body: dict) -> Any:
    return client.post("/api/v1/matching/hybrid", json=body)


class TestModelVersioningMetadata:
    def test_semantic_metadata_reports_version_identity(self, fake_provider: Any) -> None:
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200, resp.text
        metadata = resp.json()["metadata"]
        assert metadata["method"] == "local-semantic-embedding"
        assert metadata["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert metadata["model_version"] == metadata["model_name"]
        assert metadata["implementation_version"] == SEMANTIC_IMPLEMENTATION_VERSION
        assert metadata["model_source"] == "local Hugging Face cache"
        assert metadata["model_license"] == "Apache-2.0"
        assert metadata["model_dimension"] == 2

    def test_hybrid_carries_semantic_versioning(self, fake_provider: Any) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        semantic_metadata = body["metadata"]["semantic_metadata"]
        assert semantic_metadata is not None
        assert semantic_metadata["implementation_version"] == (
            SEMANTIC_IMPLEMENTATION_VERSION
        )
        assert semantic_metadata["model_version"]

    def test_empty_result_metadata_still_versioned(self) -> None:
        # No comparable content on either side should still expose model facts.
        from app.semantic_matching.service import compute_semantic_match  # noqa: PLC0415
        from app.job_parsing.schemas import JobDescription, JobMetadata  # noqa: PLC0415
        from app.parsing.schemas import Resume, ResumeMetadata  # noqa: PLC0415

        empty_resume = Resume(
            metadata=ResumeMetadata(
                word_count=0, file_type="txt", overall_confidence="low"
            )
        )
        empty_job = JobDescription(
            metadata=JobMetadata(word_count=0, overall_confidence="low")
        )
        result = compute_semantic_match(
            empty_resume, empty_job, provider=ApiFakeProvider()
        )
        assert result.overall_similarity is None
        assert result.metadata.model_version == "sentence-transformers/all-MiniLM-L6-v2"
        assert result.metadata.implementation_version == SEMANTIC_IMPLEMENTATION_VERSION
        assert result.metadata.model_source == "local Hugging Face cache"
        assert result.metadata.model_license == "Apache-2.0"


class TestModelDisclosure:
    def test_hybrid_metadata_discloses_local_model(self, fake_provider: Any) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        disclosure = body["metadata"]["model_disclosure"]
        assert disclosure == HYBRID_MODEL_DISCLOSURE
        assert "local" in disclosure
        assert "open-source" in disclosure
        assert "No external" in disclosure

    def test_disclosure_survives_model_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken(*args: Any, **kwargs: Any) -> ModelMetadata | None:
            raise ModelLoadError("missing library")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        assert body["metadata"]["mode"] == "deterministic-only"
        assert body["metadata"]["semantic_metadata"] is None
        # Disclosure stays present and accurate even when the model is unused.
        assert body["metadata"]["model_disclosure"] == HYBRID_MODEL_DISCLOSURE


class TestGuardrailSemanticCanNotImplyPossession:
    def test_semantic_similarity_does_not_move_skill_into_matched(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        # Kubernetes is required by the job but absent from the resume.
        assert "Kubernetes" in body["missing_required"]
        assert "Kubernetes" not in body["matched_requirements"]
        # Python is explicitly present -> deterministic match.
        assert any(
            "Python" in requirement for requirement in body["matched_requirements"]
        )

    def test_semantic_result_note_says_relatedness_not_possession(
        self, fake_provider: Any
    ) -> None:
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        note = resp.json()["note"]
        assert note == RESULT_NOTE
        assert "not a hiring probability" in note
        assert "relatedness" in note

    def test_insight_wording_uses_relatedness_not_skill_possession(
        self, fake_provider: Any
    ) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        statements = " | ".join(i["statement"] for i in body["semantic_insights"])
        assert "related" in statements.lower()
        assert "not explicitly verified" in statements.lower()


class TestBlendAndContractStable:
    def test_70_30_weights_unchanged(self, fake_provider: Any) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        assert body["metadata"]["weights"] == {"deterministic": 0.70, "semantic": 0.30}
        assert body["metadata"]["label"] == HYBRID_LABEL

    def test_component_scores_present_in_stable_layout(self, fake_provider: Any) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        comps = body["component_scores"]
        for key in (
            "deterministic_overall",
            "semantic_overall",
            "hybrid_overall",
            "deterministic_skills",
            "semantic_skills",
        ):
            assert key in comps

    def test_semantic_field_types_and_notes_unchanged_contract(
        self, fake_provider: Any
    ) -> None:
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for field in (
            "overall_similarity",
            "summary_similarity",
            "skill_similarity",
            "matched_semantic_items",
            "related_items",
            "note",
            "metadata",
        ):
            assert field in body


class TestTransientAndNoVectorExposure:
    def test_no_embeddings_or_vectors_in_hybrid_payload(
        self, fake_provider: Any
    ) -> None:
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        stack: list[Any] = [body]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                assert "embedding" not in {str(k).lower() for k in node}
                assert "vector" not in {str(k).lower() for k in node}
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    def test_semantic_metadata_schema_fields_are_safe(self) -> None:
        fields = set(SemanticMatchMetadata.model_fields)
        assert not any("vector" in f or "embedding" in f for f in fields)