"""API contract tests for the Resume Copilot endpoints (Phase 7A).

Offline by construction: the router is pointed at a deterministic-only
service, so no local Ollama call is ever made.
"""

from __future__ import annotations

import pytest
from copilot_factories import make_resume
from fastapi.testclient import TestClient

from app.copilot.config import COPILOT_IMPLEMENTATION_VERSION, CopilotSettings
from app.copilot.providers import DeterministicFallbackProvider
from app.copilot.service import CopilotService
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _deterministic_only_service(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api.v1 import copilot as copilot_router

    monkeypatch.setattr(
        copilot_router,
        "_service",
        CopilotService(providers=[DeterministicFallbackProvider()]),
    )


def _payload(operation: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "operation": operation,
        "resume": make_resume().model_dump(mode="json"),
    }
    payload.update(overrides)
    return payload


def _error(response: object) -> dict[str, str]:
    return response.json()["error"]  # type: ignore[union-attr]


def test_status_reports_provider_availability() -> None:
    response = client.get("/api/v1/copilot/status")
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_available"] is True
    assert body["version"] == COPILOT_IMPLEMENTATION_VERSION
    kinds = {p["provider"] for p in body["providers"]}
    assert "deterministic" in kinds
    deterministic = next(
        p for p in body["providers"] if p["provider"] == "deterministic"
    )
    assert deterministic["available"] is True


def test_suggest_returns_structured_response() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_summary"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["operation"] == "improve_summary"
    assert body["provider"]["provider"] == "deterministic"
    assert body["provider"]["fallback_used"] is True
    assert body["disclaimer"]
    assert body["explanation"]
    assert body["suggestions"]
    suggestion = body["suggestions"][0]
    assert suggestion["id"]
    assert suggestion["category"] in {
        "summary",
        "bullet",
        "skills",
        "quantification",
        "evidence",
        "alignment",
        "structure",
        "priority",
        "explanation",
        "clarity",
        "action_wording",
    }
    assert suggestion["verification"] in {
        "verified",
        "inferred",
        "advisory",
        "unverified",
    }


def test_suggest_rejects_unknown_operation() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("unlimited_free_text"),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "validation_error"


def test_suggest_rejects_job_alignment_without_job() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("job_alignment"),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_rejects_explain_finding_without_finding_ref() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("explain_finding"),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_rejects_bullet_without_target() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_bullet"),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_rejects_oversized_target() -> None:
    oversized = "x" * (CopilotSettings().max_target_text_chars + 1)
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_bullet", target_text=oversized),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_free_form_requires_user_request() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("free-form"),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_free_form_rejects_blank_user_request() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("free-form", user_request="   "),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_free_form_rejects_oversized_user_request() -> None:
    oversized = "y" * (CopilotSettings().max_request_chars + 1)
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("free-form", user_request=oversized),
    )
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_request"


def test_suggest_free_form_routes_to_advisory_by_default() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("free-form", user_request="What should I know about resumes?"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["operation"] == "free-form"
    assert body["provider"]["provider"] == "deterministic"
    assert body["suggestions"]
    for suggestion in body["suggestions"]:
        assert suggestion["verification"] in {
            "verified",
            "inferred",
            "advisory",
            "unverified",
        }


def test_suggest_free_form_routes_summary_intent_to_summary_rewrite() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "free-form",
            user_request="Rewrite my professional summary to be more concise.",
        ),
    )
    assert response.status_code == 200
    body = response.json()
    categories = {s["category"] for s in body["suggestions"]}
    assert "summary" in categories or "clarity" in categories


def test_suggest_free_form_routes_bullet_intent_to_bullet_rewrite() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "free-form",
            user_request="Improve the wording of this bullet.",
            target_ref="experience[0].achievements[0]",
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert any(
        s["category"] in {"clarity", "bullet", "quantification", "action_wording"}
        for s in body["suggestions"]
    )


def test_suggest_free_form_routes_job_intent_to_alignment() -> None:
    from copilot_factories import make_job

    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "free-form",
            user_request="How do I better match this job description?",
            job_description=make_job().model_dump(mode="json"),
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert any(s["category"] == "alignment" for s in body["suggestions"])


def test_error_response_has_request_id() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("job_alignment"),
    )
    error = _error(response)
    assert error["message"]
    assert error["request_id"]


def test_job_alignment_succeeds_with_job() -> None:
    from copilot_factories import make_job

    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "job_alignment",
            job_description=make_job().model_dump(mode="json"),
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["operation"] == "job_alignment"
    for suggestion in body["suggestions"]:
        assert suggestion["category"] == "alignment"


def test_response_does_not_echo_pii_from_resume() -> None:
    from app.parsing.schemas import ContactInfo

    resume = make_resume(
        contact=ContactInfo(
            name="Ada Lovelace",
            email="ada@example.com",
            phone="+1-555-0100",
        )
    )
    response = client.post(
        "/api/v1/copilot/suggest",
        json={
            "operation": "improve_summary",
            "resume": resume.model_dump(mode="json"),
        },
    )
    assert response.status_code == 200
    payload = response.content.decode()
    for forbidden in ("ada@example.com", "+1-555-0100", "Ada Lovelace"):
        assert forbidden not in payload
