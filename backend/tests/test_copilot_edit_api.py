"""API contract tests for Phase 7C edit proposals.

Verifies the additive ``edit`` block on ``POST /api/v1/copilot/suggest`` and
that the 7B response contract is preserved unchanged. Offline by construction:
a deterministic-only service is injected, so no local Ollama call is made.
"""

from __future__ import annotations

import pytest
from copilot_factories import make_resume
from fastapi.testclient import TestClient

from app.copilot.config import COPILOT_IMPLEMENTATION_VERSION
from app.copilot.providers import DeterministicFallbackProvider
from app.copilot.service import CopilotService
from app.main import app
from app.parsing.schemas import ContactInfo, Project, WorkExperience

client = TestClient(app)

_EDIT_CHECK_CATEGORIES = {"metric", "date", "skill", "source_term"}


@pytest.fixture(autouse=True)
def _deterministic_only_service(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api.v1 import copilot as copilot_router

    monkeypatch.setattr(
        copilot_router,
        "_service",
        CopilotService(providers=[DeterministicFallbackProvider()]),
    )


def _payload(operation: str, resume: object, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "operation": operation,
        "resume": resume.model_dump(mode="json"),  # type: ignore[union-attr]
    }
    payload.update(overrides)
    return payload


def _rewritable_summary_resume():  # type: ignore[no-untyped-def]
    return make_resume(summary="I built scalable systems with Python.")


def test_improve_summary_returns_edit_proposal() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_summary", _rewritable_summary_resume()),
    )
    assert response.status_code == 200
    suggestion = response.json()["suggestions"][0]
    edit = suggestion["edit"]
    assert edit is not None
    assert edit["edit_id"] == f"edit_{suggestion['id']}"
    assert edit["operation"] == "improve_summary"
    assert edit["target"]["path"] == "summary"
    assert edit["target"]["section"] == "summary"
    assert edit["target"]["index"] is None
    assert edit["original_value"] == "I built scalable systems with Python."
    assert edit["proposed_value"] == "Built scalable systems with Python."
    assert edit["reason"]
    assert edit["status"] == "proposed"
    assert edit["validation"]["requires_user_confirmation"] is False
    categories = {check["category"] for check in edit["validation"]["checks"]}
    assert categories == _EDIT_CHECK_CATEGORIES
    assert all(check["passed"] for check in edit["validation"]["checks"])


def test_improve_bullet_with_target_ref_returns_edit() -> None:
    resume = make_resume(
        experience=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                achievements=["I built a billing service."],
            )
        ]
    )
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "improve_bullet",
            resume,
            target_ref="experience[0].achievements[0]",
        ),
    )
    assert response.status_code == 200
    edit = response.json()["suggestions"][0]["edit"]
    assert edit is not None
    assert edit["target"]["path"] == "experience[0].achievements[0]"
    assert edit["target"]["field"] == "achievements"
    assert edit["target"]["sub_index"] == 0
    assert edit["original_value"] == "I built a billing service."
    assert edit["proposed_value"] == "Built a billing service."


def test_improve_bullet_with_text_only_has_no_edit() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "improve_bullet",
            make_resume(),
            target_text="I built a billing service.",
        ),
    )
    assert response.status_code == 200
    assert all(
        suggestion["edit"] is None for suggestion in response.json()["suggestions"]
    )


def test_free_form_with_target_ref_returns_edit() -> None:
    resume = make_resume(
        experience=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                achievements=["I built a billing service."],
            )
        ]
    )
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "free-form",
            resume,
            user_request="Improve the wording of this bullet.",
            target_ref="experience[0].achievements[0]",
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["operation"] == "free-form"
    edits = [
        suggestion["edit"]
        for suggestion in body["suggestions"]
        if suggestion["edit"] is not None
    ]
    assert edits
    assert edits[0]["target"]["path"] == "experience[0].achievements[0]"
    assert edits[0]["original_value"] == "I built a billing service."
    assert edits[0]["proposed_value"] == "Built a billing service."


def test_free_form_without_target_is_advisory_has_no_edit() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "free-form",
            make_resume(),
            user_request="What should I focus on to look stronger?",
        ),
    )
    assert response.status_code == 200
    for suggestion in response.json()["suggestions"]:
        assert suggestion["edit"] is None


def test_free_form_with_inferred_description_target_returns_edit() -> None:
    resume = make_resume(
        projects=[
            Project(
                name="Python Data Pipeline",
                description=(
                    "Built a Python data pipeline that processes millions of "
                    "rows for analytics dashboards."
                ),
                technologies=["Python", "PyTorch"],
            )
        ]
    )
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "free-form",
            resume,
            user_request=(
                "Rewrite my Python project description to be stronger for a "
                "Machine Learning Engineer role while preserving every fact "
                "and not adding technologies I don't already have."
            ),
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["operation"] == "free-form"
    edits = [
        suggestion["edit"]
        for suggestion in body["suggestions"]
        if suggestion["edit"] is not None
    ]
    assert edits
    assert edits[0]["target"]["path"] == "projects[0].description"
    assert edits[0]["target"]["field"] == "description"
    assert edits[0]["original_value"] == resume.projects[0].description
    assert edits[0]["proposed_value"].endswith("Key technologies: PyTorch.")
    assert edits[0]["status"] == "unverified"
    assert edits[0]["validation"]["requires_user_confirmation"] is True


def test_advisory_operations_have_no_edit() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("identify_priorities", make_resume()),
    )
    assert response.status_code == 200
    assert response.json()["suggestions"]
    assert all(
        suggestion["edit"] is None for suggestion in response.json()["suggestions"]
    )


def test_advisory_summary_suggestion_has_no_edit() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_summary", make_resume()),
    )
    assert response.status_code == 200
    for suggestion in response.json()["suggestions"]:
        if suggestion["suggested_text"]:
            continue
        assert suggestion["edit"] is None


def test_malicious_target_ref_has_no_edit() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload(
            "improve_bullet",
            make_resume(),
            target_ref="contact.email",
            target_text="I built a billing service.",
        ),
    )
    assert response.status_code == 200
    assert all(
        suggestion["edit"] is None for suggestion in response.json()["suggestions"]
    )


def test_7b_suggestion_contract_is_still_present() -> None:
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_summary", _rewritable_summary_resume()),
    )
    suggestion = response.json()["suggestions"][0]
    for field in (
        "id",
        "operation",
        "category",
        "original_text",
        "suggested_text",
        "rationale",
        "evidence",
        "verification",
        "requires_user_confirmation",
    ):
        assert field in suggestion


def test_status_reports_current_version() -> None:
    response = client.get("/api/v1/copilot/status")
    assert response.status_code == 200
    assert response.json()["version"] == COPILOT_IMPLEMENTATION_VERSION
    assert COPILOT_IMPLEMENTATION_VERSION == "7d-copilot-1.0"


def test_edit_response_does_not_echo_contact_pii() -> None:
    resume = make_resume(
        summary="I am a software engineer with Python.",
        contact=ContactInfo(
            name="Ada Lovelace",
            email="ada@example.com",
            phone="+1-555-0100",
        ),
    )
    response = client.post(
        "/api/v1/copilot/suggest",
        json=_payload("improve_summary", resume),
    )
    assert response.status_code == 200
    payload = response.content.decode()
    for forbidden in ("ada@example.com", "+1-555-0100", "Ada Lovelace"):
        assert forbidden not in payload
