"""Schema contract tests for the Resume Copilot (Phase 7A).

These pin the stable, versioned shapes the frontend and downstream phases
depend on. Changing a value here is a breaking change.
"""

from __future__ import annotations

import pytest
from copilot_factories import make_resume
from pydantic import ValidationError

from app.copilot.providers.deterministic import DeterministicFallbackProvider
from app.copilot.schemas import (
    CopilotEvidence,
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotSuggestion,
    EvidenceKind,
    ProviderKind,
    ProviderMetadata,
    SuggestionCategory,
    VerificationLevel,
)


def test_copilot_operation_values_are_stable() -> None:
    assert [op.value for op in CopilotOperation] == [
        "improve_summary",
        "improve_bullet",
        "identify_priorities",
        "explain_finding",
        "job_alignment",
        "free-form",
    ]


def test_enums_are_stable() -> None:
    assert [k.value for k in EvidenceKind] == [
        "fact",
        "job_requirement",
        "inference",
        "suggestion",
        "unverified",
    ]
    assert [v.value for v in VerificationLevel] == [
        "verified",
        "inferred",
        "advisory",
        "unverified",
    ]
    assert ProviderKind.LOCAL_OLLAMA.value == "local-ollama"
    assert ProviderKind.DETERMINISTIC.value == "deterministic"
    assert SuggestionCategory.PRIORITY.value == "priority"


def test_request_requires_a_resume() -> None:
    with pytest.raises(ValidationError):
        CopilotRequest(operation=CopilotOperation.IMPROVE_SUMMARY)


def test_request_rejects_unknown_operation_string() -> None:
    raw = {
        "operation": "unlimited_free_text",
        "resume": make_resume().model_dump(mode="json"),
    }
    with pytest.raises(ValidationError):
        CopilotRequest.model_validate(raw)


def test_free_form_request_accepts_user_request() -> None:
    req = CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=make_resume(),
        user_request="Make my summary more concise.",
    )
    decoded = CopilotRequest.model_validate_json(req.model_dump_json())
    assert decoded.operation == CopilotOperation.FREE_FORM
    assert decoded.user_request == "Make my summary more concise."


def test_request_resume_round_trip_is_content_identical() -> None:
    resume = make_resume(
        summary="A summary containing email-like text user@example.com.",
    )
    req = CopilotRequest(
        operation=CopilotOperation.EXPLAIN_FINDING,
        resume=resume,
        finding_ref="exp-action-verbs",
    )
    decoded = CopilotRequest.model_validate_json(
        req.model_dump_json(),
    )
    assert decoded.operation == req.operation
    assert decoded.resume.summary == resume.summary


def test_suggestion_ids_are_stable_per_response() -> None:
    provider = DeterministicFallbackProvider()
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.IMPROVE_BULLET,
            resume=make_resume(),
            target_text="I built a service.",
        )
    )
    ids = [s.id for s in resp.suggestions]
    assert ids == [f"sug_{i}" for i in range(1, len(ids) + 1)]


def test_suggestion_phase_7b_fields_default_to_neutral() -> None:
    suggestion = CopilotSuggestion(
        id="sug_1",
        operation=CopilotOperation.IMPROVE_SUMMARY,
        category=SuggestionCategory.SUMMARY,
        rationale="Because.",
        verification=VerificationLevel.ADVISORY,
    )
    assert suggestion.priority is None
    assert suggestion.issue == ""
    assert suggestion.recommendation == ""
    assert suggestion.impact == ""


def test_priority_fields_round_trip() -> None:
    suggestion = CopilotSuggestion(
        id="sug_1",
        operation=CopilotOperation.IDENTIFY_PRIORITIES,
        category=SuggestionCategory.QUANTIFICATION,
        rationale="Because.",
        verification=VerificationLevel.ADVISORY,
        priority=2,
        issue="Bullets lack measurable outcomes.",
        recommendation="Add real numbers where you have them.",
        impact="high",
    )
    decoded = CopilotSuggestion.model_validate_json(suggestion.model_dump_json())
    assert decoded.priority == 2
    assert decoded.issue == "Bullets lack measurable outcomes."
    assert decoded.recommendation == "Add real numbers where you have them."
    assert decoded.impact == "high"


def test_response_shape_serialises_to_expected_json() -> None:
    suggestion = CopilotSuggestion(
        id="sug_1",
        operation=CopilotOperation.IMPROVE_SUMMARY,
        category=SuggestionCategory.SUMMARY,
        original_text="Original",
        suggested_text="Suggested",
        rationale="Because.",
        evidence=[CopilotEvidence(kind=EvidenceKind.FACT, source="s", statement="t")],
        verification=VerificationLevel.VERIFIED,
    )
    response = CopilotResponse(
        operation=CopilotOperation.IMPROVE_SUMMARY,
        explanation="Explanation.",
        suggestions=[suggestion],
        provider=ProviderMetadata(
            provider=ProviderKind.DETERMINISTIC,
            provider_label="Deterministic fallback (offline rules)",
            available=True,
            fallback_used=True,
            version="7b-copilot-1.0",
        ),
        disclaimer="Disclaimer.",
    )
    data = response.model_dump(mode="json")
    assert data["operation"] == "improve_summary"
    assert data["provider"]["provider"] == "deterministic"
    assert data["suggestions"][0]["id"] == "sug_1"
    assert data["suggestions"][0]["verification"] == "verified"
