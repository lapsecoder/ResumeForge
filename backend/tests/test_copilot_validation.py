"""Factual validation tests for the Resume Copilot (Phase 7B).

These tests pin the hallucination guard: untraceable numbers, unsupported job
skills, and fabricated wording are downgraded to unverified, confirmation-
required suggestions; traceable rewrites pass untouched; and unresolvable
evidence paths are dropped. No network, no model, no persistence.
"""

from __future__ import annotations

import pytest
from copilot_factories import make_job, make_resume

from app.copilot.schemas import (
    CopilotEvidence,
    CopilotOperation,
    CopilotRequest,
    CopilotSuggestion,
    EvidenceKind,
    SuggestionCategory,
    VerificationLevel,
)
from app.copilot.validation import (
    _canonical_number,
    _word_present,
    validate_llm_suggestions,
)


def _evidence(reference: str = "resume.experience") -> CopilotEvidence:
    return CopilotEvidence(
        kind=EvidenceKind.FACT,
        source="resume.experience",
        statement="Grounding statement.",
        reference=reference,
    )


def _suggestion(
    suggested_text: str,
    *,
    original_text: str = "",
    verification: VerificationLevel = VerificationLevel.VERIFIED,
    evidence: list[CopilotEvidence] | None = None,
) -> CopilotSuggestion:
    return CopilotSuggestion(
        id="sug_1",
        operation=CopilotOperation.IMPROVE_BULLET,
        category=SuggestionCategory.CLARITY,
        original_text=original_text,
        suggested_text=suggested_text,
        rationale="Because.",
        evidence=[_evidence()] if evidence is None else evidence,
        verification=verification,
    )


def _validate(
    suggestion: CopilotSuggestion, request: CopilotRequest
) -> CopilotSuggestion:
    return validate_llm_suggestions(request, [suggestion])[0]


def _bullet_request(**overrides: object) -> CopilotRequest:
    base = CopilotRequest(
        operation=CopilotOperation.IMPROVE_BULLET,
        resume=make_resume(),
        target_text="I built stuff.",
    )
    return base.model_copy(update=overrides)


# ---------------------------------------------------------------------------
# Traceable rewrites pass untouched
# ---------------------------------------------------------------------------


def test_traceable_rewrite_is_preserved() -> None:
    suggestion = _suggestion(
        "Built a billing service used by thousands of users."
    )
    result = _validate(suggestion, _bullet_request())
    assert result.verification == VerificationLevel.VERIFIED
    assert result.requires_user_confirmation is False


def test_advisory_guidance_without_text_is_not_checked() -> None:
    suggestion = _suggestion("", verification=VerificationLevel.ADVISORY)
    result = _validate(suggestion, _bullet_request())
    assert result.verification == VerificationLevel.ADVISORY
    assert result.requires_user_confirmation is False


# ---------------------------------------------------------------------------
# Metric validation
# ---------------------------------------------------------------------------


def test_hallucinated_metric_is_demoted() -> None:
    suggestion = _suggestion(
        "Built a billing service used by 1,000,000 users each month."
    )
    result = _validate(suggestion, _bullet_request())
    assert result.verification == VerificationLevel.UNVERIFIED
    assert result.requires_user_confirmation is True
    assert any(
        evidence.source == "copilot.validation" for evidence in result.evidence
    )


def test_metric_supplied_in_target_text_is_allowed() -> None:
    request = _bullet_request(target_text="Supported 42 production services.")
    suggestion = _suggestion(
        "Supported 42 production services.",
        original_text="Supported 42 production services.",
    )
    result = _validate(suggestion, request)
    assert result.verification == VerificationLevel.VERIFIED


def test_number_equivalence_ignores_formatting() -> None:
    request = _bullet_request(target_text="Reduced deployment time by 30%.")
    suggestion = _suggestion(
        "Reduced deployment time by 30%.",
        original_text="Reduced deployment time by 30%.",
    )
    assert _validate(suggestion, request).verification == VerificationLevel.VERIFIED


# ---------------------------------------------------------------------------
# Skill validation
# ---------------------------------------------------------------------------


def test_unsupported_job_skill_claim_is_demoted() -> None:
    request = _bullet_request(
        job_description=make_job(required_skills=["Kubernetes"]),
    )
    suggestion = _suggestion("Experienced with Kubernetes and Python.")
    result = _validate(suggestion, request)
    assert result.verification == VerificationLevel.UNVERIFIED
    assert result.requires_user_confirmation is True


def test_supported_job_skill_claim_is_allowed() -> None:
    request = _bullet_request(
        job_description=make_job(required_skills=["Python", "PostgreSQL"]),
    )
    suggestion = _suggestion("Experienced with Python and PostgreSQL.")
    result = _validate(suggestion, request)
    assert result.verification == VerificationLevel.VERIFIED


def test_job_listing_alone_does_not_support_a_skill_claim() -> None:
    # The JD is not a source of resume facts: even though the job lists
    # Kubernetes, the resume does not, so the claim is demoted.
    request = _bullet_request(
        job_description=make_job(required_skills=["Kubernetes"]),
    )
    suggestion = _suggestion("Kubernetes specialist.")
    result = _validate(suggestion, request)
    assert result.verification == VerificationLevel.UNVERIFIED


def test_word_boundary_prevents_substring_false_positives() -> None:
    assert _word_present("experienced with java", "java") is True
    assert _word_present("experienced with javascript", "java") is False
    assert _word_present("c++ developer", "c++") is True


# ---------------------------------------------------------------------------
# New-term validation
# ---------------------------------------------------------------------------


def test_fabricated_wording_is_demoted() -> None:
    suggestion = _suggestion(
        "Built a blockchain ledger for the company."
    )
    result = _validate(suggestion, _bullet_request())
    assert result.verification == VerificationLevel.UNVERIFIED
    assert result.requires_user_confirmation is True


def test_new_employer_in_rewrite_is_demoted() -> None:
    suggestion = _suggestion(
        "Built a billing service at Globex Corporation."
    )
    result = _validate(suggestion, _bullet_request())
    assert result.verification == VerificationLevel.UNVERIFIED


# ---------------------------------------------------------------------------
# Evidence path validation
# ---------------------------------------------------------------------------


def test_out_of_range_evidence_path_is_dropped() -> None:
    suggestion = _suggestion(
        "Built a billing service used by thousands of users.",
        evidence=[_evidence("experience[99]")],
    )
    result = _validate(suggestion, _bullet_request())
    references = [evidence.reference for evidence in result.evidence]
    assert "experience[99]" not in references


def test_valid_structured_evidence_path_is_kept() -> None:
    suggestion = _suggestion(
        "Built a billing service used by thousands of users.",
        evidence=[_evidence("experience[0].achievements[0]")],
    )
    result = _validate(suggestion, _bullet_request())
    references = [evidence.reference for evidence in result.evidence]
    assert "experience[0].achievements[0]" in references


def test_resume_namespace_evidence_is_kept() -> None:
    suggestion = _suggestion(
        "Built a billing service used by thousands of users.",
        evidence=[_evidence("resume.skills")],
    )
    result = _validate(suggestion, _bullet_request())
    references = [evidence.reference for evidence in result.evidence]
    assert "resume.skills" in references


def test_empty_evidence_gains_a_safe_fallback() -> None:
    suggestion = _suggestion(
        "Built a billing service used by thousands of users.", evidence=[]
    )
    result = _validate(suggestion, _bullet_request())
    assert result.evidence
    assert result.evidence[0].source == "copilot.validation"


# ---------------------------------------------------------------------------
# Independence and helpers
# ---------------------------------------------------------------------------


def test_multiple_suggestions_are_validated_independently() -> None:
    good = _suggestion("Built a billing service used by thousands of users.")
    bad = _suggestion("Built a billing service used by 9,999 users.")
    validated = validate_llm_suggestions(_bullet_request(), [good, bad])
    assert validated[0].verification == VerificationLevel.VERIFIED
    assert validated[1].verification == VerificationLevel.UNVERIFIED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000", "1000"),
        ("30", "30"),
        ("2.50", "2.5"),
        ("007", "7"),
        ("0", "0"),
    ],
)
def test_number_canonicalisation(raw: str, expected: str) -> None:
    assert _canonical_number(raw) == expected
