"""Deterministic fallback provider tests for the Resume Copilot (Phase 7A).

Offline by construction: no model, no Ollama, no internet. Verifies the
fallback executes every operation safely and never fabricates evidence.
"""

from __future__ import annotations

import re

import pytest
from copilot_factories import (
    make_ats_result,
    make_context,
    make_finding,
    make_job,
    make_match_result,
    make_resume,
)

from app.copilot.config import COPILOT_IMPLEMENTATION_VERSION, copilot_settings
from app.copilot.errors import InvalidRequestError
from app.copilot.providers import DeterministicFallbackProvider
from app.copilot.schemas import (
    CopilotOperation,
    CopilotRequest,
    EvidenceKind,
    ProviderKind,
    SuggestionCategory,
    VerificationLevel,
)
from app.parsing.schemas import Project, SkillSet

provider = DeterministicFallbackProvider()

_NUMERIC_VALUE = re.compile(r"\d+(?:\.\d+)?")


def _suggest(operation: CopilotOperation, **kwargs: object):
    return provider.run(
        CopilotRequest(operation=operation, resume=make_resume(), **kwargs)
    )


def test_provider_is_always_available_and_metadata_is_clear() -> None:
    assert provider.available() is True
    assert provider.provider_kind() == ProviderKind.DETERMINISTIC
    meta = provider.metadata()
    assert meta.available is True
    assert meta.fallback_used is True
    assert meta.version == COPILOT_IMPLEMENTATION_VERSION
    assert "Ollama" not in (meta.provider_label or "")


def test_improve_bullet_rewrites_first_person_and_preserves_facts() -> None:
    resp = _suggest(
        CopilotOperation.IMPROVE_BULLET,
        target_text="I built a billing service used by thousands of users.",
    )
    assert resp.provider.provider == ProviderKind.DETERMINISTIC
    rewrite = next(s for s in resp.suggestions if s.suggested_text)
    assert rewrite.suggested_text == (
        "Built a billing service used by thousands of users."
    )
    assert rewrite.verification == VerificationLevel.VERIFIED
    for token in rewrite.suggested_text.lower().split():
        assert token not in {"i", "me", "my"}
    original_words = set(
        re.split(r"[^a-z]+", "built a billing service used by thousands of users")
    ) - {""}
    rewritten_words = set(re.split(r"[^a-z]+", rewrite.suggested_text.lower())) - {""}
    assert original_words <= rewritten_words


def test_improve_bullet_never_invents_metrics() -> None:
    resp = _suggest(
        CopilotOperation.IMPROVE_BULLET,
        target_text="Built a billing service.",
    )
    for suggestion in resp.suggestions:
        assert not _NUMERIC_VALUE.search(suggestion.suggested_text)
    quantification = next(
        s for s in resp.suggestions if s.category == SuggestionCategory.QUANTIFICATION
    )
    assert quantification.original_text == "Built a billing service."
    assert quantification.suggested_text == ""
    assert quantification.requires_user_confirmation is True
    assert quantification.verification == VerificationLevel.UNVERIFIED


def test_improve_summary_suggests_when_missing() -> None:
    resume = make_resume(summary=None)
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.IMPROVE_SUMMARY,
            resume=resume,
        )
    )
    assert resp.suggestions[0].category == SuggestionCategory.SUMMARY
    assert resp.suggestions[0].verification == VerificationLevel.INFERRED
    assert resp.suggestions[0].suggested_text


def test_improve_summary_draft_is_built_only_from_resume_facts() -> None:
    resume = make_resume(
        summary=None,
        experience=[],
        skills=SkillSet(technical=[], tools=[], languages=[], soft=[], all=[]),
    )
    resp = provider.run(
        CopilotRequest(operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume)
    )
    assert resp.suggestions[0].verification == VerificationLevel.ADVISORY
    assert resp.suggestions[0].suggested_text == ""


def test_improve_summary_rewrite_is_fact_preserving() -> None:
    resume = make_resume(
        summary="I am a software engineer with five years of experience."
    )
    resp = provider.run(
        CopilotRequest(operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume)
    )
    rewritten = [s for s in resp.suggestions if s.suggested_text]
    assert rewritten
    for token in ("engineer", "five", "years", "experience"):
        assert token in rewritten[0].suggested_text.lower()


def test_identify_priorities_reflects_findings_when_supplied() -> None:
    resume = make_resume()
    context = make_context(
        ats=make_ats_result(
            make_finding(
                "exp-action-verbs",
                "Achievements should open with action verbs.",
                impact=2.0,
            )
        )
    )
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.IDENTIFY_PRIORITIES,
            resume=resume,
            analysis=context,
        )
    )
    assert resp.suggestions
    priority = resp.suggestions[0]
    assert priority.category == SuggestionCategory.BULLET
    assert "action verbs" in priority.rationale.lower()
    assert priority.priority == 1
    assert priority.issue
    assert priority.recommendation
    assert priority.impact == "medium"


def test_identify_priorities_flags_missing_required_skill_without_claiming_it() -> None:
    resume = make_resume()
    job = make_job(required_skills=["Python", "Kubernetes"])
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.IDENTIFY_PRIORITIES,
            resume=resume,
            job_description=job,
            analysis=make_context(),
        )
    )
    labels = " ".join(s.rationale for s in resp.suggestions).lower()
    assert "kubernetes" in labels
    kubernetes_suggestions = [
        s for s in resp.suggestions if "kubernetes" in s.rationale.lower()
    ]
    assert kubernetes_suggestions
    for suggestion in kubernetes_suggestions:
        assert "do not add" in suggestion.rationale.lower()
        assert "if you" in suggestion.rationale.lower()


def test_identify_priorities_works_without_any_analysis() -> None:
    resp = _suggest(CopilotOperation.IDENTIFY_PRIORITIES)
    assert resp.suggestions


def test_explain_finding_grounds_in_supplied_finding() -> None:
    context = make_context(
        ats=make_ats_result(
            make_finding(
                "exp-vague",
                "Avoid vague phrasing.",
                recommendation="Replace vague phrasing with specifics.",
            )
        )
    )
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.EXPLAIN_FINDING,
            resume=make_resume(),
            finding_ref="exp-vague",
            analysis=context,
        )
    )
    assert resp.suggestions[0].suggested_text == (
        "Replace vague phrasing with specifics."
    )
    assert resp.suggestions[0].verification == VerificationLevel.INFERRED
    assert resp.suggestions[0].evidence[0].reference == "exp-vague"


def test_explain_finding_unknown_ref_is_graceful() -> None:
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.EXPLAIN_FINDING,
            resume=make_resume(),
            finding_ref="does-not-exist",
            analysis=make_context(),
        )
    )
    assert resp.suggestions
    assert "does-not-exist" in resp.suggestions[0].rationale
    assert resp.suggestions[0].verification == VerificationLevel.ADVISORY


def test_job_alignment_highlights_existing_matches_as_verified() -> None:
    resume = make_resume()
    job = make_job(required_skills=["Python"], preferred_skills=["PostgreSQL"])
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.JOB_ALIGNMENT,
            resume=resume,
            job_description=job,
            analysis=make_context(
                match=make_match_result(
                    matched_required=["Python"],
                    matched_preferred=["PostgreSQL"],
                    missing_required=["Kubernetes"],
                )
            ),
        )
    )
    validators = [
        s for s in resp.suggestions if s.verification == VerificationLevel.VERIFIED
    ]
    assert "python" in validators[0].rationale.lower()


def test_job_alignment_never_asserts_possession_of_missing_skill() -> None:
    resume = make_resume()
    job = make_job(required_skills=["Kubernetes", "Spark"])
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.JOB_ALIGNMENT,
            resume=resume,
            job_description=job,
            analysis=make_context(),
        )
    )
    missing = [
        s for s in resp.suggestions if s.verification == VerificationLevel.UNVERIFIED
    ]
    verified = [
        s for s in resp.suggestions if s.verification == VerificationLevel.VERIFIED
    ]
    assert missing
    for suggestion in missing:
        assert suggestion.suggested_text == ""
        kinds = [e.kind for e in suggestion.evidence]
        assert EvidenceKind.JOB_REQUIREMENT in kinds
        assert EvidenceKind.UNVERIFIED in kinds
        assert (
            "you have not used it" in suggestion.rationale.lower()
            or "do not add" in suggestion.rationale.lower()
        )
    for suggestion in verified:
        assert "spark" not in suggestion.rationale.lower()
        assert "kubernetes" not in suggestion.rationale.lower()


def test_all_operations_respect_max_suggestions_bounds() -> None:
    cases = [
        (CopilotOperation.IMPROVE_SUMMARY, {}),
        (
            CopilotOperation.IMPROVE_BULLET,
            {"target_text": "I built a service with Python."},
        ),
        (CopilotOperation.IDENTIFY_PRIORITIES, {}),
        (CopilotOperation.EXPLAIN_FINDING, {}),
        (
            CopilotOperation.JOB_ALIGNMENT,
            {"job_description": make_job()},
        ),
        (
            CopilotOperation.FREE_FORM,
            {"user_request": "Improve my summary"},
        ),
        (
            CopilotOperation.FREE_FORM,
            {"user_request": "Tell me a joke"},
        ),
    ]
    for operation, kwargs in cases:
        resp = provider.run(
            CopilotRequest(operation=operation, resume=make_resume(), **kwargs)
        )
        assert 1 <= len(resp.suggestions) <= copilot_settings.max_suggestions


def test_supports_contract_input_requirements() -> None:
    summary_request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY, resume=make_resume()
    )
    assert provider.supports(summary_request)
    bullet_without_target = CopilotRequest(
        operation=CopilotOperation.IMPROVE_BULLET, resume=make_resume()
    )
    assert provider.supports(bullet_without_target) is False
    alignment_without_job = CopilotRequest(
        operation=CopilotOperation.JOB_ALIGNMENT, resume=make_resume()
    )
    assert provider.supports(alignment_without_job) is False
    free_form_without_request = CopilotRequest(
        operation=CopilotOperation.FREE_FORM, resume=make_resume()
    )
    assert provider.supports(free_form_without_request) is False
    free_form_with_request = CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=make_resume(),
        user_request="Improve my summary.",
    )
    assert provider.supports(free_form_with_request) is True


def test_free_form_routes_summary_intent_and_preserves_facts() -> None:
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=make_resume(),
            user_request="Rewriting my summary",
        )
    )
    assert resp.operation == CopilotOperation.FREE_FORM
    rewritten = [s for s in resp.suggestions if s.suggested_text]
    if rewritten:
        for token in ("engineer", "python", "postgresql"):
            assert token in rewritten[0].suggested_text.lower()


def test_free_form_routes_bullet_intent_to_bullet_rewrite() -> None:
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=make_resume(),
            user_request="Improve the wording of this bullet.",
            target_text="I built a billing service.",
        )
    )
    categories = {s.category for s in resp.suggestions}
    assert categories & {
        SuggestionCategory.CLARITY,
        SuggestionCategory.BULLET,
        SuggestionCategory.QUANTIFICATION,
        SuggestionCategory.ACTION_WORDING,
    }
    for suggestion in resp.suggestions:
        assert not _NUMERIC_VALUE.search(suggestion.suggested_text)


def test_free_form_routes_description_edit_to_strengthened_rewrite() -> None:
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
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request=(
                "Rewrite my Python project description to be stronger for a "
                "Machine Learning Engineer role while preserving every fact "
                "and not adding technologies I don't already have."
            ),
        )
    )
    rewrite = next((s for s in resp.suggestions if s.suggested_text), None)
    assert rewrite is not None
    assert rewrite.category == SuggestionCategory.CLARITY
    assert rewrite.verification == VerificationLevel.INFERRED
    assert rewrite.suggested_text == (
        "Built a Python data pipeline that processes millions of rows for "
        "analytics dashboards. Key technologies: PyTorch."
    )


def test_free_form_routes_job_intent_to_alignment() -> None:
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=make_resume(),
            job_description=make_job(required_skills=["Python", "Kubernetes"]),
            user_request="How do I match this job better?",
            analysis=make_context(
                match=make_match_result(
                    matched_required=["Python"],
                    missing_required=["Kubernetes"],
                )
            ),
        )
    )
    categories = {s.category for s in resp.suggestions}
    assert SuggestionCategory.ALIGNMENT in categories
    for suggestion in resp.suggestions:
        assert suggestion.suggested_text == ""


def test_free_form_routes_priority_intent_to_priorities() -> None:
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=make_resume(),
            user_request="What are the most important things to fix?",
            analysis=make_context(
                ats=make_ats_result(
                    make_finding(
                        "exp-action-verbs",
                        "Achievements should open with action verbs.",
                        impact=2.0,
                    )
                )
            ),
        )
    )
    assert SuggestionCategory.BULLET in {s.category for s in resp.suggestions}


def test_free_form_unknown_request_is_advisory_and_factsafe() -> None:
    resp = provider.run(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=make_resume(),
            user_request="Tell me a joke about resumes.",
        )
    )
    assert resp.suggestions
    first = resp.suggestions[0]
    assert first.verification == VerificationLevel.ADVISORY
    assert first.suggested_text == ""
    assert not _NUMERIC_VALUE.search(first.rationale)


def test_run_rejects_unsupported_inputs() -> None:
    with pytest.raises(InvalidRequestError):
        provider.run(
            CopilotRequest(
                operation=CopilotOperation.JOB_ALIGNMENT, resume=make_resume()
            )
        )


def test_deterministic_run_is_stateless_and_never_mutates_inputs() -> None:
    """Running the provider leaves request and resume byte-identical."""
    resume = make_resume(summary="Summary that must not change.")
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY,
        resume=resume,
        target_text="Unused because operation ignores it.",
    )
    before_request = request.model_dump(mode="json")
    before_summary = resume.summary
    provider.run(request)
    assert resume.summary == before_summary
    assert request.model_dump(mode="json") == before_request
