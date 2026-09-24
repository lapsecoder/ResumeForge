"""Unit tests for the Phase 7C Copilot edit-proposal layer.

All content is synthetic. These tests exercise the allowlist, the pure
apply/revert primitive, and the deterministic fact-preservation checks with no
network, no LLM, and no persistence.
"""

from __future__ import annotations

import pytest
from copilot_factories import make_job, make_resume

from app.copilot.editing import (
    apply_edit,
    attach_edit_proposals,
    build_edit_proposal,
    canonical_path,
    is_editable_target,
    parse_edit_path,
    resolve_target_value,
    target_for_request,
    validate_edit_value,
)
from app.copilot.schemas import (
    CopilotEditTarget,
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotSuggestion,
    EditCheckCategory,
    EditStatus,
    ProviderKind,
    ProviderMetadata,
    SuggestionCategory,
    VerificationLevel,
)
from app.parsing.schemas import Project, WorkExperience

# ---------------------------------------------------------------------------
# Allowlist parsing
# ---------------------------------------------------------------------------

_ACCEPTED_PATHS = [
    "summary",
    "experience[0].title",
    "experience[0].company",
    "experience[0].description",
    "experience[0].achievements[0]",
    "experience[0].bullets[1]",
    "projects[2].description",
    "projects[0].technologies",
]

_REJECTED_PATHS = [
    "",
    "   ",
    "summary.x",
    "summary[0]",
    "summary[0].text",
    "experience",
    "experience[0]",
    "experience.title",
    "experience[0].name",
    "experience[0].skills",
    "experience[0].start_date",
    "experience[0].end_date",
    "experience[0].achievements",
    "experience[0].achievements[x]",
    "experience[0].title[0]",
    "experience[-1].title",
    "education[0].institution",
    "certifications[0].name",
    "skills.technical[0]",
    "contact.email",
    "contact.phone",
    "contact.linkedin",
    "projects[0].name",
    "projects[0].technologies[0]",
    "custom_sections[0].content[0]",
    "metadata.word_count",
    "resume.summary",
    "../../etc/passwd",
    "experience[0].description; DROP TABLE resumes",
    "summary\nDROP",
]


@pytest.mark.parametrize("path", _ACCEPTED_PATHS)
def test_parse_edit_path_accepts_allowlisted_paths(path: str) -> None:
    target = parse_edit_path(path)
    assert target is not None
    assert is_editable_target(target)


@pytest.mark.parametrize("path", _REJECTED_PATHS)
def test_parse_edit_path_rejects_everything_else(path: str) -> None:
    assert parse_edit_path(path) is None


def test_bullets_alias_canonicalises_to_achievements() -> None:
    target = parse_edit_path("experience[0].bullets[1]")
    assert target is not None
    assert target.field == "achievements"
    assert target.sub_index == 1
    assert target.path == "experience[0].achievements[1]"


def test_parse_edit_path_does_not_mutate_leading_zeros() -> None:
    target = parse_edit_path("experience[007].title")
    assert target is not None
    assert target.index == 7
    assert target.path == "experience[7].title"


def test_is_editable_target_rejects_inconsistent_structs() -> None:
    forged = CopilotEditTarget(
        path="summary",
        section="experience",
        index=0,
        field="title",
    )
    assert is_editable_target(forged) is False
    assert apply_edit(make_resume(), forged, "x") is None


def test_canonical_path_round_trips() -> None:
    for path in _ACCEPTED_PATHS:
        target = parse_edit_path(path)
        assert target is not None
        assert canonical_path(target) == target.path


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def test_resolve_summary_empty_string_is_insertion_point() -> None:
    resume = make_resume(summary=None)
    assert resolve_target_value(resume, parse_edit_path("summary")) == ""  # type: ignore[arg-type]


def test_resolve_target_value_reads_from_resume() -> None:
    resume = make_resume()
    target = parse_edit_path("experience[0].achievements[1]")
    assert target is not None
    assert (
        resolve_target_value(resume, target)
        == "Reduced deployment time by 30%."
    )


def test_resolve_target_value_none_for_out_of_range() -> None:
    resume = make_resume()
    assert (
        resolve_target_value(resume, parse_edit_path("experience[9].title"))
        is None
    )
    assert (
        resolve_target_value(
            resume, parse_edit_path("experience[0].achievements[9]")
        )
        is None
    )


def test_resolve_technologies_joins_list() -> None:
    resume = make_resume(
        projects=[
            Project(name="Copilot", description="Local.", technologies=["Python", "Go"])
        ]
    )
    target = parse_edit_path("projects[0].technologies")
    assert target is not None
    assert resolve_target_value(resume, target) == "Python, Go"


def test_target_for_request_maps_operations() -> None:
    resume = make_resume()
    summary_request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume
    )
    assert target_for_request(summary_request) is not None
    bullet_request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_BULLET,
        resume=resume,
        target_ref="experience[0].achievements[0]",
    )
    assert target_for_request(bullet_request) is not None
    advisory = CopilotRequest(
        operation=CopilotOperation.IDENTIFY_PRIORITIES, resume=resume
    )
    assert target_for_request(advisory) is None


# ---------------------------------------------------------------------------
# apply_edit (pure; mirrors the frontend allowlist)
# ---------------------------------------------------------------------------


def test_apply_edit_summary_does_not_mutate_input() -> None:
    resume = make_resume(summary="Old summary.")
    target = parse_edit_path("summary")
    assert target is not None
    updated = apply_edit(resume, target, "New summary.")
    assert updated is not None
    assert updated.summary == "New summary."
    assert resume.summary == "Old summary."


def test_apply_edit_scalar_and_bullet_fields() -> None:
    resume = make_resume()
    title = parse_edit_path("experience[0].title")
    bullet = parse_edit_path("experience[0].achievements[0]")
    assert title is not None and bullet is not None
    with_title = apply_edit(resume, title, "Senior Engineer")
    assert with_title is not None
    assert with_title.experience[0].title == "Senior Engineer"
    with_bullet = apply_edit(with_title, bullet, "Built a billing service.")
    assert with_bullet is not None
    assert with_bullet.experience[0].achievements[0] == "Built a billing service."
    assert with_bullet.experience[0].title == "Senior Engineer"


def test_apply_edit_technologies_splits_on_commas() -> None:
    resume = make_resume(
        projects=[Project(name="Copilot", description="Local.")]
    )
    target = parse_edit_path("projects[0].technologies")
    assert target is not None
    updated = apply_edit(resume, target, "Python, Go, Rust")
    assert updated is not None
    assert updated.projects[0].technologies == ["Python", "Go", "Rust"]


def test_apply_edit_rejects_out_of_range_and_non_allowlisted() -> None:
    resume = make_resume()
    out_of_range = parse_edit_path("experience[5].title")
    assert out_of_range is not None
    assert apply_edit(resume, out_of_range, "x") is None
    bullet_out_of_range = parse_edit_path("experience[0].achievements[9]")
    assert bullet_out_of_range is not None
    assert apply_edit(resume, bullet_out_of_range, "x") is None


def test_apply_edit_rejects_oversized_value() -> None:
    from app.copilot.config import copilot_settings

    resume = make_resume(summary="Old.")
    target = parse_edit_path("summary")
    assert target is not None
    oversized = "x" * (copilot_settings.max_edit_value_chars + 1)
    assert apply_edit(resume, target, oversized) is None


# ---------------------------------------------------------------------------
# Fact-preservation checks
# ---------------------------------------------------------------------------


def _suggestion(
    operation: CopilotOperation,
    suggested: str,
    *,
    verification: VerificationLevel = VerificationLevel.VERIFIED,
    requires_confirmation: bool = False,
) -> CopilotSuggestion:
    return CopilotSuggestion(
        id="sug_1",
        operation=operation,
        category=SuggestionCategory.SUMMARY,
        suggested_text=suggested,
        rationale="Synthetic test suggestion.",
        verification=verification,
        requires_user_confirmation=requires_confirmation,
    )


def test_verified_rewrite_passes_every_check() -> None:
    resume = make_resume(summary="I am a software engineer with Python.")
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_SUMMARY,
        "Software engineer with Python.",
    )
    validation = validate_edit_value(
        resume, None, suggestion, resume.summary, suggestion.suggested_text
    )
    assert validation.all_passed is True
    assert validation.requires_user_confirmation is False
    assert {check.category for check in validation.checks} == set(EditCheckCategory)


def test_new_metric_is_rejected() -> None:
    resume = make_resume()
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_BULLET,
        "Reduced deployment time by 45%.",
    )
    validation = validate_edit_value(
        resume,
        None,
        suggestion,
        "Reduced deployment time by 30%.",
        suggestion.suggested_text,
    )
    metric = next(
        c for c in validation.checks if c.category == EditCheckCategory.METRIC
    )
    assert metric.passed is False
    assert validation.requires_user_confirmation is True
    assert validation.all_passed is False


def test_preserved_metric_is_accepted() -> None:
    resume = make_resume()
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_BULLET,
        "Reduced deployment time by 30% and built systems.",
    )
    validation = validate_edit_value(
        resume,
        None,
        suggestion,
        "Reduced deployment time by 30%.",
        suggestion.suggested_text,
    )
    assert validation.all_passed is True


def test_new_date_is_rejected() -> None:
    resume = make_resume()
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_BULLET,
        "Built data processing systems in 2023.",
    )
    validation = validate_edit_value(
        resume,
        None,
        suggestion,
        "Built data processing systems.",
        suggestion.suggested_text,
    )
    date = next(c for c in validation.checks if c.category == EditCheckCategory.DATE)
    assert date.passed is False


def test_invented_employer_is_rejected_as_source_term() -> None:
    resume = make_resume()
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_BULLET,
        "Built data processing systems at Google.",
    )
    validation = validate_edit_value(
        resume,
        None,
        suggestion,
        "Built data processing systems.",
        suggestion.suggested_text,
    )
    source = next(
        c for c in validation.checks if c.category == EditCheckCategory.SOURCE_TERM
    )
    assert source.passed is False


def test_unsupported_job_skill_is_rejected() -> None:
    resume = make_resume()
    job = make_job(required_skills=["Kubernetes"])
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_SUMMARY,
        "Software engineer experienced with Kubernetes.",
    )
    validation = validate_edit_value(
        resume, job, suggestion, resume.summary, suggestion.suggested_text
    )
    skill = next(c for c in validation.checks if c.category == EditCheckCategory.SKILL)
    assert skill.passed is False
    assert "Kubernetes" in skill.detail


def test_supported_job_skill_is_accepted() -> None:
    resume = make_resume()
    job = make_job(required_skills=["Python"])
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_SUMMARY,
        "Software engineer experienced with Python.",
    )
    validation = validate_edit_value(
        resume, job, suggestion, resume.summary, suggestion.suggested_text
    )
    assert validation.all_passed is True


def test_failed_check_downgrades_suggestion_status() -> None:
    resume = make_resume()
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_SUMMARY,
        "I am a software engineer who raised revenue by 90%.",
    )
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume
    )
    proposal = build_edit_proposal(
        request,
        suggestion.model_copy(
            update={"verification": VerificationLevel.VERIFIED}
        ),
    )
    assert proposal is not None
    assert proposal.status == EditStatus.UNVERIFIED
    assert proposal.validation.requires_user_confirmation is True


# ---------------------------------------------------------------------------
# Proposal construction and attachment
# ---------------------------------------------------------------------------


def test_build_proposal_uses_server_resolved_original() -> None:
    resume = make_resume(summary="I am a software engineer with Python.")
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume
    )
    suggestion = _suggestion(
        CopilotOperation.IMPROVE_SUMMARY, "Software engineer with Python."
    )
    proposal = build_edit_proposal(request, suggestion)
    assert proposal is not None
    assert proposal.edit_id == "edit_sug_1"
    assert proposal.target.path == "summary"
    assert proposal.original_value == "I am a software engineer with Python."
    assert proposal.status == EditStatus.PROPOSED


def test_build_proposal_none_for_advisory_or_unmapped() -> None:
    resume = make_resume()
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume
    )
    advisory = _suggestion(
        CopilotOperation.IMPROVE_SUMMARY,
        "",
        verification=VerificationLevel.ADVISORY,
    )
    assert build_edit_proposal(request, advisory) is None
    priorities = _suggestion(
        CopilotOperation.IDENTIFY_PRIORITIES, "Do something."
    )
    assert build_edit_proposal(request, priorities) is None


def test_build_proposal_none_when_target_ref_not_allowlisted() -> None:
    resume = make_resume()
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_BULLET,
        resume=resume,
        target_ref="contact.email",
    )
    suggestion = _suggestion(CopilotOperation.IMPROVE_BULLET, "ada@evil.example")
    assert build_edit_proposal(request, suggestion) is None


def test_attach_proposals_dedupes_duplicate_targets() -> None:
    resume = make_resume(summary="I am a software engineer.")
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_SUMMARY, resume=resume
    )
    response = CopilotResponse(
        operation=CopilotOperation.IMPROVE_SUMMARY,
        explanation="test",
        suggestions=[
            _suggestion(CopilotOperation.IMPROVE_SUMMARY, "Software engineer."),
            _suggestion(CopilotOperation.IMPROVE_SUMMARY, "An engineer."),
        ],
        provider=ProviderMetadata(
            provider=ProviderKind.DETERMINISTIC,
            provider_label="Deterministic",
            available=True,
            fallback_used=True,
            version="test",
        ),
        disclaimer="test",
    )
    attached = attach_edit_proposals(request, response)
    assert attached.suggestions[0].edit is not None
    assert attached.suggestions[1].edit is None


def test_attach_proposals_leaves_non_edits_unchanged() -> None:
    resume = make_resume()
    request = CopilotRequest(
        operation=CopilotOperation.IDENTIFY_PRIORITIES, resume=resume
    )
    response = CopilotResponse(
        operation=CopilotOperation.IDENTIFY_PRIORITIES,
        explanation="test",
        suggestions=[
            _suggestion(CopilotOperation.IDENTIFY_PRIORITIES, "Prioritise X.")
        ],
        provider=ProviderMetadata(
            provider=ProviderKind.DETERMINISTIC,
            provider_label="Deterministic",
            available=True,
            fallback_used=True,
            version="test",
        ),
        disclaimer="test",
    )
    attached = attach_edit_proposals(request, response)
    assert attached.suggestions[0].edit is None


def test_work_experience_factory_override_used_for_bullets() -> None:
    resume = make_resume(
        experience=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                achievements=["I built a billing service."],
            )
        ]
    )
    target = parse_edit_path("experience[0].achievements[0]")
    assert target is not None
    assert resolve_target_value(resume, target) == "I built a billing service."
