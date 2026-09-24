"""Tests for the role analysis API client and profile resolution."""

from __future__ import annotations

from __future__ import annotations

from app.job_parsing.schemas import JobDescription, JobMetadata
from app.matching.service import match_resume_to_job
from app.parsing.schemas import (
    ConfidenceLevel,
    Education,
    Resume,
    ResumeMetadata,
    SkillSet,
    WorkExperience,
)
from app.role_analysis.profiles import (
    SUPPORTED_ROLES,
    resolve_role_title,
)
from app.role_analysis.service import analyze_role_compatibility
from app.role_analysis.schemas import RoleAnalysisRequest

REFERENCE = __import__("datetime").date(2026, 9, 9)


def _resume(
    *,
    skills: list[str] | None = None,
    experience: list[WorkExperience] | None = None,
    education: list[Education] | None = None,
) -> Resume:
    return Resume(
        summary=None,
        experience=experience or [],
        education=education or [],
        skills=SkillSet(
            technical=skills or [],
            soft=[],
            tools=[],
            languages=[],
            all=skills or [],
        ),
        projects=[],
        certifications=[],
        custom_sections=[],
        metadata=ResumeMetadata(
            word_count=0, file_type="text", overall_confidence=ConfidenceLevel.HIGH
        ),
    )


def _stint(start: str, end: str) -> WorkExperience:
    return WorkExperience(
        company="Acme", title="Engineer", start_date=start, end_date=end
    )


class TestProfileResolutionType:
    def test_supported_roles_all_have_titles(self) -> None:
        for role in SUPPORTED_ROLES:
            assert role.title is not None
            assert len(role.title) > 0

    def test_supported_roles_have_required_skills_field(self) -> None:
        for role in SUPPORTED_ROLES:
            assert isinstance(role.required_skills, tuple)

    def test_supported_roles_have_aliases(self) -> None:
        for role in SUPPORTED_ROLES:
            assert isinstance(role.aliases, tuple)


class TestAnalyzeRoleCompatibilityDeterministic:
    def test_deterministic_with_same_input(self) -> None:
        resume = _resume(
            skills=["Python", "JavaScript", "Git", "SQL"],
            experience=[_stint("2020", "2024")],
        )
        req1 = RoleAnalysisRequest(role_title="Software Engineer", resume=resume)
        req2 = RoleAnalysisRequest(role_title="Software Engineer", resume=resume)
        result1 = analyze_role_compatibility(req1, reference_date=REFERENCE)
        result2 = analyze_role_compatibility(req2, reference_date=REFERENCE)
        assert result1.compatibility["overall"] == result2.compatibility["overall"]
        assert result1.skills.missing == result2.skills.missing

    def test_different_roles_different_results(self) -> None:
        resume = _resume(
            skills=["Python", "Docker", "Kubernetes"],
            experience=[_stint("2020", "2024")],
        )
        result_swe = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        result_devops = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="DevOps Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result_swe.role_title != result_devops.role_title


class TestSchemaValidation:
    def test_role_analysis_request_valid(self) -> None:
        from app.role_analysis.schemas import RoleAnalysisRequest

        resume = _resume(skills=["Python"])
        req = RoleAnalysisRequest(role_title="Software Engineer", resume=resume)
        assert req.role_title == "Software Engineer"

    def test_role_analysis_request_accepts_empty_title(self) -> None:
        resume = _resume(skills=["Python"])
        req = RoleAnalysisRequest(role_title="", resume=resume)
        assert req.role_title == ""
        result = analyze_role_compatibility(req, reference_date=REFERENCE)
        assert result.profile.supported is False


class TestJobDescriptionFromProfile:
    def test_profile_to_job_has_required_fields(self) -> None:
        from app.role_analysis.profiles import resolve_role_title
        from app.role_analysis.service import analyze_role_compatibility

        profile = resolve_role_title("Software Engineer")
        assert profile is not None
        resume = _resume(skills=profile.required_skills[:1] or ["Python"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title=profile.title, resume=resume),
            reference_date=REFERENCE,
        )
        assert result.matched_job is not None
        assert isinstance(result.matched_job, JobDescription)
        assert len(result.matched_job.required_skills) > 0
