"""Role-level compatibility analysis service.

Resolves a target job role to a local profile, builds a
JobDescription from that profile, then reuses the existing
deterministic matching pipeline (skills, experience, education,
qualifications) to produce compatibility results, skill gaps,
experience gaps, and improvement recommendations.

All processing is transient and stateless — nothing is persisted,
logged, or sent externally. Role profiles are local data only.
Ollama is not used for determining requirements; it may be used
optionally by callers for generating natural-language explanations
from the structured results.
"""

from __future__ import annotations

import datetime as _dt

from app.ingestion.schemas import FileTypeEnum
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.matching.service import match_resume_to_job
from app.parsing.schemas import ConfidenceLevel, SectionConfidence
from app.role_analysis.profiles import RoleProfile, resolve_role_title
from app.role_analysis.schemas import (
    EducationComparison,
    ExperienceComparison,
    RoleAnalysisRequest,
    RoleAnalysisResult,
    RoleProfileInfo,
    SkillComparison,
)


def _profile_to_job_description(
    profile: RoleProfile,
) -> JobDescription:
    """Build a JobDescription from a role profile's structured expectations."""
    return JobDescription(
        title=profile.title,
        required_skills=list(profile.required_skills),
        preferred_skills=list(profile.preferred_skills),
        experience_requirements=list(profile.experience_requirements),
        education_requirements=list(profile.education_requirements),
        qualifications=list(profile.qualifications),
        responsibilities=list(profile.responsibilities),
        metadata=JobMetadata(
            word_count=0,
            file_type=FileTypeEnum.TXT,
            overall_confidence=ConfidenceLevel.HIGH,
            section_confidence=[
                SectionConfidence(
                    section="required_skills", level=ConfidenceLevel.HIGH
                ),
                SectionConfidence(
                    section="preferred_skills", level=ConfidenceLevel.HIGH
                ),
                SectionConfidence(
                    section="experience", level=ConfidenceLevel.HIGH
                ),
                SectionConfidence(
                    section="education", level=ConfidenceLevel.HIGH
                ),
            ],
        ),
    )


def _years_gap(candidate: float | None, required: float | None) -> float | None:
    if candidate is None or required is None:
        return None
    return round(required - candidate, 1)


def _build_recommendations(
    skills: SkillComparison,
    experience: ExperienceComparison,
    education: EducationComparison,
    gaps: list[str],
) -> list[str]:
    """Generate concrete improvement recommendations from gaps."""
    recs: list[str] = []

    if skills.missing:
        recs.append(
            "Add the following skills to your resume: "
            + ", ".join(skills.missing[:5])
            + ("." if len(skills.missing) <= 5 else ", ...")
        )

    if experience.gap_years is not None and experience.gap_years > 0:
        recs.append(
            f"Your experience is ~{experience.candidate_years} years; "
            f"this role expects ~{experience.required_years} years. "
            f"Gain ~{experience.gap_years} more years of relevant "
            "experience or highlight transferable accomplishments."
        )
    elif experience.met is True:
        recs.append(
            f"Your ~{experience.candidate_years} years of experience "
            f"meets the ~{experience.required_years} year requirement."
        )

    if education.met is False:
        recs.append(
            f"This role requires {education.required_level}-level "
            "education. Consider upskilling or highlighting relevant "
            "certifications and self-study."
        )

    for gap in gaps[:5]:
        if gap not in recs:
            recs.append(gap)

    return recs


def analyze_role_compatibility(
    request: RoleAnalysisRequest,
    reference_date: _dt.date | None = None,
) -> RoleAnalysisResult:
    """Analyze resume compatibility with a target job role.

    ``reference_date`` (default today) keeps date calculations
    traceable for tests.
    """
    reference = reference_date or _dt.date.today()
    profile = resolve_role_title(request.role_title)

    if profile is None:
        unknown_profile = RoleProfileInfo(
            title=request.role_title,
            aliases=[],
            supported=False,
        )
        return RoleAnalysisResult(
            role_title=request.role_title,
            profile=unknown_profile,
            recommendations=[
                f"The role '{request.role_title}' is not in the local "
                "profile database. Try a different role, or upload a "
                "specific job description for targeted analysis.",
            ],
            compatibility={"overall": None},
        )

    job = _profile_to_job_description(profile)

    match = match_resume_to_job(request.resume, job, reference_date=reference)

    required_skills = list(job.required_skills)
    preferred_skills = list(job.preferred_skills)

    have_set = set()
    for bucket in (
        request.resume.skills.technical,
        request.resume.skills.soft,
        request.resume.skills.tools,
        request.resume.skills.languages,
        request.resume.skills.all,
    ):
        for s in bucket:
            if s:
                have_set.add(s.lower())

    for proj in request.resume.projects:
        for tech in proj.technologies:
            if tech:
                have_set.add(tech.lower())

    for entry in request.resume.experience:
        for s in entry.skills_mentioned:
            if s:
                have_set.add(s.lower())

    have_lower = have_set
    skills_have = [
        s for s in required_skills + preferred_skills if s.lower() in have_lower
    ]
    skills_missing = [
        s for s in required_skills + preferred_skills if s.lower() not in have_lower
    ]
    required_have = [s for s in required_skills if s.lower() in have_lower]
    preferred_have = [s for s in preferred_skills if s.lower() in have_lower]

    if match.skill_match is not None:
        matched_required = list(match.skill_match.matched_required)
        matched_preferred = list(match.skill_match.matched_preferred)
    else:
        matched_required = required_have
        matched_preferred = preferred_have

    exp = ExperienceComparison(
        candidate_years=match.experience_match.candidate_years,
        required_years=match.experience_match.required_years,
        required_level=match.experience_match.required_level,
        gap_years=_years_gap(
            match.experience_match.candidate_years,
            match.experience_match.required_years,
        ),
        met=(
            match.experience_match.score == 100.0
            if match.experience_match.score is not None
            else None
        ),
        resume_experience_count=len(request.resume.experience),
    )

    edu = EducationComparison(
        required_level=match.education_match.required_level,
        candidate_level=match.education_match.candidate_level,
        met=match.education_match.matched,
    )

    if (
        match.experience_match.required_years is not None
        and exp.candidate_years is not None
    ):
        exp.gap_years = round(
            match.experience_match.required_years - exp.candidate_years, 1
        )

    matched_reqs = list(match.matched_requirements)
    missing_reqs = list(match.skill_match.missing_required)
    strengths = list(match.strengths)
    gaps = list(match.gaps)

    recommendations = _build_recommendations(
        SkillComparison(
            have=skills_have,
            missing=[s for s in skills_missing if s.lower() not in have_lower],
            matched_count=len(matched_required) + len(matched_preferred),
            required_total=len(required_skills),
            preferred_total=len(preferred_skills),
        ),
        exp,
        edu,
        gaps,
    )

    overall = match.overall_score

    return RoleAnalysisResult(
        role_title=profile.title,
        profile=RoleProfileInfo(
            title=profile.title,
            aliases=list(profile.aliases),
            supported=True,
        ),
        compatibility={
            "overall": overall,
            "skills": match.skill_match.score,
            "experience": match.experience_match.score,
            "education": match.education_match.score,
        },
        skills=SkillComparison(
            have=skills_have,
            missing=skills_missing,
            matched_count=len(matched_required) + len(matched_preferred),
            required_total=len(required_skills),
            preferred_total=len(preferred_skills),
        ),
        experience=exp,
        education=edu,
        matched_requirements=matched_reqs,
        missing_required=missing_reqs,
        strengths=strengths,
        gaps=gaps,
        recommendations=recommendations,
        matched_job=job,
        deterministic_match=match,
    )
