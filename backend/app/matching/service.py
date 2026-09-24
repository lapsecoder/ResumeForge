"""Deterministic, explainable baseline matching: Resume vs JobDescription.

Pure/local service: no database, no filesystem, no external services, no
embeddings, no LLM. The pipeline is intentionally layered so a later phase can
add semantic signals without rewriting the deterministic baseline:

    Deterministic signals (this phase)
    +   Semantic signals (future phase, intentionally absent today)
    =   Hybrid MatchResult
"""

from __future__ import annotations

import datetime as _dt

from app.job_parsing.schemas import JobDescription
from app.matching.education_matcher import match_education
from app.matching.experience_matcher import match_experience
from app.matching.qualification_matcher import match_qualifications
from app.matching.schemas import MatchResult
from app.matching.scorer import (
    build_metadata,
    build_narratives,
    overall_score,
)
from app.matching.skill_matcher import match_skills
from app.parsing.schemas import Resume


def match_resume_to_job(
    resume: Resume,
    job: JobDescription,
    *,
    reference_date: _dt.date | None = None,
) -> MatchResult:
    """Return a transient, explainable Baseline Match Score for resume vs job.

    ``reference_date`` (default today) resolves 'Present' experience end dates
    so every calculation is traceable and tests can be fully deterministic.
    """
    reference = reference_date or _dt.date.today()

    skill = match_skills(resume, job)
    experience = match_experience(resume, job, reference)
    education = match_education(resume, job)
    qualification = match_qualifications(resume, job)

    components: dict[str, float | None] = {
        "required_skill": (
            skill.required_coverage * 100
            if skill.required_coverage is not None
            else None
        ),
        "preferred_skill": (
            skill.preferred_coverage * 100
            if skill.preferred_coverage is not None
            else None
        ),
        "experience": experience.score,
        "education": education.score,
        "qualification": qualification.score,
    }

    overall, applied_weights = overall_score(components)
    strengths, gaps, matched_requirements, unmet_requirements = build_narratives(
        skill, experience, education, qualification
    )

    return MatchResult(
        overall_score=overall,
        skill_match=skill,
        experience_match=experience,
        education_match=education,
        qualification_match=qualification,
        strengths=strengths,
        gaps=gaps,
        matched_requirements=matched_requirements,
        unmet_requirements=unmet_requirements,
        metadata=build_metadata(applied_weights, reference),
    )
