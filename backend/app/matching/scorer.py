"""Overall scoring and deterministic explanation generation.

The Baseline Match Score is a weighted average of the component scores:

- required skills: 50  (coverage of job.required_skills)
- preferred skills: 15 (coverage of job.preferred_skills)
- experience:       20 (candidate years vs required years)
- education:        10 (degree-level requirement met)
- qualifications:    5 (explicit certification qualification coverage)

When a criterion is not evaluable -- the job defines no requirement for it, or
the resume data is insufficient -- that component is EXCLUDED and the remaining
weights are redistributed proportionally. A candidate is never penalised for a
criterion the job did not ask for, and never scored 0 or 100 merely because a
criterion could not be evaluated. If NO component is evaluable the overall
score is ``None`` (insufficient requirements to score against).

The result is explicitly NOT a probability of hiring, an ATS score, a recruiter
decision, or a validated employability measure -- it is a deterministic
"Baseline Match Score" used purely for explainable triage.
"""

from __future__ import annotations

import datetime as _dt

from app.matching.schemas import (
    SCORE_WEIGHTS_DEFAULT,
    EducationMatch,
    ExperienceMatch,
    MatchMetadata,
    QualificationMatch,
    SkillMatch,
)

_COMPONENT_ORDER = (
    "required_skill",
    "preferred_skill",
    "experience",
    "education",
    "qualification",
)


def overall_score(
    components: dict[str, float | None],
) -> tuple[float | None, dict[str, float]]:
    """Return (overall 0-100 | None, applied weights after redistribution)."""
    applied: dict[str, float] = {}
    weighted_sum = 0.0
    for component in _COMPONENT_ORDER:
        score = components.get(component)
        if score is None:
            continue
        weight = SCORE_WEIGHTS_DEFAULT[component]
        applied[component] = weight
        weighted_sum += weight * score

    if not applied:
        return None, {}

    total_weight = sum(applied.values())
    return round(weighted_sum / total_weight, 1), applied


def build_narratives(
    skill: SkillMatch,
    experience: ExperienceMatch,
    education: EducationMatch,
    qualification: QualificationMatch,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (strengths, gaps, matched_requirements, unmet_requirements)."""
    strengths: list[str] = []
    gaps: list[str] = []
    matched_requirements: list[str] = []
    unmet_requirements: list[str] = []

    if skill.total_required and skill.matched_required:
        strengths.append(
            f"Matches {len(skill.matched_required)} of {skill.total_required} "
            "required skills."
        )
    if skill.total_preferred and skill.matched_preferred:
        strengths.append(
            f"Matches {len(skill.matched_preferred)} of {skill.total_preferred} "
            "preferred skills."
        )
    for missing in skill.missing_required:
        gaps.append(f"Missing required skill: {missing}.")
        unmet_requirements.append(f"Required skill missing: {missing}")
    for missing in skill.missing_preferred:
        gaps.append(f"Missing preferred skill: {missing}.")
    for matched in skill.matched_required:
        matched_requirements.append(f"Required skill matched: {matched}")

    if experience.required_years is not None:
        if experience.candidate_years is None:
            gaps.append(
                "Experience requirement could not be verified from the available "
                "dates (insufficient data — not treated as zero experience)."
            )
        elif experience.score == 100.0:
            strengths.append(
                f"Resume shows ~{experience.candidate_years} years; job requests "
                f"{_fmt_required(experience.required_years)} years — requirement met."
            )
            matched_requirements.append(
                f"Experience requirement met "
                f"({_fmt_required(experience.required_years)}+ years)."
            )
        else:
            gaps.append(
                f"Experience below requirement: ~{experience.candidate_years} years "
                f"vs {_fmt_required(experience.required_years)}+ years."
            )
            unmet_requirements.append(
                f"Experience requirement not met "
                f"({_fmt_required(experience.required_years)}+ years requested)."
            )

    if education.matched is True:
        strengths.append(
            f"Education requirement satisfied ({education.required_level}-level)."
        )
        matched_requirements.append(
            f"Education requirement met ({education.required_level}-level)."
        )
    elif education.matched is False:
        gaps.append(
            f"Education requirement not met (requires {education.required_level}-level "
            "education)."
        )
        unmet_requirements.append(
            f"Education requirement not met "
            f"({education.required_level}-level)."
        )

    for matched in qualification.matched:
        strengths.append(f"Qualification matched: {matched}.")
        matched_requirements.append(f"Qualification matched: {matched}")
    for unmet in qualification.unmet:
        gaps.append(
            f"Requirement listed explicitly in job but not found in resume: "
            f"{unmet}."
        )
        unmet_requirements.append(f"Qualification not found: {unmet}")
    for item in qualification.unknown:
        gaps.append(
            f"Could not verify qualification from structured resume data: {item} "
            "(evidence limitation)."
        )

    return strengths, gaps, matched_requirements, unmet_requirements


def _fmt_required(years: float) -> str:
    return str(int(years)) if years.is_integer() else f"{years:g}"


def build_metadata(
    applied_weights: dict[str, float], reference_date: _dt.date
) -> MatchMetadata:
    return MatchMetadata(
        weights=dict(SCORE_WEIGHTS_DEFAULT),
        applied_weights=applied_weights,
        reference_date=reference_date,
    )
