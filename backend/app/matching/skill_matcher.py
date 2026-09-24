"""Deterministic skill matching between a Resume and a JobDescription."""

from __future__ import annotations

from app.job_parsing.schemas import JobDescription
from app.matching.normalizer import normalize_skill
from app.matching.schemas import SkillMatch
from app.parsing.schemas import Resume

REQUIRED_WEIGHT = 50.0
PREFERRED_WEIGHT = 15.0


def collect_resume_skills(resume: Resume) -> set[str]:
    """Normalised set of every skill the structured resume reports."""
    skills = set(normalize_skill(s) for s in resume.skills.all)
    buckets = (
        resume.skills.technical,
        resume.skills.soft,
        resume.skills.tools,
        resume.skills.languages,
    )
    for bucket in buckets:
        for skill in bucket:
            norm = normalize_skill(skill)
            if norm:
                skills.add(norm)
    for project in resume.projects:
        for tech in project.technologies:
            norm = normalize_skill(tech)
            if norm:
                skills.add(norm)
    for entry in resume.experience:
        for skill in entry.skills_mentioned:
            norm = normalize_skill(skill)
            if norm:
                skills.add(norm)
    skills.discard("")
    return skills


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        norm = normalize_skill(item)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(item)
    return unique


def _match(requirements: list[str], candidate: set[str]) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []
    for req in _dedupe(requirements):
        if normalize_skill(req) in candidate:
            matched.append(req)
        else:
            missing.append(req)
    return matched, missing


def match_skills(resume: Resume, job: JobDescription) -> SkillMatch:
    candidate = collect_resume_skills(resume)

    matched_required, missing_required = _match(job.required_skills, candidate)
    matched_preferred, missing_preferred = _match(job.preferred_skills, candidate)

    total_required = len(matched_required) + len(missing_required)
    total_preferred = len(matched_preferred) + len(missing_preferred)

    required_coverage: float | None
    preferred_coverage: float | None
    if total_required:
        required_coverage = len(matched_required) / total_required
    else:
        required_coverage = None
    if total_preferred:
        preferred_coverage = len(matched_preferred) / total_preferred
    else:
        preferred_coverage = None

    notes: list[str] = []
    if not total_required and not total_preferred:
        notes.append("Job lists no skill requirements — skills are not evaluated.")
    else:
        if not total_required:
            notes.append("Job lists no required skills.")
        if not total_preferred:
            notes.append("Job lists no preferred skills.")

    score: float | None = None
    if total_required or total_preferred:
        if required_coverage is not None and preferred_coverage is not None:
            score = round(
                (
                    REQUIRED_WEIGHT * required_coverage
                    + PREFERRED_WEIGHT * preferred_coverage
                )
                / (REQUIRED_WEIGHT + PREFERRED_WEIGHT)
                * 100,
                1,
            )
        elif required_coverage is not None:
            score = round(required_coverage * 100, 1)
        elif preferred_coverage is not None:
            score = round(preferred_coverage * 100, 1)

    matched = list(matched_required)
    for skill in matched_preferred:
        if skill not in matched:
            matched.append(skill)

    return SkillMatch(
        score=score,
        matched_required=matched_required,
        matched_preferred=matched_preferred,
        missing_required=missing_required,
        missing_preferred=missing_preferred,
        matched=matched,
        total_required=total_required,
        total_preferred=total_preferred,
        required_coverage=required_coverage,
        preferred_coverage=preferred_coverage,
        notes=notes,
    )
