"""Deterministic evidence extraction for the Resume Copilot (Phase 7A).

All evidence is produced by fixed rules over the structured Resume,
JobDescription, and supplied ResumeForge analyses. No LLM, no randomness, no
persistence. The functions here are the only way providers obtain grounding:
a suggestion may never reference evidence that was not produced by this layer
(or, for the local-LLM provider, by the same structured fields rendered into
the prompt).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence, TypeVar

from app.ats_analysis.job_specific.schemas import (
    JobSpecificATSResult,
    TermMatch,
)
from app.ats_analysis.schemas import ATSReadinessResult, Finding
from app.matching.schemas import MatchResult
from app.parsing.schemas import Resume

_T = TypeVar("_T")

#: Structured-path parser for target references such as
#: ``experience[0].achievements[1]`` or ``summary``.
_REF_RE = re.compile(r"^([a-z_]+)(?:\[(\d+)\])?(?:\.([a-z_]+)(?:\[(\d+)\])?)?$")


def resume_skill_names(resume: Resume) -> list[str]:
    """All resume skill names in canonical order, deduplicated by lowercase."""
    skills = resume.skills
    ordered = [
        *skills.technical,
        *skills.tools,
        *skills.languages,
        *skills.soft,
        *skills.all,
    ]
    seen: set[str] = set()
    result: list[str] = []
    for skill in ordered:
        key = skill.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(skill.strip())
    return result


def resolve_target(resume: Resume, target_ref: str) -> str | None:
    """Return the text at a structured path, or None when it cannot be found.

    Supported paths: ``summary``, ``experience[i].achievements[j]``,
    ``experience[i].description``, ``projects[i].description``,
    ``projects[i].name``, ``skills.<bucket>[i]`` where bucket is technical,
    soft, tools, languages, or all. Everything else resolves to None.
    """
    m = _REF_RE.match(target_ref.strip())
    if not m:
        return None
    section, s_idx, attr, e_idx = m.groups()
    if section == "summary":
        return resume.summary or None
    if section == "experience" and s_idx is not None:
        entry = _at(resume.experience, s_idx)
        if entry is None:
            return None
        if attr is None or attr == "description":
            return entry.description or None
        if attr == "achievements" and e_idx is not None:
            return _at(entry.achievements, e_idx) or None
        return None
    if section == "projects" and s_idx is not None:
        project = _at(resume.projects, s_idx)
        if project is None:
            return None
        if attr is None or attr == "description":
            return project.description or None
        if attr == "name":
            return project.name or None
        return None
    if section == "skills" and attr is not None:
        bucket = getattr(resume.skills, attr, None)
        if isinstance(bucket, list) and s_idx is not None:
            return _at(bucket, s_idx)
        return None
    return None


def _at(items: Sequence[_T], index: str) -> _T | None:
    try:
        return items[int(index)]
    except (ValueError, IndexError):
        return None


def finding_by_rule(ats: ATSReadinessResult | None, rule_id: str) -> Finding | None:
    if ats is None:
        return None
    for finding in ats.findings:
        if finding.rule_id == rule_id:
            return finding
    return None


def job_specific_finding_by_rule(
    result: JobSpecificATSResult | None, rule_id: str
) -> Finding | None:
    if result is None:
        return None
    for finding in result.findings:
        if finding.rule_id == rule_id:
            return finding
    return None


def missing_terms(result: JobSpecificATSResult | None) -> list[TermMatch]:
    """Absent JD terms, ordered required < preferred < phrase (stable elsewhere)."""
    if result is None:
        return []
    origin_rank = {"required": 0, "preferred": 1, "phrase": 2}
    absent = [m for m in result.term_matches if m.match_type.value == "absent"]
    return sorted(absent, key=lambda m: origin_rank.get(m.origin.value, 3))


def matched_terms(result: JobSpecificATSResult | None) -> list[TermMatch]:
    """JD terms found in the resume (any non-absent match type)."""
    if result is None:
        return []
    origin_rank = {"required": 0, "preferred": 1, "phrase": 2}
    present = [m for m in result.term_matches if m.match_type.value != "absent"]
    return sorted(present, key=lambda m: origin_rank.get(m.origin.value, 3))


def match_signal(
    result: MatchResult | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (matched_required, matched_preferred, missing_required,
    missing_preferred) from the deterministic match result."""
    if result is None:
        return [], [], [], []
    return (
        list(result.skill_match.matched_required),
        list(result.skill_match.matched_preferred),
        list(result.skill_match.missing_required),
        list(result.skill_match.missing_preferred),
    )


def ordered_findings(
    ats: ATSReadinessResult | None,
    job_ats: JobSpecificATSResult | None,
    *,
    key: Callable[[Finding], tuple[int, float, str]] | None = None,
) -> list[Finding]:
    """Rank findings from both analyses, highest severity and impact first.

    Default ranking: severity (high > medium > low > info), then absolute
    impact, then rule id for stability.
    """
    findings: list[Finding] = []
    if ats is not None:
        findings.extend(ats.findings)
    if job_ats is not None:
        findings.extend(job_ats.findings)
    severity_rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
    sort_key = key or (
        lambda f: (
            severity_rank.get(f.severity.value, 9),
            -f.impact,
            f.rule_id,
        )
    )
    unique: dict[str, Finding] = {}
    for finding in findings:
        unique.setdefault(finding.rule_id, finding)
    return sorted(unique.values(), key=sort_key)
