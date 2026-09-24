"""Factual validation layer between LLM output and the Copilot contract.

Every suggestion produced by the local model passes through here before it is
allowed out. The validator is deliberately conservative and deterministic:

- numbers/percentages in a rewrite must be traceable to the supplied
  resume/job/target materials;
- skills or technologies must not be claimed by an authoritative rewrite when
  nothing in the resume supports them (JD skills are compared with the
  existing ``app.matching.normalizer`` vocabulary — no second skill list);
- a verified/inferred rewrite must not introduce distinctive wording that has
  no source in the supplied materials (catches fabricated employers, titles,
  dates expressed as words, organisations, degrees, and certifications);
- evidence references that look like structured paths must resolve against the
  resume/job; unresolvable paths are dropped.

Rather than inventing a "perfect truth detector", the validator refuses to
silently accept anything uncertain: any flagged suggestion is downgraded to
``unverified`` with ``requires_user_confirmation=true`` so the user decides.
Nothing here is ever persisted or logged.
"""

from __future__ import annotations

import re
from typing import Any

from app.copilot.config import copilot_settings
from app.copilot.schemas import (
    CopilotEvidence,
    CopilotRequest,
    CopilotSuggestion,
    EvidenceKind,
    VerificationLevel,
)
from app.job_parsing.schemas import JobDescription
from app.matching.normalizer import normalize_skill, significant_tokens
from app.parsing.schemas import Resume

_REWRITE_LEVELS = frozenset(
    {VerificationLevel.VERIFIED, VerificationLevel.INFERRED}
)

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

#: Structured-path-like references, e.g. "experience[0].achievements[1]".
_PATH_RE = re.compile(r"^([a-z_]+)(?:\[(\d+)\])?(?:\.([a-z_]+)(?:\[(\d+)\])?)?$")

_TRUSTED_EVIDENCE_ROOTS = (
    "ats",
    "job_specific",
    "deterministic_match",
    "hybrid_match",
    "copilot",
)


def validate_llm_suggestions(
    request: CopilotRequest, suggestions: list[CopilotSuggestion]
) -> list[CopilotSuggestion]:
    """Validate each LLM suggestion against the supplied evidence.

    Returns the same suggestions (possibly downgraded). Never raises: the goal
    is conservative degradation, not rejection, so fallback semantics are left
    to the provider chain for structural failures.
    """
    resume_corpus = _resume_text(request.resume)
    rewrite_corpus = _rewrite_corpus(request, resume_corpus)
    corpus_numbers = _numbers(rewrite_corpus)
    rewrite_tokens = _term_tokens(rewrite_corpus)
    resume_norm_skills = _normalized_resume_skills(request.resume)
    jd_terms = _jd_skill_terms(request.job_description)

    validated: list[CopilotSuggestion] = []
    for suggestion in suggestions:
        current = suggestion
        current = _check_metrics(
            current, corpus_numbers, _numbers(current.original_text)
        )
        current = _check_skills(
            current, resume_corpus, resume_norm_skills, jd_terms
        )
        current = _check_new_terms(current, rewrite_tokens)
        current = _sanitize_evidence(current, request.resume, request.job_description)
        validated.append(current)
    return validated


def _normalized_resume_skills(resume: Resume) -> set[str]:
    skills = resume.skills
    return {
        normalize_skill(value)
        for value in (
            *skills.technical,
            *skills.tools,
            *skills.languages,
            *skills.soft,
            *skills.all,
        )
        if normalize_skill(value)
    }


def _rewrite_corpus(request: CopilotRequest, resume_corpus: str) -> str:
    """Text a rewrite may legitimately draw facts from.

    Deliberately excludes the job description and analysis: the job lists
    skills the user may not have, so treating it as a source of facts would
    let the model smuggle JD-only terms into the resume.
    """
    parts = [resume_corpus]
    if request.target_text:
        parts.append(request.target_text)
    return "\n".join(parts)


def _resume_text(resume: Resume) -> str:
    """All text from the resume itself (no job description, no analysis)."""
    parts: list[str] = []
    if resume.summary and resume.summary.strip():
        parts.append(resume.summary)
    for entry in resume.experience:
        parts.extend(
            [
                entry.title,
                entry.company,
                entry.location or "",
                entry.start_date or "",
                entry.end_date or "",
                entry.description,
            ]
        )
        parts.extend(entry.achievements)
        parts.extend(entry.skills_mentioned)
    for project in resume.projects:
        parts.append(project.name)
        parts.append(project.description)
        parts.extend(project.technologies)
    for cert in resume.certifications:
        parts.append(cert.name)
        parts.append(cert.issuer)
    for edu in resume.education:
        parts.append(edu.institution)
        parts.extend([edu.degree or "", edu.field or "", edu.location or ""])
    parts.extend(resume.skills.technical)
    parts.extend(resume.skills.tools)
    parts.extend(resume.skills.languages)
    parts.extend(resume.skills.soft)
    parts.extend(resume.skills.all)
    for section in resume.custom_sections:
        parts.append(section.heading)
        parts.extend(section.content)
    return "\n".join(part for part in parts if part)


def _numbers(text: str) -> set[str]:
    return {_canonical_number(raw) for raw in _NUMBER_RE.findall(text)}


def _canonical_number(raw: str) -> str:
    value = raw.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    else:
        value = value.lstrip("0") or "0"
    return value


def _term_tokens(text: str) -> set[str]:
    """Significant tokens with trailing periods stripped.

    ``significant_tokens`` keeps internal periods (``node.js``) but sentence-
    final words keep a trailing ``.`` which breaks token equality.
    """
    return {token.strip(".") for token in significant_tokens(text)} - {""}


def _check_metrics(
    suggestion: CopilotSuggestion,
    corpus_numbers: set[str],
    original_numbers: set[str],
) -> CopilotSuggestion:
    if suggestion.verification not in _REWRITE_LEVELS:
        return suggestion
    suggested = _numbers(suggestion.suggested_text)
    unsupported = suggested - corpus_numbers - original_numbers
    if not unsupported:
        return suggestion
    return _demote(
        suggestion,
        "The suggested text introduces numbers that are not traceable to the "
        "supplied resume or job materials; confirm or remove them before "
        "applying.",
    )


def _jd_skill_terms(job: JobDescription | None) -> list[tuple[str, frozenset[str]]]:
    """(label, surface forms) pairs from the JD skill vocabulary.

    Surface forms include the raw label (lowercased words) and the canonical
    ``normalize_skill`` key so alias spellings (e.g. ``k8s``/``kubernetes``)
    are recognised in both directions.
    """
    if job is None:
        return []
    terms: list[tuple[str, frozenset[str]]] = []
    seen: set[str] = set()
    for bucket in (
        job.required_skills,
        job.preferred_skills,
        job.qualifications,
        job.nice_to_have,
    ):
        for label in bucket:
            norm = normalize_skill(label)
            key = (norm or label.strip().lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            forms = {key}
            label_lower = label.strip().lower()
            if label_lower:
                forms.add(label_lower)
            terms.append((label, frozenset(forms)))
    return terms


def _check_skills(
    suggestion: CopilotSuggestion,
    resume_corpus: str,
    resume_norm_skills: set[str],
    jd_terms: list[tuple[str, frozenset[str]]],
) -> CopilotSuggestion:
    if suggestion.verification not in _REWRITE_LEVELS or not jd_terms:
        return suggestion
    text = suggestion.suggested_text.lower()
    corpus_lower = resume_corpus.lower()
    for label, forms in jd_terms:
        mentioned = any(_word_present(text, form) for form in forms)
        if not mentioned:
            continue
        supported = any(
            form in resume_norm_skills or _word_present(corpus_lower, form)
            for form in forms
        )
        if not supported:
            return _demote(
                suggestion,
                f"The suggested text asserts the job skill '{label}', which is "
                "not supported by the supplied resume; it must be confirmed "
                "before applying.",
            )
    return suggestion


def _word_present(haystack: str, needle: str) -> bool:
    """Whole-token containment so ``java`` does not match ``javascript``."""
    if not needle:
        return False
    if not re.search(r"[a-z0-9]", needle):
        return needle in haystack
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _check_new_terms(
    suggestion: CopilotSuggestion, corpus_tokens: set[str]
) -> CopilotSuggestion:
    if suggestion.verification not in _REWRITE_LEVELS:
        return suggestion
    suggested_tokens = _term_tokens(suggestion.suggested_text)
    extra = suggested_tokens - corpus_tokens
    if not extra:
        return suggestion
    return _demote(
        suggestion,
        "The suggested text adds wording not traceable to the supplied "
        "resume or job materials (for example a new employer, title, "
        "organisation, or certification). Confirm it is true before applying.",
    )


def _sanitize_evidence(
    suggestion: CopilotSuggestion,
    resume: Resume,
    job: JobDescription | None,
) -> CopilotSuggestion:
    kept = [
        evidence
        for evidence in suggestion.evidence
        if _evidence_path_ok(evidence.reference, resume, job)
    ]
    if not kept:
        kept = [
            CopilotEvidence(
                kind=EvidenceKind.SUGGESTION,
                source="copilot.validation",
                statement=(
                    "No structural evidence path could be verified for this "
                    "suggestion; treat it as guidance pending user review."
                ),
            )
        ]
    return suggestion.model_copy(update={"evidence": kept})


def _evidence_path_ok(
    reference: str, resume: Resume, job: JobDescription | None
) -> bool:
    ref = reference.strip()
    if not ref:
        return True
    match = _PATH_RE.match(ref)
    if match is None:
        return True
    root, root_index, child, child_index = match.groups()
    if root in _TRUSTED_EVIDENCE_ROOTS:
        return True
    if root in ("job", "job_description"):
        if job is None:
            return False
        return child in {
            "title",
            "summary",
            "required_skills",
            "preferred_skills",
            "qualifications",
            "experience_requirements",
            "education_requirements",
            "certifications",
            "nice_to_have",
        }
    if root == "resume":
        return _resume_namespace_ok(resume, child)
    return _resume_path_resolvable(resume, root, root_index, child, child_index)


def _resume_namespace_ok(resume: Resume, child: str | None) -> bool:
    """Loose check for ``resume.<namespace>`` style evidence references."""
    if child is None:
        return True
    if child == "summary":
        return bool((resume.summary or "").strip())
    if child == "skills":
        skills = resume.skills
        return bool(skills.all or skills.technical or skills.tools)
    if child == "achievements" or child == "bullets":
        return any(entry.achievements for entry in resume.experience)
    # Unknown namespaces cannot be disproved from the structured data; keep
    # them rather than dropping evidence that may be valid.
    return True


def _resume_path_resolvable(
    resume: Resume,
    root: str,
    root_index: str | None,
    child: str | None,
    child_index: str | None,
) -> bool:
    if root == "skills":
        skills = resume.skills
        return bool(skills.all or skills.technical or skills.tools)
    if root == "summary":
        return bool(resume.summary and resume.summary.strip())
    index = int(root_index) if root_index is not None else None
    collection: list[Any]
    if root == "experience":
        collection = resume.experience
        allowed_children: set[str] = {
            "description",
            "title",
            "company",
            "achievements",
            "bullets",
        }
    elif root == "projects":
        collection = resume.projects
        allowed_children = {"description", "name", "technologies"}
    elif root == "education":
        collection = resume.education
        allowed_children = {"institution", "degree", "field"}
    elif root == "certifications":
        collection = resume.certifications
        allowed_children = {"name", "issuer"}
    else:
        return False
    # Only an explicit, out-of-range index is treated as invalid; an
    # unindexed reference cannot be disproved, so it is kept.
    if index is None:
        return True
    if index >= len(collection):
        return False
    if child is None:
        return True
    if child not in allowed_children:
        return False
    if child in ("achievements", "bullets"):
        achievement_index = (
            int(child_index) if child_index is not None else None
        )
        achievements = getattr(collection[index], "achievements", [])
        if achievement_index is None or achievement_index >= len(achievements):
            return False
    return True


def _demote(suggestion: CopilotSuggestion, statement: str) -> CopilotSuggestion:
    """Downgrade a rewrite to an unverified, confirmation-required suggestion."""
    evidence = [
        *suggestion.evidence,
        CopilotEvidence(
            kind=EvidenceKind.UNVERIFIED,
            source="copilot.validation",
            statement=statement,
        ),
    ][: copilot_settings.max_evidence_per_suggestion]
    return suggestion.model_copy(
        update={
            "verification": VerificationLevel.UNVERIFIED,
            "requires_user_confirmation": True,
            "evidence": evidence,
        }
    )


# ---------------------------------------------------------------------------
# Read-only helpers reused by the Phase 7C edit validator.
#
# These expose the exact deterministic primitives the 7B suggestion validator
# already uses so the edit layer cannot silently diverge from it. They add no
# behaviour of their own and nothing here is persisted or logged.
# ---------------------------------------------------------------------------


def resume_fact_corpus(resume: Resume) -> str:
    """Return all resume text a rewrite may legitimately draw facts from."""
    return _resume_text(resume)


def canonical_numbers(text: str) -> set[str]:
    """Return canonicalised numeric tokens (commas stripped, zeros trimmed)."""
    return _numbers(text)


def significant_term_tokens(text: str) -> set[str]:
    """Return significant stopword-filtered tokens (trailing periods stripped)."""
    return _term_tokens(text)


def unsupported_job_skills(
    text: str, resume: Resume, job: JobDescription | None
) -> list[str]:
    """Return JD skill labels asserted in ``text`` that the resume cannot support.

    Mirrors ``_check_skills`` exactly (same whole-token and normalised-skill
    comparison). Returns labels only — never a possession claim — so callers can
    report the issue without echoing resume content.
    """
    jd_terms = _jd_skill_terms(job)
    if not jd_terms:
        return []
    lowered = text.lower()
    corpus_lower = _resume_text(resume).lower()
    resume_norm_skills = _normalized_resume_skills(resume)
    unsupported: list[str] = []
    for label, forms in jd_terms:
        if not any(_word_present(lowered, form) for form in forms):
            continue
        supported = any(
            form in resume_norm_skills or _word_present(corpus_lower, form)
            for form in forms
        )
        if not supported:
            unsupported.append(label)
    return unsupported
