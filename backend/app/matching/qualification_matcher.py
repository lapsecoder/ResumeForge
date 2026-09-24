"""Deterministic qualification matching.

Three outcomes are kept distinct on purpose:

- matched  -> the requirement is deterministically found in the structured resume.
- unmet    -> an explicit, verifiable requirement (usually a named
              certification) that the resume does not list.
- unknown  -> present in the JD but not deterministically verifiable from the
              structured resume. Absence of evidence is NOT treated as absence
              of the qualification, so unknown items never lower the score.

Generic prose qualifications are only counted as matched when every significant
token of the phrase appears in the resume's evidence corpus; otherwise they are
``unknown`` (evidence limitation). No compliance is inferred from absence.
"""

from __future__ import annotations

import re

from app.job_parsing.schemas import JobDescription
from app.matching.normalizer import normalize_skill, significant_tokens
from app.matching.schemas import QualificationMatch
from app.parsing.schemas import Resume

_CERT_KEYWORD_RE = re.compile(
    r"\b(?:certif(?:ied|ication|ications?)?|aws\s+certified|azure\s+certified|"
    r"google\s+cloud\s+certified|pmp|cissp|security\+|ccna|ccnp|ccie|ceh|"
    r"prince2|cka|cks)\b",
    re.IGNORECASE,
)

_LANGUAGE_HINT_RE = re.compile(
    r"\b(?:fluency|fluent|language|languages|speaks?|written\s+and\s+verbal)\b",
    re.IGNORECASE,
)

_LANGUAGES = (
    "english",
    "spanish",
    "hindi",
    "french",
    "german",
    "mandarin",
    "cantonese",
    "chinese",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "arabic",
    "tamil",
    "telugu",
    "bengali",
    "marathi",
    "kannada",
    "malayalam",
    "gujarati",
)


def _is_cert_requirement(text: str) -> bool:
    return bool(_CERT_KEYWORD_RE.search(text))


def _language_in(text: str) -> str | None:
    lowered = text.lower()
    for language in _LANGUAGES:
        if re.search(rf"\b{re.escape(language)}\b", lowered):
            return language
    return None


def _is_language_requirement(text: str) -> bool:
    return _LANGUAGE_HINT_RE.search(text) is not None and _language_in(text) is not None


def _cert_tokens(req: str) -> set[str]:
    return set(significant_tokens(req))


def _cert_matches(requirement: str, resume_cert: str) -> bool:
    req_norm = normalize_skill(requirement)
    cert_norm = normalize_skill(resume_cert)
    if not req_norm or not cert_norm:
        return False
    if req_norm == cert_norm:
        return True
    req_tokens = _cert_tokens(requirement)
    cert_tokens = _cert_tokens(resume_cert)
    if not req_tokens or not cert_tokens:
        return False
    return req_tokens.issubset(cert_tokens) or cert_tokens.issubset(req_tokens)


def _resume_evidence_corpus(resume: Resume) -> tuple[list[str], set[str], set[str]]:
    """Return (normalised cert names, normalised languages, significant tokens)."""
    certs = [c.name for c in resume.certifications if c.name.strip()]
    languages = {
        normalize_skill(lang)
        for lang in resume.skills.languages
        if normalize_skill(lang)
    }
    tokens: set[str] = set()

    def add(*texts: str | None) -> None:
        for text in texts:
            if text:
                tokens.update(significant_tokens(text))

    add(resume.summary)
    for entry in resume.experience:
        add(entry.title, entry.company, entry.description)
        for item in entry.achievements:
            add(item)
        for skill in entry.skills_mentioned:
            add(skill)
    for project in resume.projects:
        add(project.name, project.description)
        for tech in project.technologies:
            add(tech)
    for edu in resume.education:
        add(edu.degree, edu.field, edu.institution)
        for detail in edu.details:
            add(detail)
    for section in resume.custom_sections:
        for line in section.content:
            add(line)
    buckets = (
        resume.skills.all,
        resume.skills.technical,
        resume.skills.tools,
        resume.skills.soft,
    )
    for bucket in buckets:
        for skill in bucket:
            add(skill)
    return certs, languages, tokens


def match_qualifications(resume: Resume, job: JobDescription) -> QualificationMatch:
    required: list[str] = []
    for item in [*job.certifications, *job.qualifications]:
        text = item.strip()
        if text and text not in required:
            required.append(text)

    if not required:
        return QualificationMatch(
            score=None,
            total=0,
            assessment=(
                "Job lists no explicit qualification requirements — not evaluated."
            ),
        )

    certs, languages, corpus_tokens = _resume_evidence_corpus(resume)
    matched: list[str] = []
    unmet: list[str] = []
    unknown: list[str] = []

    for item in required:
        if _is_cert_requirement(item):
            if any(_cert_matches(item, cert) for cert in certs):
                matched.append(item)
            else:
                unmet.append(item)
            continue

        language = _language_in(item)
        if _is_language_requirement(item) and language is not None:
            if normalize_skill(language) in languages:
                matched.append(item)
            else:
                unknown.append(item)
            continue

        if _matched_by_tokens(item, corpus_tokens):
            matched.append(item)
        else:
            unknown.append(item)

    total = len(required)
    verifiable = len(matched) + len(unmet)
    score = round(len(matched) / verifiable * 100, 1) if verifiable else None

    assessment = f"Verified {len(matched)} of {len(required)} qualifications."
    if unmet:
        assessment += f" {len(unmet)} explicitly required and not listed."
    if unknown:
        assessment += (
            f" {len(unknown)} could not be verified from structured resume data "
            "(evidence limitation — absence of evidence treated as unknown, "
            "not absence)."
        )

    return QualificationMatch(
        score=score,
        matched=matched,
        unmet=unmet,
        unknown=unknown,
        total=total,
        assessment=assessment,
    )


def _matched_by_tokens(item: str, corpus_tokens: set[str]) -> bool:
    tokens = significant_tokens(item)
    if not tokens:
        return False
    return corpus_tokens.issuperset(tokens)
