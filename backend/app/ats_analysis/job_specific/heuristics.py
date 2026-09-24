"""Deterministic, token-aware matching heuristics for job-specific coverage.

Split-aware design goals:

- ``Java`` must NOT satisfy ``JavaScript``: matching is exact-token, never
  prefix-based.
- ``C`` must NOT satisfy ``C++`` and ``C++`` must NOT satisfy ``C``: terms
  are matched as complete token sequences that preserve ``+`` / ``#`` / ``.``.
- ``.NET`` must NOT satisfy ``net`` (and vice versa): the leading dot is a
  significant token character.
- ``Node.js`` must NOT satisfy ``Node``: the token sequence is ``["node","js"]``
  and a contiguous window of the same tokens is required.

No LLM, no embeddings, no external services, no persistence. All inputs are
already in memory.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.ats_analysis.job_specific.rules import (
    MAX_PHRASES,
    PHRASE_MAX_TOKENS,
    PHRASE_MIN_TOKENS,
)
from app.ats_analysis.job_specific.schemas import MatchType
from app.matching.normalizer import (
    STOPWORDS,
    alias_target,
    normalize_phrase,
    normalize_skill,
    normalized_base,
)

if TYPE_CHECKING:
    from app.job_parsing.schemas import JobDescription
    from app.parsing.schemas import Resume

# Tokens keep meaningful short forms (c, js, ++, #)
_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Extra filler words ignored ONLY when mining phrases from a job description
# (never when matching). Kept separate from the shared STOPWORDS so existing
# matching behaviour is untouched.
_PHRASE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "of", "to", "in", "on", "by", "at", "as", "or", "but",
        "from", "using", "use", "used", "our",
    }
)

_STRONG_EVIDENCE_LOCATIONS = frozenset({"experience", "projects"})


@dataclass(frozen=True)
class ResumeLocations:
    """Per-section text plus the skill keys each section exposes."""

    text: dict[str, list[str]] = field(default_factory=dict)
    canonical_keys: dict[str, frozenset[str]] = field(default_factory=dict)
    base_keys: dict[str, frozenset[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def presence_tokens(value: str) -> list[str]:
    """Tokenise preserving punctuation; lowercased, no filtering.

    Used for exact-window term/phrase presence checks so that ``C`` and
    ``C++``, ``.NET`` and ``net``, ``Node.js`` and ``Node`` stay distinct.
    """
    return _TOKEN_RE.findall(normalize_phrase(value))


def meaningful_tokens(value: str) -> list[str]:
    """Content tokens only (stopwords removed) for lenient phrase matching.

    Token discipline is preserved: filtering removes only filler words, never
    characters, so ``Java`` never collapses into ``JavaScript`` and ``.NET``
    never collapses into ``net``.
    """
    text = normalize_phrase(value)
    return [
        t
        for t in _TOKEN_RE.findall(text)
        if t not in STOPWORDS and t not in _PHRASE_STOPWORDS
    ]


def _phrase_tokens(value: str) -> list[str]:
    """Meaningful tokens only, for mining candidate phrases from a job."""
    return meaningful_tokens(value)


def _contiguous_present(window: list[str], target: list[str]) -> bool:
    if not target or len(target) > len(window):
        return False
    n, m = len(window), len(target)
    for i in range(n - m + 1):
        if window[i : i + m] == target:
            return True
    return False


def contains_phrase(text_blocks: list[str], phrase: str) -> bool:
    """True if ``phrase`` appears as a contiguous token window in any block.

    Matching is done on meaningful tokens so that filler words on either side
    do not break a genuine phrase, while token exactness (punctuation and
    letter-for-letter) is still required.
    """
    target = meaningful_tokens(phrase)
    if not target:
        return False
    for block in text_blocks:
        if _contiguous_present(meaningful_tokens(block), target):
            return True
    return False


# ---------------------------------------------------------------------------
# Resume collection
# ---------------------------------------------------------------------------


def collect_resume_locations(resume: Resume) -> ResumeLocations:
    """Split the resume into per-section text blocks and skill keys.

    ``canonical_keys`` are alias-expanded; ``base_keys`` are the literal
    normalised forms. Experience ``skills_mentioned`` and project
    ``technologies`` contribute skill keys to their sections, which powers
    the skills-only vs experience/projects evidence distinction.
    """
    text: dict[str, list[str]] = defaultdict(list)
    canonical: dict[str, set[str]] = defaultdict(set)
    base: dict[str, set[str]] = defaultdict(set)

    def add_text(location: str, value: str | None) -> None:
        if value and value.strip():
            text[location].append(value.strip())

    def add_skill(location: str, value: str) -> None:
        raw = value.strip()
        if not raw:
            return
        text[location].append(raw)
        base[location].add(normalized_base(raw))
        canonical[location].add(normalize_skill(raw))

    add_text("summary", resume.summary)

    for skill in (
        resume.skills.technical
        + resume.skills.soft
        + resume.skills.tools
        + resume.skills.languages
        + resume.skills.all
    ):
        add_skill("skills", skill)

    for entry in resume.experience:
        add_text("experience", entry.title)
        add_text("experience", entry.company)
        add_text("experience", entry.description)
        for bullet in entry.achievements:
            add_text("experience", bullet)
        for skill in entry.skills_mentioned:
            add_skill("experience", skill)

    for project in resume.projects:
        add_text("projects", project.name)
        add_text("projects", project.description)
        for tech in project.technologies:
            add_skill("projects", tech)

    for edu in resume.education:
        add_text("education", edu.institution)
        add_text("education", edu.degree)
        add_text("education", edu.field)
        for detail in edu.details:
            add_text("education", detail)

    for cert in resume.certifications:
        add_text("certifications", cert.name)
        add_text("certifications", cert.issuer)

    for section in resume.custom_sections:
        add_text("custom", section.heading)
        for line in section.content:
            add_text("custom", line)

    return ResumeLocations(
        text=dict(text),
        canonical_keys={k: frozenset(v) for k, v in canonical.items()},
        base_keys={k: frozenset(v) for k, v in base.items()},
    )


# ---------------------------------------------------------------------------
# Term resolution
# ---------------------------------------------------------------------------


def _resolve_skill_keys(
    term: str, locs: ResumeLocations
) -> tuple[MatchType, set[str]]:
    term_base = normalized_base(term)
    term_canon = normalize_skill(term)

    exact_locs = {
        k for k, keys in locs.base_keys.items() if term_base in keys
    }
    canon_locs = {
        k for k, keys in locs.canonical_keys.items() if term_canon in keys
    }

    if exact_locs:
        return MatchType.EXACT, exact_locs
    if canon_locs:
        if alias_target(term) is not None:
            return MatchType.ALIAS, canon_locs
        return MatchType.NORMALIZED, canon_locs
    return MatchType.ABSENT, set()


def _evidence_snippet(
    term: str, locs: ResumeLocations, locations: list[str]
) -> str:
    for location in locations:
        for block in locs.text.get(location, []):
            if contains_phrase([block], term):
                return block[:120]
    return ""


_MTYPE_LABEL: dict[MatchType, str] = {
    MatchType.EXACT: (
        "Found verbatim (after case/unicode normalisation) in a "
        "skills or skill-bearing section."
    ),
    MatchType.NORMALIZED: (
        "Found after canonical normalisation of the resume skills "
        "(case, unicode, or resume-side alias expansion)."
    ),
    MatchType.ALIAS: (
        "Job term is a documented abbreviation whose expanded form appears "
        "in the resume skills."
    ),
    MatchType.PHRASE: (
        "Term appears in resume text as a matching contiguous token window."
    ),
    MatchType.ABSENT: "Term was not found in any resume section.",
}


def resolve_term(
    term: str, locs: ResumeLocations
) -> tuple[MatchType, list[str], str]:
    """Resolve one job term against the resume.

    Priority: exact > normalized/alias (skill keys) > phrase (free text) >
    absent. Returned locations are deduplicated and sorted for determinism.
    """
    match_type, key_locations = _resolve_skill_keys(term, locs)

    found_locations: set[str] = set(key_locations)
    if match_type is MatchType.ABSENT:
        for location, blocks in locs.text.items():
            if contains_phrase(blocks, term):
                found_locations.add(location)
        if not found_locations:
            return MatchType.ABSENT, [], ""
        match_type = MatchType.PHRASE
    else:
        for location, blocks in locs.text.items():
            if contains_phrase(blocks, term):
                found_locations.add(location)

    ordered = sorted(found_locations)
    return match_type, ordered, _evidence_snippet(term, locs, ordered)


def has_strong_evidence(term_match_locations: list[str]) -> bool:
    """True when a matched term is backed by experience or project sections."""
    return bool(_STRONG_EVIDENCE_LOCATIONS.intersection(term_match_locations))


def is_skills_only(term_match_locations: list[str]) -> bool:
    """True when the only evidence for a term is the skills section."""
    return term_match_locations == ["skills"]


# ---------------------------------------------------------------------------
# Phrase mining from a job description
# ---------------------------------------------------------------------------


def extract_phrases(job: JobDescription) -> list[str]:
    """Mine stable candidate phrases from the job description.

    Sources, by priority: responsibilities, experience requirements and
    qualifications (the longest contiguous meaningful-token window per item,
    capped at ``PHRASE_MAX_TOKENS``), plus whole certifications and
    nice-to-have entries. Taking one maximal window per item keeps phrases
    meaningful and bounded. Output is deduplicated, kept in first-seen order,
    and capped by ``MAX_PHRASES``.
    """
    candidates: list[str] = []

    def add(phrase: str) -> None:
        joined = " ".join(_phrase_tokens(phrase)).strip()
        if joined and joined not in candidates:
            candidates.append(joined)

    for source in (
        job.responsibilities,
        job.experience_requirements,
        job.qualifications,
    ):
        for item in source:
            tokens = _phrase_tokens(item)
            if len(tokens) < PHRASE_MIN_TOKENS:
                continue
            size = min(PHRASE_MAX_TOKENS, len(tokens))
            add(" ".join(tokens[:size]))

    for source in (job.certifications, job.nice_to_have):
        for item in source:
            add(item)

    return candidates[:MAX_PHRASES]
