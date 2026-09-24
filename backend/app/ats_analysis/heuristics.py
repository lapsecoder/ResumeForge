"""Deterministic, rule-based text helpers for ATS readiness analysis.

No LLM, no embeddings, no external services, no persistence. The helpers
operate on transient structured data that is already in memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.parsing.schemas import Resume

# ---------------------------------------------------------------------------
# Action-verb set
# ---------------------------------------------------------------------------

ACTION_VERBS: frozenset[str] = frozenset(
    {
        # bases (canonical forms)
        "analyze",
        "architect",
        "automate",
        "build",
        "collaborate",
        "configure",
        "coordinate",
        "create",
        "deliver",
        "deploy",
        "design",
        "develop",
        "engineer",
        "establish",
        "evaluate",
        "execute",
        "fix",
        "govern",
        "identify",
        "implement",
        "improve",
        "increase",
        "integrate",
        "introduce",
        "lead",
        "launch",
        "manage",
        "mentor",
        "migrate",
        "monitor",
        "negotiate",
        "optimize",
        "overhaul",
        "plan",
        "reduce",
        "refactor",
        "reorganize",
        "research",
        "resolve",
        "revamp",
        "scale",
        "ship",
        "spearhead",
        "standardize",
        "streamline",
        "strengthen",
        "supervise",
        "sustain",
        "test",
        "train",
        "translate",
        "troubleshoot",
        "write",
        # common inflected forms
        "automated",
        "built",
        "collaborated",
        "configured",
        "coordinated",
        "created",
        "delivered",
        "deployed",
        "designed",
        "developed",
        "engineered",
        "established",
        "evaluated",
        "executed",
        "fixed",
        "governed",
        "identified",
        "implemented",
        "improved",
        "increased",
        "integrated",
        "introduced",
        "launched",
        "led",
        "managed",
        "mentored",
        "migrated",
        "monitored",
        "negotiated",
        "optimized",
        "overhauled",
        "planned",
        "reduced",
        "refactored",
        "reorganized",
        "researched",
        "resolved",
        "revamped",
        "scaled",
        "shipped",
        "spearheaded",
        "standardized",
        "streamlined",
        "strengthened",
        "supervised",
        "sustained",
        "tested",
        "trained",
        "translated",
        "troubleshooted",
        "wrote",
        # progressive / -ing (common in resume bullets)
        "analyzing",
        "architecting",
        "automating",
        "building",
        "collaborating",
        "configuring",
        "coordinating",
        "creating",
        "delivering",
        "deploying",
        "designing",
        "developing",
        "engineering",
        "establishing",
        "evaluating",
        "executing",
        "fixing",
        "governing",
        "identifying",
        "implementing",
        "improving",
        "increasing",
        "integrating",
        "introducing",
        "launching",
        "leading",
        "managing",
        "mentoring",
        "migrating",
        "monitoring",
        "negotiating",
        "optimizing",
        "overhauling",
        "planning",
        "reducing",
        "refactoring",
        "reorganizing",
        "researching",
        "resolving",
        "revamping",
        "scaling",
        "shipping",
        "spearheading",
        "standardizing",
        "streamlining",
        "strengthening",
        "supervising",
        "sustaining",
        "testing",
        "training",
        "translating",
        "troubleshooting",
        "writing",
        # 3rd person singular
        "analyzes",
        "builds",
        "creates",
        "delivers",
        "deploys",
        "designs",
        "develops",
        "engineers",
        "establishes",
        "evaluates",
        "executes",
        "fixes",
        "identifies",
        "implements",
        "improves",
        "increases",
        "integrates",
        "introduces",
        "launches",
        "leads",
        "manages",
        "mentors",
        "migrates",
        "monitors",
        "negotiates",
        "optimizes",
        "overhauls",
        "plans",
        "reduces",
        "refactors",
        "reorganizes",
        "researches",
        "resolves",
        "revamps",
        "scales",
        "ships",
        "spearheads",
        "standardizes",
        "streamlines",
        "strengthens",
        "supervises",
        "sustains",
        "tests",
        "trains",
        "translates",
        "troubleshoots",
        "writes",
    },
)

_SUFFIXES = ("ing", "ed", "es", "s")


def _verb_stem(word: str) -> str:
    """Return a conservative morphological stem for action-verb lookup."""
    w = word.lower()
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def is_action_start(line: str) -> bool:
    """Return True if the line's first meaningful token is an action verb."""
    tokens = line.split()
    if not tokens:
        return False
    first = tokens[0]
    if not first or not first[0].isalpha():
        return False
    w = first.lower().rstrip(".")
    if w in ACTION_VERBS:
        return True
    stem = _verb_stem(w)
    return stem != w and stem in ACTION_VERBS


# ---------------------------------------------------------------------------
# Quantitative evidence detection
# ---------------------------------------------------------------------------

_QUANT_RE = re.compile(
    r"\b\d[\d,]*\s*%"
    r"|(?:[₹$€£]\s?\d[\d,]*(?:\.\d+)?\b)"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:x|×)\b"
    r"|\b\d[\d,]*\s*"
    r"(?:hrs?|hours?|days?|weeks?|months?|years?|"
    r"users?|people|customers?|clients?|"
    r"requests?|queries?|records?|files?|lines?|rows?|"
    r"items?|instances?|nodes?|services?|endpoints?|APIs?|"
    r"tests?|bugs?|issues?|PRs?|commits?|"
    r"results?|sales?|revenue|costs?|errors?|"
    r"students?|developers?|engineers?|members?|teams?|"
    r"downloads?|visits?|sessions?|installs?|devices?|"
    r"orders?|payments?|tickets?|campaigns?|attempts?|"
    r"searches?|projects?|reports?|posts?|articles?|"
    r"ms|secs?|mins?|GB|MB|KB|TB|GHz)\b",
    re.IGNORECASE,
)


def has_quantitative(line: str) -> bool:
    """Detect any quantitative token (percentage, currency, unit, count)."""
    return bool(_QUANT_RE.search(line))


# ---------------------------------------------------------------------------
# Vague / non-informative markers
# ---------------------------------------------------------------------------

_VAGUE_MARKERS = re.compile(
    r"\b(?:responsible for|handled|worked on|helped with|helped|"
    r"assisted (?:in|with)|involved in|participated in|contributed to|"
    r"various tasks|other duties|additional duties|etc\.?|"
    r"some)\b",
    re.IGNORECASE,
)


def is_vague(line: str) -> bool:
    """Detect lines that contain vague or non-informative markers.

    Length alone is handled separately as "too short"; this helper only
    matches explicit vague phrasing so a single line is not double-counted.
    """
    return bool(_VAGUE_MARKERS.search(line))


def first_person_words(line: str) -> bool:
    """Detect first-person pronouns in a line."""
    return bool(re.search(r"\b(?:i|me|my|mine|we|us|our|ours)\b", line, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Sentence splitting (conservative)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Conservatively split prose into sentences."""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------
# Line collection from a Resume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AchievementLine:
    source: str
    text: str


def collect_lines(resume: Resume) -> list[AchievementLine]:
    """Collect all achievement-like lines from the resume.

    Sources: experience bullets, experience description sentences,
    project description sentences. These represent substantive content
    that can be evaluated for action verbs, quantitative evidence, etc.
    """
    lines: list[AchievementLine] = []
    for entry in resume.experience:
        for bullet in entry.achievements:
            stripped = bullet.strip()
            if stripped:
                lines.append(AchievementLine(source="experience", text=stripped))
        if entry.description.strip():
            for sent in split_sentences(entry.description):
                if sent.strip():
                    lines.append(AchievementLine(source="experience", text=sent))
    for project in resume.projects:
        if project.description.strip():
            for sent in split_sentences(project.description):
                if sent.strip():
                    lines.append(AchievementLine(source="project", text=sent))
    return lines


# ---------------------------------------------------------------------------
# Date value parsing
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@dataclass(frozen=True)
class DateVal:
    year: int | None
    month: int | None
    is_current: bool


def parse_date_value(s: str | None) -> DateVal | None:
    """Parse a date string into a structured ``DateVal``.

    Supports ``YYYY``, ``MM/YYYY``, ``Mon YYYY``, ``Month YYYY``,
    ``Present``/``current``. Returns ``None`` if the string cannot be parsed.
    """
    if not s:
        return None
    t = s.strip()
    low = t.lower()
    if low in ("present", "current", "now"):
        return DateVal(year=None, month=None, is_current=True)
    m = re.fullmatch(r"(\d{4})", t)
    if m:
        return DateVal(year=int(m.group(1)), month=None, is_current=False)
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", t)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return DateVal(year=int(m.group(2)), month=month, is_current=False)
        return None
    m = re.fullmatch(r"([a-zA-Z]+)\s+(\d{4})", t)
    if m:
        mn = _MONTHS.get(m.group(1).lower())
        if mn is not None:
            return DateVal(year=int(m.group(2)), month=mn, is_current=False)
    return None


def _date_val_sortable(dv: DateVal) -> tuple[int, int] | None:
    """Return (year, month|6) for comparison, or None if unparsed or current."""
    if dv.is_current or dv.year is None:
        return None
    return (dv.year, dv.month if dv.month is not None else 6)


# ---------------------------------------------------------------------------
# Word count helper
# ---------------------------------------------------------------------------


def computed_word_count(resume: Resume) -> int:
    """Estimate word count from all structured fields (fallback)."""
    parts: list[str] = []
    c = resume.contact
    for v in (c.name, c.email, c.phone, c.location, c.linkedin, c.github, c.website):
        if v:
            parts.append(v)
    if resume.summary:
        parts.append(resume.summary)
    for e in resume.experience:
        if e.title:
            parts.append(e.title)
        if e.company:
            parts.append(e.company)
        if e.description:
            parts.append(e.description)
        parts.extend(e.achievements)
    for ed in resume.education:
        if ed.institution:
            parts.append(ed.institution)
        if ed.degree:
            parts.append(ed.degree)
        if ed.field:
            parts.append(ed.field)
        parts.extend(ed.details)
    for p in resume.projects:
        if p.name:
            parts.append(p.name)
        if p.description:
            parts.append(p.description)
    for cert in resume.certifications:
        if cert.name:
            parts.append(cert.name)
    for cs in resume.custom_sections:
        parts.extend(cs.content)
    parts.extend(resume.skills.all)
    return len(" ".join(parts).split())
