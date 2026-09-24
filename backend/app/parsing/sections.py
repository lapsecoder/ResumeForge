"""Deterministic section detection for resumes.

Recognises common headings via normalised aliases and splits the document
into titled sections (plus a leading header block). Unknown headings that
look like headings (short, title-case, blank-line separated) are preserved
as custom sections rather than discarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.heuristics import (
    find_date,
    find_email,
    find_phone,
    find_url,
    is_bullet,
)

_ALIASES: dict[str, str] = {
    "summary": "summary",
    "professional summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "career objective": "summary",
    "experience": "experience",
    "work experience": "experience",
    "employment": "experience",
    "professional experience": "experience",
    "work history": "experience",
    "education": "education",
    "academic background": "education",
    "qualifications": "education",
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
    "competencies": "skills",
    "technologies": "skills",
    "professional skills": "professional_skills",
    "soft skills": "professional_skills",
    "personal skills": "professional_skills",
    "transferable skills": "professional_skills",
    "languages": "languages",
    "interests": "interests",
    "projects": "projects",
    "personal projects": "projects",
    "academic projects": "projects",
    "certifications": "certifications",
    "certificates": "certifications",
    "licenses & certifications": "certifications",
    "licenses and certifications": "certifications",
}

_STRUCTURED_SECTIONS = frozenset(
    {
        "summary",
        "experience",
        "education",
        "skills",
        "professional_skills",
        "languages",
        "interests",
        "projects",
        "certifications",
    }
)

#: Structured sections whose bodies carry their own entry headers (project
#: names, role/employer lines, schools, certificates). A short title-case line
#: inside these is an entry header — never the start of a new custom section.
_ENTRY_SECTIONS = frozenset(
    {"projects", "experience", "education", "certifications"}
)

_PUNCT_STRIP_RE = re.compile(r"^[\s#\-*.\t]+|[\s#\-*.\t]+$")
_WS_COLLAPSE_RE = re.compile(r"\s+")
_CUSTOM_HEADING_RE = re.compile(
    r"^[A-Z][A-Za-z &'\-]*(?:\s+[A-Za-z][A-Za-z &'\-]*){0,3}$"
)
# A single title-case word — e.g. the wrapped tail of a longer heading like
# "Emerging\nTechnologies". Distinct from all-caps/full headings.
_WRAP_TAIL_RE = re.compile(r"^[A-Z][a-z]+$")


@dataclass
class Section:
    key: str
    heading: str
    body: str


def normalize_heading(text: str) -> str:
    """Normalise a heading for alias matching (decoration + whitespace + case)."""
    text = _PUNCT_STRIP_RE.sub("", text)
    text = _WS_COLLAPSE_RE.sub(" ", text).strip()
    return text.lower()


def detect_section_key(heading: str) -> str | None:
    """Return the known section key for a heading, or None if it is custom."""
    return _ALIASES.get(normalize_heading(heading))


def _is_custom_heading(line: str) -> bool:
    """Whether a free-text line plausibly begins a custom section."""
    if is_bullet(line):
        return False
    if line.rstrip().endswith("."):
        return False
    if find_email(line) or find_phone(line) or find_url(line) or find_date(line):
        return False
    tokens = line.split()
    if not 1 <= len(tokens) <= 4:
        return False
    return bool(_CUSTOM_HEADING_RE.match(line.strip()))


def _suppress_alias_split(
    line: str, current: Section | None, was_blank: bool
) -> bool:
    """Whether a short title-case alias line is a wrapped tail, not a heading.

    PDF extraction often wraps long headings; the trailing word alone (e.g.
    "Technologies") matches an alias and would otherwise create a phantom
    section. A single title-case word mid-body is treated as body content
    unless it is preceded by a blank line or appears in the header region.
    """
    if was_blank or current is None:
        return False
    if current.key == "header":
        return False
    if not _WRAP_TAIL_RE.match(line.strip()):
        return False
    return True


def split_sections(text: str) -> list[Section]:
    """Split normalised resume text into sections in document order."""
    sections: list[Section] = []
    current: Section | None = None
    was_blank = True

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if current is not None:
                current.body += "\n"
            was_blank = True
            continue

        key = detect_section_key(stripped)
        if key is not None:
            if not _suppress_alias_split(stripped, current, was_blank):
                if current:
                    sections.append(current)
                current = Section(key=key, heading=stripped, body="")
            elif current is not None:
                current.body += stripped + "\n"
            else:
                current = Section(key="header", heading="", body=stripped + "\n")
            was_blank = False
            continue

        # The very first line of the document always belongs to the header.
        if current is None:
            current = Section(key="header", heading="", body=stripped + "\n")
            was_blank = False
            continue

        # Inside a structured section, short heading-looking lines separated
        # by a blank line begin a new custom section — except in entry-led
        # sections, where such lines are entry headers (e.g. a project name).
        if current.key in _STRUCTURED_SECTIONS:
            if (
                current.key not in _ENTRY_SECTIONS
                and was_blank
                and _is_custom_heading(stripped)
            ):
                sections.append(current)
                current = Section(key="custom", heading=stripped, body="")
            else:
                current.body += stripped + "\n"
            was_blank = False
            continue

        # In the header/custom region a heading-looking line starts a custom
        # section.
        if _is_custom_heading(stripped) and (
            was_blank
            or (current.body and current.key == "custom")
        ):
            sections.append(current)
            current = Section(key="custom", heading=stripped, body="")
            was_blank = False
            continue

        current.body += stripped + "\n"
        was_blank = False

    if current:
        sections.append(current)

    return sections


def header_text(sections: list[Section]) -> str:
    """Return the text of the leading header block, if any."""
    for section in sections:
        if section.key == "header":
            return section.body.strip()
        return ""
    return ""
