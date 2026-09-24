"""Deterministic section detection for job descriptions.

Recognises common JD headings (via normalised aliases) and splits the document
into titled sections plus a leading preamble block. Unknown headings that look
like headings (short, title-case, blank-line separated) inside the structured
region are preserved as custom sections rather than discarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.heuristics import find_email, find_phone, is_bullet
from app.parsing.sections import normalize_heading

_ALIASES: dict[str, str] = {
    # Summary / overview
    "summary": "summary",
    "overview": "summary",
    "about the role": "summary",
    "about this role": "summary",
    "about the position": "summary",
    "about the job": "summary",
    "job summary": "summary",
    "position summary": "summary",
    "role overview": "summary",
    "position overview": "summary",
    "role summary": "summary",
    "the role": "summary",
    "the opportunity": "summary",
    "job overview": "summary",
    # Responsibilities
    "responsibilities": "responsibilities",
    "what you'll do": "responsibilities",
    "what you will do": "responsibilities",
    "what you will be doing": "responsibilities",
    "what you'll be doing": "responsibilities",
    "role and responsibilities": "responsibilities",
    "responsibilities and duties": "responsibilities",
    "duties": "responsibilities",
    "duties and responsibilities": "responsibilities",
    "key responsibilities": "responsibilities",
    "primary responsibilities": "responsibilities",
    "principal responsibilities": "responsibilities",
    "job responsibilities": "responsibilities",
    "your responsibilities": "responsibilities",
    "core responsibilities": "responsibilities",
    "day to day": "responsibilities",
    # Requirements / qualifications
    "requirements": "requirements",
    "job requirements": "requirements",
    "position requirements": "requirements",
    "role requirements": "requirements",
    "required qualifications": "requirements",
    "minimum qualifications": "requirements",
    "must have": "requirements",
    "must haves": "requirements",
    "must have skills": "requirements",
    "what we're looking for": "requirements",
    "what we are looking for": "requirements",
    "qualifications": "requirements",
    "role requirements & qualifications": "requirements",
    "hard requirements": "requirements",
    "about you": "requirements",
    # Preferred / nice to have
    "preferred qualifications": "preferred",
    "preferred skills": "preferred",
    "nice to have": "preferred",
    "nice to haves": "preferred",
    "nice-to-have": "preferred",
    "nice-to-haves": "preferred",
    "bonus": "preferred",
    "bonus points": "preferred",
    "bonus skills": "preferred",
    "desirable": "preferred",
    "good to have": "preferred",
    "good to haves": "preferred",
    "preferred experience": "preferred",
    "preferred background": "preferred",
    "beneficial": "preferred",
    # Skills / technologies
    "skills": "skills",
    "skills and requirements": "skills",
    "technical skills": "skills",
    "required skills": "skills",
    "required technologies": "skills",
    "core skills": "skills",
    "technologies": "skills",
    "tech stack": "skills",
    "technology stack": "skills",
    "skills and tools": "skills",
    "tools and technologies": "skills",
    "skills you'll need": "skills",
    "skills you will need": "skills",
    "skills used": "skills",
    "required competencies": "skills",
    # Education
    "education": "education",
    "educational qualifications": "education",
    "academic qualifications": "education",
    "education requirements": "education",
    "educational requirements": "education",
    "formal education": "education",
    # Experience
    "experience": "experience",
    "required experience": "experience",
    "experience requirements": "experience",
    "requisite experience": "experience",
    "years of experience": "experience",
    "professional experience": "experience",
    "relevant experience": "experience",
    "work experience": "experience",
    "experience required": "experience",
    # Certifications / licenses
    "certifications": "certifications",
    "certification requirements": "certifications",
    "certificates": "certifications",
    "licenses and certifications": "certifications",
    "licenses": "certifications",
    "certifications required": "certifications",
    # Benefits
    "benefits": "benefits",
    "perks": "benefits",
    "perks and benefits": "benefits",
    "benefits and perks": "benefits",
    "what we offer": "benefits",
    "compensation and benefits": "benefits",
    "why join us": "benefits",
    "why you'll love working here": "benefits",
    "what we provide": "benefits",
    # Salary / compensation
    "salary": "salary",
    "salary range": "salary",
    "pay": "salary",
    "pay range": "salary",
    "compensation": "salary",
    "compensation package": "salary",
    "remuneration": "salary",
    "ctc": "salary",
    "annual compensation": "salary",
}

_STRUCTURED_SECTIONS = frozenset(
    {
        "summary",
        "responsibilities",
        "requirements",
        "preferred",
        "skills",
        "education",
        "experience",
        "benefits",
        "certifications",
        "salary",
    }
)

_NUMBERED_PREFIX_RE = re.compile(r"^\d+\s*[.)]\s+")
_CUSTOM_HEADING_RE = re.compile(
    r"^[A-Z][A-Za-z &'\-]*(?:\s+[A-Za-z][A-Za-z &'\-]*){0,3}$"
)


@dataclass
class Section:
    key: str
    heading: str
    body: str


def normalize_job_heading(text: str) -> str:
    """Normalise a JD heading: number prefixes, decoration, case."""
    text = _NUMBERED_PREFIX_RE.sub("", text.strip())
    text = text.strip(" :")
    return normalize_heading(text)


def detect_section_key(heading: str) -> str | None:
    """Return the known section key for a heading, or None if it is custom."""
    return _ALIASES.get(normalize_job_heading(heading))


def _is_custom_heading(line: str) -> bool:
    """Whether a free-text line plausibly begins a custom section."""
    if is_bullet(line):
        return False
    if line.rstrip().endswith("."):
        return False
    if find_email(line) or find_phone(line):
        return False
    tokens = line.split()
    if not 1 <= len(tokens) <= 4:
        return False
    return bool(_CUSTOM_HEADING_RE.match(line))


def split_job_sections(text: str) -> list[Section]:
    """Split normalised JD text into sections in document order.

    The leading block before the first recognised heading is the preamble
    (``key="header"``). Custom headings are only detected *after* a known
    heading has been seen, so the preamble (title/company/location lines) is
    never misread as a heading.
    """
    sections: list[Section] = []
    current: Section | None = None
    was_blank = True
    seen_known_heading = False

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if current is not None:
                current.body += "\n"
            was_blank = True
            continue

        key = detect_section_key(stripped)
        if key is not None:
            if current:
                sections.append(current)
            current = Section(key=key, heading=stripped, body="")
            seen_known_heading = True
            was_blank = False
            continue

        if current is None:
            current = Section(key="header", heading="", body=stripped + "\n")
            was_blank = False
            continue

        if seen_known_heading and was_blank and _is_custom_heading(stripped):
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
    """Return the text of the leading preamble block, if any."""
    for section in sections:
        if section.key == "header":
            return section.body.strip()
        return ""
    return ""
