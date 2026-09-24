"""Deterministic structured resume parsing: normalised text → transient Resume.

Parsing is rule-based and conservative. Missing or ambiguous information stays
Optional/empty rather than being invented. All output is transient — nothing is
persisted and no resume content is logged.

Confidence semantics (heuristic, NOT calibrated probability):
- HIGH   → the section clearly contained structured, parseable entries.
- MEDIUM → the section was present but its structure was only partially clear.
- LOW    → the section was absent, empty, or could not be structured at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.schemas import FileTypeEnum
from app.parsing.heuristics import (
    ends_with_date_range,
    find_date,
    find_date_range,
    find_email,
    find_github,
    find_linkedin,
    find_phone,
    find_url,
    find_website,
    is_bullet,
    remove_dates,
    strip_bullet,
)
from app.parsing.schemas import (
    Certification,
    ConfidenceLevel,
    ContactInfo,
    CustomSection,
    Education,
    Project,
    Resume,
    ResumeMetadata,
    SectionConfidence,
    SkillSet,
    WorkExperience,
)
from app.parsing.sections import Section, header_text, split_sections


class ParsingError(Exception):
    """Raised when a document cannot be parsed as a resume."""

    def __init__(self, message: str, code: str = "empty_document") -> None:
        super().__init__(message)
        self.code = code


_NAME_RE = re.compile(
    r"^[A-Za-z][A-Za-z'\-.]*(?:\s+[A-Za-z][A-Za-z'\-.]*){1,2}$"
)

_DEGREE_RE = re.compile(
    r"(?:^|[\s(])(?:"
    r"b\.?a\.?|b\.?s\.?|b\.?e\.?|b\.?tech\.?|b\.?sc\.?|b\.?b\.?a\.?|b\.?ca|"
    r"m\.?a\.?|m\.?s\.?|m\.?e\.?|m\.?tech\.?|m\.?sc\.?|m\.?ca|m\.?ba\.?|"
    r"m\.?com\.?|b\.?com\.?|"
    r"ph\.?d\.?|bachelor|master|diploma|phd|mba"
    r")(?=[\s,]|$)",
    re.IGNORECASE,
)

_TECH_LABEL_RE = re.compile(
    r"^(?:technologies?|tools?|stack|built with|languages?)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

_SCHOOL_KEYWORD_RE = re.compile(
    r"\b(?:university|college|institute|institution|school|polytechnic|academy)\b",
    re.IGNORECASE,
)

_GRADE_LINE_RE = re.compile(
    r"(?:cgpa|gpa|percentage|percent|grade|score)\b|%", re.IGNORECASE
)

_EDU_SEPARATOR_RE = re.compile(r"\s*(?:[—–|])\s*")

_SKILL_LABEL_RE = re.compile(r"^[\w&/()'+-]+(?:\s+[\w&/()'+-]+){0,4}:\s*")

_WORD_SEPARATOR_RE = re.compile(r"[•|;,]+")

_LOWERCASE_WORD_RE = re.compile(r"^[a-z][a-z0-9.'+-]*$")


@dataclass
class _WorkEntry:
    company: str = ""
    title: str = ""
    start: str | None = None
    end: str | None = None
    achievements: list[str] = field(default_factory=list)
    description: list[str] = field(default_factory=list)


def parse_resume(text: str, *, file_type: FileTypeEnum = FileTypeEnum.TXT) -> Resume:
    """Parse normalised resume text into a transient structured Resume."""
    text = text.strip()
    if not text:
        raise ParsingError("Document is empty.")

    sections = split_sections(text)
    header = header_text(sections)

    contact = _parse_contact(header, text)
    summary = _parse_summary(sections)
    skills = _parse_skills(sections)
    experience = _parse_experience(sections)
    education = _parse_education(sections)
    projects = _parse_projects(sections)
    certifications = _parse_certifications(sections)
    custom_sections = _parse_custom_sections(sections)

    metadata = _build_metadata(
        text=text,
        file_type=file_type,
        sections=sections,
        has_summary=summary is not None,
        skills_count=len(skills.all),
        experience=experience,
        education=education,
        projects=projects,
        certifications=certifications,
        contact=contact,
    )

    return Resume(
        contact=contact,
        summary=summary,
        experience=experience,
        education=education,
        skills=skills,
        projects=projects,
        certifications=certifications,
        custom_sections=custom_sections,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Contact & name
# ---------------------------------------------------------------------------


def _parse_contact(header: str, full_text: str) -> ContactInfo:
    name = _extract_name(header)
    email = find_email(full_text)
    phone = find_phone(full_text)
    linkedin = find_linkedin(full_text)
    github = find_github(full_text)
    # Only the header region is used for the portfolio/website, so a project
    # URL is never misattributed to the candidate.
    website = find_website(header)
    return ContactInfo(
        name=name,
        email=email,
        phone=phone,
        location=None,
        linkedin=linkedin,
        github=github,
        website=website,
    )


def _extract_name(header: str) -> str | None:
    """Conservatively take the first header line that looks like a name."""
    for raw in header.splitlines()[:4]:
        line = raw.strip()
        if not line:
            continue
        if find_email(line) or find_phone(line) or find_url(line):
            continue
        if _NAME_RE.fullmatch(line) and len(line.split()) >= 2:
            return line
    return None


# ---------------------------------------------------------------------------
# Summary & skills
# ---------------------------------------------------------------------------


def _parse_summary(sections: list[Section]) -> str | None:
    for section in sections:
        if section.key != "summary":
            continue
        body = " ".join(section.body.split()).strip()
        return body or None
    return None


_SKILL_BUCKETS = {
    "skills": "technical",
    "professional_skills": "soft",
    "languages": "languages",
}


def _parse_skills(sections: list[Section]) -> SkillSet:
    buckets: dict[str, list[str]] = {"technical": [], "soft": [], "languages": []}
    all_skills: list[str] = []

    for section in sections:
        bucket = _SKILL_BUCKETS.get(section.key)
        if bucket is None:
            continue
        for raw in section.body.splitlines():
            line = strip_bullet(raw)
            if not line:
                continue
            items = _skill_items(line, buckets[bucket], all_skills)
            for item in items:
                if item and item not in buckets[bucket]:
                    buckets[bucket].append(item)
                if item and item not in all_skills:
                    all_skills.append(item)

    technical, soft, languages = (
        buckets["technical"],
        buckets["soft"],
        buckets["languages"],
    )
    if not all_skills:
        return SkillSet()
    return SkillSet(
        technical=technical,
        soft=soft,
        languages=languages,
        all=all_skills,
    )


def _skill_items(line: str, bucket: list[str], all_skills: list[str]) -> list[str]:
    """Turn one skills line into clean skill names.

    Handles leading ``Label:`` prefixes ("Programming: Python"), bullet/pipe/
    comma/semicolon-separated lists, and wrapped lowercase tails ("AI-assisted"
    followed by "development") that complete the previous skill.
    """
    if "://" not in line:
        line = _SKILL_LABEL_RE.sub("", line)
    item_text = line.strip()

    if (
        _LOWERCASE_WORD_RE.fullmatch(item_text)
        and bucket
        and item_text not in bucket
    ):
        base = bucket[-1]
        combined = f"{base} {item_text}".strip()
        if combined != base:
            bucket[-1] = combined
            if all_skills and all_skills[-1] == base:
                all_skills[-1] = combined
        return []

    items = []
    for unit in _WORD_SEPARATOR_RE.split(item_text):
        skill = unit.strip().strip("()")
        if skill:
            items.append(skill)
    return items


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------


def _parse_experience(sections: list[Section]) -> list[WorkExperience]:
    for section in sections:
        if section.key == "experience":
            return _experience_from_body(section.body)
    return []


def _experience_from_body(body: str) -> list[WorkExperience]:
    entries: list[WorkExperience] = []
    for block in _blocks(body):
        header_lines: list[str] = []
        bullets: list[str] = []
        details: list[str] = []

        for line in block:
            if is_bullet(line):
                bullets.append(strip_bullet(line))
                continue
            if len(header_lines) < 2 and _is_headerish(line):
                header_lines.append(line)
            else:
                details.append(line)

        if not header_lines:
            # Bullet/detail-only block continues the previous entry.
            if entries:
                entries[-1].achievements.extend(bullets)
                if details:
                    entries[-1].description = _append_line(
                        entries[-1].description, details
                    )
            continue

        entry = _work_entry_from_header(header_lines)
        entries.append(
            WorkExperience(
                company=entry.company,
                title=entry.title,
                start_date=entry.start,
                end_date=entry.end,
                description=_append_line("", details),
                achievements=bullets,
                skills_mentioned=[],
            )
        )

    return entries


def _is_headerish(line: str) -> bool:
    """A plausible experience header line (short, no sentence punctuation)."""
    if is_bullet(line):
        return False
    tokens = line.split()
    if not tokens:
        return False
    if line.rstrip().endswith("."):
        return False
    # A trailing date range is a strong header signature, e.g.
    # "Event Management — Team Member  |  2025–Present".
    if ends_with_date_range(line):
        return True
    if len(tokens) > 6:
        return False
    if find_date_range(line):
        return True
    if len(tokens) <= 3:
        return True
    # Title-case-ish for slightly longer lines (ignore separators like "|").
    meaningful = [t for t in tokens if len(t) > 1 or not any(c in t for c in "|–—")]
    if len(meaningful) <= 3:
        return True
    return all(t[:1].isupper() or t[:1].isdigit() for t in meaningful)


def _work_entry_from_header(header_lines: list[str]) -> _WorkEntry:
    entry = _WorkEntry()
    start = end = None
    for line in header_lines:
        rng = find_date_range(line)
        if rng:
            start, end = rng
    entry.start, entry.end = start, end

    text = remove_dates(" | ".join(header_lines))
    parts = [part.strip() for part in re.split(r"\s*[|–—]\s*", text) if part.strip()]
    if parts:
        entry.title = parts[0]
    if len(parts) > 1:
        entry.company = parts[1]
    return entry


def _append_line(current: str, lines: list[str]) -> str:
    joined = " ".join(lines).strip()
    if not joined:
        return current
    return " ".join(part for part in (current, joined) if part)


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------


def _parse_education(sections: list[Section]) -> list[Education]:
    for section in sections:
        if section.key == "education":
            return _education_from_body(section.body)
    return []


def _education_from_body(body: str) -> list[Education]:
    entries: list[Education] = []
    for block in _blocks(body):
        entries.extend(_education_entries_from_block(block))
    return entries


@dataclass
class _EducationAcc:
    degree: str | None = None
    institution: str | None = None
    start: str | None = None
    end: str | None = None
    details: list[str] = field(default_factory=list)

    def is_meaningful(self) -> bool:
        return bool(self.degree or self.institution)


def _education_entries_from_block(block: list[str]) -> list[Education]:
    entries: list[Education] = []
    current: _EducationAcc | None = None

    for raw in block:
        line = strip_bullet(raw)
        if not line:
            continue
        if is_bullet(raw):
            if current is None:
                current = _EducationAcc()
            current.details.append(line)
            continue
        if _is_education_header(line):
            if current is not None and current.is_meaningful():
                entries.append(_education_model(current))
            current = _EducationAcc()
            _assign_education_header(current, line)
            continue
        if current is None:
            current = _EducationAcc()
        if _is_expected_date(line):
            current.details.append(line)
            continue
        rng = find_date_range(line)
        if rng:
            current.start, current.end = rng
            continue
        single_date = find_date(line)
        if single_date:
            current.end = single_date
            continue
        clean = remove_dates(line).strip(", ")
        if not clean:
            continue
        if _GRADE_LINE_RE.search(clean):
            current.details.append(clean)
            continue
        if current.institution is None:
            current.institution = clean
        else:
            current.details.append(clean)

    if current is not None and current.is_meaningful():
        entries.append(_education_model(current))
    return entries


def _is_expected_date(line: str) -> bool:
    return line.strip().lower().startswith("expected")


def _is_education_header(line: str) -> bool:
    """A line that begins a new education entry.

    Degree keyword lines always start an entry. School lines only start an
    entry when they carry a separator or a board acronym ("Public School — ICSE"),
    so institution continuations like "Indian Institute of Technology" stay put.
    """
    if _has_degree(line):
        return True
    if _SCHOOL_KEYWORD_RE.search(line):
        return bool(_EDU_SEPARATOR_RE.search(line))
    return False


def _assign_education_header(entry: _EducationAcc, line: str) -> None:
    parts = [p for p in _EDU_SEPARATOR_RE.split(line) if p.strip()]
    if not parts:
        return
    if _has_degree(line):
        entry.degree = parts[0].strip()
        if len(parts) > 1 and _SCHOOL_KEYWORD_RE.search(parts[1]):
            entry.institution = parts[1].strip()
        return
    entry.institution = parts[0].strip()
    if len(parts) > 1 and _SCHOOL_KEYWORD_RE.search(parts[1]):
        entry.institution = parts[1].strip()
    else:
        entry.details.extend(part.strip() for part in parts[1:] if part.strip())


def _education_model(entry: _EducationAcc) -> Education:
    return Education(
        institution=entry.institution or "",
        degree=entry.degree,
        field=None,
        location=None,
        start_date=entry.start,
        end_date=entry.end,
        details=entry.details,
    )


def _has_degree(line: str) -> bool:
    return bool(_DEGREE_RE.search(line))


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def _parse_projects(sections: list[Section]) -> list[Project]:
    for section in sections:
        if section.key == "projects":
            return _projects_from_body(section.body)
    return []


def _projects_from_body(body: str) -> list[Project]:
    entries: list[list[str]] = []
    current: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            if current:
                entries.append(current)
                current = []
            continue
        if current and _is_project_header(line):
            entries.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        entries.append(current)
    return [p for p in (_project_from_block(b) for b in entries) if p is not None]


def _is_project_header(line: str) -> bool:
    """A short, title-case-ish, non-sentence line that starts a new project."""
    if is_bullet(line) or _TECH_LABEL_RE.match(strip_bullet(line)):
        return False
    if line.rstrip().endswith("."):
        return False
    tokens = line.split()
    if len(tokens) > 8:
        return False
    if find_url(line) and len(tokens) == 1:
        return False
    return all(
        tok[:1].isupper() or tok[:1].isdigit()
        for tok in tokens
        if len(tok) > 2 or any(c.isalpha() for c in tok)
    )


def _project_from_block(block: list[str]) -> Project | None:
    name = ""
    description_lines: list[str] = []
    technologies: list[str] = []
    url: str | None = None

    for i, raw in enumerate(block):
        line = raw.strip()
        if not line:
            continue
        if url is None:
            url = find_url(line)
        label_match = _TECH_LABEL_RE.match(strip_bullet(line))
        if label_match:
            technologies = [
                part.strip().rstrip(".,;")
                for part in re.split(r"[,|]", label_match.group(1))
                if part.strip()
            ]
            continue
        if i == 0:
            name = strip_bullet(line)
            name = re.sub(r"https?://[^\s]+", "", name)
            name = name.split("|")[0].strip(" ,")
        elif is_bullet(line):
            description_lines.append(strip_bullet(line))
        else:
            description_lines.append(line)

    if not name:
        return None
    return Project(
        name=name,
        description=" ".join(description_lines),
        technologies=technologies,
        url=url,
    )


# ---------------------------------------------------------------------------
# Certifications
# ---------------------------------------------------------------------------


def _parse_certifications(sections: list[Section]) -> list[Certification]:
    for section in sections:
        if section.key == "certifications":
            return _certs_from_body(section.body)
    return []


def _certs_from_body(body: str) -> list[Certification]:
    certs: list[Certification] = []
    for raw in body.splitlines():
        line = strip_bullet(raw)
        if not line:
            continue
        date = find_date(line)
        url = find_url(line)
        name = remove_dates(line)
        name = re.sub(r"https?://[^\s]+", "", name)
        name = re.sub(r"\(\s*\)", "", name).strip(" ,–—")
        if not name:
            continue
        issuer = ""
        parts = re.split(r"\s*(?:[—–|])\s*", name, maxsplit=1)
        if len(parts) > 1:
            name, issuer = parts[0].strip(" ,;"), parts[1].strip(" ,;")
        certs.append(Certification(name=name, issuer=issuer, date=date, url=url))
    return certs


# ---------------------------------------------------------------------------
# Custom sections
# ---------------------------------------------------------------------------


def _parse_custom_sections(sections: list[Section]) -> list[CustomSection]:
    customs: list[CustomSection] = []
    known = {
        "header",
        "summary",
        "experience",
        "education",
        "skills",
        "professional_skills",
        "languages",
        "projects",
        "certifications",
    }
    for section in sections:
        if section.key in known:
            continue
        content = [strip_bullet(line) for line in section.body.splitlines()]
        content = [line for line in content if line]
        customs.append(CustomSection(heading=section.heading, content=content))
    return customs


# ---------------------------------------------------------------------------
# Metadata / confidence
# ---------------------------------------------------------------------------


def _build_metadata(
    *,
    text: str,
    file_type: FileTypeEnum,
    sections: list[Section],
    has_summary: bool,
    skills_count: int,
    experience: list[WorkExperience],
    education: list[Education],
    projects: list[Project],
    certifications: list[Certification],
    contact: ContactInfo,
) -> ResumeMetadata:
    section_confidence = [
        SectionConfidence(
            section="summary", level=_level_for_bool(has_summary)
        ),
        SectionConfidence(
            section="skills", level=_level_for_bool(skills_count > 0)
        ),
        SectionConfidence(
            section="experience", level=_level_for_count(len(experience))
        ),
        SectionConfidence(
            section="education", level=_level_for_count(len(education))
        ),
        SectionConfidence(
            section="projects", level=_level_for_count(len(projects))
        ),
        SectionConfidence(
            section="certifications", level=_level_for_count(len(certifications))
        ),
    ]

    score = 0
    score += 2 if contact.name else 0
    contact_hits = sum(
        1
        for value in (
            contact.email,
            contact.phone,
            contact.linkedin,
            contact.github,
            contact.website,
        )
        if value
    )
    score += min(contact_hits, 2)
    score += 1 if has_summary else 0
    score += 1 if skills_count > 0 else 0
    score += min(len(experience), 2)
    score += min(len(education), 2)
    score += min(len(projects), 2)
    score += min(len(certifications), 2)

    if score >= 6:
        overall = ConfidenceLevel.HIGH
    elif score >= 3:
        overall = ConfidenceLevel.MEDIUM
    else:
        overall = ConfidenceLevel.LOW

    return ResumeMetadata(
        word_count=len(text.split()),
        file_type=file_type,
        overall_confidence=overall,
        section_confidence=section_confidence,
        section_order=[section.key for section in sections],
    )


def _level_for_count(count: int) -> ConfidenceLevel:
    if count >= 2:
        return ConfidenceLevel.HIGH
    if count == 1:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _level_for_bool(present: bool) -> ConfidenceLevel:
    return ConfidenceLevel.HIGH if present else ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _blocks(body: str) -> list[list[str]]:
    """Split a section body into blocks separated by blank lines."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks
