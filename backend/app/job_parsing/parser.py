"""Deterministic structured job-description parsing.

``parse_job_description(text)`` converts normalised JD text into a transient
``JobDescription``. Parsing is rule-based and conservative: missing or
ambiguous information stays Optional/empty rather than being invented. Nothing
is persisted and no JD content is logged.

Confidence semantics (heuristic, NOT a calibrated probability):
- HIGH   → the document clearly contained several structured sections.
- MEDIUM → sections were present but their structure was only partially clear.
- LOW    → little or nothing could be structured (e.g. bare/unstructured text).
"""

from __future__ import annotations

import re

from app.ingestion.schemas import FileTypeEnum
from app.job_parsing.heuristics import (
    find_company,
    find_employment_type,
    find_labeled_value,
    find_location,
    find_remote_type,
    find_salary_statements,
    has_degree,
    is_cert_like,
    is_experience_statement,
    is_salary_line,
    is_skill_item,
    salary_from_line,
    section_items,
    split_skill_list,
)
from app.job_parsing.schemas import (
    JobDescription,
    JobMetadata,
    Salary,
)
from app.job_parsing.sections import (
    Section,
    detect_section_key,
    header_text,
    split_job_sections,
)
from app.parsing.heuristics import find_email, find_phone, find_url, is_bullet
from app.parsing.schemas import ConfidenceLevel, CustomSection, SectionConfidence


class ParsingError(Exception):
    """Raised when a document cannot be parsed as a job description."""

    def __init__(self, message: str, code: str = "empty_document") -> None:
        super().__init__(message)
        self.code = code


_TITLE_LABEL_RE = re.compile(
    r"(?:job\s+)?(?:role|position|title|designation)\s*:", re.IGNORECASE
)
_TITLE_CANDIDATE_RE = re.compile(
    r"^[A-Z][A-Za-z0-9'\-&./()]+(?:\s+[A-Za-z0-9'\-&./()]+){1,7}$"
)


def parse_job_description(
    text: str, *, file_type: FileTypeEnum = FileTypeEnum.TXT
) -> JobDescription:
    """Parse normalised job-description text into a transient JobDescription."""
    text = text.strip()
    if not text:
        raise ParsingError("Job description is empty.")

    sections = split_job_sections(text)
    header = header_text(sections)

    title = _extract_title(sections, header, text)
    company = find_company(header, text)
    location = find_location(header, text)
    employment_type = find_employment_type(header, text)
    remote_type = find_remote_type(header, text)
    summary = _parse_summary(sections, title=title, company=company)

    responsibilities = _parse_responsibilities(sections)
    (
        required_skills,
        preferred_skills,
        qualifications,
        nice_to_have,
        experience,
        education,
        certs,
    ) = _parse_requirement_sections(sections)
    experience = _merge_unique(experience, _parse_experience_section(sections))
    experience = _merge_unique(experience, _preamble_experience(header))
    education = _merge_unique(education, _parse_education_section(sections))
    certifications = _parse_certifications_section(sections)
    certifications = _merge_unique(certifications, certs)
    benefits = _parse_benefits(sections)
    salaries = _parse_salaries(sections, header, text)
    custom_sections = _parse_custom_sections(sections)

    metadata = _build_metadata(
        text=text,
        file_type=file_type,
        title=title,
        company=company,
        location=location,
        summary=summary,
        responsibilities=responsibilities,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        qualifications=qualifications,
        experience=experience,
        education=education,
        certifications=certifications,
        benefits=benefits,
        salary=salaries,
    )

    return JobDescription(
        title=title,
        company=company,
        location=location,
        employment_type=employment_type,
        remote_type=remote_type,
        summary=summary,
        responsibilities=responsibilities,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        qualifications=qualifications,
        experience_requirements=experience,
        education_requirements=education,
        certifications=certifications,
        nice_to_have=nice_to_have,
        benefits=benefits,
        salary=salaries,
        custom_sections=custom_sections,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Title / company / location / employment / remote / summary
# ---------------------------------------------------------------------------


def _extract_title(sections: list[Section], header: str, text: str) -> str | None:
    value = find_labeled_value(header, _TITLE_LABEL_RE)
    if value:
        return value
    value = find_labeled_value(text, _TITLE_LABEL_RE)
    if value:
        return value
    for line in header.splitlines():
        if _is_title_candidate(line):
            return line
    return None


def _is_title_candidate(line: str) -> bool:
    """A strong short heading near the beginning of the document."""
    stripped = line.strip()
    if not stripped or ":" in stripped:
        return False
    if is_bullet(stripped):
        return False
    if stripped.rstrip().endswith("."):
        return False
    if find_email(stripped) or find_phone(stripped) or find_url(stripped):
        return False
    if detect_section_key(stripped):
        return False
    if not _TITLE_CANDIDATE_RE.fullmatch(stripped):
        return False
    if find_employment_type(stripped, ""):
        return False
    if find_remote_type(stripped, ""):
        return False
    if is_salary_line(stripped):
        return False
    if is_experience_statement(stripped):
        return False
    return True


def _parse_summary(
    sections: list[Section], *, title: str | None, company: str | None
) -> str | None:
    for section in sections:
        if section.key == "summary":
            body = " ".join(section.body.split()).strip()
            if body:
                return body
    if title:
        return f"{title} at {company}" if company else title
    return None


# ---------------------------------------------------------------------------
# Section content
# ---------------------------------------------------------------------------


def _section_by_key(sections: list[Section], key: str) -> str | None:
    for section in sections:
        if section.key == key:
            return section.body
    return None


def _parse_responsibilities(sections: list[Section]) -> list[str]:
    body = _section_by_key(sections, "responsibilities")
    if body is None:
        return []
    return section_items(body)


def _parse_requirement_sections(
    sections: list[Section],
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Classify requirements/preferred items into their target fields."""
    required: list[str] = []
    preferred: list[str] = []
    quals: list[str] = []
    nice_to_have: list[str] = []
    experience: list[str] = []
    education: list[str] = []
    certifications: list[str] = []

    for key, body in (
        (s.key, s.body) for s in sections if s.key in ("requirements", "preferred")
    ):
        for item in section_items(body):
            if key == "preferred":
                if is_cert_like(item):
                    certifications = _append_unique(certifications, item)
                elif is_skill_item(item):
                    for skill in _split_skill_item(item):
                        preferred = _append_unique(preferred, skill)
                else:
                    nice_to_have = _append_unique(nice_to_have, item)
                continue
            if has_degree(item):
                education = _append_unique(education, item)
            elif is_experience_statement(item):
                experience = _append_unique(experience, item)
            elif is_cert_like(item):
                certifications = _append_unique(certifications, item)
            elif is_skill_item(item):
                for skill in _split_skill_item(item):
                    required = _append_unique(required, skill)
            else:
                quals = _append_unique(quals, item)

    skills_body = _section_by_key(sections, "skills")
    if skills_body:
        for skill in split_skill_list(skills_body):
            required = _append_unique(required, skill)

    return (
        required,
        preferred,
        quals,
        nice_to_have,
        experience,
        education,
        certifications,
    )


def _parse_experience_section(sections: list[Section]) -> list[str]:
    body = _section_by_key(sections, "experience")
    return section_items(body) if body is not None else []


def _parse_education_section(sections: list[Section]) -> list[str]:
    body = _section_by_key(sections, "education")
    return section_items(body) if body is not None else []


def _parse_certifications_section(sections: list[Section]) -> list[str]:
    body = _section_by_key(sections, "certifications")
    return section_items(body) if body is not None else []


def _parse_benefits(sections: list[Section]) -> list[str]:
    body = _section_by_key(sections, "benefits")
    return section_items(body) if body is not None else []


def _parse_salaries(
    sections: list[Section], header: str, text: str
) -> list[Salary]:
    salaries: list[Salary] = []
    seen: set[str] = set()

    for salary in find_salary_statements(header, text):
        if salary.text not in seen:
            seen.add(salary.text)
            salaries.append(salary)

    salary_body = _section_by_key(sections, "salary")
    if salary_body is not None:
        for line in salary_body.splitlines():
            item = line.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            salaries.append(salary_from_line(item))

    return salaries


def _parse_custom_sections(sections: list[Section]) -> list[CustomSection]:
    customs: list[CustomSection] = []
    for section in sections:
        if section.key not in ("custom",):
            continue
        content = [line.strip() for line in section.body.splitlines() if line.strip()]
        customs.append(CustomSection(heading=section.heading, content=content))
    return customs


def _preamble_experience(header: str) -> list[str]:
    """Capture short experience statements in the preamble (e.g. '3+ years')."""
    out: list[str] = []
    for line in header.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if is_experience_statement(stripped) and len(stripped.split()) <= 9:
            out = _append_unique(out, stripped)
    return out


# ---------------------------------------------------------------------------
# Metadata / confidence
# ---------------------------------------------------------------------------


def _build_metadata(
    *,
    text: str,
    file_type: FileTypeEnum,
    title: str | None,
    company: str | None,
    location: str | None,
    summary: str | None,
    responsibilities: list[str],
    required_skills: list[str],
    preferred_skills: list[str],
    qualifications: list[str],
    experience: list[str],
    education: list[str],
    certifications: list[str],
    benefits: list[str],
    salary: list[Salary],
) -> JobMetadata:
    checks: list[tuple[str, object]] = [
        ("summary", summary),
        ("responsibilities", responsibilities),
        ("required_skills", required_skills),
        ("preferred_skills", preferred_skills),
        ("qualifications", qualifications),
        ("experience_requirements", experience),
        ("education_requirements", education),
        ("certifications", certifications),
        ("benefits", benefits),
        ("salary", salary),
    ]
    section_confidence = [
        SectionConfidence(
            section=name,
            level=_level_for_count(len(value)) if isinstance(value, list) else (
                ConfidenceLevel.HIGH if value else ConfidenceLevel.LOW
            ),
        )
        for name, value in checks
    ]

    score = 0
    score += 1 if title else 0
    score += 1 if company else 0
    score += 1 if location else 0
    score += 1 if summary else 0
    score += 1 if responsibilities else 0
    score += 1 if required_skills else 0
    score += 1 if preferred_skills else 0
    score += 1 if qualifications or experience or education else 0
    score += 1 if certifications or benefits or salary else 0

    if score >= 5:
        overall = ConfidenceLevel.HIGH
    elif score >= 2:
        overall = ConfidenceLevel.MEDIUM
    else:
        overall = ConfidenceLevel.LOW

    return JobMetadata(
        word_count=len(text.split()),
        file_type=file_type,
        overall_confidence=overall,
        section_confidence=section_confidence,
    )


def _level_for_count(count: int) -> ConfidenceLevel:
    if count >= 2:
        return ConfidenceLevel.HIGH
    if count == 1:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Small list helpers
# ---------------------------------------------------------------------------


def _append_unique(items: list[str], value: str) -> list[str]:
    if value and value not in items:
        items.append(value)
    return items


def _split_skill_item(item: str) -> list[str]:
    """Split a pure list skill item (e.g. 'Go, Rust') into its entries."""
    parts = [p.strip().rstrip(".,;:") for p in re.split(r"[,|;]", item) if p.strip()]
    return [p for p in parts if len(p.split()) <= 3] or [item]


def _merge_unique(base: list[str], extra: list[str]) -> list[str]:
    for item in extra:
        base = _append_unique(base, item)
    return base
