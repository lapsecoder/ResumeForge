"""Deterministic construction of PII-free text units for semantic matching.

Each unit is independent of the others and excludes all contact information
(name, email, phone, address, personal URLs). Ordering is stable and optional
sections are simply omitted. Spoken languages listed only in the ``languages``
bucket are left out of semantic units; they are handled by the deterministic
matcher instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.job_parsing.schemas import JobDescription
from app.parsing.schemas import (
    Certification,
    Education,
    Project,
    Resume,
    WorkExperience,
)


@dataclass(frozen=True)
class SemanticUnit:
    """A self-contained text snippet plus its matching category."""

    name: str
    category: str
    text: str


def build_resume_units(resume: Resume) -> list[SemanticUnit]:
    """Return deterministic semantic units respecting the resume's sections."""
    units: list[SemanticUnit] = []

    if resume.summary and resume.summary.strip():
        units.append(
            SemanticUnit("summary", "summary", f"Summary: {resume.summary.strip()}")
        )

    for index, skill in enumerate(_resume_skills(resume)):
        units.append(SemanticUnit(f"skill:{index}", "skill", f"Skill: {skill}"))

    for index, stint in enumerate(resume.experience):
        text = _experience_text(stint)
        if text:
            units.append(SemanticUnit(f"experience:{index}", "experience", text))

    for index, project in enumerate(resume.projects):
        text = _project_text(project)
        if text:
            units.append(SemanticUnit(f"project:{index}", "project", text))

    for index, school in enumerate(resume.education):
        text = _education_text(school)
        if text:
            units.append(SemanticUnit(f"education:{index}", "education", text))

    for index, cert in enumerate(resume.certifications):
        text = _certification_text(cert)
        if text:
            units.append(SemanticUnit(f"certification:{index}", "certification", text))

    return units


def build_job_units(job: JobDescription) -> list[SemanticUnit]:
    """Return deterministic semantic units representing the job opening."""
    units: list[SemanticUnit] = []

    title = job.title.strip() if job.title else ""
    summary = job.summary.strip() if job.summary else ""
    if title or summary:
        text = " ".join(
            part for part in (f"Role: {title}" if title else "", summary) if part
        )
        units.append(SemanticUnit("summary", "summary", text))

    for index, resp in enumerate(job.responsibilities):
        if resp.strip():
            units.append(
                SemanticUnit(
                    f"responsibility:{index}", "responsibility", resp.strip()
                )
            )

    for index, skill in enumerate(job.required_skills):
        if skill.strip():
            units.append(
                SemanticUnit(
                    f"required-skill:{index}",
                    "skill",
                    f"Required skill: {skill.strip()}",
                )
            )

    for index, skill in enumerate(job.preferred_skills):
        if skill.strip():
            units.append(
                SemanticUnit(
                    f"preferred-skill:{index}",
                    "skill",
                    f"Preferred skill: {skill.strip()}",
                )
            )

    for index, qual in enumerate(job.qualifications):
        if qual.strip():
            units.append(
                SemanticUnit(f"qualification:{index}", "qualification", qual.strip())
            )

    for index, entry in enumerate(job.experience_requirements):
        if entry.strip():
            units.append(
                SemanticUnit(
                    f"experience-req:{index}",
                    "experience",
                    f"Experience requirement: {entry.strip()}",
                )
            )

    for index, entry in enumerate(job.education_requirements):
        if entry.strip():
            units.append(
                SemanticUnit(
                    f"education-req:{index}",
                    "education",
                    f"Education requirement: {entry.strip()}",
                )
            )

    for index, cert in enumerate(job.certifications):
        if cert.strip():
            units.append(
                SemanticUnit(
                    f"certification-req:{index}", "certification", cert.strip()
                )
            )

    return units


def _resume_skills(resume: Resume) -> list[str]:
    """Order-stable, deduplicated skill values.

    Spoken languages are excluded when they appear in the dedicated
    ``languages`` bucket; the flattened ``all`` bucket is included as-is
    (the parser fills it from the same sources).
    """
    skills = resume.skills
    seen: set[str] = set()
    ordered: list[str] = []
    for bucket in (skills.technical, skills.tools, skills.soft, skills.all):
        for value in bucket or []:
            value = value.strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                ordered.append(value)
    return ordered


def _experience_text(entry: WorkExperience) -> str:
    lines: list[str] = []
    for label, value in (
        ("Role", entry.title),
        ("Company", entry.company),
        ("Location", entry.location),
        ("Period", entry.start_date),
        ("Period", entry.end_date),
    ):
        if value and str(value).strip():
            lines.append(f"{label}: {str(value).strip()}")
    if entry.description and entry.description.strip():
        lines.append(f"Work: {entry.description.strip()}")
    for achievement in entry.achievements:
        if achievement.strip():
            lines.append(f"Achievement: {achievement.strip()}")
    for skill in entry.skills_mentioned:
        if skill.strip():
            lines.append(f"Skill used: {skill.strip()}")
    return "\n".join(lines)


def _project_text(project: Project) -> str:
    lines: list[str] = []
    if project.name.strip():
        lines.append(f"Project: {project.name.strip()}")
    if project.description and project.description.strip():
        lines.append(f"Project detail: {project.description.strip()}")
    for technology in project.technologies:
        if technology.strip():
            lines.append(f"Technology: {technology.strip()}")
    return "\n".join(lines)


def _education_text(entry: Education) -> str:
    lines: list[str] = []
    for label, value in (
        ("School", entry.institution),
        ("Degree", entry.degree),
        ("Field", entry.field),
        ("Dates", entry.start_date),
        ("Dates", entry.end_date),
    ):
        if value and str(value).strip():
            lines.append(f"{label}: {str(value).strip()}")
    for detail in entry.details:
        if detail.strip():
            lines.append(f"Details: {detail.strip()}")
    return "\n".join(lines)


def _certification_text(cert: Certification) -> str:
    lines: list[str] = []
    for label, value in (
        ("Certification", cert.name),
        ("Issuer", cert.issuer),
        ("Date", cert.date),
    ):
        if value and str(value).strip():
            lines.append(f"{label}: {str(value).strip()}")
    return "\n".join(lines)
