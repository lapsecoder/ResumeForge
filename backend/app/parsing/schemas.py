"""Transient Pydantic schemas for the structured resume parse result.

These models exist only during request processing. They are not database
tables, are not persisted to disk, and are garbage-collected after the
response is sent. All fields that cannot be extracted reliably are Optional.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Heuristic confidence, NOT a calibrated probability.

    Describes how confidently a section or value was recognised by the
    deterministic rules, for rough quality signalling only.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContactInfo(BaseModel):
    name: str | None = Field(default=None, description="Candidate name.")
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = Field(default=None, description="Portfolio / website.")


class WorkExperience(BaseModel):
    company: str
    title: str = Field(default="", description="Role / title.")
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = Field(default=None, description="'Present' if current.")
    description: str = Field(default="", description="Role description.")
    achievements: list[str] = Field(default_factory=list, description="Bullets.")
    skills_mentioned: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    details: list[str] = Field(default_factory=list)


class SkillSet(BaseModel):
    technical: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    all: list[str] = Field(default_factory=list, description="Flattened skills.")


class Project(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    url: str | None = None


class Certification(BaseModel):
    name: str
    issuer: str = ""
    date: str | None = None
    url: str | None = None


class CustomSection(BaseModel):
    heading: str
    content: list[str] = Field(default_factory=list)


class SectionConfidence(BaseModel):
    section: str
    level: ConfidenceLevel


class ResumeMetadata(BaseModel):
    word_count: int = Field(..., ge=0)
    file_type: str = Field(default="text", description='"pdf" | "docx" | "text"')
    overall_confidence: ConfidenceLevel
    section_confidence: list[SectionConfidence] = Field(default_factory=list)
    section_order: list[str] = Field(
        default_factory=list,
        description=(
            "Section keys in document order as detected by the parser "
            "(e.g. ['header', 'summary', 'experience', ...]). Empty when the "
            "ordering was not detected."
        ),
    )


class Resume(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str | None = None
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: SkillSet = Field(default_factory=SkillSet)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    custom_sections: list[CustomSection] = Field(default_factory=list)
    metadata: ResumeMetadata
