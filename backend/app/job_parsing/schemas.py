"""Transient Pydantic schemas for the structured job-description parse result.

These models exist only during request processing. They are not database
tables, are not persisted to disk, and are garbage-collected after the
response is sent. All fields that cannot be extracted reliably are Optional or
empty lists.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ingestion.schemas import FileTypeEnum
from app.parsing.schemas import ConfidenceLevel, CustomSection, SectionConfidence


class Salary(BaseModel):
    """A single preserved salary/compensation statement.

    ``text`` always preserves the original wording. The other fields are
    populated only when a pattern matched with confidence; no currency
    conversion or estimation is ever performed.
    """

    text: str = Field(..., description="Original salary statement, preserved.")
    currency: str | None = Field(
        default=None, description="Detected currency marker, e.g. '₹', '$' or 'INR'."
    )
    period: str | None = Field(
        default=None, description="'hourly', 'monthly', 'annual', or None."
    )
    range_text: str | None = Field(
        default=None, description="Detected numeric range, e.g. '10L–15L'."
    )


class JobMetadata(BaseModel):
    word_count: int = Field(..., ge=0)
    file_type: FileTypeEnum = FileTypeEnum.TXT
    overall_confidence: ConfidenceLevel
    section_confidence: list[SectionConfidence] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Structured, transient representation of a job description.

    Confidence semantics (heuristic, NOT a calibrated probability):
    - HIGH   → several clearly structured, populated sections were recognised.
    - MEDIUM → the document parsed but structure was only partially clear.
    - LOW    → little or nothing could be structured (e.g. bare/unstructured).
    """

    title: str | None = Field(default=None, description="Role title.")
    company: str | None = Field(default=None, description="Employer (labels only).")
    location: str | None = Field(default=None, description="Work location.")
    employment_type: str | None = Field(
        default=None, description="e.g. Full-time, Part-time, Contract, Internship."
    )
    remote_type: str | None = Field(
        default=None, description="e.g. Remote, Fully remote, Hybrid, On-site."
    )
    summary: str | None = Field(default=None, description="Role summary/overview.")
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(
        default_factory=list, description="Generic, unclassifiable requirements."
    )
    experience_requirements: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    salary: list[Salary] = Field(default_factory=list)
    custom_sections: list[CustomSection] = Field(default_factory=list)
    metadata: JobMetadata
