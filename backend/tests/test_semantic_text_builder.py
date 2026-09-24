"""Tests for PII-free, deterministic semantic text-unit construction."""

from __future__ import annotations

from app.job_parsing.schemas import JobDescription, JobMetadata
from app.parsing.schemas import (
    Certification,
    ConfidenceLevel,
    Education,
    Project,
    Resume,
    ResumeMetadata,
    SkillSet,
    WorkExperience,
)
from app.semantic_matching.text_builder import (
    build_job_units,
    build_resume_units,
)


def _resume_metadata() -> ResumeMetadata:
    return ResumeMetadata(
        word_count=0, file_type="pdf", overall_confidence=ConfidenceLevel.HIGH
    )


def rich_resume() -> Resume:
    return Resume(
        contact={
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "(555) 010-2030",
            "location": "111 Main St, Springfield",
            "linkedin": "linkedin.com/in/janedoe",
            "github": "github.com/janedoe",
            "website": "jane-doe.dev",
        },
        summary="Backend engineer focused on distributed systems.",
        skills=SkillSet(
            technical=["Python", "PostgreSQL"],
            tools=["Docker"],
            soft=["Leadership"],
            languages=["Spanish"],
            all=["Python", "PostgreSQL", "Docker", "Leadership"],
        ),
        experience=[
            WorkExperience(
                company="Acme Corp",
                title="Senior Engineer",
                location="Remote",
                start_date="2020",
                end_date="Present",
                description="Built REST APIs for the payments platform.",
                achievements=["Cut p99 latency by 40%."],
                skills_mentioned=["Kafka"],
            )
        ],
        projects=[
            Project(
                name="Realtime Dashboard",
                description="Streams metrics.",
                technologies=["Redis"],
            )
        ],
        education=[Education(institution="IIT Delhi", degree="B.Tech", field="CS")],
        certifications=[Certification(name="AWS Solutions Architect", issuer="AWS")],
        metadata=_resume_metadata(),
    )


def generic_job() -> JobDescription:
    return JobDescription(
        title="Backend Engineer",
        summary="Design and scale payment APIs.",
        responsibilities=[
            "Ship reliable REST APIs.",
            "Operate PostgreSQL in production.",
        ],
        required_skills=["Python"],
        preferred_skills=["Kafka"],
        qualifications=["Bachelor's in CS"],
        experience_requirements=["3+ years"],
        education_requirements=["Bachelor's degree"],
        certifications=["AWS Certified"],
        metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
    )


class TestResumeUnits:
    def test_all_sections_are_represented(self) -> None:
        units = build_resume_units(rich_resume())
        text = "\n".join(unit.text for unit in units)
        assert "distributed systems" in text
        assert "Python" in text
        assert "Acme Corp" in text
        assert "Realtime Dashboard" in text
        assert "IIT Delhi" in text
        assert "AWS Solutions Architect" in text

    def test_contact_and_pii_are_excluded(self) -> None:
        text = "\n".join(
            unit.text for unit in build_resume_units(rich_resume())
        ).lower()
        secrets = {
            "jane doe",
            "jane.doe@example.com",
            "010-2030",
            "main st",
            "linkedin.com",
            "github.com/",
            "jane-doe.dev",
        }
        for secret in secrets:
            assert secret not in text

    def test_spoken_languages_are_not_semantic_units(self) -> None:
        units = build_resume_units(rich_resume())
        assert not any("Spanish" in unit.text for unit in units)

    def test_deterministic_across_calls(self) -> None:
        first = [unit.text for unit in build_resume_units(rich_resume())]
        second = [unit.text for unit in build_resume_units(rich_resume())]
        assert first == second

    def test_empty_optional_sections_are_omitted(self) -> None:
        units = build_resume_units(Resume(metadata=_resume_metadata()))
        assert units == []

    def test_unit_names_are_stable_and_unique(self) -> None:
        units = build_resume_units(rich_resume())
        names = [unit.name for unit in units]
        assert len(names) == len(set(names))
        assert [u.name for u in units] == names


class TestJobUnits:
    def test_all_job_sections_are_represented(self) -> None:
        units = build_job_units(generic_job())
        text = "\n".join(unit.text for unit in units)
        assert "Backend Engineer" in text
        assert "Ship reliable REST APIs" in text
        assert "Required skill: Python" in text
        assert "Preferred skill: Kafka" in text
        assert "Bachelor's in CS" in text
        assert "3+ years" in text
        assert "AWS Certified" in text

    def test_deterministic_across_calls(self) -> None:
        first = [unit.text for unit in build_job_units(generic_job())]
        second = [unit.text for unit in build_job_units(generic_job())]
        assert first == second

    def test_empty_job_produces_no_units(self) -> None:
        job = JobDescription(
            metadata=JobMetadata(
                word_count=0, overall_confidence=ConfidenceLevel.HIGH
            )
        )
        assert build_job_units(job) == []

    def test_required_and_preferred_skills_are_distinct_units(self) -> None:
        units = build_job_units(generic_job())
        required = [
            u
            for u in units
            if u.category == "skill" and u.name.startswith("required-skill")
        ]
        preferred = [
            u
            for u in units
            if u.category == "skill" and u.name.startswith("preferred-skill")
        ]
        assert required and preferred
        assert "Python" in required[0].text
        assert "Kafka" in preferred[0].text


class TestPIIDetection:
    def test_no_phone_or_email_in_any_unit(self) -> None:
        text = "\n".join(unit.text for unit in build_resume_units(rich_resume()))
        for marker in ("(555)", "@", "example.com", "+1-", "010-2030"):
            assert marker not in text
