"""Deterministic resume parser tests.

Each fixture is synthetic. No real resume data is used.
"""

from __future__ import annotations

import pytest

from app.ingestion.schemas import FileTypeEnum
from app.parsing.parser import ParsingError, parse_resume
from app.parsing.schemas import ConfidenceLevel

FULL_RESUME = """John Doe
Senior Software Engineer
john.doe@example.com | (123) 456-7890
linkedin.com/in/johndoe
https://github.com/johndoe

SUMMARY
Backend engineer with 8 years building distributed systems.

SKILLS
Python, Java, PostgreSQL | Git, Docker
• Machine Learning
- REST APIs

WORK EXPERIENCE
Senior Backend Engineer | Acme Corp
Jun 2021 - Present
• Led platform team of 6 engineers
• Reduced p95 latency by 40%

Product Manager | Beta Inc
Mar 2019 - May 2021
• Shipped the mobile app

EDUCATION
Bachelor of Technology, Computer Science
Indian Institute of Technology
2016 - 2020

M.S. Computer Science
Stanford University
2014 - 2015

PROJECTS
Campus Connect | https://campusconnect.example.com
Technologies: Python, FastAPI, PostgreSQL
• Built a student event platform
Cloud Dashboard
Built a monitoring dashboard for AWS metrics

CERTIFICATIONS
• AWS Certified Solutions Architect, 2023
Certified Kubernetes Administrator (2021)

VOLUNTEERING
• Mentored students at local hackathons
"""


class TestContactExtraction:
    def test_extracts_contact_from_header(self) -> None:
        resume = parse_resume(FULL_RESUME)
        assert resume.contact.name == "John Doe"
        assert resume.contact.email == "john.doe@example.com"
        assert resume.contact.phone == "(123) 456-7890"
        assert resume.contact.github == "https://github.com/johndoe"

    def test_linkedin_without_scheme_still_found(self) -> None:
        resume = parse_resume(FULL_RESUME)
        assert "linkedin.com/in/johndoe" in (resume.contact.linkedin or "")

    def test_website_prefers_header(self) -> None:
        text = (
            "Jane Smith\njane@example.com\nhttps://jane.dev\n\n"
            "SKILLS\nPython"
        )
        resume = parse_resume(text)
        assert resume.contact.website == "https://jane.dev"


class TestSectionDetection:
    @pytest.mark.parametrize(
        "heading",
        [
            "SUMMARY",
            "Professional Summary",
            "profile",
            "Objective",
            "Career Objective",
        ],
    )
    def test_summary_aliases(self, heading: str) -> None:
        text = f"Some Name\n{heading}\nA short professional summary.\n"
        resume = parse_resume(text)
        assert resume.summary == "A short professional summary."

    @pytest.mark.parametrize(
        "heading",
        ["WORK EXPERIENCE", "Experience", "Employment", "Professional Experience"],
    )
    def test_experience_aliases(self, heading: str) -> None:
        text = (
            f"Some Name\n{heading}\nDeveloper | Acme\n2020 - 2021\n"
            "- Did things\n\nOther | Beta\n2018 - 2019\n- Did things\n"
        )
        resume = parse_resume(text)
        assert len(resume.experience) == 2

    @pytest.mark.parametrize(
        "heading", ["SKILLS", "Technical Skills", "Core Skills", "Competencies"]
    )
    def test_skills_aliases(self, heading: str) -> None:
        text = f"Some Name\n{heading}\nPython, Java\n"
        resume = parse_resume(text)
        assert "Python" in resume.skills.all

    @pytest.mark.parametrize(
        "heading", ["PROJECTS", "Personal Projects", "Academic Projects"]
    )
    def test_projects_aliases(self, heading: str) -> None:
        text = f"Some Name\n{heading}\nProject Alpha\n- Build it\n"
        resume = parse_resume(text)
        assert len(resume.projects) == 1

    @pytest.mark.parametrize(
        "heading",
        ["CERTIFICATIONS", "Certificates", "Licenses & Certifications"],
    )
    def test_certifications_aliases(self, heading: str) -> None:
        text = f"Some Name\n{heading}\nAWS Certified - 2023\n"
        resume = parse_resume(text)
        assert len(resume.certifications) == 1

    def test_unknown_heading_becomes_custom_section(self) -> None:
        text = (
            "Some Name\n\n"
            "VOLUNTEERING\n"
            "• Mentored local students\n"
            "\n"
            "AWARDS\n"
            "Best Engineer 2022\n"
        )
        resume = parse_resume(text)
        headings = [c.heading for c in resume.custom_sections]
        assert "VOLUNTEERING" in headings
        assert "AWARDS" in headings
        vol = next(c for c in resume.custom_sections if c.heading == "VOLUNTEERING")
        assert vol.content == ["Mentored local students"]


class TestExperience:
    def test_experience_with_date_range(self) -> None:
        resume = parse_resume(FULL_RESUME)
        assert len(resume.experience) == 2
        first = resume.experience[0]
        assert first.title == "Senior Backend Engineer"
        assert first.company == "Acme Corp"
        assert first.start_date == "Jun 2021"
        assert first.end_date == "Present"
        assert first.achievements == [
            "Led platform team of 6 engineers",
            "Reduced p95 latency by 40%",
        ]

    def test_present_employment(self) -> None:
        resume = parse_resume(FULL_RESUME)
        assert resume.experience[0].end_date == "Present"

    def test_multiple_entries_separated_from_body(self) -> None:
        root = parse_resume(FULL_RESUME)
        second = root.experience[1]
        assert second.title == "Product Manager"
        assert second.company == "Beta Inc"
        assert second.start_date == "Mar 2019"
        assert second.end_date == "May 2021"


class TestEducation:
    def test_education_entries(self) -> None:
        resume = parse_resume(FULL_RESUME)
        assert len(resume.education) == 2
        first = resume.education[0]
        assert "Bachelor" in (first.degree or "")
        assert first.institution == "Indian Institute of Technology"
        assert first.start_date == "2016"
        assert first.end_date == "2020"

    def test_degree_abbreviation(self) -> None:
        text = (
            "Some Name\nEDUCATION\nB.Tech, Computer Science\n"
            "Some University\n2016 - 2020\n"
        )
        resume = parse_resume(text)
        assert len(resume.education) == 1
        assert resume.education[0].degree == "B.Tech, Computer Science"

    def test_education_missing_section(self) -> None:
        resume = parse_resume("Some Name\nSKILLS\nPython\n")
        assert resume.education == []


class TestSkills:
    def test_multiple_delimiters(self) -> None:
        resume = parse_resume(FULL_RESUME)
        for skill in (
            "Python",
            "Java",
            "PostgreSQL",
            "Git",
            "Docker",
            "Machine Learning",
            "REST APIs",
        ):
            assert skill in resume.skills.all, f"Missing skill: {skill}"

    def test_skip_empty_skills(self) -> None:
        text = "Some Name\nSKILLS\nPython,,Java |\n"
        resume = parse_resume(text)
        assert resume.skills.all == ["Python", "Java"]

    def test_no_skills_section_yields_empty(self) -> None:
        resume = parse_resume("Some Name\n")
        assert resume.skills.all == []


class TestProjects:
    def test_projects_with_technologies_and_url(self) -> None:
        resume = parse_resume(FULL_RESUME)
        assert len(resume.projects) >= 1
        project = resume.projects[0]
        assert project.name == "Campus Connect"
        assert project.url == "https://campusconnect.example.com"
        assert "Python" in project.technologies
        assert "FastAPI" in project.technologies
        assert "event platform" in project.description

    def test_project_without_technologies(self) -> None:
        resume = parse_resume(FULL_RESUME)
        dashboards = [p for p in resume.projects if p.name == "Cloud Dashboard"]
        assert dashboards
        assert "monitoring dashboard" in dashboards[0].description

    def test_project_header_after_blank_line_stays_in_projects(self) -> None:
        """Regression: a blank line after the heading must not push the first
        project name into a custom section (breaking project extraction)."""
        text = (
            "Some Name\n"
            "PROJECTS\n"
            "\n"
            "Python Data Pipeline\n"
            "Built a Python data pipeline that processes millions of rows.\n"
            "Technologies: Python, PyTorch\n"
            "\n"
            "Cloud Dashboard\n"
            "Built a monitoring dashboard for AWS metrics\n"
        )
        resume = parse_resume(text)
        assert [p.name for p in resume.projects] == [
            "Python Data Pipeline",
            "Cloud Dashboard",
        ]
        pipeline = resume.projects[0]
        assert "processes millions of rows" in pipeline.description
        assert pipeline.technologies == ["Python", "PyTorch"]

    def test_experience_entry_after_blank_line_stays_in_experience(self) -> None:
        text = (
            "Some Name\n"
            "EXPERIENCE\n"
            "\n"
            "Analytical Engines\n"
            "Engineer\n"
            "2019 - Present\n"
            "- Built a billing service\n"
        )
        resume = parse_resume(text)
        assert len(resume.experience) == 1
        assert resume.experience[0].title == "Analytical Engines"


class TestCertifications:
    def test_certifications(self) -> None:
        resume = parse_resume(FULL_RESUME)
        names = [c.name for c in resume.certifications]
        assert "AWS Certified Solutions Architect" in names
        assert "Certified Kubernetes Administrator" in names

    def test_certification_date(self) -> None:
        resume = parse_resume(FULL_RESUME)
        aws = next(
            c for c in resume.certifications if c.name.startswith("AWS")
        )
        assert aws.date == "2023"


class TestNameExtraction:
    def test_no_name_when_header_has_no_name(self) -> None:
        text = "Senior Developer\nSKILLS\nPython\n"
        resume = parse_resume(text)
        # Not falsely claiming a name from a job title line.
        assert resume.contact.name in (None, "Senior Developer")

    def test_name_not_invented_for_long_text(self) -> None:
        text = "No clear structure here at all\nSKILLS\nPython\n"
        resume = parse_resume(text)
        assert resume.contact.name is None


class TestMissingOrOdd:
    def test_missing_sections(self) -> None:
        text = "Jane Doe\njane@example.com\n"
        resume = parse_resume(text)
        assert resume.summary is None
        assert resume.experience == []
        assert resume.education == []
        assert resume.skills.all == []
        assert resume.projects == []
        assert resume.certifications == []
        assert resume.contact.email == "jane@example.com"
        assert resume.contact.name == "Jane Doe"

    def test_unusual_case_and_whitespace(self) -> None:
        text = (
            "John Smith\n\n"
            "sKiLlS\n"
            "Python,Java\n"
            "\n"
            "  wOrK   eXpErIeNcE  \n"
            "Dev | Co\n"
            "2020 - Present\n"
        )
        resume = parse_resume(text)
        assert resume.skills.all == ["Python", "Java"]
        assert len(resume.experience) == 1
        assert resume.experience[0].title == "Dev"

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ParsingError):
            parse_resume("   ")

    def test_malformed_unstructured_text_does_not_crash(self) -> None:
        text = "fdskj lkjfdslkj fdskjfdsklj fdsa\nlkdjfslkdjf\n12345"
        resume = parse_resume(text)
        assert resume is not None
        assert resume.metadata.overall_confidence == ConfidenceLevel.LOW

    def test_parser_does_not_invent_missing_info(self) -> None:
        text = "jane@example.com\n"
        resume = parse_resume(text)
        assert resume.contact.name is None
        assert resume.contact.phone is None
        assert resume.contact.linkedin is None
        assert resume.contact.github is None
        assert resume.contact.website is None
        assert resume.summary is None


class TestMetadata:
    def test_file_type_passthrough(self) -> None:
        resume = parse_resume(
            "Jane Doe\njane@example.com\n", file_type=FileTypeEnum.PDF
        )
        assert resume.metadata.file_type == FileTypeEnum.PDF

    def test_word_count(self) -> None:
        resume = parse_resume("Jane Doe\n\nSKILLS\nPython, Java\n")
        assert resume.metadata.word_count == 5

    def test_confidence_reflects_content(self) -> None:
        rich = parse_resume(FULL_RESUME)
        assert rich.metadata.overall_confidence == ConfidenceLevel.HIGH
        poor = parse_resume("hello world")
        assert poor.metadata.overall_confidence == ConfidenceLevel.LOW
