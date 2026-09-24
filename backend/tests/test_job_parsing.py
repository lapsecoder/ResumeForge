"""Deterministic job-description parser tests.

Each fixture is synthetic. No real job descriptions, companies, or people are
used. The parser must never invent missing information.
"""

from __future__ import annotations

import pytest

from app.ingestion.schemas import FileTypeEnum
from app.job_parsing.parser import ParsingError, parse_job_description
from app.parsing.schemas import ConfidenceLevel

FULL_JD = """Senior Software Engineer
Company: Acme Corp
Location: Bangalore
Employment Type: Full-time
Remote: Hybrid

About the role
Build and own scalable backend services for millions of users.

Responsibilities
• Design and review APIs
• Mentor junior engineers
- Own the release process

Requirements
- 5+ years of experience with Python
- Bachelor's degree in Computer Science
- AWS Certified Solutions Architect
- Go, Rust
- Strong communication skills

Nice to have
- Kubernetes
- Familiarity with fintech payments

Experience
- At least 2 years of distributed systems

Education
- Master's degree preferred

Certifications
- PMP

Benefits
- Health insurance
- Learning budget

Salary: ₹18L–25L annual
"""


class TestFullJobDescription:
    def test_extracts_metadata_fields(self) -> None:
        job = parse_job_description(FULL_JD)
        assert job.title == "Senior Software Engineer"
        assert job.company == "Acme Corp"
        assert job.location == "Bangalore"
        assert job.employment_type == "Full-time"
        assert job.remote_type == "Hybrid"

    def test_extracts_summary_verbatim(self) -> None:
        job = parse_job_description(FULL_JD)
        assert (
            job.summary
            == "Build and own scalable backend services for millions of users."
        )

    def test_extracts_responsibilities_verbatim(self) -> None:
        job = parse_job_description(FULL_JD)
        assert job.responsibilities == [
            "Design and review APIs",
            "Mentor junior engineers",
            "Own the release process",
        ]

    def test_extracts_required_skills(self) -> None:
        job = parse_job_description(FULL_JD)
        assert "Go" in job.required_skills
        assert "Rust" in job.required_skills
        assert "Python" not in job.required_skills  # experience statement, not skill

    def test_soft_skill_item_stays_a_requirement(self) -> None:
        job = parse_job_description(FULL_JD)
        assert "Strong communication skills" in job.qualifications

    def test_extracts_preferred_skills_and_nice_to_have(self) -> None:
        job = parse_job_description(FULL_JD)
        assert job.preferred_skills == ["Kubernetes"]
        assert "Familiarity with fintech payments" in job.nice_to_have

    def test_extracts_experience_requirements(self) -> None:
        job = parse_job_description(FULL_JD)
        assert "5+ years of experience with Python" in job.experience_requirements
        assert "At least 2 years of distributed systems" in job.experience_requirements

    def test_extracts_education_requirements(self) -> None:
        job = parse_job_description(FULL_JD)
        assert "Bachelor's degree in Computer Science" in job.education_requirements
        assert "Master's degree preferred" in job.education_requirements

    def test_extracts_certifications(self) -> None:
        job = parse_job_description(FULL_JD)
        assert "AWS Certified Solutions Architect" in job.certifications
        assert "PMP" in job.certifications

    def test_extracts_benefits(self) -> None:
        job = parse_job_description(FULL_JD)
        assert job.benefits == ["Health insurance", "Learning budget"]

    def test_extracts_salary(self) -> None:
        job = parse_job_description(FULL_JD)
        assert len(job.salary) == 1
        salary = job.salary[0]
        assert salary.text == "Salary: ₹18L–25L annual"
        assert salary.currency == "₹"
        assert salary.period == "annual"
        assert salary.range_text == "18L–25L"

    def test_metadata_confidence(self) -> None:
        job = parse_job_description(FULL_JD)
        assert job.metadata.overall_confidence == ConfidenceLevel.HIGH
        assert job.metadata.word_count == len(FULL_JD.split())
        assert job.metadata.file_type == FileTypeEnum.TXT


class TestTitle:
    def test_title_from_label(self) -> None:
        job = parse_job_description(
            "Company: Acme Corp\nJob Title: DevOps Engineer\nLocation: Pune\n"
        )
        assert job.title == "DevOps Engineer"

    def test_title_from_position_label(self) -> None:
        job = parse_job_description(
            "Position: Data Scientist\n\nResponsibilities\n- x\n"
        )
        assert job.title == "Data Scientist"

    def test_title_from_strong_heading(self) -> None:
        job = parse_job_description(
            "Front-End Developer\nCompany: Beta Inc\n\nResponsibilities\n- x\n"
        )
        assert job.title == "Front-End Developer"

    def test_title_optional_when_ambiguous(self) -> None:
        job = parse_job_description("We are looking for a backend engineer today.\n")
        assert job.title is None


class TestCompany:
    param_company_labels = [
        "Company: Acme Corp",
        "Employer: Acme Corp",
        "Organization: Acme Corp",
        "Organisation: Acme Corp",
    ]

    @pytest.mark.parametrize("label", param_company_labels)
    def test_company_from_explicit_label(self, label: str) -> None:
        job = parse_job_description(f"{label}\n\nResponsibilities\n- x\n")
        assert job.company == "Acme Corp"

    def test_company_not_guessed_from_prose(self) -> None:
        job = parse_job_description(
            "Engineer\nWe are a fast growing startup in Bengaluru.\n"
        )
        assert job.company is None


class TestLocation:
    @pytest.mark.parametrize(
        "label",
        ["Location: Bangalore", "Job location: Chennai", "Work location: Hyderabad"],
    )
    def test_location_from_label(self, label: str) -> None:
        job = parse_job_description(f"{label}\n\nResponsibilities\n- x\n")
        assert job.location in ("Bangalore", "Chennai", "Hyderabad")

    def test_location_remote_sets_remote_type(self) -> None:
        job = parse_job_description("Location: Remote\nEmployment Type: Full-time\n")
        assert job.location is None
        assert job.remote_type == "Remote"

    def test_location_not_inferred_from_unrelated_text(self) -> None:
        job = parse_job_description(
            "Engineer\nWorks across Delhi, Mumbai and Bangalore time zones.\n"
        )
        assert job.location is None


class TestEmploymentType:
    def test_from_label(self) -> None:
        job = parse_job_description(
            "Job Type: Part-time\nCompany: Acme\n\nResponsibilities\n- x\n"
        )
        assert job.employment_type == "Part-time"

    def test_from_header_token(self) -> None:
        job = parse_job_description("Internship\nCompany: Acme\n\nSkills\nPython\n")
        assert job.employment_type == "Internship"

    def test_contract_label(self) -> None:
        job = parse_job_description("Employment Type: Contract\n\nSkills\nPython\n")
        assert job.employment_type == "Contract"

    def test_temporary(self) -> None:
        job = parse_job_description("Employment Type: Temporary\n\nSkills\nPython\n")
        assert job.employment_type == "Temporary"

    def test_not_inferred_from_body(self) -> None:
        job = parse_job_description(
            "Engineer\n\nResponsibilities\n- Manage a full-time holiday party plan\n"
        )
        assert job.employment_type is None


class TestRemoteType:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Remote\n", "Remote"),
            ("Work Arrangement: Fully remote\n", "Fully remote"),
            ("Workplace Type: Hybrid\n", "Hybrid"),
            ("Work Mode: On-site\n", "On-site"),
            ("Work Mode: Onsite\n", "On-site"),
            ("Remote: Yes\n", "Remote"),
            ("Work from home anywhere in India\n", "Work from home"),
        ],
    )
    def test_remote_hybrid_onsite(self, text: str, expected: str) -> None:
        job = parse_job_description(f"Engineer\nCompany: Acme\n{text}")
        assert job.remote_type == expected

    def test_remote_not_inferred_from_missing_location(self) -> None:
        job = parse_job_description("Engineer\nCompany: Acme\n\nSkills\nPython\n")
        assert job.remote_type is None


class TestResponsibilities:
    def test_bullets_only(self) -> None:
        text = "Responsibilities\n- Write code\n• Ship features\n* Review PRs\n"
        job = parse_job_description(text)
        assert job.responsibilities == ["Write code", "Ship features", "Review PRs"]

    def test_non_bullet_lines_preserved(self) -> None:
        text = "Responsibilities\nLead the platform team\nOwn the roadmap\n"
        job = parse_job_description(text)
        assert job.responsibilities == ["Lead the platform team", "Own the roadmap"]


class TestSkillExtraction:
    def test_comma_separated(self) -> None:
        text = "Skills\nPython, Java, SQL\n"
        job = parse_job_description(text)
        assert job.required_skills == ["Python", "Java", "SQL"]

    def test_pipe_separated(self) -> None:
        text = "Skills\nPython | React | PostgreSQL\n"
        job = parse_job_description(text)
        assert job.required_skills == ["Python", "React", "PostgreSQL"]

    def test_semicolon_separated(self) -> None:
        text = "Skills\nPython; Go; TypeScript\n"
        job = parse_job_description(text)
        assert job.required_skills == ["Python", "Go", "TypeScript"]

    def test_bullet_list(self) -> None:
        text = "Skills\n- Python\n- AWS\n- Docker\n"
        job = parse_job_description(text)
        assert job.required_skills == ["Python", "AWS", "Docker"]

    def test_tech_stack_alias(self) -> None:
        text = "Tech Stack\nFastAPI, PostgreSQL, Redis\n"
        job = parse_job_description(text)
        assert "FastAPI" in job.required_skills

    def test_required_skills_alias(self) -> None:
        text = "Required Skills\nTerraform, Ansible\n"
        job = parse_job_description(text)
        assert job.required_skills == ["Terraform", "Ansible"]

    def test_required_technologies_in_requirements(self) -> None:
        text = "Requirements\n- Java, Spring Boot\n- Understanding of microservices\n"
        job = parse_job_description(text)
        assert "Java" in job.required_skills
        assert "Spring Boot" in job.required_skills
        assert "Understanding of microservices" in job.qualifications


class TestQualifications:
    def test_generic_requirements_preserved(self) -> None:
        text = (
            "Requirements\n"
            "- Work authorization in India\n"
            "- Ability to travel 10%\n"
            "- Fluency in English\n"
        )
        job = parse_job_description(text)
        assert job.qualifications == [
            "Work authorization in India",
            "Ability to travel 10%",
            "Fluency in English",
        ]

    def test_degree_routed_to_education(self) -> None:
        text = "Requirements\n- M.Tech in Computer Science\n"
        job = parse_job_description(text)
        assert "M.Tech in Computer Science" in job.education_requirements
        assert not job.qualifications


class TestExperienceRequirements:
    @pytest.mark.parametrize(
        "line",
        [
            "2+ years",
            "3 years of experience",
            "minimum 5 years of experience",
            "at least 1 year of related experience",
            "entry level",
            "senior-level experience",
        ],
    )
    def test_patterns_recognised(self, line: str) -> None:
        text = f"Experience\n{line}\n"
        job = parse_job_description(text)
        assert line in job.experience_requirements

    def test_experience_pattern_from_requirements(self) -> None:
        text = "Requirements\n- 7+ years of total experience\n- B.Sc degree\n"
        job = parse_job_description(text)
        assert "7+ years of total experience" in job.experience_requirements

    def test_header_experience_statement(self) -> None:
        text = "Engineer\n3-5 years of experience required\n\nSkills\nPython\n"
        job = parse_job_description(text)
        assert "3-5 years of experience required" in job.experience_requirements

    def test_never_interpreted_as_applicant_experience(self) -> None:
        text = "Experience\n2+ years building production APIs\n"
        job = parse_job_description(text)
        assert "2+ years building production APIs" in job.experience_requirements


class TestEducationRequirements:
    @pytest.mark.parametrize(
        "line",
        [
            "Bachelor's degree",
            "Master's degree in Computer Science",
            "B.Tech",
            "B.E. in Electronics",
            "B.Sc in Physics",
            "MCA preferred",
            "M.Tech",
            "MBA",
            "PhD in a related field",
            "Diploma",
        ],
    )
    def test_degree_lines_recognised(self, line: str) -> None:
        text = f"Education\n{line}\n"
        job = parse_job_description(text)
        assert line in job.education_requirements

    def test_education_not_inferred_from_prose(self) -> None:
        text = "Summary\nWe educate our customers with great documentation.\n"
        job = parse_job_description(text)
        assert job.education_requirements == []

    def test_degree_in_requirements(self) -> None:
        text = "Requirements\n- B.E. or B.Tech in CSE\n"
        job = parse_job_description(text)
        assert "B.E. or B.Tech in CSE" in job.education_requirements


class TestCertifications:
    def test_certification_section(self) -> None:
        text = (
            "Certifications\n"
            "- AWS Certified Solutions Architect\n"
            "- Cisco CCNA\n"
        )
        job = parse_job_description(text)
        assert job.certifications == [
            "AWS Certified Solutions Architect",
            "Cisco CCNA",
        ]

    def test_cert_like_in_requirements(self) -> None:
        text = "Requirements\n- PMP certification is a plus\n- Azure Certified\n"
        job = parse_job_description(text)
        assert "PMP certification is a plus" in job.certifications
        assert "Azure Certified" in job.certifications

    def test_no_certs_invented(self) -> None:
        text = "Requirements\n- AWS experience in production\n"
        job = parse_job_description(text)
        assert job.certifications == []


class TestSalary:
    @pytest.mark.parametrize(
        ("line", "currency", "period", "range_text"),
        [
            ("Salary: ₹10L–15L\n", "₹", None, "10L–15L"),
            (
                "Salary: ₹1200000 - ₹1500000 per year\n",
                "₹",
                "annual",
                "1200000–1500000",
            ),
            ("Compensation: $100k-$130k\n", "$", None, "100k–130k"),
            (
                "Pay range: 60,000 - 80,000 per month\n",
                None,
                "monthly",
                "60000–80000",
            ),
            ("Hourly rate: $40-50\n", "$", "hourly", "40–50"),
            ("CTC: 20 LPA\n", None, "annual", None),
        ],
    )
    def test_salary_formats(
        self,
        line: str,
        currency: str | None,
        period: str | None,
        range_text: str | None,
    ) -> None:
        job = parse_job_description(f"Engineer\n{line}")
        assert job.salary, f"no salary captured for {line!r}"
        salary = job.salary[0]
        assert salary.text.strip() == line.strip()
        assert salary.currency == currency
        assert salary.period == period
        assert salary.range_text == range_text

    def test_salary_section(self) -> None:
        text = "Salary\n₹18L–25L annual\n"
        job = parse_job_description(text)
        assert job.salary[0].text == "₹18L–25L annual"

    def test_salary_preserved_when_uncertain(self) -> None:
        text = "Salary\nCompetitive salary based on experience\n"
        job = parse_job_description(text)
        assert len(job.salary) == 1
        assert job.salary[0].text == "Competitive salary based on experience"
        assert job.salary[0].currency is None

    def test_no_salary_inferred_without_statement(self) -> None:
        text = "Summary\nWe offer a great working environment.\n"
        job = parse_job_description(text)
        assert job.salary == []

    def test_no_currency_conversion(self) -> None:
        text = "Salary: $80,000 - $95,000\n"
        job = parse_job_description(text)
        assert job.salary[0].currency == "$"
        assert job.salary[0].range_text == "80000–95000"


class TestBenefits:
    def test_benefits_extracted(self) -> None:
        text = (
            "Benefits\n"
            "- Health insurance\n"
            "- 25 days paid leave\n"
            "- Flexible hours\n"
        )
        job = parse_job_description(text)
        assert job.benefits == [
            "Health insurance",
            "25 days paid leave",
            "Flexible hours",
        ]

    def test_perks_alias(self) -> None:
        text = "Perks\n- Free meals\n"
        job = parse_job_description(text)
        assert job.benefits == ["Free meals"]

    def test_benefits_not_invented(self) -> None:
        text = "Summary\nWe take good care of our people.\n"
        job = parse_job_description(text)
        assert job.benefits == []


class TestCustomSections:
    def test_unknown_section_preserved(self) -> None:
        text = (
            "Responsibilities\n- Do things\n\n"
            "Our Culture\n"
            "We value ownership and curiosity.\n"
            "Hack culture is encouraged.\n"
        )
        job = parse_job_description(text)
        assert len(job.custom_sections) == 1
        section = job.custom_sections[0]
        assert section.heading == "Our Culture"
        assert section.content == [
            "We value ownership and curiosity.",
            "Hack culture is encouraged.",
        ]

    def test_unknown_section_before_first_known_heading(self) -> None:
        text = (
            "About Us\nWe are a small team.\n\n"
            "Responsibilities\n- Ship software\n"
        )
        job = parse_job_description(text)
        assert job.custom_sections == []
        assert job.responsibilities == ["Ship software"]


class TestSectionAliases:
    @pytest.mark.parametrize(
        "heading",
        [
            "Responsibilities",
            "WHAT YOU'LL DO",
            "What you will do",
            "Role and Responsibilities",
            "Duties",
            "Key responsibilities",
        ],
    )
    def test_responsibility_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\n- Do the thing\n")
        assert job.responsibilities == ["Do the thing"]

    @pytest.mark.parametrize(
        "heading",
        [
            "Requirements",
            "Required Qualifications",
            "Minimum Qualifications",
            "Must Have",
            "What We're Looking For",
            "Qualifications",
        ],
    )
    def test_requirements_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\n- Java\n- Fast learner\n")
        assert "Java" in job.required_skills
        assert "Fast learner" in job.qualifications

    @pytest.mark.parametrize(
        "heading",
        [
            "Preferred Qualifications",
            "Preferred Skills",
            "Nice to Have",
            "Bonus",
            "Desirable",
        ],
    )
    def test_preferred_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\n- Rust\n")
        assert job.preferred_skills == ["Rust"]

    @pytest.mark.parametrize(
        "heading",
        [
            "Skills",
            "Technical Skills",
            "Technologies",
            "Tech Stack",
            "Required Skills",
        ],
    )
    def test_skills_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\nPython\n")
        assert job.required_skills == ["Python"]

    @pytest.mark.parametrize(
        "heading",
        [
            "Education",
            "Educational Qualifications",
            "Academic Qualifications",
        ],
    )
    def test_education_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\nB.E. Computer Science\n")
        assert "B.E. Computer Science" in job.education_requirements

    @pytest.mark.parametrize(
        "heading",
        [
            "Experience",
            "Required Experience",
            "Years of Experience",
            "Professional Experience",
        ],
    )
    def test_experience_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\n4+ years\n")
        assert "4+ years" in job.experience_requirements

    @pytest.mark.parametrize(
        "heading",
        [
            "Benefits",
            "Perks",
            "What We Offer",
        ],
    )
    def test_benefits_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\n- Gym membership\n")
        assert job.benefits == ["Gym membership"]

    @pytest.mark.parametrize(
        "heading",
        [
            "Salary",
            "Compensation",
            "Pay",
            "Remuneration",
        ],
    )
    def test_salary_aliases(self, heading: str) -> None:
        job = parse_job_description(f"{heading}\n₹10L-12L\n")
        assert len(job.salary) == 1
        assert job.salary[0].range_text == "10L–12L"

    def test_summary_aliases(self) -> None:
        job = parse_job_description("Overview\nWe ship things.\n")
        assert job.summary == "We ship things."

    def test_numbered_heading(self) -> None:
        job = parse_job_description("1. Responsibilities\n- Do things\n")
        assert job.responsibilities == ["Do things"]

    def test_heading_with_colon(self) -> None:
        job = parse_job_description("Responsibilities:\n- Do things\n")
        assert job.responsibilities == ["Do things"]


class TestVariations:
    def test_capitalization(self) -> None:
        text = "RESPONSIBILITIES\n- Do things\n\nSKILLS\nPython\n"
        job = parse_job_description(text)
        assert job.responsibilities == ["Do things"]
        assert job.required_skills == ["Python"]

    def test_whitespace_variations(self) -> None:
        text = (
            "   Responsibilities   \n"
            "   - Do things   \n"
            "\n"
            "   Skills   \n"
            "   Python   \n"
        )
        job = parse_job_description(text)
        assert job.responsibilities == ["Do things"]
        assert job.required_skills == ["Python"]

    def test_mixed_bullet_styles(self) -> None:
        text = "Responsibilities\n1. First\n2. Second\n3. Third\n"
        job = parse_job_description(text)
        assert job.responsibilities == ["First", "Second", "Third"]


class TestDegenerateInputs:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(ParsingError):
            parse_job_description("   \n\n  ")

    def test_unstructured_text_does_not_crash(self) -> None:
        job = parse_job_description("We are hiring engineers for our new platform.")
        assert job.title is None
        assert job.company is None
        assert job.responsibilities == []
        assert job.metadata.overall_confidence == ConfidenceLevel.LOW

    def test_minimal_job_description(self) -> None:
        job = parse_job_description(
            "Backend Engineer\nCompany: Acme\n\nSkills\nPython\n"
        )
        assert job.title == "Backend Engineer"
        assert job.company == "Acme"
        assert job.required_skills == ["Python"]
        assert job.metadata.overall_confidence == ConfidenceLevel.MEDIUM

    def test_no_invention_of_missing_information(self) -> None:
        job = parse_job_description("We need someone.\n\nOur Culture\nWe move fast.\n")
        assert job.company is None
        assert job.location is None
        assert job.certifications == []
        assert job.salary == []
        assert job.benefits == []
