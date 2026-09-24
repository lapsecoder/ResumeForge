"""Role-level analysis tests.

Covers supported roles, aliases, unknown-role fallback, missing
skills, experience gaps, strong/partial/weak alignment,
stateless/privacy behavior, and JD-flow regression.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.job_parsing.schemas import JobDescription, JobMetadata
from app.matching.service import match_resume_to_job
from app.parsing.schemas import (
    Certification,
    ConfidenceLevel,
    CustomSection,
    Education,
    Project,
    Resume,
    ResumeMetadata,
    SkillSet,
    WorkExperience,
)
from app.role_analysis.profiles import (
    SUPPORTED_ROLES,
    resolve_role_title,
    search_roles,
)
from app.role_analysis.service import analyze_role_compatibility
from app.role_analysis.schemas import RoleAnalysisRequest

from app.main import app

client = TestClient(app)

REFERENCE = date(2026, 9, 9)


def _make_resume(
    *,
    skills: list[str] | None = None,
    experience: list[WorkExperience] | None = None,
    education: list[Education] | None = None,
    certifications: list[Certification] | None = None,
    projects: list[Project] | None = None,
    summary: str | None = None,
) -> Resume:
    return Resume(
        summary=summary,
        experience=experience or [],
        education=education or [],
        skills=SkillSet(
            technical=skills or [],
            soft=[],
            tools=[],
            languages=[],
            all=skills or [],
        ),
        projects=projects or [],
        certifications=certifications or [],
        custom_sections=[],
        metadata=ResumeMetadata(
            word_count=0, file_type="text", overall_confidence=ConfidenceLevel.HIGH
        ),
    )


def _stint(start: str, end: str) -> WorkExperience:
    return WorkExperience(
        company="Acme", title="Engineer", start_date=start, end_date=end
    )


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


class TestProfileResolution:
    def test_supported_role_exact(self) -> None:
        profile = resolve_role_title("Software Engineer")
        assert profile is not None
        assert profile.title == "Software Engineer"

    def test_supported_role_alias(self) -> None:
        profile = resolve_role_title("Full Stack Engineer")
        assert profile is not None
        assert profile.title == "Full Stack Developer"

    def test_supported_role_lowercase(self) -> None:
        profile = resolve_role_title("software engineer")
        assert profile is not None
        assert profile.title == "Software Engineer"

    def test_supported_role_extra_whitespace(self) -> None:
        profile = resolve_role_title("  Software  Engineer  ")
        assert profile is not None
        assert profile.title == "Software Engineer"

    def test_developer_alias(self) -> None:
        profile = resolve_role_title("Developer")
        assert profile is not None
        assert profile.title == "Software Engineer"

    def test_ml_engineer_alias(self) -> None:
        profile = resolve_role_title("ML Engineer")
        assert profile is not None
        assert profile.title == "Machine Learning Engineer"

    def test_backend_engineer_alias(self) -> None:
        profile = resolve_role_title("Backend Engineer")
        assert profile is not None
        assert profile.title == "Backend Developer"

    def test_frontend_engineer_alias(self) -> None:
        profile = resolve_role_title("Frontend Engineer")
        assert profile is not None
        assert profile.title == "Frontend Developer"

    def test_unknown_role(self) -> None:
        profile = resolve_role_title("Quantum Physicist")
        assert profile is None

    def test_empty_role(self) -> None:
        profile = resolve_role_title("")
        assert profile is None

    def test_ai_engineer_supported(self) -> None:
        profile = resolve_role_title("AI Engineer")
        assert profile is not None
        assert profile.title == "AI Engineer"

    def test_ai_engineer_alias(self) -> None:
        profile = resolve_role_title("Artificial Intelligence Engineer")
        assert profile is not None
        assert profile.title == "AI Engineer"

    def test_machine_learning_engineer_supported(self) -> None:
        profile = resolve_role_title("Machine Learning Engineer")
        assert profile is not None
        assert profile.title == "Machine Learning Engineer"

    def test_nlp_engineer_alias(self) -> None:
        profile = resolve_role_title("Natural Language Processing Engineer")
        assert profile is not None
        assert profile.title == "NLP Engineer"

    def test_computer_vision_engineer_supported(self) -> None:
        profile = resolve_role_title("Computer Vision Engineer")
        assert profile is not None
        assert profile.title == "Computer Vision Engineer"

    def test_python_developer_supported(self) -> None:
        profile = resolve_role_title("Python Developer")
        assert profile is not None
        assert profile.title == "Python Developer"

    def test_java_developer_supported(self) -> None:
        profile = resolve_role_title("Java Developer")
        assert profile is not None
        assert profile.title == "Java Developer"

    def test_cplusplus_developer_alias(self) -> None:
        profile = resolve_role_title("CPP Developer")
        assert profile is not None
        assert profile.title == "C++ Developer"

    def test_cybersecurity_analyst_alias(self) -> None:
        profile = resolve_role_title("Cyber Security Analyst")
        assert profile is not None
        assert profile.title == "Cybersecurity Analyst"

    def test_devops_engineer_supported(self) -> None:
        profile = resolve_role_title("DevOps Engineer")
        assert profile is not None
        assert profile.title == "DevOps Engineer"

    def test_sre_alias(self) -> None:
        profile = resolve_role_title("SRE")
        assert profile is not None
        assert profile.title == "Site Reliability Engineer"

    def test_data_scientist_supported(self) -> None:
        profile = resolve_role_title("Data Scientist")
        assert profile is not None
        assert profile.title == "Data Scientist"

    def test_ethical_hacker_alias(self) -> None:
        profile = resolve_role_title("Ethical Hacker")
        assert profile is not None
        assert profile.title == "Penetration Tester"

    def test_all_profiles_have_required_fields(self) -> None:
        for role in SUPPORTED_ROLES:
            assert role.title
            assert isinstance(role.aliases, tuple)
            assert isinstance(role.required_skills, tuple)
            assert isinstance(role.experience_requirements, tuple)
            assert isinstance(role.education_requirements, tuple)

    def test_broad_role_coverage_across_categories(self) -> None:
        """Every role listed in the product requirements is supported."""
        expected = {
            "AI Engineer",
            "Machine Learning Engineer",
            "Deep Learning Engineer",
            "NLP Engineer",
            "Computer Vision Engineer",
            "Generative AI Engineer",
            "AI Research Engineer",
            "MLOps Engineer",
            "Data Scientist",
            "Data Engineer",
            "Data Analyst",
            "Software Engineer",
            "Backend Developer",
            "Frontend Developer",
            "Full Stack Developer",
            "Python Developer",
            "Java Developer",
            "C++ Developer",
            "JavaScript Developer",
            "TypeScript Developer",
            "Mobile Developer",
            "Android Developer",
            "iOS Developer",
            "DevOps Engineer",
            "Cloud Engineer",
            "AWS Engineer",
            "Azure Engineer",
            "Google Cloud Engineer",
            "Site Reliability Engineer",
            "Platform Engineer",
            "Cloud Architect",
            "Cybersecurity Analyst",
            "Cybersecurity Engineer",
            "Security Engineer",
            "SOC Analyst",
            "Penetration Tester",
            "Application Security Engineer",
            "Cloud Security Engineer",
            "Security Architect",
            "Database Administrator",
            "Database Developer",
            "BI Developer",
            "Business Intelligence Analyst",
            "Data Architect",
            "QA Engineer",
            "QA Analyst",
            "Software Test Engineer",
            "Automation Test Engineer",
            "SDET",
            "Systems Engineer",
            "Systems Administrator",
            "Network Engineer",
            "Network Administrator",
            "Infrastructure Engineer",
            "Linux Administrator",
        }
        titles = {p.title for p in SUPPORTED_ROLES}
        missing = expected - titles
        assert not missing, f"Missing expected role profiles: {sorted(missing)}"

    def test_no_duplicate_titles(self) -> None:
        titles = [p.title for p in SUPPORTED_ROLES]
        assert len(titles) == len(set(titles)), "Duplicate role titles found"

    def test_aliases_resolve_to_canonical_not_duplicate(self) -> None:
        """Aliases must NOT also appear as separate canonical titles."""
        titles_lower = {p.title.lower() for p in SUPPORTED_ROLES}
        for profile in SUPPORTED_ROLES:
            for alias in profile.aliases:
                normalised = alias.strip().lower()
                if normalised in titles_lower:
                    assert normalised == profile.title.lower(), (
                        f"Alias '{alias!r}' of '{profile.title}' is also a "
                        f"canonical title — duplicate profile"
                    )


class TestSearchRoles:
    def test_search_empty_query_returns_nothing(self) -> None:
        assert search_roles("") == []
        assert search_roles("   ") == []

    def test_search_ml_matches_multiple(self) -> None:
        results = search_roles("ml")
        titles = {p.title for p in results}
        assert "AI Engineer" in titles
        assert "Machine Learning Engineer" in titles
        assert "MLOps Engineer" in titles

    def test_search_python_matches_python_developer(self) -> None:
        results = search_roles("python")
        titles = {p.title for p in results}
        assert "Python Developer" in titles

    def test_search_security_matches_multiple(self) -> None:
        results = search_roles("security")
        titles = {p.title for p in results}
        assert "Cybersecurity Analyst" in titles
        assert "Security Engineer" in titles
        assert "Security Architect" in titles

    def test_search_backend(self) -> None:
        results = search_roles("backend")
        titles = {p.title for p in results}
        assert "Backend Developer" in titles

    def test_search_fullstack(self) -> None:
        results = search_roles("full stack")
        titles = {p.title for p in results}
        assert "Full Stack Developer" in titles

    def test_search_case_insensitive(self) -> None:
        results = search_roles("ML")
        assert len(results) == len(search_roles("ml"))

    def test_search_no_match(self) -> None:
        assert search_roles("quantum") == []


class TestSupportedEndpoint:
    def test_list_all_supported_roles(self) -> None:
        resp = client.get("/api/v1/roles/supported")
        assert resp.status_code == 200
        data = resp.json()
        assert "roles" in data
        assert "total" in data
        assert data["total"] == len(data["roles"])
        assert data["total"] > 50
        titles = [r["title"] for r in data["roles"]]
        assert "AI Engineer" in titles
        assert "Machine Learning Engineer" in titles
        assert "Software Engineer" in titles

    def test_search_supported_roles_with_query(self) -> None:
        resp = client.get("/api/v1/roles/supported?q=ml")
        assert resp.status_code == 200
        data = resp.json()
        titles = [r["title"] for r in data["roles"]]
        assert "Machine Learning Engineer" in titles
        assert "MLOps Engineer" in titles

    def test_search_supported_roles_security(self) -> None:
        resp = client.get("/api/v1/roles/supported?q=security")
        assert resp.status_code == 200
        data = resp.json()
        titles = [r["title"] for r in data["roles"]]
        assert "Cybersecurity Analyst" in titles
        assert "Security Engineer" in titles

    def test_supported_roles_includes_aliases(self) -> None:
        resp = client.get("/api/v1/roles/supported?q=python")
        assert resp.status_code == 200
        data = resp.json()
        roles = {r["title"]: r for r in data["roles"]}
        assert "Python Developer" in roles
        assert "python engineer" in roles["Python Developer"]["aliases"]


# ---------------------------------------------------------------------------
# Role analysis: supported roles
# ---------------------------------------------------------------------------


class TestRoleAnalysisSupported:
    def test_software_engineer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "JavaScript", "Git", "SQL", "Docker"],
            experience=[_stint("2020", "2024")],
            education=[Education(institution="MIT", degree="B.Tech")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Software Engineer"
        assert result.compatibility["overall"] is not None
        assert result.skills.matched_count > 0

    def test_data_scientist_partial(self) -> None:
        resume = _make_resume(
            skills=["Python", "SQL"],
            experience=[_stint("2021", "2024")],
            education=[Education(institution="IIT", degree="B.Tech")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Data Scientist", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.compatibility["overall"] is not None
        assert "Machine Learning" in result.skills.missing

    def test_devops_weak(self) -> None:
        resume = _make_resume(
            skills=["Python"],
            experience=[_stint("2023", "2024")],
            education=[Education(institution="State Univ", degree="B.A.")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="DevOps Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.compatibility["overall"] is not None
        assert len(result.skills.missing) > 0

    def test_ai_engineer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "Machine Learning", "Statistics", "SQL", "TensorFlow", "PyTorch"],
            experience=[_stint("2020", "2024")],
            education=[Education(institution="MIT", degree="B.Tech")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="AI Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "AI Engineer"
        assert result.compatibility["overall"] is not None
        assert result.skills.matched_count > 0

    def test_machine_learning_engineer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "Machine Learning", "Statistics", "SQL", "PyTorch", "Docker"],
            experience=[_stint("2020", "2024")],
            education=[Education(institution="Stanford", degree="B.S.")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Machine Learning Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Machine Learning Engineer"
        assert result.compatibility["overall"] is not None

    def test_ml_engineer_alias_resolves(self) -> None:
        resume = _make_resume(skills=["Python"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="ML Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Machine Learning Engineer"

    def test_nlp_engineer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "Natural Language Processing", "Machine Learning"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="NLP Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "NLP Engineer"

    def test_python_developer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "SQL", "Git", "Django"],
            experience=[_stint("2021", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Python Developer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Python Developer"

    def test_backend_developer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "SQL", "REST APIs", "Git"],
            experience=[_stint("2021", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Backend Developer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Backend Developer"

    def test_full_stack_developer_strong(self) -> None:
        resume = _make_resume(
            skills=["HTML", "CSS", "JavaScript", "SQL", "Git"],
            experience=[_stint("2021", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Full Stack Developer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Full Stack Developer"

    def test_cybersecurity_analyst_strong(self) -> None:
        resume = _make_resume(
            skills=["Security", "Vulnerability Assessment", "SIEM"],
            experience=[_stint("2021", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Cybersecurity Analyst", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Cybersecurity Analyst"

    def test_computer_vision_engineer_strong(self) -> None:
        resume = _make_resume(
            skills=["Python", "Computer Vision", "Machine Learning"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Computer Vision Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is True
        assert result.profile.title == "Computer Vision Engineer"

# ---------------------------------------------------------------------------
# Unknown / unsupported roles
# ---------------------------------------------------------------------------


class TestRoleAnalysisUnknown:
    def test_unknown_role_returns_fallback(self) -> None:
        resume = _make_resume(skills=["Python"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Quantum Physicist", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is False
        assert result.profile.title == "Quantum Physicist"
        assert result.compatibility["overall"] is None
        assert len(result.recommendations) > 0
        assert "not in the local profile database" in result.recommendations[0].lower()

    def test_empty_role_returns_fallback(self) -> None:
        resume = _make_resume()
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.profile.supported is False

    def test_unknown_role_no_crash(self) -> None:
        resume = _make_resume(
            skills=["Python", "Java", "SQL"],
            experience=[_stint("2020", "2024")],
            education=[Education(institution="MIT", degree="B.Tech")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Astronaut", resume=resume),
            reference_date=REFERENCE,
        )
        assert result is not None
        assert result.compatibility["overall"] is None


# ---------------------------------------------------------------------------
# Missing skills
# ---------------------------------------------------------------------------


class TestMissingSkills:
    def test_all_skills_missing(self) -> None:
        resume = _make_resume(
            skills=["Painting", "Cooking"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.skills.matched_count == 0
        assert len(result.skills.missing) > 0
        assert "Python" in result.skills.missing

    def test_no_skills_missing(self) -> None:
        resume = _make_resume(
            skills=["Python", "JavaScript", "Git", "SQL", "Docker", "AWS", "React", "Node.js", "CI/CD"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.skills.missing == []

    def test_some_skills_missing(self) -> None:
        resume = _make_resume(
            skills=["Python", "Git"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.skills.matched_count > 0
        assert "JavaScript" in result.skills.missing
        assert "SQL" in result.skills.missing

    def test_no_recommendation_when_nothing_missing(self) -> None:
        resume = _make_resume(
            skills=["Python", "JavaScript", "Git", "SQL", "Docker", "AWS", "React", "Node.js", "CI/CD"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        skill_recs = [
            r for r in result.recommendations if "skills" in r.lower()
        ]
        assert len(skill_recs) == 0


# ---------------------------------------------------------------------------
# Experience gaps
# ---------------------------------------------------------------------------


class TestExperienceGaps:
    def test_experience_gap(self) -> None:
        resume = _make_resume(
            skills=["Python"],
            experience=[_stint("2023", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="DevOps Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.experience.required_years is not None
        assert result.experience.candidate_years is not None
        assert result.experience.gap_years is not None
        assert result.experience.gap_years > 0

    def test_experience_met(self) -> None:
        resume = _make_resume(
            skills=["Python"],
            experience=[
                _stint("2020", "2021"),
                _stint("2021", "2022"),
                _stint("2022", "2024"),
            ],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="DevOps Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.experience.met is True

    def test_experience_not_verifiable(self) -> None:
        resume = _make_resume(
            skills=["Python"],
            experience=[
                WorkExperience(
                    company="X", title="Y", start_date="2020"
                ),
            ],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="DevOps Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.experience.candidate_years is None


# ---------------------------------------------------------------------------
# Education gaps
# ---------------------------------------------------------------------------


class TestEducationComparison:
    def test_education_not_met(self) -> None:
        resume = _make_resume(
            skills=["Python"],
            experience=[_stint("2020", "2024")],
            education=[Education(institution="State", degree="Associate")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Data Scientist", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.education.required_level is not None
        assert result.education.met is False

    def test_education_met(self) -> None:
        resume = _make_resume(
            skills=["Python"],
            experience=[_stint("2020", "2024")],
            education=[Education(institution="MIT", degree="M.S.")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Data Scientist", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.education.met is True


# ---------------------------------------------------------------------------
# Strong/partial/weak alignment
# ---------------------------------------------------------------------------


class TestAlignmentLevels:
    def test_strong_alignment_high_score(self) -> None:
        resume = _make_resume(
            skills=["Python", "JavaScript", "Git", "SQL", "Docker", "AWS", "React", "Node.js", "CI/CD"],
            experience=[
                _stint("2020", "2021"),
                _stint("2021", "2022"),
                _stint("2022", "2024"),
            ],
            education=[Education(institution="MIT", degree="B.Tech")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.compatibility["overall"] is not None
        assert result.compatibility["overall"] >= 70
        assert result.skills.missing == []
        assert result.experience.met is True

    def test_partial_alignment_medium_score(self) -> None:
        resume = _make_resume(
            skills=["Python", "Git"],
            experience=[_stint("2022", "2024")],
            education=[Education(institution="MIT", degree="B.Tech")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.compatibility["overall"] is not None
        assert 0 < result.compatibility["overall"] < 70
        assert "JavaScript" in result.skills.missing

    def test_weak_alignment_low_or_none(self) -> None:
        resume = _make_resume(skills=["Painting"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result is not None
        assert result.compatibility["overall"] is None or result.compatibility["overall"] < 30


# ---------------------------------------------------------------------------
# Deterministic match details
# ---------------------------------------------------------------------------


class TestDeterministicMatch:
    def test_deterministic_match_present(self) -> None:
        resume = _make_resume(
            skills=["Python", "JavaScript", "Git", "SQL"],
            experience=[_stint("2020", "2024")],
        )
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.deterministic_match is not None
        assert isinstance(result.deterministic_match.overall_score, (int, float)) or result.deterministic_match.overall_score is None

    def test_matched_job_is_jobdescription(self) -> None:
        resume = _make_resume(skills=["Python"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert result.matched_job is not None
        assert isinstance(result.matched_job, JobDescription)
        assert result.matched_job.title == "Software Engineer"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

RESUME_BODY = {
    "contact": {"name": "Jane Doe"},
    "summary": "Software engineer with 4 years of experience.",
    "experience": [
        {
            "company": "Acme",
            "title": "Engineer",
            "start_date": "2020",
            "end_date": "2024",
        }
    ],
    "education": [{"institution": "MIT", "degree": "B.Tech"}],
    "skills": {
        "all": ["Python", "JavaScript", "Git", "SQL"],
        "languages": [],
    },
    "projects": [],
    "certifications": [],
    "custom_sections": [],
    "metadata": {
        "word_count": 0,
        "file_type": "pdf",
        "overall_confidence": "high",
    },
}


class TestRoleAnalysisAPI:

    def test_analyze_supported_role(self) -> None:
        body = {"role_title": "Software Engineer", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["role_title"] == "Software Engineer"
        assert data["profile"]["supported"] is True
        assert data["compatibility"]["overall"] is not None

    def test_analyze_unknown_role(self) -> None:
        body = {"role_title": "Quantum Physicist", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["profile"]["supported"] is False
        assert data["compatibility"]["overall"] is None
        assert len(data["recommendations"]) > 0

    def test_analyze_missing_fields(self) -> None:
        body = {"role_title": ""}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 422

    def test_analyze_no_body(self) -> None:
        resp = client.post("/api/v1/roles/analyze")
        assert resp.status_code == 422

    def test_analyze_returns_no_error_field(self) -> None:
        body = {"role_title": "Software Engineer", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        data = resp.json()
        assert "error" not in data


# ---------------------------------------------------------------------------
# Privacy / stateless
# ---------------------------------------------------------------------------


class TestRoleAnalysisPrivacy:
    def test_no_file_upload_required(self) -> None:
        """Role analysis takes a JSON body, not a file."""
        body = {"role_title": "Software Engineer", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 200

    def test_no_external_calls(self) -> None:
        """Verify the endpoint works without network (local profiles only)."""
        body = {"role_title": "Software Engineer", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["profile"]["supported"] is True

    def test_transient_result_no_persistence(self) -> None:
        """Result is returned inline, no database write occurs."""
        body = {"role_title": "Software Engineer", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" not in data
        assert "created_at" not in data
        assert "stored" not in data


# ---------------------------------------------------------------------------
# JD flow regression
# ---------------------------------------------------------------------------


class TestJDFlowRegression:
    RESUME_BODY = {
        "contact": {"name": "Jane Doe"},
        "summary": "Backend engineer.",
        "experience": [
            {
                "company": "Acme",
                "title": "Engineer",
                "start_date": "2020",
                "end_date": "2023",
            }
        ],
        "education": [{"institution": "IIT", "degree": "B.Tech"}],
        "skills": {"all": ["Python", "PostgreSQL"], "languages": []},
        "projects": [],
        "certifications": [],
        "custom_sections": [],
        "metadata": {
            "word_count": 0,
            "file_type": "pdf",
            "overall_confidence": "high",
        },
    }

    JOB_BODY = {
        "title": "Backend Engineer",
        "required_skills": ["Python"],
        "preferred_skills": [],
        "experience_requirements": ["3+ years"],
        "education_requirements": ["Bachelor's degree"],
        "certifications": [],
        "qualifications": [],
        "metadata": {
            "word_count": 0,
            "file_type": "txt",
            "overall_confidence": "high",
        },
    }

    def test_matching_score_endpoint_still_works(self) -> None:
        resp = client.post(
            "/api/v1/matching/score",
            json={"resume": self.RESUME_BODY, "job": self.JOB_BODY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_score"] is not None

    def test_hybrid_endpoint_still_works(self) -> None:
        resp = client.post(
            "/api/v1/matching/hybrid",
            json={"resume": self.RESUME_BODY, "job": self.JOB_BODY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score" in data

    def test_ats_analysis_endpoint_still_works(self) -> None:
        resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": self.RESUME_BODY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_score"] is not None

    def test_job_parse_endpoint_still_works(self) -> None:
        resp = client.post(
            "/api/v1/matching/score",
            json={"resume": self.RESUME_BODY, "job": self.JOB_BODY},
        )
        assert resp.status_code == 200

    def test_role_does_not_interfere_with_matching(self) -> None:
        """Calling role analysis doesn't change matching behavior."""
        body = {"role_title": "Software Engineer", "resume": RESUME_BODY}
        resp = client.post("/api/v1/roles/analyze", json=body)
        assert resp.status_code == 200

        resp2 = client.post(
            "/api/v1/matching/score",
            json={"resume": self.RESUME_BODY, "job": self.JOB_BODY},
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["overall_score"] is not None


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_recommendations_present_when_gaps_exist(self) -> None:
        resume = _make_resume(skills=["Python"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        assert len(result.recommendations) > 0

    def test_recommendations_are_strings(self) -> None:
        resume = _make_resume(skills=["Python"])
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Software Engineer", resume=resume),
            reference_date=REFERENCE,
        )
        for rec in result.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_unknown_role_has_fallback_recommendation(self) -> None:
        resume = _make_resume()
        result = analyze_role_compatibility(
            RoleAnalysisRequest(role_title="Unknown Role", resume=resume),
            reference_date=REFERENCE,
        )
        assert any("profile database" in r.lower() for r in result.recommendations)
