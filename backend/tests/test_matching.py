"""Deterministic matching unit tests (skills, experience, education,
qualifications, overall score, and explanations).

All fixtures are synthetic. A fixed ``reference_date`` is used so every
calculation is deterministic and traceable.
"""

from __future__ import annotations

from datetime import date

import pytest

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

REFERENCE = date(2026, 9, 9)


def _resume_metadata() -> ResumeMetadata:
    return ResumeMetadata(
        word_count=0, file_type="pdf", overall_confidence=ConfidenceLevel.HIGH
    )


def make_resume(
    *,
    skills: list[str] | None = None,
    languages: list[str] | None = None,
    experience: list[WorkExperience] | None = None,
    education: list[Education] | None = None,
    certifications: list[Certification] | None = None,
    projects: list[Project] | None = None,
    summary: str | None = None,
    custom_sections: list[CustomSection] | None = None,
) -> Resume:
    return Resume(
        summary=summary,
        experience=experience or [],
        education=education or [],
        skills=SkillSet(all=skills or [], languages=languages or []),
        projects=projects or [],
        certifications=certifications or [],
        custom_sections=custom_sections or [],
        metadata=_resume_metadata(),
    )


def make_job(
    *,
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    experience_requirements: list[str] | None = None,
    education_requirements: list[str] | None = None,
    certifications: list[str] | None = None,
    qualifications: list[str] | None = None,
) -> JobDescription:
    return JobDescription(
        required_skills=required or [],
        preferred_skills=preferred or [],
        experience_requirements=experience_requirements or [],
        education_requirements=education_requirements or [],
        certifications=certifications or [],
        qualifications=qualifications or [],
        metadata=JobMetadata(
            word_count=0, overall_confidence=ConfidenceLevel.HIGH
        ),
    )


def match(resume: Resume, job: JobDescription):
    return match_resume_to_job(resume, job, reference_date=REFERENCE)


def stint(start: str, end: str) -> WorkExperience:
    return WorkExperience(
        company="Acme", title="Engineer", start_date=start, end_date=end
    )


class TestSkillNormalization:
    def test_exact_match(self) -> None:
        result = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python"]),
        )
        assert result.skill_match.matched_required == ["Python"]
        assert result.skill_match.required_coverage == 1.0

    def test_case_insensitive(self) -> None:
        result = match(
            make_resume(skills=["PYTHON", "Go"]),
            make_job(required=["python", "golang"]),
        )
        assert result.skill_match.required_coverage == 1.0

    def test_whitespace_variations(self) -> None:
        result = match(
            make_resume(skills=["  Python  "]),
            make_job(required=["python", "python"]),
        )
        assert result.skill_match.matched_required == ["python"]
        assert result.skill_match.required_coverage == 1.0

    def test_trailing_punctuation_stripped(self) -> None:
        result = match(
            make_resume(skills=["Docker"]),
            make_job(required=["docker."]),
        )
        assert result.skill_match.required_coverage == 1.0

    def test_meaningful_punctuation_preserved(self) -> None:
        assert match(
            make_resume(skills=["C++"]), make_job(required=["C++"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["C"]), make_job(required=["C++"])
        ).skill_match.required_coverage == 0.0
        assert match(
            make_resume(skills=["C#"]), make_job(required=["C#"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=[".NET"]), make_job(required=[".NET"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["Node.js"]), make_job(required=["node.js"])
        ).skill_match.required_coverage == 1.0

    def test_documented_aliases(self) -> None:
        assert match(
            make_resume(skills=["JS"]), make_job(required=["javascript"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["python"]), make_job(required=["py"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["TS"]), make_job(required=["typescript"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["postgresql"]), make_job(required=["postgres"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["k8s"]), make_job(required=["kubernetes"])
        ).skill_match.required_coverage == 1.0
        assert match(
            make_resume(skills=["go"]), make_job(required=["golang"])
        ).skill_match.required_coverage == 1.0

    def test_no_false_positive_equivalences(self) -> None:
        assert match(
            make_resume(skills=["machine learning"]),
            make_job(required=["deep learning"]),
        ).skill_match.required_coverage == 0.0
        assert match(
            make_resume(skills=["java"]), make_job(required=["javascript"])
        ).skill_match.required_coverage == 0.0
        assert match(
            make_resume(skills=["aws"]), make_job(required=["azure"])
        ).skill_match.required_coverage == 0.0


class TestSkillMatching:
    def test_required_vs_preferred_separated(self) -> None:
        result = match(
            make_resume(skills=["Python", "Go"]),
            make_job(required=["Python"], preferred=["Go"]),
        )
        assert result.skill_match.matched_required == ["Python"]
        assert result.skill_match.matched_preferred == ["Go"]

    def test_missing_required_skills(self) -> None:
        result = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python", "Docker"]),
        )
        assert result.skill_match.missing_required == ["Docker"]
        assert result.skill_match.required_coverage == 0.5

    def test_missing_preferred_only(self) -> None:
        result = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python"], preferred=["Kubernetes"]),
        )
        assert result.skill_match.missing_preferred == ["Kubernetes"]
        assert result.skill_match.preferred_coverage == 0.0

    def test_duplicate_requirements_counted_once(self) -> None:
        result = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python", "Python", " python "]),
        )
        assert result.skill_match.total_required == 1
        assert result.skill_match.required_coverage == 1.0

    def test_no_skill_requirements(self) -> None:
        result = match(make_resume(skills=["Python"]), make_job())
        assert result.skill_match.score is None
        assert result.skill_match.total_required == 0
        assert result.skill_match.total_preferred == 0
        assert any("no skill requirements" in n for n in result.skill_match.notes)

    def test_preferred_only_composite(self) -> None:
        result = match(
            make_resume(skills=["Python", "Docker"]),
            make_job(required=[], preferred=["Python", "Docker", "Kafka"]),
        )
        assert result.skill_match.preferred_coverage == pytest.approx(2 / 3)
        assert result.skill_match.score == pytest.approx(
            round(2 / 3 * 100, 1)
        )

    def test_composite_score_mixes_required_and_preferred(self) -> None:
        result = match(
            make_resume(skills=["Python", "Go", "Docker"]),
            make_job(required=["Python", "Go"], preferred=["Docker", "Kafka"]),
        )
        assert result.skill_match.required_coverage == 1.0
        assert result.skill_match.preferred_coverage == 0.5
        expected = round(
            (50.0 * 1.0 + 15.0 * 0.5) / 65.0 * 100, 1
        )
        assert result.skill_match.score == pytest.approx(expected)

    def test_skills_collected_from_all_resume_buckets(self) -> None:
        resume = Resume(
            skills=SkillSet(
                all=["python"],
                technical=["go"],
                tools=["docker"],
                soft=["leadership"],
                languages=["english"],
            ),
            projects=[Project(name="P", technologies=["kafka"])],
            experience=[stint("2020", "2023")],
            metadata=_resume_metadata(),
        )
        resume.experience[0].skills_mentioned.append("kubernetes")
        result = match(
            resume,
            make_job(
                required=[
                    "python",
                    "go",
                    "docker",
                    "leadership",
                    "english",
                    "kafka",
                    "k8s",
                ]
            ),
        )
        assert result.skill_match.required_coverage == 1.0


class TestExperienceMatching:
    def test_full_year_ranges(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2023")]),
            make_job(experience_requirements=["3+ years"]),
        )
        assert result.experience_match.candidate_years == 3.0
        assert result.experience_match.required_years == 3.0
        assert result.experience_match.score == 100.0

    def test_month_year_ranges(self) -> None:
        result = match(
            make_resume(experience=[stint("Jan 2020", "Dec 2022")]),
            make_job(experience_requirements=["3+ years"]),
        )
        assert result.experience_match.candidate_years == pytest.approx(2.9)
        assert result.experience_match.score == pytest.approx(
            round(2.9 / 3 * 100, 1)
        )

    def test_mm_yyyy_ranges(self) -> None:
        result = match(
            make_resume(experience=[stint("01/2020", "12/2022")]),
            make_job(experience_requirements=["2+ years"]),
        )
        assert result.experience_match.candidate_years == pytest.approx(2.9)
        assert result.experience_match.score == 100.0

    def test_present_resolved_to_reference_date(self) -> None:
        result = match(
            make_resume(experience=[stint("01/2024", "Present")]),
            make_job(experience_requirements=["2+ years"]),
        )
        assert result.experience_match.candidate_years == pytest.approx(2.7)
        assert result.experience_match.score == 100.0
        assert result.metadata.reference_date == REFERENCE

    def test_no_experience_requirement(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2023")]),
            make_job(),
        )
        assert result.experience_match.score is None
        assert result.experience_match.required_years is None
        assert "no explicit experience requirement" in (
            result.experience_match.assessment
        )

    def test_insufficient_dates_is_unknown_not_zero(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", None)]),
            make_job(experience_requirements=["2+ years"]),
        )
        assert result.experience_match.candidate_years is None
        assert result.experience_match.score is None
        assert result.experience_match.roles_measured == 0
        assert result.experience_match.roles_total == 1
        assert "insufficient" in result.experience_match.assessment

    def test_exact_requirement_met(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2023")]),
            make_job(experience_requirements=["3 years"]),
        )
        assert result.experience_match.score == 100.0

    def test_candidate_exceeds_requirement(self) -> None:
        result = match(
            make_resume(experience=[stint("2019", "2025")]),
            make_job(experience_requirements=["3+ years"]),
        )
        assert result.experience_match.candidate_years == pytest.approx(6.0)
        assert result.experience_match.score == 100.0

    def test_candidate_below_requirement(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2021")]),
            make_job(experience_requirements=["4+ years"]),
        )
        assert result.experience_match.candidate_years == pytest.approx(1.0)
        assert result.experience_match.score == pytest.approx(25.0)

    def test_range_requirement(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2023")]),
            make_job(experience_requirements=["2-5 years"]),
        )
        assert result.experience_match.required_years == 2.0
        assert result.experience_match.required_max_years == 5.0
        assert result.experience_match.score == 100.0

    def test_level_only_requirement_not_quantified(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2023")]),
            make_job(experience_requirements=["Senior-level engineer"]),
        )
        assert result.experience_match.required_level == "senior"
        assert result.experience_match.required_years is None
        assert result.experience_match.score is None
        assert "not quantitatively scored" in result.experience_match.assessment

    def test_unparseable_dates_excluded(self) -> None:
        result = match(
            make_resume(
                experience=[stint("unknown", "2023"), stint("2020", "2023")]
            ),
            make_job(experience_requirements=["2+ years"]),
        )
        assert result.experience_match.roles_total == 2
        assert result.experience_match.roles_measured == 1
        assert result.experience_match.candidate_years == 3.0


class TestEducationMatching:
    def test_bachelor_matches_btech(self) -> None:
        result = match(
            make_resume(education=[Education(institution="IIT", degree="B.Tech")]),
            make_job(education_requirements=["Bachelor's degree in CS"]),
        )
        assert result.education_match.matched is True
        assert result.education_match.score == 100.0

    def test_bachelor_matches_be(self) -> None:
        result = match(
            make_resume(education=[Education(institution="IIT", degree="B.E.")]),
            make_job(education_requirements=["Bachelor's degree"]),
        )
        assert result.education_match.matched is True

    def test_bsc_is_bachelor(self) -> None:
        result = match(
            make_resume(education=[Education(institution="College", degree="B.Sc")]),
            make_job(education_requirements=["Bachelor's degree"]),
        )
        assert result.education_match.matched is True

    def test_master_matches_mtech_and_mca(self) -> None:
        for degree in ("M.Tech", "MCA", "M.Sc", "M.A."):
            resume = make_resume(
                education=[Education(institution="College", degree=degree)]
            )
            result = match(
                resume,
                make_job(education_requirements=["Master's degree"]),
            )
            assert result.education_match.matched is True, degree

    def test_master_satisfies_bachelor(self) -> None:
        result = match(
            make_resume(education=[Education(institution="College", degree="M.Tech")]),
            make_job(education_requirements=["Bachelor's degree"]),
        )
        assert result.education_match.matched is True

    def test_diploma_below_bachelor(self) -> None:
        result = match(
            make_resume(education=[Education(institution="Poly", degree="Diploma")]),
            make_job(education_requirements=["Bachelor's degree"]),
        )
        assert result.education_match.matched is False
        assert result.education_match.score == 0.0

    def test_unrelated_degrees_do_not_match(self) -> None:
        result = match(
            make_resume(education=[Education(institution="Inst", degree="BBA")]),
            make_job(education_requirements=["Ph.D. in Physics"]),
        )
        assert result.education_match.matched is False

    def test_no_education_requirement(self) -> None:
        result = match(
            make_resume(education=[Education(institution="IIT", degree="B.Tech")]),
            make_job(),
        )
        assert result.education_match.score is None
        assert result.education_match.matched is None
        assert "not evaluated" in result.education_match.assessment

    def test_missing_education_data(self) -> None:
        result = match(
            make_resume(education=[]),
            make_job(education_requirements=["Bachelor's degree"]),
        )
        assert result.education_match.matched is False
        assert result.education_match.score == 0.0
        assert "no recognisable degrees" in result.education_match.assessment


class TestQualificationMatching:
    def test_certification_exact_match(self) -> None:
        result = match(
            make_resume(
                certifications=[Certification(name="AWS Certified Solutions Architect")]
            ),
            make_job(certifications=["aws certified solutions architect"]),
        )
        assert (
            result.qualification_match.matched
            == ["aws certified solutions architect"]
        )
        assert result.qualification_match.score == 100.0

    def test_certification_substring_match(self) -> None:
        result = match(
            make_resume(certifications=[Certification(name="PMP")]),
            make_job(certifications=["PMP Certification"]),
        )
        assert result.qualification_match.matched == ["PMP Certification"]
        assert result.qualification_match.score == 100.0

    def test_missing_certification_is_unmet(self) -> None:
        result = match(
            make_resume(certifications=[Certification(name="PMP")]),
            make_job(certifications=["CISSP"]),
        )
        assert result.qualification_match.unmet == ["CISSP"]
        assert result.qualification_match.score == 0.0

    def test_language_requirement_matched(self) -> None:
        result = match(
            make_resume(languages=["Spanish"]),
            make_job(qualifications=["Fluency in Spanish"]),
        )
        assert result.qualification_match.matched == ["Fluency in Spanish"]

    def test_language_not_listed_is_unknown(self) -> None:
        result = match(
            make_resume(languages=["English"]),
            make_job(qualifications=["Fluency in Spanish"]),
        )
        assert result.qualification_match.matched == []
        assert result.qualification_match.unknown == ["Fluency in Spanish"]
        assert result.qualification_match.score is None

    def test_generic_qualification_matched_by_tokens(self) -> None:
        result = match(
            make_resume(summary="Experienced with containerization at scale."),
            make_job(qualifications=["containerization"]),
        )
        assert result.qualification_match.matched == ["containerization"]
        assert result.qualification_match.score == 100.0

    def test_generic_qualification_unverifiable_is_unknown(self) -> None:
        result = match(
            make_resume(),
            make_job(qualifications=["Valid work authorization"]),
        )
        assert result.qualification_match.matched == []
        assert result.qualification_match.unknown == ["Valid work authorization"]
        assert result.qualification_match.score is None
        assert "evidence limitation" in result.qualification_match.assessment

    def test_mixed_matched_and_unmet(self) -> None:
        result = match(
            make_resume(certifications=[Certification(name="PMP")]),
            make_job(certifications=["PMP", "AWS Certified Developer"]),
        )
        assert len(result.qualification_match.matched) == 1
        assert len(result.qualification_match.unmet) == 1
        assert result.qualification_match.score == 50.0

    def test_no_qualification_requirements(self) -> None:
        result = match(make_resume(), make_job())
        assert result.qualification_match.score is None
        assert result.qualification_match.total == 0
        assert "no explicit qualification requirements" in (
            result.qualification_match.assessment
        )


class TestOverallScore:
    def test_zero_requirements_gives_no_overall_score(self) -> None:
        result = match(make_resume(), make_job())
        assert result.overall_score is None
        assert result.metadata.applied_weights == {}
        assert result.metadata.weights["required_skill"] == 50.0

    def test_all_components_max_out_at_100(self) -> None:
        resume = make_resume(
            skills=["Python", "Docker", "Kubernetes"],
            experience=[stint("2019", "2025")],
            education=[Education(institution="IIT", degree="B.Tech")],
            certifications=[Certification(name="PMP")],
        )
        job = make_job(
            required=["Python", "Docker"],
            preferred=["Kubernetes"],
            experience_requirements=["3+ years"],
            education_requirements=["Bachelor's degree"],
            certifications=["PMP"],
        )
        result = match(resume, job)
        assert result.overall_score == 100.0

    def test_missing_required_skill_reduces_overall(self) -> None:
        full = match(
            make_resume(skills=["Python", "Go"]),
            make_job(required=["Python", "Go"]),
        )
        partial = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python", "Go"]),
        )
        assert partial.overall_score == pytest.approx(50.0)
        assert full.overall_score == pytest.approx(100.0)
        assert partial.overall_score < full.overall_score  # type: ignore[operator]

    def test_missing_required_hurts_more_than_missing_preferred(self) -> None:
        missing_required = match(
            make_resume(skills=["Go", "Docker"]),
            make_job(required=["Python", "Go"], preferred=["Docker", "Kafka"]),
        )
        missing_preferred = match(
            make_resume(skills=["Python", "Go", "Kafka"]),
            make_job(required=["Python", "Go"], preferred=["Docker", "Kafka"]),
        )
        assert missing_required.overall_score < missing_preferred.overall_score  # type: ignore[operator]

    def test_preferred_skill_only_slightly_reduces_overall(self) -> None:
        result = match(
            make_resume(skills=["Python", "Go"]),
            make_job(required=["Python", "Go"], preferred=["Kafka"]),
        )
        applied = result.metadata.applied_weights
        assert applied.get("preferred_skill") == 15.0
        expected = round((50.0 * 100.0 + 15.0 * 0.0) / 65.0, 1)
        assert result.overall_score == pytest.approx(expected)

    def test_reweighted_formula_excludes_absent_criteria(self) -> None:
        result = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python"]),
        )
        assert result.metadata.applied_weights == {"required_skill": 50.0}
        assert result.overall_score == 100.0

    def test_unrelated_skills_do_not_inflate_score(self) -> None:
        result = match(
            make_resume(skills=["Photography", "Cooking", "Magic"]),
            make_job(required=["Python"]),
        )
        assert result.skill_match.matched_required == []
        assert result.overall_score == 0.0

    def test_no_nan_or_division_by_zero(self) -> None:
        result = match(make_resume(skills=["Python"]), make_job())
        assert result.overall_score is None
        assert not any(
            v != v for v in result.metadata.weights.values()
        )
        result = match(
            make_resume(skills=["Python"]),
            make_job(required=["Python"]),
        )
        assert isinstance(result.overall_score, float)
        assert 0.0 <= result.overall_score <= 100.0


class TestExplanations:
    def test_strengths_and_gaps_are_traceable(self) -> None:
        result = match(
            make_resume(skills=["Python"], experience=[stint("2020", "2023")]),
            make_job(
                required=["Python", "Docker"],
                experience_requirements=["4+ years"],
            ),
        )
        assert "Matches 1 of 2 required skills." in result.strengths
        assert "Missing required skill: Docker." in result.gaps
        assert "Required skill missing: Docker" in result.unmet_requirements
        assert "Required skill matched: Python" in result.matched_requirements

    def test_experience_shortfall_is_explained(self) -> None:
        result = match(
            make_resume(experience=[stint("2020", "2021")]),
            make_job(experience_requirements=["4+ years"]),
        )
        assert any("Experience below requirement" in g for g in result.gaps)
        assert any(
            "Experience requirement not met" in u
            for u in result.unmet_requirements
        )

    def test_unverifiable_qualification_is_not_a_matched_claim(self) -> None:
        job = make_job(
            certifications=["CISSP"],
            qualifications=["Valid work authorization"],
        )
        result = match(make_resume(), job)
        assert "CISSP" in " ".join(result.unmet_requirements)
        assert not any("work authorization" in m for m in result.matched_requirements)
        assert any("Could not verify" in g for g in result.gaps)

    def test_no_invented_evidence(self) -> None:
        result = match(
            make_resume(skills=["Python"]),
            make_job(certifications=["CISSP"]),
        )
        assert all("CISSP" not in s for s in result.strengths)
        assert all("CISSP" not in m for m in result.matched_requirements)
