"""Scenario evaluation tests for the Job-Specific ATS Coverage engine.

Each scenario is a concrete (resume, job) pair with a fully deterministic
expected score. Structural invariants (bounds, weight redistribution, category
non-applicability, match logging, finding rule IDs) are asserted for every
scenario.

Scenarios 24–28 in the project list (see test_job_specific_ats.py):
  24 perfect alignment       -> 100.0 Strong, no negative findings
  25 skills-only evidence    -> 50.0 Needs Improvement
  26 one missing required    -> 70.0 Good
  27 completely empty job    -> None / None with explanation
  28 Java vs JavaScript      -> 0.0 Weak, Java genuinely absent
"""

from __future__ import annotations

import pytest

from app.ats_analysis.job_specific.rules import CATEGORY_ORDER, WEIGHTS
from app.ats_analysis.job_specific.schemas import (
    JobSpecificATSResult,
    MatchType,
)
from app.ats_analysis.job_specific.service import analyze_job_specific_ats
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.parsing.schemas import ConfidenceLevel, WorkExperience
from tests.test_ats_analysis import make_resume, skill_set


def _job(
    *,
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    responsibilities: list[str] | None = None,
    certifications: list[str] | None = None,
    nice_to_have: list[str] | None = None,
) -> JobDescription:
    return JobDescription(
        required_skills=required or [],
        preferred_skills=preferred or [],
        responsibilities=responsibilities or [],
        certifications=certifications or [],
        nice_to_have=nice_to_have or [],
        metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
    )


def _aligned_resume():
    return make_resume(
        summary=None,
        skills=skill_set(technical=["Python", "Docker", "PostgreSQL"]),
        experience=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                achievements=[
                    "Build scalable backend microservices using Python and Docker.",
                    "Design data pipelines for analytics dashboards.",
                    "Deploy Kubernetes workloads and expose REST APIs.",
                    "Holds CKA certification.",
                ],
                skills_mentioned=["Python", "PostgreSQL", "Docker"],
            )
        ],
    )


def _skills_only_resume():
    return make_resume(
        summary=None,
        skills=skill_set(technical=["Python", "PostgreSQL", "Docker"]),
        experience=[],
    )


def _one_missing_resume():
    return make_resume(
        summary=None,
        skills=skill_set(technical=["Python", "Docker"]),
        experience=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                achievements=[
                    "Build Python services.",
                    "Deploy Linux containers.",
                ],
            )
        ],
    )


def _java_resume():
    return make_resume(
        summary="JavaScript developer.",
        skills=skill_set(technical=["JavaScript"]),
        experience=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                achievements=["Wrote JavaScript tooling."],
            )
        ],
    )


_SCENARIOS: dict[str, tuple[object, JobDescription]] = {
    "aligned": (
        _aligned_resume(),
        _job(
            required=["Python", "Docker", "PostgreSQL"],
            preferred=["Kubernetes", "REST APIs"],
            responsibilities=[
                "Build scalable backend microservices using Python and Docker.",
                "Design data pipelines for analytics.",
            ],
            certifications=["CKA"],
            nice_to_have=["Kubernetes"],
        ),
    ),
    "skills_only": (
        _skills_only_resume(),
        _job(
            required=["Python", "PostgreSQL", "Docker"],
            preferred=["Go", "Spark"],
            responsibilities=["Design data pipelines for analytics."],
        ),
    ),
    "one_missing": (
        _one_missing_resume(),
        _job(
            required=["Python", "Kafka"],
            preferred=["Docker"],
            responsibilities=["Deploy Linux containers."],
        ),
    ),
    "empty_job": (
        make_resume(skills=skill_set(technical=["Python"])),
        _job(),
    ),
    "java_vs_js": (
        _java_resume(),
        _job(required=["Java"]),
    ),
}


def _assert_structural(result: JobSpecificATSResult) -> None:
    for cs in result.category_scores:
        assert cs.key in CATEGORY_ORDER
        assert cs.max_weight == WEIGHTS[cs.key]
        if cs.applicable:
            assert cs.score is not None
            assert cs.weight == WEIGHTS[cs.key]
        else:
            assert cs.score is None
            assert cs.weight == 0.0
        if cs.score is not None:
            assert 0.0 <= cs.score <= 100.0

    assert set(result.metadata.applied_weights) == {
        cs.key for cs in result.category_scores if cs.applicable
    }
    for k, w in result.metadata.applied_weights.items():
        assert w == WEIGHTS[k]

    if result.overall_score is not None:
        assert 0.0 <= result.overall_score <= 100.0

    coverage = result.coverage
    for value in (
        coverage.required_coverage,
        coverage.preferred_coverage,
        coverage.overall_coverage,
        coverage.evidence_supported_required,
        coverage.evidence_supported_preferred,
    ):
        assert value is None or (0.0 <= value <= 1.0)

    for term in result.term_matches:
        assert term.term
        assert term.match_type in set(MatchType)


class TestScenarioAligned:
    EXPECTED = 100.0

    def test_expected_score(self) -> None:
        resume, job = _SCENARIOS["aligned"]
        result = analyze_job_specific_ats(resume, job)
        assert result.overall_score == self.EXPECTED
        assert result.score_label == "Strong"

    def test_no_negative_findings(self) -> None:
        resume, job = _SCENARIOS["aligned"]
        result = analyze_job_specific_ats(resume, job)
        assert not [f for f in result.findings if f.impact > 0]

    def test_structural_invariants(self) -> None:
        resume, job = _SCENARIOS["aligned"]
        _assert_structural(analyze_job_specific_ats(resume, job))


class TestScenarioSkillsOnly:
    EXPECTED = 50.0

    def test_expected_score(self) -> None:
        resume, job = _SCENARIOS["skills_only"]
        result = analyze_job_specific_ats(resume, job)
        assert result.overall_score == self.EXPECTED
        assert result.score_label == "Needs Improvement"

    def test_missing_preferred_is_penalised(self) -> None:
        resume, job = _SCENARIOS["skills_only"]
        result = analyze_job_specific_ats(resume, job)
        assert any(
            f.rule_id == "ats.job.preferred_coverage" and f.impact > 0
            for f in result.findings
        )

    def test_skills_only_flag_present(self) -> None:
        resume, job = _SCENARIOS["skills_only"]
        result = analyze_job_specific_ats(resume, job)
        assert any(f.rule_id == "ats.job.skills_only" for f in result.findings)

    def test_structural_invariants(self) -> None:
        resume, job = _SCENARIOS["skills_only"]
        _assert_structural(analyze_job_specific_ats(resume, job))


class TestScenarioOneMissing:
    EXPECTED = 70.0

    def test_expected_score(self) -> None:
        resume, job = _SCENARIOS["one_missing"]
        result = analyze_job_specific_ats(resume, job)
        assert result.overall_score == self.EXPECTED
        assert result.score_label == "Good"

    def test_required_missing_finding(self) -> None:
        resume, job = _SCENARIOS["one_missing"]
        result = analyze_job_specific_ats(resume, job)
        missing = [
            f for f in result.findings
            if f.rule_id == "ats.job.required_term_missing"
        ]
        assert missing and "Kafka" in missing[0].title

    def test_required_coverage_fraction(self) -> None:
        resume, job = _SCENARIOS["one_missing"]
        result = analyze_job_specific_ats(resume, job)
        assert result.coverage.required_coverage == 0.5

    def test_structural_invariants(self) -> None:
        resume, job = _SCENARIOS["one_missing"]
        _assert_structural(analyze_job_specific_ats(resume, job))


class TestScenarioEmptyJob:
    def test_overall_is_none_with_explanation(self) -> None:
        resume, job = _SCENARIOS["empty_job"]
        result = analyze_job_specific_ats(resume, job)
        assert result.overall_score is None
        assert result.score_label is None
        assert result.metadata.applied_weights == {}
        assert any(
            f.rule_id == "ats.job.limited_terminology" for f in result.findings
        )

    def test_structural_invariants(self) -> None:
        resume, job = _SCENARIOS["empty_job"]
        _assert_structural(analyze_job_specific_ats(resume, job))


class TestScenarioJavaVsJs:
    EXPECTED = 0.0

    def test_expected_score(self) -> None:
        resume, job = _SCENARIOS["java_vs_js"]
        result = analyze_job_specific_ats(resume, job)
        assert result.overall_score == self.EXPECTED
        assert result.score_label == "Weak"

    def test_java_is_reported_absent(self) -> None:
        resume, job = _SCENARIOS["java_vs_js"]
        result = analyze_job_specific_ats(resume, job)
        java = next(m for m in result.term_matches if m.term == "Java")
        assert java.match_type is MatchType.ABSENT
        assert java.evidence_locations == []

    def test_structural_invariants(self) -> None:
        resume, job = _SCENARIOS["java_vs_js"]
        _assert_structural(analyze_job_specific_ats(resume, job))


class TestCrossScenarioInvariants:
    def test_determinism(self) -> None:
        resume, job = _SCENARIOS["aligned"]
        assert analyze_job_specific_ats(resume, job) == analyze_job_specific_ats(
            resume, job
        )

    def test_weights_sum_to_100(self) -> None:
        assert sum(WEIGHTS.values()) == pytest.approx(100.0)

    def test_findings_reference_documented_rule_ids(self) -> None:
        known = {
            "ats.job.required_term_missing",
            "ats.job.preferred_term_missing",
            "ats.job.phrase_missing",
            "ats.job.skills_only",
            "ats.job.evidence_present",
            "ats.job.required_coverage",
            "ats.job.preferred_coverage",
            "ats.job.phrase_coverage",
            "ats.job.evidence_representation",
            "ats.job.limited_terminology",
        }
        for name in ("skills_only", "one_missing"):
            resume, job = _SCENARIOS[name]
            result = analyze_job_specific_ats(resume, job)
            for finding in result.findings:
                assert finding.rule_id in known

    def test_every_deduction_has_aggregate_finding(self) -> None:
        resume, job = _SCENARIOS["skills_only"]
        result = analyze_job_specific_ats(resume, job)
        impact_by_rule = {f.rule_id: f.impact for f in result.findings if f.impact > 0}
        aggregate = (
            impact_by_rule.get("ats.job.required_coverage", 0.0)
            + impact_by_rule.get("ats.job.preferred_coverage", 0.0)
            + impact_by_rule.get("ats.job.phrase_coverage", 0.0)
            + impact_by_rule.get("ats.job.evidence_representation", 0.0)
        )
        assert aggregate == pytest.approx(100.0 - result.overall_score)
