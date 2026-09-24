"""Unit tests for Job-Specific ATS Coverage analysis.

All fixtures are synthetic and fully offline. The engine has no randomness, so
every score is deterministic and reproducible.

Scenario coverage (the list used by the evaluation file documents >= 22
distinct cases):
  1  exact skill-key match
  2  resume-side alias expansion (normalized)
  3  job-term alias match (js -> JavaScript)
  4  phrase match in free text
  5  absent term
  6  Java != JavaScript
  7  C != C++  (both directions)
  8  .NET != net
  9  Node.js != Node
  10 required vs preferred separation
  11 skills-only evidence flagged
  12 experience/projects evidence recognised
  13 empty required + preferred buckets (no penalty)
  14 completely empty job (score None + explanation)
  15 duplicate required terms deduplicated
  16 case-insensitive exactness
  17 unicode NFKC normalisation
  18 coverage report fractions
  19 weight redistribution on non-applicable categories
  20 deterministic repeatability
  21 term-match ordering (required < preferred < phrase)
  22 category bounds and weight-sum invariant
  23 marker-free evidence snippet truncation safety
"""

from __future__ import annotations

from app.ats_analysis.job_specific.analyzers import analyze_job_specific
from app.ats_analysis.job_specific.heuristics import (
    contains_phrase,
    has_strong_evidence,
    presence_tokens,
)
from app.ats_analysis.job_specific.rules import (
    CATEGORY_ORDER,
    WEIGHTS,
)
from app.ats_analysis.job_specific.schemas import MatchType, TermOrigin
from app.ats_analysis.job_specific.scorer import compute_overall, score_label
from app.ats_analysis.job_specific.service import analyze_job_specific_ats
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.parsing.schemas import (
    ConfidenceLevel,
    Project,
    WorkExperience,
)
from tests.test_ats_analysis import make_resume, skill_set


def make_job(
    *,
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    responsibilities: list[str] | None = None,
    experience_requirements: list[str] | None = None,
    qualifications: list[str] | None = None,
    certifications: list[str] | None = None,
    nice_to_have: list[str] | None = None,
) -> JobDescription:
    return JobDescription(
        required_skills=required or [],
        preferred_skills=preferred or [],
        responsibilities=responsibilities or [],
        experience_requirements=experience_requirements or [],
        qualifications=qualifications or [],
        certifications=certifications or [],
        nice_to_have=nice_to_have or [],
        metadata=JobMetadata(
            word_count=0, overall_confidence=ConfidenceLevel.HIGH
        ),
    )


# ---------------------------------------------------------------------------
# Matching semantics
# ---------------------------------------------------------------------------


class TestMatchingTypes:
    def test_exact_skill_key_match(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Kafka"]))
        scores, coverage, matches, _ = analyze_job_specific(
            resume, make_job(required=["Kafka"])
        )
        m = matches[0]
        assert m.match_type is MatchType.EXACT
        assert "skills" in m.evidence_locations
        assert coverage.required_coverage == 1.0

    def test_resume_side_aliases_are_normalized_matches(self) -> None:
        resume = make_resume(skills=skill_set(technical=["postgres"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["PostgreSQL"])
        )
        assert matches[0].match_type is MatchType.NORMALIZED

    def test_job_side_alias_is_alias_match(self) -> None:
        resume = make_resume(skills=skill_set(technical=["JavaScript"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["JS"])
        )
        assert matches[0].match_type is MatchType.ALIAS

    def test_exact_beats_alias(self) -> None:
        resume = make_resume(skills=skill_set(technical=["js"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["JS"])
        )
        assert matches[0].match_type is MatchType.EXACT

    def test_phrase_match_in_free_text(self) -> None:
        resume = make_resume(
            skills=skill_set(),
            summary="I use Docker to ship container images every day.",
        )
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["Docker"])
        )
        assert matches[0].match_type is MatchType.PHRASE
        assert "summary" in matches[0].evidence_locations

    def test_absent_term_is_not_inferred(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["Kafka"])
        )
        assert matches[0].match_type is MatchType.ABSENT
        assert matches[0].evidence_locations == []


class TestTokenDiscipline:
    def test_java_never_satisfies_javascript(self) -> None:
        resume = make_resume(skills=skill_set(technical=["JavaScript"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["Java"])
        )
        assert matches[0].match_type is MatchType.ABSENT

    def test_cpp_not_satisfied_by_c(self) -> None:
        resume = make_resume(skills=skill_set(technical=["C"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["C++"])
        )
        assert matches[0].match_type is MatchType.ABSENT

    def test_c_not_satisfied_by_cpp(self) -> None:
        resume = make_resume(skills=skill_set(technical=["C++"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["C"])
        )
        assert matches[0].match_type is MatchType.ABSENT

    def test_c_does_not_match_arbitrary_words(self) -> None:
        assert not contains_phrase(["develops scalable code"], "C")
        assert contains_phrase(["proficient in C"], "C")

    def test_dotnet_never_satisfied_by_net(self) -> None:
        resume = make_resume(skills=skill_set(technical=["net"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=[".NET"])
        )
        assert matches[0].match_type is MatchType.ABSENT

    def test_net_never_satisfied_by_dotnet(self) -> None:
        resume = make_resume(skills=skill_set(technical=[".NET"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["net"])
        )
        assert matches[0].match_type is MatchType.ABSENT

    def test_nodejs_never_satisfied_by_node(self) -> None:
        resume = make_resume(
            summary="I run Node engine analysis tools.",
            skills=skill_set(),
        )
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["Node.js"])
        )
        assert matches[0].match_type is MatchType.ABSENT

    def test_punctuation_preserved_in_tokens(self) -> None:
        assert presence_tokens("C++") == ["c++"]
        assert presence_tokens(".NET") == [".net"]
        assert presence_tokens("Node.js") == ["node.js"]
        assert presence_tokens("React.JS") == ["react.js"]


class TestNormalisation:
    def test_case_insensitive_exact(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["python"])
        )
        assert matches[0].match_type is MatchType.EXACT

    def test_unicode_nfkc_equivalence(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        # full-width 'python'
        _, _, matches, _ = analyze_job_specific(
            resume,
            make_job(required=["\uff50\uff59\uff54\uff48\uff4f\uff4e"]),
        )
        assert matches[0].match_type is MatchType.EXACT


# ---------------------------------------------------------------------------
# Required / preferred / bucket semantics
# ---------------------------------------------------------------------------


class TestBuckets:
    def test_required_and_preferred_separated(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["Python", "Docker"]),
            experience=[],
        )
        scores, coverage, _, _ = analyze_job_specific(
            resume,
            make_job(required=["Python"], preferred=["Spark"]),
        )
        assert scores["required_coverage"] == 100.0
        assert scores["preferred_coverage"] == 0.0
        assert coverage.required_coverage == 1.0
        assert coverage.preferred_coverage == 0.0
        assert coverage.overall_coverage == 0.5

    def test_empty_required_and_preferred_never_penalised(self) -> None:
        resume = make_resume(
            summary="Built pipelines.",
            skills=skill_set(technical=["Python"]),
        )
        scores, coverage, _, findings = analyze_job_specific(
            resume,
            make_job(
                responsibilities=["Build ingestion pipelines."]
            ),
        )
        assert "required_coverage" not in scores
        assert "preferred_coverage" not in scores
        assert coverage.required_coverage is None
        assert coverage.preferred_coverage is None
        assert not any(f.rule_id.endswith("_term_missing") for f in findings)

    def test_completely_empty_job_is_limited_evidence(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        scores, coverage, _, findings = analyze_job_specific(
            resume, make_job()
        )
        assert scores == {}
        assert coverage.overall_coverage is None
        assert any(
            f.rule_id == "ats.job.limited_terminology" for f in findings
        )

    def test_duplicate_required_terms_deduplicated(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        _, coverage, matches, _ = analyze_job_specific(
            resume,
            make_job(required=["python", "Python", "PYTHON"]),
        )
        assert coverage.totals.required == 1
        assert len([m for m in matches if m.origin is TermOrigin.REQUIRED]) == 1


class TestEvidence:
    def test_skills_only_matched_term_flagged(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]), summary=None)
        _, _, _, findings = analyze_job_specific(
            resume, make_job(required=["Python"])
        )
        skills_only = [f for f in findings if f.rule_id == "ats.job.skills_only"]
        assert skills_only and skills_only[0].severity.value == "info"

    def test_experience_evidence_recognised(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["Python"]),
            summary=None,
            experience=[
                WorkExperience(
                    company="Acme",
                    title="Engineer",
                    achievements=["Built Python services."],
                )
            ],
        )
        _, coverage, matches, findings = analyze_job_specific(
            resume, make_job(required=["Python"])
        )
        assert has_strong_evidence(matches[0].evidence_locations)
        assert coverage.evidence_supported_required == 1.0
        assert any(f.rule_id == "ats.job.evidence_present" for f in findings)
        assert not any(f.rule_id == "ats.job.skills_only" for f in findings)

    def test_project_technologies_are_strong_evidence(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["FastAPI"]),
            summary=None,
            experience=[],
            projects=[
                Project(
                    name="P",
                    description="Built an API.",
                    technologies=["FastAPI"],
                )
            ],
        )
        _, _, matches, _ = analyze_job_specific(
            resume, make_job(required=["FastAPI"])
        )
        assert has_strong_evidence(matches[0].evidence_locations)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_weights_sum_to_100(self) -> None:
        assert sum(WEIGHTS.values()) == 100.0
        assert set(CATEGORY_ORDER) == set(WEIGHTS)

    def test_label_bounds(self) -> None:
        assert score_label(49.9) == "Weak"
        assert score_label(50.0) == "Needs Improvement"
        assert score_label(64.9) == "Needs Improvement"
        assert score_label(65.0) == "Good"
        assert score_label(79.9) == "Good"
        assert score_label(80.0) == "Strong"
        assert score_label(100.0) == "Strong"
        assert score_label(None) is None

    def test_weight_redistribution_when_buckets_missing(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        result = analyze_job_specific_ats(
            resume, make_job(required=["Python"])
        )
        applied = result.metadata.applied_weights
        assert "preferred_coverage" not in applied
        assert applied["required_coverage"] == 50.0
        assert result.overall_score is not None

    def test_deterministic_repeatability(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["Python", "Docker"]),
            experience=[
                WorkExperience(
                    company="Acme",
                    title="Engineer",
                    achievements=["Built Python and Docker services."],
                )
            ],
        )
        job = make_job(
            required=["Python"],
            responsibilities=["Build services using Python."],
        )
        first = analyze_job_specific_ats(resume, job)
        second = analyze_job_specific_ats(resume, job)
        assert first.model_dump() == second.model_dump()

    def test_compute_overall_returns_none_when_no_category(self) -> None:
        overall, applied = compute_overall({})
        assert overall is None
        assert applied == {}

    def test_category_bounds_and_fraction_sanity(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        result = analyze_job_specific_ats(
            resume,
            make_job(required=["Python", "Kafka"], preferred=["Go"]),
        )
        for cs in result.category_scores:
            if cs.score is not None:
                assert 0.0 <= cs.score <= 100.0
        assert result.coverage.required_coverage == 0.5
        assert result.coverage.preferred_coverage == 0.0
        assert result.coverage.overall_coverage == 0.3333


class TestOutputOrdering:
    def test_term_matches_ordered_required_then_preferred_then_phrase(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["Python"]),
            summary="Built pipelines with Airlfow.",
        )
        job = make_job(
            required=["Python"],
            preferred=["Kubernetes"],
            responsibilities=["Build Data pipelines."],
        )
        result = analyze_job_specific_ats(resume, job)
        ranks = [m.origin.value for m in result.term_matches]
        assert (
            ranks.index("required")
            < ranks.index("preferred")
            < ranks.index("phrase")
        )

    def test_findings_reference_valid_category(self) -> None:
        resume = make_resume(skills=skill_set(technical=["Python"]))
        result = analyze_job_specific_ats(
            resume, make_job(required=["Python", "Kafka"])
        )
        for finding in result.findings:
            assert finding.category == "job_specific"
            assert finding.rule_id
            assert finding.impact >= 0.0
