"""Shared synthetic factories for Phase 7A Copilot tests.

All content is fictional. No real resume, job, or user data is used.
"""

from __future__ import annotations

import datetime as _dt

from app.ats_analysis.job_specific.schemas import (
    CoverageReport,
    CoverageTotals,
    JobSpecificATSResult,
    MatchType,
    TermMatch,
    TermOrigin,
)
from app.ats_analysis.schemas import (
    AnalysisMetadata,
    ATSReadinessResult,
    CategoryScore,
    Finding,
    Severity,
)
from app.copilot.schemas import CopilotAnalysisContext
from app.job_parsing.schemas import JobDescription, JobMetadata, Salary
from app.matching.schemas import MatchMetadata, MatchResult, SkillMatch
from app.parsing.schemas import (
    Certification,
    ConfidenceLevel,
    ContactInfo,
    CustomSection,
    Education,
    Project,
    Resume,
    ResumeMetadata,
    SkillSet,
    WorkExperience,
)


def make_resume(**overrides: object) -> Resume:
    base = Resume(
        contact=ContactInfo(
            name="Ada Lovelace",
            email="ada@example.com",
            phone="+1-555-0100",
            location="London, UK",
        ),
        summary="Software engineer experienced with Python, PostgreSQL, and "
        "large-scale data pipelines.",
        experience=[
            WorkExperience(
                company="Analytical Engines",
                title="Engineer",
                start_date="2019",
                end_date="Present",
                description="Built data processing systems and maintained "
                "production pipelines.",
                achievements=[
                    "Built a billing service used by thousands of users.",
                    "Reduced deployment time by 30%.",
                ],
                skills_mentioned=["Python", "PostgreSQL"],
            )
        ],
        education=[
            Education(institution="MIT", degree="B.S.", field="Computer Science")
        ],
        skills=SkillSet(
            technical=["Python", "PostgreSQL", "Docker"],
            languages=["Python"],
            all=["Python", "PostgreSQL", "Docker"],
        ),
        projects=[Project(name="Copilot", description="Local resume copilot.")],
        certifications=[Certification(name="AWS Solutions Architect")],
        custom_sections=[
            CustomSection(heading="Volunteering", content=["Founded a meetup."])
        ],
        metadata=ResumeMetadata(
            word_count=120,
            file_type="text",
            overall_confidence=ConfidenceLevel.HIGH,
        ),
    )
    return base.model_copy(update=overrides)


def make_job(**overrides: object) -> JobDescription:
    base = JobDescription(
        title="Backend Engineer",
        company="Acme Corp",
        location="Remote",
        summary="Build reliable APIs and data pipelines.",
        responsibilities=["Ship reliable software."],
        required_skills=["Python", "Kubernetes"],
        preferred_skills=["PostgreSQL"],
        qualifications=["Bachelor's degree in CS"],
        experience_requirements=["3+ years"],
        nice_to_have=["Docker"],
        salary=[Salary(text="100k")],
        metadata=JobMetadata(word_count=40, overall_confidence=ConfidenceLevel.HIGH),
    )
    return base.model_copy(update=overrides)


def make_finding(
    rule_id: str,
    title: str,
    *,
    severity: str = "medium",
    impact: float = 0.0,
    recommendation: str = "Act on this finding.",
) -> Finding:
    """Public factory for a deterministic ATS finding."""
    return Finding(
        category="experience",
        severity=Severity(severity),
        rule_id=rule_id,
        title=title,
        explanation=f"Explanation for {rule_id}.",
        evidence="Original excerpt that triggered the finding.",
        recommendation=recommendation,
        impact=impact,
    )


def make_ats_result(*findings: Finding) -> ATSReadinessResult:
    return ATSReadinessResult(
        overall_score=70.0,
        score_label="Moderate",
        category_scores=[
            CategoryScore(
                key="experience",
                label="Experience Quality",
                score=60.0,
                applicable=True,
                weight=1.0,
                max_weight=1.0,
            )
        ],
        findings=list(findings),
        metadata=AnalysisMetadata(
            method="ats-readiness-heuristic",
            version="6a-ats-1.0",
            weights={},
            applied_weights={},
            score_labels={},
            disclaimer="A deterministic, explainable heuristic, not a hiring "
            "or ATS-pass prediction.",
        ),
    )


def make_job_ats_result(
    *term_matches: TermMatch, findings: list[Finding] | None = None
) -> JobSpecificATSResult:
    return JobSpecificATSResult(
        overall_score=70.0,
        score_label="Moderate",
        category_scores=[
            CategoryScore(
                key="coverage",
                label="Terminology Coverage",
                score=60.0,
                applicable=True,
                weight=1.0,
                max_weight=1.0,
            )
        ],
        coverage=CoverageReport(
            required_coverage=0.5,
            preferred_coverage=0.5,
            overall_coverage=0.5,
            evidence_supported_required=0.5,
            evidence_supported_preferred=0.5,
            totals=CoverageTotals(
                required=3,
                required_matched=2,
                preferred=2,
                preferred_matched=1,
                phrases=1,
                phrases_matched=1,
            ),
        ),
        term_matches=list(term_matches),
        findings=list(findings or []),
        metadata=AnalysisMetadata(
            method="job-specific-ats-coverage",
            version="6b-ats-1.0",
            weights={},
            applied_weights={},
            score_labels={},
            disclaimer="A deterministic, explainable heuristic, not a hiring "
            "or ATS-pass prediction.",
        ),
    )


def make_match_result(
    *,
    matched_required: list[str] | None = None,
    matched_preferred: list[str] | None = None,
    missing_required: list[str] | None = None,
) -> MatchResult:
    return MatchResult(
        skill_match=SkillMatch(
            matched_required=list(matched_required or []),
            matched_preferred=list(matched_preferred or []),
            missing_required=list(missing_required or ["Kubernetes"]),
            score=40.0,
        ),
        metadata=MatchMetadata(reference_date=_dt.date(2026, 9, 17)),
    )


def make_context(
    *,
    ats: ATSReadinessResult | None = None,
    job_ats: JobSpecificATSResult | None = None,
    match: MatchResult | None = None,
) -> CopilotAnalysisContext:
    return CopilotAnalysisContext(
        ats_readiness=ats,
        job_specific_ats=job_ats,
        deterministic_match=match,
    )


def present_term(term: str, *, origin: str = "required") -> TermMatch:
    return TermMatch(
        term=term,
        origin=TermOrigin(origin),
        match_type=MatchType.EXACT,
        evidence_locations=["skills.technical"],
        evidence=term,
        explanation="Found in the reference skill list.",
    )


def absent_term(term: str, *, origin: str = "required") -> TermMatch:
    return TermMatch(
        term=term,
        origin=TermOrigin(origin),
        match_type=MatchType.ABSENT,
        evidence_locations=[],
        evidence="",
        explanation="Not found in the resume.",
    )
