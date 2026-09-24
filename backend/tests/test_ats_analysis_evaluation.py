"""Scenario evaluation tests for the ATS readiness engine.

Checks a variety of resume archetypes against expected scores and
structural invariants (bounds, weight redistribution, deduction traceability,
determinism).  All data is synthetic; no real user data is used.
"""

from __future__ import annotations

import pytest

from app.ats_analysis.rules import CATEGORY_ORDER, WEIGHTS
from app.ats_analysis.schemas import ATSReadinessResult
from app.ats_analysis.service import analyze_resume
from app.parsing.schemas import (
    Certification,
    Education,
    Project,
    Resume,
    WorkExperience,
)
from tests.test_ats_analysis import (
    contact_info,
    make_resume,
    meta,
    skill_set,
)

# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

_ORDER = [
    "header",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
]


def _strong_resume() -> Resume:
    """Phenomenal resume with full data and nothing to deduct."""
    return make_resume(
        contact=contact_info(github="https://github.com/jane"),
        summary="Experienced software engineer with 8 years in backend systems.",
        experience=[
            WorkExperience(
                company="Acme",
                title="Senior Engineer",
                start_date="Jan 2020",
                end_date="Present",
                description="Platform team.",
                achievements=[
                    "Built a resume parsing service in Python serving 10,000 users.",
                    "Reduced processing time by 35% through caching.",
                    "Led a team of 6 engineers across two time zones.",
                    "Designed the REST API architecture for the platform.",
                ],
            ),
            WorkExperience(
                company="Globex",
                title="Engineer",
                start_date="Jun 2016",
                end_date="Dec 2019",
                description="Backend development.",
                achievements=[
                    "Automated deployment pipelines reducing release time by 40%.",
                    "Optimized PostgreSQL queries improving latency by 200ms.",
                    "Integrated third-party payment APIs used by 5,000 customers.",
                ],
            ),
        ],
        education=[
            Education(
                institution="IIT",
                degree="B.Tech",
                field="Computer Science",
                start_date="2012",
                end_date="2016",
            )
        ],
        skills=skill_set(
            technical=[
                "Python",
                "FastAPI",
                "PostgreSQL",
                "Docker",
                "Kubernetes",
                "Redis",
            ],
            soft=["Leadership", "Communication"],
            tools=["Git", "CI/CD", "Linux"],
            languages=["English", "Hindi"],
        ),
        projects=[
            Project(
                name="ResumeForge",
                description="Open-source resume parser used by 5,000 developers.",
                technologies=["Python", "FastAPI"],
            )
        ],
        certifications=[
            Certification(
                name="AWS Certified Developer",
                issuer="Amazon",
                date="2021",
            )
        ],
        metadata=meta(900, section_order=_ORDER),
    )


def _sparse_resume() -> Resume:
    """Minimal resume — only contact (partial), one skill, no sections."""
    return make_resume(
        contact=contact_info(linkedin=None),
        summary=None,
        experience=[],
        education=[],
        skills=skill_set(technical=["Python"]),
        projects=[],
        certifications=[],
        metadata=meta(40),
    )


def _student_resume() -> Resume:
    """Early-career / student with strong projects but no work experience."""
    return make_resume(
        contact=contact_info(linkedin="https://linkedin.com/in/jane"),
        summary="Computer Science undergraduate interested in full-stack development.",
        experience=[],
        education=[
            Education(
                institution="State University",
                degree="B.Tech",
                field="Computer Science",
                start_date="2022",
                end_date="2026",
            )
        ],
        skills=skill_set(
            technical=["Python", "JavaScript", "React", "Node.js"],
            tools=["Git"],
            languages=["English"],
        ),
        projects=[
            Project(
                name="Campus Events",
                description="Built a full-stack event platform serving 2,000 students.",
                technologies=["React", "Node.js"],
            ),
            Project(
                name="Weather Bot",
                description="Automated a Telegram bot delivering 500 daily forecasts.",
                technologies=["Python"],
            ),
        ],
        certifications=[],
        metadata=meta(300),
    )


def _experienced_resume() -> Resume:
    """Senior/staff engineer with long tenure and strong bullets."""
    return make_resume(
        contact=contact_info(github="https://github.com/jane"),
        summary="Staff engineer.",
        experience=[
            WorkExperience(
                company="F",
                title="Staff Engineer",
                start_date="2021",
                end_date="Present",
                achievements=[
                    "Led the platform team of 12 engineers.",
                    "Reduced cloud spend by 30%.",
                    "Designed the service mesh architecture.",
                ],
            ),
            WorkExperience(
                company="E",
                title="Senior Engineer",
                start_date="2018",
                end_date="2021",
                achievements=[
                    "Designed the event streaming pipeline.",
                    "Improved test coverage to 90%.",
                    "Mentored 5 junior engineers.",
                ],
            ),
            WorkExperience(
                company="D",
                title="Engineer",
                start_date="2015",
                end_date="2018",
                achievements=[
                    "Built the monolith migration tooling.",
                    "Automated the release cadence.",
                    "Documented system architecture.",
                ],
            ),
        ],
        education=[
            Education(
                institution="IIT",
                degree="B.Tech",
                field="CS",
                start_date="2011",
                end_date="2015",
            )
        ],
        skills=skill_set(
            technical=["Go", "Kubernetes", "PostgreSQL", "Redis", "Kafka"],
            tools=["Terraform", "ArgoCD"],
            languages=["English"],
        ),
        projects=[],
        certifications=[
            Certification(name="CKA", issuer="Linux Foundation", date="2022")
        ],
        metadata=meta(1200),
    )


def _project_heavy_resume() -> Resume:
    """Project-oriented resume with no work experience."""
    return make_resume(
        contact=contact_info(
            linkedin="https://linkedin.com/in/jane",
            github="https://github.com/jane",
        ),
        summary="Independent developer building full-stack products.",
        experience=[],
        education=[
            Education(
                institution="Community College",
                degree="B.Sc",
                field="Computer Science",
                start_date="2020",
                end_date="2024",
            )
        ],
        skills=skill_set(
            technical=["Python", "FastAPI", "React", "TypeScript"],
            tools=["Docker", "Git"],
            languages=["English"],
        ),
        projects=[
            Project(
                name="Analytics Dashboard",
                description=(
                    "Built a real-time dashboard processing "
                    "100k events per day."
                ),
                technologies=["React", "WebSockets"],
            ),
            Project(
                name="E-Commerce Store",
                description="Developed a store serving 1,200 customers with payments.",
                technologies=["Next.js", "Stripe"],
            ),
            Project(
                name="CLI Tool",
                description="Shipped a developer CLI downloaded 8,000 times.",
                technologies=["Go"],
            ),
        ],
        certifications=[],
        metadata=meta(500),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cat_scores(result: ATSReadinessResult) -> dict[str, float | None]:
    return {cs.key: cs.score for cs in result.category_scores}


def _cat_applicable(result: ATSReadinessResult) -> dict[str, bool]:
    return {cs.key: cs.applicable for cs in result.category_scores}


def _finding_impacts(result: ATSReadinessResult) -> dict[str, float]:
    """Sum of nonzero deductions per category."""
    out: dict[str, float] = {}
    for f in result.findings:
        if f.impact and f.impact > 0:
            out[f.category] = out.get(f.category, 0.0) + f.impact
    return out


def _assert_structural(result: ATSReadinessResult) -> None:
    """Common structural invariants that hold for every result."""
    assert result.overall_score is not None
    assert 0.0 <= result.overall_score <= 100.0
    assert result.score_label in (
        "Weak",
        "Needs Improvement",
        "Good",
        "Strong",
    )

    # All category scores in bounds or None
    for cs in result.category_scores:
        if cs.score is not None:
            assert 0.0 <= cs.score <= 100.0, f"{cs.key} out of bounds"
        assert cs.key in WEIGHTS
        # Original weight always correct
        assert cs.max_weight == WEIGHTS[cs.key]
        # If applicable, stored weight matches original; otherwise zero
        if cs.applicable:
            assert cs.weight == WEIGHTS[cs.key], f"{cs.key} weight mismatch"
        else:
            assert cs.weight == 0.0
            assert cs.score is None

    # Applied weights == original weights of applicable categories only
    applicable_keys = {cs.key for cs in result.category_scores if cs.applicable}
    assert set(result.metadata.applied_weights.keys()) == applicable_keys
    for k in applicable_keys:
        assert result.metadata.applied_weights[k] == WEIGHTS[k]

    # All findings reference valid categories and have non-empty rule_ids
    for f in result.findings:
        assert f.rule_id
        assert f.category in WEIGHTS
        assert f.impact >= 0.0

    # Weight sum = sum of applicable WEIGHTS
    weight_sum = sum(result.metadata.applied_weights.values())
    assert weight_sum == pytest.approx(sum(WEIGHTS[k] for k in applicable_keys))

    # Manual recomputation matches overall
    scores = _cat_scores(result)
    total_w = sum(WEIGHTS[k] for k in applicable_keys)
    expected_overall = round(
        sum(WEIGHTS[k] * scores[k] for k in applicable_keys) / total_w, 1
    )
    assert result.overall_score == expected_overall


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


class TestStrongScenario:
    EXPECTED_OVERALL = 100.0
    EXPECTED_LABEL = "Strong"
    EXPECTED_ALL_CATEGORIES = 100.0

    def test_expected_score(self) -> None:
        result = analyze_resume(_strong_resume())
        assert result.overall_score == self.EXPECTED_OVERALL
        assert result.score_label == self.EXPECTED_LABEL

    def test_all_categories_maxed(self) -> None:
        result = analyze_resume(_strong_resume())
        for cs in result.category_scores:
            assert cs.score == self.EXPECTED_ALL_CATEGORIES, cs.key

    def test_no_non_info_findings(self) -> None:
        result = analyze_resume(_strong_resume())
        non_info = [f for f in result.findings if f.severity != "info"]
        assert non_info == []

    def test_structure_invariants(self) -> None:
        _assert_structural(analyze_resume(_strong_resume()))


class TestSparseScenario:
    EXPECTED_OVERALL = 78.9
    EXPECTED_LABEL = "Good"

    def test_expected_score(self) -> None:
        result = analyze_resume(_sparse_resume())
        assert result.overall_score == self.EXPECTED_OVERALL
        assert result.score_label == self.EXPECTED_LABEL

    def test_not_applicable_categories_none(self) -> None:
        result = analyze_resume(_sparse_resume())
        na = {
            "experience",
            "bullets",
            "quantified",
            "education",
            "projects_certs",
            "dates",
        }
        for cs in result.category_scores:
            if cs.key in na:
                assert cs.score is None, f"{cs.key} should be None"
                assert not cs.applicable

    def test_applicable_categories_present(self) -> None:
        result = analyze_resume(_sparse_resume())
        applicable = {"contact", "structure", "skills", "parsing"}
        for cs in result.category_scores:
            if cs.key in applicable:
                assert cs.score is not None, f"{cs.key} should be scored"

    def test_redistribution_weight_sum_less_than_100(self) -> None:
        result = analyze_resume(_sparse_resume())
        wsum = sum(result.metadata.applied_weights.values())
        assert wsum == pytest.approx(40.0)  # 10+15+10+5

    def test_sparse_resume_not_penalised_for_absent_sections(self) -> None:
        result = analyze_resume(_sparse_resume())
        # Skills section present → no skills penalty
        skills_cs = next(cs for cs in result.category_scores if cs.key == "skills")
        assert skills_cs.score == 100.0

    def test_structure_invariants(self) -> None:
        _assert_structural(analyze_resume(_sparse_resume()))


class TestStudentScenario:
    EXPECTED_OVERALL = 96.8
    EXPECTED_LABEL = "Strong"

    def test_expected_score(self) -> None:
        result = analyze_resume(_student_resume())
        assert result.overall_score == self.EXPECTED_OVERALL
        assert result.score_label == self.EXPECTED_LABEL

    def test_experience_not_applicable(self) -> None:
        result = analyze_resume(_student_resume())
        exp_cs = next(cs for cs in result.category_scores if cs.key == "experience")
        assert exp_cs.score is None

    def test_projects_scored(self) -> None:
        result = analyze_resume(_student_resume())
        proj_cs = next(
            cs
            for cs in result.category_scores
            if cs.key == "projects_certs"
        )
        assert proj_cs.score == 100.0

    def test_education_present(self) -> None:
        result = analyze_resume(_student_resume())
        edu_cs = next(cs for cs in result.category_scores if cs.key == "education")
        assert edu_cs.score == 100.0

    def test_structure_invariants(self) -> None:
        _assert_structural(analyze_resume(_student_resume()))


class TestExperiencedScenario:
    EXPECTED_OVERALL = 100.0
    EXPECTED_LABEL = "Strong"

    def test_expected_score(self) -> None:
        result = analyze_resume(_experienced_resume())
        assert result.overall_score == self.EXPECTED_OVERALL
        assert result.score_label == self.EXPECTED_LABEL

    def test_all_categories_maxed(self) -> None:
        result = analyze_resume(_experienced_resume())
        for cs in result.category_scores:
            assert cs.score == 100.0, cs.key

    def test_bullets_entry_count_accepted(self) -> None:
        result = analyze_resume(_experienced_resume())
        entry_cs = next(cs for cs in result.category_scores if cs.key == "experience")
        assert entry_cs.score == 100.0

    def test_structure_invariants(self) -> None:
        _assert_structural(analyze_resume(_experienced_resume()))


class TestProjectHeavyScenario:
    EXPECTED_OVERALL = 97.8
    EXPECTED_LABEL = "Strong"

    def test_expected_score(self) -> None:
        result = analyze_resume(_project_heavy_resume())
        assert result.overall_score == self.EXPECTED_OVERALL
        assert result.score_label == self.EXPECTED_LABEL

    def test_experience_not_applicable(self) -> None:
        result = analyze_resume(_project_heavy_resume())
        exp_cs = next(cs for cs in result.category_scores if cs.key == "experience")
        assert exp_cs.score is None

    def test_projects_scored(self) -> None:
        result = analyze_resume(_project_heavy_resume())
        proj_cs = next(
            cs
            for cs in result.category_scores
            if cs.key == "projects_certs"
        )
        assert proj_cs.score == 100.0

    def test_structure_invariants(self) -> None:
        _assert_structural(analyze_resume(_project_heavy_resume()))


class TestCrossCuttingInvariants:
    def test_determinism_across_calls(self) -> None:
        resume = _strong_resume()
        assert analyze_resume(resume) == analyze_resume(resume)

    def test_no_overlap_bullets_vs_experience(self) -> None:
        """Bullets analyzer runs on achievement lines; experience runs on
        entry counts only — no double-counting."""
        resume = _strong_resume()
        r = analyze_resume(resume)
        b_findings = [f for f in r.findings if f.category == "bullets"]
        e_findings = [f for f in r.findings if f.category == "experience"]
        # Both categories present; bullets focus on content, experience on count
        b_rules = {f.rule_id for f in b_findings}
        e_rules = {f.rule_id for f in e_findings}
        assert not b_rules & e_rules, "Overlap between bullets and experience rules"

    def test_deductions_match_findings(self) -> None:
        """For every category, the sum of nonzero deduction impacts equals
        100 - score (clamped at floor for contact)."""
        resume = _experienced_resume()
        result = analyze_resume(resume)
        impacts = _finding_impacts(result)
        for cs in result.category_scores:
            if cs.score is None:
                continue
            total_impact = impacts.get(cs.key, 0.0)
            expected = max(0.0, 100.0 - total_impact)
            # Contact has a floor at 17
            if cs.key == "contact":
                expected = max(17.0, expected)
            assert cs.score == pytest.approx(expected), (
                f"{cs.key}: score {cs.score} != 100 - {total_impact}"
            )

    def test_all_non_applicable_weights_zero(self) -> None:
        """Non-applicable categories must have weight=0 and score=None."""
        resume = _student_resume()
        result = analyze_resume(resume)
        for cs in result.category_scores:
            if not cs.applicable:
                assert cs.weight == 0.0
                assert cs.score is None

    def test_metadata_fields_populated(self) -> None:
        result = analyze_resume(_strong_resume())
        m = result.metadata
        assert m.method == "ats-readiness-heuristic"
        assert m.version
        assert "not a probability" in m.disclaimer.lower()
        assert len(m.weights) == len(CATEGORY_ORDER)

    def test_findings_unique_rule_ids_per_category(self) -> None:
        """No duplicate rule_id within the same category."""
        result = analyze_resume(_experienced_resume())
        seen: set[tuple[str, str]] = set()
        for f in result.findings:
            assert (f.category, f.rule_id) not in seen
            seen.add((f.category, f.rule_id))
