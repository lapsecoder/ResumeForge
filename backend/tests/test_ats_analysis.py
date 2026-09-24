"""Unit tests for the deterministic ATS readiness / resume-quality engine.

All fixtures are synthetic and fully offline. The engine has no randomness, so
every score is deterministic and reproducible.
"""

from __future__ import annotations

from app.ats_analysis.analyzers import (
    analyze_bullets,
    analyze_contact,
    analyze_dates,
    analyze_education,
    analyze_experience,
    analyze_parsing,
    analyze_projects_certs,
    analyze_quantified,
    analyze_skills,
    analyze_structure,
)
from app.ats_analysis.rules import CATEGORY_ORDER, WEIGHTS
from app.ats_analysis.scorer import compute_overall, score_label
from app.ats_analysis.service import analyze_resume
from app.parsing.schemas import (
    Certification,
    ConfidenceLevel,
    ContactInfo,
    CustomSection,
    Education,
    Project,
    Resume,
    ResumeMetadata,
    SectionConfidence,
    SkillSet,
    WorkExperience,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def meta(
    word_count: int = 300,
    *,
    section_order: list[str] | None = None,
    section_confidence: list[SectionConfidence] | None = None,
) -> ResumeMetadata:
    return ResumeMetadata(
        word_count=word_count,
        file_type="pdf",
        overall_confidence=ConfidenceLevel.HIGH,
        section_confidence=section_confidence or [],
        section_order=section_order or [],
    )


def contact_info(
    *,
    name: str | None = "Jane Doe",
    email: str | None = "jane@example.com",
    phone: str | None = "123-456-7890",
    linkedin: str | None = "https://linkedin.com/in/jane",
    github: str | None = None,
    website: str | None = None,
) -> ContactInfo:
    return ContactInfo(
        name=name,
        email=email,
        phone=phone,
        linkedin=linkedin,
        github=github,
        website=website,
    )


def skill_set(
    *,
    technical: list[str] | None = None,
    soft: list[str] | None = None,
    tools: list[str] | None = None,
    languages: list[str] | None = None,
    all_skills: list[str] | None = None,
) -> SkillSet:
    return SkillSet(
        technical=technical or [],
        soft=soft or [],
        tools=tools or [],
        languages=languages or [],
        all=all_skills or [],
    )


def exp(
    *,
    company: str = "Acme",
    title: str = "Engineer",
    start: str | None = "Jan 2020",
    end: str | None = "Dec 2024",
    description: str = "",
    achievements: list[str] | None = None,
) -> WorkExperience:
    return WorkExperience(
        company=company,
        title=title,
        start_date=start,
        end_date=end,
        description=description,
        achievements=achievements or [],
    )


def make_resume(
    *,
    contact: ContactInfo | None = None,
    summary: str | None = "Experienced engineer.",
    experience: list[WorkExperience] | None = None,
    education: list[Education] | None = None,
    skills: SkillSet | None = None,
    projects: list[Project] | None = None,
    certifications: list[Certification] | None = None,
    custom_sections: list[CustomSection] | None = None,
    metadata: ResumeMetadata | None = None,
) -> Resume:
    return Resume(
        contact=contact if contact is not None else contact_info(),
        summary=summary,
        experience=experience or [],
        education=education or [],
        skills=skills if skills is not None else skill_set(technical=["Python"]),
        projects=projects or [],
        certifications=certifications or [],
        custom_sections=custom_sections or [],
        metadata=metadata if metadata is not None else meta(),
    )


ORDER_ALL = [
    "header",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
]


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------


class TestContact:
    def test_full_contact_scores_100(self) -> None:
        score, findings = analyze_contact(
            make_resume(contact=contact_info(github="https://github.com/jane"))
        )
        assert score == 100.0
        assert not any(f.impact > 0 for f in findings)

    def test_missing_name_and_email(self) -> None:
        score, findings = analyze_contact(
            make_resume(contact=contact_info(name=None, email=None))
        )
        # phone + one link present -> name -30, email -30, one link -8
        assert score == 32.0
        rules = {f.rule_id for f in findings}
        assert "ats.contact.name" in rules
        assert "ats.contact.email" in rules

    def test_optional_links_not_required_but_penalised_lightly(self) -> None:
        score, _ = analyze_contact(
            make_resume(contact=contact_info(linkedin=None, github=None, website=None))
        )
        # name+email+phone present, no links -> -15
        assert score == 85.0

    def test_one_link_is_small_penalty(self) -> None:
        score, _ = analyze_contact(
            make_resume(
                contact=contact_info(linkedin="https://linkedin.com/in/jane")
            )
        )
        # only one of three optional links present -> -8
        assert score == 92.0

    def test_all_optional_missing_are_info_only(self) -> None:
        _, findings = analyze_contact(
            make_resume(contact=contact_info(github=None, website=None))
        )
        info = [f for f in findings if f.rule_id == "ats.contact.github"]
        assert info and info[0].severity.value == "info"
        assert info[0].impact == 0.0


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_all_sections_and_sensible_order(self) -> None:
        resume = make_resume(
            experience=[exp()],
            education=[Education(institution="IIT", degree="B.Tech")],
            skills=skill_set(technical=["Python"]),
            projects=[Project(name="P", description="Built something useful.")],
            certifications=[Certification(name="C")],
            metadata=meta(section_order=ORDER_ALL),
        )
        score, _ = analyze_structure(resume)
        assert score == 100.0

    def test_no_experience_with_projects_less_penalty(self) -> None:
        with_projects, _ = analyze_structure(
            make_resume(
                experience=[],
                education=[Education(institution="IIT", degree="B.Tech")],
                projects=[Project(name="P", description="Did a thing.")],
            )
        )
        without_projects, _ = analyze_structure(
            make_resume(
                experience=[],
                education=[Education(institution="IIT", degree="B.Tech")],
                projects=[],
            )
        )
        assert with_projects is not None
        assert without_projects is not None
        assert with_projects > without_projects

    def test_missing_summary_not_heavily_penalised(self) -> None:
        score, findings = analyze_structure(
            make_resume(
                summary=None,
                experience=[exp()],
                education=[Education(institution="IIT", degree="B.Tech")],
                skills=skill_set(technical=["Python"]),
            )
        )
        assert score is not None
        assert score >= 90.0
        assert any(f.rule_id == "ats.structure.summary" for f in findings)

    def test_no_penalty_when_projects_or_certs_absent(self) -> None:
        _, findings = analyze_structure(
            make_resume(
                experience=[exp()],
                education=[Education(institution="IIT", degree="B.Tech")],
                skills=skill_set(technical=["Python"]),
                projects=[],
                certifications=[],
            )
        )
        deducted = [f for f in findings if f.impact > 0]
        assert not any("project" in f.rule_id for f in deducted)
        assert not any("cert" in f.rule_id for f in deducted)

    def test_order_unavailable_is_not_penalised(self) -> None:
        score, findings = analyze_structure(
            make_resume(
                experience=[exp()],
                education=[Education(institution="IIT", degree="B.Tech")],
                skills=skill_set(technical=["Python"]),
                metadata=meta(section_order=[]),
            )
        )
        assert score == 100.0
        assert any(
            f.rule_id == "ats.structure.order.unavailable" for f in findings
        )

    def test_education_before_experience_is_flagged(self) -> None:
        score, findings = analyze_structure(
            make_resume(
                experience=[exp()],
                education=[Education(institution="IIT", degree="B.Tech")],
                skills=skill_set(technical=["Python"]),
                metadata=meta(
                    section_order=[
                        "header",
                        "summary",
                        "education",
                        "experience",
                        "skills",
                    ]
                ),
            )
        )
        assert score is not None
        assert score < 100.0
        assert any(
            f.rule_id == "ats.structure.order.education_first" for f in findings
        )


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------


class TestExperience:
    def test_no_experience_is_not_applicable(self) -> None:
        score, findings = analyze_experience(make_resume(experience=[]))
        assert score is None
        assert findings == []

    def test_strong_entries_score_100(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    company="Acme",
                    start="2020",
                    end="2024",
                    achievements=[
                        "Built a service used widely.",
                        "Reduced API latency significantly.",
                        "Led the platform team.",
                    ],
                ),
                exp(
                    company="Globex",
                    start="2016",
                    end="2020",
                    achievements=[
                        "Designed the data model.",
                        "Automated the release process.",
                        "Improved test coverage.",
                    ],
                ),
            ]
        )
        score, _ = analyze_experience(resume)
        assert score == 100.0

    def test_missing_title_and_company_penalised(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    title="",
                    company="",
                    achievements=["Built a service used widely."],
                )
            ]
        )
        score, findings = analyze_experience(resume)
        assert score is not None
        rules = {f.rule_id for f in findings}
        assert "ats.experience.title" in rules
        assert "ats.experience.company" in rules

    def test_missing_dates_reported_not_assumed(self) -> None:
        resume = make_resume(
            experience=[
                exp(start=None, end=None, achievements=["Built a service."])
            ]
        )
        _, findings = analyze_experience(resume)
        rules = {f.rule_id for f in findings}
        assert "ats.experience.start_date" in rules
        assert "ats.experience.end_date" in rules

    def test_empty_entry_is_high_severity(self) -> None:
        resume = make_resume(
            experience=[exp(description="", achievements=[])]
        )
        _, findings = analyze_experience(resume)
        detail = [f for f in findings if f.rule_id == "ats.experience.detail"]
        assert detail and detail[0].severity.value == "high"

    def test_no_bullets_but_description_is_medium(self) -> None:
        resume = make_resume(
            experience=[exp(description="Worked on the platform.", achievements=[])]
        )
        _, findings = analyze_experience(resume)
        assert any(f.rule_id == "ats.experience.bullets" for f in findings)

    def test_single_entry_is_info_only(self) -> None:
        resume = make_resume(experience=[exp(achievements=["Built a service."])])
        _, findings = analyze_experience(resume)
        single = [f for f in findings if f.rule_id == "ats.experience.single_entry"]
        assert single and single[0].impact == 0.0


# ---------------------------------------------------------------------------
# Bullets
# ---------------------------------------------------------------------------


class TestBullets:
    def test_no_lines_not_applicable(self) -> None:
        score, _ = analyze_bullets(make_resume(experience=[], projects=[]))
        assert score is None

    def test_all_action_verbs_no_deduction(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    achievements=[
                        "Built the parser.",
                        "Reduced the latency.",
                        "Led the team.",
                        "Designed the schema.",
                    ]
                )
            ]
        )
        score, _ = analyze_bullets(resume)
        assert score == 100.0

    def test_no_action_verbs_penalised(self) -> None:
        resume = make_resume(
            experience=[
                exp(achievements=["The parser was built.", "Latency was reduced."])
            ]
        )
        score, findings = analyze_bullets(resume)
        assert score is not None and score < 100.0
        assert any(f.rule_id == "ats.bullets.no_action" for f in findings)

    def test_first_person_penalised(self) -> None:
        resume = make_resume(
            experience=[exp(achievements=["I built the parser.", "Led my team."])]
        )
        _, findings = analyze_bullets(resume)
        assert any(f.rule_id == "ats.bullets.first_person" for f in findings)

    def test_short_bullets_penalised(self) -> None:
        resume = make_resume(
            experience=[exp(achievements=["Team player.", "Hard worker.", "Built."])]
        )
        _, findings = analyze_bullets(resume)
        assert any(f.rule_id == "ats.bullets.too_short" for f in findings)

    def test_vague_bullets_penalised(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    achievements=[
                        "Responsible for various tasks.",
                        "Worked on stuff.",
                        "Helped with things.",
                    ]
                )
            ]
        )
        _, findings = analyze_bullets(resume)
        assert any(f.rule_id == "ats.bullets.vague" for f in findings)

    def test_repetitive_starts_flagged(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    achievements=[
                        "Built service A.",
                        "Built service B.",
                        "Built service C.",
                        "Built service D.",
                    ]
                )
            ]
        )
        _, findings = analyze_bullets(resume)
        assert any(f.rule_id == "ats.bullets.repetitive" for f in findings)

    def test_project_descriptions_are_analysed(self) -> None:
        resume = make_resume(
            experience=[],
            projects=[
                Project(
                    name="P",
                    description="Built a web application serving 5,000 users.",
                )
            ],
        )
        score, _ = analyze_bullets(resume)
        assert score == 100.0


# ---------------------------------------------------------------------------
# Quantified evidence
# ---------------------------------------------------------------------------


class TestQuantified:
    def test_no_lines_not_applicable(self) -> None:
        score, _ = analyze_quantified(make_resume(experience=[], projects=[]))
        assert score is None

    def test_quantified_evidence_detected(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    achievements=[
                        "Reduced processing time by 30%.",
                        "Served 10,000 users.",
                        "Improved accuracy from 82% to 91%.",
                    ]
                )
            ]
        )
        score, findings = analyze_quantified(resume)
        assert score == 100.0
        assert any(f.rule_id == "ats.quantified.good" for f in findings)

    def test_no_quantified_evidence_penalised(self) -> None:
        resume = make_resume(
            experience=[
                exp(achievements=["Built the parser.", "Led the team."])
            ]
        )
        score, findings = analyze_quantified(resume)
        assert score == 75.0
        assert any(f.rule_id == "ats.quantified.none" for f in findings)

    def test_currency_detected(self) -> None:
        resume = make_resume(
            experience=[
                exp(achievements=["Managed a $50,000 budget.", "Led the team."])
            ]
        )
        _, findings = analyze_quantified(resume)
        assert not any(f.rule_id == "ats.quantified.none" for f in findings)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class TestSkills:
    def test_no_skills_not_applicable(self) -> None:
        score, _ = analyze_skills(make_resume(skills=skill_set()))
        assert score is None

    def test_structured_skills_score_100(self) -> None:
        resume = make_resume(
            skills=skill_set(
                technical=["Python", "Go"],
                soft=["Leadership", "Communication"],
                tools=["Docker", "Git"],
                languages=["English", "Hindi"],
            )
        )
        score, _ = analyze_skills(resume)
        assert score == 100.0

    def test_duplicates_penalised(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["Python", "python", "Go", "go"])
        )
        score, findings = analyze_skills(resume)
        assert score is not None and score < 100.0
        assert any(f.rule_id == "ats.skills.duplicates" for f in findings)

    def test_large_skill_list_penalised(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=[f"Skill{i}" for i in range(50)])
        )
        _, findings = analyze_skills(resume)
        assert any(f.rule_id == "ats.skills.large" for f in findings)

    def test_flat_only_skills_flagged(self) -> None:
        resume = make_resume(
            skills=skill_set(
                all_skills=["Python", "Go", "Docker", "Git", "SQL", "Redis"]
            )
        )
        _, findings = analyze_skills(resume)
        assert any(
            f.rule_id == "ats.skills.unusual_categories" for f in findings
        )

    def test_empty_categories_are_info_only(self) -> None:
        resume = make_resume(
            skills=skill_set(technical=["Python", "Go", "Docker"])
        )
        _, findings = analyze_skills(resume)
        empty = [
            f for f in findings if f.rule_id == "ats.skills.empty_categories"
        ]
        assert empty and empty[0].impact == 0.0


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------


class TestEducation:
    def test_no_education_not_applicable(self) -> None:
        score, _ = analyze_education(make_resume(education=[]))
        assert score is None

    def test_complete_education_scores_100(self) -> None:
        resume = make_resume(
            education=[
                Education(
                    institution="IIT",
                    degree="B.Tech",
                    field="Computer Science",
                    start_date="2014",
                    end_date="2018",
                )
            ]
        )
        score, _ = analyze_education(resume)
        assert score == 100.0

    def test_incomplete_education_penalised(self) -> None:
        resume = make_resume(education=[Education(institution="IIT")])
        score, findings = analyze_education(resume)
        assert score is not None and score < 100.0
        assert any(f.rule_id == "ats.education.degree" for f in findings)

    def test_gpa_not_required_or_invented(self) -> None:
        resume = make_resume(
            education=[
                Education(
                    institution="IIT",
                    degree="B.Tech",
                    field="CS",
                    start_date="2014",
                    end_date="2018",
                )
            ]
        )
        score, _ = analyze_education(resume)
        assert score == 100.0


# ---------------------------------------------------------------------------
# Projects & certifications
# ---------------------------------------------------------------------------


class TestProjectsCerts:
    def test_neither_is_not_applicable(self) -> None:
        score, _ = analyze_projects_certs(
            make_resume(projects=[], certifications=[])
        )
        assert score is None

    def test_complete_project_and_cert_score_100(self) -> None:
        resume = make_resume(
            projects=[
                Project(
                    name="ResumeForge",
                    description="Built a resume parsing service used by 5,000 users.",
                    technologies=["Python", "FastAPI"],
                )
            ],
            certifications=[
                Certification(
                    name="AWS Certified Developer", issuer="Amazon", date="2021"
                )
            ],
        )
        score, _ = analyze_projects_certs(resume)
        assert score == 100.0

    def test_projects_without_certs_still_ok(self) -> None:
        resume = make_resume(
            projects=[
                Project(
                    name="P",
                    description="Built a web application used by 5,000 users.",
                    technologies=["Python"],
                )
            ],
            certifications=[],
        )
        score, _ = analyze_projects_certs(resume)
        assert score == 100.0

    def test_incomplete_cert_penalised(self) -> None:
        resume = make_resume(certifications=[Certification(name="PMP")])
        score, findings = analyze_projects_certs(resume)
        assert score is not None and score < 100.0
        assert any(f.rule_id == "ats.certs.issuer" for f in findings)


# ---------------------------------------------------------------------------
# Date consistency
# ---------------------------------------------------------------------------


class TestDates:
    def test_no_dates_not_applicable(self) -> None:
        resume = make_resume(
            experience=[exp(start=None, end=None, achievements=["Built."])]
        )
        score, _ = analyze_dates(resume)
        assert score is None

    def test_consistent_dates_score_100(self) -> None:
        resume = make_resume(
            experience=[
                exp(company="A", start="2019", end="2023"),
                exp(company="B", start="2016", end="2019"),
            ]
        )
        score, _ = analyze_dates(resume)
        assert score == 100.0

    def test_end_before_start_flagged(self) -> None:
        resume = make_resume(
            experience=[exp(start="2020", end="2018")]
        )
        score, findings = analyze_dates(resume)
        assert score is not None and score < 100.0
        assert any(f.rule_id == "ats.dates.end_before_start" for f in findings)

    def test_overlap_is_low_severity_not_invalid(self) -> None:
        resume = make_resume(
            experience=[
                exp(company="A", start="Jan 2019", end="Dec 2021"),
                exp(company="B", start="Jun 2020", end="Dec 2022"),
            ]
        )
        _, findings = analyze_dates(resume)
        overlap = [f for f in findings if f.rule_id == "ats.dates.overlap"]
        assert overlap and overlap[0].severity.value == "low"
        assert "not inherently invalid" in overlap[0].explanation

    def test_adjacent_year_ranges_not_flagged(self) -> None:
        resume = make_resume(
            experience=[
                exp(company="A", start="2018", end="2020"),
                exp(company="B", start="2020", end="2024"),
            ]
        )
        _, findings = analyze_dates(resume)
        assert not any(f.rule_id == "ats.dates.overlap" for f in findings)

    def test_current_role_excluded_from_consistency(self) -> None:
        resume = make_resume(
            experience=[exp(start="2020", end="Present")]
        )
        score, _ = analyze_dates(resume)
        assert score is None  # no fully-dated non-current pair


# ---------------------------------------------------------------------------
# Parsing / readability
# ---------------------------------------------------------------------------


class TestParsing:
    def test_rich_resume_scores_100(self) -> None:
        resume = make_resume(
            experience=[exp(achievements=["Built a service used widely."])],
            metadata=meta(600),
        )
        score, _ = analyze_parsing(resume)
        assert score == 100.0

    def test_sparse_resume_penalised(self) -> None:
        resume = make_resume(metadata=meta(20))
        score, findings = analyze_parsing(resume)
        assert score is not None and score < 100.0
        assert any(f.rule_id == "ats.parsing.sparse" for f in findings)

    def test_dense_resume_penalised(self) -> None:
        resume = make_resume(metadata=meta(2500))
        score, _ = analyze_parsing(resume)
        assert score is not None and score < 100.0

    def test_custom_sections_flagged(self) -> None:
        resume = make_resume(
            custom_sections=[
                CustomSection(heading="Hobbies", content=["Chess"]),
                CustomSection(heading="Volunteering", content=["Teaching"]),
                CustomSection(heading="Awards", content=["Best paper"]),
            ]
        )
        _, findings = analyze_parsing(resume)
        assert any(f.rule_id == "ats.parsing.custom_sections" for f in findings)

    def test_formatting_not_assessed_always_present(self) -> None:
        resume = make_resume()
        _, findings = analyze_parsing(resume)
        fmt = [
            f for f in findings
            if f.rule_id == "ats.parsing.format_not_assessed"
        ]
        assert fmt and fmt[0].severity.value == "info"
        assert "cannot be detected" in fmt[0].explanation

    def test_computed_word_count_fallback(self) -> None:
        resume = make_resume(
            experience=[exp(achievements=["Built a service."])],
            metadata=meta(0),
        )
        score, _ = analyze_parsing(resume)
        assert score is not None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class TestScorer:
    def test_weights_sum_to_100(self) -> None:
        assert sum(WEIGHTS.values()) == 100.0

    def test_non_applicable_weights_redistribute(self) -> None:
        scores = {key: None for key in CATEGORY_ORDER}
        scores["contact"] = 100.0
        overall, applied = compute_overall(scores)
        assert overall == 100.0
        assert applied == {"contact": 10.0}

    def test_weighted_average(self) -> None:
        scores = {key: None for key in CATEGORY_ORDER}
        scores["contact"] = 100.0
        scores["structure"] = 0.0
        overall, applied = compute_overall(scores)
        assert applied == {"contact": 10.0, "structure": 15.0}
        assert overall == round(100.0 * 10.0 / 25.0, 1)

    def test_no_applicable_categories_is_none(self) -> None:
        scores = {key: None for key in CATEGORY_ORDER}
        overall, applied = compute_overall(scores)
        assert overall is None
        assert applied == {}

    def test_score_labels(self) -> None:
        assert score_label(0.0) == "Weak"
        assert score_label(49.9) == "Weak"
        assert score_label(50.0) == "Needs Improvement"
        assert score_label(64.9) == "Needs Improvement"
        assert score_label(65.0) == "Good"
        assert score_label(79.9) == "Good"
        assert score_label(80.0) == "Strong"
        assert score_label(100.0) == "Strong"
        assert score_label(None) is None


# ---------------------------------------------------------------------------
# Service / result shape
# ---------------------------------------------------------------------------


class TestService:
    def test_result_bounds_and_shape(self) -> None:
        result = analyze_resume(make_resume())
        assert result.overall_score is not None
        assert 0.0 <= result.overall_score <= 100.0
        assert result.score_label in {"Weak", "Needs Improvement", "Good", "Strong"}
        assert len(result.category_scores) == len(CATEGORY_ORDER)
        assert result.metadata.method == "ats-readiness-heuristic"
        assert result.metadata.weights == WEIGHTS

    def test_deterministic_repeatability(self) -> None:
        resume = make_resume(
            experience=[exp(achievements=["Built a service.", "Reduced latency."])],
            metadata=meta(400),
        )
        first = analyze_resume(resume)
        second = analyze_resume(resume)
        assert first.model_dump() == second.model_dump()

    def test_every_deduction_has_traceable_finding(self) -> None:
        resume = make_resume(
            contact=contact_info(email=None),
            experience=[exp(title="", achievements=["Responsible for stuff."])],
            metadata=meta(40),
        )
        result = analyze_resume(resume)
        rules = {f.rule_id for f in result.findings}
        assert "ats.contact.email" in rules
        assert "ats.experience.title" in rules

    def test_category_scores_do_not_exceed_bounds(self) -> None:
        resume = make_resume(
            experience=[
                exp(
                    title="",
                    company="",
                    start=None,
                    end=None,
                    achievements=["Stuff."],
                )
            ],
            skills=skill_set(technical=[f"S{i}" for i in range(80)]),
            metadata=meta(10),
        )
        result = analyze_resume(resume)
        for cs in result.category_scores:
            if cs.score is not None:
                assert 0.0 <= cs.score <= 100.0
