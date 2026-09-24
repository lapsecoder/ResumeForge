"""Regression tests for parser failure patterns found on a real PDF.

The fixture mirrors the structure of a real extracted resume: em-dash
separators, DEL-rendered bullet glyphs, wrapped headings without blank lines,
inline bullet-separated skill lists, and label-prefixed skill lines. No real
personal data is used; names and contact details are synthetic.

The document flows through ``normalize_text`` before ``parse_resume`` exactly
like the ingestion service, so mojibake/DEL handling is exercised too.
"""

# ruff: noqa: E501  -- fixture lines mirror realistic long resume lines.
from __future__ import annotations

from app.ingestion.normalizer import normalize_text
from app.parsing.parser import parse_resume
from app.parsing.schemas import CustomSection

REALISH_RESUME = """MOHAMMED AYAAN AHMED
 Bangalore, India  |  8050425980  |  ayaanmsrit@gmail.com
PROFESSIONAL SUMMARY
Computer Science diploma student with a strong interest in Artificial Intelligence, Machine Learning, Generative AI, and software development.
EDUCATION
Diploma in Computer Science — Ramaiah Polytechnic
Expected 2027
CGPA: 9.11/10
Mc Nay Doons Public School — ICSE
2024
83%
EXPERIENCE
Event Management — Team Member  |  2025–Present
• Collaborated with teams to plan and execute events.
• Communicated with vendors and stakeholders.
• Managed multiple tasks under time constraints.
 TECHNICAL SKILLS
Programming: Python
AI & Machine Learning: Artificial Intelligence, Machine Learning, Generative AI, Prompt Engineering, AI-assisted
development
Development & Tools: Python libraries, VS Code, Jupyter Notebook
Hardware: PC Building & Assembly, Basic Hardware Troubleshooting
PROFESSIONAL SKILLS
Communication • Tele-calling • Customer Interaction • Team Coordination
CERTIFICATIONS
• Python and Generative AI — Infosys Springboard
• Cybersecurity — Infosys Springboard
LANGUAGES
English • Hindi • Urdu
 INTERESTS
Artificial Intelligence & Machine Learning • Generative AI • Software Development • Emerging
Technologies
"""


def _parse(text: str):
    return parse_resume(normalize_text(text))


class TestExperienceRegression:
    def test_entry_with_em_dash_and_trailing_date_is_not_lost(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert len(resume.experience) == 1
        entry = resume.experience[0]
        assert entry.title == "Event Management"
        assert entry.company == "Team Member"
        assert entry.start_date == "2025"
        assert entry.end_date == "Present"
        assert entry.achievements == [
            "Collaborated with teams to plan and execute events.",
            "Communicated with vendors and stakeholders.",
            "Managed multiple tasks under time constraints.",
        ]

    def test_header_withlonger_body_dates_still_detail(self) -> None:
        text = (
            "Jane\nEXPERIENCE\nDeveloper | Acme\n2021 - Present\n"
            "• Did A\n\n"
            "Coordinated annual events 2020 - 2021 across regions\n"
            "• Did B\n"
        )
        resume = _parse(text)
        assert len(resume.experience) == 1
        assert resume.experience[0].achievements[1] == "Did B"
        assert "Coordinated annual events" in resume.experience[0].description


class TestEducationRegression:
    def test_two_entries_share_one_block(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert len(resume.education) == 2

        diploma, school = resume.education
        assert diploma.degree == "Diploma in Computer Science"
        assert diploma.institution == "Ramaiah Polytechnic"
        assert diploma.end_date is None
        assert diploma.details == ["Expected 2027", "CGPA: 9.11/10"]

        assert school.degree is None
        assert school.institution == "Mc Nay Doons Public School"
        assert school.end_date == "2024"
        assert school.details == ["ICSE", "83%"]

    def test_school_continuation_is_not_a_new_entry(self) -> None:
        text = (
            "Jane\nEDUCATION\nBachelor of Technology, Computer Science\n"
            "Indian Institute of Technology\n2016 - 2020\n"
        )
        resume = _parse(text)
        assert len(resume.education) == 1
        assert resume.education[0].degree == "Bachelor of Technology, Computer Science"
        assert resume.education[0].institution == "Indian Institute of Technology"


class TestSkillBucketsRegression:
    def test_technical_soft_and_language_buckets(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert "Python" in resume.skills.technical
        assert "Communication" in resume.skills.soft
        assert "Tele-calling" in resume.skills.soft
        assert resume.skills.languages == ["English", "Hindi", "Urdu"]
        assert "Python" in resume.skills.all
        assert "Communication" in resume.skills.all
        assert "English" in resume.skills.all

    def test_colon_labels_are_not_skills(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert "Programming" not in resume.skills.all
        assert "AI & Machine Learning" not in resume.skills.all
        assert "Hardware" not in resume.skills.technical
        assert "Python" in resume.skills.technical

    def test_wrapped_lowercase_tail_completes_previous_skill(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert "AI-assisted development" in resume.skills.technical
        assert "AI-assisted" not in resume.skills.technical

    def test_bullet_separated_inline_skills(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert "Customer Interaction" in resume.skills.soft
        assert "Team Coordination" in resume.skills.soft

    def test_languages_never_reach_professional_skills(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert "English" not in resume.skills.soft
        assert "Hindi" not in resume.skills.technical


class TestCertificationRegression:
    def test_issuer_split_on_em_dash(self) -> None:
        resume = _parse(REALISH_RESUME)
        certs = {c.name: c.issuer for c in resume.certifications}
        assert certs["Python and Generative AI"] == "Infosys Springboard"
        assert certs["Cybersecurity"] == "Infosys Springboard"
        assert len(resume.certifications) == 2


class TestSectionBoundaryRegression:
    def test_languages_and_interests_not_absorbed_into_certs(self) -> None:
        resume = _parse(REALISH_RESUME)
        assert [c.name for c in resume.certifications] == [
            "Python and Generative AI",
            "Cybersecurity",
        ]

    def test_interests_surface_as_custom_section(self) -> None:
        resume = _parse(REALISH_RESUME)
        headings = [c.heading for c in resume.custom_sections]
        assert "INTERESTS" in headings
        interests = next(
            c for c in resume.custom_sections if c.heading == "INTERESTS"
        )
        assert interests.content

    def test_wrapped_tail_stays_with_previous_section(self) -> None:
        text = (
            "Jane\n\nINTERESTS\nAI & ML • Software • Emerging\nTechnologies\n"
        )
        resume = _parse(text)
        assert "skills" not in resume.metadata.section_order
        interests = _interest_section(resume)
        body = " ".join(interests.content)
        assert "Emerging Technologies" in body

    def test_section_order_has_no_duplicate_skills(self) -> None:
        resume = _parse(REALISH_RESUME)
        order = resume.metadata.section_order
        assert order.count("skills") == 1
        assert order == [
            "header",
            "summary",
            "education",
            "experience",
            "skills",
            "professional_skills",
            "certifications",
            "languages",
            "interests",
        ]


class TestMojibakeBulletInDocument:
    def test_cp437_bullet_mojibake_parses_as_separator(self) -> None:
        # The bullet "•" whose UTF-8 bytes were decoded as cp437/ANSI renders
        # as "ΓÇó"; it must be repaired before section parsing.
        text = (
            "Jane\n\nINTERESTS\nAI & ML \u0393\u00c7\u00f3 Software \u0393\u00c7\u00f3 "
            "Hardware\n"
        )
        resume = _parse(text)
        interests = _interest_section(resume)
        body = " ".join(interests.content)
        assert "\u0393\u00c7\u00f3" not in body
        assert "•" in body

    def test_del_and_cp437_bullets_are_interchangeable(self) -> None:
        del_doc = _parse("Jane\n\nINTERESTS\nA \x7f B \x7f C\n")
        cp437_doc = _parse("Jane\n\nINTERESTS\nA \u0393\u00c7\u00f3 B \u0393\u00c7\u00f3 C\n")
        assert del_doc.custom_sections == cp437_doc.custom_sections


class TestEducationSpacePreservation:
    def test_institution_keeps_internal_space(self) -> None:
        # Extraction keeps the source PDF's explicit space; the parser must
        # never merge the words back together.
        text = (
            "Jane\nEDUCATION\nDiploma in Computer Science — Ramaiah Polytechnic\n"
            "Expected 2027\n"
        )
        resume = _parse(text)
        assert len(resume.education) == 1
        assert resume.education[0].institution == "Ramaiah Polytechnic"
        assert resume.education[0].degree == "Diploma in Computer Science"
        assert "RamaiahPolytechnic" not in resume.education[0].institution


def _interest_section(resume) -> CustomSection:
    return next(
        c for c in resume.custom_sections if c.heading == "INTERESTS"
    )
