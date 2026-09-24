"""Rule catalogue and documented constants for ATS Readiness analysis.

The scoring model is intentionally transparent: fixed weights, fixed labels,
fixed thresholds, and deterministic rules. Nothing here depends on random
numbers, language models, or external references.
"""

from __future__ import annotations

CATEGORY_ORDER = (
    "contact",
    "structure",
    "experience",
    "bullets",
    "quantified",
    "skills",
    "education",
    "projects_certs",
    "dates",
    "parsing",
)

CATEGORY_LABELS = {
    "contact": "Contact Completeness",
    "structure": "Section Completeness & Structure",
    "experience": "Experience Quality",
    "bullets": "Bullet & Action Quality",
    "quantified": "Quantified Achievements",
    "skills": "Skills Presentation",
    "education": "Education Quality",
    "projects_certs": "Projects & Certifications",
    "dates": "Date Consistency",
    "parsing": "Parsing & Readability",
}

# Documented default weights (sum = 100). Categories that are not applicable to
# a given resume redistribute their weight proportionally instead of
# penalising the resume.
WEIGHTS: dict[str, float] = {
    "contact": 10.0,
    "structure": 15.0,
    "experience": 20.0,
    "bullets": 15.0,
    "quantified": 10.0,
    "skills": 10.0,
    "education": 5.0,
    "projects_certs": 5.0,
    "dates": 5.0,
    "parsing": 5.0,
}

# Score label thresholds (lower bound inclusive, upper bound exclusive).
SCORE_LABELS_DOC: dict[str, str] = {
    "Weak": "< 50",
    "Needs Improvement": ">= 50 and < 65",
    "Good": ">= 65 and < 80",
    "Strong": ">= 80",
}

METHOD_NAME = "ats-readiness-heuristic"
VERSION = "6a-ats-1.0"

DISCLAIMER = (
    "This ATS Readiness Score is a deterministic, heuristic resume-quality "
    "signal. It is NOT a probability of passing any specific Applicant "
    "Tracking System, of being hired, or of a recruiter decision, and it is "
    "not a scientifically validated ATS prediction. It is computed only from "
    "the structured data visible to this analyzer."
)

CATEGORY_DISCLAIMER = {
    "ats.parsing.format_not_assessed": (
        "Visual formatting risks (multi-column layouts, tables, images, fonts) "
        "cannot be detected from parsed structured text alone. This analysis "
        "covers structural risks only."
    ),
    "ats.parsing.unverified_dates": (
        "Date consistency could not be assessed because the structured data "
        "contains no date pairs. Missing date evidence is reported as "
        "'unknown/missing evidence', never treated as invalid."
    ),
}
