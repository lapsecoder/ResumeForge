"""Rule catalogue and documented constants for Job-Specific ATS Coverage.

The scoring model is intentionally transparent: fixed weights, fixed labels,
fixed thresholds, split-aware matching types, and deterministic rules. Nothing
here depends on random numbers, language models, or external references.
"""

from __future__ import annotations

CATEGORY_ORDER = (
    "required_coverage",
    "preferred_coverage",
    "phrase_coverage",
    "evidence_representation",
)

CATEGORY_LABELS = {
    "required_coverage": "Required Terminology Coverage",
    "preferred_coverage": "Preferred Terminology Coverage",
    "phrase_coverage": "Job-Specific Phrase Coverage",
    "evidence_representation": "Evidence Representation",
}

# Documented default weights (sum = 100). Categories that are not applicable
# to a given pair (e.g. a job with no preferred skills, or a resume with no
# matched terms) redistribute their weight proportionally instead of
# penalising the candidate.
WEIGHTS: dict[str, float] = {
    "required_coverage": 50.0,
    "preferred_coverage": 20.0,
    "phrase_coverage": 15.0,
    "evidence_representation": 15.0,
}

# Score label thresholds (lower bound inclusive, upper bound exclusive).
SCORE_LABELS_DOC: dict[str, str] = {
    "Weak": "< 50",
    "Needs Improvement": ">= 50 and < 65",
    "Good": ">= 65 and < 80",
    "Strong": ">= 80",
}

METHOD_NAME = "job-specific-ats-coverage-heuristic"
VERSION = "6b-ats-1.0"

DISCLAIMER = (
    "This Job-Specific ATS Coverage score is a deterministic, heuristic "
    "measure of how completely the job description's explicit terminology is "
    "represented in the resume. It is NOT a probability of passing any "
    "specific Applicant Tracking System, of being hired, or of a recruiter "
    "decision. It measures explicit representation only: an absent term does "
    "not prove the candidate lacks the skill, and a present term does not "
    "prove proficiency."
)

# Terminology extraction bounds. A job description rarely needs more than
# these, so the caps keep the result payload predictable and cheap.
MAX_REQUIRED_TERMS = 100
MAX_PREFERRED_TERMS = 100
MAX_PHRASES = 80
PHRASE_MIN_TOKENS = 2
PHRASE_MAX_TOKENS = 4
MAX_TERM_MATCHES = 250

# Term-origin severity mapping for missing-term findings.
SEVERITY_FOR_ABSENT = {
    "required": "medium",
    "preferred": "low",
    "phrase": "low",
}
