"""Feature registry for the candidate-matching experiments.

Every feature family is declared here together with its *include/exclude*
decision, leakage classification and evaluation note. This is the single
source of truth Track C consults, so no experiment can silently hand the
generation rule (or the label table) to the model.

Leakage vocabulary used below:

* ``identity``       - the value is a primary key; including it lets the model
  memorise the label table verbatim.
* ``generative``     - the value *is* an input of the published generation
  rule (raw must-have skills, coverage threshold, cap); including it turns
  the model into a rule re-implementation.
* ``quasi-generative`` - the value is strongly derived from the rule inputs
  (e.g. overlap of job terms with resume skills); it predicts labels well but
  only because it approximates the generator. Keep it available for ablations
  and report gains separately; do not claim it as general "matching skill".
* ``target-derived`` - the value is constructed from the labels themselves.
* ``unknown-provenance`` - provided by the dataset but its meaning/integrity
  is not documented well enough to use as a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Columns that are the raw generation rule or its inputs - never model inputs.
GENERATIVE_RULE_FIELDS = (
    "must_have_skills",
    "nice_to_have_skills",
    "coverage_threshold",
    "max_relevant_per_job",
)

# The labels themselves (the only legitimate use is ``train``/``eval`` split).
TARGET_FIELDS = ("relevant_resume_ids",)

# Identity columns that would memorise the label table if used as features.
IDENTITY_FIELDS = ("resume_id", "job_id")


@dataclass(frozen=True)
class FeatureFamily:
    """One declared feature source and its Track-C decision."""

    fid: str
    name: str
    inputs: tuple[str, ...]
    at_inference: str
    leakage_risk: str
    include: bool
    rationale: str
    excluded_subfields: tuple[str, ...] = field(default_factory=tuple)


_FAMILIES: tuple[FeatureFamily, ...] = (
    FeatureFamily(
        fid="resume_text",
        name="Resume narrative text",
        inputs=("summary", "experience_bullets", "skills-as-text"),
        at_inference="parsed resume (user upload)",
        leakage_risk="none",
        include=True,
        rationale=(
            "Narrative evidence of skill use. TF-IDF/boosting over these terms "
            "requires a vocabulary fitted on the training split only."
        ),
    ),
    FeatureFamily(
        fid="job_text",
        name="Job description text",
        inputs=("description", "responsibilities", "requirements-as-text"),
        at_inference="job description supplied by the user",
        leakage_risk="none",
        include=True,
        rationale="The job's own narrative; legitimate at inference time.",
    ),
    FeatureFamily(
        fid="structured_resume",
        name="Structured resume fields",
        inputs=("role", "seniority", "years_experience", "industry", "education"),
        at_inference="parsed resume (deterministic extraction)",
        leakage_risk="none",
        include=True,
        rationale="Deterministic attributes; class-level information prevails.",
    ),
    FeatureFamily(
        fid="structured_job",
        name="Structured job fields",
        inputs=("job_title", "seniority", "industry"),
        at_inference="job description (deterministic extraction)",
        leakage_risk="none",
        include=True,
        rationale="Deterministic job attributes plus role/seniority match vs resume.",
    ),
    FeatureFamily(
        fid="deterministic_match",
        name="Phase-5 deterministic matching scores",
        inputs=("keyword_overlap", "semantic_cosine", "hybrid_score"),
        at_inference="computed from resume + job at request time",
        leakage_risk="quasi-generative",
        include=True,
        rationale=(
            "Keyword overlap approximates the generator's coverage rule and "
            "therefore predicts labels strongly. Include for ablations; report "
            "rule-reproduction separately via the canary set."
        ),
    ),
    FeatureFamily(
        fid="ats_6a",
        name="General ATS quality scores (phase 6A)",
        inputs=("section_scores", "overall_score"),
        at_inference="recomputed per request (deterministic heuristics)",
        leakage_risk="none",
        include=True,
        rationale="General resume-quality heuristics with no rule adjacency.",
    ),
    FeatureFamily(
        fid="ats_6b",
        name="Job-specific ATS coverage/evidence (phase 6B)",
        inputs=("required_coverage", "preferred_coverage", "overall_coverage",
                "evidence_supported_required", "evidence_supported_preferred"),
        at_inference="recomputed per request (deterministic heuristics)",
        leakage_risk="quasi-generative",
        include=True,
        excluded_subfields=("required_coverage", "overall_coverage"),
        rationale=(
            "required_coverage and overall_coverage encode the generation "
            "rule's coverage fraction almost directly and are excluded from "
            "trained models. preferred_coverage and evidence-supported "
            "fractions are not rule inputs and stay available."
        ),
    ),
    FeatureFamily(
        fid="minilm",
        name="Local MiniLM semantic embeddings",
        inputs=("embedding_cosine", "per_block_cosines"),
        at_inference="embedded locally with sentence-transformers",
        leakage_risk="none",
        include=True,
        rationale=(
            "Local, GPU-embedded semantics. No external API; no fine-tuning "
            "assumed. Changes to the embedding model bump feature_version."
        ),
    ),
    FeatureFamily(
        fid="identity",
        name="Primary-key identity columns",
        inputs=("resume_id", "job_id"),
        at_inference="always known, always meaningless",
        leakage_risk="identity",
        include=False,
        rationale="The label table maps resume_id/job_id to relevance by construction.",
    ),
    FeatureFamily(
        fid="raw_must_have",
        name="Raw generation-rule skill lists",
        inputs=("must_have_skills", "nice_to_have_skills"),
        at_inference="job description (difficult to extract that cleanly)",
        leakage_risk="generative",
        include=False,
        rationale=(
            "These lists ARE the generator's inputs; feeding them in exactly "
            "reproduces the published labels. Content may still reach the "
            "model through job_text/structured_job naturally."
        ),
    ),
    FeatureFamily(
        fid="generative_rule",
        name="Derived rule membership flags",
        inputs=("candidate", "coverage_ge_0.6", "capped_membership"),
        at_inference="not computable without the raw lists",
        leakage_risk="target-derived",
        include=False,
        rationale="Re-implements the generation formula; exclusion is mandatory.",
    ),
    FeatureFamily(
        fid="provided_embeddings",
        name="Dataset-provided embedding arrays",
        inputs=("embeddings/resume", "embeddings/job"),
        at_inference="not reproducible from our pipeline",
        leakage_risk="unknown-provenance",
        include=False,
        rationale=(
            "The dataset ships embedding arrays of undocumented provenance. "
            "We embed with a pinned local model instead so feature meaning is "
            "fully reproducible."
        ),
    ),
)

FEATURE_FAMILIES: dict[str, FeatureFamily] = {f.fid: f for f in _FAMILIES}


def included_families() -> tuple[str, ...]:
    """Family ids that trained models are allowed to consume."""
    return tuple(sorted(f.fid for f in _FAMILIES if f.include))


def excluded_families() -> tuple[str, ...]:
    """Family ids that trained models must NOT consume."""
    return tuple(sorted(f.fid for f in _FAMILIES if not f.include))


def lookup(fid: str) -> FeatureFamily:
    if fid not in FEATURE_FAMILIES:
        raise KeyError(f"unknown feature family: {fid}")
    return FEATURE_FAMILIES[fid]


def excluded_subfields(fid: str) -> tuple[str, ...]:
    """Subfields to drop from an included family before training."""
    return lookup(fid).excluded_subfields


def quasi_generative_families() -> tuple[str, ...]:
    """Families that partly encode the generation rule (ablation canaries)."""
    return tuple(
        sorted(
            f.fid for f in _FAMILIES
            if f.include and f.leakage_risk == "quasi-generative"
        )
    )
