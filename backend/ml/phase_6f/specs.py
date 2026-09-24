"""Phase 6F feature and ablation specifications.

Central place for the feature sets under investigation.  These constants are
the single source of truth for shortcut analysis, ablations, feature
importance groupings, and the focused tests, so analysis and tests stay in
sync.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.features.registry import quasi_generative_families

# Structured features flagged in the Phase 6F remit as being directly aligned
# with the synthetic generator's profile templates.  Ablation B removes them.
SUSPICIOUS_STRUCTURED: tuple[str, ...] = (
    "same_role",
    "same_industry",
    "same_seniority",
    "years_experience",
)

# Every feature produced by the quasi-generative (rule-reconstructing)
# families.  These approximate the dataset's generation rule (must-have
# coverage / text overlap) — the "canary" signals for rule leakage.
QUASI_GENERATIVE_FEATURES: tuple[str, ...] = (
    "keyword_overlap_required",
    "keyword_overlap_preferred",
    "keyword_overlap_combined",
    "text_token_overlap",
    "hybrid_score",
    "preferred_coverage",
    "evidence_supported_required",
    "evidence_supported_preferred",
)

# Feature -> family map used by the importance report and shortcut analysis.
FEATURE_FAMILY: dict[str, str] = {
    "years_experience": "structured",
    "years_experience_bucket": "structured",
    "degree_phd": "structured",
    "degree_masters": "structured",
    "degree_bachelors": "structured",
    "degree_diploma": "structured",
    "degree_high_school": "structured",
    "degree_unknown": "structured",
    "same_seniority": "structured",
    "same_industry": "structured",
    "same_role": "structured",
    "keyword_overlap_required": "deterministic_match",
    "keyword_overlap_preferred": "deterministic_match",
    "keyword_overlap_combined": "deterministic_match",
    "text_token_overlap": "deterministic_match",
    "hybrid_score": "deterministic_match",
    "action_verb_count": "ats_6a",
    "quantitative_count": "ats_6a",
    "section_completeness": "ats_6a",
    "preferred_coverage": "ats_6b",
    "evidence_supported_required": "ats_6b",
    "evidence_supported_preferred": "ats_6b",
    "minilm_cosine": "minilm",
}


def quasi_generative_feature_names() -> tuple[str, ...]:
    """Return the quasi-generative feature names from the leakage registry.

    Uses the same authoritative families as Phase 6D's leakage audit so the
    definition cannot drift from the locked exclusion list.
    """
    qg_families = set(quasi_generative_families())
    return tuple(
        name for name, family in FEATURE_FAMILY.items() if family in qg_families
    )


def all_rule_derived_features() -> tuple[str, ...]:
    """Union of suspicious-structured and quasi-generative features.

    This is the full "clearly synthetic-rule-derived" feature set removed by
    ablation C.
    """
    return SUSPICIOUS_STRUCTURED + quasi_generative_feature_names()


@dataclass(frozen=True)
class AblationSpec:
    """One ablation: a unique id, description, and features to exclude."""

    id: str
    description: str
    excluded: tuple[str, ...]


def ablation_specs() -> tuple[AblationSpec, ...]:
    """The Phase 6F ablation set (all trained on the job_grouped split)."""
    suspicious = SUSPICIOUS_STRUCTURED
    qg = quasi_generative_feature_names()

    def _spec(
        ablation_id: str,
        desc: str,
        removed: tuple[str, ...],
    ) -> AblationSpec:
        return AblationSpec(
            ablation_id,
            f"{desc}; removed={', '.join(removed) or 'none'} ({len(removed)})",
            removed,
        )

    return (
        _spec("6f-ablation-A", "full tier 4 features", ()),
        _spec(
            "6f-ablation-B",
            "tier 4 without suspicious structured features",
            suspicious,
        ),
        _spec(
            "6f-ablation-C",
            "tier 4 without all clearly synthetic-rule-derived features",
            suspicious + qg,
        ),
        _spec(
            "6f-ablation-D-same_role",
            "tier 4 without same_role",
            ("same_role",),
        ),
        _spec(
            "6f-ablation-D-same_industry",
            "tier 4 without same_industry",
            ("same_industry",),
        ),
        _spec(
            "6f-ablation-D-same_seniority",
            "tier 4 without same_seniority",
            ("same_seniority",),
        ),
        _spec(
            "6f-ablation-D-years_experience",
            "tier 4 without years_experience",
            ("years_experience",),
        ),
        _spec(
            "6f-ablation-E",
            "tier 4 without quasi-generative features only",
            qg,
        ),
    )


__all__ = [
    "SUSPICIOUS_STRUCTURED",
    "QUASI_GENERATIVE_FEATURES",
    "FEATURE_FAMILY",
    "quasi_generative_feature_names",
    "all_rule_derived_features",
    "AblationSpec",
    "ablation_specs",
]
