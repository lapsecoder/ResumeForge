"""Offline tests for Phase 6D ML experiment design.

Covers: target definition (already in test_ml_data_pairs), feature registry
exclusions, experiment config fingerprint determinism, preprocessing
determinism and technical-term token discipline, metric arithmetic, and the
Phase 6E training/evaluator contracts.
"""

from __future__ import annotations

import pytest

from ml.evaluation.contracts import (
    PRIMARY_METRIC,
    EvaluationReport,
    default_evaluation_config,
)
from ml.experiments.config import (
    MODEST_GRID,
    TIERS,
    ExperimentConfig,
)
from ml.features.registry import (
    excluded_families,
    excluded_subfields,
    included_families,
    lookup,
    quasi_generative_families,
)
from ml.metrics import (
    balanced_accuracy,
    confusion_counts,
    f1_negative,
    f1_positive,
    macro_f1,
    per_class_report,
)
from ml.models.trainer import ExperimentPlan, train
from ml.preprocessing.structured import (
    degree_level,
    matches_industry,
    matches_seniority,
    role_overlap,
    seniority_tier,
)
from ml.preprocessing.text import normalize, tokenize, unique_terms

# ------------------------------------------------------------------ registry


class TestRegistry:
    def test_always_excludes_identity_and_raw_must_have(self) -> None:
        assert "identity" in excluded_families()
        assert "raw_must_have" in excluded_families()
        assert "generative_rule" in excluded_families()
        assert "provided_embeddings" in excluded_families()

    def test_leakage_risk_metadata(self) -> None:
        assert lookup("identity").leakage_risk == "identity"
        assert lookup("generative_rule").leakage_risk == "target-derived"
        assert lookup("minilm").leakage_risk == "none"
        assert lookup("provided_embeddings").leakage_risk == "unknown-provenance"

    def test_ats_6b_excludes_required_coverage(self) -> None:
        subfields = excluded_subfields("ats_6b")
        assert "required_coverage" in subfields
        assert "overall_coverage" in subfields
        assert "preferred_coverage" not in subfields

    def test_quasi_generative_tagged(self) -> None:
        qg = quasi_generative_families()
        assert "deterministic_match" in qg
        assert "ats_6b" in qg
        assert "minilm" not in qg

    def test_included_count_exceeds_excluded(self) -> None:
        assert len(included_families()) > len(excluded_families())


# --------------------------------------------------------------- preprocessing


class TestPreprocessing:
    def test_normalise_idempotent(self) -> None:
        text = "  a    b  c\n\n "
        assert normalize(normalize(text)) == normalize(text)

    def test_nfkc_applied(self) -> None:
        cjk = "\uff12"  # fullwidth digit two
        assert "2" in normalize(cjk)

    def test_technical_tokens_preserved(self) -> None:
        assert tokenize("Node.js + C++ + .NET") == ["node.js", "c++", ".net"]

    def test_ordering_preserved(self) -> None:
        terms = unique_terms("b a b a c")
        assert terms == ["b", "a", "c"]


# ----------------------------------------------------------- structured rules


class TestStructured:
    def test_seniority_tier_boundaries(self) -> None:
        assert seniority_tier(2.9) == "junior"
        assert seniority_tier(3.0) == "mid"
        assert seniority_tier(7.0) == "senior"

    def test_seniority_tier_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            seniority_tier(-1)

    def test_degree_level_match(self) -> None:
        assert degree_level("B.Tech in CS") == "bachelors"
        assert degree_level("Ph.D") == "phd"
        assert degree_level("high school") == "high_school"
        assert degree_level("diploma") == "diploma"

    def test_matches_casefold(self) -> None:
        assert matches_seniority("mid", "MID") is True
        assert matches_industry("tech", "finance") is False
        assert role_overlap("Engineer", "engineer") is True


# -------------------------------------------------------------------- metrics


class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y = [1, 1, 0, 0]
        assert macro_f1(y, y) == pytest.approx(1.0)
        assert balanced_accuracy(y, y) == pytest.approx(1.0)

    def test_all_wrong_is_zero(self) -> None:
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        assert macro_f1(y_true, y_pred) == pytest.approx(0.0)
        assert confusion_counts(y_true, y_pred) == (0, 0, 4, 0)

    def test_symmetry(self) -> None:
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        report = per_class_report(y_true, y_pred)
        assert 0.0 < report["positive"]["recall"] < 1.0
        assert 0.0 < report["negative"]["recall"] < 1.0

    def test_per_class_f1_vals(self) -> None:
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        assert f1_positive(y_true, y_pred) == pytest.approx(2 / 3, abs=1e-9)
        assert f1_negative(y_true, y_pred) == pytest.approx(1 / 2, abs=1e-9)

    def test_macro_f1_both_f1s_averaged(self) -> None:
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 1, 0]
        expected = (f1_positive(y_true, y_pred) + f1_negative(y_true, y_pred)) / 2
        assert macro_f1(y_true, y_pred) == pytest.approx(expected)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            confusion_counts([], [])


# --------------------------------------------------------------- experiment cfg


class TestExperimentConfig:
    def test_default_deterministic(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.fingerprint() == ExperimentConfig.default().fingerprint()

    def test_seed_shift_changes_fingerprint(self) -> None:
        cfg2 = ExperimentConfig(seed=42)
        cfg3 = ExperimentConfig(seed=43)
        assert cfg2.fingerprint() != cfg3.fingerprint()

    def test_all_tiers_valid(self) -> None:
        cfg = ExperimentConfig.default()
        cfg.validate_tiers()
        known = {tier.tier for tier in TIERS}
        assert all(t in known for t in cfg.tiers)

    def test_modest_grid_nonempty(self) -> None:
        assert "logistic_regression" in MODEST_GRID
        assert "linear_svm" in MODEST_GRID
        assert "classical" in MODEST_GRID


# ------------------------------------------------------------ trainer contract


class TestTrainerContract:
    def test_train_creates_fitted_model(self) -> None:
        import numpy as np
        import pandas as pd

        from ml.splits import SplitManifest

        manifest = SplitManifest(jobs={}, resumes={})
        plan = ExperimentPlan(
            config=ExperimentConfig(),
            tier=0,
            model_spec="majority",
            manifest=manifest,
        )
        X = pd.DataFrame({"feature_0": [0.0, 1.0, 2.0, 3.0]})
        y = np.array([0, 0, 1, 1])
        model = train(plan, X, y)
        assert model.train_samples == 4
        assert model.estimator is not None
        preds = model.predict(X)
        assert len(preds) == 4

    def test_plan_describe(self) -> None:
        from ml.splits import SplitManifest

        manifest = SplitManifest(jobs={"a": "train"}, resumes={"b": "train"})
        plan = ExperimentPlan(
            config=ExperimentConfig(),
            tier=0,
            model_spec="majority",
            manifest=manifest,
        )
        desc = plan.describe()
        assert desc["name"] == "majority baseline"
        assert "fingerprint" in desc


# ---------------------------------------------------------- evaluator contract


class TestEvaluatorContract:
    def test_valid_report_succeeds(self) -> None:
        report = EvaluationReport(
            metrics={PRIMARY_METRIC: 0.5, "accuracy": 0.6},
            confusion={"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            per_class={"positive": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                        "negative": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
            metadata={},
        )
        assert report.primary_missing is False
        assert report.confusion_keys_invalid is False

    def test_missing_primary_raises(self) -> None:
        with pytest.raises(ValueError):
            EvaluationReport(
                metrics={}, confusion={"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                per_class={}, metadata={},
            )

    def test_default_config_includes_macro_f1(self) -> None:
        cfg = default_evaluation_config()
        assert PRIMARY_METRIC in cfg.metrics
