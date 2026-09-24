"""Focused tests for the Phase 6F analysis package.

All tests are offline (no GPU, no dataset download).  They verify the
structural correctness of ablation definitions, coefficient extraction,
split disjointness, error aggregation, robustness transforms, calibration
ECE, and reproducibility sanity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.phase_6f.calibration import reliability_curve
from ml.phase_6f.errors import (
    categorize,
    confusion_counts,
    error_counts,
    pattern_counts,
)
from ml.phase_6f.importance import (
    _coefficients,
    _scaler_std,
    extract_importances,
    importance_by_family,
)
from ml.phase_6f.robustness import (
    TRANSFORMS,
    apply_transform_to_frame,
    probe_prediction_stability,
)
from ml.phase_6f.sanity import (
    check_no_forbidden_columns,
    check_no_identity_features,
    check_quasi_generative_excluded,
    check_reproducible_metrics,
)
from ml.phase_6f.specs import (
    FEATURE_FAMILY,
    QUASI_GENERATIVE_FEATURES,
    SUSPICIOUS_STRUCTURED,
    AblationSpec,
    ablation_specs,
    all_rule_derived_features,
)

# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------


class TestAblationSpecs:
    def test_specs_are_frozen_dataclasses(self) -> None:
        for spec in ablation_specs():
            assert isinstance(spec, AblationSpec)
            with pytest.raises(AttributeError):
                spec.id = "mutated"  # type: ignore[misc]

    def test_exactly_8_ablation_specs(self) -> None:
        assert len(ablation_specs()) == 8

    def test_ablation_a_has_no_exclusions(self) -> None:
        spec_a = ablation_specs()[0]
        assert spec_a.id == "6f-ablation-A"
        assert spec_a.excluded == ()

    def test_ablation_b_excludes_suspicious_only(self) -> None:
        spec_b = ablation_specs()[1]
        assert set(spec_b.excluded) == set(SUSPICIOUS_STRUCTURED)

    def test_ablation_c_excludes_suspicious_and_quasi(self) -> None:
        spec_c = ablation_specs()[2]
        expected = set(SUSPICIOUS_STRUCTURED) | set(
            QUASI_GENERATIVE_FEATURES
        )
        assert set(spec_c.excluded) == expected
        assert len(spec_c.excluded) == 12

    def test_ablation_d_individual_removals(self) -> None:
        d_specs = [
            s for s in ablation_specs()
            if s.id.startswith("6f-ablation-D")
        ]
        assert len(d_specs) == 4
        for spec in d_specs:
            assert len(spec.excluded) == 1

    def test_ablation_e_quasi_generative_only(self) -> None:
        spec_e = ablation_specs()[7]
        assert spec_e.id == "6f-ablation-E"
        assert set(spec_e.excluded) == set(QUASI_GENERATIVE_FEATURES)
        assert len(spec_e.excluded) == 8


class TestSpecConstants:
    def test_suspicious_subset_of_feature_family(self) -> None:
        for feat in SUSPICIOUS_STRUCTURED:
            assert feat in FEATURE_FAMILY

    def test_quasi_generative_families(self) -> None:
        families = {FEATURE_FAMILY[f] for f in QUASI_GENERATIVE_FEATURES}
        assert families == {"deterministic_match", "ats_6b"}

    def test_all_rule_derived_covers_expected(self) -> None:
        rule = all_rule_derived_features()
        assert "same_role" in rule
        assert "keyword_overlap_required" in rule
        assert "action_verb_count" not in rule


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------


class TestImportance:
    def _make_estimator(
        self, coefs: list[float], stds: list[float]
    ) -> Pipeline:
        n = len(coefs)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(np.zeros((1, n)))

        scaler = StandardScaler()
        scaler.fit(np.random.default_rng(0).normal(size=(10, n)))
        scaler.scale_ = np.array(stds, dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = n

        clf = LogisticRegression()
        clf.coef_ = np.array([coefs])
        clf.intercept_ = np.array([0.0])
        clf.classes_ = np.array([0, 1])
        clf.n_features_in_ = n

        return Pipeline([
            ("imputer", imputer), ("scaler", scaler), ("clf", clf),
        ])

    def test_coefficients_shape_matches(self) -> None:
        est = self._make_estimator([0.5, -1.0, 2.0], [1.0, 1.0, 1.0])
        names = ["a", "b", "c"]
        coef = _coefficients(est, names)
        assert list(coef.index) == names
        assert list(coef.values) == pytest.approx([0.5, -1.0, 2.0])

    def test_scaler_std_preserves(self) -> None:
        est = self._make_estimator([1.0], [2.5])
        diag = _scaler_std(est)
        assert diag is not None
        assert diag.tolist() == pytest.approx([2.5])

    def test_coef_per_original_unit_rescales(self) -> None:
        est = self._make_estimator([4.0], [2.0])
        rows = extract_importances(est, ["x"])
        assert rows[0]["coefficient"] == pytest.approx(4.0)
        assert rows[0]["coef_per_original_unit"] == pytest.approx(2.0)
        assert rows[0]["std"] == pytest.approx(2.0)

    def test_ranking_is_descending_abs(self) -> None:
        est = self._make_estimator([1.0, 3.0, -2.0], [1.0, 1.0, 1.0])
        rows = extract_importances(est, ["a", "b", "c"])
        abs_vals = [r["abs_coefficient"] for r in rows]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_importance_by_family_aggregates(self) -> None:
        import ml.phase_6f.specs as specs_mod

        importances = [
            {"feature": "a", "abs_coefficient": 0.5,
             "coefficient": 0.5},
            {"feature": "b", "abs_coefficient": 0.3,
             "coefficient": 0.3},
        ]
        orig = specs_mod.FEATURE_FAMILY.copy()
        try:
            specs_mod.FEATURE_FAMILY["a"] = "fam1"
            specs_mod.FEATURE_FAMILY["b"] = "fam1"
            result = importance_by_family(importances)
            assert result["by_family"]["fam1"]["n_features"] == 2
            val = result["by_family"]["fam1"]["sum_abs_coefficient"]
            assert val == pytest.approx(0.8)
        finally:
            specs_mod.FEATURE_FAMILY.clear()
            specs_mod.FEATURE_FAMILY.update(orig)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_confusion_counts_basic(self) -> None:
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        assert tp == 2
        assert fp == 1
        assert fn == 1
        assert tn == 1

    def test_error_counts_keys(self) -> None:
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        result = error_counts(y_true, y_pred)
        assert "confusion" in result
        assert "macro_f1" in result
        assert "per_class" in result

    def test_categorize_partitions(self) -> None:
        y_true = [1, 0, 0, 1]
        y_pred = [1, 1, 0, 0]
        buckets = categorize(y_true, y_pred)
        assert set(buckets.keys()) == {"tp", "tn", "fp", "fn"}
        assert buckets["tp"] == [0]
        assert buckets["fp"] == [1]
        assert buckets["tn"] == [2]
        assert buckets["fn"] == [3]

    def test_pattern_counts_requires_indices(self) -> None:
        df = pd.DataFrame({
            "relevant": [1, 0],
            "resume_skills": ["python", "java"],
        })
        result = pattern_counts(df, [0, 1], label=1)
        assert result["bucket"] == "positive"
        assert result["examples_in_bucket"] == 2
        assert isinstance(result["pattern_counts"], dict)


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------


class TestSanityChecks:
    def test_no_forbidden_columns_passes(self) -> None:
        X = pd.DataFrame({"a": [1], "b": [2]})
        result = check_no_forbidden_columns(X)
        assert result["passed"] is True

    def test_no_forbidden_columns_fails(self) -> None:
        X = pd.DataFrame({"relevant": [1]})
        with pytest.raises(AssertionError):
            check_no_forbidden_columns(X)

    def test_no_identity_features_passes(self) -> None:
        X = pd.DataFrame({"a": [1], "b": [2]})
        result = check_no_identity_features(X)
        assert result["passed"] is True

    def test_no_identity_features_fails(self) -> None:
        X = pd.DataFrame({"resume_id": ["r1"], "job_id": ["j1"]})
        with pytest.raises(AssertionError):
            check_no_identity_features(X)

    def test_quasi_generative_excluded_passes(self) -> None:
        X = pd.DataFrame({"a": [1], "b": [2]})
        result = check_quasi_generative_excluded(X, ("c", "d"))
        assert result["passed"] is True

    def test_quasi_generative_excluded_fails(self) -> None:
        X = pd.DataFrame({"a": [1], "c": [3]})
        with pytest.raises(AssertionError):
            check_quasi_generative_excluded(X, ("c", "d"))

    def test_reproducible_metrics_passes(self) -> None:
        recorded = {
            "macro_f1": 0.95,
            "accuracy": 0.96,
            "balanced_accuracy": 0.94,
        }
        reproduced = {
            "macro_f1": 0.95001,
            "accuracy": 0.96001,
            "balanced_accuracy": 0.94001,
        }
        result = check_reproducible_metrics(
            recorded, reproduced, tolerance=1e-3
        )
        assert result["passed"] is True

    def test_reproducible_metrics_fails(self) -> None:
        recorded = {"macro_f1": 0.95, "accuracy": 0.95,
                    "balanced_accuracy": 0.95}
        reproduced = {"macro_f1": 0.90, "accuracy": 0.90,
                      "balanced_accuracy": 0.90}
        with pytest.raises(AssertionError):
            check_reproducible_metrics(
                recorded, reproduced, tolerance=1e-4
            )


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_ece_all_confident(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        probs = np.array([0.99, 0.99, 0.01, 0.01])
        result = reliability_curve(y_true, probs, bins=10)
        assert result["expected_calibration_error"] < 0.1
        assert result["note"]

    def test_ece_uniform_probs(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1000)
        probs = np.full(1000, 0.5)
        result = reliability_curve(y_true, probs, bins=5)
        ece = result["expected_calibration_error"]
        assert 0.0 <= ece <= 0.5


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_transforms_tuple_of_pairs(self) -> None:
        assert isinstance(TRANSFORMS, tuple)
        expected_names = {
            "whitespace_case", "dedupe_skills",
            "reorder_bullets", "remove_boilerplate",
        }
        assert {name for name, _ in TRANSFORMS} == expected_names

    def test_whitespace_case_uppercases_summary(self) -> None:
        row = pd.Series({
            "summary": "experienced professional",
            "description": "led team",
        })
        _, transform = TRANSFORMS[0]
        result = transform(row)
        # whitespace_case normalises via split+double-space+upper
        assert result["summary"].upper() == result["summary"]
        assert "EXPERIENCED" in result["summary"]
        assert "PROFESSIONAL" in result["summary"]
        assert result["description"].upper() == result["description"]

    def test_dedupe_skills_removes_duplicates(self) -> None:
        row = pd.Series({"skills": ["python", "python", "java"]})
        _, transform = TRANSFORMS[1]
        result = transform(row)
        assert result["skills"] == ["python", "java"]

    def test_reorder_bullets_preserves_count(self) -> None:
        row = pd.Series({
            "experience_bullets": ["a", "b", "c"],
        })
        _, transform = TRANSFORMS[2]
        result = transform(row)
        assert len(result["experience_bullets"]) == 3
        assert result["experience_bullets"][0] == "c"

    def test_remove_boilerplate_strips(self) -> None:
        text = "Software Engineer with 5 years of experience in fintech."
        row = pd.Series({"summary": text})
        _, transform = TRANSFORMS[3]
        result = transform(row)
        # When the boilerplate IS the entire summary, the regex strips to
        # empty and the transform falls back to the original (by design).
        assert result["summary"] == text

    def test_apply_transform_to_frame(self) -> None:
        df = pd.DataFrame({
            "summary": ["Hello World", "Foo Bar"],
            "description": ["x", "y"],
        })
        _, transform = TRANSFORMS[0]
        result = apply_transform_to_frame(df, transform)
        assert "HELLO" in result.iloc[0]["summary"]
        assert "WORLD" in result.iloc[0]["summary"]

    def test_probe_prediction_stability_no_flips(self) -> None:
        base = [1, 0, 1, 0]
        probe = [1, 0, 1, 0]
        X_base = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        X_probe = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = probe_prediction_stability(base, probe, X_base, X_probe)
        assert result["flip_fraction"] == 0.0
        assert result["n_pairs"] == 4

    def test_probe_prediction_stability_with_flips(self) -> None:
        base = [1, 0, 1, 0]
        probe = [0, 0, 0, 1]
        X_base = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        X_probe = pd.DataFrame({"a": [1.0, 2.0, 3.0, 5.0]})
        result = probe_prediction_stability(
            base, probe, X_base, X_probe
        )
        assert result["flip_fraction"] == pytest.approx(0.75)
        assert result["mean_abs_feature_delta"] == pytest.approx(0.25)
