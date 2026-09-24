"""Pure-function tests for the hybrid scoring formula (Phase 5C).

Validates the exact 70/30 blend, the documented fallbacks (deterministic-only,
semantic-only, no-evidence), boundary scores, and the no-NaN / no-divide-by-zero
invariant. No model library is required.
"""

from __future__ import annotations

import math

import pytest

from app.hybrid_matching.scoring import (
    HYBRID_WEIGHTS,
    evidence_quality,
    hybrid_overall,
    semantic_on_100,
)


class TestBlendFormula:
    def test_exact_7030_blend(self) -> None:
        overall, mode = hybrid_overall(80.0, 0.6)
        assert mode == "hybrid"
        assert overall == pytest.approx(0.70 * 80.0 + 0.30 * 60.0)

    def test_full_match_at_100(self) -> None:
        overall, mode = hybrid_overall(100.0, 1.0)
        assert mode == "hybrid"
        assert overall == pytest.approx(100.0)

    def test_zero_deterministic_zero_semantic(self) -> None:
        overall, mode = hybrid_overall(0.0, 0.0)
        assert mode == "hybrid"
        assert overall == pytest.approx(0.0)

    def test_semantic_lifts_a_zero_deterministic_only_partially(self) -> None:
        overall, mode = hybrid_overall(0.0, 1.0)
        assert mode == "hybrid"
        assert overall == pytest.approx(30.0)

    def test_weights_are_documented(self) -> None:
        assert HYBRID_WEIGHTS == {"deterministic": 0.70, "semantic": 0.30}

    def test_both_ends_of_the_range_are_finite(self) -> None:
        for det, sem in ((0.0, 0.0), (100.0, 1.0), (3.3, 0.07), (99.9, 0.99)):
            overall, _ = hybrid_overall(det, sem)
            assert overall is not None
            assert math.isfinite(overall)
            assert 0.0 <= overall <= 100.0


class TestFallbacks:
    def test_deterministic_only_returns_deterministic_untouched(self) -> None:
        overall, mode = hybrid_overall(66.7, None)
        assert mode == "deterministic-only"
        assert overall == 66.7

    def test_semantic_only_is_limited_not_a_match_score(self) -> None:
        overall, mode = hybrid_overall(None, 0.8)
        assert mode == "semantic-only"
        assert overall is None

    def test_no_evidence_is_none(self) -> None:
        overall, mode = hybrid_overall(None, None)
        assert mode == "no-evidence"
        assert overall is None


class TestScaleMapping:
    def test_semantic_maps_onto_0_100(self) -> None:
        assert semantic_on_100(0.5) == 50.0
        assert semantic_on_100(0.0) == 0.0
        assert semantic_on_100(1.0) == 100.0


class TestEvidenceQuality:
    def test_high_needs_both_signals_and_two_or_more_criteria(self) -> None:
        assert evidence_quality("hybrid", 2, True) == "high"
        assert evidence_quality("hybrid", 5, True) == "high"

    def test_medium_when_only_one_signal_is_rich(self) -> None:
        assert evidence_quality("hybrid", 1, True) == "medium"
        assert evidence_quality("deterministic-only", 2, True) == "medium"
        assert evidence_quality("deterministic-only", 1, False) == "medium"
        assert evidence_quality("semantic-only", 0, True) == "medium"

    def test_limited_when_no_meaningful_evidence(self) -> None:
        assert evidence_quality("no-evidence", 0, False) == "limited"


class TestNeverInvalid:
    def test_no_nan_from_partial_signals(self) -> None:
        for det, sem in ((None, None), (None, 0.0), (0.0, None), (100.0, None)):
            overall, _ = hybrid_overall(det, sem)
            assert overall is None or math.isfinite(overall)

    def test_no_division_by_zero_in_blend(self) -> None:
        overall, mode = hybrid_overall(0.0, 1.0)
        assert isinstance(overall, float) and math.isfinite(overall)
