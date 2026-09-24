"""Pure-math tests for the deterministic cosine similarity utilities.

No model library is needed: vectors are plain Python list[float].
"""

from __future__ import annotations

import math

import pytest

from app.semantic_matching.similarity import (
    DimensionMismatchError,
    cosine_similarity,
    normalized_similarity,
    similarity_level,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0

    def test_near_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 1e-9]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_zero_vector_returns_zero_not_nan(self) -> None:
        assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
        value = cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert value == value  # not NaN

    def test_positive_scalar_multiple_is_identical(self) -> None:
        assert cosine_similarity([2.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_negative_scaling_is_opposite(self) -> None:
        assert cosine_similarity([2.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_magnitude_irrelevant(self) -> None:
        assert cosine_similarity([100.0, 0.0], [0.0, 5.0]) == 0.0

    def test_clipped_against_float_error(self) -> None:
        value = cosine_similarity([1e8, 1e8], [1e8, 1e8])
        assert -1.0 <= value <= 1.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(DimensionMismatchError):
            cosine_similarity([1.0, 0.0], [1.0])

    def test_deterministic(self) -> None:
        a = [0.3, -0.7, 0.2]
        b = [0.1, 0.4, 0.9]
        assert cosine_similarity(a, b) == pytest.approx(
            cosine_similarity(list(a), list(b))
        )


class TestNormalizedSimilarity:
    def test_maps_cosine_to_unit_interval(self) -> None:
        assert normalized_similarity(1.0) == 1.0
        assert normalized_similarity(-1.0) == 0.0
        assert normalized_similarity(0.0) == 0.5

    def test_clipped_outside_range(self) -> None:
        assert normalized_similarity(2.0) == 1.0
        assert normalized_similarity(-2.0) == 0.0

    def test_not_a_probability_but_scaling(self) -> None:
        assert normalized_similarity(0.5) == pytest.approx(0.75)


class TestSimilarityLevel:
    def test_bucketing_matches_documented_thresholds(self) -> None:
        assert similarity_level(0.70, high=0.65, moderate=0.40) == "high"
        assert similarity_level(0.65, high=0.65, moderate=0.40) == "high"
        assert similarity_level(0.50, high=0.65, moderate=0.40) == "moderate"
        assert similarity_level(0.40, high=0.65, moderate=0.40) == "moderate"
        assert similarity_level(0.10, high=0.65, moderate=0.40) == "low"

    def test_math_helpers_are_floats(self) -> None:
        assert not any(
            math.isnan(v)
            for v in (
                cosine_similarity([0.0, 0.0], [1.0, 1.0]),
                normalized_similarity(-0.2),
            )
        )
