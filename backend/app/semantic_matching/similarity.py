"""Deterministic cosine similarity utilities.

Implemented in pure Python so the unit suite runs without numpy/torch.
No NaN, no division by zero: zero vectors score 0.0 by convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

SimilarityLevel = Literal["high", "moderate", "low"]


class DimensionMismatchError(ValueError):
    """Raised when comparing vectors of different lengths."""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length dense vectors.

    Returns a float in [-1.0, 1.0] (clipped against float error). Zero or
    mismatched-length vectors yield 0.0, never NaN or an exception.
    """
    if len(a) != len(b):
        raise DimensionMismatchError("vector dimension mismatch")
    if not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    value = dot / denom
    return max(-1.0, min(1.0, value))


def normalized_similarity(cosine: float) -> float:
    """Map cosine [-1, 1] into [0, 1]; kept distinct from raw cosine.

    This is a scaling, not a probability.
    """
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def similarity_level(
    normalized: float,
    *,
    high: float,
    moderate: float,
) -> SimilarityLevel:
    """Bucket a normalized score into high / moderate / low."""
    if normalized >= high:
        return "high"
    if normalized >= moderate:
        return "moderate"
    return "low"
