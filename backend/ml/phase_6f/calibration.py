"""Calibration diagnostics for the 6F analysis.

Logistic regression with ``class_weight="balanced"`` produces a well-ranked
but not frequency-calibrated score; the diagnostics here quantify that on a
frozen test split.  Used for the report's calibration section and to reason
about whether the score is a safe "probability" (it is not, on its own).
"""

from __future__ import annotations

from typing import Any, Sequence


def reliability_curve(
    y_true: Sequence[int],
    probs_positive: Sequence[float],
    bins: int = 10,
) -> dict[str, Any]:
    """Equal-width reliability: predicted-mean vs actual-positive-rate.

    ``bins`` buckets of width 1/bins over the probability range are used
    (equal-width rather than equal-frequency so the curve is readable).  Only
    bins with at least one sample are reported.
    """
    if len(y_true) != len(probs_positive):
        raise ValueError("y_true and probs_positive must have equal length")
    if len(y_true) == 0:
        raise ValueError("cannot compute calibration on an empty set")

    rows: list[dict[str, Any]] = []
    total = len(y_true)
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        indices = [
            j for j, p in enumerate(probs_positive) if lo <= p < hi
        ]
        if not indices:
            continue
        pred_mean = sum(probs_positive[j] for j in indices) / len(indices)
        pos_rate = sum(1 for j in indices if y_true[j] == 1) / len(indices)
        rows.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "count": len(indices),
            "predicted_mean": round(pred_mean, 6),
            "actual_positive_rate": round(pos_rate, 6),
            "abs_error": round(abs(pred_mean - pos_rate), 6),
        })

    expected_calibration_error = sum(
        row["abs_error"] * row["count"] for row in rows
    ) / total

    return {
        "bins": rows,
        "expected_calibration_error": round(expected_calibration_error, 6),
        "note": (
            "Diagnostic only.  The phase-6E heads use class_weight='balanced'; "
            "the score must NOT be interpreted as a hiring probability."
        ),
    }
