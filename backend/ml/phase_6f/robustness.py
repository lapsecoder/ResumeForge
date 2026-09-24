"""Robustness probes for the 6F analysis.

Applies harmless formatting/content dilutions to a sample of test pairs and
records how often a model's prediction flips.  The intent is to flag
fragility to benign input variation, not to change any production code.

Transformations (in order of expected harmlessness):
  * ``whitespace/case``   - extra whitespace + uppercase text; the text
                            builders normalise case, so a no-op is expected.
  * ``dedupe skills``     - duplicate skills; set-based builders dedupe, so
                            a no-op is expected.
  * ``reorder bullets``   - reverse ``experience_bullets``; all builders are
                            order-insensitive.
  * ``remove boilerplate``- strip the template prefix "{role} with {n} years
                            of experience in {industry}"; this MAY change
                            text-derived features, so it is reported
                            separately and expected to flip rarely.

The probe returns per-transform flip counts and feature deltas on a fixed
sample.  It never retrains and never touches production code.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import pandas as pd

BOILERPLATE_RE = re.compile(
    r"^\s*[^.]*\bwith\s+\d+ years? of experience in\b[^.]*\.\s*"
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _transform_whitespace_case(
    row: pd.Series,
) -> pd.Series:
    out = row.copy()
    for col in ("summary", "description"):
        val = _text(row.get(col))
        out[col] = "  ".join(val.split()).upper() if val else val
    return out


def _transform_dedupe_skills(row: pd.Series) -> pd.Series:
    out = row.copy()
    skills = _as_list(row.get("skills"))
    out["skills"] = list(dict.fromkeys(skills))
    return out


def _transform_reorder_bullets(row: pd.Series) -> pd.Series:
    out = row.copy()
    out["experience_bullets"] = list(reversed(_as_list(row.get("experience_bullets"))))
    return out


def _transform_remove_boilerplate(row: pd.Series) -> pd.Series:
    out = row.copy()
    summary = _text(row.get("summary"))
    stripped = BOILERPLATE_RE.sub("", summary).strip()
    out["summary"] = stripped or summary
    return out


TRANSFORMS: tuple[tuple[str, Callable[[pd.Series], pd.Series]], ...] = (
    ("whitespace_case", _transform_whitespace_case),
    ("dedupe_skills", _transform_dedupe_skills),
    ("reorder_bullets", _transform_reorder_bullets),
    ("remove_boilerplate", _transform_remove_boilerplate),
)


def probe_prediction_stability(
    y_base: list[int],
    y_probe: list[int],
    x_base: pd.DataFrame,
    x_probe: pd.DataFrame,
) -> dict[str, Any]:
    """Count prediction flips and mean |delta| between two feature frames."""
    if len(y_base) != len(y_probe):
        raise ValueError("base and probe predictions must have equal length")
    if len(x_base) != len(x_probe):
        raise ValueError("base and probe feature frames must have equal length")
    n = len(y_base)
    if n == 0:
        raise ValueError("cannot probe an empty sample")
    flips = sum(1 for a, b in zip(y_base, y_probe, strict=True) if a != b)
    numeric_delta = x_base.select_dtypes(include="number") - x_probe.select_dtypes(
        include="number"
    )
    mean_abs_delta = 0.0
    if not numeric_delta.empty:
        mean_abs_delta = float(numeric_delta.abs().mean().mean())
    return {
        "n_pairs": n,
        "flipped": flips,
        "flip_fraction": round(flips / n, 6),
        "mean_abs_feature_delta": round(mean_abs_delta, 6),
    }


def apply_transform_to_frame(
    pairs: pd.DataFrame,
    transform: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Apply a row-level transform to every row, returning a fresh frame."""
    rows = [transform(row) for _, row in pairs.iterrows()]
    return pd.DataFrame(rows, index=pairs.index)
