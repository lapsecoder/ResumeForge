"""Shared value-flattening helpers for audit modules.

Parquet list columns can surface as Python lists, tuples, sets, numpy
``ndarray`` (possibly nested) or a single scalar/None. All audit modules use
this single helper so row-level values are handled consistently.
"""

from __future__ import annotations

from typing import Any


def flatten_value(value: Any) -> list[Any]:
    """Flatten one cell into a plain list of items (empty list for None)."""
    if value is None:
        return []
    if _is_numeric_scalar(value):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "shape"):
        flat: list[Any] = []
        _stack(value, flat)
        return flat
    if isinstance(value, str):
        return [value]
    return [value]


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, complex, bool))


def _stack(value: Any, out: list[Any]) -> None:
    if _is_numeric_scalar(value):
        out.append(value)
        return
    if isinstance(value, str):
        out.append(value)
        return
    try:
        for item in value:
            _stack(item, out)
    except TypeError:
        out.append(value)
