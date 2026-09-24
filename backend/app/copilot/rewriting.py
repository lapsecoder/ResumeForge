"""Safety-preserving deterministic rewrites for the Resume Copilot.

A rewrite only ever changes surface wording whose factual meaning is
invariant to the transformation: stripping a leading first-person subject,
fixing a missing subject-capitalisation, or trimming empty phrasing. If a
transform would change (or guess) a fact, the rule returns ``None`` so the
caller falls back to a confirmation request instead of silently rewriting.
"""

from __future__ import annotations

import re

_FIRST_PERSON_RE = re.compile(r"^(?:i|my|me|mine|we|our|ours|us)\s+", re.IGNORECASE)


def strip_first_person(text: str) -> str | None:
    """Drop a leading first-person subject and re-capitalise the stem.

    Returns None when the text does not start with a first-person pronoun
    (no transformation is safe, so the caller must not produce a rewrite).
    """
    stripped = text.strip()
    if not stripped or not _FIRST_PERSON_RE.match(stripped):
        return None
    remainder = _FIRST_PERSON_RE.sub("", stripped, count=1).lstrip()
    if not remainder:
        return None
    return remainder[0].upper() + remainder[1:]


def capitalise_start(text: str) -> str | None:
    """Capitalise a text that starts with a lowercase alpha token.

    Used for bullet rewrites after subject stripping. Returns None when the
    first character is already a capital letter or is not alphabetic.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if not stripped[0].isalpha() or stripped[0].isupper():
        return None
    return stripped[0].upper() + stripped[1:]


def trim_excess_whitespace(text: str) -> str | None:
    """Collapse runs of spaces; return None when nothing changed."""
    collapsed = " ".join(text.split())
    return collapsed if collapsed != text else None


def apply_safe_rewrite(text: str) -> str | None:
    """Compose the safe deterministic bullet transforms in order.

    Returns the rewritten bullet, or ``None`` when no safe transformation
    applied. Where several transforms could apply they are applied in a fixed,
    tested order.
    """
    current = text.strip()
    transformed = strip_first_person(current)
    if transformed is not None:
        current = transformed
    capitalised = capitalise_start(current)
    if capitalised is not None:
        current = capitalised
    trimmed = trim_excess_whitespace(current)
    if trimmed is not None:
        current = trimmed
    if current == text.strip():
        return None
    return current
