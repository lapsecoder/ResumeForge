"""Deterministic text normalisation for matching.

Skills are compared on a canonical form that is case-insensitive, unicode-
normalised and whitespace-collapsed, while deliberately PRESERVING meaningful
punctuation that is part of a technology name (``C++``, ``C#``, ``.NET``,
``Node.js``, ``React.js``).

Only a small, documented set of obvious abbreviations is canonicalised via an
exact whole-string alias map. Nothing else is guessed.
"""

from __future__ import annotations

import re
import unicodedata

_SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "postgres": "postgresql",
    "golang": "go",
    "k8s": "kubernetes",
}

_WS_COLLAPSE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[.:;,]+$")

_SIGNIFICANT_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "with",
        "for",
        "have",
        "has",
        "must",
        "will",
        "work",
        "able",
        "ability",
        "strong",
        "good",
        "required",
        "requirement",
        "requirements",
        "experience",
        "knowledge",
        "knowledgeable",
        "degree",
        "skills",
        "skill",
        "role",
        "position",
        "you",
        "etc",
    }
)

#: Public stopword set reused by other analyzers (e.g. phase 6B phrase mining).
STOPWORDS: frozenset[str] = _SIGNIFICANT_STOPWORDS


def normalized_base(value: str) -> str:
    """Return the literal normalised form of a skill, WITHOUT alias expansion.

    Same normalisation steps as ``normalize_skill`` (unicode, case,
    whitespace, trailing punctuation) but never maps abbreviations like
    ``js`` — use ``alias_target`` for that.
    """
    text = unicodedata.normalize("NFKC", value).strip()
    text = text.lower()
    text = _WS_COLLAPSE.sub(" ", text).strip()
    text = _TRAILING_PUNCT.sub("", text).strip()
    return text


def alias_target(value: str) -> str | None:
    """Return the canonical expansion when ``value`` is a known alias.

    Returns ``None`` for non-alias terms so callers can distinguish a plain
    exact/normalised match from an alias-expanded one.
    """
    base = normalized_base(value)
    return _SKILL_ALIASES.get(base) if base else None


def normalize_skill(value: str) -> str:
    """Return the canonical comparison key for a single skill.

    1. Unicode normalise (NFKC collapses full-width forms etc.).
    2. Lowercase.
    3. Collapse repeated whitespace and trim.
    4. Strip surplus trailing ``.``/``:``/``;``/``,`` separators only.
    5. Apply the exact whole-string alias map.
    """
    base = normalized_base(value)
    if not base:
        return ""
    return _SKILL_ALIASES.get(base, base)


def normalize_phrase(value: str) -> str:
    """Whitespace/unicode/case normalised phrase key (punctuation preserved)."""
    text = unicodedata.normalize("NFKC", value).lower()
    text = _WS_COLLAPSE.sub(" ", text).strip()
    text = _TRAILING_PUNCT.sub("", text).strip()
    return text


def significant_tokens(value: str) -> list[str]:
    """Split a phrase into its byte-significant tokens for containment checks."""
    text = normalize_phrase(value)
    tokens = [t for t in re.findall(r"[a-z0-9+#.]+", text) if len(t) >= 3]
    return [t for t in tokens if t not in _SIGNIFICANT_STOPWORDS]
