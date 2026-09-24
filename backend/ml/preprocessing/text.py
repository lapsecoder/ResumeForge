"""Deterministic, stateless text normalisation.

Pipeline: NFKC -> collapse whitespace -> lowercase -> tokenise while
preserving technical terms (``C++`` -> ``c++``, ``.NET`` -> ``.net``,
``Node.js`` -> ``node.js``; the ``.`` stays inside the token).

Stateless by design: no vocabulary, no fitted object, so the same input
always maps to the same output. Learned representations (TF-IDF vectors,
embeddings) are applied on top of these tokens later and are strictly
fit-on-train.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def normalize(text: str) -> str:
    """NFKC-normalise and collapse all whitespace runs to single spaces."""
    nfkc = unicodedata.normalize("NFKC", text)
    return " ".join(nfkc.split())


def tokenize(text: str) -> list[str]:
    """Lowercase technical-term-preserving tokens of ``text``."""
    tokens = _TOKEN_RE.findall(normalize(text).lower())
    return [token for token in tokens if _has_alphanumeric(token)]


def _has_alphanumeric(token: str) -> bool:
    return any(character.isalnum() for character in token)


def unique_terms(text: str) -> list[str]:
    """Tokenised terms with duplicates removed while preserving first order."""
    seen: set[str] = set()
    terms: list[str] = []
    for term in tokenize(text):
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms
