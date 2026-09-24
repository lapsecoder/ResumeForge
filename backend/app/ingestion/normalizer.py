"""Deterministic text normalisation for extracted resume text.

Handles line-ending normalisation, excessive blank lines, repeated whitespace,
Unicode normalisation, and common extraction artefacts — while preserving
meaningful structure for downstream parsing.
"""

from __future__ import annotations

import re
import unicodedata

# Codepages that a resume text layer can be mis-decoded with. UTF-8 bytes for
# a typographic character decode to a stable mojibake triplet under each one
# (e.g. U+2022 -> "â€¢" under cp1252, "ΓÇó" under cp437). The two families
# use distinct character shapes (cp1252 lowercase â, cp437 uppercase Γ Ç) so
# all keys are unique and can be matched case-sensitively without collision.
_MOJIBAKE_CODEPAGES = ("cp1252", "cp437")


def _build_mojibake_repair() -> dict[str, str]:
    repair: dict[str, str] = {}
    for char in "—–\u201c\u201d\u2018\u2019…•":
        utf8 = char.encode("utf-8")
        for codepage in _MOJIBAKE_CODEPAGES:
            try:
                mojibake = utf8.decode(codepage)
            except UnicodeDecodeError:
                continue
            if "\ufffd" in mojibake:
                continue
            repair[mojibake] = char
    return repair


_MOJIBAKE_REPAIR = _build_mojibake_repair()
_MOJIBAKE_RE = re.compile(
    "|".join(map(re.escape, _MOJIBAKE_REPAIR)),
)


# pypdf renders unknown glyphs such as bullet points as the DEL control char.
_BULLET_DELIMITER_RE = re.compile(r"\x7f+")


def normalize_text(text: str) -> str:
    """Return normalised text ready for downstream resume parsing."""
    text = _normalize_line_endings(text)
    text = _repair_mojibake(text)
    text = _normalize_unicode(text)
    text = _replace_bullet_delimiters(text)
    text = _collapse_excessive_blank_lines(text)
    text = _strip_trailing_whitespace_per_line(text)
    text = text.strip()
    return text


def _normalize_line_endings(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def _repair_mojibake(text: str) -> str:
    return _MOJIBAKE_RE.sub(
        lambda match: _MOJIBAKE_REPAIR[match.group(0)], text
    )


def _replace_bullet_delimiters(text: str) -> str:
    return _BULLET_DELIMITER_RE.sub("•", text)


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _collapse_excessive_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def _strip_trailing_whitespace_per_line(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(line.rstrip() for line in lines)
