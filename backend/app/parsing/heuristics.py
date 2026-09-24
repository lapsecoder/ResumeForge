"""Deterministic regex/heuristic helpers for resume parsing.

Conservative by design: these helpers extract only what can be matched with
high confidence and never invent missing information.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Loose but conservative: requires a 3-3-4 digit pattern.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
)

_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+", re.IGNORECASE
)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[\w-]+", re.IGNORECASE
)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)"
)
_MONTH_YEAR = rf"(?:{_MONTH}\s?\d{{4}})"
_MM_YYYY = r"\d{1,2}/\d{4}"
_YEAR = r"(?:19|20)\d{2}"
_DATE = rf"(?:{_MONTH_YEAR}|{_MM_YYYY}|{_YEAR})"

_DATE_RANGE_RE = re.compile(
    rf"({_DATE})\s*(?:-|–|—|to)\s*({_DATE}|present|current)", re.IGNORECASE
)
_DATE_RE = re.compile(rf"\b{_DATE}\b", re.IGNORECASE)


def find_email(text: str) -> str | None:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def find_phone(text: str) -> str | None:
    match = _PHONE_RE.search(text)
    return match.group(0).strip() if match else None


def find_linkedin(text: str) -> str | None:
    match = _LINKEDIN_RE.search(text)
    return _clean_url(match.group(0)) if match else None


def find_github(text: str) -> str | None:
    match = _GITHUB_RE.search(text)
    return _clean_url(match.group(0)) if match else None


def find_website(text: str) -> str | None:
    """Find a generic website that is not a LinkedIn/GitHub URL."""
    for match in _URL_RE.finditer(text):
        url = _clean_url(match.group(0))
        low = url.lower()
        if "linkedin.com" not in low and "github.com" not in low:
            return url
    return None


def find_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    return _clean_url(match.group(0)) if match else None


def find_date_range(text: str) -> tuple[str, str] | None:
    """Return (start, end) for a date range like ``Jun 2021 - Present``."""
    match = _DATE_RANGE_RE.search(text)
    if not match:
        return None
    start, end = match.group(1), match.group(2)
    return start, end


def ends_with_date_range(text: str) -> bool:
    """Whether a line ends with a complete date range (a header signature)."""
    match = _DATE_RANGE_RE.search(text)
    return bool(match and text.rstrip().endswith(match.group(0)))


def find_date(text: str) -> str | None:
    """Return the first single date token (``MM/YYYY``, ``Month YYYY``, ``YYYY``)."""
    match = _DATE_RE.search(text)
    return match.group(0) if match else None


def remove_dates(text: str) -> str:
    """Strip date ranges and single dates, collapsing leftover separators."""
    text = _DATE_RANGE_RE.sub("", text)
    text = _DATE_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().strip(" ,-–—")


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:)'\"")


_BULLET_RE = re.compile(r"^(?:[-*•◦▪–—>]\s+|\d+[.)]\s+)")


def is_bullet(line: str) -> bool:
    return bool(_BULLET_RE.match(line.strip()))


def strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line.strip()).strip()
