"""Deterministic regex/heuristic helpers for job-description parsing.

Conservative by design: only explicit labels and clear patterns are captured,
and nothing is ever invented. "Precision over aggressive inference".
"""

from __future__ import annotations

import re

from app.job_parsing.schemas import Salary
from app.parsing.heuristics import find_url, is_bullet, strip_bullet

# ---------------------------------------------------------------------------
# Label-based metadata (title / company / location / employment / remote)
# ---------------------------------------------------------------------------

_TITLE_LABEL_RE = re.compile(
    r"(?:job\s+)?(?:role|position|title|designation)\s*:", re.IGNORECASE
)
_COMPANY_LABEL_RE = re.compile(
    r"(?:hiring\s+)?(?:company|employer|organization|organisation)\s*(?:name\s*)?:",
    re.IGNORECASE,
)
_LOCATION_LABEL_RE = re.compile(
    r"(?:(?:office|work|job|workplace)\s+)?location\s*:", re.IGNORECASE
)
_EMPLOYMENT_LABEL_RE = re.compile(
    r"(?:(?:employment|job|position|engagement)\s+type|"
    r"type\s+of\s+(?:employment|position)|employment)\s*:",
    re.IGNORECASE,
)
_REMOTE_LABEL_RE = re.compile(
    r"(?:remote|work\s+arrangement|workplace\s+type|work\s+mode|"
    r"work\s+environment|location\s+type)\s*:",
    re.IGNORECASE,
)
_SALARY_LABEL_RE = re.compile(
    r"(?:salary\s+range|pay\s+range|annual\s+compensation|"
    r"compensation\s+package|salary|compensation|remuneration|ctc|pay)\s*:",
    re.IGNORECASE,
)

_EMPLOYMENT_CANONICAL: dict[str, str] = {
    "full-time": "Full-time",
    "fulltime": "Full-time",
    "part-time": "Part-time",
    "parttime": "Part-time",
    "contract": "Contract",
    "contractual": "Contract",
    "internship": "Internship",
    "intern": "Internship",
    "temporary": "Temporary",
    "temporarycontract": "Temporary",
    "freelance": "Freelance",
}

_REMOTE_CANONICAL: dict[str, str] = {
    "fully remote": "Fully remote",
    "100% remote": "Fully remote",
    "remote": "Remote",
    "hybrid": "Hybrid",
    "on-site": "On-site",
    "onsite": "On-site",
    "in-office": "On-site",
    "work from home": "Work from home",
    "work from anywhere": "Remote",
    "office-based": "Office-based",
    "office based": "Office-based",
}

_EMPLOYMENT_TOKEN_RE = re.compile(
    r"\b(?:full[- ]?time|part[- ]?time|contract(?:ual)?|"
    r"internship|intern|temporary|freelance)\b",
    re.IGNORECASE,
)
_REMOTE_TOKEN_RE = re.compile(
    r"\b(?:fully[- ]?remote|100\s*%\s*remote|work[- ]*from[- ]*home|"
    r"work[- ]from[- ]anywhere|hybrid|on[- ]?site|onsite|in[- ]office|"
    r"office[- ]based|remote)\b",
    re.IGNORECASE,
)


def find_labeled_value(text: str, label: re.Pattern[str]) -> str | None:
    """Return the value after the first ``Label:`` occurrence in ``text``.

    Lines may contain several ``Label: value`` pairs separated by ``|``.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for segment in re.split(r"\s*\|\s*", line):
            match = label.search(segment)
            if not match:
                continue
            value = _clean_value(segment[match.end() :])
            if value:
                return value
    return None


def _clean_value(value: str) -> str:
    value = strip_bullet(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t,;:.|")


def _employment_token(text: str) -> str | None:
    match = _EMPLOYMENT_TOKEN_RE.search(text)
    if not match:
        return None
    token = re.sub(r"\s+", "-", match.group(0).lower())
    return _EMPLOYMENT_CANONICAL.get(token, token.title())


def _remote_token(text: str) -> str | None:
    match = _REMOTE_TOKEN_RE.search(text)
    if not match:
        return None
    token = re.sub(r"\s+", " ", match.group(0).lower())
    for key, value in _REMOTE_CANONICAL.items():
        if re.search(rf"\b{re.escape(key)}\b", token):
            return value
    return None


def find_employment_type(header: str, full_text: str) -> str | None:
    """Recognise an explicit employment type (labels first, then header tokens)."""
    for text in (header, full_text):
        if not text:
            continue
        value = find_labeled_value(text, _EMPLOYMENT_LABEL_RE)
        if value:
            token = _employment_token(value)
            if token:
                return token
    for line in header.splitlines():
        token = _employment_token(line)
        if token:
            return token
    return None


def find_remote_type(header: str, full_text: str) -> str | None:
    """Recognise an explicit remote/hybrid/on-site statement."""
    for text in (header, full_text):
        if not text:
            continue
        value = find_labeled_value(text, _REMOTE_LABEL_RE)
        if value:
            if value.lower().strip() == "yes":
                return "Remote"
            token = _remote_token(value)
            if token:
                return token
    for line in header.splitlines():
        token = _remote_token(line)
        if token:
            return token
    return None


def find_company(header: str, full_text: str) -> str | None:
    """Return the explicitly labelled company, or None (never guessed)."""
    for text in (header, full_text):
        if not text:
            continue
        value = find_labeled_value(text, _COMPANY_LABEL_RE)
        if value:
            return value
    return None


def find_location(header: str, full_text: str) -> str | None:
    """Return the explicitly labelled location, or None when it is remote-like."""
    for text in (header, full_text):
        if not text:
            continue
        value = find_labeled_value(text, _LOCATION_LABEL_RE)
        if value:
            if _remote_token(value):
                return None
            return value
    return None


def find_salary_statements(header: str, full_text: str) -> list[Salary]:
    """Extract explicit salary/compensation statements.

    Only lines that clearly mention compensation (a salary hint or a labelled
    salary field) are included. Original wording is always preserved.
    """
    seen: set[str] = set()
    out: list[Salary] = []

    def add(line: str) -> None:
        text = line.strip()
        if not text or text in seen or not is_salary_line(text):
            return
        seen.add(text)
        out.append(salary_from_line(text))

    for line in header.splitlines():
        add(line)
    for line in full_text.splitlines():
        add(line)
    return out


_SALARY_HINT_RE = re.compile(
    r"\b(?:salary|compensation|remuneration|ctc|pay|stipend)\b|"
    r"[₹$€£]|\b(?:USD|INR|EUR|GBP)\b",
    re.IGNORECASE,
)
_PAY_RANGE_RE = re.compile(
    r"(?:[₹$€£])?\d[\d,]*(?:\.\d+)?\s*(?:k|K|lakh|lpa|L|crore|cr)?\s*"
    r"(?:to|-|–|—)\s*"
    r"(?:[₹$€£])?\d[\d,]*(?:\.\d+)?\s*(?:k|K|lakh|lpa|L|crore|cr)?"
)
_CURRENCY_RE = re.compile(r"[₹$€£]|\b(?:USD|INR|EUR|GBP)\b", re.IGNORECASE)
_ANNUAL_PERIOD_RE = re.compile(
    r"\b(?:p\.?a\.?|annual|per\s+year|/year|/yr|per\s+annum|annum|lpa|pa)\b",
    re.IGNORECASE,
)
_MONTHLY_PERIOD_RE = re.compile(
    r"\b(?:monthly|per\s+month|/month)\b", re.IGNORECASE
)
_HOURLY_PERIOD_RE = re.compile(
    r"\b(?:hourly|per\s+hour|/hour|/hr)\b", re.IGNORECASE
)


def is_salary_line(text: str) -> bool:
    if not re.search(r"\d", text):
        return False
    if _SALARY_HINT_RE.search(text):
        return True
    return bool(_PAY_RANGE_RE.search(text))


def salary_from_line(text: str) -> Salary:
    currency_match = _CURRENCY_RE.search(text)
    currency = currency_match.group(0).strip() if currency_match else None

    period: str | None = None
    if _ANNUAL_PERIOD_RE.search(text):
        period = "annual"
    elif _MONTHLY_PERIOD_RE.search(text):
        period = "monthly"
    elif _HOURLY_PERIOD_RE.search(text):
        period = "hourly"

    range_match = _PAY_RANGE_RE.search(text)
    range_text = range_match.group(0).strip() if range_match else None
    if range_text:
        range_text = re.sub(r"\s*(?:to|-|–|—)\s*", "–", range_text)
        range_text = re.sub(r"[₹$€£]", "", range_text)
        range_text = re.sub(r"(?<=\d),(?=\d)", "", range_text)

    return Salary(text=text, currency=currency, period=period, range_text=range_text)


# ---------------------------------------------------------------------------
# Requirement classification (degrees, experience, certifications, skills)
# ---------------------------------------------------------------------------

_DEGREE_TERMS = (
    r"b\.?a\.?",
    r"b\.?s\.?",
    r"b\.?e\.?",
    r"b\.?tech\.?",
    r"b\.?sc\.?",
    r"b\.?b\.?a\.?",
    r"bca",
    r"m\.?a\.?",
    r"m\.?s\.?",
    r"m\.?e\.?",
    r"m\.?tech\.?",
    r"m\.?sc\.?",
    r"mca",
    r"m\.?ba\.?",
    r"m\.?com\.?",
    r"b\.?com\.?",
    r"ph\.?d\.?",
    r"phd",
    r"bachelor(?:'s)?",
    r"master(?:'s)?",
    r"diploma",
)
_DEGREE_RE = re.compile(
    r"(?:^|[^\w])(" + "|".join(_DEGREE_TERMS) + r")(?:[\s.'/]|degree|$)",
    re.IGNORECASE,
)

_EXPERIENCE_RE = re.compile(
    r"(?:\d+\+?\s*(?:to|–|-)\s*\d+"
    r"|(?:minimum|at\s+least|min\.?|up\s+to|max(?:imum)?)\s*\d+"
    r"|\d+\+?)\s*(?:years?|yrs?\.?|yr\.?)\b",
    re.IGNORECASE,
)
_LEVEL_EXPERIENCE_RE = re.compile(
    r"\b(?:entry[- ]level|junior[- ]level|mid[- ]level|"
    r"senior[- ]level|senior[- ]level[-\s]experience|freshers?)\b",
    re.IGNORECASE,
)

_CERT_RE = re.compile(
    r"\b(?:certified|certifications?|certification)\b"
    r"|\b(?:licen[cs]e[ds]?)\b"
    r"|\b(?:PMP|CISSP|Security\+|CEH|CCNA|CCNP|CCIE|PRINCE2|CFA|CPA|LEED|"
    r"CKA|CKS|AWS Certified|Azure Certified)\b",
    re.IGNORECASE,
)


def has_degree(text: str) -> bool:
    return bool(_DEGREE_RE.search(text))


def is_experience_statement(text: str) -> bool:
    return bool(_EXPERIENCE_RE.search(text) or _LEVEL_EXPERIENCE_RE.search(text))


def is_cert_like(text: str) -> bool:
    return bool(_CERT_RE.search(text))


def is_skill_item(text: str) -> bool:
    """Whether a requirement-style item is a short skill entry rather than prose.

    Conservative: simple single-token entries and explicit comma/pipe/semicolon
    lists count as skills; anything sentence-like is left to qualifications.
    """
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return False
    if re.search(r"[.!?][\s)]+[A-Z]", cleaned):
        return False
    words = cleaned.split()
    if len(words) > 5:
        return False
    if re.search(r"[,|;]", cleaned):
        return True
    if len(words) == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# List splitting
# ---------------------------------------------------------------------------


def split_skill_list(body: str) -> list[str]:
    """Split a skills-section body into items, one per bullet/comma/pipe/semicolon."""
    items: list[str] = []
    for raw in body.splitlines():
        line = strip_bullet(raw)
        if not line:
            continue
        for part in re.split(r"[,|;]", line):
            item = part.strip().rstrip(".,;:")
            if item and item not in items:
                items.append(item)
    return items


def section_items(body: str) -> list[str]:
    """Return verbatim items for a section: bullets if present, else lines."""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    bullets = [strip_bullet(line) for line in lines if is_bullet(line)]
    bullets = [b for b in bullets if b]
    return bullets if bullets else lines


def looks_like_url_or_contact(text: str) -> bool:
    """Guard against treating contact/URL lines as headings or titles."""
    if find_url(text):
        return True
    return False
