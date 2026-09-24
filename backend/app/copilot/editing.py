"""Deterministic edit-proposal layer for the Resume Copilot (Phase 7C).

Phase 7B produces suggestions. Phase 7C turns the *replacement-text* subset of
those suggestions into explicit, structured **edit proposals** that a user may
preview, apply, revert, or dismiss on a transient in-memory copy of the resume.

Hard guarantees enforced here:

- The model never supplies a target path. A proposal's target is derived from
  the requested operation plus the client's ``target_ref`` — or, for free-form
  requests, deterministic inference from the request wording — and must parse
  against an explicit, hard-coded allowlist of editable fields.
- ``original_value`` is always resolved from the resume server-side (the
  source of truth), never taken from model output.
- Every proposed value runs through deterministic fact-preservation checks
  (metrics, dates, skills/technologies, and all remaining distinctive wording
  such as employers, titles, education, certifications, projects, locations).
  Anything uncertain is marked ``unverified`` and requires confirmation.
- The backend never mutates a resume. ``apply_edit`` is a pure function used by
  tests and the frontend's mirrored allowlist; the API only ever *proposes*.

Nothing here is persisted, logged, or scored.
"""

from __future__ import annotations

import re

from app.copilot.config import copilot_settings
from app.copilot.rewriting import apply_safe_rewrite
from app.copilot.schemas import (
    CopilotClarification,
    CopilotClarificationOption,
    CopilotEditProposal,
    CopilotEditTarget,
    CopilotEditValidation,
    CopilotEvidence,
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotSuggestion,
    EditCheck,
    EditCheckCategory,
    EditStatus,
    EvidenceKind,
    ProviderKind,
    SuggestionCategory,
    VerificationLevel,
)
from app.copilot.validation import (
    canonical_numbers,
    resume_fact_corpus,
    significant_term_tokens,
    unsupported_job_skills,
)
from app.job_parsing.schemas import JobDescription
from app.parsing.schemas import Project, Resume

#: Operations allowed to emit replacement-text edit proposals.
_EDITABLE_OPERATIONS = frozenset(
    {
        CopilotOperation.IMPROVE_SUMMARY,
        CopilotOperation.IMPROVE_BULLET,
        CopilotOperation.FREE_FORM,
    }
)

#: Readable target-path grammar. Intentionally narrow: no arbitrary JSON paths,
#: no contact fields, no filesystem-like segments.
_EDIT_PATH_RE = re.compile(
    r"^(summary|experience|projects)(?:\[(\d+)\])?(?:\.([a-z_]+)(?:\[(\d+)\])?)?$"
)

#: Scalar fields that may be replaced, keyed by section.
_EDITABLE_SCALAR_FIELDS: dict[str, frozenset[str]] = {
    "experience": frozenset({"title", "company", "description"}),
    "projects": frozenset({"description"}),
}

#: List fields that may be replaced wholesale (comma-separated value).
_EDITABLE_LIST_FIELDS: dict[str, frozenset[str]] = {
    "experience": frozenset({"achievements"}),
    "projects": frozenset({"technologies"}),
}

#: Accepted aliases for the achievements field.
_FIELD_ALIASES: dict[str, str] = {"bullets": "achievements"}

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "present",
        "current",
    }
)


# ---------------------------------------------------------------------------
# Target allowlist
# ---------------------------------------------------------------------------


def parse_edit_path(path: str) -> CopilotEditTarget | None:
    """Parse ``path`` into a structured target, or None if not allowlisted.

    Accepts exactly: ``summary``, ``experience[i].title|company|description``,
    ``experience[i].achievements[j]`` (alias ``bullets[j]``),
    ``projects[i].description``, ``projects[i].technologies``. Everything else —
    contact fields, skills, education, certifications, custom sections,
    metadata, arbitrary JSON, or malformed indices — returns None.
    """
    match = _EDIT_PATH_RE.match((path or "").strip())
    if match is None:
        return None
    section, raw_index, raw_field, raw_sub_index = match.groups()

    if section == "summary":
        if raw_index is not None or raw_field is not None:
            return None
        return CopilotEditTarget(path="summary", section="summary")

    if raw_index is None or raw_field is None:
        return None
    try:
        index = int(raw_index)
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None
    field = _FIELD_ALIASES.get(raw_field, raw_field)

    if section == "experience":
        if field in _EDITABLE_SCALAR_FIELDS["experience"]:
            if raw_sub_index is not None:
                return None
            return CopilotEditTarget(
                path=f"experience[{index}].{field}",
                section="experience",
                index=index,
                field=field,
            )
        if field in _EDITABLE_LIST_FIELDS["experience"]:
            if raw_sub_index is None:
                return None
            sub_index = int(raw_sub_index)
            return CopilotEditTarget(
                path=f"experience[{index}].{field}[{sub_index}]",
                section="experience",
                index=index,
                field=field,
                sub_index=sub_index,
            )
        return None

    if section == "projects":
        if raw_sub_index is not None:
            return None
        if field in _EDITABLE_SCALAR_FIELDS["projects"]:
            return CopilotEditTarget(
                path=f"projects[{index}].{field}",
                section="projects",
                index=index,
                field=field,
            )
        if field in _EDITABLE_LIST_FIELDS["projects"]:
            return CopilotEditTarget(
                path=f"projects[{index}].{field}",
                section="projects",
                index=index,
                field=field,
            )
        return None

    return None


def is_editable_target(target: CopilotEditTarget) -> bool:
    """True when ``target`` is allowlisted and internally consistent."""
    parsed = parse_edit_path(target.path)
    if parsed is None:
        return False
    return (
        parsed.section == target.section
        and parsed.index == target.index
        and parsed.field == target.field
        and parsed.sub_index == target.sub_index
    )


def canonical_path(target: CopilotEditTarget) -> str:
    """Return the canonical readable path for a target."""
    if target.section == "summary":
        return "summary"
    if target.field in _EDITABLE_LIST_FIELDS.get(target.section, frozenset()):
        if target.section == "experience":
            return f"experience[{target.index}].{target.field}[{target.sub_index}]"
        return f"{target.section}[{target.index}].{target.field}"
    return f"{target.section}[{target.index}].{target.field}"


def resolve_target_value(resume: Resume, target: CopilotEditTarget) -> str | None:
    """Return the current value at an allowlisted target, or None if unusable.

    ``summary`` resolves to "" when absent so a proposal may fill an empty
    summary; collection entries resolve only when the indices are in range.
    """
    if not is_editable_target(target):
        return None
    if target.section == "summary":
        return resume.summary or ""
    if target.index is None:
        return None

    if target.section == "experience":
        if target.index >= len(resume.experience):
            return None
        entry = resume.experience[target.index]
        if target.field in ("title", "company", "description"):
            return getattr(entry, target.field) or ""
        if target.field == "achievements":
            if target.sub_index is None or target.sub_index >= len(entry.achievements):
                return None
            return entry.achievements[target.sub_index]
        return None

    if target.section == "projects":
        if target.index >= len(resume.projects):
            return None
        project = resume.projects[target.index]
        if target.field == "description":
            return project.description or ""
        if target.field == "technologies":
            return ", ".join(project.technologies)
        return None

    return None


def target_for_request(request: CopilotRequest) -> CopilotEditTarget | None:
    """Derive the editable target for a request, or None when there is none.

    ``improve_summary`` targets ``summary``. ``improve_bullet`` targets the
    client's ``target_ref`` only when it is an allowlisted path; pasted custom
    text (no ref) is not in the resume, so it yields no edit proposal.
    ``free-form`` honours an explicit allowlisted ``target_ref`` (the
    client-selected "Act on (optional)" target) and otherwise falls back to
    deterministic inference from the request wording; requests with no clear
    editable target stay advisory.
    """
    if request.operation == CopilotOperation.IMPROVE_SUMMARY:
        return parse_edit_path("summary")
    explicit = parse_edit_path((request.target_ref or "").strip())
    if explicit is not None:
        return explicit
    if request.operation == CopilotOperation.FREE_FORM:
        return infer_edit_target(request.resume, request.user_request or "")
    return None


def description_target_for_request(request: CopilotRequest) -> CopilotEditTarget | None:
    """Resolve the editable long-form description target for a request.

    Returns the resolved target only when it points at an allowlisted
    ``description`` field (project or experience); everything else is None.
    """
    target = target_for_request(request)
    if target is not None and target.field == "description":
        return target
    return None


# ---------------------------------------------------------------------------
# Free-form target inference
# ---------------------------------------------------------------------------

#: Signs that the user is asking to change/reword text rather than for advice.
_EDIT_INTENT_RE = re.compile(
    r"\b(rewrit|improv|shorten|concise|polish|refin|tighten|strengthen|strong"
    r"|rephrase|reword|revise|updat|edit|clean|fix|writ|draft|creat|trim"
    r"|boost|refresh|sharpen|punch|make)\w*",
    re.IGNORECASE,
)

_SUMMARY_RE = re.compile(r"\b(summar(y|ies)|profile|objective)\b", re.IGNORECASE)

_BULLET_RE = re.compile(r"\b(bullet|bullet\s*point|achievement(s)?)\b", re.IGNORECASE)

_ORDINAL_WORDS = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
    "seventh": 6,
    "eighth": 7,
    "ninth": 8,
    "tenth": 9,
}

_ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    r"|\d{1,2}(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)

_EXPERIENCE_WORD_RE = re.compile(r"\bexperience\b", re.IGNORECASE)

_EXPERIENCE_CONTEXT_RE = re.compile(
    r"\b(role|position|job|experience|description)\b", re.IGNORECASE
)

#: Single-word job titles too generic to uniquely identify an experience entry
#: (e.g. a request mentioning an unrelated "Machine Learning Engineer role").
_COMMON_ROLE_WORDS = frozenset(
    {
        "engineer",
        "engineering",
        "analyst",
        "manager",
        "developer",
        "intern",
        "consultant",
        "specialist",
        "associate",
        "lead",
        "senior",
        "staff",
        "junior",
        "head",
        "director",
        "architect",
        "scientist",
        "researcher",
        "designer",
        "supervisor",
        "coordinator",
        "administrator",
        "assistant",
        "executive",
        "officer",
        "president",
    }
)

#: Tokens that carry no identifying signal when matching a project by name,
#: technology, or description text (both sides of the comparison).
_PROJECT_STOP = frozenset(
    {
        "project",
        "projects",
        "description",
        "technology",
        "technologies",
        "my",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "for",
        "with",
        "from",
        "into",
        "not",
        "all",
        "every",
        "already",
        "have",
        "has",
        "role",
        "roles",
        "section",
        "make",
        "improve",
        "rewrite",
        "stronger",
        "strongest",
        "strong",
        "engineer",
        "engineers",
        "engineering",
        "while",
        "preserving",
        "preserve",
        "fact",
        "facts",
        "adding",
        "add",
        "one",
        "using",
        "use",
        "about",
        "this",
        "that",
        "your",
        "resume",
        "job",
        "position",
        "to",
        "as",
        "it",
        "in",
        "on",
        "of",
        "at",
        "be",
        "being",
        "is",
        "are",
        "was",
        "were",
        "would",
        "should",
        "can",
        "could",
        "will",
        "more",
        "much",
        "better",
        "then",
        "than",
        "so",
        "just",
        "please",
        "very",
        "also",
        "really",
        "want",
        "need",
        "help",
        "there",
        "their",
        "how",
        "what",
        "why",
        "which",
        "who",
    }
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _ordinal_index(raw: str) -> int | None:
    lowered = raw.lower()
    if lowered in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[lowered]
    digits = re.match(r"\d+", lowered)
    if digits is None:
        return None
    zero_based = int(digits.group(0)) - 1
    return zero_based if zero_based >= 0 else None


def _company_matched_index(resume: Resume, text: str) -> int | None:
    """Index of the experience entry whose company/role appears in ``text``.

    Short values and single generic role words (e.g. "Engineer") are ignored:
    they appear far too often to identify a specific entry.
    """
    for index, entry in enumerate(resume.experience):
        for phrase in (entry.company or "", entry.title or ""):
            lowered = phrase.strip().lower()
            if not lowered or len(lowered) < 3:
                continue
            tokens = lowered.split()
            if len(tokens) == 1 and tokens[0] in _COMMON_ROLE_WORDS:
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(lowered) + r"(?![a-z0-9])", text):
                return index
    return None


def _project_candidate_indices(
    resume: Resume, text: str
) -> tuple[list[int], list[int]]:
    """Return ``(winners, ties)`` project indices matching the request.

    ``winners`` is ``[i]`` when exactly one project resolves (name-token
    overlap preferred over a union of name/technology/description tokens).
    ``ties`` holds the tied highest-scoring indices when matching is
    ambiguous — never non-empty alongside a winner. Both lists are orderless;
    callers sort when they need resume order.
    """
    request_tokens = {
        token for token in _tokens(text) - _PROJECT_STOP if len(token) > 1
    }
    if not request_tokens:
        return [], []

    def _name_overlap(project: Project) -> int:
        return len(request_tokens & (_tokens(project.name or "") - _PROJECT_STOP))

    name_scores = [_name_overlap(project) for project in resume.projects]
    best_name = max(name_scores, default=0)
    name_winners = [i for i, score in enumerate(name_scores) if score == best_name]
    if best_name > 0:
        if len(name_winners) == 1:
            return [name_winners[0]], []
        if len(name_winners) > 1:
            return [], name_winners

    scores: list[tuple[int, int]] = []
    for index, project in enumerate(resume.projects):
        project_tokens = (
            _tokens(project.name or "")
            | set().union(*(_tokens(tech) for tech in project.technologies))
            | _tokens(project.description or "")
        ) - _PROJECT_STOP
        scores.append((len(request_tokens & project_tokens), index))
    best_score = max((score for score, _index in scores), default=0)
    winners = [index for score, index in scores if score == best_score]
    if best_score > 0:
        if len(winners) == 1:
            return [winners[0]], []
        if len(winners) > 1:
            return [], winners
    return [], []


def infer_edit_target(resume: Resume, user_request: str) -> CopilotEditTarget | None:
    """Infer a safe editable target from a free-form request, or None.

    The request must ask to change/reword specific text (an edit intent) and
    resolve to EXACTLY ONE allowlisted target. Ambiguous wording, advice
    requests, and non-editable sections (e.g. skills) return None so the
    request is handled advisory/clarification, never guessed.

    Supported resolutions: ``summary``; ``projects[i].description`` (matched by
    name/technology/description tokens); a specifically ordered achievement
    bullet ``experience[i].achievements[n]``; or ``experience[i].description``
    when the user names an experience section or a matching employer/role.
    """
    text = (user_request or "").lower()
    if not _EDIT_INTENT_RE.search(text):
        return None

    candidates: set[str] = set()

    if _SUMMARY_RE.search(text):
        candidates.add("summary")

    if re.search(r"\bproject\b", text):
        winners, _ties = _project_candidate_indices(resume, text)
        if len(winners) == 1:
            candidates.add(f"projects[{winners[0]}].description")

    bullet_requested = _BULLET_RE.search(text) is not None
    if bullet_requested:
        ordinal_match = _ORDINAL_RE.search(text)
        if ordinal_match is not None:
            bullet_index = _ordinal_index(ordinal_match.group(1))
        else:
            bullet_index = None
        if bullet_index is not None:
            entry_index = _company_matched_index(resume, text)
            if entry_index is None and len(resume.experience) == 1:
                entry_index = 0
            if entry_index is not None:
                achievements = resume.experience[entry_index].achievements
                if bullet_index < len(achievements):
                    candidates.add(
                        f"experience[{entry_index}].achievements[{bullet_index}]"
                    )

    if not bullet_requested:
        experience_index: int | None = None
        if _EXPERIENCE_WORD_RE.search(text):
            experience_index = _company_matched_index(resume, text)
            if experience_index is None and len(resume.experience) == 1:
                experience_index = 0
        else:
            matched = _company_matched_index(resume, text)
            if matched is not None and _EXPERIENCE_CONTEXT_RE.search(text):
                experience_index = matched
        if experience_index is not None:
            candidates.add(f"experience[{experience_index}].description")

    if len(candidates) != 1:
        return None
    return parse_edit_path(next(iter(candidates)))


def edit_target_options(request: CopilotRequest) -> list[CopilotEditTarget]:
    """Candidate allowlisted targets when a free-form edit is ambiguous.

    Returns the tied ``projects[i].description`` targets when the request asks
    to rewrite a *project* but no project uniquely matches (e.g. the wording
    names a technology that several projects share). The service surfaces these
    as an explicit user choice — the system never guesses which one.

    Empty for everything else: non-free-form operations, requests with an
    explicit ``target_ref``/``target_text``, non-edit advice, requests naming
    another section (summary/bullet/experience) as well, and requests where
    inference already resolves a unique project.
    """
    if request.operation != CopilotOperation.FREE_FORM:
        return []
    if (request.target_ref or "").strip() or (request.target_text or "").strip():
        return []
    text = (request.user_request or "").lower()
    if not _EDIT_INTENT_RE.search(text):
        return []
    if not re.search(r"\bproject\b", text):
        return []
    if _SUMMARY_RE.search(text) or _BULLET_RE.search(text):
        return []
    if _EXPERIENCE_WORD_RE.search(text):
        return []
    if _company_matched_index(request.resume, text) is not None:
        return []
    _winners, ties = _project_candidate_indices(request.resume, text)
    if not ties:
        return []

    targets: list[CopilotEditTarget] = []
    for index in sorted(set(ties)):
        target = parse_edit_path(f"projects[{index}].description")
        if target is not None:
            targets.append(target)
    return targets


def _project_option_label(resume: Resume, target: CopilotEditTarget) -> str:
    """Short human-readable label for one project clarification option."""
    if target.section != "projects" or target.index is None:
        return "Project"
    project = resume.projects[target.index]
    name = (project.name or "").strip()
    techs = [value.strip() for value in project.technologies if value.strip()]
    head = name if name else f"Project {target.index + 1}"
    if techs:
        head = f"{head} — {', '.join(techs[:4])}"
    if project.description:
        snippet = re.sub(r"\s+", " ", project.description).strip()
        if len(snippet) > 80:
            snippet = f"{snippet[:77].rstrip()}\u2026"
        head = f"{head} · {snippet}"
    return head[:200]


def build_target_clarification(
    request: CopilotRequest, options: list[CopilotEditTarget]
) -> CopilotClarification:
    """Structured 'which project did you mean?' prompt for ambiguous requests.

    The user picks one option; the frontend resubmits the same request with
    that option's path as the explicit ``target_ref`` so the normal
    fact-preserving EditProposal flow runs on exactly that section.
    """
    return CopilotClarification(
        question="Which project's description should I rewrite?",
        reason=(
            "Your request names a project by a skill or technology that "
            "appears in several projects. Choose one below to get a "
            "fact-safe rewrite proposal for exactly that project."
        ),
        options=[
            CopilotClarificationOption(
                target=target, label=_project_option_label(request.resume, target)
            )
            for target in options
        ],
    )


# ---------------------------------------------------------------------------
# Fact preservation
# ---------------------------------------------------------------------------


def validate_edit_value(
    resume: Resume,
    job: JobDescription | None,
    suggestion: CopilotSuggestion,
    original: str,
    proposed: str,
) -> CopilotEditValidation:
    """Run the deterministic fact-preservation checks for one edit value."""
    corpus = resume_fact_corpus(resume)
    if original:
        corpus = f"{original}\n{corpus}"
    corpus_numbers = canonical_numbers(corpus)
    corpus_years = _years(corpus)
    proposed_numbers = canonical_numbers(proposed)
    proposed_years = _years(proposed)

    unsupported_metrics = (proposed_numbers - proposed_years) - (
        corpus_numbers - corpus_years
    )
    unsupported_dates = _date_tokens(proposed) - _date_tokens(corpus)
    unsupported_skills = unsupported_job_skills(proposed, resume, job)
    new_terms = {
        token
        for token in significant_term_tokens(proposed) - significant_term_tokens(corpus)
        if not any(character.isdigit() for character in token)
    }

    checks = [
        EditCheck(
            category=EditCheckCategory.METRIC,
            passed=not unsupported_metrics,
            detail=(
                ""
                if not unsupported_metrics
                else "The proposed text adds numeric values that do not appear "
                "in your resume."
            ),
        ),
        EditCheck(
            category=EditCheckCategory.DATE,
            passed=not unsupported_dates,
            detail=(
                ""
                if not unsupported_dates
                else "The proposed text adds a date that does not appear in "
                "your resume."
            ),
        ),
        EditCheck(
            category=EditCheckCategory.SKILL,
            passed=not unsupported_skills,
            detail=(
                ""
                if not unsupported_skills
                else "The proposed text claims job skills your resume does not "
                "show: " + ", ".join(sorted(unsupported_skills)) + "."
            ),
        ),
        EditCheck(
            category=EditCheckCategory.SOURCE_TERM,
            passed=not new_terms,
            detail=(
                ""
                if not new_terms
                else "The proposed text adds wording not present in your resume "
                "(for example a different employer, job title, organisation, "
                "project, school, certification, or location)."
            ),
        ),
    ]
    failed = any(not check.passed for check in checks)
    base = suggestion.verification
    if failed or base not in (VerificationLevel.VERIFIED, VerificationLevel.INFERRED):
        status = VerificationLevel.UNVERIFIED
        requires_confirmation = True
    else:
        status = base
        requires_confirmation = suggestion.requires_user_confirmation
    return CopilotEditValidation(
        status=status,
        requires_user_confirmation=requires_confirmation,
        checks=checks,
    )


def _years(text: str) -> set[str]:
    result: set[str] = set()
    for year in _YEAR_RE.findall(text):
        result |= canonical_numbers(year) or {year}
    return result


def _date_tokens(text: str) -> set[str]:
    tokens = _years(text)
    lowered = text.lower()
    for month in _MONTHS:
        if re.search(r"(?<![a-z])" + month + r"(?![a-z])", lowered):
            tokens.add(month)
    return tokens


# ---------------------------------------------------------------------------
# Proposal construction
# ---------------------------------------------------------------------------


def build_edit_proposal(
    request: CopilotRequest, suggestion: CopilotSuggestion
) -> CopilotEditProposal | None:
    """Build a validated edit proposal for one suggestion, or None."""
    if suggestion.operation not in _EDITABLE_OPERATIONS:
        return None
    if not suggestion.suggested_text.strip():
        return None
    target = target_for_request(request)
    if target is None:
        return None
    original = resolve_target_value(request.resume, target)
    if original is None:
        return None

    proposed = suggestion.suggested_text.strip()[
        : copilot_settings.max_edit_value_chars
    ]
    validation = validate_edit_value(
        request.resume, request.job_description, suggestion, original, proposed
    )
    status = (
        EditStatus.PROPOSED
        if validation.status in (VerificationLevel.VERIFIED, VerificationLevel.INFERRED)
        and not validation.requires_user_confirmation
        else EditStatus.UNVERIFIED
    )
    return CopilotEditProposal(
        edit_id=f"edit_{suggestion.id}",
        operation=suggestion.operation,
        target=target,
        original_value=original,
        proposed_value=proposed,
        reason=suggestion.rationale,
        evidence=list(suggestion.evidence),
        validation=validation,
        status=status,
    )


def attach_edit_proposals(
    request: CopilotRequest, response: CopilotResponse
) -> CopilotResponse:
    """Attach at most one edit proposal per distinct target to a response.

    Only one edit per target path is emitted (first wins) so the UI can never
    present two conflicting replacements for the same field. Suggestions that
    are not editable are returned unchanged with ``edit=None``.
    """
    seen_targets: set[str] = set()
    updated: list[CopilotSuggestion] = []
    for suggestion in response.suggestions:
        proposal = build_edit_proposal(request, suggestion)
        if proposal is not None:
            path = canonical_path(proposal.target)
            if path in seen_targets:
                proposal = None
            else:
                seen_targets.add(path)
        updated.append(
            suggestion.model_copy(update={"edit": proposal})
            if proposal is not None
            else suggestion
        )
    return response.model_copy(update={"suggestions": updated})


def section_stack_for_target(resume: Resume, target: CopilotEditTarget) -> list[str]:
    """Skills/technologies already stated on the resume for a section."""
    if target.section == "projects":
        if target.index is None or target.index >= len(resume.projects):
            return []
        return [
            value.strip()
            for value in resume.projects[target.index].technologies
            if value.strip()
        ]
    if target.section == "experience":
        if target.index is None or target.index >= len(resume.experience):
            return []
        entry = resume.experience[target.index]
        stack = [value.strip() for value in entry.skills_mentioned if value.strip()]
        if not stack:
            stack = [
                value.strip() for value in resume.skills.technical if value.strip()
            ]
        return stack
    return []


def deterministic_description_suggestion(
    request: CopilotRequest,
    target: CopilotEditTarget,
    *,
    suggestion_id: str = "sug_1",
) -> CopilotSuggestion | None:
    """Strengthen a description using only facts already stated on the resume.

    Surface-fixes the wording (first-person subject, capitalisation) and, when
    the section's own resume-stated stack is not already named, surfaces it.
    Nothing is invented: appended items come from the resume itself. Returns
    None when nothing fact-safe changed.
    """
    original = (
        resolve_target_value(request.resume, target)
        or (request.target_text or "").strip()
    )
    if not original:
        return None

    base = apply_safe_rewrite(original) or original
    surface_changed = base != original

    stack = section_stack_for_target(request.resume, target)
    missing = [value for value in stack if value.lower() not in base.lower()]
    appended = ""
    if missing:
        label = "Key skills" if target.section == "experience" else "Key technologies"
        separator = " " if base.endswith((".", "?", "!")) else ". "
        appended = f"{separator}{label}: {', '.join(missing[:8])}."
    strengthened = f"{base}{appended}"
    if strengthened == original:
        return None

    rationale_parts: list[str] = []
    if surface_changed:
        rationale_parts.append(
            "removed a leading first-person subject and normalised the opening"
        )
    if appended:
        rationale_parts.append(
            "surfaced the skills/technologies already listed for this section "
            "on your resume"
        )
    rationale = (
        "Deterministic rewrite: "
        + "; ".join(rationale_parts)
        + ". Every fact is preserved and no skill or technology was added "
        "that is not already on your resume."
    )
    return CopilotSuggestion(
        id=suggestion_id,
        operation=request.operation,
        category=SuggestionCategory.CLARITY,
        original_text=original[:250],
        suggested_text=strengthened[:400],
        rationale=rationale,
        evidence=[
            CopilotEvidence(
                kind=EvidenceKind.FACT,
                source="copilot.inference",
                statement=(
                    "Original wording preserved; appended items are taken "
                    "from the resume itself."
                ),
                reference=target.path,
                quote=original[:120],
            )
        ],
        verification=VerificationLevel.INFERRED,
        requires_user_confirmation=bool(appended),
    )


def reconcile_description_edit(
    request: CopilotRequest, response: CopilotResponse
) -> CopilotResponse:
    """Keep LLM-path proposals for description targets fact-preserving.

    On the LLM path, rewrite text for a description target becomes an editable
    proposal only when every fact-preservation check passes (nothing new needs
    the user's confirmation). When the model invents facts — or rewrote a
    different field than the resolved description — its text stays advisory
    and the deterministic fact-preserving description rewrite is attached
    instead, so an edit-intent free-form request still yields exactly one safe
    EditProposal. Advice-only requests (no resolved target) are unchanged.
    """
    if request.operation != CopilotOperation.FREE_FORM:
        return response
    if (
        response.provider is None
        or response.provider.provider != ProviderKind.LOCAL_OLLAMA
    ):
        return response
    target = description_target_for_request(request)
    if target is None:
        return response
    path = canonical_path(target)

    for suggestion in response.suggestions:
        edit = suggestion.edit
        if (
            edit is not None
            and canonical_path(edit.target) == path
            and not edit.validation.requires_user_confirmation
        ):
            return response

    updated = [
        suggestion.model_copy(update={"edit": None})
        if (
            suggestion.edit is not None
            and canonical_path(suggestion.edit.target) == path
        )
        else suggestion
        for suggestion in response.suggestions
    ]
    fallback = deterministic_description_suggestion(
        request,
        target,
        suggestion_id=f"sug_det_{len(updated) + 1}",
    )
    if fallback is None:
        return response.model_copy(update={"suggestions": updated})
    proposal = build_edit_proposal(request, fallback)
    if proposal is None:
        return response.model_copy(update={"suggestions": updated})
    return response.model_copy(
        update={
            "suggestions": updated + [fallback.model_copy(update={"edit": proposal})]
        }
    )


# ---------------------------------------------------------------------------
# Pure apply/revert primitives (frontend mirrors this allowlist)
# ---------------------------------------------------------------------------


def apply_edit(resume: Resume, target: CopilotEditTarget, value: str) -> Resume | None:
    """Return a new resume with ``target`` set to ``value``, or None if invalid.

    Never mutates the input. Returns None for non-allowlisted targets,
    out-of-range indices, or values over ``max_edit_value_chars``. The frontend
    keeps the same allowlist so a tampered target cannot be applied.
    """
    if not is_editable_target(target):
        return None
    if value is None or len(value) > copilot_settings.max_edit_value_chars:
        return None
    if target.section != "summary":
        if target.index is None or target.index < 0:
            return None
    if not _target_in_range(resume, target):
        return None

    updated = resume.model_copy(deep=True)
    if target.section == "summary":
        updated.summary = value
    elif target.section == "experience":
        entry = updated.experience[target.index]  # type: ignore[index]
        if target.field in ("title", "company", "description"):
            setattr(entry, target.field, value)
        elif target.field == "achievements":
            entry.achievements[target.sub_index] = value  # type: ignore[index]
    elif target.section == "projects":
        project = updated.projects[target.index]  # type: ignore[index]
        if target.field == "description":
            project.description = value
        elif target.field == "technologies":
            project.technologies = [
                item.strip() for item in value.split(",") if item.strip()
            ]
    return updated


def _target_in_range(resume: Resume, target: CopilotEditTarget) -> bool:
    if target.section == "summary":
        return True
    if target.index is None or target.index < 0:
        return False
    if target.section == "experience":
        if target.index >= len(resume.experience):
            return False
        if target.field == "achievements":
            if target.sub_index is None or target.sub_index < 0:
                return False
            return target.sub_index < len(resume.experience[target.index].achievements)
        return target.field in _EDITABLE_SCALAR_FIELDS["experience"]
    if target.section == "projects":
        if target.index >= len(resume.projects):
            return False
        return target.field in (
            _EDITABLE_SCALAR_FIELDS["projects"] | _EDITABLE_LIST_FIELDS["projects"]
        )
    return False
