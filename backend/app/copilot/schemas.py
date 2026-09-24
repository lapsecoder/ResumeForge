"""Transient Pydantic schemas for the Resume Copilot (Phase 7A).

These models exist only for the lifetime of a Copilot request. They are not
database tables, are not persisted to disk, and are garbage-collected after
the response is sent. No prompt, response, resume text, or job text is ever
stored or logged.

The contract is intentionally *controlled*: the client requests a finite set
of ``CopilotOperation`` values with structured inputs rather than an arbitrary
chat prompt. The one natural-language input — ``user_request`` on the
``free-form`` operation — is size-bounded, never treated as instructions, and
routed into the same task pipeline as the structured operations. Every
suggestion distinguishes grounding (``CopilotEvidence``) from proposed
wording, and carries a verification state so the UI can choose what to apply
directly and what needs the user's confirmation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.ats_analysis.job_specific.schemas import JobSpecificATSResult
from app.ats_analysis.schemas import ATSReadinessResult
from app.hybrid_matching.schemas import HybridMatchResult
from app.job_parsing.schemas import JobDescription
from app.matching.schemas import MatchResult
from app.parsing.schemas import Resume


class CopilotOperation(str, Enum):
    """The finite set of operations the Copilot can perform.

    Phase 7A defines the full contract for each operation; only the
    deterministic fallback executes them now. Phase 7B will build the
    local-LLM implementations on top of these contracts.
    """

    IMPROVE_SUMMARY = "improve_summary"
    IMPROVE_BULLET = "improve_bullet"
    IDENTIFY_PRIORITIES = "identify_priorities"
    EXPLAIN_FINDING = "explain_finding"
    JOB_ALIGNMENT = "job_alignment"
    FREE_FORM = "free-form"


class EvidenceKind(str, Enum):
    """Grounding of a single evidence statement.

    The system never silently converts an inference into a resume fact; every
    suggested claim or reworded bullet is labelled with the strongest grounding
    the pipeline actually has.
    """

    FACT = "fact"
    JOB_REQUIREMENT = "job_requirement"
    INFERENCE = "inference"
    SUGGESTION = "suggestion"
    UNVERIFIED = "unverified"


class VerificationLevel(str, Enum):
    """How confident the pipeline is that a suggestion can be applied as-is.

    - ``verified``: rewording only; every fact is preserved exactly.
    - ``inferred``: reasonable linguistic interpretation, no new factual claim.
    - ``advisory``: guidance only; no rewritten text is supplied.
    - ``unverified``: applying it would add information the user must confirm.
    """

    VERIFIED = "verified"
    INFERRED = "inferred"
    ADVISORY = "advisory"
    UNVERIFIED = "unverified"


class SuggestionCategory(str, Enum):
    """Deterministic category used by the UI to group suggestions."""

    SUMMARY = "summary"
    BULLET = "bullet"
    SKILLS = "skills"
    QUANTIFICATION = "quantification"
    EVIDENCE = "evidence"
    ALIGNMENT = "alignment"
    STRUCTURE = "structure"
    PRIORITY = "priority"
    EXPLANATION = "explanation"
    CLARITY = "clarity"
    ACTION_WORDING = "action_wording"


class ProviderKind(str, Enum):
    """Which engine produced the response."""

    LOCAL_OLLAMA = "local-ollama"
    DETERMINISTIC = "deterministic"


class EditStatus(str, Enum):
    """Lifecycle of a structured edit proposal (Phase 7C).

    The backend only ever emits ``proposed`` (safe to apply) or ``unverified``
    (needs the user's explicit confirmation). ``applied``/``reverted``/
    ``dismissed`` are transitions the UI performs on its in-memory working copy;
    they are never persisted or sent back as authoritative state.
    """

    PROPOSED = "proposed"
    APPLIED = "applied"
    REVERTED = "reverted"
    DISMISSED = "dismissed"
    UNVERIFIED = "unverified"


class EditCheckCategory(str, Enum):
    """Deterministic fact-preservation checks run against a proposed edit.

    Together these validate every fact class the resume can contain: numbers
    and metrics (:attr:`METRIC`), dates (:attr:`DATE`), skills and technologies
    (:attr:`SKILL`), and all remaining distinctive wording such as employers,
    job titles, education, certifications, project names, and locations
    (:attr:`SOURCE_TERM`).
    """

    METRIC = "metric"
    DATE = "date"
    SKILL = "skill"
    SOURCE_TERM = "source_term"


class CopilotEvidence(BaseModel):
    """One verifiable statement backing a suggestion.

    ``source`` names the structured origin (e.g. ``resume.skills``,
    ``job.required_skills``, ``ats.findings``, ``job_specific_ats``,
    ``deterministic_match``, ``copilot.inference``). ``reference`` points at
    the specific structured field path or finding rule id when available.
    ``quote`` is a short verbatim excerpt from the supplied materials (it is
    user content and is returned to the user, never logged or persisted).
    """

    kind: EvidenceKind
    source: str
    statement: str
    reference: str = Field(default="")
    quote: str = Field(default="", description="Short verbatim excerpt.")


class CopilotEditTarget(BaseModel):
    """A structured, server-resolved location in the resume that may be edited.

    The path is *never* supplied by the model. It is derived from the requested
    operation plus the client's ``target_ref``, then validated against the
    backend's explicit editable-field allowlist. ``path`` is human-readable
    (e.g. ``experience[0].achievements[1]``); the structured fields let the
    frontend apply the change without parsing path strings.
    """

    path: str = Field(..., description="Canonical readable path of the field.")
    section: str = Field(..., description="'summary' | 'experience' | 'projects'.")
    index: int | None = Field(
        default=None, description="Index within the section's collection."
    )
    field: str | None = Field(
        default=None, description="Field name within an entry, when applicable."
    )
    sub_index: int | None = Field(
        default=None, description="Index within a list field (e.g. achievements)."
    )


class CopilotClarificationOption(BaseModel):
    """One selectable choice for an ambiguous free-form edit request."""

    target: CopilotEditTarget = Field(
        ...,
        description=(
            "Allowlisted editable target that becomes the ``target_ref`` of a "
            "resubmitted copy of the original request."
        ),
    )
    label: str = Field(..., description="Short, human-readable label for the option.")


class CopilotClarification(BaseModel):
    """Structured 'which target did you mean?' prompt for ambiguous requests.

    Produced when a free-form edit request asks to rewrite text that inference
    cannot uniquely resolve (e.g. a technology shared by several projects). The
    backend never guesses: it offers the tied allowlisted targets, and the
    frontend resubmits the SAME request with the chosen ``target_ref``.
    """

    question: str
    reason: str = Field(
        default="",
        description="PII-safe explanation of why a choice is required.",
    )
    options: list[CopilotClarificationOption] = Field(
        ...,
        description="Tied allowlisted targets, ordered by resume position.",
    )


class EditCheck(BaseModel):
    """One deterministic fact-preservation check on a proposed edit value."""

    category: EditCheckCategory
    passed: bool
    detail: str = Field(
        default="",
        description="Human-readable, PII-safe explanation when the check fails.",
    )


class CopilotEditValidation(BaseModel):
    """Outcome of the fact-preservation checks for one proposed edit."""

    status: VerificationLevel
    requires_user_confirmation: bool
    checks: list[EditCheck] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)


class CopilotEditProposal(BaseModel):
    """A structured, explicit proposal to replace one resume value.

    An edit is never applied by the backend. It is offered to the user, who
    may preview, apply, revert, or dismiss it on a transient in-memory copy of
    the resume. ``original_value`` is resolved from the resume server-side (the
    source of truth), not taken from model output.
    """

    edit_id: str = Field(
        ..., description='Stable within the response, e.g. "edit_sug_1".'
    )
    operation: CopilotOperation
    target: CopilotEditTarget
    original_value: str = Field(
        default="", description="Server-resolved current value."
    )
    proposed_value: str = Field(default="", description="Validated replacement value.")
    reason: str = Field(default="", description="Why the change is suggested.")
    evidence: list[CopilotEvidence] = Field(default_factory=list)
    validation: CopilotEditValidation
    status: EditStatus = EditStatus.PROPOSED


class CopilotSuggestion(BaseModel):
    """A single suggestion produced by a Copilot provider.

    ``verification`` and ``requires_user_confirmation`` let the UI distinguish
    rewrites that are safe to apply directly from guidance or claims that need
    the user's own confirmation. ``priority``/``impact`` are populated only by
    ``identify_priorities`` so the UI can order and weight improvement advice.
    """

    id: str = Field(..., description='Stable within the response, e.g. "sug_1".')
    operation: CopilotOperation
    category: SuggestionCategory
    original_text: str = Field(
        default="", description="Source text from the resume, when applicable."
    )
    suggested_text: str = Field(
        default="",
        description="Proposed wording, or '' for advisory/guidance suggestions.",
    )
    rationale: str
    evidence: list[CopilotEvidence] = Field(default_factory=list)
    verification: VerificationLevel
    requires_user_confirmation: bool = False
    priority: int | None = Field(
        default=None,
        description="Rank within identify_priorities (1 = highest impact).",
    )
    issue: str = Field(
        default="", description="The concrete problem addressed (priorities)."
    )
    recommendation: str = Field(
        default="", description="Practical fix for the issue (priorities)."
    )
    impact: str = Field(
        default="", description="Expected weight of the fix (priorities)."
    )
    edit: CopilotEditProposal | None = Field(
        default=None,
        description=(
            "Structured edit proposal for replacement-text operations "
            "(Phase 7C). None for advisory/guidance suggestions and for "
            "targets outside the editable-field allowlist."
        ),
    )


class ProviderMetadata(BaseModel):
    """Non-sensitive facts describing how a Copilot response was produced.

    Provider logs contain only these safe fields — never resume/JD content,
    PII, or generated suggestions.
    """

    provider: ProviderKind
    provider_label: str
    model: str | None = Field(
        default=None, description="Local model name when an LLM produced the result."
    )
    available: bool
    fallback_used: bool
    version: str
    note: str = Field(
        default="",
        description="Non-sensitive explanation, e.g. 'Ollama unavailable; "
        "deterministic fallback used'.",
    )


class CopilotAnalysisContext(BaseModel):
    """Optional previously-computed ResumeForge analyses supplied with a request.

    The backend is stateless: the frontend already computed these in earlier
    steps (Phase 5A/5B/5C/6A/6B) and passes them back so the Copilot can
    reference real finding ids and skill evidence without re-computing or
    storing anything.
    """

    ats_readiness: ATSReadinessResult | None = None
    job_specific_ats: JobSpecificATSResult | None = None
    deterministic_match: MatchResult | None = None
    hybrid_match: HybridMatchResult | None = None


class CopilotRequest(BaseModel):
    """Structured request for one controlled Copilot operation.

    The primary contract is the ``operation`` enum plus the structured
    Resume/JobDescription objects. ``target_*`` and ``finding_ref`` select the
    specific item the operation should act on when it is not the whole resume.
    """

    operation: CopilotOperation
    resume: Resume
    job_description: JobDescription | None = None
    target_ref: str | None = Field(
        default=None,
        description=(
            "Structured path selecting the item to act on (e.g. 'summary', "
            "'experience[0].achievements[1]')."
        ),
    )
    target_text: str | None = Field(
        default=None,
        description=(
            "Bounded text to improve (e.g. a single bullet) when it is not yet "
            "in the resume. Never an arbitrary chat prompt."
        ),
    )
    finding_ref: str | None = Field(
        default=None,
        description="Finding rule id to explain (matches Phase 6A/6B findings).",
    )
    user_request: str | None = Field(
        default=None,
        description=(
            "Bounded free-text request (operation ``free-form``). Optional; "
            "trimmed, length-capped, and routed deterministically on the "
            "backend. Never treated as a prompt injection channel: it selects "
            "the task, it does not define the system behaviour."
        ),
    )
    analysis: CopilotAnalysisContext | None = None


class CopilotResponse(BaseModel):
    """Structured result of one Copilot operation.

    Only safe metadata (``provider``) and the suggestions are returned. No
    chain-of-thought or internal reasoning is ever exposed.
    """

    operation: CopilotOperation
    explanation: str
    suggestions: list[CopilotSuggestion]
    provider: ProviderMetadata
    disclaimer: str
    clarification: CopilotClarification | None = Field(
        default=None,
        description=(
            "Target-options prompt for ambiguous free-form edit requests; "
            "None otherwise."
        ),
    )


class ProviderStatus(BaseModel):
    """Availability of a single provider, for the status endpoint."""

    provider: ProviderKind
    provider_label: str
    model: str | None = None
    available: bool
    note: str = ""


class CopilotStatus(BaseModel):
    """Availability of every provider, used by the frontend to label state."""

    providers: list[ProviderStatus]
    fallback_available: bool = Field(
        description="True whenever the deterministic fallback can answer requests."
    )
    version: str
