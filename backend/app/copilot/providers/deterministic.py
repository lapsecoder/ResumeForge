"""Deterministic fallback provider for the Resume Copilot (Phase 7A).

This provider answers every supported operation using fixed rules over the
structured resume, job, and supplied ResumeForge analyses. It is always
available, requires no Ollama, no internet, and no GPU, and it never pretends
to be a generative model: its metadata clearly reports
``provider = deterministic`` and ``fallback_used = true``.

It also cannot fabricate facts by construction: every suggestion is assembled
from either (a) an existing finding/term match, (b) an existing resume fact,
or (c) a fixed guidance template that never introduces a concrete value.
"""

from __future__ import annotations

from app.ats_analysis.heuristics import (
    collect_lines,
    has_quantitative,
    is_action_start,
    is_vague,
)
from app.copilot.config import (
    COPILOT_DISCLAIMER,
    COPILOT_IMPLEMENTATION_VERSION,
    copilot_settings,
)
from app.copilot.editing import (
    description_target_for_request,
    deterministic_description_suggestion,
)
from app.copilot.errors import InvalidRequestError
from app.copilot.evidence import (
    finding_by_rule,
    job_specific_finding_by_rule,
    matched_terms,
    missing_terms,
    ordered_findings,
    resume_skill_names,
)
from app.copilot.operations import get_contract, requires_job, requires_target
from app.copilot.rewriting import apply_safe_rewrite
from app.copilot.schemas import (
    CopilotAnalysisContext,
    CopilotEvidence,
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotSuggestion,
    EvidenceKind,
    ProviderKind,
    ProviderMetadata,
    SuggestionCategory,
    VerificationLevel,
)
from app.job_parsing.schemas import JobDescription
from app.parsing.schemas import Resume

_MAX_MATCHED_FOR_ALIGNMENT = 5

#: ATS finding category -> Copilot suggestion category. Keeps the
#: identify_priorities list accurately labelled instead of dumping everything
#: into a single bucket.
_ATS_CATEGORY_MAP: dict[str, SuggestionCategory] = {
    "contact": SuggestionCategory.STRUCTURE,
    "structure": SuggestionCategory.STRUCTURE,
    "experience": SuggestionCategory.BULLET,
    "bullets": SuggestionCategory.ACTION_WORDING,
    "quantified": SuggestionCategory.QUANTIFICATION,
    "skills": SuggestionCategory.SKILLS,
    "education": SuggestionCategory.STRUCTURE,
    "projects_certs": SuggestionCategory.EVIDENCE,
    "dates": SuggestionCategory.STRUCTURE,
    "parsing": SuggestionCategory.STRUCTURE,
    "required_coverage": SuggestionCategory.SKILLS,
    "preferred_coverage": SuggestionCategory.ALIGNMENT,
    "phrase_coverage": SuggestionCategory.ALIGNMENT,
    "evidence_representation": SuggestionCategory.EVIDENCE,
    "job_specific": SuggestionCategory.ALIGNMENT,
}

_HIGH_IMPACT_FLOOR = 5.0
_MEDIUM_IMPACT_FLOOR = 2.0


def _impact_label(impact: float) -> str:
    if impact >= _HIGH_IMPACT_FLOOR:
        return "high"
    if impact >= _MEDIUM_IMPACT_FLOOR:
        return "medium"
    return "low"


class DeterministicFallbackProvider:
    """Rule-based provider that mirrors the Copilot contract offline."""

    def provider_kind(self) -> ProviderKind:
        return ProviderKind.DETERMINISTIC

    def available(self) -> bool:
        return True

    def supports(self, request: CopilotRequest) -> bool:
        contract = get_contract(request.operation)
        if not contract.fallback_available:
            return False
        if requires_job(request.operation) and request.job_description is None:
            return False
        if requires_target(request.operation) and not request.target_text:
            return False
        if (
            request.operation == CopilotOperation.FREE_FORM
            and not (request.user_request or "").strip()
        ):
            return False
        return True

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider=ProviderKind.DETERMINISTIC,
            provider_label="Deterministic fallback (offline rules)",
            model=None,
            available=True,
            fallback_used=True,
            version=COPILOT_IMPLEMENTATION_VERSION,
            note=(
                "Deterministic rule-based suggestions. No generative model was "
                "used; claims are grounded in the supplied structured data."
            ),
        )

    def run(self, request: CopilotRequest) -> CopilotResponse:
        if not self.supports(request):
            raise InvalidRequestError(
                "The deterministic provider cannot run this operation with the "
                "given inputs."
            )
        operation = request.operation
        if operation == CopilotOperation.IMPROVE_SUMMARY:
            suggestions, explanation = self._improve_summary(request)
        elif operation == CopilotOperation.IMPROVE_BULLET:
            suggestions, explanation = self._improve_bullet(request)
        elif operation == CopilotOperation.IDENTIFY_PRIORITIES:
            suggestions, explanation = self._identify_priorities(request)
        elif operation == CopilotOperation.EXPLAIN_FINDING:
            suggestions, explanation = self._explain_finding(request)
        elif operation == CopilotOperation.JOB_ALIGNMENT:
            suggestions, explanation = self._job_alignment(request)
        elif operation == CopilotOperation.FREE_FORM:
            suggestions, explanation = self._free_form(request)
        else:  # pragma: no cover - guarded by supports()
            raise InvalidRequestError("Unsupported operation for fallback.")

        assert request.job_description is not None or not requires_job(operation)
        return CopilotResponse(
            operation=operation,
            explanation=explanation,
            suggestions=suggestions,
            provider=self.metadata(),
            disclaimer=COPILOT_DISCLAIMER,
        )

    # ------------------------------------------------------------------
    # Sum & helper builders
    # ------------------------------------------------------------------

    def _summary(self, explanations: list[str]) -> str:
        return " ".join(explanations).strip() or (
            "The Copilot analysed the supplied materials. No deterministic "
            "action surfaced beyond the suggestions below."
        )

    def _base_explanation(self, request: CopilotRequest) -> str:
        return (
            f"Operation '{request.operation.value}' performed with the "
            "deterministic fallback provider."
        )

    def _evidence(
        self,
        kind: EvidenceKind,
        source: str,
        statement: str,
        reference: str = "",
        quote: str = "",
    ) -> CopilotEvidence:
        return CopilotEvidence(
            kind=kind,
            source=source,
            statement=statement,
            reference=reference,
            quote=quote,
        )

    # ------------------------------------------------------------------
    # improve_summary
    # ------------------------------------------------------------------

    def _draft_fact_summary(self, resume: Resume) -> str | None:
        """Assemble a summary draft using ONLY facts already in the resume."""
        parts: list[str] = []
        if resume.experience:
            entry = resume.experience[0]
            title = entry.title.strip()
            company = entry.company.strip()
            if title and company:
                parts.append(f"{title} with experience at {company}.")
            elif title:
                parts.append(f"{title}.")
            elif company:
                parts.append(f"Professional with experience at {company}.")
        skills = list(
            dict.fromkeys(
                value.strip()
                for value in (*resume.skills.technical, *resume.skills.tools)
                if value.strip()
            )
        )
        if skills:
            parts.append("Core skills: " + ", ".join(skills[:6]) + ".")
        if not parts:
            return None
        return " ".join(parts)

    def _improve_summary(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        resume = request.resume
        job = request.job_description
        suggestions: list[CopilotSuggestion] = []
        summary = (resume.summary or "").strip()

        if not summary:
            draft = self._draft_fact_summary(resume)
            if draft:
                suggestions.append(
                    self._suggestion(
                        request.operation,
                        1,
                        SuggestionCategory.SUMMARY,
                        original="",
                        suggested=draft,
                        rationale=(
                            "Your resume has no professional summary. This "
                            "draft is assembled only from facts already in "
                            "your resume (your most recent role and listed "
                            "skills); edit it freely and add a real outcome "
                            "if you have one."
                        ),
                        evidence=[
                            self._evidence(
                                EvidenceKind.FACT,
                                "resume.experience",
                                "Role and employer taken from the resume.",
                                "resume.experience",
                            ),
                            self._evidence(
                                EvidenceKind.FACT,
                                "resume.skills",
                                "Skills taken from the resume's skill list.",
                                "resume.skills",
                            ),
                        ],
                        verification=VerificationLevel.INFERRED,
                    )
                )
            else:
                suggestions.append(
                    self._suggestion(
                        request.operation,
                        1,
                        SuggestionCategory.SUMMARY,
                        original="",
                        suggested="",
                        rationale=(
                            "Your resume has no professional summary. Add 2–3 "
                            "sentences covering your role, the technologies or "
                            "domains you work in, and one notable real outcome. "
                            "Do not invent numbers; describe scope honestly."
                        ),
                        evidence=[
                            self._evidence(
                                EvidenceKind.FACT,
                                "resume.summary",
                                "No summary section is present in the resume.",
                                "resume.summary",
                            ),
                            self._evidence(
                                EvidenceKind.SUGGESTION,
                                "copilot.suggestion",
                                "Add a professional summary if you want one.",
                            ),
                        ],
                        verification=VerificationLevel.ADVISORY,
                    )
                )
        else:
            rewrite = apply_safe_rewrite(summary)
            if rewrite is not None and rewrite != summary:
                suggestions.append(
                    self._suggestion(
                        request.operation,
                        1,
                        SuggestionCategory.CLARITY,
                        original=summary,
                        suggested=rewrite,
                        rationale=(
                            "Removed a leading first-person subject and "
                            "normalised the opening. Every factual claim is "
                            "unchanged."
                        ),
                        evidence=[
                            self._evidence(
                                EvidenceKind.FACT,
                                "resume.summary",
                                "Original summary wording, factually preserved.",
                                "resume.summary",
                                summary[:120],
                            )
                        ],
                        verification=VerificationLevel.VERIFIED,
                    )
                )
            if len(summary.split()) < 30:
                suggestions.append(
                    self._suggestion(
                        request.operation,
                        len(suggestions) + 1,
                        SuggestionCategory.SUMMARY,
                        original=summary[:120],
                        suggested="",
                        rationale=(
                            "Your summary is short. Consider a sentence about "
                            "the scope of the work you own and one concrete "
                            "outcome you can actually demonstrate — without "
                            "inventing metrics."
                        ),
                        evidence=[
                            self._evidence(
                                EvidenceKind.FACT,
                                "resume.summary",
                                f"Summary is about {len(summary.split())} words.",
                                "resume.summary",
                            )
                        ],
                        verification=VerificationLevel.ADVISORY,
                    )
                )
            if job is not None:
                skill_defaults = [
                    skill
                    for skill in resume_skill_names(resume)
                    if _job_mentions_skill(job, skill)
                ]
                unseen = [
                    skill for skill in skill_defaults if _mentions(summary, skill)
                ]
                if unseen and len(suggestions) < copilot_settings.max_suggestions:
                    suggestions.append(
                        self._suggestion(
                            request.operation,
                            len(suggestions) + 1,
                            SuggestionCategory.ALIGNMENT,
                            original=summary[:120],
                            suggested="",
                            rationale=(
                                "These skills appear in both your resume and "
                                "the job description. Consider naming them in "
                                "the summary if they are central to the role."
                            ),
                            evidence=[
                                self._evidence(
                                    EvidenceKind.FACT,
                                    "resume.skills",
                                    "Skill is listed in the resume.",
                                    "resume.skills",
                                ),
                                self._evidence(
                                    EvidenceKind.JOB_REQUIREMENT,
                                    "job.required_skills",
                                    "Skill appears in the job requirements.",
                                    "job.skills",
                                ),
                            ],
                            verification=VerificationLevel.INFERRED,
                        )
                    )

        if not suggestions:
            suggestions.append(
                self._suggestion(
                    request.operation,
                    1,
                    SuggestionCategory.SUMMARY,
                    original=summary[:120],
                    suggested="",
                    rationale=(
                        "Your summary is clear and contains a factual basis. "
                        "No safety-preserving automated rewrite was needed."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume.summary",
                            "Summary present and free of the deterministic "
                            "issues the Copilot checks.",
                            "resume.summary",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        return suggestions, self._base_explanation(request)

    # ------------------------------------------------------------------
    # improve_bullet
    # ------------------------------------------------------------------

    def _improve_bullet(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        text = (request.target_text or "").strip()
        suggestions: list[CopilotSuggestion] = []

        rewrite = apply_safe_rewrite(text)
        if rewrite is not None:
            suggestions.append(
                self._suggestion(
                    request.operation,
                    1,
                    SuggestionCategory.CLARITY,
                    original=text,
                    suggested=rewrite,
                    rationale=(
                        "Deterministic rewrite: removed a leading first-person "
                        "subject and capitalised the opening. No fact was "
                        "changed."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "copilot.inference",
                            "Original wording, factually preserved in full.",
                            quote=text[:120],
                        )
                    ],
                    verification=VerificationLevel.VERIFIED,
                )
            )

        if is_vague(text):
            suggestions.append(
                self._suggestion(
                    request.operation,
                    len(suggestions) + 1,
                    SuggestionCategory.CLARITY,
                    original=text,
                    suggested="",
                    rationale=(
                        "The bullet relies on vague phrasing such as "
                        "'responsible for', 'worked on', or 'helped with'. "
                        "Replace it with the specific action you took and the "
                        "outcome you observed."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.INFERENCE,
                            "copilot.inference",
                            "Bullet contains vague, low-information markers.",
                            quote=text[:120],
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )

        if not has_quantitative(text) and not is_vague(text):
            suggestions.append(
                self._suggestion(
                    request.operation,
                    len(suggestions) + 1,
                    SuggestionCategory.QUANTIFICATION,
                    original=text,
                    suggested="",
                    rationale=(
                        "Add a measurable result here if you have one (for "
                        "example, the time saved, users affected, or volume "
                        "handled). Do not invent a number — only real values "
                        "you can stand behind."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.UNVERIFIED,
                            "copilot.inference",
                            "No quantitative value is present; a real figure "
                            "would strengthen the bullet if one exists.",
                            quote=text[:120],
                        )
                    ],
                    verification=VerificationLevel.UNVERIFIED,
                    requires_confirmation=True,
                )
            )

        if not is_action_start(text) and rewrite is None:
            suggestions.append(
                self._suggestion(
                    request.operation,
                    len(suggestions) + 1,
                    SuggestionCategory.ACTION_WORDING,
                    original=text,
                    suggested="",
                    rationale=(
                        "Start the bullet with an action verb such as 'Led', "
                        "'Built', 'Reduced', or 'Designed' for stronger, "
                        "scannable wording."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.INFERENCE,
                            "copilot.inference",
                            "Bullet does not open with a recognised action verb.",
                            quote=text[:120],
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )

        if not suggestions:
            suggestions.append(
                self._suggestion(
                    request.operation,
                    1,
                    SuggestionCategory.BULLET,
                    original=text,
                    suggested="",
                    rationale=(
                        "This bullet is clear, specific, and begins with an "
                        "action verb. No safety-preserving automated rewrite "
                        "is needed."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "copilot.inference",
                            "Bullet passed the deterministic quality checks.",
                            quote=text[:120],
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        return suggestions, self._base_explanation(request)

    # ------------------------------------------------------------------
    # identify_priorities
    # ------------------------------------------------------------------

    def _identify_priorities(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        analysis = request.analysis
        findings = ordered_findings(
            analysis.ats_readiness if analysis else None,
            analysis.job_specific_ats if analysis else None,
        )
        suggestions: list[CopilotSuggestion] = []
        index = 0

        for finding in findings:
            if index >= copilot_settings.max_suggestions:
                break
            if finding.impact == 0 and finding.severity.value == "info":
                continue
            index += 1
            suggestions.append(
                self._suggestion(
                    request.operation,
                    index,
                    _ATS_CATEGORY_MAP.get(
                        finding.category, SuggestionCategory.PRIORITY
                    ),
                    original="",
                    suggested="",
                    rationale=(f"{finding.title}. {finding.recommendation}"),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "ats.findings",
                            finding.explanation,
                            finding.rule_id,
                            finding.evidence[:120],
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                    priority=index,
                    issue=finding.title,
                    recommendation=finding.recommendation,
                    impact=_impact_label(finding.impact),
                )
            )

        if request.job_description is not None:
            missing = self._missing_skill_labels(request, analysis)
            for label in missing:
                if index >= copilot_settings.max_suggestions:
                    break
                index += 1
                suggestions.append(
                    self._suggestion(
                        request.operation,
                        index,
                        SuggestionCategory.SKILLS,
                        original="",
                        suggested="",
                        rationale=(
                            f"The job lists '{label}' as a requirement that "
                            "was not found explicitly in the resume. If you "
                            "have real experience with it, make sure it "
                            "appears; do not add it if you have not used it."
                        ),
                        evidence=[
                            self._evidence(
                                EvidenceKind.JOB_REQUIREMENT,
                                "job.required_skills",
                                f"The job explicitly asks for '{label}'.",
                            ),
                            self._evidence(
                                EvidenceKind.UNVERIFIED,
                                "copilot.inference",
                                "Whether the user genuinely has this skill is "
                                "unverified and needs the user's confirmation.",
                            ),
                        ],
                        verification=VerificationLevel.UNVERIFIED,
                        requires_confirmation=True,
                        priority=index,
                        issue=f"The job requires '{label}', which is not "
                        "explicitly present in the resume.",
                        recommendation=f"If you genuinely have '{label}' "
                        "experience, add it to your skills or a relevant "
                        "bullet; otherwise leave it out.",
                        impact="high",
                    )
                )

        for check in self._resume_heuristic_priorities(request.resume):
            if index >= copilot_settings.max_suggestions:
                break
            index += 1
            suggestions.append(check)

        suggestions = [
            suggestion
            if suggestion.priority is not None
            else suggestion.model_copy(update={"priority": position})
            for position, suggestion in enumerate(suggestions, start=1)
        ]

        if not suggestions:
            suggestions.append(
                self._suggestion(
                    request.operation,
                    1,
                    SuggestionCategory.PRIORITY,
                    original="",
                    suggested="",
                    rationale=(
                        "No deterministic priorities were found in the "
                        "supplied context. Consider adding quantified outcomes "
                        "you can actually demonstrate."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.SUGGESTION,
                            "copilot.suggestion",
                            "Generic strengthening advice, no new facts added.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        return suggestions, self._base_explanation(request)

    def _resume_heuristic_priorities(self, resume: Resume) -> list[CopilotSuggestion]:
        checks: list[CopilotSuggestion] = []
        lines = collect_lines(resume)
        word_count = resume.metadata.word_count

        has_no_summary = not (resume.summary or "").strip()
        has_no_skills = not (resume.skills.technical or resume.skills.all)
        if has_no_summary and has_no_skills:
            checks.append(
                self._suggestion(
                    CopilotOperation.IDENTIFY_PRIORITIES,
                    len(checks) + 1,
                    SuggestionCategory.STRUCTURE,
                    rationale=(
                        "The resume lacks a summary and a technical skills "
                        "list, two sections recruiters and ATS look for first."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume",
                            "No summary and no technical skills were found.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        elif not (resume.summary or "").strip():
            checks.append(
                self._suggestion(
                    CopilotOperation.IDENTIFY_PRIORITIES,
                    len(checks) + 1,
                    SuggestionCategory.STRUCTURE,
                    rationale=(
                        "Add a 2–3 sentence professional summary; it helps a "
                        "reader quickly establish fit."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume.summary",
                            "No summary section is present.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        if lines and not any(has_quantitative(line.text) for line in lines):
            checks.append(
                self._suggestion(
                    CopilotOperation.IDENTIFY_PRIORITIES,
                    len(checks) + 1,
                    SuggestionCategory.QUANTIFICATION,
                    rationale=(
                        "None of the achievement lines include a measurable "
                        "value. Add real numbers (percentages, counts, or "
                        "volumes) where you actually have them."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume.achievements",
                            "No quantitative tokens found in achievement lines.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        if lines and not any(is_action_start(line.text) for line in lines):
            checks.append(
                self._suggestion(
                    CopilotOperation.IDENTIFY_PRIORITIES,
                    len(checks) + 1,
                    SuggestionCategory.ACTION_WORDING,
                    rationale=(
                        "Achievement lines should open with action verbs for "
                        "stronger, scannable wording."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume.achievements",
                            "No action-verb openings detected.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        if 0 < word_count < 80:
            checks.append(
                self._suggestion(
                    CopilotOperation.IDENTIFY_PRIORITIES,
                    len(checks) + 1,
                    SuggestionCategory.EVIDENCE,
                    rationale=(
                        f"The resume is sparse (about {word_count} words). Add "
                        "concrete achievements and context so ATS and "
                        "recruiters can find the substance."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume.metadata",
                            f"Approximate word count is {word_count}.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        return checks[: copilot_settings.max_suggestions]

    def _missing_skill_labels(
        self, request: CopilotRequest, analysis: CopilotAnalysisContext | None
    ) -> list[str]:
        labels: list[str] = []
        if analysis and analysis.job_specific_ats is not None:
            labels.extend(
                term.term for term in missing_terms(analysis.job_specific_ats)
            )
        elif analysis and analysis.deterministic_match is not None:
            labels.extend(analysis.deterministic_match.skill_match.missing_required)
        elif request.job_description is not None:
            labels.extend(request.job_description.required_skills)
        seen: set[str] = set()
        result: list[str] = []
        for label in labels:
            key = label.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(label.strip())
        return result

    # ------------------------------------------------------------------
    # explain_finding
    # ------------------------------------------------------------------

    def _explain_finding(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        ref = (request.finding_ref or "").strip()
        analysis = request.analysis
        finding = None
        source = ""
        if analysis:
            finding = finding_by_rule(analysis.ats_readiness, ref)
            source = "ats.findings"
            if finding is None:
                finding = job_specific_finding_by_rule(analysis.job_specific_ats, ref)
                source = "job_specific_ats.findings"

        if finding is None:
            suggestions = [
                self._suggestion(
                    request.operation,
                    1,
                    SuggestionCategory.EXPLANATION,
                    rationale=(
                        f"No finding with rule id '{ref}' was found in the "
                        "supplied analysis context. Pass the Phase 6A/6B "
                        "analysis that produced this finding so it can be "
                        "explained."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.SUGGESTION,
                            "copilot.inference",
                            "The referenced finding id is not present in the "
                            "provided analyses.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            ]
            return suggestions, self._base_explanation(request)

        suggestions = [
            self._suggestion(
                request.operation,
                1,
                SuggestionCategory.EXPLANATION,
                original="",
                suggested=finding.recommendation,
                rationale=(f"{finding.title}. {finding.explanation}"),
                evidence=[
                    self._evidence(
                        EvidenceKind.FACT,
                        source,
                        finding.explanation,
                        finding.rule_id,
                        finding.evidence[:120],
                    )
                ],
                verification=VerificationLevel.INFERRED,
            )
        ]
        return suggestions, self._base_explanation(request)

    # ------------------------------------------------------------------
    # job_alignment
    # ------------------------------------------------------------------

    def _job_alignment(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        job = request.job_description
        assert job is not None
        analysis = request.analysis
        suggestions: list[CopilotSuggestion] = []
        index = 0

        matched = self._matched_skill_labels(request, analysis)
        for skill in matched[:_MAX_MATCHED_FOR_ALIGNMENT]:
            if index >= copilot_settings.max_suggestions:
                break
            index += 1
            suggestions.append(
                self._suggestion(
                    request.operation,
                    index,
                    SuggestionCategory.ALIGNMENT,
                    original="",
                    suggested="",
                    rationale=(
                        f"'{skill}' is both on your resume and explicitly "
                        "sought by this role. Consider mentioning it earlier "
                        "or more prominently (summary, skill list, or a recent "
                        "bullet) so the reader sees the alignment immediately."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.FACT,
                            "resume.skills",
                            f"'{skill}' is listed in the resume.",
                            "resume.skills",
                        ),
                        self._evidence(
                            EvidenceKind.JOB_REQUIREMENT,
                            "job.required_skills",
                            f"'{skill}' is a requirement of the job.",
                            "job.required_skills",
                        ),
                    ],
                    verification=VerificationLevel.VERIFIED,
                )
            )

        missing = self._missing_skill_labels(request, analysis)
        for label in missing:
            if index >= copilot_settings.max_suggestions:
                break
            index += 1
            suggestions.append(
                self._suggestion(
                    request.operation,
                    index,
                    SuggestionCategory.ALIGNMENT,
                    original="",
                    suggested="",
                    rationale=(
                        f"The job lists '{label}' but it was not found "
                        "explicitly in your resume. If you have genuine "
                        "experience with it, make sure it appears; do not add "
                        "it solely to match the job description."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.JOB_REQUIREMENT,
                            "job.required_skills",
                            f"The job explicitly requires '{label}'.",
                        ),
                        self._evidence(
                            EvidenceKind.UNVERIFIED,
                            "copilot.inference",
                            "Presence of this skill in the candidate's history "
                            "is unverified.",
                        ),
                    ],
                    verification=VerificationLevel.UNVERIFIED,
                )
            )

        if not suggestions:
            suggestions.append(
                self._suggestion(
                    request.operation,
                    1,
                    SuggestionCategory.ALIGNMENT,
                    original="",
                    suggested="",
                    rationale=(
                        "No explicit skill alignment signal was detected from "
                        "the structured data. Review your experience bullets "
                        "against the job's responsibilities and surface the "
                        "most relevant existing work without adding anything "
                        "you did not do."
                    ),
                    evidence=[
                        self._evidence(
                            EvidenceKind.JOB_REQUIREMENT,
                            "job",
                            "Job requirements available; no deterministic "
                            "skill overlap detected.",
                        )
                    ],
                    verification=VerificationLevel.ADVISORY,
                )
            )
        return suggestions, self._base_explanation(request)

    def _matched_skill_labels(
        self, request: CopilotRequest, analysis: CopilotAnalysisContext | None
    ) -> list[str]:
        labels: list[str] = []
        job = request.job_description
        if job is not None:
            job_skills = _job_skill_bag(job)
            for skill in resume_skill_names(request.resume):
                if skill.lower() in job_skills:
                    labels.append(skill)
        if analysis and analysis.deterministic_match is not None:
            labels.extend(analysis.deterministic_match.skill_match.matched_required)
            labels.extend(analysis.deterministic_match.skill_match.matched_preferred)
        elif analysis and analysis.job_specific_ats is not None:
            labels.extend(
                term.term
                for term in matched_terms(analysis.job_specific_ats)
                if term.origin.value == "required"
            )
        seen: set[str] = set()
        result: list[str] = []
        for label in labels:
            key = label.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(label.strip())
        return result

    # ------------------------------------------------------------------
    # free-form
    # ------------------------------------------------------------------

    def _free_form(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        text = (request.user_request or "").strip().lower()
        job = request.job_description
        analysis = request.analysis

        summary_intent = _mentions_any(text, "summary", "profile", "objective")
        improvement_verb = _mentions_any(
            text,
            "improve",
            "rewrite",
            "polish",
            "writ",
            "refine",
            "shorten",
            "short",
            "long",
            "concise",
            "professional",
            "add",
            "create",
            "missing",
        )
        bullet_intent = bool((request.target_text or "").strip()) and (
            summary_intent
            or _mentions_any(
                text,
                "bullet",
                "wording",
                "clarity",
                "grammar",
                "rephrase",
                "stronger",
                "action",
            )
        )
        job_intent = _mentions_any(
            text, "align", "match", "job", "keyword", "requirement", "recruiter"
        )
        priority_intent = _mentions_any(
            text,
            "priority",
            "most important",
            "fix first",
            "what should i",
            "review",
            "overall",
            "audit",
            "improve my resume",
        )
        description_intent = (
            not job_intent
            and not priority_intent
            and description_target_for_request(request) is not None
            and (improvement_verb or _mentions_any(text, "stronger", "strengthen"))
        )

        if summary_intent and improvement_verb:
            return self._improve_summary(request)
        if description_intent:
            return self._improve_description(request)
        if bullet_intent:
            return self._improve_bullet(request)
        if job is not None and job_intent:
            return self._job_alignment(request)
        if analysis is not None and priority_intent:
            return self._identify_priorities(request)
        return self._advisory_free_form(request)

    def _advisory_free_form(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        suggestions = [
            self._suggestion(
                request.operation,
                1,
                SuggestionCategory.CLARITY,
                original="",
                suggested="",
                rationale=(
                    "To help most effectively, say exactly what you would "
                    "like: rewrite a specific bullet or the summary, point "
                    "out the most important weaknesses to fix first, or "
                    "suggest how to align the resume with the job "
                    "description. The Copilot edits wording and gives advice "
                    "rather than inventing new facts."
                ),
                evidence=[
                    self._evidence(
                        EvidenceKind.SUGGESTION,
                        "copilot.suggestion",
                        "Generic guidance; no facts were added or changed.",
                    )
                ],
                verification=VerificationLevel.ADVISORY,
            )
        ]
        return suggestions, self._base_explanation(request)

    # ------------------------------------------------------------------
    # improve_description (long-form project/experience descriptions)
    # ------------------------------------------------------------------

    def _improve_description(
        self, request: CopilotRequest
    ) -> tuple[list[CopilotSuggestion], str]:
        """Strengthen an editable description using only resume facts.

        Surface-fixes the wording (first-person, capitalisation) and, when the
        section's own resume-stated stack is not already named, surfaces it.
        Nothing is invented: appended items come from the resume itself. If
        nothing fact-safe can be added, the request stays advisory.
        """
        target = description_target_for_request(request)
        if target is None:
            return self._advisory_free_form(request)
        suggestion = deterministic_description_suggestion(request, target)
        if suggestion is None:
            return self._advisory_free_form(request)
        return [suggestion], self._base_explanation(request)

    # ------------------------------------------------------------------
    # Suggestion builder
    # ------------------------------------------------------------------

    def _suggestion(
        self,
        operation: CopilotOperation,
        index: int,
        category: SuggestionCategory,
        rationale: str,
        evidence: list[CopilotEvidence],
        verification: VerificationLevel,
        original: str = "",
        suggested: str = "",
        requires_confirmation: bool = False,
        priority: int | None = None,
        issue: str = "",
        recommendation: str = "",
        impact: str = "",
    ) -> CopilotSuggestion:
        return CopilotSuggestion(
            id=f"sug_{index}",
            operation=operation,
            category=category,
            original_text=original[:250],
            suggested_text=suggested[:400],
            rationale=rationale,
            evidence=evidence[: copilot_settings.max_evidence_per_suggestion],
            verification=verification,
            requires_user_confirmation=requires_confirmation,
            priority=priority,
            issue=issue,
            recommendation=recommendation,
            impact=impact,
        )


def _mentions(text: str, token: str) -> bool:
    """True when ``token`` (case-insensitive) appears in ``text``."""
    return token.lower() in text.lower()


def _mentions_any(text: str, *keys: str) -> bool:
    """True when ``text`` (lowercased) contains any of ``keys``."""
    return any(key in text for key in keys)


def _job_mentions_skill(job: JobDescription, skill: str) -> bool:
    return skill.lower() in _job_skill_bag(job)


def _job_skill_bag(job: JobDescription) -> set[str]:
    values: list[str] = []
    for bucket in (
        job.required_skills,
        job.preferred_skills,
        job.qualifications,
        job.nice_to_have,
    ):
        values.extend(bucket)
    return {v.strip().lower() for v in values if v.strip()}
