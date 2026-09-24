"""Tests for free-form edit-target inference and the resulting proposals.

Offline by construction: no network, no LLM, no persistence. Verifies that
free-form edit requests resolve to the exact allowlisted resume target, that
advice/ambiguous requests stay advisory, and that explicit "Act on (optional)"
targets always win.
"""

from __future__ import annotations

from copilot_factories import make_resume

from app.copilot.config import CopilotSettings
from app.copilot.editing import (
    edit_target_options,
    infer_edit_target,
    target_for_request,
)
from app.copilot.prompting import build_messages
from app.copilot.providers import DeterministicFallbackProvider
from app.copilot.schemas import (
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotSuggestion,
    EditCheckCategory,
    EditStatus,
    ProviderKind,
    ProviderMetadata,
    SuggestionCategory,
    VerificationLevel,
)
from app.copilot.service import CopilotService
from app.parsing.schemas import Project, WorkExperience


def _python_project_resume():
    return make_resume(
        projects=[
            Project(
                name="Python Data Pipeline",
                description=(
                    "Built a Python data pipeline that processes millions of "
                    "rows for analytics dashboards."
                ),
                technologies=["Python", "PyTorch"],
            )
        ]
    )


def _multi_python_project_resume():
    return make_resume(
        projects=[
            Project(
                name="Churn Predictor",
                description=(
                    "Predicted customer churn with gradient boosting for a "
                    "subscription analytics product."
                ),
                technologies=["Python", "XGBoost"],
            ),
            Project(
                name="Python Data Pipeline",
                description=(
                    "Built a Python data pipeline that processes millions of "
                    "rows for analytics dashboards."
                ),
                technologies=["Python", "PyTorch"],
            ),
            Project(
                name="Tweet Analyzer",
                description=(
                    "Classified tweet sentiment with rule heuristics and a "
                    "small logistic regression model."
                ),
                technologies=["Python", "scikit-learn"],
            ),
        ]
    )


def _all_python_projects_resume():
    """Three Python-tagged projects; no project name contains Python."""
    return make_resume(
        projects=[
            Project(
                name="Churn Predictor",
                description=(
                    "Predicted customer churn with gradient boosting for a "
                    "subscription analytics product."
                ),
                technologies=["Python", "XGBoost"],
            ),
            Project(
                name="Recommendation Engine",
                description=(
                    "Built a product recommendation engine that ranks items "
                    "from click-stream logs."
                ),
                technologies=["Python", "scikit-learn"],
            ),
            Project(
                name="ETL Pipeline",
                description=(
                    "Developed an ETL pipeline that normalises and loads "
                    "billing data for reporting."
                ),
                technologies=["Python", "SQL"],
            ),
        ]
    )


_EXACT_EDIT_REQUEST = (
    "Rewrite my Python project description to be stronger for a Machine "
    "Learning Engineer role while preserving every fact and not adding "
    "technologies I don't already have."
)


# ---------------------------------------------------------------------------
# infer_edit_target: unit behaviour
# ---------------------------------------------------------------------------


def test_infer_python_project_description() -> None:
    resume = _python_project_resume()
    target = infer_edit_target(
        resume,
        "Rewrite my Python project description to be stronger for a Machine "
        "Learning Engineer role while preserving every fact and not adding "
        "technologies I don't already have.",
    )
    assert target is not None
    assert target.path == "projects[0].description"


def test_infer_resolves_python_project_by_name_when_technologies_tie() -> None:
    resume = _multi_python_project_resume()
    target = infer_edit_target(resume, _EXACT_EDIT_REQUEST)
    assert target is not None
    assert target.path == "projects[1].description"


def test_infer_summary_edit() -> None:
    resume = make_resume()
    target = infer_edit_target(resume, "Make my professional summary more concise.")
    assert target is not None
    assert target.path == "summary"


def test_infer_profile_alias_targets_summary() -> None:
    resume = make_resume()
    target = infer_edit_target(resume, "Rewrite the profile to open stronger.")
    assert target is not None
    assert target.path == "summary"


def test_infer_first_bullet() -> None:
    resume = make_resume()
    target = infer_edit_target(resume, "Rewrite my first bullet to sound stronger.")
    assert target is not None
    assert target.path == "experience[0].achievements[0]"


def test_infer_second_bullet() -> None:
    resume = make_resume()
    target = infer_edit_target(resume, "Polish my 2nd bullet.")
    assert target is not None
    assert target.path == "experience[0].achievements[1]"


def test_infer_bullet_under_named_company() -> None:
    resume = make_resume(
        experience=[
            WorkExperience(
                company="Analytical Engines",
                title="Engineer",
                achievements=["Built a billing service.", "Reduced deploy time."],
            ),
            WorkExperience(
                company="Other Corp",
                title="Analyst",
                achievements=["Sliced spreadsheets.", "Wrote reports."],
            ),
        ]
    )
    target = infer_edit_target(
        resume, "Improve my first bullet under my Analytical Engines role."
    )
    assert target is not None
    assert target.path == "experience[0].achievements[0]"


def test_infer_experience_section_single_entry() -> None:
    resume = make_resume()
    target = infer_edit_target(resume, "Make my experience section more concise.")
    assert target is not None
    assert target.path == "experience[0].description"


def test_infer_experience_role_by_company_match() -> None:
    resume = make_resume(
        experience=[
            WorkExperience(
                company="Analytical Engines",
                title="Engineer",
                description="Built data processing systems.",
            ),
            WorkExperience(
                company="Other Corp",
                title="Analyst",
                description="Prepared analytical reports.",
            ),
        ]
    )
    target = infer_edit_target(
        resume, "Rewrite the description under my Analytical Engines role."
    )
    assert target is not None
    assert target.path == "experience[0].description"


def test_infer_returns_none_for_pure_advice_request() -> None:
    resume = make_resume()
    assert (
        infer_edit_target(resume, "What are the biggest weaknesses in my resume?")
        is None
    )


def test_infer_returns_none_for_skills_request() -> None:
    resume = make_resume()
    assert infer_edit_target(resume, "How do I make my skills section better?") is None


def test_infer_returns_none_for_ambiguous_edit_request() -> None:
    resume = make_resume()
    assert infer_edit_target(resume, "Make it better") is None


def test_infer_returns_none_for_ambiguous_multi_target_request() -> None:
    resume = _python_project_resume()
    assert (
        infer_edit_target(resume, "Rewrite my summary to mention my Python project.")
        is None
    )


def test_infer_returns_none_without_edit_intent() -> None:
    resume = make_resume()
    assert infer_edit_target(resume, "A summary explains who you are.") is None


def test_infer_returns_none_for_out_of_range_bullet() -> None:
    resume = make_resume()
    assert infer_edit_target(resume, "Rewrite my 9th bullet.") is None


def test_infer_returns_none_when_bullet_entry_ambiguous() -> None:
    resume = make_resume(
        experience=[
            WorkExperience(company="A", title="X", achievements=["One."]),
            WorkExperience(company="B", title="Y", achievements=["Two."]),
        ]
    )
    assert infer_edit_target(resume, "Rewrite my first bullet.") is None


# ---------------------------------------------------------------------------
# target_for_request: explicit override wins
# ---------------------------------------------------------------------------


def test_target_for_request_free_form_prefers_explicit_ref() -> None:
    resume = _python_project_resume()
    request = CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=resume,
        target_ref="experience[0].achievements[0]",
        user_request="Rewrite my Python project description.",
    )
    target = target_for_request(request)
    assert target is not None
    assert target.path == "experience[0].achievements[0]"


def test_target_for_request_free_form_uses_inference_when_no_ref() -> None:
    resume = _python_project_resume()
    request = CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=resume,
        user_request="Rewrite my Python project description to be stronger.",
    )
    target = target_for_request(request)
    assert target is not None
    assert target.path == "projects[0].description"


def test_target_for_request_improve_bullet_custom_text_stays_advisory() -> None:
    resume = make_resume()
    request = CopilotRequest(
        operation=CopilotOperation.IMPROVE_BULLET,
        resume=resume,
        target_text="I built stuff.",
    )
    assert target_for_request(request) is None


# ---------------------------------------------------------------------------
# Service integration: inference feeds the provider and the edit layer
# ---------------------------------------------------------------------------


class _RecordingProvider:
    """Fake generative provider that records the request it receives."""

    def __init__(self, response: CopilotResponse) -> None:
        self._response = response
        self.requests: list[CopilotRequest] = []

    def provider_kind(self) -> ProviderKind:
        return ProviderKind.LOCAL_OLLAMA

    def available(self) -> bool:
        return True

    def supports(self, request: CopilotRequest) -> bool:
        return True

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider=ProviderKind.LOCAL_OLLAMA,
            provider_label="fake",
            available=True,
            fallback_used=False,
            version="test",
        )

    def run(self, request: CopilotRequest) -> CopilotResponse:
        self.requests.append(request)
        return self._response


def _response(suggested: str, *, original: str = "") -> CopilotResponse:
    return CopilotResponse(
        operation=CopilotOperation.FREE_FORM,
        explanation="Fake provider rewrite.",
        suggestions=[
            CopilotSuggestion(
                id="sug_1",
                operation=CopilotOperation.FREE_FORM,
                category=SuggestionCategory.CLARITY,
                original_text=original,
                suggested_text=suggested,
                rationale="Synthetic rewrite.",
                evidence=[],
                verification=VerificationLevel.VERIFIED,
                requires_user_confirmation=False,
            )
        ],
        provider=ProviderMetadata(
            provider=ProviderKind.LOCAL_OLLAMA,
            provider_label="fake",
            available=True,
            fallback_used=False,
            version="test",
        ),
        disclaimer="test",
    )


def test_service_injects_inferred_target_and_prompt_text() -> None:
    resume = _python_project_resume()
    expected = resume.projects[0].description
    provider = _RecordingProvider(_response("Rewritten pipeline blurb."))
    service = CopilotService(providers=[provider])

    service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request="Rewrite my Python project description to be stronger.",
        )
    )

    sent = provider.requests[0]
    assert sent.target_ref == "projects[0].description"
    assert sent.target_text == expected
    user_content = build_messages(sent, CopilotSettings())[1]["content"]
    assert "TARGET TEXT" in user_content
    assert expected in user_content


def test_service_attaches_edit_proposal_for_inferred_edit_request() -> None:
    resume = _python_project_resume()
    provider = _RecordingProvider(_response("Rewritten pipeline blurb."))
    service = CopilotService(providers=[provider])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request="Rewrite my Python project description to be stronger.",
        )
    )

    llm_suggestion = response.suggestions[0]
    assert llm_suggestion.suggested_text == "Rewritten pipeline blurb."
    assert llm_suggestion.edit is None
    edits = [s.edit for s in response.suggestions if s.edit is not None]
    assert len(edits) == 1
    edit = edits[0]
    assert edit.target.path == "projects[0].description"
    assert edit.original_value == resume.projects[0].description
    assert edit.status in (EditStatus.PROPOSED, EditStatus.UNVERIFIED)


def test_llm_hallucinated_description_gets_deterministic_fact_safe_edit() -> None:
    """Regression: the exact local-Ollama/fallback scenario from the browser.

    The real qwen2.5-coder:7b model rewrote this description with invented
    facts ("over a billion rows", "real-time"). That text must stay a plain
    advice card — never an editable proposal — and the deterministic
    fact-preserving description rewrite must supply the EditProposal instead.
    """
    resume = _multi_python_project_resume()
    hallucinated = (
        "Developed a high-performance Python data pipeline that processes "
        "over a billion rows to support real-time analytics dashboards."
    )
    llm_response = CopilotResponse(
        operation=CopilotOperation.FREE_FORM,
        explanation="Fake qwen-style response.",
        suggestions=[
            CopilotSuggestion(
                id="sug_1",
                operation=CopilotOperation.FREE_FORM,
                category=SuggestionCategory.CLARITY,
                original_text=resume.experience[0].achievements[0],
                suggested_text=(
                    "XGBoost model boosted retention forecasts using "
                    "experience[0].achievements[0] material."
                ),
                rationale="Verified squash of the achievement bullet.",
                evidence=[],
                verification=VerificationLevel.VERIFIED,
                requires_user_confirmation=False,
            ),
            CopilotSuggestion(
                id="sug_3",
                operation=CopilotOperation.FREE_FORM,
                category=SuggestionCategory.ALIGNMENT,
                original_text=resume.projects[1].description,
                suggested_text=hallucinated,
                rationale="Aligns the description with the role requirements.",
                evidence=[],
                verification=VerificationLevel.VERIFIED,
                requires_user_confirmation=False,
            ),
        ],
        provider=ProviderMetadata(
            provider=ProviderKind.LOCAL_OLLAMA,
            provider_label="fake",
            available=True,
            fallback_used=False,
            version="test",
        ),
        disclaimer="test",
    )
    service = CopilotService(providers=[_RecordingProvider(llm_response)])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request=_EXACT_EDIT_REQUEST,
        )
    )

    # The hallucinated card is advice only; its text is never editable.
    hallucinated_card = next(
        s for s in response.suggestions if s.suggested_text == hallucinated
    )
    assert hallucinated_card.edit is None
    # Exactly one edit exists: the deterministic fact-preserving rewrite.
    edits = [s.edit for s in response.suggestions if s.edit is not None]
    assert len(edits) == 1
    edit = edits[0]
    assert edit.target.path == "projects[1].description"
    assert edit.original_value == resume.projects[1].description
    expected = (
        "Built a Python data pipeline that processes millions of rows for "
        "analytics dashboards. Key technologies: PyTorch."
    )
    assert edit.proposed_value == expected
    for invented in ("billion", "real-time", "high-performance"):
        assert invented not in edit.proposed_value.lower()


def test_llm_fact_preserving_description_rewrite_stays_editable() -> None:
    resume = _python_project_resume()
    safe = (
        "Built a Python data pipeline that processes millions of rows for "
        "analytics dashboards with Python and PyTorch."
    )
    service = CopilotService(
        providers=[
            _RecordingProvider(_response(safe, original=resume.projects[0].description))
        ]
    )

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request=_EXACT_EDIT_REQUEST,
        )
    )

    assert len(response.suggestions) == 1
    edit = response.suggestions[0].edit
    assert edit is not None
    assert edit.target.path == "projects[0].description"
    assert edit.proposed_value == safe
    assert edit.status == EditStatus.PROPOSED
    assert edit.validation.requires_user_confirmation is False


def test_deterministic_service_attaches_edit_for_inferred_description() -> None:
    resume = _python_project_resume()
    service = CopilotService(providers=[DeterministicFallbackProvider()])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request=(
                "Rewrite my Python project description to be stronger for a "
                "Machine Learning Engineer role while preserving every fact "
                "and not adding technologies I don't already have."
            ),
        )
    )

    edit = response.suggestions[0].edit
    assert edit is not None
    assert edit.target.path == "projects[0].description"
    assert edit.original_value == resume.projects[0].description
    assert edit.proposed_value == (
        "Built a Python data pipeline that processes millions of rows for "
        "analytics dashboards. Key technologies: PyTorch."
    )
    assert edit.status == EditStatus.UNVERIFIED
    assert edit.validation.requires_user_confirmation is True


def test_service_keeps_hallucinated_metric_unverified() -> None:
    resume = make_resume()
    provider = _RecordingProvider(
        _response(
            "Software engineer who reduced deploy time by 45%.",
            original=resume.summary or "",
        )
    )
    service = CopilotService(providers=[provider])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request="Make my summary more concise.",
        )
    )

    edit = response.suggestions[0].edit
    assert edit is not None
    assert edit.target.path == "summary"
    assert edit.status == EditStatus.UNVERIFIED
    assert edit.validation.requires_user_confirmation is True
    metric = next(
        check
        for check in edit.validation.checks
        if check.category == EditCheckCategory.METRIC
    )
    assert metric.passed is False


def test_service_advice_request_produces_no_edit() -> None:
    resume = make_resume()
    advisory = CopilotResponse(
        operation=CopilotOperation.FREE_FORM,
        explanation="Fake provider advice.",
        suggestions=[
            CopilotSuggestion(
                id="sug_1",
                operation=CopilotOperation.FREE_FORM,
                category=SuggestionCategory.CLARITY,
                original_text="",
                suggested_text="",
                rationale="Focus on quantified outcomes you can prove.",
                evidence=[],
                verification=VerificationLevel.ADVISORY,
            )
        ],
        provider=ProviderMetadata(
            provider=ProviderKind.LOCAL_OLLAMA,
            provider_label="fake",
            available=True,
            fallback_used=False,
            version="test",
        ),
        disclaimer="test",
    )
    provider = _RecordingProvider(advisory)
    service = CopilotService(providers=[provider])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request="What are the biggest weaknesses in my resume?",
        )
    )

    assert provider.requests[0].target_ref is None
    assert all(s.edit is None for s in response.suggestions)


def test_explicit_target_override_beats_inference() -> None:
    resume = _python_project_resume()
    provider = _RecordingProvider(_response("Rewritten first bullet.", original=""))
    service = CopilotService(providers=[provider])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            target_ref="experience[0].achievements[0]",
            user_request="Rewrite my Python project description.",
        )
    )

    sent = provider.requests[0]
    assert sent.target_ref == "experience[0].achievements[0]"
    assert sent.target_text == resume.experience[0].achievements[0]
    edit = response.suggestions[0].edit
    assert edit is not None
    assert edit.target.path == "experience[0].achievements[0]"


# ---------------------------------------------------------------------------
# Target clarification: ambiguous edit requests never guess
# ---------------------------------------------------------------------------


def _ambiguous_request(resume):
    return CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=resume,
        user_request=_EXACT_EDIT_REQUEST,
    )


def test_ambiguous_python_project_returns_clarification() -> None:
    """Regression: 'Rewrite my Python project...' with several Python projects
    must return a structured clarification, never a blank advice card and
    never a guessed target."""
    resume = _all_python_projects_resume()
    service = CopilotService(providers=[DeterministicFallbackProvider()])

    response = service.suggest(_ambiguous_request(resume))

    assert response.clarification is not None
    assert response.clarification.options
    # Options cover every tied project, in resume order; no guess.
    assert [option.target.path for option in response.clarification.options] == [
        "projects[0].description",
        "projects[1].description",
        "projects[2].description",
    ]
    for option in response.clarification.options:
        assert option.label
    # The advisory body never carries an edit.
    assert all(s.edit is None for s in response.suggestions)


def test_clarified_target_resubmit_yields_edit_proposal() -> None:
    """Selecting an option and resubmitting the SAME request with its path as
    the explicit target_ref produces the existing fact-safe EditProposal."""
    resume = _all_python_projects_resume()
    service = CopilotService(providers=[DeterministicFallbackProvider()])

    first = service.suggest(_ambiguous_request(resume))
    assert first.clarification is not None
    chosen = first.clarification.options[1].target.path

    resubmitted = _ambiguous_request(resume).model_copy(update={"target_ref": chosen})
    response = service.suggest(resubmitted)

    assert response.clarification is None
    edits = [s.edit for s in response.suggestions if s.edit is not None]
    assert len(edits) == 1
    edit = edits[0]
    assert edit.target.path == chosen
    assert edit.original_value == resume.projects[1].description
    assert edit.proposed_value
    assert edit.status == EditStatus.UNVERIFIED


def test_ambiguous_request_with_explicit_target_skips_clarification() -> None:
    """An explicit 'Act on (optional)' target beats inference: no clarification
    is shown even when the wording alone is ambiguous."""
    resume = _all_python_projects_resume()
    service = CopilotService(providers=[DeterministicFallbackProvider()])

    response = service.suggest(
        _ambiguous_request(resume).model_copy(
            update={"target_ref": "projects[2].description"}
        )
    )

    assert response.clarification is None
    edits = [s.edit for s in response.suggestions if s.edit is not None]
    assert len(edits) == 1
    assert edits[0].target.path == "projects[2].description"


def test_edit_target_options_empty_for_other_scopes() -> None:
    """No clarification for non-edit advice, non-project sections, or a
    uniquely resolvable project."""
    resume = _all_python_projects_resume()
    assert (
        edit_target_options(
            CopilotRequest(
                operation=CopilotOperation.FREE_FORM,
                resume=resume,
                user_request="What are the biggest weaknesses in my resume?",
            )
        )
        == []
    )
    assert (
        edit_target_options(
            CopilotRequest(
                operation=CopilotOperation.FREE_FORM,
                resume=resume,
                user_request="Rewrite my summary to open stronger.",
            )
        )
        == []
    )
    assert (
        edit_target_options(
            CopilotRequest(
                operation=CopilotOperation.FREE_FORM,
                resume=resume,
                user_request=(
                    "Rewrite my Python project description and my first bullet."
                ),
            )
        )
        == []
    )
    assert (
        edit_target_options(
            CopilotRequest(
                operation=CopilotOperation.FREE_FORM,
                resume=resume,
                target_ref="projects[0].description",
                user_request=_EXACT_EDIT_REQUEST,
            )
        )
        == []
    )
    named = make_resume(
        projects=[
            Project(
                name="Python Data Pipeline",
                description="Built a data pipeline for analytics.",
                technologies=["Python"],
            ),
            Project(
                name="Tweet Analyzer",
                description="Classified tweet sentiment.",
                technologies=["Python"],
            ),
        ]
    )
    assert (
        edit_target_options(
            CopilotRequest(
                operation=CopilotOperation.FREE_FORM,
                resume=named,
                user_request=_EXACT_EDIT_REQUEST,
            )
        )
        == []
    )


def test_normal_advice_free_form_stays_advisory() -> None:
    """Pure advice requests never produce a clarification and stay advice-only."""
    resume = make_resume()
    service = CopilotService(providers=[DeterministicFallbackProvider()])

    response = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=resume,
            user_request="What are the biggest weaknesses in my resume?",
        )
    )

    assert response.clarification is None
    assert response.suggestions
    assert all(s.edit is None for s in response.suggestions)
