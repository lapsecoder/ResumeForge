import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { CopilotEditProposal, CopilotStatus } from "@/lib/copilot";
import type { Resume } from "@/lib/resume";

import { CopilotPanel } from "./CopilotPanel";

const { suggestCopilotMock, getCopilotStatusMock, analyzeAtsMock, analyzeJobAtsMock } =
  vi.hoisted(() => ({
    suggestCopilotMock: vi.fn(),
    getCopilotStatusMock: vi.fn(),
    analyzeAtsMock: vi.fn(),
    analyzeJobAtsMock: vi.fn(),
  }));

vi.mock("@/lib/copilot", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/copilot")>();
  return {
    ...actual,
    suggestCopilot: suggestCopilotMock,
    getCopilotStatus: getCopilotStatusMock,
  };
});

vi.mock("@/lib/ats", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/ats")>();
  return {
    ...actual,
    analyzeAtsReadiness: analyzeAtsMock,
    analyzeJobSpecificAts: analyzeJobAtsMock,
  };
});

const RESUME: Resume = {
  contact: { name: "Ada Lovelace" },
  summary: "Analyst with a background in mathematics.",
  experience: [
    {
      company: "Analytical Engines",
      title: "Analyst",
      achievements: ["I built a billing service used by thousands of users."],
    },
  ],
  projects: [
    {
      name: "Churn Predictor",
      description: "Predicted customer churn with gradient boosting.",
      technologies: ["Python", "XGBoost"],
    },
    {
      name: "ETL Pipeline",
      description: "Developed an ETL pipeline for billing data.",
      technologies: ["Python", "SQL"],
    },
  ],
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

const STATUS: CopilotStatus = {
  providers: [
    { provider: "deterministic", provider_label: "Deterministic fallback", available: true },
  ],
  fallback_available: true,
  version: "7d-copilot-1.0",
};

// Both analyses report the SAME rule id, reproducing the duplicate-key case.
const SHARED_RULE_ID = "ats_education_field";

const READINESS = {
  overall_score: 74,
  score_label: "Good",
  category_scores: [],
  findings: [
    {
      category: "structure",
      severity: "medium",
      rule_id: SHARED_RULE_ID,
      title: "Education section could be clearer",
      explanation: "The education section lacks consistent detail.",
      evidence: "No degree field on some entries.",
      recommendation: "Standardise the education section.",
      impact: 1,
    },
  ],
  metadata: { method: "test", version: "1", weights: {}, applied_weights: {}, score_labels: {}, disclaimer: "" },
};

const JOB_ATS = {
  overall_score: 55,
  score_label: "Weak",
  category_scores: [],
  coverage: {
    required_coverage: 0.5,
    preferred_coverage: null,
    overall_coverage: 0.5,
    evidence_supported_required: null,
    evidence_supported_preferred: null,
    totals: { required: 2, required_matched: 1, preferred: 0, preferred_matched: 0, phrases: 0, phrases_matched: 0 },
  },
  term_matches: [],
  findings: [
    {
      category: "job_specific",
      severity: "medium",
      rule_id: SHARED_RULE_ID,
      title: "Education requirement emphasis",
      explanation: "The role stresses a degree requirement.",
      evidence: "Job lists a degree requirement.",
      recommendation: "Surface your education more clearly.",
      impact: 1,
    },
  ],
  metadata: { method: "test", version: "1", weights: {}, applied_weights: {}, score_labels: {}, disclaimer: "" },
};

const SUGGEST_RESPONSE = {
  operation: "free-form",
  explanation: "Tightened the summary wording.",
  suggestions: [
    {
      id: "sug_1",
      operation: "free-form",
      category: "clarity",
      original_text: "Analyst with a background in mathematics.",
      suggested_text: "Mathematician with a background in quantitative analysis.",
      rationale: "Leads with the clearer professional identity you described.",
      evidence: [],
      verification: "verified",
      requires_user_confirmation: false,
    },
  ],
  provider: {
    provider: "deterministic",
    provider_label: "Deterministic fallback",
    available: true,
    fallback_used: true,
    version: "7d-copilot-1.0",
  },
  disclaimer: "Assistance only.",
};

function editProposal(): CopilotEditProposal {
  return {
    edit_id: "edit_sug_1",
    operation: "free-form",
    target: {
      path: "summary",
      section: "summary",
      index: null,
      field: null,
      sub_index: null,
    },
    original_value: "Analyst with a background in mathematics.",
    proposed_value: "Mathematician with a background in quantitative analysis.",
    reason: "Matches your request for a crisper opener.",
    evidence: [],
    validation: {
      status: "verified",
      requires_user_confirmation: false,
      checks: [
        { category: "metric", passed: true, detail: "" },
        { category: "date", passed: true, detail: "" },
        { category: "skill", passed: true, detail: "" },
        { category: "source_term", passed: true, detail: "" },
      ],
    },
    status: "proposed",
  };
}

function responseWithEdit(edit: CopilotEditProposal) {
  return {
    ...SUGGEST_RESPONSE,
    suggestions: [
      {
        ...SUGGEST_RESPONSE.suggestions[0],
        edit,
      },
    ],
  };
}

beforeEach(() => {
  suggestCopilotMock.mockReset();
  getCopilotStatusMock.mockReset();
  analyzeAtsMock.mockReset();
  analyzeJobAtsMock.mockReset();
  getCopilotStatusMock.mockResolvedValue(STATUS);
  analyzeAtsMock.mockResolvedValue(READINESS);
  analyzeJobAtsMock.mockResolvedValue(JOB_ATS);
  suggestCopilotMock.mockResolvedValue(SUGGEST_RESPONSE);
});

const JOB = {
  title: "Senior Analyst",
  company: "Analytical Engines",
  summary: "Senior Analyst with a focus on quantitative analysis.",
  responsibilities: ["Build analysis tooling"],
  required_skills: ["Python"],
  preferred_skills: [],
  qualifications: [],
  experience_requirements: [],
  education_requirements: [],
  certifications: [],
  nice_to_have: [],
  benefits: [],
  custom_sections: [],
  salary: [],
  metadata: {
    word_count: 12,
    file_type: "paste",
    overall_confidence: "high" as const,
  },
};

const MATCH_RESULT = {
  overall_score: 0.6,
  deterministic: {
    overall_score: 0.6,
    skill_match: { score: 0.6 },
    experience_match: { score: null },
    education_match: { score: null },
    matched_requirements: [],
    missing_required: [],
    strengths: [],
    gaps: [],
    metadata: { method: "test", label: "Possible match", version: "1" },
  },
  semantic: {
    overall_similarity: 0.6,
    note: "",
    metadata: {
      method: "test",
      model_name: "test",
      model_version: "1",
      implementation_version: "1",
      model_source: "test",
      model_license: "test",
      device: "cpu",
    },
  },
  component_scores: {
    deterministic_overall: 0.6,
    semantic_overall: 0.6,
    hybrid_overall: 0.6,
    deterministic_skills: 0.6,
    semantic_skills: 0.6,
    deterministic_experience: null,
    semantic_experience: null,
  },
  matched_requirements: [],
  missing_required: [],
  strengths: [],
  gaps: [],
  semantic_insights: [],
  metadata: {
    method: "test",
    label: "Possible match",
    version: "1",
    weights: { deterministic: 0.5, semantic: 0.5 },
    mode: "hybrid" as const,
    semantic_availability: { available: true, status: "ready", note: "" },
    note: "",
    model_disclosure: "",
  },
};

async function setupPanel(overrides: Partial<Parameters<typeof CopilotPanel>[0]> = {}) {
  const user = userEvent.setup();
  render(<CopilotPanel resume={RESUME} {...overrides} />);
  await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));
  return user;
}

async function setupPanelWithJob() {
  const user = userEvent.setup();
  render(<CopilotPanel resume={RESUME} jobMatch={{ job: JOB, result: MATCH_RESULT }} />);
  await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(analyzeJobAtsMock).toHaveBeenCalledTimes(1));
  return user;
}

describe("CopilotPanel free-text requests", () => {
  it("renders the free-text input and Ask Copilot button", async () => {
    await setupPanel();

    expect(screen.getByLabelText(/Free-text request for Copilot/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ask Copilot/ })).toBeInTheDocument();
    expect(screen.getByLabelText(/Act on/)).toBeInTheDocument();
  });

  it("blocks an empty request with a validation message", async () => {
    const user = await setupPanel();

    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(await screen.findByText(/Enter a request first/)).toBeInTheDocument();
    expect(suggestCopilotMock).not.toHaveBeenCalled();
  });

  it("blocks an oversized request with a length message", async () => {
    const user = await setupPanel();
    const textarea = screen.getByLabelText(/Free-text request for Copilot/);
    fireEvent.change(textarea, { target: { value: "x".repeat(2001) } });

    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(await screen.findByText(/Keep your request under 2000 characters/)).toBeInTheDocument();
    expect(suggestCopilotMock).not.toHaveBeenCalled();
  });

  it("posts a bounded free-form request with the analysis context", async () => {
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "Make my summary more concise"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    await waitFor(() => expect(suggestCopilotMock).toHaveBeenCalledTimes(1));
    const request = suggestCopilotMock.mock.calls[0][0];
    expect(request.operation).toBe("free-form");
    expect(request.user_request).toBe("Make my summary more concise");
    expect(request.analysis.ats_readiness.overall_score).toBe(74);
    expect(request.analysis.job_specific_ats).toBeNull();
    expect(request.analysis.deterministic_match).toBeNull();
  });

  it("posts a selected target so the rewrite can be applied", async () => {
    const user = await setupPanel();

    await user.selectOptions(screen.getByLabelText(/Act on/), "summary");
    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "Rewrite the summary"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    await waitFor(() => expect(suggestCopilotMock).toHaveBeenCalledTimes(1));
    const request = suggestCopilotMock.mock.calls[0][0];
    expect(request.target_ref).toBe("summary");
    expect(request.target_text).toBe("Analyst with a background in mathematics.");
  });

  it("enters a generating state that disables both submit actions", async () => {
    let resolveSuggest = () => {};
    suggestCopilotMock.mockReturnValue(
      new Promise((resolve) => {
        resolveSuggest = () => resolve(SUGGEST_RESPONSE);
      })
    );
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "Improve the opener"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    const generating = screen.getAllByRole("button", { name: /Generating…/ });
    expect(generating).toHaveLength(2);
    expect(generating[0]).toBeDisabled();
    expect(generating[1]).toBeDisabled();
    expect(
      screen.queryByRole("button", { name: /Ask Copilot/ })
    ).not.toBeInTheDocument();

    await act(async () => {
      resolveSuggest();
    });
    await waitFor(() =>
      expect(screen.getByText(/Tightened the summary wording/)).toBeInTheDocument()
    );
  });

  it("shows a friendly message when a free-form request times out", async () => {
    suggestCopilotMock.mockRejectedValue(
      new ApiError(
        "request_timeout",
        "The request took too long and was cancelled. Please try again."
      )
    );
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "Rewrite my summary"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(
      await screen.findByText(/took too long and was cancelled/)
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Tightened the summary wording/)
    ).not.toBeInTheDocument();
  });

  it("renders duplicate ATS rule ids from both analyses without key collisions", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const user = await setupPanelWithJob();

    await user.selectOptions(screen.getByLabelText(/Operation/), "explain_finding");

    const findingSelect = await screen.findByLabelText(/Finding to explain/);
    const options = Array.from(findingSelect.querySelectorAll("option")).map(
      (option) => option.textContent
    );
    expect(options.filter((text) => text?.includes("Education"))).toHaveLength(2);

    const keyWarnings = consoleError.mock.calls.filter((args) =>
      String(args[0]).includes("Encountered two children with the same key")
    );
    expect(keyWarnings).toHaveLength(0);
    consoleError.mockRestore();
  });

  it("applies a free-form edit proposal to the working resume", async () => {
    suggestCopilotMock.mockResolvedValue(responseWithEdit(editProposal()));
    const user = await setupPanel();

    await user.selectOptions(screen.getByLabelText(/Act on/), "summary");
    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "Rewrite the summary to be crisper"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(
      await screen.findByRole("button", { name: /Apply to resume/ })
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Apply to resume/ }));

    expect(await screen.findByText("Applied")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Revert$/ })).toBeInTheDocument();
  });

  it("renders an editable proposal for a rewrite without an explicit target", async () => {
    suggestCopilotMock.mockResolvedValue(responseWithEdit(editProposal()));
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "Rewrite my summary to be crisper"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(
      await screen.findByRole("button", { name: /Apply to resume/ })
    ).toBeInTheDocument();
    expect(screen.getByText(/Mathematician with a background/)).toBeInTheDocument();
  });

  it("does not offer Apply for advisory free-form suggestions", async () => {
    suggestCopilotMock.mockResolvedValue({
      ...SUGGEST_RESPONSE,
      suggestions: [
        {
          id: "sug_1",
          operation: "free-form",
          category: "clarity",
          original_text: "",
          suggested_text: "",
          rationale: "Focus on quantified outcomes you can prove.",
          evidence: [],
          verification: "advisory",
          requires_user_confirmation: false,
        },
      ],
    });
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      "What should I focus on?"
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(await screen.findByText(/Advice only/)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Apply to resume/ })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Which project's description/ })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Churn Predictor/ })
    ).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Target clarification: ambiguous free-form edit requests
// ---------------------------------------------------------------------------

const PROJECT = {
  name: "Churn Predictor",
  description: "Predicted customer churn with gradient boosting.",
  technologies: ["Python", "XGBoost"],
};

const CLARIFICATION = {
  question: "Which project's description should I rewrite?",
  reason:
    "Your request names a project by a skill or technology that appears in several projects.",
  options: [
    {
      target: {
        path: "projects[0].description",
        section: "projects",
        index: 0,
        field: "description",
        sub_index: null,
      },
      label: "Churn Predictor — Python, XGBoost · Predicted customer churn with gradient boosting.",
    },
    {
      target: {
        path: "projects[1].description",
        section: "projects",
        index: 1,
        field: "description",
        sub_index: null,
      },
      label: "ETL Pipeline — Python, SQL · Developed an ETL pipeline for billing data.",
    },
  ],
};

const AMBIGUOUS_REQUEST =
  "Rewrite my Python project description to be stronger for a Machine Learning Engineer role while preserving every fact.";

describe("CopilotPanel target clarification", () => {
  it("shows selectable project choices for an ambiguous rewrite request", async () => {
    suggestCopilotMock.mockResolvedValue({
      ...SUGGEST_RESPONSE,
      suggestions: [
        {
          id: "sug_1",
          operation: "free-form",
          category: "clarity",
          original_text: "",
          suggested_text: "",
          rationale: "Advisory body carried with the clarification.",
          evidence: [],
          verification: "advisory",
          requires_user_confirmation: false,
        },
      ],
      clarification: CLARIFICATION,
    });
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      AMBIGUOUS_REQUEST
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));

    expect(
      await screen.findByText(/Which project's description should I rewrite\?/)
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Churn Predictor — Python, XGBoost/ })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /ETL Pipeline — Python, SQL/ })
    ).toBeInTheDocument();
    // Ambiguity is never presented as a generic advice card.
    expect(screen.queryByText(/Advice only/)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Apply to resume/ })
    ).not.toBeInTheDocument();
  });

  it("resubmits the SAME request with the chosen target_ref", async () => {
    suggestCopilotMock
      .mockResolvedValueOnce({ ...SUGGEST_RESPONSE, clarification: CLARIFICATION })
      .mockResolvedValueOnce({ ...SUGGEST_RESPONSE });
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      AMBIGUOUS_REQUEST
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));
    const option = await screen.findByRole("button", {
      name: /ETL Pipeline — Python, SQL/,
    });
    await user.click(option);

    await waitFor(() => expect(suggestCopilotMock).toHaveBeenCalledTimes(2));
    const first = suggestCopilotMock.mock.calls[0][0];
    const second = suggestCopilotMock.mock.calls[1][0];
    expect(first.target_ref).toBeUndefined();
    expect(second.target_ref).toBe("projects[1].description");
    expect(second.user_request).toBe(first.user_request);
    expect(second.operation).toBe("free-form");
    // The chosen target is pinned in the "Act on" select.
    expect(screen.getByLabelText(/Act on/)).toHaveValue("projects[1].description");
  });

  it("renders the EditProposal flow after a project is selected", async () => {
    suggestCopilotMock
      .mockResolvedValueOnce({ ...SUGGEST_RESPONSE, clarification: CLARIFICATION })
      .mockResolvedValueOnce({
        ...responseWithEdit({
          ...editProposal(),
          target: CLARIFICATION.options[0].target,
          original_value: PROJECT.description,
          proposed_value:
            "Predicted customer churn with gradient boosting for a subscription analytics product. Key technologies: XGBoost.",
          status: "unverified",
          validation: {
            status: "unverified",
            requires_user_confirmation: true,
            checks: [
              { category: "metric", passed: true, detail: "" },
              { category: "date", passed: true, detail: "" },
              { category: "skill", passed: true, detail: "" },
              { category: "source_term", passed: false, detail: "" },
            ],
          },
        }),
      });
    const user = await setupPanel();

    await user.type(
      screen.getByLabelText(/Free-text request for Copilot/),
      AMBIGUOUS_REQUEST
    );
    await user.click(screen.getByRole("button", { name: /Ask Copilot/ }));
    const option = await screen.findByRole("button", { name: /Churn Predictor/ });
    await user.click(option);

    expect(
      await screen.findByRole("button", { name: /Apply anyway/ })
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Predicted customer churn with gradient boosting for a subscription analytics product\. Key technologies: XGBoost\./
      )
    ).toBeInTheDocument();
    expect(screen.queryByText(/Which project's description/)).not.toBeInTheDocument();
  });
});