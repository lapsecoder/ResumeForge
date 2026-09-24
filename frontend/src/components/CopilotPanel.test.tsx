import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { CopilotSuggestion, CopilotStatus } from "@/lib/copilot";
import type { JobDescription, HybridMatchResult } from "@/lib/matcher";
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
  return { ...actual, suggestCopilot: suggestCopilotMock, getCopilotStatus: getCopilotStatusMock };
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
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

const JOB: JobDescription = {
  title: "Backend Engineer",
  required_skills: ["Python", "Docker"],
  metadata: { word_count: 0, file_type: "txt", overall_confidence: "high" },
};

const MATCH_RESULT = {
  overall_score: 62,
  deterministic: { overall_score: 62 },
  metadata: { mode: "hybrid" },
} as unknown as HybridMatchResult;

const READINESS = {
  overall_score: 74,
  score_label: "Good",
  category_scores: [],
  findings: [
    {
      category: "quantification",
      severity: "high",
      rule_id: "ats_quant_01",
      title: "Achievements lack numbers",
      explanation: "Few achievements include measurable impact.",
      evidence: "No quantified achievement found.",
      recommendation: "Add metrics such as percentages or user counts.",
      impact: 6,
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
  findings: [],
  metadata: { method: "test", version: "1", weights: {}, applied_weights: {}, score_labels: {}, disclaimer: "" },
};

const STATUS: CopilotStatus = {
  providers: [
    {
      provider: "local-ollama",
      provider_label: "Local Ollama",
      model: "qwen2.5-coder:7b",
      available: false,
      note: "Ollama unavailable; deterministic fallback used.",
    },
    { provider: "deterministic", provider_label: "Deterministic fallback", available: true },
  ],
  fallback_available: true,
  version: "7b-copilot-1.0",
};

function suggestion(overrides: Partial<CopilotSuggestion> = {}): CopilotSuggestion {
  return {
    id: "sug_1",
    operation: "identify_priorities",
    category: "quantification",
    original_text: "I built a billing service used by thousands of users.",
    suggested_text: "Built a billing service used by 1,200+ users.",
    rationale: "A specific number makes the impact measurable.",
    evidence: [
      {
        kind: "inference",
        source: "ats_analysis.findings[0]",
        statement: "The ATS finding recommends quantifying achievements.",
        reference: "ats_quant_01",
      },
    ],
    verification: "unverified",
    requires_user_confirmation: true,
    priority: 1,
    issue: "Achievement is not quantified",
    recommendation: "Replace vague scale with a concrete figure.",
    impact: "high",
    ...overrides,
  };
}

const SUGGEST_RESPONSE = {
  operation: "identify_priorities",
  explanation: "Prioritised the highest-impact fixes for your resume.",
  suggestions: [suggestion()],
  provider: {
    provider: "deterministic",
    provider_label: "Deterministic fallback",
    available: true,
    fallback_used: true,
    version: "7b-copilot-1.0",
  },
  disclaimer:
    "The Copilot provides resume-writing assistance. It does not predict hiring outcomes.",
};

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

describe("CopilotPanel", () => {
  it("renders the operation selector and loads ATS grounding", async () => {
    render(<CopilotPanel resume={RESUME} />);

    expect(screen.getByText("AI Copilot")).toBeInTheDocument();
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));
    expect(getCopilotStatusMock).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: /Generate suggestions/ })
    ).toBeInTheDocument();
  });

  it("runs identify_priorities and renders the 7b priority fields", async () => {
    const user = userEvent.setup();
    render(<CopilotPanel resume={RESUME} />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));

    await waitFor(() => {
      expect(screen.getByText(/Prioritised the highest-impact fixes/)).toBeInTheDocument();
    });
    expect(suggestCopilotMock).toHaveBeenCalledTimes(1);
    const request = suggestCopilotMock.mock.calls[0][0];
    expect(request.operation).toBe("identify_priorities");
    expect(request.analysis.ats_readiness.overall_score).toBe(74);

    expect(screen.getByText("Priority 1")).toBeInTheDocument();
    expect(screen.getByText(/Achievement is not quantified/)).toBeInTheDocument();
    expect(screen.getByText(/Replace vague scale with a concrete figure/)).toBeInTheDocument();
    expect(screen.getByText(/expected impact: high/)).toBeInTheDocument();
    expect(screen.getByText("Review needed")).toBeInTheDocument();
    expect(screen.getByText(/Deterministic fallback · fallback · 7b-copilot-1.0/)).toBeInTheDocument();
  });

  it("never offers a resume-changing Apply on advisory suggestions (no edit proposal)", async () => {
    const user = userEvent.setup();
    const onResumeChange = vi.fn();
    render(<CopilotPanel resume={RESUME} onResumeChange={onResumeChange} />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));
    await screen.findByText(/Prioritised the highest-impact fixes/);

    // Regression: this suggestion has no edit proposal, so it must not render
    // an Apply button that only toggles card-local state without touching the
    // canonical workingResume.
    expect(screen.queryByRole("button", { name: /^Apply$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Undo$/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Advice only/)).toBeInTheDocument();
    expect(onResumeChange).not.toHaveBeenCalled();
  });

  it("posts a selected bullet ref and text for improve_bullet", async () => {
    const user = userEvent.setup();
    render(<CopilotPanel resume={RESUME} />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));

    await user.selectOptions(
      screen.getByLabelText(/Operation/),
      "improve_bullet"
    );
    await user.selectOptions(
      await screen.findByLabelText(/Target bullet/),
      "experience[0].achievements[0]"
    );
    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));

    await waitFor(() => expect(suggestCopilotMock).toHaveBeenCalledTimes(1));
    const request = suggestCopilotMock.mock.calls[0][0];
    expect(request.target_ref).toBe("experience[0].achievements[0]");
    expect(request.target_text).toBe("I built a billing service used by thousands of users.");
  });

  it("gates job_alignment behind a matched job", async () => {
    const user = userEvent.setup();
    render(<CopilotPanel resume={RESUME} />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));

    await user.selectOptions(
      screen.getByLabelText(/Operation/),
      "job_alignment"
    );

    expect(
      screen.getByRole("button", { name: /Generate suggestions/ })
    ).toBeDisabled();
    expect(
      screen.getByText(/Job alignment requires a job description/)
    ).toBeInTheDocument();
  });

  it("runs job_alignment with a matched job and loads job-specific ATS", async () => {
    const user = userEvent.setup();
    render(<CopilotPanel resume={RESUME} jobMatch={{ job: JOB, result: MATCH_RESULT }} />);

    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(analyzeJobAtsMock).toHaveBeenCalledTimes(1));

    await user.selectOptions(
      screen.getByLabelText(/Operation/),
      "job_alignment"
    );
    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));

    await waitFor(() => expect(suggestCopilotMock).toHaveBeenCalledTimes(1));
    const request = suggestCopilotMock.mock.calls[0][0];
    expect(request.job_description).toBe(JOB);
    expect(request.analysis.job_specific_ats.overall_score).toBe(55);
    expect(request.analysis.deterministic_match.overall_score).toBe(62);
  });

  it("offers ATS findings for explain_finding", async () => {
    const user = userEvent.setup();
    render(<CopilotPanel resume={RESUME} />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));

    await user.selectOptions(
      screen.getByLabelText(/Operation/),
      "explain_finding"
    );

    const findingSelect = await screen.findByLabelText(/Finding to explain/);
    expect(findingSelect).toBeInTheDocument();
    await user.selectOptions(findingSelect, "ats_quant_01");
    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));

    await waitFor(() => expect(suggestCopilotMock).toHaveBeenCalledTimes(1));
    expect(suggestCopilotMock.mock.calls[0][0].finding_ref).toBe("ats_quant_01");
  });

  it("shows a friendly error when the backend cannot be reached", async () => {
    suggestCopilotMock.mockRejectedValue(
      new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.")
    );
    const user = userEvent.setup();
    render(<CopilotPanel resume={RESUME} />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));

    expect(
      await screen.findByText(/We couldn't reach the ResumeForge backend/)
    ).toBeInTheDocument();
  });
});