import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  CopilotEditProposal,
  CopilotSuggestion,
  CopilotStatus,
} from "@/lib/copilot";
import { useHistory } from "@/lib/history";
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
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

const STATUS: CopilotStatus = {
  providers: [
    { provider: "deterministic", provider_label: "Deterministic fallback", available: true },
  ],
  fallback_available: true,
  version: "7c-copilot-1.0",
};

const READINESS = {
  overall_score: 74,
  score_label: "Good",
  category_scores: [],
  findings: [],
  metadata: { method: "test", version: "1", weights: {}, applied_weights: {}, score_labels: {}, disclaimer: "" },
};

function makeEdit(overrides: Partial<CopilotEditProposal> = {}): CopilotEditProposal {
  return {
    edit_id: "edit_sug_1",
    operation: "improve_summary",
    target: {
      path: "summary",
      section: "summary",
      index: null,
      field: null,
      sub_index: null,
    },
    original_value: "Analyst with a background in mathematics.",
    proposed_value: "Mathematician and software engineer.",
    reason: "Lead with your strongest professional identity.",
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
    ...overrides,
  };
}

function makeSuggestion(edit: CopilotEditProposal): CopilotSuggestion {
  return {
    id: "sug_1",
    operation: "improve_summary",
    category: "summary",
    original_text: edit.original_value,
    suggested_text: edit.proposed_value,
    rationale: "Clearer professional identity.",
    evidence: [],
    verification: "verified",
    requires_user_confirmation: false,
    edit,
  };
}

function mockResponse(edit: CopilotEditProposal) {
  suggestCopilotMock.mockResolvedValue({
    operation: "improve_summary",
    explanation: "Prepared a summary rewrite for your review.",
    suggestions: [makeSuggestion(edit)],
    provider: {
      provider: "deterministic",
      provider_label: "Deterministic fallback",
      available: true,
      fallback_used: true,
      version: "7c-copilot-1.0",
    },
    disclaimer: "Assistance only.",
  });
}

function Harness() {
  const [resume, setResume] = useState<Resume>(RESUME);
  return (
    <>
      <p data-testid="summary">{resume.summary}</p>
      <p data-testid="bullet">{resume.experience?.[0]?.achievements?.[0]}</p>
      <p data-testid="original">{RESUME.summary}</p>
      <CopilotPanel resume={resume} onResumeChange={setResume} />
    </>
  );
}

function HistoryHarness() {
  const [original] = useState<Resume>(RESUME);
  const {
    present: working,
    commit,
    undo,
    redo,
    canUndo,
    canRedo,
    reset,
  } = useHistory<Resume>(original);
  return (
    <>
      <p data-testid="summary">{working.summary}</p>
      <p data-testid="bullet">{working.experience?.[0]?.achievements?.[0]}</p>
      <p data-testid="original">{original.summary}</p>
      <button type="button" onClick={undo} disabled={!canUndo}>
        Undo
      </button>
      <button type="button" onClick={redo} disabled={!canRedo}>
        Redo
      </button>
      <button type="button" onClick={() => reset(original)}>
        Reset
      </button>
      <CopilotPanel resume={working} onResumeChange={commit} />
    </>
  );
}

async function generate() {
  const user = userEvent.setup();
  render(<Harness />);
  await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalled());
  await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));
  await screen.findByRole("button", { name: /Preview checks/ });
  return user;
}

function makeAdvisory(
  overrides: Partial<CopilotSuggestion> = {}
): CopilotSuggestion {
  return {
    id: "sug_adv",
    operation: "free-form",
    category: "clarity",
    original_text: "Analyst with a background in mathematics.",
    suggested_text: "Mathematician and software engineer.",
    rationale: "General guidance only — no editable target was selected.",
    evidence: [],
    verification: "verified",
    requires_user_confirmation: false,
    ...overrides,
  };
}

async function generateAdvisory() {
  suggestCopilotMock.mockResolvedValue({
    operation: "free-form",
    explanation: "Prepared general advice for your review.",
    suggestions: [makeAdvisory()],
    provider: {
      provider: "deterministic",
      provider_label: "Deterministic fallback",
      available: true,
      fallback_used: true,
      version: "7c-copilot-1.0",
    },
    disclaimer: "Assistance only.",
  });
  const user = userEvent.setup();
  render(<Harness />);
  await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalled());
  await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));
  await screen.findByText(/Prepared general advice for your review/);
  return user;
}

beforeEach(() => {
  suggestCopilotMock.mockReset();
  getCopilotStatusMock.mockReset();
  analyzeAtsMock.mockReset();
  analyzeJobAtsMock.mockReset();
  getCopilotStatusMock.mockResolvedValue(STATUS);
  analyzeAtsMock.mockResolvedValue(READINESS);
  analyzeJobAtsMock.mockResolvedValue(READINESS);
});

describe("CopilotPanel edit proposals", () => {
  it("applies an edit to the working resume and offers Revert", async () => {
    mockResponse(makeEdit());
    const user = await generate();

    expect(screen.getByTestId("summary")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );

    await user.click(screen.getByRole("button", { name: /Apply to resume/ }));

    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Mathematician and software engineer."
      )
    );
    expect(screen.getByText("Applied")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Revert$/ })).toBeInTheDocument();
  });

  it("reverts an applied edit back to the original value", async () => {
    mockResponse(makeEdit());
    const user = await generate();

    await user.click(screen.getByRole("button", { name: /Apply to resume/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent("Mathematician")
    );

    await user.click(screen.getByRole("button", { name: /^Revert$/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Analyst with a background in mathematics."
      )
    );
    expect(screen.getByRole("button", { name: /Apply to resume/ })).toBeInTheDocument();
  });

  it("dismisses a proposal without changing the resume", async () => {
    mockResponse(makeEdit());
    const user = await generate();

    await user.click(screen.getByRole("button", { name: /^Dismiss$/ }));

    await waitFor(() =>
      expect(screen.queryByText("Professional summary")).not.toBeInTheDocument()
    );
    expect(screen.getByTestId("summary")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );
  });

  it("previews the deterministic fact checks", async () => {
    mockResponse(makeEdit());
    const user = await generate();

    await user.click(screen.getByRole("button", { name: /Preview checks/ }));
    expect(screen.getByText(/Passed · metric/)).toBeInTheDocument();
    expect(screen.getByText(/Passed · source_term/)).toBeInTheDocument();
  });

  it("labels an unverified edit and warns before applying", async () => {
    mockResponse(
      makeEdit({
        status: "unverified",
        validation: {
          status: "unverified",
          requires_user_confirmation: true,
          checks: [
            { category: "metric", passed: false, detail: "Adds a number not in your resume." },
            { category: "date", passed: true, detail: "" },
            { category: "skill", passed: true, detail: "" },
            { category: "source_term", passed: true, detail: "" },
          ],
        },
      })
    );
    await generate();

    expect(screen.getByText("Needs your confirmation")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Apply anyway/ })).toBeInTheDocument();
    expect(
      screen.getByText(/introduces information the fact checks could not trace/)
    ).toBeInTheDocument();
  });

  it("shows an error instead of applying an out-of-range target", async () => {
    mockResponse(
      makeEdit({
        target: {
          path: "experience[9].title",
          section: "experience",
          index: 9,
          field: "title",
          sub_index: null,
        },
      })
    );
    const user = await generate();

    await user.click(screen.getByRole("button", { name: /Apply to resume/ }));

    expect(
      await screen.findByText(/This edit can no longer be applied/)
    ).toBeInTheDocument();
    expect(screen.getByTestId("summary")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );
  });

  it("regression: advisory-only suggestions never offer a resume-changing Apply", async () => {
    await generateAdvisory();

    expect(screen.getByText(/Advice only/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Apply$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Undo$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Apply to resume/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Apply anyway/ })).not.toBeInTheDocument();

    expect(screen.getByTestId("summary")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );
    expect(screen.getByTestId("bullet")).toHaveTextContent(
      "I built a billing service used by thousands of users."
    );
  });

  it("regression: a verified targeted edit updates the canonical workingResume and passes through history undo/redo", async () => {
    mockResponse(makeEdit());
    const user = userEvent.setup();
    render(<HistoryHarness />);
    await waitFor(() => expect(analyzeAtsMock).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: /Generate suggestions/ }));
    await screen.findByRole("button", { name: /Preview checks/ });

    expect(screen.getByTestId("original")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );
    expect(screen.getByTestId("summary")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );

    await user.click(screen.getByRole("button", { name: /Apply to resume/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Mathematician and software engineer."
      )
    );
    expect(screen.getByTestId("original")).toHaveTextContent(
      "Analyst with a background in mathematics."
    );
    expect(screen.getByRole("button", { name: /^Undo$/ })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /^Undo$/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Analyst with a background in mathematics."
      )
    );
    expect(screen.getByRole("button", { name: /^Redo$/ })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /^Redo$/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Mathematician and software engineer."
      )
    );

    await user.click(screen.getByRole("button", { name: /^Reset$/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Analyst with a background in mathematics."
      )
    );
  });

  it("regression: applying twice from one click never double-applies", async () => {
    mockResponse(makeEdit());
    const user = await generate();

    await user.click(screen.getByRole("button", { name: /Apply to resume/ }));
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toHaveTextContent(
        "Mathematician and software engineer."
      )
    );
    expect(screen.getByRole("button", { name: /^Revert$/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Apply to resume/ })).not.toBeInTheDocument();

    expect(screen.getByText("Applied")).toBeInTheDocument();
    expect(screen.getByTestId("summary")).toHaveTextContent(
      "Mathematician and software engineer."
    );
  });
});
