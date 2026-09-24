import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Resume } from "@/lib/resume";

import { MatchFlow } from "./MatchFlow";

const { parseJobMock, matchHybridMock } = vi.hoisted(() => ({
  parseJobMock: vi.fn(),
  matchHybridMock: vi.fn(),
}));

vi.mock("@/lib/matcher", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/matcher")>();
  return { ...actual, parseJobDescription: parseJobMock, matchHybrid: matchHybridMock };
});

const RESUME: Resume = {
  contact: { name: "Ada Lovelace" },
  summary: "First programmer.",
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

const JOB_RESULT = {
  overall_score: 80,
  deterministic: {
    overall_score: 80,
    skill_match: { score: 80 },
    experience_match: { score: null },
    education_match: { score: null },
    matched_requirements: ["Required skill matched: Python"],
    missing_required: [],
    strengths: [],
    gaps: [],
    metadata: { method: "deterministic-baseline", label: "Baseline Match Score", version: "1.0" },
  },
  semantic: {
    overall_similarity: 0.9,
    note: "Semantic similarity is a measure of relatedness, not a hiring probability.",
    metadata: {
      method: "local-semantic-embedding",
      model_name: "sentence-transformers/all-MiniLM-L6-v2",
      model_version: "sentence-transformers/all-MiniLM-L6-v2",
      implementation_version: "1.0",
      model_source: "local Hugging Face cache",
      model_license: "Apache-2.0",
      device: "cpu",
    },
  },
  component_scores: {
    deterministic_overall: 80,
    semantic_overall: 0.9,
    hybrid_overall: 80,
    deterministic_skills: 80,
    semantic_skills: 0.9,
    deterministic_experience: null,
    semantic_experience: null,
  },
  matched_requirements: ["Required skill matched: Python"],
  missing_required: [],
  strengths: [],
  gaps: [],
  semantic_insights: [],
  metadata: {
    method: "hybrid-match",
    label: "Hybrid Match Score",
    version: "5c-hybrid-1.0",
    weights: { deterministic: 0.7, semantic: 0.3 },
    mode: "hybrid",
    semantic_availability: { available: true, status: "available", note: "" },
    note: "The Hybrid Match Score is a heuristic relevance score. It is not a hiring probability.",
    model_disclosure:
      "The semantic signal is produced entirely locally by an open-source sentence-embedding model.",
  },
};

function jobFile(): File {
  return new File(["job description content"], "backend-job.txt", { type: "text/plain" });
}

describe("MatchFlow", () => {
  beforeEach(() => {
    parseJobMock.mockReset();
    matchHybridMock.mockReset();
  });

  it("renders the matching section heading", () => {
    render(<MatchFlow resume={RESUME} />);
    expect(screen.getByText(/Match against a job description/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Choose a job description file/ })).toBeInTheDocument();
  });

  it("resolves a full match flow and shows the result", async () => {
    parseJobMock.mockResolvedValue({
      title: "Backend Engineer",
      required_skills: ["Python"],
      metadata: { word_count: 0, file_type: "txt", overall_confidence: "high" },
    });
    matchHybridMock.mockResolvedValue(JOB_RESULT);
    const user = userEvent.setup();
    render(<MatchFlow resume={RESUME} />);

    await user.upload(
      screen.getByLabelText(/Choose a job description file/),
      jobFile()
    );
    await user.click(screen.getByRole("button", { name: /Run match/ }));

    await waitFor(() => {
      expect(screen.getByText(/Hybrid Match Score/)).toBeInTheDocument();
    });
    expect(parseJobMock).toHaveBeenCalledTimes(1);
    expect(matchHybridMock).toHaveBeenCalledTimes(1);
    expect(screen.getAllByText(/80 \/ 100/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Run match/)).toBeNull();
  });

  it("shows a friendly message when parsing fails", async () => {
    parseJobMock.mockRejectedValue(new Error("network"));
    const user = userEvent.setup();
    render(<MatchFlow resume={RESUME} />);

    await user.upload(screen.getByLabelText(/Choose a job description file/), jobFile());
    await user.click(screen.getByRole("button", { name: /Run match/ }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/reach the resumeforge backend/i);
    });
    expect(matchHybridMock).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /Run match/ })).toBeInTheDocument();
  });

  it("allows replacing the selected job file", async () => {
    const user = userEvent.setup();
    render(<MatchFlow resume={RESUME} />);

    await user.upload(screen.getByLabelText(/Choose a job description file/), jobFile());
    expect(screen.getByText("backend-job.txt")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Replace/ }));
    expect(screen.queryByText("backend-job.txt")).toBeNull();
    expect(screen.getByRole("button", { name: /Choose a job description file/ })).toBeInTheDocument();
  });

  it("calls onJobFileChange with the file name and clears on reset", async () => {
    const onJobFileChange = vi.fn();
    const user = userEvent.setup();
    render(<MatchFlow resume={RESUME} onJobFileChange={onJobFileChange} />);

    await user.upload(screen.getByLabelText(/Choose a job description file/), jobFile());
    expect(onJobFileChange).toHaveBeenCalledWith("backend-job.txt");

    await user.click(screen.getByRole("button", { name: /Replace/ }));
    expect(onJobFileChange).toHaveBeenCalledWith(null);
  });

  it("does not log job content on a successful run", async () => {
    parseJobMock.mockResolvedValue({
      title: "Backend Engineer",
      required_skills: ["Python"],
      metadata: { word_count: 0, file_type: "txt", overall_confidence: "high" },
    });
    matchHybridMock.mockResolvedValue(JOB_RESULT);
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const user = userEvent.setup();
    render(<MatchFlow resume={RESUME} />);

    await user.upload(screen.getByLabelText(/Choose a job description file/), jobFile());
    await user.click(screen.getByRole("button", { name: /Run match/ }));
    await waitFor(() => {
      expect(screen.getByText(/Hybrid Match Score/)).toBeInTheDocument();
    });

    const logText = errorLog.mock.calls.concat(logSpy.mock.calls).join("|");
    expect(logText).not.toContain("job description content");
  });
});