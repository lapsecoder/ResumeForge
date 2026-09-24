import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { HybridMatchResult } from "@/lib/matcher";

import { MatchResultView } from "./MatchResultView";

const RESULT: HybridMatchResult = {
  overall_score: 72,
  deterministic: {
    overall_score: 70,
    skill_match: { score: 70 },
    experience_match: { score: 60 },
    education_match: { score: 50 },
    matched_requirements: ["Required skill matched: Python"],
    missing_required: ["Kubernetes"],
    strengths: [],
    gaps: [],
    metadata: { method: "deterministic-baseline", label: "Baseline Match Score", version: "1.0" },
  },
  semantic: {
    overall_similarity: 0.8,
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
    deterministic_overall: 70,
    semantic_overall: 0.8,
    hybrid_overall: 72,
    deterministic_skills: 70,
    semantic_skills: 0.8,
    deterministic_experience: 60,
    semantic_experience: 0.7,
  },
  matched_requirements: ["Required skill matched: Python"],
  missing_required: ["Kubernetes"],
  strengths: [],
  gaps: [],
  semantic_insights: [
    {
      category: "skill",
      evidence_level: "high",
      similarity: 0.8,
      statement:
        "The resume is semantically related to Python but the match is not explicitly verified.",
    },
  ],
  metadata: {
    method: "hybrid-match",
    label: "Hybrid Match Score",
    version: "5c-hybrid-1.0",
    weights: { deterministic: 0.7, semantic: 0.3 },
    mode: "hybrid",
    semantic_availability: { available: true, status: "available", note: "" },
    note: "The Hybrid Match Score is a heuristic relevance score. It is not a hiring probability.",
    model_disclosure:
      "The semantic signal is produced entirely locally by an open-source sentence-embedding model (Apache-2.0).",
  },
};

function degradedResult(): HybridMatchResult {
  return {
    ...RESULT,
    overall_score: 70,
    semantic: null,
    semantic_insights: [],
    component_scores: { ...RESULT.component_scores, semantic_overall: null },
    metadata: {
      ...RESULT.metadata,
      mode: "deterministic-only",
      semantic_availability: {
        available: false,
        status: "model_unavailable",
        note: "The local semantic model is unavailable; the hybrid result uses the deterministic signal only.",
      },
    },
  };
}

describe("MatchResultView", () => {
  it("renders the hybrid score and both component signals", () => {
    render(<MatchResultView result={RESULT} />);

    expect(screen.getByText(/Hybrid Match Score/)).toBeInTheDocument();
    expect(screen.getByText(/72 \/ 100/)).toBeInTheDocument();
    expect(screen.getByText(/Deterministic match \(evidence-based\)/)).toBeInTheDocument();
    expect(screen.getByText(/70 \/ 100/)).toBeInTheDocument();
    expect(screen.getByText(/Semantic relatedness \(local ML model\)/)).toBeInTheDocument();
    expect(screen.getByText(/0.80 \/ 1.00/)).toBeInTheDocument();
  });

  it("shows the not-a-probability guardrail", () => {
    render(<MatchResultView result={RESULT} />);
    expect(
      screen.getByText(/not a hiring probability and does not prove/)
    ).toBeInTheDocument();
  });

  it("shows explicit matches and unverified missing skills", () => {
    render(<MatchResultView result={RESULT} />);

    expect(screen.getByText(/Explicitly matched/)).toBeInTheDocument();
    expect(screen.getByText(/Required skill matched: Python/)).toBeInTheDocument();
    expect(screen.getByText(/Not explicitly verified/)).toBeInTheDocument();
    expect(screen.getByText(/Kubernetes/)).toBeInTheDocument();
    expect(
      screen.getByText(/semantic relatedness does not convert them into verified matches/)
    ).toBeInTheDocument();
  });

  it("renders the model disclosure and metadata", () => {
    render(<MatchResultView result={RESULT} />);

    expect(
      screen.getByText(/produced entirely locally by an open-source sentence-embedding model/)
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/sentence-transformers\/all-MiniLM-L6-v2/).length
    ).toBeGreaterThan(0);
    expect(screen.getAllByText(/Apache-2.0/).length).toBeGreaterThan(0);
  });

  it("renders semantic insights with relatedness wording", () => {
    render(<MatchResultView result={RESULT} />);

    expect(
      screen.getByText(/semantically related to Python but the match is not explicitly verified/)
    ).toBeInTheDocument();
  });

  it("never uses positive hiring-probability phrasing", () => {
    const { container } = render(<MatchResultView result={RESULT} />);
    expect(container.textContent).not.toMatch(/probability of hire/i);
    expect(container.textContent).not.toMatch(/chance of (getting )?the job/i);
    expect(container.textContent).not.toMatch(/you(?:'re| are) likely to (?:get|be)/i);
  });

  it("surfaces the unavailable semantic model instead of hiding it", () => {
    render(<MatchResultView result={degradedResult()} />);

    expect(
      screen.getByText(/semantic model is unavailable; the hybrid result uses the deterministic/)
    ).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.queryByText(/0.80 \/ 1.00/)).toBeNull();
  });

  it("does not render insights when there are none", () => {
    render(<MatchResultView result={degradedResult()} />);
    expect(screen.queryByText(/Relatedness insights/)).toBeNull();
  });
});