import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "./api";
import { matchHybrid, parseJobDescription, type HybridMatchResult, type JobDescription } from "./matcher";
import type { Resume } from "./resume";

const SAMPLE_RESUME: Resume = {
  contact: { name: "Test Candidate", email: "test@example.com" },
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

function jobFile(): File {
  return new File(["synthetic job description"], "job.txt", { type: "text/plain" });
}

function mockResponse(status: number, body: unknown, ok?: boolean): Response {
  return {
    ok: ok ?? (status >= 200 && status < 300),
    status,
    json: async () => body,
  } as Response;
}

const SAMPLE_JOB: JobDescription = {
  title: "Backend Engineer",
  required_skills: ["Python"],
  metadata: { word_count: 0, file_type: "txt", overall_confidence: "high" },
};

const SAMPLE_MATCH: HybridMatchResult = {
  overall_score: 72,
  deterministic: {
    overall_score: 70,
    skill_match: { score: 70 },
    experience_match: { score: 60 },
    education_match: { score: 50 },
    matched_requirements: ["Required skill matched: Python"],
    missing_required: ["Kubernetes"],
    strengths: ["Python"],
    gaps: ["Kubernetes"],
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
  strengths: ["Python"],
  gaps: ["Kubernetes"],
  semantic_insights: [
    {
      category: "skill",
      evidence_level: "high",
      similarity: 0.8,
      statement: "The resume is semantically related to Python but it is not explicitly verified.",
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
      "The semantic signal is produced entirely locally by an open-source sentence-embedding model (Apache-2.0). No external, hosted, or paid AI model is consulted, and no content leaves the machine.",
  },
};

function matchWith(partial: Partial<HybridMatchResult>): HybridMatchResult {
  return {
    ...SAMPLE_MATCH,
    ...partial,
    deterministic: { ...SAMPLE_MATCH.deterministic, ...(partial.deterministic ?? {}) },
    semantic: partial.semantic === null ? null : { ...SAMPLE_MATCH.semantic!, ...(partial.semantic ?? {}) },
    component_scores: {
      ...SAMPLE_MATCH.component_scores,
      ...(partial.component_scores ?? {}),
    },
    metadata: { ...SAMPLE_MATCH.metadata, ...(partial.metadata ?? {}) },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("parseJobDescription", () => {
  it("uploads via POST /api/v1/jobs/parse and returns the job", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_JOB));
    vi.stubGlobal("fetch", fetchMock);

    const result = await parseJobDescription(jobFile());

    expect(result.title).toBe("Backend Engineer");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/jobs/parse");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
  });

  it("handles a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await parseJobDescription(jobFile()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("network_unreachable");
  });

  it("rejects a malformed payload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, { hello: "world" })));
    const error = await parseJobDescription(jobFile()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("invalid_response");
  });
});

describe("matchHybrid", () => {
  it("posts resume and job to /api/v1/matching/hybrid", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_MATCH));
    vi.stubGlobal("fetch", fetchMock);

    const result = await matchHybrid(SAMPLE_RESUME, SAMPLE_JOB as never);

    expect(result.overall_score).toBe(72);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/matching/hybrid");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toMatchObject({ resume: SAMPLE_RESUME });
  });

  it("treats semantic=null (model unavailable) as a valid degraded result", async () => {
    const degraded = matchWith({
      semantic: null,
      metadata: {
        ...SAMPLE_MATCH.metadata,
        mode: "deterministic-only",
        semantic_availability: {
          available: false,
          status: "model_unavailable",
          note: "The local semantic model is unavailable.",
        },
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, degraded)));
    expect((await matchHybrid(SAMPLE_RESUME, SAMPLE_JOB as never)).semantic).toBeNull();
  });

  it("handles a non-ok response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(mockResponse(500, { error: { code: "boom" } }))
    );
    const error = await matchHybrid(SAMPLE_RESUME, SAMPLE_JOB as never).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("unexpected_server_error");
  });

  it("never logs match content on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_MATCH)));
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const warnLog = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await matchHybrid(SAMPLE_RESUME, SAMPLE_JOB as never);

    expect(errorLog).not.toHaveBeenCalled();
    expect(warnLog).not.toHaveBeenCalled();
  });
});