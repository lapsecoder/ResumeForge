import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "./api";
import type { ATSReadinessResult } from "./ats";
import {
  COPILOT_SUGGEST_TIMEOUT_MS,
  getCopilotStatus,
  suggestCopilot,
  type CopilotResponse,
  type CopilotStatus,
  type CopilotSuggestRequest,
} from "./copilot";
import type { Resume } from "./resume";

const SAMPLE_RESUME: Resume = {
  contact: { name: "Test Candidate", email: "test@example.com" },
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

const SAMPLE_RESPONSE: CopilotResponse = {
  operation: "improve_bullet",
  explanation: "Reworded the bullet while preserving every fact.",
  suggestions: [
    {
      id: "sug_1",
      operation: "improve_bullet",
      category: "bullet",
      original_text: "I built stuff.",
      suggested_text: "Built stuff.",
      rationale: "Removed a leading first-person subject.",
      evidence: [
        {
          kind: "fact",
          source: "resume.result.resume_text",
          statement: "The original bullet is preserved under rewriting.",
          reference: "copilot.rewrite.safe",
        },
      ],
      verification: "verified",
      requires_user_confirmation: false,
      priority: 2,
      issue: "Passive phrasing",
      recommendation: "Start with a strong action verb.",
      impact: "medium",
    },
    {
      id: "sug_2",
      operation: "identify_priorities",
      category: "skills",
      suggested_text: "Add quantified impact.",
      rationale: "Impact quantifies the result.",
      evidence: [],
      verification: "unverified",
      requires_user_confirmation: true,
      priority: 1,
      issue: "No metrics",
      recommendation: "Add a measured outcome.",
      impact: "high",
    },
  ],
  provider: {
    provider: "deterministic",
    provider_label: "Deterministic fallback",
    available: true,
    fallback_used: true,
    version: "7b-copilot-1.0",
  },
  disclaimer: "The Copilot provides resume-writing assistance. It does not predict hiring outcomes.",
};

const SAMPLE_STATUS: CopilotStatus = {
  providers: [
    {
      provider: "local-ollama",
      provider_label: "Local Ollama (qwen2.5-coder:7b)",
      model: "qwen2.5-coder:7b",
      available: false,
      note: "Ollama unavailable; deterministic fallback used.",
    },
    { provider: "deterministic", provider_label: "Deterministic fallback", available: true },
  ],
  fallback_available: true,
  version: "7b-copilot-1.0",
};

const TEST_ATS_READINESS = {
  overall_score: 74,
  score_label: "Good",
  category_scores: [],
  findings: [],
  metadata: { method: "test", version: "1", weights: {}, applied_weights: {}, score_labels: {}, disclaimer: "" },
} satisfies ATSReadinessResult;

function bulletRequest(): CopilotSuggestRequest {
  return { operation: "improve_bullet", resume: SAMPLE_RESUME, target_text: "I built stuff." };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("suggestCopilot", () => {
  it("posts a structured operation to /api/v1/copilot/suggest", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESPONSE));
    vi.stubGlobal("fetch", fetchMock);

    const result = await suggestCopilot(bulletRequest());

    expect(result.suggestions[0].verification).toBe("verified");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/copilot/suggest");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toMatchObject({ operation: "improve_bullet" });
  });

  it("passes through the 7b priority/impact fields", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESPONSE)));

    const result = await suggestCopilot(bulletRequest());

    expect(result.suggestions[0]).toMatchObject({
      priority: 2,
      issue: "Passive phrasing",
      recommendation: "Start with a strong action verb.",
      impact: "medium",
    });
    expect(result.suggestions[1]).toMatchObject({
      priority: 1,
      verification: "unverified",
      requires_user_confirmation: true,
    });
  });

  it("posts the analysis grounding context with the request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESPONSE));
    vi.stubGlobal("fetch", fetchMock);

    await suggestCopilot({
      ...bulletRequest(),
      analysis: { ats_readiness: TEST_ATS_READINESS },
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(init.body));
    expect(body.analysis).toBeDefined();
    expect(body.analysis.ats_readiness.overall_score).toBe(74);
  });

  it("surfaces the 7b implementation version on status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_STATUS)));

    const status = await getCopilotStatus();

    expect(status.version).toBe("7b-copilot-1.0");
  });

  it("rejects a degraded (503) provider availability like the backend", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse(503, { error: { code: "provider_unavailable", message: "unavailable" } })
      )
    );
    const error = await suggestCopilot(bulletRequest()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("unexpected_server_error");
  });

  it("handles a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await suggestCopilot(bulletRequest()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("network_unreachable");
  });

  it("rejects a malformed payload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, { hello: "world" })));
    const error = await suggestCopilot(bulletRequest()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("invalid_response");
  });

  it("posts free-form requests with a bounded user_request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESPONSE));
    vi.stubGlobal("fetch", fetchMock);

    await suggestCopilot({
      operation: "free-form",
      resume: SAMPLE_RESUME,
      user_request: "Make my summary more concise.",
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toMatchObject({
      operation: "free-form",
      user_request: "Make my summary more concise.",
    });
  });

  it("attaches an abort signal so requests are cancellable", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESPONSE));
    vi.stubGlobal("fetch", fetchMock);

    await suggestCopilot(bulletRequest());

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("rejects with request_timeout when generation exceeds the deadline", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockImplementation(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError"))
          );
        })
    );
    vi.stubGlobal("fetch", fetchMock);

    const promise = suggestCopilot(bulletRequest());
    const settled = promise.then(
      () => undefined,
      (error) => error
    );
    await vi.advanceTimersByTimeAsync(COPILOT_SUGGEST_TIMEOUT_MS + 1);
    const error = await settled;

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("request_timeout");
    vi.useRealTimers();
  });
});

describe("getCopilotStatus", () => {
  it("returns provider availability from /api/v1/copilot/status", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_STATUS));
    vi.stubGlobal("fetch", fetchMock);

    const status = await getCopilotStatus();

    expect(status.fallback_available).toBe(true);
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toContain("/api/v1/copilot/status");
  });

  it("attaches an abort signal for status probes", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_STATUS));
    vi.stubGlobal("fetch", fetchMock);

    await getCopilotStatus();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("rejects a malformed status payload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, { nope: true })));
    const error = await getCopilotStatus().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("invalid_response");
  });
});