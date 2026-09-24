import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiErrorMessage, parseResume } from "./api";
import type { Resume } from "./resume";

const SAMPLE_RESUME: Resume = {
  contact: { name: "Test Candidate", email: "test@example.com" },
  summary: "Synthetic summary",
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

function mockResponse(status: number, body: unknown, ok?: boolean): Response {
  return {
    ok: ok ?? (status >= 200 && status < 300),
    status,
    json: async () => body,
  } as Response;
}

function resumeFile(): File {
  return new File(["synthetic resume content"], "resume.pdf", { type: "application/pdf" });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("parseResume", () => {
  it("uploads via POST /api/v1/resumes/parse and returns the resume", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESUME));
    vi.stubGlobal("fetch", fetchMock);

    const result = await parseResume(resumeFile());

    expect(result.contact?.name).toBe("Test Candidate");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/resumes/parse");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
  });

  it("maps a server error code to a human-readable message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse(422, {
          error: { code: "malformed_file", message: "nope", request_id: "abc" },
        })
      )
    );

    await expect(parseResume(resumeFile())).rejects.toThrow(
      "The file appears to be malformed or corrupted."
    );
  });

  it("maps a file_too_large 413", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse(413, { error: { code: "file_too_large", message: "too big" } })
      )
    );

    const error = await parseResume(resumeFile()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("file_too_large");
    expect((error as ApiError).message).toContain("10 MB");
  });

  it("handles a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const error = await parseResume(resumeFile()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("network_unreachable");
  });

  it("rejects a 500 with a generic message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(500, {})));

    await expect(parseResume(resumeFile())).rejects.toThrow(
      "Something went wrong while analyzing your resume."
    );
  });

  it("rejects a malformed success payload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, { hello: "world" })));

    const error = await parseResume(resumeFile()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("invalid_response");
  });

  it("does not log resume content on success", async () => {
    const content = "SECRET-SURNAME-THREE";
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse(200, SAMPLE_RESUME)));
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const warnLog = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await parseResume(new File([content], "resume.txt"));

    expect(errorLog).not.toHaveBeenCalled();
    expect(warnLog).not.toHaveBeenCalled();
  });
});

describe("apiErrorMessage", () => {
  it("prefers ApiError messages", () => {
    expect(apiErrorMessage(new ApiError("network_unreachable", "Network unreachable"))).toBe(
      "Network unreachable"
    );
  });

  it("maps unknown fetch errors to a network message", () => {
    expect(apiErrorMessage(new TypeError("Failed to fetch"))).toContain("backend");
  });

  it("has a generic fallback", () => {
    expect(apiErrorMessage("garbage")).toBe(
      "Something went wrong while analyzing your resume. Please try again."
    );
  });
});