import { describe, expect, it } from "vitest";

import {
  MAX_COMPANY_CHARS,
  addApplication,
  changeStatus,
  createApplication,
  normalizeApplication,
  normalizeDate,
  removeApplication,
  replaceApplication,
  safeExternalUrl,
  updateApplication,
  validateApplicationDraft,
  type Application,
} from "./applications";

const FIXED_NOW = new Date("2026-09-01T12:00:00.000Z");

function sampleApp(overrides: Partial<Application> = {}): Application {
  return {
    id: "app_1",
    company: "Acme Corp",
    role: "Software Engineer",
    location: "Remote",
    url: "https://acme.example/jobs/1",
    applied_date: "2026-09-01",
    status: "applied",
    notes: null,
    resume_template: "classic",
    scores: {
      baseline_match: 70,
      hybrid_match: 75,
      ats_readiness: 80,
      job_specific_ats_coverage: 0.7,
    },
    created_at: "2026-09-01T00:00:00.000Z",
    updated_at: "2026-09-01T00:00:00.000Z",
    ...overrides,
  };
}

describe("safeExternalUrl", () => {
  it("keeps https and http URLs", () => {
    expect(safeExternalUrl("https://acme.example/jobs/1")).toBe("https://acme.example/jobs/1");
    expect(safeExternalUrl("http://acme.example/jobs/1")).toBe("http://acme.example/jobs/1");
  });

  it("upgrades a bare domain to https", () => {
    expect(safeExternalUrl("acme.example/jobs/1")).toBe("https://acme.example/jobs/1");
  });

  it("rejects unsafe schemes", () => {
    for (const url of ["javascript:alert(1)", "data:text/html,hi", "file:///etc/passwd", "vbscript:x"]) {
      expect(safeExternalUrl(url)).toBeNull();
    }
  });

  it("rejects malformed, whitespace-laden, and way-too-long values", () => {
    expect(safeExternalUrl("not a domain")).toBeNull();
    expect(safeExternalUrl("https://exc\\.com/a b")).toBeNull();
    expect(safeExternalUrl("x".repeat(2_001))).toBeNull();
    expect(safeExternalUrl(42)).toBeNull();
  });
});

describe("normalizeDate", () => {
  it("passes through a valid YYYY-MM-DD", () => {
    expect(normalizeDate("2026-09-01")).toBe("2026-09-01");
  });

  it("extracts the date part from an ISO datetime", () => {
    expect(normalizeDate("2026-09-01T00:00:00.000Z")).toBe("2026-09-01");
  });

  it("rejects impossible and malformed dates", () => {
    expect(normalizeDate("2026-02-30")).toBeNull();
    expect(normalizeDate("01-01-2026")).toBeNull();
    expect(normalizeDate("tomorrow")).toBeNull();
    expect(normalizeDate(0)).toBeNull();
  });
});

describe("createApplication", () => {
  it("creates with only required fields and safe defaults", () => {
    const app = createApplication({ company: " Acme Corp ", role: "Engineer" }, { now: FIXED_NOW });
    expect(app).not.toBeNull();
    expect(app!.company).toBe("Acme Corp");
    expect(app!.role).toBe("Engineer");
    expect(app!.status).toBe("saved");
    expect(app!.location).toBeNull();
    expect(app!.url).toBeNull();
    expect(app!.applied_date).toBeNull();
    expect(app!.scores).toEqual({
      baseline_match: null,
      hybrid_match: null,
      ats_readiness: null,
      job_specific_ats_coverage: null,
    });
    expect(app!.created_at).toBe("2026-09-01T12:00:00.000Z");
    expect(app!.updated_at).toBe("2026-09-01T12:00:00.000Z");
  });

  it("returns null when company or role is missing or overlong", () => {
    expect(createApplication({})).toBeNull();
    expect(createApplication({ company: "Acme", role: "" })).toBeNull();
    expect(createApplication({ company: "x".repeat(MAX_COMPANY_CHARS + 1), role: "Engineer" })).toBeNull();
  });

  it("strips control characters and trims", () => {
    const app = createApplication({ company: "Ac\u0000me Corp", role: "En\u0007gineer\n" });
    expect(app!.company).toBe("Acme Corp");
    expect(app!.role).toBe("Engineer");
  });

  it("sanitizes the URL and applied date", () => {
    const app = createApplication({
      company: "Acme",
      role: "Engineer",
      url: "javascript:alert(1)",
      applied_date: "2026-09-01T00:00:00.000Z",
    });
    expect(app!.url).toBeNull();
    expect(app!.applied_date).toBe("2026-09-01");
  });

  it("falls back to saved for an invalid status and uses a provided id", () => {
    const app = createApplication(
      { company: "Acme", role: "Engineer", status: "nope" as never },
      { id: "custom-id", now: FIXED_NOW }
    );
    expect(app!.status).toBe("saved");
    expect(app!.id).toBe("custom-id");
  });

  it("ignores unknown draft keys and never copies prototype keys", () => {
    const draft = {
      company: "Acme",
      role: "Engineer",
      __proto__: { injected: true },
      constructor: { prototype: {} },
    };
    const app = createApplication(draft as never);
    expect(app).not.toBeNull();
    expect(Object.prototype.hasOwnProperty.call(app, "__proto__")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(app, "constructor")).toBe(false);
  });

  it("stores hostile text inert (rendering escapes it later)", () => {
    const app = createApplication({ company: "<script>alert(1)</script>", role: "Engineer" });
    expect(app!.company).toBe("<script>alert(1)</script>");
  });

  it("clamps out-of-range scores to null", () => {
    const app = createApplication({
      company: "Acme",
      role: "Engineer",
      scores: { baseline_match: 140, hybrid_match: -5, ats_readiness: 99, job_specific_ats_coverage: 9 },
    });
    expect(app!.scores.baseline_match).toBeNull();
    expect(app!.scores.hybrid_match).toBeNull();
    expect(app!.scores.ats_readiness).toBe(99);
    expect(app!.scores.job_specific_ats_coverage).toBeNull();
  });
});

describe("normalizeApplication", () => {
  it("round-trips a trusted application object", () => {
    const app = sampleApp();
    expect(normalizeApplication(app)).toEqual(app);
  });

  it("returns null for malformed or incomplete input", () => {
    expect(normalizeApplication(null)).toBeNull();
    expect(normalizeApplication({ company: "Acme" })).toBeNull();
    expect(normalizeApplication({ company: "Acme", role: "Engineer", id: "" })).toBeNull();
    expect(normalizeApplication([])).toBeNull();
  });

  it("repairs bad timestamps, statuses, and scores", () => {
    const app = normalizeApplication({
      ...sampleApp(),
      created_at: "not-a-date",
      status: "weird",
      scores: { baseline_match: 1234 },
    });
    expect(app!.created_at).toBe(new Date(0).toISOString());
    expect(app!.status).toBe("saved");
    expect(app!.scores.baseline_match).toBeNull();
  });
});

describe("updateApplication", () => {
  it("applies a patch immutably", () => {
    const app = sampleApp();
    const updated = updateApplication(app, { location: "Berlin", notes: "Referral" }, FIXED_NOW);
    expect(updated!.location).toBe("Berlin");
    expect(updated!.notes).toBe("Referral");
    expect(updated!.updated_at).toBe("2026-09-01T12:00:00.000Z");
    expect(app.location).toBe("Remote");
    expect(app.notes).toBeNull();
    expect(app.updated_at).toBe("2026-09-01T00:00:00.000Z");
  });

  it("rejects blanking required fields and invalid statuses", () => {
    expect(updateApplication(sampleApp(), { company: "" })).toBeNull();
    expect(updateApplication(sampleApp(), { role: "   " })).toBeNull();
    expect(updateApplication(sampleApp(), { status: "nope" as never })).toBeNull();
  });

  it("sanitizes the URL on update", () => {
    const updated = updateApplication(sampleApp(), { url: "javascript:alert(1)" });
    expect(updated!.url).toBeNull();
  });

  it("preserves scores unless patched", () => {
    const preserved = updateApplication(sampleApp(), { status: "offer" });
    expect(preserved!.scores.hybrid_match).toBe(75);
    const replaced = updateApplication(sampleApp(), { scores: { hybrid_match: 90 } });
    expect(replaced!.scores.hybrid_match).toBe(90);
    expect(replaced!.scores.baseline_match).toBeNull();
  });
});

describe("changeStatus", () => {
  it("changes the status and bumps updated_at", () => {
    const updated = changeStatus(sampleApp(), "interview", FIXED_NOW);
    expect(updated!.status).toBe("interview");
    expect(updated!.updated_at).toBe("2026-09-01T12:00:00.000Z");
  });
});

describe("validateApplicationDraft", () => {
  it("flags missing company and role", () => {
    const issues = validateApplicationDraft({ company: "", role: "  " });
    expect(issues.some((issue) => issue.path === "company")).toBe(true);
    expect(issues.some((issue) => issue.path === "role")).toBe(true);
  });

  it("flags overlong fields", () => {
    const issues = validateApplicationDraft({
      company: "x".repeat(MAX_COMPANY_CHARS + 1),
      role: "Engineer",
      notes: "n".repeat(5_001),
    });
    expect(issues.some((issue) => issue.path === "company")).toBe(true);
    expect(issues.some((issue) => issue.path === "notes")).toBe(true);
  });

  it("flags unsafe URLs, bad dates, and unknown statuses", () => {
    const issues = validateApplicationDraft({
      company: "Acme",
      role: "Engineer",
      url: "javascript:alert(1)",
      applied_date: "2026-02-30",
      status: "nope",
    });
    expect(issues.some((issue) => issue.path === "url")).toBe(true);
    expect(issues.some((issue) => issue.path === "applied_date")).toBe(true);
    expect(issues.some((issue) => issue.path === "status")).toBe(true);
  });

  it("accepts a clean draft", () => {
    expect(
      validateApplicationDraft({ company: "Acme", role: "Engineer", url: "https://acme.example/jobs/1" })
    ).toEqual([]);
  });
});

describe("addApplication / replaceApplication / removeApplication", () => {
  it("appends an application and refuses duplicate ids", () => {
    const apps: Application[] = [];
    const added = addApplication(apps, { company: "Acme", role: "Engineer" }, { id: "a1" });
    expect(added!.map((app) => app.id)).toEqual(["a1"]);
    expect(addApplication(added!, { company: "Other", role: "PM" }, { id: "a1" })).toBeNull();
  });

  it("replaces a tracked application and does nothing for unknown ids", () => {
    const apps = [sampleApp()];
    const replaced = replaceApplication(apps, { ...sampleApp(), role: "Senior Engineer" });
    expect(replaced![0].role).toBe("Senior Engineer");
    expect(replaceApplication(apps, { ...sampleApp(), id: "unknown" })).toBeNull();
  });

  it("removes a tracked application and does nothing for unknown ids", () => {
    const apps = [sampleApp()];
    expect(removeApplication(apps, "app_1")).toEqual([]);
    expect(removeApplication(apps, "nope")).toBeNull();
  });
});