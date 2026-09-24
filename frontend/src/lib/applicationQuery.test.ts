import { describe, expect, it } from "vitest";

import type { ATSReadinessResult, JobSpecificATSResult } from "./ats";
import type { HybridMatchResult } from "./matcher";
import {
  buildPrefill,
  computeDashboard,
  filterApplications,
  sortApplications,
} from "./applicationQuery";
import type { Application } from "./applications";

function sampleApp(overrides: Partial<Application> = {}): Application {
  return {
    id: "app_1",
    company: "Acme Corp",
    role: "Software Engineer",
    location: "Remote",
    url: null,
    applied_date: "2026-09-01",
    status: "applied",
    notes: null,
    resume_template: null,
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

describe("filterApplications", () => {
  it("is empty for an empty list", () => {
    expect(filterApplications([])).toEqual([]);
  });

  it("searches company, role, and location case-insensitively", () => {
    const apps = [
      sampleApp(),
      sampleApp({ id: "app_2", company: "Globex", role: "Product Manager", location: "London" }),
    ];
    expect(filterApplications(apps, { search: "acme" }).map((app) => app.id)).toEqual(["app_1"]);
    expect(filterApplications(apps, { search: "LONDON" }).map((app) => app.id)).toEqual(["app_2"]);
    expect(filterApplications(apps, { search: "engin" }).map((app) => app.id)).toEqual(["app_1"]);
    expect(filterApplications(apps, { search: "nope" })).toEqual([]);
  });

  it("filters by status and combines status with search", () => {
    const apps = [
      sampleApp({ status: "applied" }),
      sampleApp({ id: "app_2", company: "Globex", status: "offer" }),
      sampleApp({ id: "app_3", company: "Initech", status: "applied" }),
    ];
    expect(filterApplications(apps, { status: "offer" }).map((app) => app.id)).toEqual(["app_2"]);
    expect(filterApplications(apps, { status: "all" }).map((app) => app.id)).toHaveLength(3);
    expect(filterApplications(apps, { search: "itech", status: "applied" }).map((app) => app.id)).toEqual([
      "app_3",
    ]);
  });

  it("tolerates malformed entries without throwing", () => {
    const apps: Application[] = [sampleApp(), null as never, {} as never];
    expect(filterApplications(apps, { search: "acme" }).map((app) => app.id)).toEqual(["app_1"]);
    expect(filterApplications(apps, { search: "zzz" })).toEqual([]);
  });

  it("bounds the search term", () => {
    const apps = [sampleApp()];
    expect(filterApplications(apps, { search: "a".repeat(500) })).toEqual([]);
  });
});

describe("sortApplications", () => {
  const apps = [
    sampleApp({ id: "a", applied_date: null, created_at: "2026-08-01T00:00:00.000Z" }),
    sampleApp({ id: "b", applied_date: "2026-09-10" }),
    sampleApp({ id: "c", applied_date: "2026-01-15" }),
  ];

  it("sorts by date desc and asc using applied_date with created_at fallback", () => {
    // a has created_at 2026-08-01, b has applied_date 2026-09-10, c has applied_date 2026-01-15
    // desc: b (09-10), a (08-01), c (01-15)
    // asc: c (01-15), a (08-01), b (09-10)
    expect(sortApplications(apps, "date", "desc").map((app) => app.id)).toEqual(["b", "a", "c"]);
    expect(sortApplications(apps, "date", "asc").map((app) => app.id)).toEqual(["c", "a", "b"]);
  });

  it("sorts by company and role text", () => {
    const mixed = [
      sampleApp({ id: "x", company: "zeta", role: "Engineer" }),
      sampleApp({ id: "y", company: "alpha", role: "PM" }),
    ];
    expect(sortApplications(mixed, "company", "asc").map((app) => app.id)).toEqual(["y", "x"]);
    expect(sortApplications(mixed, "company", "desc").map((app) => app.id)).toEqual(["x", "y"]);
  });

  it("sorts by score using hybrid then baseline, nulls last", () => {
    const mixed = [
      sampleApp({ id: "low", scores: { hybrid_match: 30, baseline_match: 30, ats_readiness: null, job_specific_ats_coverage: null } }),
      sampleApp({ id: "high", scores: { hybrid_match: 90, baseline_match: 90, ats_readiness: null, job_specific_ats_coverage: null } }),
      sampleApp({ id: "none", scores: { hybrid_match: null, baseline_match: null, ats_readiness: null, job_specific_ats_coverage: null } }),
    ];
    expect(sortApplications(mixed, "score", "desc").map((app) => app.id)).toEqual(["high", "low", "none"]);
    expect(sortApplications(mixed, "score", "asc").map((app) => app.id)).toEqual(["low", "high", "none"]);
  });

  it("does not mutate its input", () => {
    const before = apps.map((app) => app.id).join(",");
    sortApplications(apps, "date");
    expect(apps.map((app) => app.id).join(",")).toBe(before);
  });
});

describe("computeDashboard", () => {
  it("returns zeroed stats for an empty list", () => {
    const stats = computeDashboard([]);
    expect(stats.total).toBe(0);
    expect(stats.active).toBe(0);
    expect(stats.averages.hybrid_match).toBeNull();
    expect(stats.timeline).toEqual([]);
    expect(stats.recent).toEqual([]);
  });

  it("counts by status and active pipeline", () => {
    const stats = computeDashboard([
      sampleApp({ status: "applied" }),
      sampleApp({ id: "s", status: "saved" }),
      sampleApp({ id: "i", status: "interview" }),
      sampleApp({ id: "o", status: "offer" }),
      sampleApp({ id: "r", status: "rejected" }),
      sampleApp({ id: "w", status: "withdrawn" }),
    ]);
    expect(stats.total).toBe(6);
    expect(stats.active).toBe(2);
    expect(stats.interviews).toBe(1);
    expect(stats.offers).toBe(1);
    expect(stats.rejections).toBe(1);
    expect(stats.withdrawn).toBe(1);
  });

  it("averages scores rounded to one decimal", () => {
    const stats = computeDashboard([
      sampleApp({ scores: { baseline_match: 70, hybrid_match: 75, ats_readiness: 80, job_specific_ats_coverage: 0.7 } }),
      sampleApp({ id: "b", scores: { baseline_match: 30, hybrid_match: 25, ats_readiness: null, job_specific_ats_coverage: null } }),
    ]);
    expect(stats.averages.baseline_match).toBe(50);
    expect(stats.averages.hybrid_match).toBe(50);
    expect(stats.averages.ats_readiness).toBe(80);
    expect(stats.averages.job_specific_ats_coverage).toBe(0.7);
  });

  it("lists the most recent five and buckets the timeline by month", () => {
    const apps = Array.from({ length: 6 }, (_, index) =>
      sampleApp({
        id: `app_${index}`,
        applied_date: `2026-0${9 - index}-01`,
        created_at: "2026-09-01T05:00:00.000Z",
      })
    );
    const stats = computeDashboard(apps);
    expect(stats.recent).toHaveLength(5);
    expect(stats.timeline.map((point) => point.month)).toEqual([
      "2026-04",
      "2026-05",
      "2026-06",
      "2026-07",
      "2026-08",
      "2026-09",
    ]);
    expect(stats.timeline.find((point) => point.month === "2026-09")!.count).toBe(1);
  });
});

describe("buildPrefill", () => {
  it("produces a blank draft when no sources exist", () => {
    const draft = buildPrefill({});
    expect(draft).toEqual({
      company: null,
      role: null,
      location: null,
      url: null,
      applied_date: null,
      status: "saved",
      notes: null,
      resume_template: null,
      scores: {
        baseline_match: null,
        hybrid_match: null,
        ats_readiness: null,
        job_specific_ats_coverage: null,
      },
    });
  });

  it("copies only what exists from a matched job and carries scores", () => {
    const job = {
      title: "Engineer",
      company: "Acme",
      location: "Remote",
      metadata: { word_count: 40, file_type: "pdf", overall_confidence: "high" as const },
    };
    const match = {
      overall_score: 81,
      deterministic: { overall_score: 80 },
    } as unknown as HybridMatchResult;
    const ats = { overall_score: 70 } as ATSReadinessResult;
    const jobAts = { coverage: { overall_coverage: 0.6 } } as unknown as JobSpecificATSResult;

    const draft = buildPrefill({ job, match, ats, jobAts, resumeTemplate: "modern" });
    expect(draft.company).toBe("Acme");
    expect(draft.role).toBe("Engineer");
    expect(draft.location).toBe("Remote");
    expect(draft.url).toBeNull();
    expect(draft.status).toBe("saved");
    expect(draft.resume_template).toBe("modern");
    expect(draft.scores).toEqual({
      baseline_match: 80,
      hybrid_match: 81,
      ats_readiness: 70,
      job_specific_ats_coverage: 0.6,
    });
  });
});