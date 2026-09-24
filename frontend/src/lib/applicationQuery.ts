/**
 * Phase 7E — pure query/aggregate helpers for the application tracker.
 *
 * Search, filter, sort, and dashboard statistics are deterministic functions
 * over an in-memory `Application[]`. None of them mutate their input, none of
 * them perform I/O, and all of them tolerate malformed entries defensively.
 */

import type { ATSReadinessResult, JobSpecificATSResult } from "./ats";
import type { HybridMatchResult, JobDescription } from "./matcher";
import {
  ACTIVE_STATUSES,
  APPLICATION_STATUSES,
  MAX_SEARCH_CHARS,
  RECENT_LIMIT,
  isApplicationStatus,
  normalizeScores,
  type Application,
  type ApplicationDraft,
  type ApplicationStatus,
} from "./applications";

export type SortKey = "date" | "company" | "role" | "score";
export type SortDirection = "asc" | "desc";

export interface TrackerQuery {
  search?: string;
  status?: ApplicationStatus | "all";
}

export interface DashboardAverages {
  baseline_match: number | null;
  hybrid_match: number | null;
  ats_readiness: number | null;
  job_specific_ats_coverage: number | null;
}

export interface TimelinePoint {
  month: string;
  count: number;
}

export interface DashboardStats {
  total: number;
  byStatus: Record<ApplicationStatus, number>;
  saved: number;
  active: number;
  interviews: number;
  offers: number;
  rejections: number;
  withdrawn: number;
  recent: Application[];
  averages: DashboardAverages;
  timeline: TimelinePoint[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function recordEntries(applications: Application[]): Application[] {
  return applications.filter((application) => isRecord(application));
}

function readText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/**
 * Case-insensitive search across company, role, and location, combined with an
 * optional status filter. The query is bounded and never throws.
 */
export function filterApplications(
  applications: Application[],
  query: TrackerQuery = {}
): Application[] {
  const search =
    typeof query.search === "string"
      ? query.search.trim().toLowerCase().slice(0, MAX_SEARCH_CHARS)
      : "";
  const status = query.status && query.status !== "all" ? query.status : null;

  return applications.filter((application) => {
    if (!isRecord(application)) return false;
    if (status && application.status !== status) return false;
    if (!search) return true;
    const haystack = [application.company, application.role, application.location]
      .map(readText)
      .join(" ")
      .toLowerCase();
    return haystack.includes(search);
  });
}

function compareText(a: string, b: string): number {
  return a.toLowerCase().localeCompare(b.toLowerCase());
}

function compareNullableText(a: string | null, b: string | null, factor: number): number {
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  return compareText(a, b) * factor;
}

function scoreOf(application: Application): number | null {
  const scores = normalizeScores(application.scores);
  return scores.hybrid_match ?? scores.baseline_match ?? null;
}

function dateOf(application: Application): string | null {
  if (typeof application.applied_date === "string") return application.applied_date;
  if (typeof application.created_at === "string") return application.created_at.slice(0, 10);
  return null;
}

/**
 * Deterministic sort with stable tie-breaking by company then id. Entries with
 * no date or no score always sort last, regardless of direction.
 */
export function sortApplications(
  applications: Application[],
  key: SortKey,
  direction: SortDirection = "desc"
): Application[] {
  const factor = direction === "asc" ? 1 : -1;
  return [...applications].sort((a, b) => {
    let primary = 0;
    if (key === "company") {
      primary = compareText(readText(a.company), readText(b.company)) * factor;
    } else if (key === "role") {
      primary = compareText(readText(a.role), readText(b.role)) * factor;
    } else if (key === "date") {
      primary = compareNullableText(dateOf(a), dateOf(b), factor);
    } else {
      const aScore = scoreOf(a);
      const bScore = scoreOf(b);
      if (aScore === null && bScore === null) primary = 0;
      else if (aScore === null) primary = 1;
      else if (bScore === null) primary = -1;
      else primary = (aScore - bScore) * factor;
    }
    if (primary !== 0) return primary;
    const company = compareText(readText(a.company), readText(b.company));
    if (company !== 0) return company;
    return compareText(readText(a.id), readText(b.id));
  });
}

function average(values: (number | null)[]): number | null {
  const present = values.filter((value): value is number => typeof value === "number");
  if (present.length === 0) return null;
  const sum = present.reduce((total, value) => total + value, 0);
  return Math.round((sum / present.length) * 10) / 10;
}

/** Descriptive statistics over the tracked applications. Never predictive. */
export function computeDashboard(applications: Application[]): DashboardStats {
  const list = recordEntries(applications);
  const byStatus = Object.fromEntries(
    APPLICATION_STATUSES.map((status) => [status, 0])
  ) as Record<ApplicationStatus, number>;

  for (const application of list) {
    if (isApplicationStatus(application.status)) byStatus[application.status] += 1;
  }

  const active = ACTIVE_STATUSES.reduce((total, status) => total + byStatus[status], 0);

  const recent = [...list]
    .sort((a, b) => compareNullableText(dateOf(a), dateOf(b), -1))
    .slice(0, RECENT_LIMIT);

  const scores = list.map((application) => normalizeScores(application.scores));
  const averages: DashboardAverages = {
    baseline_match: average(scores.map((score) => score.baseline_match)),
    hybrid_match: average(scores.map((score) => score.hybrid_match)),
    ats_readiness: average(scores.map((score) => score.ats_readiness)),
    job_specific_ats_coverage: average(
      scores.map((score) => score.job_specific_ats_coverage)
    ),
  };

  const monthCounts = new Map<string, number>();
  for (const application of list) {
    const date = typeof application.applied_date === "string" ? application.applied_date : "";
    if (date.length < 7) continue;
    const month = date.slice(0, 7);
    monthCounts.set(month, (monthCounts.get(month) ?? 0) + 1);
  }
  const timeline = [...monthCounts.entries()]
    .map(([month, count]) => ({ month, count }))
    .sort((a, b) => (a.month < b.month ? -1 : a.month > b.month ? 1 : 0));

  return {
    total: list.length,
    byStatus,
    saved: byStatus.saved,
    active,
    interviews: byStatus.interview,
    offers: byStatus.offer,
    rejections: byStatus.rejected,
    withdrawn: byStatus.withdrawn,
    recent,
    averages,
    timeline,
  };
}

export interface PrefillSources {
  job?: JobDescription | null;
  match?: HybridMatchResult | null;
  ats?: ATSReadinessResult | null;
  jobAts?: JobSpecificATSResult | null;
  resumeTemplate?: string | null;
}

/**
 * Build an editable draft from a parsed/matched job. Only values that actually
 * exist are copied; missing values stay null instead of being guessed.
 */
export function buildPrefill(sources: PrefillSources): ApplicationDraft {
  const { job, match, ats, jobAts } = sources;
  return {
    company: job?.company ?? null,
    role: job?.title ?? null,
    location: job?.location ?? null,
    url: null,
    applied_date: null,
    status: "saved",
    notes: null,
    resume_template: sources.resumeTemplate ?? null,
    scores: {
      baseline_match: match?.deterministic?.overall_score ?? null,
      hybrid_match: match?.overall_score ?? null,
      ats_readiness: ats?.overall_score ?? null,
      job_specific_ats_coverage: jobAts?.coverage?.overall_coverage ?? null,
    },
  };
}
