/**
 * Phase 7E — client-side application tracker model.
 *
 * Everything here is pure and in-memory. Applications are created, updated,
 * searched, sorted, and aggregated from a plain array held in React state for
 * the current browser session; nothing is ever persisted or uploaded.
 *
 * All user-controlled text is sanitized and all job URLs are treated as
 * untrusted (only http/https survive). Scores keep their original meaning and
 * units: match/ATS readiness scores are 0-100, coverage is a 0-1 ratio.
 */

export const APPLICATION_STATUSES = [
  "saved",
  "applied",
  "screening",
  "interview",
  "offer",
  "rejected",
  "withdrawn",
] as const;

export type ApplicationStatus = (typeof APPLICATION_STATUSES)[number];

export const STATUS_LABELS: Record<ApplicationStatus, string> = {
  saved: "Saved",
  applied: "Applied",
  screening: "Screening",
  interview: "Interview",
  offer: "Offer",
  rejected: "Rejected",
  withdrawn: "Withdrawn",
};

/** Statuses that represent an application still in progress (not a decision). */
export const ACTIVE_STATUSES: readonly ApplicationStatus[] = ["applied", "screening", "interview"];

export const MAX_COMPANY_CHARS = 200;
export const MAX_ROLE_CHARS = 200;
export const MAX_LOCATION_CHARS = 200;
export const MAX_URL_CHARS = 2000;
export const MAX_NOTES_CHARS = 5000;
export const MAX_TEMPLATE_CHARS = 60;
export const MAX_ID_CHARS = 128;
export const MAX_SEARCH_CHARS = 100;
export const RECENT_LIMIT = 5;

export interface ApplicationScores {
  /** Baseline (deterministic) Match Score, 0-100. */
  baseline_match: number | null;
  /** Hybrid Match Score, 0-100. */
  hybrid_match: number | null;
  /** ATS Readiness overall score, 0-100. */
  ats_readiness: number | null;
  /** Job-Specific ATS Coverage, 0-1 ratio. */
  job_specific_ats_coverage: number | null;
}

export interface Application {
  id: string;
  company: string;
  role: string;
  location: string | null;
  url: string | null;
  applied_date: string | null;
  status: ApplicationStatus;
  notes: string | null;
  resume_template: string | null;
  scores: ApplicationScores;
  created_at: string;
  updated_at: string;
}

export interface ApplicationDraft {
  company?: string | null;
  role?: string | null;
  location?: string | null;
  url?: string | null;
  applied_date?: string | null;
  status?: string | null;
  notes?: string | null;
  resume_template?: string | null;
  scores?: Partial<ApplicationScores> | null;
}

export interface ValidationIssue {
  path: string;
  message: string;
}

export interface CreateApplicationOptions {
  id?: string;
  now?: Date;
}

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const ISO_PREFIX_RE = /^(\d{4}-\d{2}-\d{2})/;
const BARE_DOMAIN_RE = /^[a-z0-9-]+(\.[a-z0-9-]+)+(\/[^\s]*)?$/i;

export function isApplicationStatus(value: unknown): value is ApplicationStatus {
  return typeof value === "string" && (APPLICATION_STATUSES as readonly string[]).includes(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stripControl(value: string): string {
  return value.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "");
}

/** Trimmed, control-stripped text bounded by `max`; null when blank/invalid. */
export function sanitizeOptionalText(value: unknown, max: number): string | null {
  if (typeof value !== "string") return null;
  const cleaned = stripControl(value).trim();
  if (!cleaned || cleaned.length > max) return null;
  return cleaned;
}

/** Like `sanitizeOptionalText`, but null also means "missing or too long". */
export function sanitizeRequiredText(value: unknown, max: number): string | null {
  return sanitizeOptionalText(value, max);
}

function sanitizeId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const cleaned = stripControl(value).trim();
  if (!cleaned || cleaned.length > MAX_ID_CHARS) return null;
  return cleaned;
}

/** Normalize a YYYY-MM-DD (or ISO datetime) value; null when not a real date. */
export function normalizeDate(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const raw = value.trim();
  const match = ISO_PREFIX_RE.exec(raw);
  if (!match) return null;
  const datePart = match[1];
  if (!DATE_RE.test(datePart)) return null;
  const [year, month, day] = datePart.split("-").map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() !== month - 1 ||
    parsed.getUTCDate() !== day
  ) {
    return null;
  }
  return datePart;
}

function normalizeBoundedNumber(value: unknown, max: number): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (value < 0 || value > max) return null;
  return Math.round(value * 100) / 100;
}

/** Rebuild a trusted scores object; invalid values become null. */
export function normalizeScores(input: unknown): ApplicationScores {
  const record = isRecord(input) ? input : {};
  return {
    baseline_match: normalizeBoundedNumber(record.baseline_match, 100),
    hybrid_match: normalizeBoundedNumber(record.hybrid_match, 100),
    ats_readiness: normalizeBoundedNumber(record.ats_readiness, 100),
    job_specific_ats_coverage: normalizeBoundedNumber(record.job_specific_ats_coverage, 1),
  };
}

/**
 * Only http/https URLs survive. Everything else (javascript:, data:, file:,
 * vbscript:, malformed, whitespace-laden) becomes null. Bare domains are
 * upgraded to https. The URL is never fetched.
 */
export function safeExternalUrl(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const raw = value.trim();
  if (!raw || raw.length > MAX_URL_CHARS || /\s/.test(raw)) return null;
  let candidate = raw;
  if (!/^[a-z][a-z0-9+.-]*:/i.test(raw)) {
    if (!BARE_DOMAIN_RE.test(raw)) return null;
    candidate = `https://${raw}`;
  }
  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    return null;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
  return parsed.toString();
}

export function generateApplicationId(): string {
  const cryptoObject = globalThis.crypto;
  if (cryptoObject && typeof cryptoObject.randomUUID === "function") {
    return cryptoObject.randomUUID();
  }
  const random = Math.random().toString(36).slice(2, 10);
  return `app-${Date.now().toString(36)}-${random}`;
}

export function createApplication(
  draft: ApplicationDraft,
  options: CreateApplicationOptions = {}
): Application | null {
  if (!isRecord(draft)) return null;
  const company = sanitizeRequiredText(draft.company, MAX_COMPANY_CHARS);
  const role = sanitizeRequiredText(draft.role, MAX_ROLE_CHARS);
  if (!company || !role) return null;

  const now = (options.now ?? new Date()).toISOString();
  const id = sanitizeId(options.id) ?? generateApplicationId();
  return {
    id,
    company,
    role,
    location: sanitizeOptionalText(draft.location, MAX_LOCATION_CHARS),
    url: safeExternalUrl(draft.url),
    applied_date: normalizeDate(draft.applied_date),
    status: isApplicationStatus(draft.status) ? draft.status : "saved",
    notes: sanitizeOptionalText(draft.notes, MAX_NOTES_CHARS),
    resume_template: sanitizeOptionalText(draft.resume_template, MAX_TEMPLATE_CHARS),
    scores: normalizeScores(draft.scores),
    created_at: now,
    updated_at: now,
  };
}

/**
 * Rebuild a trusted Application from arbitrary input. Returns null when the
 * object is malformed or its required fields are unusable.
 */
export function normalizeApplication(input: unknown): Application | null {
  if (!isRecord(input)) return null;
  const company = sanitizeRequiredText(input.company, MAX_COMPANY_CHARS);
  const role = sanitizeRequiredText(input.role, MAX_ROLE_CHARS);
  const id = sanitizeId(input.id);
  if (!company || !role || !id) return null;

  const createdAt =
    typeof input.created_at === "string" && !Number.isNaN(Date.parse(input.created_at))
      ? input.created_at
      : new Date(0).toISOString();
  const updatedAt =
    typeof input.updated_at === "string" && !Number.isNaN(Date.parse(input.updated_at))
      ? input.updated_at
      : createdAt;

  return {
    id,
    company,
    role,
    location: sanitizeOptionalText(input.location, MAX_LOCATION_CHARS),
    url: safeExternalUrl(input.url),
    applied_date: normalizeDate(input.applied_date),
    status: isApplicationStatus(input.status) ? input.status : "saved",
    notes: sanitizeOptionalText(input.notes, MAX_NOTES_CHARS),
    resume_template: sanitizeOptionalText(input.resume_template, MAX_TEMPLATE_CHARS),
    scores: normalizeScores(input.scores),
    created_at: createdAt,
    updated_at: updatedAt,
  };
}

/**
 * Apply a partial patch. Required fields may not be blanked; invalid values are
 * rejected (null) rather than silently coerced. Immutable: returns a new object.
 */
export function updateApplication(
  application: Application,
  patch: ApplicationDraft,
  now: Date = new Date()
): Application | null {
  const base = normalizeApplication(application);
  if (!base || !isRecord(patch)) return null;

  const next: Application = { ...base };
  if ("company" in patch) {
    const company = sanitizeRequiredText(patch.company, MAX_COMPANY_CHARS);
    if (!company) return null;
    next.company = company;
  }
  if ("role" in patch) {
    const role = sanitizeRequiredText(patch.role, MAX_ROLE_CHARS);
    if (!role) return null;
    next.role = role;
  }
  if ("location" in patch) next.location = sanitizeOptionalText(patch.location, MAX_LOCATION_CHARS);
  if ("url" in patch) next.url = safeExternalUrl(patch.url);
  if ("applied_date" in patch) next.applied_date = normalizeDate(patch.applied_date);
  if ("status" in patch) {
    if (!isApplicationStatus(patch.status)) return null;
    next.status = patch.status;
  }
  if ("notes" in patch) next.notes = sanitizeOptionalText(patch.notes, MAX_NOTES_CHARS);
  if ("resume_template" in patch) {
    next.resume_template = sanitizeOptionalText(patch.resume_template, MAX_TEMPLATE_CHARS);
  }
  if ("scores" in patch) next.scores = normalizeScores(patch.scores);
  next.updated_at = now.toISOString();
  return next;
}

export function changeStatus(
  application: Application,
  status: ApplicationStatus,
  now: Date = new Date()
): Application | null {
  return updateApplication(application, { status }, now);
}

export function validateApplicationDraft(draft: ApplicationDraft): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const company = draft.company;
  if (typeof company !== "string" || !company.trim()) {
    issues.push({ path: "company", message: "Company is required." });
  } else if (company.trim().length > MAX_COMPANY_CHARS) {
    issues.push({ path: "company", message: `Company must be ${MAX_COMPANY_CHARS} characters or fewer.` });
  }
  const role = draft.role;
  if (typeof role !== "string" || !role.trim()) {
    issues.push({ path: "role", message: "Job title is required." });
  } else if (role.trim().length > MAX_ROLE_CHARS) {
    issues.push({ path: "role", message: `Job title must be ${MAX_ROLE_CHARS} characters or fewer.` });
  }
  if (typeof draft.location === "string" && draft.location.trim().length > MAX_LOCATION_CHARS) {
    issues.push({ path: "location", message: `Location must be ${MAX_LOCATION_CHARS} characters or fewer.` });
  }
  if (typeof draft.notes === "string" && draft.notes.trim().length > MAX_NOTES_CHARS) {
    issues.push({ path: "notes", message: `Notes must be ${MAX_NOTES_CHARS} characters or fewer.` });
  }
  if (draft.url !== undefined && draft.url !== null && String(draft.url).trim()) {
    if (!safeExternalUrl(draft.url)) {
      issues.push({ path: "url", message: "Enter a valid http(s) job link." });
    }
  }
  if (draft.applied_date !== undefined && draft.applied_date !== null && String(draft.applied_date).trim()) {
    if (!normalizeDate(draft.applied_date)) {
      issues.push({ path: "applied_date", message: "Enter a valid application date." });
    }
  }
  if (draft.status !== undefined && draft.status !== null && !isApplicationStatus(draft.status)) {
    issues.push({ path: "status", message: "Choose a valid status." });
  }
  return issues;
}

/** Append an application. Refuses to overwrite an existing id. */
export function addApplication(
  applications: Application[],
  draft: ApplicationDraft,
  options: CreateApplicationOptions = {}
): Application[] | null {
  const application = createApplication(draft, options);
  if (!application) return null;
  if (applications.some((existing) => existing.id === application.id)) return null;
  return [...applications, application];
}

/** Replace an application by id; null when it is not tracked. */
export function replaceApplication(
  applications: Application[],
  updated: Application
): Application[] | null {
  const normalized = normalizeApplication(updated);
  if (!normalized) return null;
  if (!applications.some((existing) => existing.id === normalized.id)) return null;
  return applications.map((existing) => (existing.id === normalized.id ? normalized : existing));
}

/** Remove an application by id; null when it is not tracked. */
export function removeApplication(applications: Application[], id: string): Application[] | null {
  if (!applications.some((existing) => existing.id === id)) return null;
  return applications.filter((existing) => existing.id !== id);
}
