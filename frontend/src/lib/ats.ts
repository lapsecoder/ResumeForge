/**
 * Typed API client for the deterministic ATS analyses (Phase 6A/6B).
 *
 * These endpoints evaluate an already-parsed resume (optionally against a
 * parsed job description) using explainable heuristics. They are transient:
 * nothing is stored, and results are never hiring/ATS-pass predictions. The
 * Copilot consumes these analyses as grounding for its suggestions.
 */

import type { JobDescription } from "./matcher";
import type { Resume } from "./resume";
import { getApiBaseUrl, ApiError } from "./api";

export type Severity = "info" | "low" | "medium" | "high";

export interface Finding {
  category: string;
  severity: Severity;
  rule_id: string;
  title: string;
  explanation: string;
  evidence: string;
  recommendation: string;
  impact: number;
}

export interface CategoryScore {
  key: string;
  label: string;
  score: number | null;
  applicable: boolean;
  weight: number;
  max_weight: number;
}

export interface AtsMetadata {
  method: string;
  version: string;
  weights: Record<string, number>;
  applied_weights: Record<string, number>;
  score_labels: Record<string, string>;
  disclaimer: string;
}

export interface ATSReadinessResult {
  overall_score: number | null;
  score_label: string | null;
  category_scores: CategoryScore[];
  findings: Finding[];
  metadata: AtsMetadata;
}

export type MatchType = "exact" | "normalized" | "alias" | "phrase" | "absent";
export type TermOrigin = "required" | "preferred" | "phrase";

export interface TermMatch {
  term: string;
  origin: TermOrigin;
  match_type: MatchType;
  evidence_locations: string[];
  evidence: string;
  explanation: string;
}

export interface CoverageTotals {
  required: number;
  required_matched: number;
  preferred: number;
  preferred_matched: number;
  phrases: number;
  phrases_matched: number;
}

export interface CoverageReport {
  required_coverage: number | null;
  preferred_coverage: number | null;
  overall_coverage: number | null;
  evidence_supported_required: number | null;
  evidence_supported_preferred: number | null;
  totals: CoverageTotals;
}

export interface JobSpecificATSResult {
  overall_score: number | null;
  score_label: string | null;
  category_scores: CategoryScore[];
  coverage: CoverageReport;
  term_matches: TermMatch[];
  findings: Finding[];
  metadata: AtsMetadata;
}

function isAtsReadinessResult(payload: unknown): payload is ATSReadinessResult {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    Array.isArray(record.category_scores) &&
    Array.isArray(record.findings) &&
    typeof record.metadata === "object" &&
    record.metadata !== null
  );
}

function isJobSpecificATSResult(payload: unknown): payload is JobSpecificATSResult {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    Array.isArray(record.category_scores) &&
    Array.isArray(record.findings) &&
    typeof record.coverage === "object" &&
    record.coverage !== null &&
    Array.isArray(record.term_matches)
  );
}

/**
 * Run a deterministic ATS Readiness analysis on a parsed resume.
 */
export async function analyzeAtsReadiness(resume: Resume): Promise<ATSReadinessResult> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/resumes/ats-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume }),
    });
  } catch {
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError(
      "unexpected_server_error",
      "Something went wrong while analysing the resume. Please try again."
    );
  }

  try {
    const payload: unknown = await response.json();
    if (!isAtsReadinessResult(payload)) {
      throw new ApiError(
        "invalid_response",
        "The backend returned an unexpected response. Please try again."
      );
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      "invalid_response",
      "The backend returned an unexpected response. Please try again."
    );
  }
}

/**
 * Run a Job-Specific ATS Coverage analysis against a parsed job description.
 */
export async function analyzeJobSpecificAts(
  resume: Resume,
  job: JobDescription
): Promise<JobSpecificATSResult> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/resumes/job-specific-ats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume, job_description: job }),
    });
  } catch {
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError(
      "unexpected_server_error",
      "Something went wrong while running the job analysis. Please try again."
    );
  }

  try {
    const payload: unknown = await response.json();
    if (!isJobSpecificATSResult(payload)) {
      throw new ApiError(
        "invalid_response",
        "The backend returned an unexpected response. Please try again."
      );
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      "invalid_response",
      "The backend returned an unexpected response. Please try again."
    );
  }
}