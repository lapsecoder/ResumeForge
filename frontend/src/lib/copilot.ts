/**
 * Typed API client for the Resume Copilot (Phase 7A).
 *
 * Mirrors the backend's controlled-operation contract
 * (app/copilot). The Copilot is NOT a chat: the client requests one of a
 * finite set of operations with structured inputs, and receives grounded,
 * verification-labelled suggestions. Everything is transient and local —
 * nothing is stored, and no content is sent off the machine.
 */

import type { ATSReadinessResult, JobSpecificATSResult } from "./ats";
import type { HybridMatchResult, JobDescription } from "./matcher";
import type { Resume } from "./resume";
import { getApiBaseUrl, ApiError } from "./api";

export type CopilotOperation =
  | "improve_summary"
  | "improve_bullet"
  | "identify_priorities"
  | "explain_finding"
  | "job_alignment"
  | "free-form";

export type EvidenceKind =
  | "fact"
  | "job_requirement"
  | "inference"
  | "suggestion"
  | "unverified";

export type VerificationLevel = "verified" | "inferred" | "advisory" | "unverified";

export type SuggestionCategory =
  | "summary"
  | "bullet"
  | "skills"
  | "quantification"
  | "evidence"
  | "alignment"
  | "structure"
  | "priority"
  | "explanation"
  | "clarity"
  | "action_wording";

export type ProviderKind = "local-ollama" | "deterministic";

export type EditStatus = "proposed" | "applied" | "reverted" | "dismissed" | "unverified";

export type EditCheckCategory = "metric" | "date" | "skill" | "source_term";

export interface CopilotEvidence {
  kind: EvidenceKind;
  source: string;
  statement: string;
  reference?: string;
  quote?: string;
}

/** A structured, server-resolved resume location that may be edited. */
export interface CopilotEditTarget {
  path: string;
  section: string;
  index?: number | null;
  field?: string | null;
  sub_index?: number | null;
}

/** One selectable choice in a target clarification prompt. */
export interface CopilotClarificationOption {
  target: CopilotEditTarget;
  label: string;
}

/**
 * Structured "which target did you mean?" prompt for ambiguous free-form edit
 * requests. Produced when inference cannot uniquely resolve the target (e.g. a
 * technology shared by several projects); the backend never guesses. Selecting
 * an option resubmits the SAME request with that target's path as target_ref.
 */
export interface CopilotClarification {
  question: string;
  reason?: string;
  options: CopilotClarificationOption[];
}

export interface EditCheck {
  category: EditCheckCategory;
  passed: boolean;
  detail?: string;
}

export interface CopilotEditValidation {
  status: VerificationLevel;
  requires_user_confirmation: boolean;
  checks: EditCheck[];
}

/** A structured proposal to replace one resume value (Phase 7C). */
export interface CopilotEditProposal {
  edit_id: string;
  operation: CopilotOperation;
  target: CopilotEditTarget;
  original_value?: string;
  proposed_value?: string;
  reason?: string;
  evidence: CopilotEvidence[];
  validation: CopilotEditValidation;
  status: EditStatus;
}

export interface CopilotSuggestion {
  id: string;
  operation: CopilotOperation;
  category: SuggestionCategory;
  original_text?: string;
  suggested_text?: string;
  rationale: string;
  evidence: CopilotEvidence[];
  verification: VerificationLevel;
  requires_user_confirmation?: boolean;
  /** Rank within identify_priorities (1 = highest impact). */
  priority?: number | null;
  /** The concrete problem addressed (identify_priorities). */
  issue?: string;
  /** Practical fix for the issue (identify_priorities). */
  recommendation?: string;
  /** Expected weight of the fix: "high" | "medium" | "low". */
  impact?: string;
  /**
   * Structured edit proposal for replacement-text operations (Phase 7C).
   * Null/absent for advisory suggestions and for targets outside the
   * editable-field allowlist.
   */
  edit?: CopilotEditProposal | null;
}

export interface CopilotAnalysisContext {
  ats_readiness?: ATSReadinessResult | null;
  job_specific_ats?: JobSpecificATSResult | null;
  deterministic_match?: HybridMatchResult | null;
}

export interface ProviderMetadata {
  provider: ProviderKind;
  provider_label: string;
  model?: string | null;
  available: boolean;
  fallback_used: boolean;
  version: string;
  note?: string;
}

export interface CopilotResponse {
  operation: CopilotOperation;
  explanation: string;
  suggestions: CopilotSuggestion[];
  provider: ProviderMetadata;
  disclaimer: string;
  clarification?: CopilotClarification | null;
}

export interface CopilotStatus {
  providers: {
    provider: ProviderKind;
    provider_label: string;
    model?: string | null;
    available: boolean;
    note?: string;
  }[];
  fallback_available: boolean;
  version: string;
}

export interface CopilotSuggestRequest {
  operation: CopilotOperation;
  resume: Resume;
  job_description?: JobDescription;
  target_ref?: string;
  target_text?: string;
  finding_ref?: string;
  user_request?: string;
  analysis?: CopilotAnalysisContext;
}

/**
 * Hard upper bound for the generation call. Generation runs through a local
 * Ollama model and legitimately takes tens of seconds; anything beyond this is
 * treated as failed so the UI can never stay in "Generating…" indefinitely.
 */
export const COPILOT_SUGGEST_TIMEOUT_MS = 180_000;

/** Bounded wait for the lightweight status probe. */
export const COPILOT_STATUS_TIMEOUT_MS = 10_000;

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (controller.signal.aborted) {
      throw new ApiError(
        "request_timeout",
        "The request took too long and was cancelled. Please try again.",
      );
    }
    return response;
  } catch (error) {
    if (controller.signal.aborted) {
      throw new ApiError(
        "request_timeout",
        "The request took too long and was cancelled. Please try again.",
      );
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function isCopilotResponse(payload: unknown): payload is CopilotResponse {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    typeof record.operation === "string" &&
    typeof record.explanation === "string" &&
    Array.isArray(record.suggestions) &&
    typeof record.disclaimer === "string"
  );
}

function isCopilotStatus(payload: unknown): payload is CopilotStatus {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    Array.isArray(record.providers) &&
    typeof record.fallback_available === "boolean" &&
    typeof record.version === "string"
  );
}

/**
 * Run one controlled Copilot operation and return grounded suggestions.
 */
export async function suggestCopilot(request: CopilotSuggestRequest): Promise<CopilotResponse> {
  let response: Response;
  try {
    response = await fetchWithTimeout(
      `${getApiBaseUrl()}/api/v1/copilot/suggest`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      },
      COPILOT_SUGGEST_TIMEOUT_MS,
    );
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (response.status === 503) {
    throw new ApiError(
      "unexpected_server_error",
      "The resume assistant is unavailable right now. Your structured analysis still works."
    );
  }

  if (!response.ok) {
    throw new ApiError(
      "unexpected_server_error",
      "Something went wrong while preparing the suggestion. Please try again."
    );
  }

  try {
    const payload: unknown = await response.json();
    if (!isCopilotResponse(payload)) {
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
 * Report whether the local Copilot engines are available.
 */
export async function getCopilotStatus(): Promise<CopilotStatus> {
  let response: Response;
  try {
    response = await fetchWithTimeout(
      `${getApiBaseUrl()}/api/v1/copilot/status`,
      { method: "GET" },
      COPILOT_STATUS_TIMEOUT_MS,
    );
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError(
      "unexpected_server_error",
      "Something went wrong while checking the resume assistant. Please try again."
    );
  }

  try {
    const payload: unknown = await response.json();
    if (!isCopilotStatus(payload)) {
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