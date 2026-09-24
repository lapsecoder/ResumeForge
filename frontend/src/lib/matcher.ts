/**
 * Typed API client for job-description parsing and hybrid matching.
 *
 * Mirrors the backend transient schemas (app/job_parsing and
 * app/hybrid_matching). Matching is transient: nothing is stored, and the
 * semantic score is the local MiniLM model's relatedness signal — never a
 * hiring probability.
 */

import type { Resume } from "./resume";
import { getApiBaseUrl, ApiError } from "./api";

export interface JobMetadata {
  word_count: number;
  file_type: string;
  overall_confidence: "high" | "medium" | "low";
  section_confidence?: { section: string; level: string }[];
}

export interface JobDescription {
  title?: string | null;
  company?: string | null;
  location?: string | null;
  employment_type?: string | null;
  remote_type?: string | null;
  summary?: string | null;
  responsibilities?: string[];
  required_skills?: string[];
  preferred_skills?: string[];
  qualifications?: string[];
  experience_requirements?: string[];
  education_requirements?: string[];
  certifications?: string[];
  nice_to_have?: string[];
  benefits?: string[];
  salary?: unknown[];
  custom_sections?: unknown[];
  metadata: JobMetadata;
}

export interface HybridMatchResult {
  overall_score: number | null;
  deterministic: {
    overall_score: number | null;
    skill_match: { score: number | null };
    experience_match: { score: number | null };
    education_match: { score: number | null };
    matched_requirements: string[];
    missing_required: string[];
    strengths: string[];
    gaps: string[];
    metadata: { method: string; label: string; version: string };
  };
  semantic: {
    overall_similarity: number | null;
    note: string;
    metadata: {
      method: string;
      model_name: string;
      model_version: string;
      implementation_version: string;
      model_source: string;
      model_license: string;
      device: string;
    };
  } | null;
  component_scores: {
    deterministic_overall: number | null;
    semantic_overall: number | null;
    hybrid_overall: number | null;
    deterministic_skills: number | null;
    semantic_skills: number | null;
    deterministic_experience: number | null;
    semantic_experience: number | null;
  };
  matched_requirements: string[];
  missing_required: string[];
  strengths: string[];
  gaps: string[];
  semantic_insights: {
    category: string;
    evidence_level: "high" | "moderate" | "low";
    similarity: number | null;
    statement: string;
  }[];
  metadata: {
    method: string;
    label: string;
    version: string;
    weights: { deterministic: number; semantic: number };
    mode: "hybrid" | "deterministic-only" | "semantic-only" | "no-evidence";
    semantic_availability: {
      available: boolean;
      status: string;
      note: string;
    };
    note: string;
    model_disclosure: string;
  };
}

function isJobDescription(payload: unknown): payload is JobDescription {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    typeof record.metadata === "object" &&
    record.metadata !== null &&
    typeof (record.metadata as Record<string, unknown>).overall_confidence === "string"
  );
}

function isHybridMatchResult(payload: unknown): payload is HybridMatchResult {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    typeof record.deterministic === "object" &&
    record.deterministic !== null &&
    typeof record.metadata === "object" &&
    record.metadata !== null
  );
}

export async function parseJobDescription(file: File): Promise<JobDescription> {
  const formData = new FormData();
  formData.append("file", file, file.name);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/jobs/parse`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError(
      "parsing_failed",
      "We couldn't parse that job description. Please try another file."
    );
  }

  try {
    const payload: unknown = await response.json();
    if (!isJobDescription(payload)) {
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

export async function matchHybrid(
  resume: Resume,
  job: JobDescription
): Promise<HybridMatchResult> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/matching/hybrid`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume, job }),
    });
  } catch {
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError(
      "unexpected_server_error",
      "Something went wrong while running the match. Please try again."
    );
  }

  try {
    const payload: unknown = await response.json();
    if (!isHybridMatchResult(payload)) {
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