import { getApiBaseUrl, ApiError } from "./api";
import type { Resume } from "./resume";

export interface RoleProfileInfo {
  title: string;
  aliases: string[];
  supported: boolean;
}

export interface RoleInfo {
  title: string;
  aliases: string[];
}

export interface SupportedRolesResponse {
  roles: RoleInfo[];
  total: number;
}

export interface SkillComparison {
  have: string[];
  missing: string[];
  matched_count: number;
  required_total: number;
  preferred_total: number;
}

export interface ExperienceComparison {
  candidate_years: number | null;
  required_years: number | null;
  required_level: string | null;
  gap_years: number | null;
  met: boolean | null;
  resume_experience_count: number;
}

export interface EducationComparison {
  required_level: string | null;
  candidate_level: string | null;
  met: boolean | null;
}

export interface RoleAnalysisResult {
  role_title: string;
  profile: RoleProfileInfo;
  compatibility: Record<string, number | null>;
  skills: SkillComparison;
  experience: ExperienceComparison;
  education: EducationComparison;
  matched_requirements: string[];
  missing_required: string[];
  strengths: string[];
  gaps: string[];
  recommendations: string[];
}

function isRoleAnalysisResult(payload: unknown): payload is RoleAnalysisResult {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  return (
    typeof record.role_title === "string" &&
    typeof record.profile === "object" &&
    record.profile !== null &&
    typeof (record.profile as Record<string, unknown>).supported === "boolean" &&
    Array.isArray(record.compatibility) === false &&
    typeof record.compatibility === "object" &&
    Array.isArray((record.skills as Record<string, unknown>)?.have) &&
    Array.isArray(record.recommendations)
  );
}

export async function analyzeRoleCompatibility(
  roleTitle: string,
  resume: Resume,
): Promise<RoleAnalysisResult> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/roles/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role_title: roleTitle, resume }),
    });
  } catch {
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError("unexpected_server_error", "Something went wrong while analyzing the role. Please try again.");
  }

  try {
    const payload: unknown = await response.json();
    if (!isRoleAnalysisResult(payload)) {
      throw new ApiError("invalid_response", "The backend returned an unexpected response. Please try again.");
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("invalid_response", "The backend returned an unexpected response. Please try again.");
  }
}

function isSupportedRolesResponse(payload: unknown): payload is SupportedRolesResponse {
  if (typeof payload !== "object" || payload === null) return false;
  const record = payload as Record<string, unknown>;
  if (!Array.isArray(record.roles)) return false;
  return record.roles.every(
    (r): r is RoleInfo =>
      typeof r === "object" &&
      r !== null &&
      typeof (r as Record<string, unknown>).title === "string" &&
      Array.isArray((r as Record<string, unknown>).aliases),
  );
}

export async function getSupportedRoles(query?: string): Promise<RoleInfo[]> {
  const base = `${getApiBaseUrl()}/api/v1/roles/supported`;
  const url = query ? `${base}?q=${encodeURIComponent(query)}` : base;
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new ApiError("network_unreachable", "We couldn't reach the ResumeForge backend.");
  }

  if (!response.ok) {
    throw new ApiError("unexpected_server_error", "Something went wrong while loading roles. Please try again.");
  }

  try {
    const payload: unknown = await response.json();
    if (!isSupportedRolesResponse(payload)) {
      throw new ApiError("invalid_response", "The backend returned an unexpected response. Please try again.");
    }
    return payload.roles;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("invalid_response", "The backend returned an unexpected response. Please try again.");
  }
}