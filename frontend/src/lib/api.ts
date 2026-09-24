/**
 * Typed API client for the ResumeForge backend.
 *
 * Thin, centralised layer over fetch. The backend URL is read from
 * NEXT_PUBLIC_API_BASE_URL; localhost is only a convenience fallback for
 * development when the variable is unset.
 *
 * Privacy: this module never logs resume contents, filenames, or contact
 * information. Errors are logged only by their stable code so PII cannot leak
 * into the browser console.
 */

import type { Resume } from "./resume";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL;
}

export type ApiErrorCode =
  | "unsupported_file_type"
  | "file_too_large"
  | "empty_content"
  | "empty_file"
  | "empty_document"
  | "malformed_file"
  | "ocr_required"
  | "extraction_failed"
  | "parsing_failed"
  | "network_unreachable"
  | "invalid_response"
  | "request_timeout"
  | "unexpected_server_error";

export class ApiError extends Error {
  readonly code: ApiErrorCode;

  constructor(code: ApiErrorCode, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

const ERROR_MESSAGES: Record<ApiErrorCode, string> = {
  unsupported_file_type: "Unsupported file type. Please upload a PDF, DOCX, or TXT file.",
  file_too_large: `The file is too large. Maximum size is 10 MB.`,
  empty_content: "The file does not contain any readable text. Please try another file.",
  empty_file: "The selected file is empty. Please choose a different file.",
  empty_document: "The file does not contain any readable text. Please try another file.",
  malformed_file: "The file appears to be malformed or corrupted. Please try another file.",
  ocr_required:
    "We couldn't read this file. Scanned or image-only resumes are not supported yet.",
  extraction_failed: "We couldn't extract your resume. Please try again.",
  parsing_failed: "We couldn't structure your resume. Please try again.",
  network_unreachable:
    "We couldn't reach the ResumeForge backend. Please make sure it is running, then try again.",
  invalid_response: "The backend returned an unexpected response. Please try again.",
  request_timeout: "The request took too long and was cancelled. Please try again.",
  unexpected_server_error:
    "Something went wrong while analyzing your resume. Please try again.",
};

export function apiErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error && error.message) {
    // Network-layer failures (TypeError) surface here.
    return ERROR_MESSAGES.network_unreachable;
  }
  return ERROR_MESSAGES.unexpected_server_error;
}

function isResumePayload(value: unknown): value is Resume {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.metadata === "object" &&
    record.metadata !== null &&
    typeof (record.metadata as Record<string, unknown>).overall_confidence === "string"
  );
}

async function readErrorCode(response: Response): Promise<ApiErrorCode | null> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null) {
      const code = (body as { error?: { code?: unknown } }).error?.code;
      if (typeof code === "string") {
        const known: ApiErrorCode[] = [
          "unsupported_file_type",
          "file_too_large",
          "empty_content",
          "empty_file",
          "empty_document",
          "malformed_file",
          "ocr_required",
          "extraction_failed",
          "parsing_failed",
        ];
        return known.includes(code as ApiErrorCode) ? (code as ApiErrorCode) : null;
      }
    }
  } catch {
    // Non-JSON bodies are handled below with status-based mapping.
  }
  return null;
}

/**
 * Upload a resume and return the parsed result. Transient end-to-end:
 * the backend processes the file in memory and never stores it.
 */
export async function parseResume(file: File): Promise<Resume> {
  const formData = new FormData();
  formData.append("file", file, file.name);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/resumes/parse`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new ApiError("network_unreachable", ERROR_MESSAGES.network_unreachable);
  }

  if (!response.ok) {
    const code = (await readErrorCode(response)) ?? mapStatusToCode(response.status);
    console.error("Resume parse failed with code:", code);
    throw new ApiError(code, ERROR_MESSAGES[code]);
  }

  try {
    const payload: unknown = await response.json();
    if (!isResumePayload(payload)) {
      throw new ApiError("invalid_response", ERROR_MESSAGES.invalid_response);
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("invalid_response", ERROR_MESSAGES.invalid_response);
  }
}

function mapStatusToCode(status: number): ApiErrorCode {
  if (status === 413) return "file_too_large";
  if (status === 422) return "unsupported_file_type";
  if (status >= 500) return "unexpected_server_error";
  return "unexpected_server_error";
}