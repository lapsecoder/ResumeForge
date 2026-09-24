/**
 * Upload validation for resume files.
 *
 * Client-side validation is UX-only and MUST NOT be treated as a security
 * boundary — the backend re-validates every upload.
 */

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
export const MAX_FILE_SIZE_MB = 10;
export const ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"] as const;
export const ALLOWED_EXTENSIONS_LABEL = "PDF, DOCX, TXT";
export const ACCEPT_ATTR = ".pdf,.docx,.txt";

export type AllowedExtension = (typeof ALLOWED_EXTENSIONS)[number];

export interface FileValidationResult {
  ok: boolean;
  message?: string;
}

export function extensionOf(filename: string): string | null {
  const match = /\.([^.]+)$/.exec(filename.trim());
  return match ? match[1].toLowerCase() : null;
}

export function isAllowedExtension(ext: string | null): ext is AllowedExtension {
  return ext !== null && (ALLOWED_EXTENSIONS as readonly string[]).includes(ext);
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function validateResumeFile(file: File): FileValidationResult {
  if (file.size === 0) {
    return { ok: false, message: "The selected file is empty. Please choose a different file." };
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      ok: false,
      message: `The file is larger than ${MAX_FILE_SIZE_MB} MB. Please choose a smaller file.`,
    };
  }
  const ext = extensionOf(file.name);
  if (!isAllowedExtension(ext)) {
    return {
      ok: false,
      message: `Unsupported file type. Please upload a ${ALLOWED_EXTENSIONS_LABEL} file.`,
    };
  }
  return { ok: true };
}