import { describe, expect, it } from "vitest";

import {
  ALLOWED_EXTENSIONS_LABEL,
  MAX_FILE_SIZE_BYTES,
  extensionOf,
  formatFileSize,
  isAllowedExtension,
  validateResumeFile,
} from "./upload";

function makeFile(name: string, contents: string | BlobPart[], type = ""): File {
  const parts = typeof contents === "string" ? [contents] : contents;
  return new File(parts, name, { type });
}

describe("extensionOf", () => {
  it("extracts a lowercased extension", () => {
    expect(extensionOf("Resume.PDF")).toBe("pdf");
    expect(extensionOf("resume.docx")).toBe("docx");
    expect(extensionOf("notes.txt")).toBe("txt");
  });

  it("returns null without an extension", () => {
    expect(extensionOf("resume")).toBeNull();
    expect(extensionOf("")).toBeNull();
  });
});

describe("isAllowedExtension", () => {
  it("accepts only pdf, docx, txt", () => {
    expect(isAllowedExtension("pdf")).toBe(true);
    expect(isAllowedExtension("docx")).toBe(true);
    expect(isAllowedExtension("txt")).toBe(true);
    expect(isAllowedExtension("exe")).toBe(false);
    expect(isAllowedExtension(null)).toBe(false);
  });
});

describe("formatFileSize", () => {
  it("formats bytes, KB and MB", () => {
    expect(formatFileSize(512)).toBe("512 B");
    expect(formatFileSize(2048)).toBe("2 KB");
    expect(formatFileSize(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("validateResumeFile", () => {
  it("accepts a valid PDF", () => {
    const result = validateResumeFile(makeFile("resume.pdf", "text", "application/pdf"));
    expect(result.ok).toBe(true);
  });

  it("accepts a valid DOCX", () => {
    const result = validateResumeFile(makeFile("resume.docx", "text"));
    expect(result.ok).toBe(true);
  });

  it("accepts a valid TXT", () => {
    const result = validateResumeFile(makeFile("resume.txt", "Hello"));
    expect(result.ok).toBe(true);
  });

  it("rejects an unsupported extension", () => {
    const result = validateResumeFile(makeFile("resume.exe", "text"));
    expect(result.ok).toBe(false);
    expect(result.message).toContain(ALLOWED_EXTENSIONS_LABEL);
  });

  it("rejects a file without an extension", () => {
    const result = validateResumeFile(makeFile("resume", "text"));
    expect(result.ok).toBe(false);
  });

  it("rejects an oversized file", () => {
    const big = new File([new Uint8Array(MAX_FILE_SIZE_BYTES + 1)], "big.pdf", {
      type: "application/pdf",
    });
    const result = validateResumeFile(big);
    expect(result.ok).toBe(false);
    expect(result.message).toContain("10 MB");
  });

  it("rejects an empty file", () => {
    const result = validateResumeFile(makeFile("empty.pdf", ""));
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/empty/i);
  });
});