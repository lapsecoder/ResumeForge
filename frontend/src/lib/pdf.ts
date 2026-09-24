/**
 * Client-side resume PDF export.
 *
 * Builds a structured pdfmake document definition from the canonical Resume
 * object and the selected template (see ./pdfDocument), then renders real,
 * selectable PDF text with pdfmake. No DOM cloning, no canvas screenshot,
 * no browser print dialog, and no server round-trip: pdfmake embeds a font
 * via its virtual file system and downloads the finished A4 document.
 */

import type { TemplateId } from "@/components/templates";
import type { TCreatedPdf, TDocumentDefinitions, TVirtualFileSystem } from "pdfmake/interfaces";

import { buildResumePdfDocument } from "./pdfDocument";
import type { Resume } from "./resume";

const WINDOWS_RESERVED = new Set([
  "con",
  "prn",
  "aux",
  "nul",
  "com1",
  "com2",
  "com3",
  "com4",
  "com5",
  "com6",
  "com7",
  "com8",
  "com9",
  "lpt1",
  "lpt2",
  "lpt3",
  "lpt4",
  "lpt5",
  "lpt6",
  "lpt7",
  "lpt8",
  "lpt9",
]);

const MAX_BASE_CHARS = 60;

/**
 * Reduce arbitrary text to a safe filename base: no path separators, no
 * traversal, no control characters, no reserved device names, bounded length.
 */
export function sanitizeFilenameBase(raw: string, fallback = "resume"): string {
  let value = (raw ?? "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  value = value.replace(/[^A-Za-z0-9 ._-]/g, "_");
  value = value.replace(/\.{2,}/g, ".");
  value = value.replace(/\s+/g, "_");
  value = value.replace(/_{2,}/g, "_");
  value = value.replace(/^[^A-Za-z0-9]+/, "");
  value = value.replace(/[^A-Za-z0-9]+$/, "");
  value = value.slice(0, MAX_BASE_CHARS).replace(/[^A-Za-z0-9]+$/, "");
  if (!value) return fallback;
  if (WINDOWS_RESERVED.has(value.toLowerCase())) return `_${value}`;
  return value;
}

/** Suggested export filename, e.g. "Ada_Lovelace_Resume.pdf". */
export function buildExportFilename(resume: Resume): string {
  const name = resume?.contact?.name?.trim();
  if (!name) return "resume.pdf";
  return `${sanitizeFilenameBase(name)}_Resume.pdf`;
}

interface PdfMakeStatic {
  createPdf(documentDefinitions: TDocumentDefinitions): TCreatedPdf;
  addVirtualFileSystem?: (vfs: TVirtualFileSystem) => void;
}

/**
 * Lazily load the pdfmake browser build and register its embedded Roboto
 * fonts so exported text renders fully client-side.
 */
export async function loadPdfMake(): Promise<PdfMakeStatic> {
  const pdfMakeModule = await import("pdfmake");
  const pdfMake = (pdfMakeModule as unknown as { default?: PdfMakeStatic }).default ??
    (pdfMakeModule as unknown as PdfMakeStatic);

  const vfsModule = await import("pdfmake/build/vfs_fonts");
  const vfs = (vfsModule as unknown as { default?: TVirtualFileSystem }).default ??
    (vfsModule as unknown as TVirtualFileSystem);

  if (vfs && typeof pdfMake.addVirtualFileSystem === "function") {
    pdfMake.addVirtualFileSystem(vfs);
  }
  return pdfMake;
}

/** Trigger a clean download for the generated blob. */
export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Generate a real text PDF from the resume data and start a download.
 * Resolves with the suggested filename on success.
 */
export async function exportResumePdf(resume: Resume, templateId: TemplateId): Promise<string> {
  const filename = buildExportFilename(resume);

  if (typeof document === "undefined" || typeof window === "undefined") {
    return filename;
  }

  const documentDefinitions = buildResumePdfDocument(resume, templateId);
  const pdfMake = await loadPdfMake();
  const pdfDoc = pdfMake.createPdf(documentDefinitions);
  const blob = await pdfDoc.getBlob();
  triggerDownload(blob, filename);
  return filename;
}