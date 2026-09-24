import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildExportFilename, sanitizeFilenameBase } from "./pdf";
import { makeFixtureResume } from "./pdfTestFixture";
import type { Resume } from "./resume";

function makeResume(name?: string | null): Resume {
  return {
    contact: name === undefined ? {} : { name },
    summary: "Summary",
    experience: [{ company: "Co", title: "Engineer" }],
    skills: { technical: ["Python"] },
    metadata: { overall_confidence: "high" },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  document.title = "";
  document.body.innerHTML = "";
});

describe("filename sanitization", () => {
  it("turns a display name into a safe base", () => {
    expect(sanitizeFilenameBase("Ada Lovelace")).toBe("Ada_Lovelace");
    expect(sanitizeFilenameBase("José Álvarez")).toBe("Jose_Alvarez");
  });

  it("neutralises path traversal and separators", () => {
    const base = sanitizeFilenameBase("../../etc/passwd");
    expect(base).not.toContain("/");
    expect(base).not.toContain("\\");
    expect(base).not.toContain("..");
    expect(base).toBe("etc_passwd");

    expect(sanitizeFilenameBase("C:\\Windows\\system32")).not.toContain(":");
  });

  it("avoids reserved Windows device names", () => {
    expect(sanitizeFilenameBase("CON")).toBe("_CON");
    expect(sanitizeFilenameBase("nul")).toBe("_nul");
  });

  it("falls back for empty input and bounds length", () => {
    expect(sanitizeFilenameBase("")).toBe("resume");
    expect(sanitizeFilenameBase("   ...   ")).toBe("resume");
    expect(sanitizeFilenameBase("a".repeat(200)).length).toBeLessThanOrEqual(60);
  });
});

describe("buildExportFilename", () => {
  it("derives a name from sanitized contact data", () => {
    expect(buildExportFilename(makeResume("Ada Lovelace"))).toBe("Ada_Lovelace_Resume.pdf");
  });

  it("falls back to resume.pdf without a name", () => {
    expect(buildExportFilename(makeResume(null))).toBe("resume.pdf");
    expect(buildExportFilename(makeResume(undefined))).toBe("resume.pdf");
  });

  it("never emits an arbitrary path", () => {
    const filename = buildExportFilename(makeResume("../../evil/x"));
    expect(filename).not.toContain("/");
    expect(filename.endsWith(".pdf")).toBe(true);
  });
});

const getBlobMock = vi.fn<() => Promise<Blob>>().mockResolvedValue(new Blob(["pdf-bytes"]));
const createPdfMock = vi.fn().mockReturnValue({ getBlob: getBlobMock });
const addVirtualFileSystemMock = vi.fn();

const pdfMakeMock = { createPdf: createPdfMock, addVirtualFileSystem: addVirtualFileSystemMock };

vi.mock("pdfmake", () => ({
  ...pdfMakeMock,
  default: pdfMakeMock,
}));

vi.mock("pdfmake/build/vfs_fonts", () => ({
  default: { "Roboto-Regular.ttf": "BASE64_FONT" },
}));

import { exportResumePdf } from "./pdf";

describe("exportResumePdf", () => {
  beforeEach(() => {
    createPdfMock.mockClear();
    getBlobMock.mockClear();
    addVirtualFileSystemMock.mockClear();
    getBlobMock.mockResolvedValue(new Blob(["pdf-bytes"]));
    createPdfMock.mockReturnValue({ getBlob: getBlobMock });
  });

  it("resolves the suggested filename and downloads a blob", async () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click");
    const filename = await exportResumePdf(makeFixtureResume(), "classic");

    expect(filename).toBe("Ada_Lovelace_Resume.pdf");
    expect(createPdfMock).toHaveBeenCalledTimes(1);
    expect(getBlobMock).toHaveBeenCalledTimes(1);
    expect(clickSpy).toHaveBeenCalledTimes(1);
    const anchor = clickSpy.mock.instances[0] as HTMLAnchorElement;
    expect(anchor.download).toBe("Ada_Lovelace_Resume.pdf");
    clickSpy.mockRestore();
  });

  it("builds an A4 document definition from template data", async () => {
    await exportResumePdf(makeFixtureResume(), "classic");

    const dd = createPdfMock.mock.calls[0][0];
    expect(dd.pageSize).toBe("A4");
    expect(dd.defaultStyle?.font).toBe("Roboto");
    expect(dd.info?.title).toBe("Ada Lovelace");
    expect(typeof createPdfMock.mock.calls[0][0].content).toBe("object");
  });

  it("registers the embedded font virtual file system", async () => {
    await exportResumePdf(makeFixtureResume(), "compact");
    expect(addVirtualFileSystemMock).toHaveBeenCalledTimes(1);
    expect((addVirtualFileSystemMock.mock.calls[0][0] as Record<string, string>)["Roboto-Regular.ttf"]).toBe(
      "BASE64_FONT"
    );
  });

  it("propagates render failures", async () => {
    getBlobMock.mockRejectedValueOnce(new Error("pdfmake render failed"));
    await expect(exportResumePdf(makeFixtureResume(), "modern")).rejects.toThrow("pdfmake render failed");
  });
});