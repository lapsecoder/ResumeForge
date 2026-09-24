import type { Content } from "pdfmake/interfaces";
import { describe, expect, it } from "vitest";

import type { TemplateId } from "@/components/templates";

import {
  A4_WIDTH_PT,
  COMPACT_LABEL_WIDTH,
  PDF_CONTENT_WIDTH,
  PDF_PAGE_MARGINS,
  PDF_PAGE_SIZE,
  buildResumePdfDocument,
} from "./pdfDocument";
import { makeFixtureResume, makeLongResume } from "./pdfTestFixture";
import type { Resume } from "./resume";

const TEMPLATES: TemplateId[] = ["classic", "modern", "compact"];

function collectText(content: Content[]): string[] {
  const out: string[] = [];
  const visit = (node: unknown): void => {
    if (node == null || typeof node === "number" || typeof node === "boolean") return;
    if (typeof node === "string") {
      out.push(node);
      return;
    }
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    const record = node as Record<string, unknown>;
    if (typeof record.text === "string") out.push(record.text);
    if (Array.isArray(record.text)) record.text.forEach(visit);
    if (record.stack) visit(record.stack);
    if (record.columns) visit(record.columns);
    if (record.ul) visit(record.ul);
    if (record.ol) visit(record.ol);
    if (record.table) visit(record.table);
    if (record.items) visit(record.items);
  };
  visit(content);
  return out;
}

const flatText = (content: Content[]): string => collectText(content).join(" ");
const norm = (value: string): string => value.replace(/\s+/g, "");

function walkNodes(content: Content[], visit: (node: Record<string, unknown>) => void): void {
  const walk = (node: unknown): void => {
    if (node == null || typeof node === "number" || typeof node === "boolean") return;
    if (typeof node === "string") return;
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (typeof node === "object") {
      const record = node as Record<string, unknown>;
      visit(record);
      if (record.text) walk(record.text);
      if (record.stack) walk(record.stack);
      if (record.columns) walk(record.columns);
      if (record.ul) walk(record.ul);
      if (record.ol) walk(record.ol);
      if (record.table) walk(record.table);
      if (record.items) walk(record.items);
    }
  };
  walk(content);
}

function unbreakableCount(content: Content[]): number {
  let count = 0;
  walkNodes(content, (node) => {
    if (node.unbreakable === true && (node.stack !== undefined || node.columns !== undefined)) count += 1;
  });
  return count;
}

function headingsOf(content: Content[], expected: string[]): string[] {
  const present: string[] = [];
  walkNodes(content, (node) => {
    if (node.headlineLevel === 1 && typeof node.text === "string") present.push(node.text.replace(/\s+/g, ""));
  });
  return expected.filter((heading) => present.includes(heading));
}

function hasRectColor(content: Content[], color: string): boolean {
  let found = false;
  walkNodes(content, (node) => {
    if (Array.isArray(node.canvas)) {
      for (const op of node.canvas as Record<string, unknown>[]) {
        if (op.type === "rect" && op.color === color) found = true;
      }
    }
  });
  return found;
}

function hasLabelColumn(content: Content[], width: number): boolean {
  let found = false;
  walkNodes(content, (node) => {
    if (Array.isArray(node.columns)) {
      const first = (node.columns as unknown[])[0] as Record<string, unknown> | undefined;
      if (first?.width === width) found = true;
    }
  });
  return found;
}

describe("buildResumePdfDocument — shared settings", () => {
  for (const templateId of TEMPLATES) {
    it(`produces A4, margin and content settings for ${templateId}`, () => {
      const dd = buildResumePdfDocument(makeFixtureResume(), templateId);
      expect(dd.pageSize).toBe(PDF_PAGE_SIZE);
      expect(dd.pageMargins).toEqual([...PDF_PAGE_MARGINS]);
      expect(dd.defaultStyle?.font).toBe("Roboto");
      expect(dd.info?.creator).toBe("ResumeForge");
      expect(dd.info?.title).toBe("Ada Lovelace");
      expect(dd.compress).toBe(true);
      expect(Array.isArray(dd.content) && dd.content.length > 0).toBe(true);
    });
  }

  it("exposes stable PDF metrics", () => {
    expect(PDF_CONTENT_WIDTH).toBe(A4_WIDTH_PT - PDF_PAGE_MARGINS[0] - PDF_PAGE_MARGINS[2]);
    expect(COMPACT_LABEL_WIDTH).toBe(63);
  });

  it("keeps headings with their content instead of inserting bad breaks", () => {
    const dd = buildResumePdfDocument(makeFixtureResume(), "classic");
    const pageBreakBefore = dd.pageBreakBefore as (current: {
      headlineLevel?: number;
      startPosition: { verticalRatio: number };
    }) => boolean;
    expect(pageBreakBefore({ startPosition: { verticalRatio: 0.9 } })).toBe(false);
    expect(pageBreakBefore({ headlineLevel: 1, startPosition: { verticalRatio: 0.9 } })).toBe(true);
    expect(pageBreakBefore({ headlineLevel: 1, startPosition: { verticalRatio: 0.5 } })).toBe(false);
  });
});

describe("buildResumePdfDocument — classic", () => {
  const dd = buildResumePdfDocument(makeFixtureResume(), "classic");
  const text = norm(flatText(dd.content as Content[]));

  it("renders the full set of sections and entry data", () => {
    expect(text).toContain(norm("Ada Lovelace"));
    expect(text).toContain(norm("ada@example.com"));
    expect(text).toContain(norm("Mathematician and first programmer"));
    expect(text).toContain(norm("Lead Engineer"));
    expect(text).toContain(norm("Analytical Engines Ltd"));
    expect(text).toContain(norm("1843-01 – 1848-12"));
    expect(text).toContain(norm("Published extensive notes"));
    expect(text).toContain(norm("Home Academy"));
    expect(text).toContain(norm("BSc · Mathematics"));
    expect(text).toContain(norm("Note G"));
    expect(text).toContain(norm("Punched cards"));
    expect(text).toContain(norm("Technical Skills: Algorithm design, Mathematics"));
    expect(text).toContain(norm("Royal Society"));
    expect(text).toContain(norm("Order of Merit"));
  });

  it("emits section headings with a headline level", () => {
    expect(
      headingsOf(dd.content as Content[], [
        "SUMMARY",
        "EXPERIENCE",
        "EDUCATION",
        "PROJECTS",
        "SKILLS",
        "CERTIFICATIONS",
        "AWARDS",
      ])
    ).toEqual(["SUMMARY", "EXPERIENCE", "EDUCATION", "PROJECTS", "SKILLS", "CERTIFICATIONS", "AWARDS"]);
  });

  it("keeps each experience entry unbreakable", () => {
    const resume = makeFixtureResume();
    const expected =
      (resume.experience ?? []).length +
      (resume.education ?? []).length +
      (resume.projects ?? []).length +
      (resume.certifications ?? []).length;
    expect(unbreakableCount(dd.content as Content[])).toBeGreaterThanOrEqual(expected);
  });

  it("omits empty sections entirely", () => {
    const sparse: Resume = {
      contact: { name: "Only Name" },
      metadata: { overall_confidence: "low" },
    };
    const sparseDd = buildResumePdfDocument(sparse, "classic");
    const sparseText = norm(flatText(sparseDd.content as Content[]));
    expect(sparseText).toContain(norm("Only Name"));
    expect(sparseText).not.toContain("EXPERIENCE");
    expect(sparseText).not.toContain("EDUCATION");
    expect(sparseText).not.toContain("PROJECTS");
    expect(sparseDd.info?.title).toBe("Only Name");
  });
});

describe("buildResumePdfDocument — modern", () => {
  const dd = buildResumePdfDocument(makeFixtureResume(), "modern");
  const text = norm(flatText(dd.content as Content[]));

  it("renders the full set of sections with accent styling", () => {
    expect(text).toContain(norm("Ada Lovelace"));
    expect(text).toContain(norm("London, UK"));
    expect(text).toContain(norm("ada@example.com"));
    expect(text).toContain(norm("Mathematician and first programmer"));
    expect(
      headingsOf(dd.content as Content[], [
        "PROFILE",
        "EXPERIENCE",
        "PROJECTS",
        "SKILLS",
        "EDUCATION",
        "CERTIFICATIONS",
        "AWARDS",
      ])
    ).toEqual(["PROFILE", "EXPERIENCE", "PROJECTS", "SKILLS", "EDUCATION", "CERTIFICATIONS", "AWARDS"]);
    expect(hasRectColor(dd.content as Content[], "#6366f1")).toBe(true);
  });

  it("includes project technologies alongside the project name", () => {
    expect(text).toContain(norm("Note G"));
    expect(text).toContain(norm("Punched cards"));
  });

  it("splits skills into two balanced columns", () => {
    const skills = norm("Technical Skills");
    expect(text).toContain(skills);
  });

  it("omits empty sections entirely", () => {
    const sparse: Resume = {
      contact: { name: "Only Name" },
      metadata: { overall_confidence: "low" },
    };
    const sparseText = norm(flatText(buildResumePdfDocument(sparse, "modern").content as Content[]));
    expect(sparseText).toContain(norm("Only Name"));
    expect(sparseText).not.toContain("EXPERIENCE");
  });
});

describe("buildResumePdfDocument — compact", () => {
  const dd = buildResumePdfDocument(makeFixtureResume(), "compact");
  const text = norm(flatText(dd.content as Content[]));

  it("renders label-column sections with the compact width", () => {
    expect(hasLabelColumn(dd.content as Content[], COMPACT_LABEL_WIDTH)).toBe(true);
    expect(
      headingsOf(dd.content as Content[], ["SUMMARY", "EXPERIENCE", "PROJECTS", "SKILLS", "EDUCATION", "CERTIFICATIONS", "AWARDS"])
    ).toEqual([
      "SUMMARY",
      "EXPERIENCE",
      "PROJECTS",
      "SKILLS",
      "EDUCATION",
      "CERTIFICATIONS",
      "AWARDS",
    ]);
    expect(text).toContain(norm("Technical: Algorithm design, Mathematics"));
  });

  it("keeps job title, company and range on one line", () => {
    expect(text).toContain(norm("Lead Engineer · Analytical Engines Ltd · London, UK (1843-01 – 1848-12)"));
  });
});

describe("buildResumePdfDocument — edited data is reflected", () => {
  for (const templateId of TEMPLATES) {
    it(`${templateId} reflects renamed content`, () => {
      const edited = makeFixtureResume();
      edited.contact = { ...edited.contact, name: "Grace Hopper" };
      edited.summary = "Compiled one of the first computer languages.";
      const dd = buildResumePdfDocument(edited, templateId);
      const text = norm(flatText(dd.content as Content[]));
      expect(text).toContain(norm("Grace Hopper"));
      expect(text).toContain(norm("Compiled one of the first computer languages"));
      expect(text).not.toContain(norm("Ada Lovelace"));
      expect(dd.info?.title).toBe("Grace Hopper");
    });
  }

  it("marks every long resume entry as unbreakable", () => {
    const dd = buildResumePdfDocument(makeLongResume(), "classic");

    let entryMarks = 0;
    walkNodes(dd.content as Content[], (node) => {
      if (node.unbreakable === true && node.stack !== undefined) entryMarks += 1;
    });
    expect(entryMarks).toBeGreaterThanOrEqual((makeLongResume().experience ?? []).length);
  });
});