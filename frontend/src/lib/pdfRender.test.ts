// @vitest-environment node
/**
 * Renders real pdfmake PDFs (node build + embedded Roboto) and verifies the
 * output with Mozilla's pdf.js parser: A4 geometry, real selectable text, and
 * multi-page flow. No DOM, no canvas — this is the definitive proof that the
 * exported file is a genuine text PDF rather than a screenshot.
 */
import path from "node:path";

import pdfMake from "pdfmake";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { describe, expect, it } from "vitest";

import type { TDocumentDefinitions } from "pdfmake/interfaces";
import type { TemplateId } from "@/components/templates";

import { buildResumePdfDocument } from "./pdfDocument";
import { makeFixtureResume, makeLongResume } from "./pdfTestFixture";

const FONT_DIR = path.resolve("node_modules/pdfmake/build/fonts/Roboto");
pdfMake.fonts = {
  Roboto: {
    normal: path.join(FONT_DIR, "Roboto-Regular.ttf"),
    bold: path.join(FONT_DIR, "Roboto-Medium.ttf"),
    italics: path.join(FONT_DIR, "Roboto-Italic.ttf"),
    bolditalics: path.join(FONT_DIR, "Roboto-MediumItalic.ttf"),
  },
};

async function render(dd: TDocumentDefinitions): Promise<Uint8Array> {
  const buffer = await pdfMake.createPdf(dd).getBuffer();
  return new Uint8Array(buffer);
}

interface ParsedPdf {
  pageCount: number;
  pages: { width: number; height: number; text: string }[];
  fullText: string;
  raw: string;
}

async function parse(buffer: Uint8Array): Promise<ParsedPdf> {
  const doc = await getDocument({ data: buffer }).promise;
  const pages: ParsedPdf["pages"] = [];
  for (let index = 1; index <= doc.numPages; index += 1) {
    const page = await doc.getPage(index);
    const viewport = page.getViewport({ scale: 1 });
    const content = await page.getTextContent();
    const text = content.items
      .map((item) => ("str" in item ? String(item.str) : ""))
      .join("\n");
    pages.push({ width: viewport.width, height: viewport.height, text });
  }
  return {
    pageCount: doc.numPages,
    pages,
    fullText: pages.map((page) => page.text).join("\n"),
    raw: Buffer.from(buffer).toString("latin1"),
  };
}

const norm = (value: string): string => value.replace(/\s+/g, "");

async function parsedResume(templateId: TemplateId): Promise<ParsedPdf> {
  const buffer = await render(buildResumePdfDocument(makeFixtureResume(), templateId));
  return parse(buffer);
}

describe("pdfmake render — classic", () => {
  it("produces A4 pages with selectable text and no images", async () => {
    const pdf = await parsedResume("classic");

    for (const page of pdf.pages) {
      expect(page.width).toBeCloseTo(595.28, 1);
      expect(page.height).toBeCloseTo(841.89, 1);
    }
    expect(pdf.raw).not.toContain("/Subtype /Image");

    const text = norm(pdf.fullText);
    expect(text).toContain(norm("Ada Lovelace"));
    expect(text).toContain(norm("ada@example.com"));
    expect(text).toContain(norm("Mathematician and first programmer"));
    expect(text).toContain(norm("Lead Engineer"));
    expect(text).toContain(norm("Published extensive notes"));
    expect(text).toContain(norm("EXPERIENCE"));
    expect(text).toContain(norm("SKILLS"));
    expect(text).toContain(norm("Technical Skills"));
  });
});

describe("pdfmake render — modern", () => {
  it("renders all sections as real text", async () => {
    const pdf = await parsedResume("modern");

    for (const page of pdf.pages) {
      expect(page.width).toBeCloseTo(595.28, 1);
    }
    expect(pdf.raw).not.toContain("/Subtype /Image");

    const text = norm(pdf.fullText);
    expect(text).toContain(norm("Ada Lovelace"));
    expect(text).toContain(norm("PROFILE"));
    expect(text).toContain(norm("EXPERIENCE"));
    expect(text).toContain(norm("PROJECTS"));
    expect(text).toContain(norm("Note G"));
    expect(text).toContain(norm("Punched cards"));
  });
});

describe("pdfmake render — compact", () => {
  it("renders label-column sections as real text", async () => {
    const pdf = await parsedResume("compact");

    const text = norm(pdf.fullText);
    expect(text).toContain(norm("Ada Lovelace"));
    expect(text).toContain(norm("EXPERIENCE"));
    expect(text).toContain(norm("Lead Engineer · Analytical Engines Ltd · London, UK (1843-01 – 1848-12)"));
    expect(text).toContain(norm("Technical"));
  });
});

describe("pdfmake render — multi-page flow", () => {
  it("flows long content onto multiple A4 pages without clipping", async () => {
    const longResume = makeLongResume();
    const experience = longResume.experience ?? [];
    const buffer = await render(buildResumePdfDocument(longResume, "classic"));
    const pdf = await parse(buffer);

    expect(pdf.pageCount).toBeGreaterThan(1);
    for (const page of pdf.pages) {
      expect(page.width).toBeCloseTo(595.28, 1);
      expect(page.height).toBeCloseTo(841.89, 1);
    }
    expect(pdf.raw).not.toContain("/Subtype /Image");

    const text = norm(pdf.fullText);
    const lastRole = experience[experience.length - 1]?.title ?? "";
    expect(text).toContain(norm(lastRole));
    expect(pdf.pages[pdf.pages.length - 1].text.length).toBeGreaterThan(0);
  });
});