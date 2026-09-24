import { describe, expect, it, vi } from "vitest";

import {
  addEntry,
  addEntryListItem,
  addSkill,
  MAX_FIELD_CHARS,
  moveEntry,
  normalizeResume,
  removeEntry,
  removeSkill,
  updateContactField,
  updateEntryField,
  updateEntryListItem,
  updateSummary,
} from "./builder";
import type { Resume } from "./resume";

function makeResume(): Resume {
  return {
    contact: { name: "Ada Lovelace", email: "ada@example.com" },
    summary: "First programmer.",
    experience: [
      {
        company: "Analytical Engines",
        title: "Engineer",
        achievements: ["Wrote notes"],
      },
    ],
    education: [],
    skills: { technical: ["Python"], soft: [], tools: [], languages: [] },
    projects: [{ name: "Engine", technologies: ["Brass"] }],
    certifications: [],
    custom_sections: [],
    metadata: { overall_confidence: "high", word_count: 0 },
  };
}

describe("builder security", () => {
  it("refuses prototype-polluting field names without touching Object.prototype", () => {
    const resume = makeResume();
    for (const field of ["__proto__", "constructor", "prototype", "achievements.constructor"]) {
      expect(updateEntryField(resume, "experience", 0, field, "polluted")).toBeNull();
    }
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    expect((Object.prototype as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("refuses invalid, fractional, and out-of-range indexes", () => {
    const resume = makeResume();
    expect(updateEntryField(resume, "experience", -1, "company", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", 1.5, "company", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", 50, "company", "x")).toBeNull();
    expect(updateEntryListItem(resume, "experience", 0, "achievements", 99, "x")).toBeNull();
    expect(removeEntry(resume, "experience", 50)).toBeNull();
    expect(moveEntry(resume, "experience", 0, -1)).toBeNull();
  });

  it("refuses oversized values before they reach state", () => {
    const resume = makeResume();
    const payload = "x".repeat(MAX_FIELD_CHARS + 1);
    expect(updateSummary(resume, payload)).toBeNull();
    expect(updateContactField(resume, "name", payload)).toBeNull();
    expect(addEntryListItem(resume, "experience", 0, "achievements", payload)).toBeNull();
    expect(addSkill(resume, "technical", payload)).toBeNull();
  });

  it("rejects writes to sections that have no such list field", () => {
    const resume = makeResume();
    expect(addEntryListItem(resume, "certifications", 0, "achievements")).toBeNull();
    expect(updateEntryListItem(resume, "education", 0, "content", 0, "x")).toBeNull();
    expect(removeSkill(resume, "technical", 99)).toBeNull();
  });

  it("stores hostile text as inert data and never mutates the input", () => {
    const resume = makeResume();
    const before = JSON.stringify(resume);
    const hostile = '<img src=x onerror="alert(1)"><script>alert(2)</script>';

    const next = updateSummary(resume, hostile);
    expect(next?.summary).toBe(hostile);
    expect(typeof next?.summary).toBe("string");

    const contact = updateContactField(next!, "name", "<b>Ada</b>");
    expect(contact?.contact?.name).toBe("<b>Ada</b>");

    expect(JSON.stringify(resume)).toBe(before);
  });

  it("strips control characters from text values", () => {
    const next = updateSummary(makeResume(), "hello\u0000\u0007world");
    expect(next?.summary).toBe("helloworld");
  });

  it("works against a tampered resume without throwing or writing globally", () => {
    const tampered = {
      ...makeResume(),
      experience: "not-an-array",
      skills: null,
    } as unknown as Resume;
    expect(() => normalizeResume(tampered)).not.toThrow();
    const normalized = normalizeResume(tampered);
    expect(Array.isArray(normalized.experience)).toBe(true);
    expect(Array.isArray(normalized.skills?.technical)).toBe(true);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("never touches browser storage or the network", () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const fetchSpy = vi.fn();
    const originalFetch = globalThis.fetch;
    Reflect.set(globalThis, "fetch", fetchSpy);
    try {
      const resume = makeResume();
      const added = addEntry(resume, "experience");
      const withSkill = addSkill(added!, "technical", "TypeScript");
      const withSummary = updateSummary(withSkill!, "Updated");
      const withItem = addEntryListItem(withSummary!, "experience", 0, "achievements", "New");
      expect(setItem).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
      expect(withItem).not.toBeNull();
    } finally {
      setItem.mockRestore();
      Reflect.set(globalThis, "fetch", originalFetch);
    }
  });
});
