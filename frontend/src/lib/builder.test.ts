import { describe, expect, it } from "vitest";

import { applyEdit } from "./edit";
import type { Resume } from "./resume";
import {
  addEntry,
  addEntryListItem,
  addSkill,
  emptyResume,
  isResumeEmpty,
  MAX_FIELD_CHARS,
  moveEntry,
  moveEntryListItem,
  moveSkill,
  normalizeResume,
  removeEntry,
  removeEntryListItem,
  removeSkill,
  updateContactField,
  updateEntryField,
  updateEntryListItem,
  updateSkill,
  updateSummary,
} from "./builder";

function makeResume(): Resume {
  return {
    contact: {
      name: "Ada Lovelace",
      email: "ada@example.com",
      location: "London",
      linkedin: "linkedin.com/in/ada",
    },
    summary: "First programmer.",
    experience: [
      {
        company: "Analytical Engines",
        title: "Engineer",
        location: "London",
        start_date: "1842",
        end_date: "1843",
        description: "Built engines",
        achievements: ["Wrote notes", "Drew diagrams"],
      },
      { company: "Second Co", title: "Analyst", achievements: [] },
    ],
    education: [{ institution: "Home Academy", degree: "BSc", field: "Math", details: ["Self taught"] }],
    skills: {
      technical: ["Python", "React"],
      soft: ["Writing"],
      tools: ["Git"],
      languages: ["English"],
      all: [],
    },
    projects: [{ name: "Engine", description: "An engine", technologies: ["Brass"] }],
    certifications: [{ name: "Pioneer", issuer: "Academy", date: "1843" }],
    custom_sections: [{ heading: "Awards", content: ["Best notes"] }],
    metadata: { overall_confidence: "high", word_count: 0 },
  };
}

describe("builder core operations", () => {
  it("updates contact fields and clears them when emptied", () => {
    const updated = updateContactField(makeResume(), "name", "Ada L.");
    expect(updated?.contact?.name).toBe("Ada L.");
    const cleared = updateContactField(updated!, "name", "");
    expect(cleared?.contact?.name).toBeNull();
  });

  it("updates and clears the summary", () => {
    const updated = updateSummary(makeResume(), "New summary");
    expect(updated?.summary).toBe("New summary");
    expect(updateSummary(updated!, "")?.summary).toBeNull();
  });

  it("updates an entry field without mutating the input", () => {
    const resume = makeResume();
    const updated = updateEntryField(resume, "experience", 0, "company", "New Co");
    expect(updated?.experience?.[0].company).toBe("New Co");
    expect(resume.experience?.[0].company).toBe("Analytical Engines");
  });

  it("rejects unknown fields, prototype keys, and out-of-range indexes", () => {
    const resume = makeResume();
    expect(updateEntryField(resume, "experience", 0, "salaries", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", 0, "__proto__", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", 0, "constructor", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", 9, "company", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", -1, "company", "x")).toBeNull();
    expect(updateEntryField(resume, "experience", 1.5, "company", "x")).toBeNull();
  });

  it("rejects over-long values", () => {
    const tooLong = "a".repeat(MAX_FIELD_CHARS + 1);
    expect(updateSummary(makeResume(), tooLong)).toBeNull();
    expect(updateEntryField(makeResume(), "experience", 0, "title", tooLong)).toBeNull();
  });

  it("adds, removes, and reorders entries", () => {
    const added = addEntry(makeResume(), "experience");
    expect(added?.experience).toHaveLength(3);
    expect(added?.experience?.[2].company).toBe("");

    const removed = removeEntry(added!, "experience", 0);
    expect(removed?.experience).toHaveLength(2);
    expect(removed?.experience?.[0].company).toBe("Second Co");

    const moved = moveEntry(added!, "experience", 2, -1);
    expect(moved?.experience?.[1].company).toBe("");
    expect(moveEntry(added!, "experience", 0, -1)).toBeNull();
    expect(moveEntry(added!, "experience", 2, 1)).toBeNull();
    expect(removeEntry(added!, "experience", 9)).toBeNull();
  });

  it("manages list items inside an entry", () => {
    const resume = makeResume();
    const added = addEntryListItem(resume, "experience", 0, "achievements", "New bullet");
    expect(added?.experience?.[0].achievements).toEqual(["Wrote notes", "Drew diagrams", "New bullet"]);

    const updated = updateEntryListItem(added!, "experience", 0, "achievements", 0, "Edited");
    expect(updated?.experience?.[0].achievements?.[0]).toBe("Edited");

    const moved = moveEntryListItem(updated!, "experience", 0, "achievements", 0, 1);
    expect(moved?.experience?.[0].achievements).toEqual(["Drew diagrams", "Edited", "New bullet"]);

    const removed = removeEntryListItem(moved!, "experience", 0, "achievements", 2);
    expect(removed?.experience?.[0].achievements).toEqual(["Drew diagrams", "Edited"]);

    expect(updateEntryListItem(resume, "experience", 0, "achievements", 9, "x")).toBeNull();
    expect(addEntryListItem(resume, "certifications", 0, "achievements")).toBeNull();
  });

  it("manages skills and keeps the flattened list in sync", () => {
    const added = addSkill(makeResume(), "technical", "TypeScript");
    expect(added?.skills?.technical).toEqual(["Python", "React", "TypeScript"]);
    expect(added?.skills?.all).toContain("TypeScript");

    expect(addSkill(added!, "technical", "python")).toBeNull();
    expect(addSkill(added!, "technical", "  ")).toBeNull();

    const updated = updateSkill(added!, "technical", 0, "Py");
    expect(updated?.skills?.technical?.[0]).toBe("Py");

    const moved = moveSkill(updated!, "technical", 0, 1);
    expect(moved?.skills?.technical).toEqual(["React", "Py", "TypeScript"]);

    const removed = removeSkill(moved!, "technical", 0);
    expect(removed?.skills?.technical).toEqual(["Py", "TypeScript"]);
    expect(removed?.skills?.all).not.toContain("React");

    expect(removeSkill(removed!, "technical", 9)).toBeNull();
    expect(updateSkill(removed!, "technical", 1.2, "x")).toBeNull();
  });

  it("normalizes derived fields and shape", () => {
    const normalized = normalizeResume(makeResume());
    expect(normalized.metadata.word_count).toBeGreaterThan(0);
    expect(normalized.skills?.all).toEqual(["Python", "React", "Git", "English", "Writing"]);
  });

  it("produces a valid empty resume", () => {
    const resume = emptyResume();
    expect(isResumeEmpty(resume)).toBe(true);
    expect(resume.metadata.overall_confidence).toBe("low");
    expect(addEntry(resume, "projects")?.projects).toHaveLength(1);
  });

  it("detects a non-empty resume", () => {
    expect(isResumeEmpty(makeResume())).toBe(false);
  });

  it("applies a Copilot edit into a builder-compatible working resume", () => {
    const edited = applyEdit(
      makeResume(),
      { path: "experience[0].title", section: "experience", index: 0, field: "title", sub_index: null },
      "Lead Engineer"
    );
    expect(edited).not.toBeNull();
    const working = normalizeResume(edited!);
    expect(working.experience?.[0].title).toBe("Lead Engineer");
    expect(working.metadata.word_count).toBeGreaterThan(0);
  });
});
