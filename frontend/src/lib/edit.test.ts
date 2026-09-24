import { describe, expect, it } from "vitest";

import {
  MAX_EDIT_VALUE_CHARS,
  applyEdit,
  editTargetLabel,
  isEditableTarget,
  parseEditPath,
} from "./edit";
import type { CopilotEditTarget } from "./copilot";
import type { Resume } from "./resume";

const RESUME: Resume = {
  contact: { name: "Ada Lovelace", email: "ada@example.com" },
  summary: "Analyst with a background in mathematics.",
  experience: [
    {
      company: "Analytical Engines",
      title: "Analyst",
      description: "Built analytical tooling.",
      achievements: ["Built a billing service.", "Reduced costs."],
    },
  ],
  projects: [
    {
      name: "Copilot",
      description: "Local resume copilot.",
      technologies: ["Python", "Postgres"],
    },
  ],
  skills: { technical: ["Python"] },
  metadata: { overall_confidence: "high", section_confidence: [] },
};

const ACCEPTED = [
  "summary",
  "experience[0].title",
  "experience[0].company",
  "experience[0].description",
  "experience[0].achievements[1]",
  "experience[0].bullets[0]",
  "projects[0].description",
  "projects[0].technologies",
];

const REJECTED = [
  "",
  "summary.x",
  "summary[0]",
  "experience",
  "experience[0]",
  "experience.title",
  "experience[0].name",
  "experience[0].start_date",
  "experience[0].achievements",
  "experience[0].achievements[x]",
  "education[0].institution",
  "certifications[0].name",
  "skills.technical[0]",
  "contact.email",
  "contact.phone",
  "contact.github",
  "metadata.word_count",
  "projects[0].name",
  "projects[0].technologies[0]",
  "custom_sections[0].content[0]",
  "../../etc/passwd",
  "summary; rm -rf /",
];

describe("parseEditPath", () => {
  it.each(ACCEPTED)("accepts %s", (path) => {
    const target = parseEditPath(path);
    expect(target).not.toBeNull();
    expect(isEditableTarget(target)).toBe(true);
  });

  it.each(REJECTED)("rejects %s", (path) => {
    expect(parseEditPath(path)).toBeNull();
  });

  it("canonicalises the bullets alias", () => {
    const target = parseEditPath("experience[0].bullets[1]");
    expect(target).toMatchObject({
      path: "experience[0].achievements[1]",
      section: "experience",
      field: "achievements",
      sub_index: 1,
    });
  });
});

describe("isEditableTarget", () => {
  it("rejects a tampered section", () => {
    const tampered: CopilotEditTarget = {
      path: "experience[0].title",
      section: "contact",
      index: 0,
      field: "email",
      sub_index: null,
    };
    expect(isEditableTarget(tampered)).toBe(false);
  });

  it("rejects a mismatched index", () => {
    const tampered: CopilotEditTarget = {
      path: "experience[0].title",
      section: "experience",
      index: 3,
      field: "title",
      sub_index: null,
    };
    expect(isEditableTarget(tampered)).toBe(false);
  });
});

describe("applyEdit", () => {
  it("applies a summary edit without mutating the input", () => {
    const target = parseEditPath("summary") as CopilotEditTarget;
    const updated = applyEdit(RESUME, target, "Mathematician and engineer.");
    expect(updated?.summary).toBe("Mathematician and engineer.");
    expect(RESUME.summary).toBe("Analyst with a background in mathematics.");
  });

  it("applies a bullet edit by index", () => {
    const target = parseEditPath("experience[0].achievements[1]") as CopilotEditTarget;
    const updated = applyEdit(RESUME, target, "Reduced costs by 30%.");
    expect(updated?.experience?.[0].achievements?.[1]).toBe("Reduced costs by 30%.");
    expect(updated?.experience?.[0].achievements?.[0]).toBe("Built a billing service.");
  });

  it("splits technologies on commas", () => {
    const target = parseEditPath("projects[0].technologies") as CopilotEditTarget;
    const updated = applyEdit(RESUME, target, "Rust, Go, Elixir");
    expect(updated?.projects?.[0].technologies).toEqual(["Rust", "Go", "Elixir"]);
  });

  it("applies a project description edit", () => {
    const target = parseEditPath("projects[0].description") as CopilotEditTarget;
    const updated = applyEdit(RESUME, target, "A local, private resume copilot.");
    expect(updated?.projects?.[0].description).toBe("A local, private resume copilot.");
  });

  it("refuses out-of-range indices", () => {
    const target = parseEditPath("experience[9].title") as CopilotEditTarget;
    expect(applyEdit(RESUME, target, "x")).toBeNull();
    const bullet = parseEditPath("experience[0].achievements[9]") as CopilotEditTarget;
    expect(applyEdit(RESUME, bullet, "x")).toBeNull();
  });

  it("refuses non-allowlisted targets", () => {
    const target = {
      path: "contact.email",
      section: "contact",
      index: null,
      field: "email",
      sub_index: null,
    } as unknown as CopilotEditTarget;
    expect(applyEdit(RESUME, target, "attacker@evil.example")).toBeNull();
  });

  it("refuses an oversized value", () => {
    const target = parseEditPath("summary") as CopilotEditTarget;
    const oversized = "x".repeat(MAX_EDIT_VALUE_CHARS + 1);
    expect(applyEdit(RESUME, target, oversized)).toBeNull();
  });
});

describe("editTargetLabel", () => {
  it("describes allowlisted targets", () => {
    expect(editTargetLabel(parseEditPath("summary"))).toBe("Professional summary");
    expect(editTargetLabel(parseEditPath("experience[0].title"))).toBe(
      "Experience 1 · job title"
    );
    expect(editTargetLabel(parseEditPath("experience[0].achievements[1]"))).toBe(
      "Experience 1 · achievement 2"
    );
    expect(editTargetLabel(parseEditPath("projects[0].technologies"))).toBe(
      "Project 1 · technologies"
    );
  });
});
