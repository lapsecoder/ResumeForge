import { describe, expect, it } from "vitest";

import { emptyResume, MAX_FIELD_CHARS, MAX_LIST_ITEMS, WARN_FIELD_CHARS } from "./builder";
import { isSafeEmail, validateResume } from "./builderValidation";
import type { Resume } from "./resume";

function makeResume(): Resume {
  return {
    contact: { name: "Ada Lovelace", email: "ada@example.com", linkedin: "linkedin.com/in/ada" },
    summary: "First programmer.",
    experience: [{ company: "Analytical Engines", title: "Engineer", achievements: ["Wrote notes"] }],
    education: [{ institution: "Home Academy", degree: "BSc" }],
    skills: { technical: ["Python"], soft: [], tools: [], languages: [] },
    projects: [{ name: "Engine", description: "An engine", technologies: ["Brass"] }],
    certifications: [{ name: "Pioneer", issuer: "Academy" }],
    custom_sections: [{ heading: "Awards", content: ["Best notes"] }],
    metadata: { overall_confidence: "high", word_count: 12 },
  };
}

function asResume(value: unknown): Resume {
  return value as unknown as Resume;
}

describe("validateResume", () => {
  it("accepts a well-formed resume", () => {
    const { errors, warnings } = validateResume(makeResume());
    expect(errors).toEqual([]);
    expect(warnings).toEqual([]);
  });

  it("accepts a sparse resume with optional fields absent", () => {
    const sparse = asResume({ contact: { name: "Only Name" }, metadata: { overall_confidence: "low" } });
    const { errors } = validateResume(sparse);
    expect(errors).toEqual([]);
  });

  it("reports an empty resume", () => {
    const { errors } = validateResume(emptyResume());
    expect(errors.some((issue) => /empty/i.test(issue.message))).toBe(true);
  });

  it("reports an invalid email", () => {
    const resume = makeResume();
    resume.contact = { ...resume.contact, email: "not-an-email" };
    const { errors } = validateResume(resume);
    expect(errors.some((issue) => issue.path === "contact.email")).toBe(true);
  });

  it("reports unsafe link schemes", () => {
    const resume = makeResume();
    resume.contact = { ...resume.contact, website: "javascript:alert(1)" };
    const { errors } = validateResume(resume);
    expect(errors.some((issue) => issue.path === "contact.website")).toBe(true);
  });

  it("reports over-long fields and warns on long fields", () => {
    const tooLong = makeResume();
    tooLong.summary = "a".repeat(MAX_FIELD_CHARS + 1);
    expect(validateResume(tooLong).errors.some((issue) => issue.path === "summary")).toBe(true);

    const long = makeResume();
    long.summary = "a".repeat(WARN_FIELD_CHARS + 1);
    const result = validateResume(long);
    expect(result.errors).toEqual([]);
    expect(result.warnings.some((issue) => issue.path === "summary")).toBe(true);
  });

  it("reports malformed sections and entries", () => {
    const notArray = asResume({ ...makeResume(), experience: { company: "x" } });
    expect(validateResume(notArray).errors.some((issue) => issue.path === "experience")).toBe(true);

    const badEntry = asResume({ ...makeResume(), experience: [null] });
    expect(validateResume(badEntry).errors.some((issue) => issue.path === "experience[0]")).toBe(true);
  });

  it("warns when a required-ish field is blank but does not block", () => {
    const resume = makeResume();
    resume.experience = [{ company: "", title: "" }];
    const { errors, warnings } = validateResume(resume);
    expect(errors).toEqual([]);
    expect(warnings.some((issue) => issue.path === "experience[0].company")).toBe(true);
  });

  it("reports wrong field types", () => {
    const resume = asResume({ ...makeResume(), summary: 123, skills: { technical: "Python" } });
    const { errors } = validateResume(resume);
    expect(errors.some((issue) => issue.path === "summary")).toBe(true);
    expect(errors.some((issue) => issue.path === "skills.technical")).toBe(true);
  });

  it("reports too many list items", () => {
    const resume = makeResume();
    resume.skills = { technical: Array.from({ length: MAX_LIST_ITEMS + 1 }, (_, i) => `s${i}`) };
    const { errors } = validateResume(resume);
    expect(errors.some((issue) => issue.path === "skills.technical")).toBe(true);
  });

  it("validates email shape helper", () => {
    expect(isSafeEmail("a@b.co")).toBe(true);
    expect(isSafeEmail("a@b")).toBe(false);
    expect(isSafeEmail("a b@c.co")).toBe(false);
  });
});
