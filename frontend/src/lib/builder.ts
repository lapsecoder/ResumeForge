/**
 * Phase 7D — safe, in-memory resume builder operations.
 *
 * Every function is pure: it returns a NEW resume object (or null when the
 * request is not allowed) and never mutates its input. Mutation is restricted
 * to an explicit allowlist of sections/fields and validated integer indexes,
 * mirroring the Phase 7C edit-target philosophy — there is no generic
 * object-path or prototype write surface here.
 *
 * Nothing in this module persists or transmits resume content.
 */

import type {
  ContactInfo,
  Resume,
  SkillSet,
} from "./resume";

/** Hard bound on a single editable text value (mirrors the Copilot bound). */
export const MAX_FIELD_CHARS = 5000;
/** Soft bound: warn above this so the user can tighten wording. */
export const WARN_FIELD_CHARS = 1000;
/** Hard bound on the number of entries/items in any one list. */
export const MAX_LIST_ITEMS = 200;

export type ArraySectionKey =
  | "experience"
  | "education"
  | "projects"
  | "certifications"
  | "custom_sections";

export type SkillGroupKey = "technical" | "soft" | "tools" | "languages";

export type ContactFieldKey =
  | "name"
  | "email"
  | "phone"
  | "location"
  | "linkedin"
  | "github"
  | "website";

export const ARRAY_SECTION_KEYS: readonly ArraySectionKey[] = [
  "experience",
  "education",
  "projects",
  "certifications",
  "custom_sections",
];

export const SKILL_GROUP_KEYS: readonly SkillGroupKey[] = [
  "technical",
  "soft",
  "tools",
  "languages",
];

export const CONTACT_FIELD_KEYS: readonly ContactFieldKey[] = [
  "name",
  "email",
  "phone",
  "location",
  "linkedin",
  "github",
  "website",
];

const ENTRY_SCALAR_FIELDS: Record<ArraySectionKey, readonly string[]> = {
  experience: ["company", "title", "location", "start_date", "end_date", "description"],
  education: ["institution", "degree", "field", "location", "start_date", "end_date"],
  projects: ["name", "description", "url"],
  certifications: ["name", "issuer", "date", "url"],
  custom_sections: ["heading"],
};

const ENTRY_LIST_FIELDS: Record<ArraySectionKey, readonly string[]> = {
  experience: ["achievements", "skills_mentioned"],
  education: ["details"],
  projects: ["technologies"],
  certifications: [],
  custom_sections: ["content"],
};

const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);

export const SECTION_LABELS: Record<ArraySectionKey, string> = {
  experience: "Experience",
  education: "Education",
  projects: "Projects",
  certifications: "Certifications",
  custom_sections: "Custom section",
};

export const SKILL_GROUP_LABELS: Record<SkillGroupKey, string> = {
  technical: "Technical skills",
  soft: "Soft skills",
  tools: "Tools",
  languages: "Languages",
};

export const CONTACT_FIELD_LABELS: Record<ContactFieldKey, string> = {
  name: "Full name",
  email: "Email",
  phone: "Phone",
  location: "Location",
  linkedin: "LinkedIn",
  github: "GitHub",
  website: "Website",
};

export const EMPTY_RESUME: Resume = {
  contact: {},
  summary: null,
  experience: [],
  education: [],
  skills: { technical: [], soft: [], tools: [], languages: [], all: [] },
  projects: [],
  certifications: [],
  custom_sections: [],
  metadata: { word_count: 0, overall_confidence: "low" },
};

export function emptyResume(): Resume {
  return cloneResume(EMPTY_RESUME);
}

export function cloneResume(resume: Resume): Resume {
  return JSON.parse(JSON.stringify(resume)) as Resume;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeText(value: unknown): value is string {
  return typeof value === "string" && value.length <= MAX_FIELD_CHARS;
}

function sanitizeText(value: string): string {
  // Strip NUL and other C0 control characters except tab/newline.
  return value.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "");
}

function sectionArray(resume: Resume, section: ArraySectionKey): unknown[] {
  const value = resume[section];
  return Array.isArray(value) ? (value as unknown[]) : [];
}

function writeSection(resume: Resume, section: ArraySectionKey, items: unknown[]): void {
  (resume as unknown as Record<string, unknown>)[section] = items;
}

function entryAt(
  resume: Resume,
  section: ArraySectionKey,
  index: number
): Record<string, unknown> | null {
  if (!Number.isInteger(index) || index < 0) return null;
  const items = sectionArray(resume, section);
  if (index >= items.length) return null;
  const entry = items[index];
  return isRecord(entry) ? entry : null;
}

function isAllowedScalarField(section: ArraySectionKey, field: string): boolean {
  return ENTRY_SCALAR_FIELDS[section].includes(field);
}

function isAllowedListField(section: ArraySectionKey, field: string): boolean {
  return ENTRY_LIST_FIELDS[section].includes(field);
}

function isSafeKey(field: string): boolean {
  return !FORBIDDEN_KEYS.has(field);
}

function flattenSkills(skills: SkillSet): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const groups = [skills.technical, skills.tools, skills.languages, skills.soft];
  for (const group of groups) {
    if (!Array.isArray(group)) continue;
    for (const item of group) {
      if (typeof item !== "string") continue;
      const key = item.trim();
      const dedupe = key.toLowerCase();
      if (key && !seen.has(dedupe)) {
        seen.add(dedupe);
        out.push(key);
      }
    }
  }
  return out;
}

function countWords(resume: Resume): number {
  const parts: string[] = [];
  const pushText = (value: unknown) => {
    if (typeof value === "string" && value.trim()) parts.push(value);
  };
  const pushList = (value: unknown) => {
    if (Array.isArray(value)) value.forEach(pushText);
  };

  pushText(resume.summary);
  for (const entry of sectionArray(resume, "experience")) {
    if (!isRecord(entry)) continue;
    pushText(entry.title);
    pushText(entry.company);
    pushText(entry.description);
    pushList(entry.achievements);
  }
  for (const entry of sectionArray(resume, "education")) {
    if (!isRecord(entry)) continue;
    pushText(entry.institution);
    pushText(entry.degree);
    pushText(entry.field);
    pushList(entry.details);
  }
  for (const entry of sectionArray(resume, "projects")) {
    if (!isRecord(entry)) continue;
    pushText(entry.name);
    pushText(entry.description);
    pushList(entry.technologies);
  }
  for (const entry of sectionArray(resume, "certifications")) {
    if (!isRecord(entry)) continue;
    pushText(entry.name);
    pushText(entry.issuer);
  }
  for (const entry of sectionArray(resume, "custom_sections")) {
    if (!isRecord(entry)) continue;
    pushText(entry.heading);
    pushList(entry.content);
  }
  pushList(resume.skills?.all);
  return parts.join(" ").trim().split(/\s+/).filter(Boolean).length;
}

/**
 * Recompute derived fields (flattened skills, word count) and guarantee that
 * optional collections exist. Never invents user-authored content.
 */
export function normalizeResume(resume: Resume): Resume {
  const next = cloneResume(resume ?? emptyResume());
  if (!isRecord(next.contact)) next.contact = {};
  if (!isRecord(next.skills)) {
    next.skills = { technical: [], soft: [], tools: [], languages: [], all: [] };
  }
  for (const section of ARRAY_SECTION_KEYS) {
    const value = next[section];
    if (!Array.isArray(value)) writeSection(next, section, []);
  }
  for (const group of SKILL_GROUP_KEYS) {
    const value = next.skills[group];
    if (!Array.isArray(value)) next.skills[group] = [];
  }
  next.skills = { ...next.skills, all: flattenSkills(next.skills) };

  const metadata: Record<string, unknown> = isRecord(next.metadata) ? next.metadata : {};
  const confidence = metadata.overall_confidence;
  next.metadata = {
    word_count: countWords(next),
    file_type: typeof metadata.file_type === "string" ? metadata.file_type : undefined,
    overall_confidence:
      confidence === "high" || confidence === "medium" || confidence === "low"
        ? confidence
        : "low",
    section_confidence: Array.isArray(metadata.section_confidence)
      ? (metadata.section_confidence as Resume["metadata"]["section_confidence"])
      : undefined,
  };
  return next;
}

export function isResumeEmpty(resume: Resume): boolean {
  const contact = resume.contact ?? {};
  const hasContact = CONTACT_FIELD_KEYS.some((field) => {
    const value = contact[field];
    return typeof value === "string" && value.trim().length > 0;
  });
  if (hasContact) return false;
  if (typeof resume.summary === "string" && resume.summary.trim()) return false;
  for (const section of ARRAY_SECTION_KEYS) {
    if (sectionArray(resume, section).length > 0) return false;
  }
  for (const group of SKILL_GROUP_KEYS) {
    const value = resume.skills?.[group];
    if (Array.isArray(value) && value.some((item) => typeof item === "string" && item.trim())) {
      return false;
    }
  }
  return true;
}

function setScalar(target: Record<string, unknown>, field: string, value: string): void {
  target[field] = sanitizeText(value);
}

/** Update one contact field. Empty strings clear the field. */
export function updateContactField(
  resume: Resume,
  field: ContactFieldKey,
  value: string
): Resume | null {
  if (!CONTACT_FIELD_KEYS.includes(field)) return null;
  if (!isSafeText(value)) return null;
  const next = normalizeResume(resume);
  const contact: Record<string, unknown> = { ...(next.contact ?? {}) };
  const cleaned = sanitizeText(value);
  contact[field] = cleaned.trim() === "" ? null : cleaned;
  next.contact = contact as ContactInfo;
  return normalizeResume(next);
}

/** Update the professional summary. Empty string clears it. */
export function updateSummary(resume: Resume, value: string): Resume | null {
  if (!isSafeText(value)) return null;
  const next = normalizeResume(resume);
  const cleaned = sanitizeText(value);
  next.summary = cleaned.trim() === "" ? null : cleaned;
  return normalizeResume(next);
}

/** Update a scalar field on one entry (e.g. experience[0].company). */
export function updateEntryField(
  resume: Resume,
  section: ArraySectionKey,
  index: number,
  field: string,
  value: string
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!isSafeKey(field) || !isAllowedScalarField(section, field)) return null;
  if (!isSafeText(value)) return null;
  const next = cloneResume(resume);
  const entry = entryAt(next, section, index);
  if (!entry) return null;
  setScalar(entry, field, value);
  return normalizeResume(next);
}

/** Update one item inside an entry list (e.g. experience[0].achievements[1]). */
export function updateEntryListItem(
  resume: Resume,
  section: ArraySectionKey,
  index: number,
  field: string,
  itemIndex: number,
  value: string
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!isSafeKey(field) || !isAllowedListField(section, field)) return null;
  if (!isSafeText(value)) return null;
  if (!Number.isInteger(itemIndex) || itemIndex < 0) return null;
  const next = cloneResume(resume);
  const entry = entryAt(next, section, index);
  if (!entry) return null;
  const list = entry[field];
  if (!Array.isArray(list) || itemIndex >= list.length) return null;
  list[itemIndex] = sanitizeText(value);
  return normalizeResume(next);
}

/** Append an item to an entry list. */
export function addEntryListItem(
  resume: Resume,
  section: ArraySectionKey,
  index: number,
  field: string,
  value = ""
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!isSafeKey(field) || !isAllowedListField(section, field)) return null;
  if (!isSafeText(value)) return null;
  const next = cloneResume(resume);
  const entry = entryAt(next, section, index);
  if (!entry) return null;
  const list = Array.isArray(entry[field]) ? (entry[field] as unknown[]) : [];
  if (list.length >= MAX_LIST_ITEMS) return null;
  list.push(sanitizeText(value));
  entry[field] = list;
  return normalizeResume(next);
}

export function removeEntryListItem(
  resume: Resume,
  section: ArraySectionKey,
  index: number,
  field: string,
  itemIndex: number
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!isSafeKey(field) || !isAllowedListField(section, field)) return null;
  if (!Number.isInteger(itemIndex) || itemIndex < 0) return null;
  const next = cloneResume(resume);
  const entry = entryAt(next, section, index);
  if (!entry) return null;
  const list = entry[field];
  if (!Array.isArray(list) || itemIndex >= list.length) return null;
  const copy = list.slice();
  copy.splice(itemIndex, 1);
  entry[field] = copy;
  return normalizeResume(next);
}

export function moveEntryListItem(
  resume: Resume,
  section: ArraySectionKey,
  index: number,
  field: string,
  itemIndex: number,
  direction: -1 | 1
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!isSafeKey(field) || !isAllowedListField(section, field)) return null;
  if (!Number.isInteger(itemIndex) || itemIndex < 0) return null;
  const next = cloneResume(resume);
  const entry = entryAt(next, section, index);
  if (!entry) return null;
  const list = entry[field];
  if (!Array.isArray(list)) return null;
  const target = itemIndex + direction;
  if (target < 0 || target >= list.length) return null;
  const copy = list.slice();
  const [moved] = copy.splice(itemIndex, 1);
  copy.splice(target, 0, moved);
  entry[field] = copy;
  return normalizeResume(next);
}

function createEntry(section: ArraySectionKey): Record<string, unknown> {
  switch (section) {
    case "experience":
      return { company: "", title: "", description: "", achievements: [] };
    case "education":
      return { institution: "", degree: "", details: [] };
    case "projects":
      return { name: "", description: "", technologies: [] };
    case "certifications":
      return { name: "", issuer: "" };
    case "custom_sections":
      return { heading: "", content: [] };
  }
}

export function addEntry(resume: Resume, section: ArraySectionKey): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  const next = cloneResume(resume);
  const items = sectionArray(next, section);
  if (items.length >= MAX_LIST_ITEMS) return null;
  writeSection(next, section, [...items, createEntry(section)]);
  return normalizeResume(next);
}

export function removeEntry(
  resume: Resume,
  section: ArraySectionKey,
  index: number
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!Number.isInteger(index) || index < 0) return null;
  const next = cloneResume(resume);
  const items = sectionArray(next, section);
  if (index >= items.length) return null;
  const copy = items.slice();
  copy.splice(index, 1);
  writeSection(next, section, copy);
  return normalizeResume(next);
}

export function moveEntry(
  resume: Resume,
  section: ArraySectionKey,
  index: number,
  direction: -1 | 1
): Resume | null {
  if (!ARRAY_SECTION_KEYS.includes(section)) return null;
  if (!Number.isInteger(index) || index < 0) return null;
  const next = cloneResume(resume);
  const items = sectionArray(next, section);
  const target = index + direction;
  if (target < 0 || target >= items.length) return null;
  const copy = items.slice();
  const [moved] = copy.splice(index, 1);
  copy.splice(target, 0, moved);
  writeSection(next, section, copy);
  return normalizeResume(next);
}

function skillList(resume: Resume, group: SkillGroupKey): string[] {
  const value = resume.skills?.[group];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function writeSkills(resume: Resume, group: SkillGroupKey, items: string[]): Resume {
  const skills: SkillSet = { ...(resume.skills ?? {}), [group]: items };
  resume.skills = skills;
  return resume;
}

export function addSkill(
  resume: Resume,
  group: SkillGroupKey,
  value: string
): Resume | null {
  if (!SKILL_GROUP_KEYS.includes(group)) return null;
  if (!isSafeText(value)) return null;
  const cleaned = sanitizeText(value).trim();
  if (!cleaned) return null;
  const next = cloneResume(resume);
  const items = skillList(next, group);
  if (items.length >= MAX_LIST_ITEMS) return null;
  if (items.some((item) => item.toLowerCase() === cleaned.toLowerCase())) return null;
  writeSkills(next, group, [...items, cleaned]);
  return normalizeResume(next);
}

export function updateSkill(
  resume: Resume,
  group: SkillGroupKey,
  index: number,
  value: string
): Resume | null {
  if (!SKILL_GROUP_KEYS.includes(group)) return null;
  if (!isSafeText(value)) return null;
  if (!Number.isInteger(index) || index < 0) return null;
  const next = cloneResume(resume);
  const items = skillList(next, group);
  if (index >= items.length) return null;
  const copy = items.slice();
  copy[index] = sanitizeText(value).trim();
  writeSkills(next, group, copy);
  return normalizeResume(next);
}

export function removeSkill(
  resume: Resume,
  group: SkillGroupKey,
  index: number
): Resume | null {
  if (!SKILL_GROUP_KEYS.includes(group)) return null;
  if (!Number.isInteger(index) || index < 0) return null;
  const next = cloneResume(resume);
  const items = skillList(next, group);
  if (index >= items.length) return null;
  const copy = items.slice();
  copy.splice(index, 1);
  writeSkills(next, group, copy);
  return normalizeResume(next);
}

export function moveSkill(
  resume: Resume,
  group: SkillGroupKey,
  index: number,
  direction: -1 | 1
): Resume | null {
  if (!SKILL_GROUP_KEYS.includes(group)) return null;
  if (!Number.isInteger(index) || index < 0) return null;
  const next = cloneResume(resume);
  const items = skillList(next, group);
  const target = index + direction;
  if (target < 0 || target >= items.length) return null;
  const copy = items.slice();
  const [moved] = copy.splice(index, 1);
  copy.splice(target, 0, moved);
  writeSkills(next, group, copy);
  return normalizeResume(next);
}
