/**
 * Phase 7D — pre-export validation for the in-memory working resume.
 *
 * Validation is defensive: it treats the working object as untrusted and
 * checks structure, field types, lengths, and a few semantic rules (email
 * shape, unsafe URL schemes). Optional fields are never required, so a sparse
 * but valid resume is not blocked. `errors` block export; `warnings` do not.
 */

import {
  ARRAY_SECTION_KEYS,
  CONTACT_FIELD_KEYS,
  CONTACT_FIELD_LABELS,
  MAX_FIELD_CHARS,
  MAX_LIST_ITEMS,
  SECTION_LABELS,
  SKILL_GROUP_KEYS,
  SKILL_GROUP_LABELS,
  WARN_FIELD_CHARS,
  isResumeEmpty,
  type ArraySectionKey,
} from "./builder";
import type { Resume } from "./resume";

export interface ValidationIssue {
  path: string;
  message: string;
}

export interface ResumeValidation {
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const UNSAFE_SCHEME_RE = /^\s*(javascript|data|vbscript|file):/i;
const URL_FIELDS: readonly string[] = ["linkedin", "github", "website", "url"];
const CONFIDENCE_LEVELS = new Set(["high", "medium", "low"]);

export function isSafeEmail(value: string): boolean {
  return EMAIL_RE.test(value.trim());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validateResume(resume: Resume): ResumeValidation {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  const raw = resume as unknown;

  if (!isRecord(raw)) {
    return {
      errors: [{ path: "resume", message: "The resume is not a valid object." }],
      warnings,
    };
  }

  const source = raw;

  const metadata = source.metadata;
  if (!isRecord(metadata)) {
    errors.push({ path: "metadata", message: "Resume metadata is missing or malformed." });
  } else if (!CONFIDENCE_LEVELS.has(String(metadata.overall_confidence))) {
    errors.push({
      path: "metadata.overall_confidence",
      message: "Resume confidence must be high, medium, or low.",
    });
  }

  const contact = source.contact;
  if (contact !== undefined && contact !== null && !isRecord(contact)) {
    errors.push({ path: "contact", message: "Contact details must be an object." });
  }

  for (const section of ARRAY_SECTION_KEYS) {
    const value = source[section];
    if (value !== undefined && value !== null && !Array.isArray(value)) {
      errors.push({
        path: section,
        message: `${SECTION_LABELS[section]} must be a list.`,
      });
      continue;
    }
    const list = Array.isArray(value) ? value : [];
    if (list.length > MAX_LIST_ITEMS) {
      errors.push({
        path: section,
        message: `${SECTION_LABELS[section]} has too many entries (max ${MAX_LIST_ITEMS}).`,
      });
    }
    list.forEach((entry, index) => {
      if (!isRecord(entry)) {
        errors.push({
          path: `${section}[${index}]`,
          message: `${SECTION_LABELS[section]} entry ${index + 1} is malformed.`,
        });
      }
    });
  }

  const skills = source.skills;
  if (skills !== undefined && skills !== null && !isRecord(skills)) {
    errors.push({ path: "skills", message: "Skills must be an object." });
  }

  checkText(errors, warnings, "summary", "Summary", source.summary);

  if (isRecord(contact)) {
    for (const field of CONTACT_FIELD_KEYS) {
      const label = CONTACT_FIELD_LABELS[field];
      const check = checkText(errors, warnings, `contact.${field}`, label, contact[field]);
      if (field === "email" && check && !isSafeEmail(check)) {
        errors.push({ path: "contact.email", message: "Email address is not valid." });
      }
      if (check && URL_FIELDS.includes(field)) checkUrl(errors, warnings, `contact.${field}`, label, check);
    }
  }

  checkSection(errors, warnings, source, "experience", [
    ["title", "job title"],
    ["company", "company"],
    ["location", "location"],
    ["start_date", "start date"],
    ["end_date", "end date"],
    ["description", "description"],
  ], [["achievements", "achievement"], ["skills_mentioned", "skill"]]);
  checkSection(errors, warnings, source, "education", [
    ["institution", "institution"],
    ["degree", "degree"],
    ["field", "field"],
    ["location", "location"],
    ["start_date", "start date"],
    ["end_date", "end date"],
  ], [["details", "detail"]]);
  checkSection(errors, warnings, source, "projects", [
    ["name", "name"],
    ["description", "description"],
    ["url", "url"],
  ], [["technologies", "technology"]]);
  checkSection(errors, warnings, source, "certifications", [
    ["name", "name"],
    ["issuer", "issuer"],
    ["date", "date"],
    ["url", "url"],
  ], []);
  checkSection(errors, warnings, source, "custom_sections", [["heading", "heading"]], [
    ["content", "line"],
  ]);

  if (isRecord(skills)) {
    for (const group of SKILL_GROUP_KEYS) {
      const label = SKILL_GROUP_LABELS[group];
      const value = skills[group];
      if (value !== undefined && value !== null && !Array.isArray(value)) {
        errors.push({ path: `skills.${group}`, message: `${label} must be a list.` });
        continue;
      }
      const items = Array.isArray(value) ? value : [];
      if (items.length > MAX_LIST_ITEMS) {
        errors.push({
          path: `skills.${group}`,
          message: `${label} has too many items (max ${MAX_LIST_ITEMS}).`,
        });
      }
      items.forEach((item, index) => {
        if (typeof item !== "string") {
          errors.push({
            path: `skills.${group}[${index}]`,
            message: `${label} item ${index + 1} must be text.`,
          });
          return;
        }
        checkText(errors, warnings, `skills.${group}[${index}]`, label, item);
      });
    }
  }

  validateEmptiness(errors, resume);

  return { errors, warnings };
}

function validateEmptiness(errors: ValidationIssue[], resume: Resume): void {
  if (isResumeEmpty(resume)) {
    errors.push({
      path: "resume",
      message: "This resume is empty — add some content before exporting.",
    });
  }
}

function checkSection(
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
  source: Record<string, unknown>,
  section: ArraySectionKey,
  scalarFields: readonly (readonly [string, string])[],
  listFields: readonly (readonly [string, string])[]
): void {
  const value = source[section];
  const list = Array.isArray(value) ? value : [];
  list.forEach((entry, index) => {
    if (!isRecord(entry)) return;
    const prefix = `${SECTION_LABELS[section]} ${index + 1}`;
    for (const [field, label] of scalarFields) {
      const text = checkText(
        errors,
        warnings,
        `${section}[${index}].${field}`,
        `${prefix} · ${label}`,
        entry[field]
      );
      if (text && URL_FIELDS.includes(field)) {
        checkUrl(errors, warnings, `${section}[${index}].${field}`, `${prefix} · ${label}`, text);
      }
    }
    for (const [field, label] of listFields) {
      const rawList = entry[field];
      if (rawList !== undefined && rawList !== null && !Array.isArray(rawList)) {
        errors.push({
          path: `${section}[${index}].${field}`,
          message: `${prefix} ${field} must be a list.`,
        });
        continue;
      }
      const items = Array.isArray(rawList) ? rawList : [];
      if (items.length > MAX_LIST_ITEMS) {
        errors.push({
          path: `${section}[${index}].${field}`,
          message: `${prefix} has too many ${field} (max ${MAX_LIST_ITEMS}).`,
        });
      }
      items.forEach((item, itemIndex) => {
        checkText(
          errors,
          warnings,
          `${section}[${index}].${field}[${itemIndex}]`,
          `${prefix} · ${label} ${itemIndex + 1}`,
          item
        );
      });
    }
    warnIfMissing(errors, warnings, section, index, prefix, entry);
  });
}

function warnIfMissing(
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
  section: ArraySectionKey,
  index: number,
  prefix: string,
  entry: Record<string, unknown>
): void {
  const required: Partial<Record<ArraySectionKey, string>> = {
    experience: "company",
    education: "institution",
    projects: "name",
    certifications: "name",
    custom_sections: "heading",
  };
  const field = required[section];
  if (!field) return;
  const value = entry[field];
  if (typeof value !== "string" || !value.trim()) {
    warnings.push({
      path: `${section}[${index}].${field}`,
      message: `${prefix} has no ${field.replace(/_/g, " ")}.`,
    });
  }
}

function checkText(
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
  path: string,
  label: string,
  value: unknown
): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") {
    errors.push({ path, message: `${label} must be text.` });
    return null;
  }
  if (value.length > MAX_FIELD_CHARS) {
    errors.push({
      path,
      message: `${label} is too long (max ${MAX_FIELD_CHARS} characters).`,
    });
    return value;
  }
  if (value.length > WARN_FIELD_CHARS) {
    warnings.push({
      path,
      message: `${label} is long (${value.length} characters) — consider tightening it.`,
    });
  }
  return value;
}

function checkUrl(
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
  path: string,
  label: string,
  value: string
): void {
  if (!value.trim()) return;
  if (UNSAFE_SCHEME_RE.test(value)) {
    errors.push({ path, message: `${label} uses an unsupported or unsafe link.` });
    return;
  }
  if (/\s/.test(value.trim())) {
    warnings.push({ path, message: `${label} does not look like a valid link.` });
  }
}
