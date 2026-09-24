/**
 * Client-side apply/revert for Phase 7C Copilot edit proposals.
 *
 * The backend only ever *proposes* an edit. This module mirrors the backend's
 * editable-field allowlist as defence in depth: a tampered or stale target is
 * refused here rather than written into the user's in-memory resume. Applying
 * an edit returns a NEW resume object; the input is never mutated, and nothing
 * is persisted or sent anywhere.
 */

import type { CopilotEditTarget } from "./copilot";
import type { Resume } from "./resume";

/** Mirrors `COPILOT_MAX_EDIT_VALUE_CHARS` on the backend. */
export const MAX_EDIT_VALUE_CHARS = 4000;

const EDIT_PATH_RE =
  /^(summary|experience|projects)(?:\[(\d+)\])?(?:\.([a-z_]+)(?:\[(\d+)\])?)?$/;

const FIELD_ALIASES: Record<string, string> = { bullets: "achievements" };

const SCALAR_FIELDS: Record<string, readonly string[]> = {
  experience: ["title", "company", "description"],
  projects: ["description"],
};

const LIST_FIELDS: Record<string, readonly string[]> = {
  experience: ["achievements"],
  projects: ["technologies"],
};

function cloneResume(resume: Resume): Resume {
  return JSON.parse(JSON.stringify(resume)) as Resume;
}

/**
 * Parse a readable edit path into a structured target, or null when the path
 * is not on the editable allowlist.
 */
export function parseEditPath(path: string): CopilotEditTarget | null {
  const match = EDIT_PATH_RE.exec((path ?? "").trim());
  if (!match) return null;
  const [, section, rawIndex, rawField, rawSubIndex] = match;

  if (section === "summary") {
    if (rawIndex !== undefined || rawField !== undefined) return null;
    return { path: "summary", section: "summary", index: null, field: null, sub_index: null };
  }

  if (rawIndex === undefined || rawField === undefined) return null;
  const index = Number.parseInt(rawIndex, 10);
  if (!Number.isFinite(index)) return null;
  const field = FIELD_ALIASES[rawField] ?? rawField;

  if (section === "experience") {
    if (SCALAR_FIELDS.experience.includes(field)) {
      if (rawSubIndex !== undefined) return null;
      return {
        path: `experience[${index}].${field}`,
        section,
        index,
        field,
        sub_index: null,
      };
    }
    if (LIST_FIELDS.experience.includes(field)) {
      if (rawSubIndex === undefined) return null;
      const subIndex = Number.parseInt(rawSubIndex, 10);
      if (!Number.isFinite(subIndex)) return null;
      return {
        path: `experience[${index}].${field}[${subIndex}]`,
        section,
        index,
        field,
        sub_index: subIndex,
      };
    }
    return null;
  }

  if (section === "projects") {
    if (rawSubIndex !== undefined) return null;
    if (SCALAR_FIELDS.projects.includes(field)) {
      return {
        path: `projects[${index}].${field}`,
        section,
        index,
        field,
        sub_index: null,
      };
    }
    if (LIST_FIELDS.projects.includes(field)) {
      return {
        path: `projects[${index}].${field}`,
        section,
        index,
        field,
        sub_index: null,
      };
    }
    return null;
  }

  return null;
}

/** True when a target is allowlisted and internally consistent. */
export function isEditableTarget(target: CopilotEditTarget | null | undefined): boolean {
  if (!target) return false;
  const parsed = parseEditPath(target.path);
  if (!parsed) return false;
  return (
    parsed.section === target.section &&
    (parsed.index ?? null) === (target.index ?? null) &&
    (parsed.field ?? null) === (target.field ?? null) &&
    (parsed.sub_index ?? null) === (target.sub_index ?? null)
  );
}

function targetInRange(resume: Resume, target: CopilotEditTarget): boolean {
  if (target.section === "summary") return true;
  if (target.index === null || target.index === undefined || target.index < 0) {
    return false;
  }
  if (target.section === "experience") {
    const experience = resume.experience ?? [];
    if (target.index >= experience.length) return false;
    if (target.field === "achievements") {
      if (target.sub_index === null || target.sub_index === undefined || target.sub_index < 0) {
        return false;
      }
      return target.sub_index < (experience[target.index].achievements ?? []).length;
    }
    return SCALAR_FIELDS.experience.includes(target.field ?? "");
  }
  if (target.section === "projects") {
    const projects = resume.projects ?? [];
    if (target.index >= projects.length) return false;
    return (
      SCALAR_FIELDS.projects.includes(target.field ?? "") ||
      LIST_FIELDS.projects.includes(target.field ?? "")
    );
  }
  return false;
}

/**
 * Return a new resume with `target` set to `value`, or null when the target is
 * not editable, out of range, or the value exceeds the length bound.
 */
export function applyEdit(
  resume: Resume,
  target: CopilotEditTarget | null | undefined,
  value: string
): Resume | null {
  if (!isEditableTarget(target)) return null;
  if (typeof value !== "string" || value.length > MAX_EDIT_VALUE_CHARS) return null;
  const safeTarget = target as CopilotEditTarget;
  if (!targetInRange(resume, safeTarget)) return null;

  const next = cloneResume(resume);
  if (safeTarget.section === "summary") {
    next.summary = value;
    return next;
  }
  const index = safeTarget.index as number;
  if (safeTarget.section === "experience") {
    const entry = (next.experience ?? [])[index];
    if (!entry) return null;
    if (safeTarget.field === "achievements") {
      const subIndex = safeTarget.sub_index as number;
      entry.achievements = entry.achievements ?? [];
      entry.achievements[subIndex] = value;
    } else if (safeTarget.field === "title") {
      entry.title = value;
    } else if (safeTarget.field === "company") {
      entry.company = value;
    } else if (safeTarget.field === "description") {
      entry.description = value;
    }
    return next;
  }
  if (safeTarget.section === "projects") {
    const project = (next.projects ?? [])[index];
    if (!project) return null;
    if (safeTarget.field === "technologies") {
      project.technologies = value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    } else if (safeTarget.field === "description") {
      project.description = value;
    }
    return next;
  }
  return null;
}

/** Human-readable label for an edit target, used in the UI. */
export function editTargetLabel(target: CopilotEditTarget | null | undefined): string {
  if (!target) return "Resume";
  const oneBased = (value: number | null | undefined) => (value ?? 0) + 1;
  switch (target.section) {
    case "summary":
      return "Professional summary";
    case "experience":
      if (target.field === "title") return `Experience ${oneBased(target.index)} · job title`;
      if (target.field === "company") return `Experience ${oneBased(target.index)} · company`;
      if (target.field === "description") return `Experience ${oneBased(target.index)} · description`;
      if (target.field === "achievements") {
        return `Experience ${oneBased(target.index)} · achievement ${oneBased(target.sub_index)}`;
      }
      return `Experience ${oneBased(target.index)}`;
    case "projects":
      if (target.field === "description") return `Project ${oneBased(target.index)} · description`;
      if (target.field === "technologies") return `Project ${oneBased(target.index)} · technologies`;
      return `Project ${oneBased(target.index)}`;
    default:
      return target.path;
  }
}
