import { ClassicTemplate } from "./ClassicTemplate";
import { CompactTemplate } from "./CompactTemplate";
import { ModernTemplate } from "./ModernTemplate";
import type { ResumeTemplate, TemplateId } from "./types";

export type { ResumeTemplate, TemplateId } from "./types";

export const TEMPLATES: readonly ResumeTemplate[] = [
  {
    id: "classic",
    name: "Classic",
    description: "Single-column, ATS-friendly layout with plain typography.",
    component: ClassicTemplate,
  },
  {
    id: "modern",
    name: "Modern",
    description: "Accent headings and a right-aligned contact block.",
    component: ModernTemplate,
  },
  {
    id: "compact",
    name: "Compact",
    description: "Dense two-column label layout that fits more on one page.",
    component: CompactTemplate,
  },
];

export const DEFAULT_TEMPLATE_ID: TemplateId = "classic";

export function isTemplateId(value: unknown): value is TemplateId {
  return typeof value === "string" && TEMPLATES.some((template) => template.id === value);
}

/** Resolve a template by id, falling back to the default for unknown ids. */
export function getTemplate(id: string | undefined | null): ResumeTemplate {
  return TEMPLATES.find((template) => template.id === id) ?? TEMPLATES[0];
}
