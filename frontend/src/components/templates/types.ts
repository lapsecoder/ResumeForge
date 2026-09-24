import type { ComponentType } from "react";

import type { Resume } from "@/lib/resume";

export type TemplateId = "classic" | "modern" | "compact";

export interface ResumeTemplate {
  id: TemplateId;
  name: string;
  description: string;
  component: ComponentType<{ resume: Resume }>;
}
