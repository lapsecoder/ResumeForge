"use client";

import { useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";

import type { Resume } from "@/lib/resume";

import { getTemplate, type TemplateId } from "./templates";

const emptySubscribe = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

/**
 * Mounts the resume into a body-level, print-only root. Keeping the print
 * output separate from the app shell means the browser prints only the resume
 * (no clipping from surrounding scroll containers, no blank trailing pages).
 */
function PrintRoot({ children }: { children: ReactNode }) {
  const isClient = useSyncExternalStore(emptySubscribe, getClientSnapshot, getServerSnapshot);
  if (!isClient) return null;
  return createPortal(<div className="resume-print-root">{children}</div>, document.body);
}

interface TemplatePreviewProps {
  resume: Resume;
  templateId: TemplateId;
  /** Accessible label for the preview region. */
  label?: string;
}

export function TemplatePreview({ resume, templateId, label = "Resume preview" }: TemplatePreviewProps) {
  const Template = getTemplate(templateId).component;
  return (
    <>
      <div
        data-testid="resume-preview"
        role="region"
        aria-label={label}
        className="max-h-[70vh] overflow-auto rounded-xl border border-slate-200 bg-slate-100 p-3 sm:p-4"
      >
        <div className="resume-paper">
          <Template resume={resume} />
        </div>
      </div>
      <PrintRoot>
        <div className="resume-paper">
          <Template resume={resume} />
        </div>
      </PrintRoot>
    </>
  );
}
