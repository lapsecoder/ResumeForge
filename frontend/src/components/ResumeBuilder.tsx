"use client";

import { useMemo, useState } from "react";

import type { HistoryCommitOptions } from "@/lib/history";
import { exportResumePdf } from "@/lib/pdf";
import type { Resume } from "@/lib/resume";
import { validateResume } from "@/lib/builderValidation";

import { BuilderEditor } from "./BuilderEditor";
import { TemplatePreview } from "./TemplatePreview";
import { TEMPLATES, type TemplateId } from "./templates";

interface ResumeBuilderProps {
  resume: Resume;
  canUndo: boolean;
  canRedo: boolean;
  onCommit: (resume: Resume, options?: HistoryCommitOptions) => void;
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
  templateId: TemplateId;
  onTemplateChange: (id: TemplateId) => void;
}

export function ResumeBuilder({
  resume,
  canUndo,
  canRedo,
  onCommit,
  onUndo,
  onRedo,
  onReset,
  templateId,
  onTemplateChange,
}: ResumeBuilderProps) {
  const [mobileTab, setMobileTab] = useState<"edit" | "preview">("edit");
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  const validation = useMemo(() => validateResume(resume), [resume]);
  const hasErrors = validation.errors.length > 0;

const handleExport = async () => {
    const current = validateResume(resume);
    if (current.errors.length > 0) {
      setExportMessage("Fix the validation errors before exporting.");
      return;
    }
    setExportMessage("Preparing PDF…");
    try {
      const filename = await exportResumePdf(resume, templateId);
      setExportMessage(`PDF exported: ${filename}`);
    } catch (error) {
      console.error("PDF export failed:", error);
      setExportMessage("PDF export failed. Please try again.");
    }
  };

  return (
    <section aria-labelledby="builder-heading" className="space-y-4">
      <div>
        <h3
          id="builder-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          Resume builder
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Edit the parsed resume directly, switch templates, and export a PDF.
          Changes live in this browser session only and are never stored.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <div>
          <label htmlFor="builder-template" className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Template
          </label>
          <select
            id="builder-template"
            value={templateId}
            onChange={(event) => onTemplateChange(event.target.value as TemplateId)}
            className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 sm:w-56"
          >
            {TEMPLATES.map((template) => (
              <option key={template.id} value={template.id}>
                {template.name}
              </option>
            ))}
          </select>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onUndo}
            disabled={!canUndo}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Undo
          </button>
          <button
            type="button"
            onClick={onRedo}
            disabled={!canRedo}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Redo
          </button>
          <button
            type="button"
            onClick={onReset}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50"
          >
            Reset changes
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={hasErrors}
            title={hasErrors ? "Fix validation errors before exporting" : undefined}
            className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Export PDF
          </button>
        </div>
      </div>

      <p className="text-xs text-slate-400">{TEMPLATES.find((t) => t.id === templateId)?.description}</p>

      <ValidationSummary resume={resume} />

      <p role="status" aria-live="polite" className="min-h-[1rem] text-xs font-medium text-slate-600">
        {exportMessage}
      </p>

      <div className="flex gap-2 lg:hidden" role="tablist" aria-label="Builder view">
        <button
          type="button"
          role="tab"
          aria-selected={mobileTab === "edit"}
          onClick={() => setMobileTab("edit")}
          className={`flex-1 rounded-lg border px-3 py-2 text-sm font-semibold ${
            mobileTab === "edit"
              ? "border-indigo-300 bg-indigo-50 text-indigo-700"
              : "border-slate-300 bg-white text-slate-600"
          }`}
        >
          Edit
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mobileTab === "preview"}
          onClick={() => setMobileTab("preview")}
          className={`flex-1 rounded-lg border px-3 py-2 text-sm font-semibold ${
            mobileTab === "preview"
              ? "border-indigo-300 bg-indigo-50 text-indigo-700"
              : "border-slate-300 bg-white text-slate-600"
          }`}
        >
          Preview
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className={mobileTab === "edit" ? "block" : "hidden lg:block"}>
          <BuilderEditor resume={resume} onChange={onCommit} />
        </div>
        <div className={mobileTab === "preview" ? "block" : "hidden lg:block"}>
          <div className="lg:sticky lg:top-4">
            <TemplatePreview resume={resume} templateId={templateId} />
          </div>
        </div>
      </div>
    </section>
  );
}

function ValidationSummary({ resume }: { resume: Resume }) {
  const { errors, warnings } = useMemo(() => validateResume(resume), [resume]);
  if (errors.length === 0 && warnings.length === 0) return null;
  return (
    <div className="space-y-2">
      {errors.length > 0 ? (
        <div role="alert" className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <p className="font-semibold">
            {errors.length} issue{errors.length === 1 ? "" : "s"} must be fixed before exporting:
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {errors.slice(0, 8).map((issue, index) => (
              <li key={`${issue.path}-${index}`}>{issue.message}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {warnings.length > 0 ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <p className="font-semibold">{warnings.length} suggestion{warnings.length === 1 ? "" : "s"}:</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {warnings.slice(0, 6).map((issue, index) => (
              <li key={`${issue.path}-${index}`}>{issue.message}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
