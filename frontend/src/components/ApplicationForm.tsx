"use client";

import { useState, type FormEvent } from "react";

import {
  APPLICATION_STATUSES,
  MAX_LOCATION_CHARS,
  MAX_NOTES_CHARS,
  MAX_TEMPLATE_CHARS,
  STATUS_LABELS,
  validateApplicationDraft,
  type ApplicationDraft,
} from "@/lib/applications";

interface ApplicationFormProps {
  /** Pre-filled draft (from a matched job or an existing application). */
  initial?: ApplicationDraft;
  submitLabel: string;
  /** Shown when the draft was pre-filled from a matched job. */
  prefillLabel?: string | null;
  onSave: (draft: ApplicationDraft) => void;
  onCancel: () => void;
}

const inputClass =
  "mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1";

const labelClass = "text-xs font-medium uppercase tracking-wide text-slate-500";

export function ApplicationForm({
  initial,
  submitLabel,
  prefillLabel,
  onSave,
  onCancel,
}: ApplicationFormProps) {
  const [draft, setDraft] = useState<ApplicationDraft>(() => ({
    company: initial?.company ?? "",
    role: initial?.role ?? "",
    location: initial?.location ?? "",
    url: initial?.url ?? "",
    applied_date: initial?.applied_date ?? "",
    status: initial?.status ?? "saved",
    notes: initial?.notes ?? "",
    resume_template: initial?.resume_template ?? "",
  }));

  const leftoverScores = initial?.scores;
  const issues = validateApplicationDraft(draft);
  const hasErrors = issues.length > 0;

  const setField = (field: keyof ApplicationDraft, value: string) => {
    setDraft((current) => ({ ...current, [field]: value }));
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (hasErrors) return;
    onSave(leftoverScores ? { ...draft, scores: leftoverScores } : draft);
  };

  return (
    <form onSubmit={handleSubmit} noValidate className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm font-semibold text-slate-800">{submitLabel}</p>
      {prefillLabel ? <p className="mt-1 text-xs text-slate-500">{prefillLabel}</p> : null}

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label htmlFor="application-company" className={labelClass}>Company</label>
          <input
            id="application-company"
            value={draft.company ?? ""}
            maxLength={200}
            onChange={(event) => setField("company", event.target.value)}
            className={inputClass}
            autoComplete="off"
          />
        </div>
        <div>
          <label htmlFor="application-role" className={labelClass}>Job title</label>
          <input
            id="application-role"
            value={draft.role ?? ""}
            maxLength={200}
            onChange={(event) => setField("role", event.target.value)}
            className={inputClass}
            autoComplete="off"
          />
        </div>
        <div>
          <label htmlFor="application-location" className={labelClass}>Location</label>
          <input
            id="application-location"
            value={draft.location ?? ""}
            maxLength={MAX_LOCATION_CHARS}
            onChange={(event) => setField("location", event.target.value)}
            className={inputClass}
            autoComplete="off"
          />
        </div>
        <div>
          <label htmlFor="application-url" className={labelClass}>Job link</label>
          <input
            id="application-url"
            value={draft.url ?? ""}
            maxLength={2000}
            onChange={(event) => setField("url", event.target.value)}
            className={inputClass}
            autoComplete="off"
            placeholder="https://…"
          />
        </div>
        <div>
          <label htmlFor="application-date" className={labelClass}>Applied date</label>
          <input
            id="application-date"
            type="date"
            value={draft.applied_date ?? ""}
            onChange={(event) => setField("applied_date", event.target.value)}
            className={inputClass}
          />
        </div>
        <div>
          <label htmlFor="application-status" className={labelClass}>Status</label>
          <select
            id="application-status"
            value={draft.status ?? "saved"}
            onChange={(event) => setField("status", event.target.value)}
            className={inputClass}
          >
            {APPLICATION_STATUSES.map((status) => (
              <option key={status} value={status}>{STATUS_LABELS[status]}</option>
            ))}
          </select>
        </div>
        <div className="sm:col-span-2">
          <label htmlFor="application-template" className={labelClass}>Resume template</label>
          <input
            id="application-template"
            value={draft.resume_template ?? ""}
            maxLength={MAX_TEMPLATE_CHARS}
            onChange={(event) => setField("resume_template", event.target.value)}
            className={inputClass}
            autoComplete="off"
          />
        </div>
        <div className="sm:col-span-2">
          <label htmlFor="application-notes" className={labelClass}>Notes</label>
          <textarea
            id="application-notes"
            value={draft.notes ?? ""}
            maxLength={MAX_NOTES_CHARS}
            onChange={(event) => setField("notes", event.target.value)}
            rows={4}
            className={inputClass}
          />
        </div>
      </div>

      {issues.length > 0 ? (
        <ul role="alert" className="mt-3 space-y-1 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {issues.map((issue, index) => (
            <li key={`${issue.path}-${index}`}>{issue.message}</li>
          ))}
        </ul>
      ) : null}

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="submit"
          disabled={hasErrors}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitLabel}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}