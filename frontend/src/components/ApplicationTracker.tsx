"use client";

import { useMemo, useState } from "react";

import {
  APPLICATION_STATUSES,
  STATUS_LABELS,
  addApplication,
  changeStatus,
  removeApplication,
  updateApplication,
  type Application,
  type ApplicationDraft,
  type ApplicationStatus,
} from "@/lib/applications";
import {
  buildPrefill,
  computeDashboard,
  filterApplications,
  sortApplications,
  type PrefillSources,
  type SortDirection,
  type SortKey,
} from "@/lib/applicationQuery";

import { ApplicationDashboard } from "./ApplicationDashboard";
import { ApplicationForm } from "./ApplicationForm";
import { formatCoverage, formatDate, formatScore } from "@/lib/applicationFormat";

type StatusFilter = ApplicationStatus | "all";

interface ApplicationTrackerProps {
  applications: Application[];
  onChange: (applications: Application[]) => void;
  /** Sources for the add form when a matched job is present. */
  prefill?: PrefillSources;
}

/**
 * Client-side, session-only job application tracker. Nothing is persisted or
 * uploaded; refreshing the page discards everything.
 */
export function ApplicationTracker({
  applications,
  onChange,
  prefill,
}: ApplicationTrackerProps) {
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<Application | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const dashboard = useMemo(() => computeDashboard(applications), [applications]);

  const filtered = useMemo(
    () =>
      sortApplications(
        filterApplications(applications, { search, status: statusFilter }),
        sortKey,
        sortDirection
      ),
    [applications, search, statusFilter, sortKey, sortDirection]
  );

  const submitLabel = editing ? "Save changes" : "Add application";
  const hasMatches = Boolean(prefill?.job);
  const prefillLabel = hasMatches
    ? "Prefilled company, role, location, and analysis scores from the current match."
    : null;

  const openAdd = () => {
    setEditing(null);
    setFormOpen(true);
    setDeleteConfirmId(null);
  };

  const openEdit = (application: Application) => {
    setEditing(application);
    setFormOpen(true);
    setDeleteConfirmId(null);
  };

  const closeForm = () => {
    setFormOpen(false);
    setEditing(null);
  };

  const handleSave = (draft: ApplicationDraft) => {
    if (editing) {
      const updated = updateApplication(editing, draft);
      if (updated) {
        onChange(applications.map((application) => (application.id === updated.id ? updated : application)));
      }
    } else {
      const next = addApplication(applications, draft);
      if (next) onChange(next);
    }
    closeForm();
  };

  const handleStatusChange = (application: Application, status: string) => {
    const updated = changeStatus(application, status as ApplicationStatus);
    if (updated) {
      onChange(applications.map((item) => (item.id === updated.id ? updated : item)));
    }
  };

  const handleDelete = (id: string) => {
    if (deleteConfirmId !== id) {
      setDeleteConfirmId(id);
      return;
    }
    const next = removeApplication(applications, id);
    if (next) onChange(next);
    setDeleteConfirmId(null);
  };

  const initialDraft: ApplicationDraft = editing
    ? {
        company: editing.company,
        role: editing.role,
        location: editing.location,
        url: editing.url,
        applied_date: editing.applied_date,
        status: editing.status,
        notes: editing.notes,
        resume_template: editing.resume_template,
      }
    : buildPrefill(prefill ?? {});

  return (
    <section aria-labelledby="tracker-heading" className="space-y-5">
      <div>
        <h3
          id="tracker-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          Application tracker
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Track job applications and analysis results. Everything lives in this
          browser session only and is never stored.
        </p>
      </div>

      {applications.length > 0 ? <ApplicationDashboard stats={dashboard} /> : null}

      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <div>
          <label htmlFor="tracker-search" className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Search
          </label>
          <input
            id="tracker-search"
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Company, role, location…"
            maxLength={100}
            className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 sm:w-56"
          />
        </div>
        <div>
          <label htmlFor="tracker-status" className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Status filter
          </label>
          <select
            id="tracker-status"
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value as StatusFilter)}
            className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 sm:w-44"
          >
            <option value="all">All statuses</option>
            {APPLICATION_STATUSES.map((status) => (
              <option key={status} value={status}>{STATUS_LABELS[status]}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="tracker-sort" className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Sort by
          </label>
          <select
            id="tracker-sort"
            value={sortKey}
            onChange={(event) => setSortKey(event.target.value as SortKey)}
            className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 sm:w-36"
          >
            <option value="date">Date</option>
            <option value="company">Company</option>
            <option value="role">Role</option>
            <option value="score">Score</option>
          </select>
        </div>
        <button
          type="button"
          onClick={() => setSortDirection((direction) => (direction === "asc" ? "desc" : "asc"))}
          className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1"
        >
          {sortDirection === "asc" ? "Ascending" : "Descending"}
        </button>
        <button
          type="button"
          onClick={openAdd}
          className="ml-auto rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
        >
          Add application
        </button>
      </div>

      {formOpen ? (
        <ApplicationForm
          key={editing?.id ?? "new"}
          initial={initialDraft}
          submitLabel={submitLabel}
          prefillLabel={prefillLabel}
          onSave={handleSave}
          onCancel={closeForm}
        />
      ) : null}

      {applications.length === 0 && !formOpen ? (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center shadow-sm">
          <p className="text-sm font-medium text-slate-700">No applications tracked yet.</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
            Add an application manually or after matching a job description; its analysis
            scores are carried over automatically. Nothing is saved to disk or uploaded.
          </p>
        </div>
      ) : null}

      {applications.length > 0 && filtered.length === 0 ? (
        <div className="rounded-xl border border-slate-200 bg-white px-6 py-8 text-center shadow-sm">
          <p className="text-sm text-slate-600">No applications match your search or filter.</p>
        </div>
      ) : null}

      {filtered.length > 0 ? (
        <ul className="space-y-3" aria-label="Tracked applications">
          {filtered.map((application) => {
            const matchScore = application.scores.hybrid_match ?? application.scores.baseline_match;
            const hasScores =
              matchScore !== null ||
              application.scores.ats_readiness !== null ||
              application.scores.job_specific_ats_coverage !== null;
            const rowLabel = `${application.company} ${application.role}`;
            return (
              <li
                key={application.id}
                className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-slate-800">
                      {application.company}
                      <span className="font-normal text-slate-500"> — {application.role}</span>
                    </p>
                    <p className="mt-0.5 text-xs text-slate-500">
                      {[application.location, application.applied_date ? `Applied ${formatDate(application.applied_date)}` : null]
                        .filter(Boolean)
                        .join(" · ") || "No location or date set"}
                    </p>
                    {hasScores ? (
                      <p className="mt-1 text-xs text-slate-500">
                        {matchScore !== null ? `Match ${formatScore(matchScore)} · ` : ""}
                        {application.scores.ats_readiness !== null
                          ? `ATS ${formatScore(application.scores.ats_readiness)} · `
                          : ""}
                        {application.scores.job_specific_ats_coverage !== null
                          ? `Coverage ${formatCoverage(application.scores.job_specific_ats_coverage)}`
                          : ""}
                      </p>
                    ) : (
                      <p className="mt-1 text-xs text-slate-400">No analysis scores attached.</p>
                    )}
                    {application.notes ? (
                      <p className="mt-1 line-clamp-2 text-xs text-slate-500">{application.notes}</p>
                    ) : null}
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <label className="sr-only" htmlFor={`status-${application.id}`}>
                      Status for {rowLabel}
                    </label>
                    <select
                      id={`status-${application.id}`}
                      value={application.status}
                      onChange={(event) => handleStatusChange(application, event.target.value)}
                      className="rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-xs font-semibold text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1"
                    >
                      {APPLICATION_STATUSES.map((status) => (
                        <option key={status} value={status}>{STATUS_LABELS[status]}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => openEdit(application)}
                      className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1"
                    >
                      Edit
                    </button>
                    {deleteConfirmId === application.id ? (
                      <button
                        type="button"
                        onClick={() => handleDelete(application.id)}
                        className="rounded-lg bg-red-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-red-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-1"
                      >
                        Confirm delete
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => handleDelete(application.id)}
                        className="rounded-lg border border-red-200 bg-white px-3 py-1.5 text-xs font-semibold text-red-600 transition-colors hover:bg-red-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-1"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      ) : null}
    </section>
  );
}