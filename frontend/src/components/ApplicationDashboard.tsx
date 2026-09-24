"use client";

import type { DashboardStats } from "@/lib/applicationQuery";
import { STATUS_LABELS } from "@/lib/applications";

import { formatCoverage, formatScore } from "@/lib/applicationFormat";

interface ApplicationDashboardProps {
  stats: DashboardStats;
}

/**
 * Descriptive, session-only summaries of the tracked applications. All numbers
 * come from `computeDashboard`; nothing here predicts hiring outcomes.
 */
export function ApplicationDashboard({ stats }: ApplicationDashboardProps) {
  const hasScores =
    stats.averages.baseline_match !== null ||
    stats.averages.hybrid_match !== null ||
    stats.averages.ats_readiness !== null ||
    stats.averages.job_specific_ats_coverage !== null;

  const maxTimelineCount = Math.max(1, ...stats.timeline.map((point) => point.count));

  return (
    <div className="space-y-5">
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <StatCard label="Total" value={stats.total} />
        <StatCard label="Active" value={stats.active} />
        <StatCard label="Interviews" value={stats.interviews} />
        <StatCard label="Offers" value={stats.offers} />
        <StatCard label="Rejections" value={stats.rejections} />
      </dl>

      {hasScores ? (
        <div className="grid grid-cols-1 gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:grid-cols-2 lg:grid-cols-4">
          <ScoreCell label="Avg. match" value={formatScore(stats.averages.hybrid_match ?? stats.averages.baseline_match)} />
          <ScoreCell label="Avg. ATS readiness" value={formatScore(stats.averages.ats_readiness)} />
          <ScoreCell
            label="Avg. job-specific ATS"
            value={formatCoverage(stats.averages.job_specific_ats_coverage)}
          />
          <ScoreCell label="Scores tracked" value={String(stats.total)} />
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Recent applications
          </p>
          {stats.recent.length > 0 ? (
            <ul className="mt-3 space-y-2">
              {stats.recent.map((application) => (
                <li key={application.id} className="flex items-baseline justify-between gap-3">
                  <span className="truncate text-sm font-medium text-slate-800">
                    {application.company} — {application.role}
                  </span>
                  <span className="shrink-0 text-xs text-slate-500">
                    {STATUS_LABELS[application.status]}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No applications yet.</p>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Applications by month
          </p>
          {stats.timeline.length > 0 ? (
            <ul className="mt-3 space-y-2">
              {stats.timeline.map((point) => (
                <li key={point.month} className="flex items-center gap-3">
                  <span className="w-14 shrink-0 text-xs font-medium text-slate-600">{point.month}</span>
                  <div
                    role="img"
                    aria-label={`${point.month}: ${point.count} application${point.count === 1 ? "" : "s"}`}
                    className="h-3 rounded bg-indigo-100"
                    style={{ width: `${Math.max(8, (point.count / maxTimelineCount) * 100)}%` }}
                  />
                  <span className="shrink-0 text-xs text-slate-500">{point.count}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No applications yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-center shadow-sm">
      <dd className="text-2xl font-bold text-slate-900">{value}</dd>
      <dt className="mt-1 text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
    </div>
  );
}

function ScoreCell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-800">{value}</p>
    </div>
  );
}