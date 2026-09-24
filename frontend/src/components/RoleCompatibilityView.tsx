"use client";

import type { RoleAnalysisResult } from "@/lib/roleAnalysis";

interface RoleCompatibilityViewProps {
  result: RoleAnalysisResult | null;
}

export function RoleCompatibilityView({ result }: RoleCompatibilityViewProps) {
  if (!result) return null;
  const overall = result.compatibility.overall;
  const isUnknown = !result.profile.supported;

  return (
    <div className="space-y-5">
      <section aria-labelledby="role-compat-heading">
        <h3
          id="role-compat-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          Role-level compatibility
        </h3>
        <p className="mt-1 text-xs text-slate-500">
          Based on local role profiles, not a specific job description.
          {isUnknown ? " Role not recognized — results are limited." : null}
        </p>

        <div className="mt-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Target role
          </p>
          <p className="mt-1 text-lg font-semibold text-slate-900">
            {result.role_title}
          </p>
          {result.profile.aliases.length > 0 && (
            <p className="mt-1 text-xs text-slate-500">
              Also known as: {result.profile.aliases.join(", ")}
            </p>
          )}
        </div>

        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <ScoreCard
            label="Overall"
            value={overall}
            note={
              isUnknown
                ? "No local profile available"
                : undefined
            }
          />
          <ScoreCard
            label="Skills"
            value={result.compatibility.skills}
          />
          <ScoreCard
            label="Experience"
            value={result.compatibility.experience}
          />
        </div>
      </section>

      <section aria-labelledby="role-skills-heading">
        <h3
          id="role-skills-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          Skills
        </h3>
        <div className="mt-3 grid grid-cols-1 gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:grid-cols-2">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-emerald-600">
              You have {result.skills.matched_count} relevant skills
            </p>
            {result.skills.have.length === 0 ? (
              <p className="mt-2 text-sm text-slate-400">None detected.</p>
            ) : (
              <ul className="mt-2 flex flex-wrap gap-2">
                {result.skills.have.map((skill) => (
                  <li
                    key={skill}
                    className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700"
                  >
                    {skill}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-amber-600">
              Missing {result.skills.missing.length} skills
            </p>
            {result.skills.missing.length === 0 ? (
              <p className="mt-2 text-sm text-slate-400">None missing.</p>
            ) : (
              <ul className="mt-2 flex flex-wrap gap-2">
                {result.skills.missing.map((skill) => (
                  <li
                    key={skill}
                    className="rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700"
                  >
                    {skill}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </section>

      {result.experience.required_years !== null && (
        <section aria-labelledby="role-exp-heading">
          <h3
            id="role-exp-heading"
            className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
          >
            Experience
          </h3>
          <div className="mt-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Required
                </dt>
                <dd className="mt-0.5 text-sm text-slate-800">
                  {result.experience.required_level
                    ? `${result.experience.required_level}-level`
                    : "Not specified"}
                  {result.experience.required_years !== null
                    ? ` · ~${result.experience.required_years} years`
                    : null}
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Your experience
                </dt>
                <dd className="mt-0.5 text-sm text-slate-800">
                  {result.experience.candidate_years !== null
                    ? `~${result.experience.candidate_years} years`
                    : "Not verifiable"}
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Gap
                </dt>
                <dd className="mt-0.5 text-sm text-slate-800">
                  {result.experience.gap_years !== null ? (
                    result.experience.gap_years > 0 ? (
                      <span className="text-amber-700">
                        ~{result.experience.gap_years} years short
                      </span>
                    ) : (
                      <span className="text-emerald-700">Meets requirement</span>
                    )
                  ) : (
                    "N/A"
                  )}
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Education
                </dt>
                <dd className="mt-0.5 text-sm text-slate-800">
                  {result.education.required_level
                    ? `Requires ${result.education.required_level}`
                    : "Not specified"}
                  {result.education.met !== null ? (
                    result.education.met ? (
                      <span className="text-emerald-700"> · Met</span>
                    ) : (
                      <span className="text-amber-700"> · Not met</span>
                    )
                  ) : null}
                </dd>
              </div>
            </dl>
          </div>
        </section>
      )}

      {result.gaps.length > 0 && (
        <section aria-labelledby="role-gaps-heading">
          <h3
            id="role-gaps-heading"
            className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
          >
            Gaps
          </h3>
          <ul className="mt-3 space-y-2">
            {result.gaps.map((gap, index) => (
              <li
                key={index}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-sm"
              >
                {gap}
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.strengths.length > 0 && (
        <section aria-labelledby="role-strengths-heading">
          <h3
            id="role-strengths-heading"
            className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
          >
            Strengths
          </h3>
          <ul className="mt-3 space-y-2">
            {result.strengths.map((strength, index) => (
              <li
                key={index}
                className="flex gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800 shadow-sm"
              >
                <span aria-hidden="true" className="mt-0.5 text-emerald-600">
                  •
                </span>
                <span>{strength}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.recommendations.length > 0 && (
        <section aria-labelledby="role-recs-heading">
          <h3
            id="role-recs-heading"
            className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
          >
            Recommendations
          </h3>
          <ul className="mt-3 space-y-2">
            {result.recommendations.map((rec, index) => (
              <li
                key={index}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-sm"
              >
                <span className="font-medium text-slate-800">
                  {index + 1}.
                </span>{" "}
                {rec}
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className="text-xs text-slate-400">
        Role-level analysis uses local profile data and deterministic
        matching. Not a hiring probability.
      </p>
    </div>
  );
}

function ScoreCard({
  label,
  value,
  note,
}: {
  label: string;
  value: number | null;
  note?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      {value !== null && value !== undefined ? (
        <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">
          {Math.round(value)} / 100
        </p>
      ) : (
        <p className="mt-1 text-2xl font-semibold text-slate-400">
          {note ?? "Not available"}
        </p>
      )}
    </div>
  );
}
