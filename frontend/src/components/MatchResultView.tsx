"use client";

import type { HybridMatchResult } from "@/lib/matcher";

interface MatchResultViewProps {
  result: HybridMatchResult;
}

/**
 * Displays a transient hybrid match result.
 *
 * Guardrails kept visible in the UI: the hybrid score is a heuristic
 * relevance score (not a hiring probability), deterministic evidence is
 * authoritative for explicit requirements, and semantic relatedness never
 * implies skill possession. When the semantic model was unavailable, that is
 * surfaced rather than hidden.
 */
export function MatchResultView({ result }: MatchResultViewProps) {
  const overall = result.overall_score;
  const semanticAvailable = result.metadata.semantic_availability.available;
  const semanticMode = result.metadata.mode !== "hybrid";

  return (
    <div className="space-y-5">
      <section aria-labelledby="match-score-heading">
        <h3
          id="match-score-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          Match score
        </h3>
        <div className="mt-3 space-y-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <ScoreRow label="Hybrid Match Score" value={overall} format="percent" />
          <DividedSignal
            deterministic={result.component_scores.deterministic_overall}
            semantic={result.component_scores.semantic_overall}
            semanticAvailable={semanticAvailable}
          />
        </div>
      </section>

      {semanticAvailable && result.semantic ? (
        <section aria-labelledby="semantic-disclosure-heading">
          <h3
            id="semantic-disclosure-heading"
            className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
          >
            Semantic signal
          </h3>
          <div className="mt-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-sm leading-relaxed text-slate-700">
              {result.semantic.note}
            </p>
            <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
              <MetadataItem label="Model" value={result.semantic.metadata.model_name} />
              <MetadataItem label="Model version" value={result.semantic.metadata.model_version} />
              <MetadataItem label="Device" value={result.semantic.metadata.device} />
              <MetadataItem label="License" value={result.semantic.metadata.model_license} />
            </dl>
            <p className="mt-3 text-xs text-slate-500">
              {result.metadata.model_disclosure}
            </p>
          </div>
        </section>
      ) : (
        <section
          aria-label="Semantic model unavailable"
          className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"
        >
          {result.metadata.semantic_availability.note ||
            "The local semantic model is unavailable; this result is supported by the deterministic signal only."}
        </section>
      )}

      {semanticMode && !semanticAvailable ? <SemanticUnavailableHint /> : null}

      <RequirementLists result={result} />
      <InsightsList insights={result.semantic_insights} />
    </div>
  );
}

function ScoreRow({
  label,
  value,
  format,
}: {
  label: string;
  value: number | null;
  format: "percent" | "raw";
}) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return (
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-sm font-medium text-slate-600">{label}</span>
        <span className="text-sm text-slate-400">Not available</span>
      </div>
    );
  }
  const display =
    format === "percent" ? `${Math.round(value)} / 100` : `${value.toFixed(2)}`;
  return (
    <div className="flex items-baseline justify-between gap-4">
      <span className="text-sm font-medium text-slate-600">{label}</span>
      <span className="text-lg font-semibold tabular-nums text-slate-900">{display}</span>
    </div>
  );
}

function DividedSignal({
  deterministic,
  semantic,
  semanticAvailable,
}: {
  deterministic: number | null;
  semantic: number | null;
  semanticAvailable: boolean;
}) {
  return (
    <div className="border-t border-slate-100 pt-3">
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Deterministic match (evidence-based)
        </span>
        <span className="text-sm font-semibold tabular-nums text-slate-800">
          {deterministic === null ? "Not available" : `${Math.round(deterministic)} / 100`}
        </span>
      </div>
      <div className="mt-1 flex items-baseline justify-between gap-4">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Semantic relatedness (local ML model)
        </span>
        <span className="text-sm font-semibold tabular-nums text-slate-800">
          {!semanticAvailable
            ? "Unavailable"
            : semantic === null
              ? "Not available"
              : `${semantic.toFixed(2)} / 1.00`}
        </span>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Relatedness is not a hiring probability and does not prove a candidate
        owns a skill. Deterministic evidence stays authoritative for explicit
        requirements.
      </p>
    </div>
  );
}

function SemanticUnavailableHint() {
  return (
    <p aria-hidden="true" className="sr-only">
      Semantic model unavailable; deterministic evidence shown only.
    </p>
  );
}

function MetadataItem({ label, value }: { label: string; value: string | undefined }) {
  if (!value) return null;
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="mt-0.5 break-words text-sm text-slate-800">{value}</dd>
    </div>
  );
}

function RequirementLists({ result }: { result: HybridMatchResult }) {
  const matched = result.matched_requirements;
  const missing = result.missing_required;
  if (matched.length === 0 && missing.length === 0) return null;

  return (
    <section aria-labelledby="requirements-heading">
      <h3
        id="requirements-heading"
        className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
      >
        Requirements
      </h3>
      <div className="mt-3 grid grid-cols-1 gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:grid-cols-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-emerald-600">
            Explicitly matched
          </p>
          {matched.length === 0 ? (
            <p className="mt-2 text-sm text-slate-400">None verified.</p>
          ) : (
            <ul className="mt-2 space-y-1.5">
              {matched.map((item, index) => (
                <li key={index} className="flex gap-2 text-sm text-slate-700">
                  <span aria-hidden="true" className="mt-0.5 text-emerald-500">
                    •
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Not explicitly verified
          </p>
          {missing.length === 0 ? (
            <p className="mt-2 text-sm text-slate-400">All required requirements verified.</p>
          ) : (
            <ul className="mt-2 space-y-1.5">
              {missing.map((item, index) => (
                <li key={index} className="flex gap-2 text-sm text-slate-700">
                  <span aria-hidden="true" className="mt-0.5 text-amber-500">
                    •
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-2 text-xs text-slate-400">
            Items here were not found in the resume; semantic relatedness does
            not convert them into verified matches.
          </p>
        </div>
      </div>
    </section>
  );
}

function InsightsList({ insights }: { insights: HybridMatchResult["semantic_insights"] }) {
  if (insights.length === 0) return null;
  return (
    <section aria-labelledby="insights-heading">
      <h3
        id="insights-heading"
        className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
      >
        Relatedness insights
      </h3>
      <ul className="mt-3 space-y-2">
        {insights.map((insight, index) => (
          <li key={index} className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
            <p className="text-sm leading-relaxed text-slate-700">{insight.statement}</p>
            {insight.similarity !== null && insight.similarity !== undefined ? (
              <p className="mt-1 text-xs text-slate-400">
                {insight.category}: {insight.similarity.toFixed(2)} similarity
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}