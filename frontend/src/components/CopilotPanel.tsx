"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiErrorMessage } from "@/lib/api";
import {
  analyzeAtsReadiness,
  analyzeJobSpecificAts,
  type ATSReadinessResult,
  type JobSpecificATSResult,
} from "@/lib/ats";
import {
  getCopilotStatus,
  suggestCopilot,
  type CopilotAnalysisContext,
  type CopilotClarification,
  type CopilotEditProposal,
  type CopilotEditTarget,
  type CopilotOperation,
  type CopilotSuggestion,
  type CopilotSuggestRequest,
  type CopilotStatus,
} from "@/lib/copilot";
import { applyEdit, editTargetLabel } from "@/lib/edit";
import type { HybridMatchResult, JobDescription } from "@/lib/matcher";
import type { Resume } from "@/lib/resume";

interface BulletTarget {
  ref: string;
  text: string;
  label: string;
}

interface JobMatchData {
  job: JobDescription;
  result: HybridMatchResult;
}

interface CopilotPanelProps {
  resume: Resume;
  jobMatch?: JobMatchData | null;
  /**
   * Called with a NEW resume object when the user applies or reverts an edit.
   * Edits live only in the caller's in-memory state; nothing is stored.
   */
  onResumeChange?: (resume: Resume) => void;
}

interface AppliedEditRecord {
  editId: string;
  originalValue: string;
  target: CopilotEditTarget;
}

interface OperationMeta {
  value: CopilotOperation;
  label: string;
  hint: string;
  needsJob: boolean;
}

const OPERATIONS: OperationMeta[] = [
  {
    value: "improve_summary",
    label: "Improve summary",
    hint: "Rewrites the professional summary while preserving facts.",
    needsJob: false,
  },
  {
    value: "improve_bullet",
    label: "Improve a bullet",
    hint: "Tightens a selected achievement or description.",
    needsJob: false,
  },
  {
    value: "identify_priorities",
    label: "Identify priorities",
    hint: "Ranks the most impactful fixes for this resume.",
    needsJob: false,
  },
  {
    value: "explain_finding",
    label: "Explain a finding",
    hint: "Explains one ATS finding and how to fix it.",
    needsJob: false,
  },
  {
    value: "job_alignment",
    label: "Job alignment",
    hint: "Aligns the resume with the job description.",
    needsJob: true,
  },
];

const VERIFICATION_LABEL: Record<string, string> = {
  verified: "Verified against the resume",
  inferred: "Inferred from resume facts",
  advisory: "Advisory (no factual claim)",
  unverified: "Unverified and needs review",
};

/** Max length of a free-text Copilot request (mirrors the backend bound). */
const MAX_FREE_REQUEST_CHARS = 2000;

function bulletTargets(resume: Resume): BulletTarget[] {
  const targets: BulletTarget[] = [];
  resume.experience?.forEach((entry, i) => {
    entry.achievements?.forEach((achievement, j) => {
      if (achievement.trim()) {
        targets.push({
          ref: `experience[${i}].achievements[${j}]`,
          text: achievement.trim(),
          label: achievement.trim(),
        });
      }
    });
    if (entry.description?.trim()) {
      targets.push({
        ref: `experience[${i}].description`,
        text: entry.description.trim(),
        label: entry.description.trim(),
      });
    }
  });
  resume.projects?.forEach((project, i) => {
    if (project.description?.trim()) {
      targets.push({
        ref: `projects[${i}].description`,
        text: project.description.trim(),
        label: project.description.trim(),
      });
    }
  });
  if (resume.summary?.trim()) {
    targets.unshift({
      ref: "summary",
      text: resume.summary.trim(),
      label: resume.summary.trim(),
    });
  }
  return targets;
}

interface AtAnalysisState<T> {
  status: "idle" | "loading" | "ready" | "error";
  result: T | null;
  error?: string;
}

const IDLE_ATS: AtAnalysisState<JobSpecificATSResult> = { status: "idle", result: null };

/**
 * Copilot panel: picks an operation, runs it against the local backend, and
 * renders grounded suggestions with evidence. Suggestions are never stored;
 * Apply/Undo only tracks an in-memory selection for the current session.
 */
export function CopilotPanel({ resume, jobMatch, onResumeChange }: CopilotPanelProps) {
  const targets = useMemo(() => bulletTargets(resume), [resume]);

  const [operation, setOperation] = useState<CopilotOperation>("identify_priorities");
  const [bulletRef, setBulletRef] = useState<string>("");
  const [customText, setCustomText] = useState<string>("");
  const [findingRule, setFindingRule] = useState<string>("");
  const [freeRequest, setFreeRequest] = useState<string>("");
  const [freeTargetRef, setFreeTargetRef] = useState<string>("");
  const [freeValidation, setFreeValidation] = useState<string | undefined>(undefined);

  const [ats, setAts] = useState<AtAnalysisState<ATSReadinessResult>>({
    status: "loading",
    result: null,
  });
  const [jobAts, setJobAts] = useState<AtAnalysisState<JobSpecificATSResult>>(IDLE_ATS);

  const [status, setStatus] = useState<CopilotStatus | null>(null);
  const [suggestions, setSuggestions] = useState<CopilotSuggestion[]>([]);
  const [clarification, setClarification] = useState<CopilotClarification | null>(null);
  const [explanation, setExplanation] = useState<string>("");
  const [providerLabel, setProviderLabel] = useState<string>("");
  const [fallbackUsed, setFallbackUsed] = useState(false);
  const [version, setVersion] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [appliedEdits, setAppliedEdits] = useState<Record<string, AppliedEditRecord>>({});
  const [dismissedEdits, setDismissedEdits] = useState<Set<string>>(new Set());

  const submittedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    getCopilotStatus()
      .then((next) => {
        if (!cancelled) setStatus(next);
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setAts({ status: "loading", result: null });
    });
    analyzeAtsReadiness(resume)
      .then((result) => {
        if (cancelled) return;
        setAts({ status: "ready", result });
        if (result.findings.length > 0) {
          setFindingRule((prev) => (prev ? prev : result.findings[0].rule_id));
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setAts({ status: "error", result: null, error: apiErrorMessage(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [resume]);

  useEffect(() => {
    let cancelled = false;
    if (!jobMatch) {
      queueMicrotask(() => {
        if (cancelled) return;
        setJobAts(IDLE_ATS);
      });
      return () => {
        cancelled = true;
      };
    }
    const { job } = jobMatch;
    queueMicrotask(() => {
      if (cancelled) return;
      setJobAts({ status: "loading", result: null });
    });
    analyzeJobSpecificAts(resume, job)
      .then((result) => {
        if (!cancelled) setJobAts({ status: "ready", result });
      })
      .catch((err) => {
        if (!cancelled) {
          setJobAts({ status: "error", result: null, error: apiErrorMessage(err) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [resume, jobMatch]);

  const meta = useMemo(
    () => OPERATIONS.find((item) => item.value === operation) ?? OPERATIONS[0],
    [operation]
  );

  const findings = useMemo(() => {
    if (meta.value !== "explain_finding") return [];
    const list: { source: string; rule_id: string; title: string }[] = [];
    if (ats.status === "ready" && ats.result) {
      for (const f of ats.result.findings) {
        list.push({ source: "ATS readiness", rule_id: f.rule_id, title: f.title });
      }
    }
    if (jobAts.status === "ready" && jobAts.result) {
      for (const f of jobAts.result.findings) {
        list.push({ source: "Job ATS", rule_id: f.rule_id, title: f.title });
      }
    }
    return list;
  }, [meta.value, ats, jobAts]);

  const selectedTarget = targets.find((target) => target.ref === bulletRef);
  const selectedFreeTarget = targets.find((target) => target.ref === freeTargetRef);

  const bulletReady =
    meta.value !== "improve_bullet" ||
    Boolean(selectedTarget) ||
    Boolean(customText.trim());
  const explainReady =
    meta.value !== "explain_finding" || (findings.length > 0 && Boolean(findingRule));
  const runnable = (!meta.needsJob || Boolean(jobMatch)) && bulletReady && explainReady;

  const performSubmit = useCallback(async (request: CopilotSuggestRequest) => {
    if (submittedRef.current) return;
    submittedRef.current = true;
    setRunning(true);
    setError(undefined);
    setSuggestions([]);
    setClarification(null);
    setExplanation("");
    setAppliedEdits({});
    setDismissedEdits(new Set());
    try {
      const response = await suggestCopilot(request);
      setSuggestions(response.suggestions);
      setClarification(response.clarification ?? null);
      setExplanation(response.explanation);
      setProviderLabel(response.provider.provider_label);
      setFallbackUsed(response.provider.fallback_used);
      setVersion(response.provider.version);
    } catch (err) {
      setError(apiErrorMessage(err));
      setSuggestions([]);
      setClarification(null);
      setExplanation("");
    } finally {
      submittedRef.current = false;
      setRunning(false);
    }
  }, []);

  const buildAnalysis = useCallback((): CopilotAnalysisContext => {
    return {
      ats_readiness: ats.status === "ready" ? ats.result : null,
      job_specific_ats: jobAts.status === "ready" ? jobAts.result : null,
      deterministic_match: jobMatch?.result ?? null,
    };
  }, [ats, jobAts, jobMatch]);

  const handleSubmit = useCallback(() => {
    if (!runnable || submittedRef.current) return;
    const request: CopilotSuggestRequest = {
      operation,
      resume,
      job_description: jobMatch?.job,
      analysis: buildAnalysis(),
      target_ref: bulletRef || undefined,
      target_text: selectedTarget ? selectedTarget.text : customText || undefined,
      finding_ref: findingRule || undefined,
    };
    void performSubmit(request);
  }, [
    runnable,
    operation,
    resume,
    jobMatch,
    buildAnalysis,
    bulletRef,
    selectedTarget,
    customText,
    findingRule,
    performSubmit,
  ]);

  const handleAskCopilot = useCallback(() => {
    const userRequest = freeRequest.trim();
    if (!userRequest) {
      setFreeValidation("Enter a request first.");
      return;
    }
    if (userRequest.length > MAX_FREE_REQUEST_CHARS) {
      setFreeValidation(
        `Keep your request under ${MAX_FREE_REQUEST_CHARS} characters.`
      );
      return;
    }
    setFreeValidation(undefined);
    const request: CopilotSuggestRequest = {
      operation: "free-form",
      resume,
      job_description: jobMatch?.job,
      analysis: buildAnalysis(),
      target_ref: freeTargetRef || undefined,
      target_text: selectedFreeTarget?.text ?? undefined,
      user_request: userRequest,
    };
    void performSubmit(request);
  }, [
    freeRequest,
    freeTargetRef,
    selectedFreeTarget,
    resume,
    jobMatch,
    buildAnalysis,
    performSubmit,
  ]);

  const handleClarify = useCallback(
    (target: CopilotEditTarget) => {
      const userRequest = freeRequest.trim();
      if (!userRequest) return;
      setClarification(null);
      setFreeTargetRef(target.path);
      const request: CopilotSuggestRequest = {
        operation: "free-form",
        resume,
        job_description: jobMatch?.job,
        analysis: buildAnalysis(),
        target_ref: target.path,
        target_text: selectedFreeTarget?.text ?? undefined,
        user_request: userRequest,
      };
      void performSubmit(request);
    },
    [freeRequest, resume, jobMatch, buildAnalysis, selectedFreeTarget, performSubmit]
  );

  const handleApplyEdit = useCallback(
    (suggestion: CopilotSuggestion) => {
      const edit = suggestion.edit;
      if (!edit) return;
      const updated = applyEdit(resume, edit.target, edit.proposed_value ?? "");
      if (!updated) {
        setError(
          "This edit can no longer be applied. Regenerate suggestions and try again."
        );
        return;
      }
      setError(undefined);
      onResumeChange?.(updated);
      setAppliedEdits((prev) => ({
        ...prev,
        [suggestion.id]: {
          editId: edit.edit_id,
          originalValue: edit.original_value ?? "",
          target: edit.target,
        },
      }));
    },
    [resume, onResumeChange]
  );

  const handleRevertEdit = useCallback(
    (suggestion: CopilotSuggestion) => {
      const record = appliedEdits[suggestion.id];
      if (!record) return;
      const updated = applyEdit(resume, record.target, record.originalValue);
      if (!updated) {
        setError("This edit could not be reverted.");
        return;
      }
      setError(undefined);
      onResumeChange?.(updated);
      setAppliedEdits((prev) => {
        const next = { ...prev };
        delete next[suggestion.id];
        return next;
      });
    },
    [appliedEdits, resume, onResumeChange]
  );

  const handleDismissEdit = useCallback((id: string) => {
    setDismissedEdits((prev) => new Set(prev).add(id));
  }, []);

  const needsJobBlocked = meta.value === "job_alignment" && !jobMatch;
  const explainBlocked = meta.value === "explain_finding" && findings.length === 0;
  const bulletBlocked = meta.value === "improve_bullet" && targets.length === 0;

  return (
    <section
      aria-labelledby="copilot-heading"
      aria-busy={running}
      className="space-y-4"
    >
      <div>
        <h3
          id="copilot-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          AI Copilot
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Get grounded, evidence-checked help with your resume. Generation runs
          on a local model; suggestions are reviewed before they can be used
          and nothing is stored.
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <label
          htmlFor="copilot-operation"
          className="text-xs font-medium uppercase tracking-wide text-slate-500"
        >
          Operation
        </label>
        <select
          id="copilot-operation"
          value={operation}
          onChange={(event) => {
            const next = event.target.value as CopilotOperation;
            setOperation(next);
            setError(undefined);
            setSuggestions([]);
            setClarification(null);
            setExplanation("");
            setAppliedEdits({});
            setDismissedEdits(new Set());
          }}
          className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
        >
          {OPERATIONS.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-slate-400">{meta.hint}</p>

        {meta.value === "improve_bullet" && targets.length > 0 ? (
          <div className="mt-4">
            <label
              htmlFor="copilot-bullet"
              className="text-xs font-medium uppercase tracking-wide text-slate-500"
            >
              Target bullet
            </label>
            <select
              id="copilot-bullet"
              value={bulletRef}
              onChange={(event) => {
                setBulletRef(event.target.value);
                setError(undefined);
              }}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
            >
              <option key="__bullet_placeholder__" value="">
                Choose from your resume…
              </option>
              {targets.map((target) => (
                <option key={target.ref} value={target.ref}>
                  {target.text.length > 90 ? `${target.text.slice(0, 90)}…` : target.text}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-slate-400">
              Select one bullet, or clear the dropdown and paste custom text below.
            </p>
            <input
              type="text"
              value={customText}
              onChange={(event) => {
                setCustomText(event.target.value);
                setError(undefined);
              }}
              placeholder="…or paste a custom bullet"
              aria-label="Custom bullet text"
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
            />
          </div>
        ) : null}

        {meta.value === "explain_finding" ? (
          <div className="mt-4">
            <label
              htmlFor="copilot-finding"
              className="text-xs font-medium uppercase tracking-wide text-slate-500"
            >
              Finding to explain
            </label>
            {explainBlocked ? (
              <p className="mt-2 text-sm text-slate-500">
                No ATS findings were reported for this resume.
              </p>
            ) : (
              <select
                id="copilot-finding"
                value={findingRule}
                onChange={(event) => setFindingRule(event.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
              >
                {findings.map((finding) => (
                  <option
                    key={`${finding.source}:${finding.rule_id}`}
                    value={finding.rule_id}
                  >
                    {finding.title} ({finding.source})
                  </option>
                ))}
              </select>
            )}
          </div>
        ) : null}

        {ats.status === "loading" || (jobMatch && jobAts.status === "loading") ? (
          <div
            role="status"
            aria-live="polite"
            className="mt-4 flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600"
          >
            <div
              className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600"
              aria-hidden="true"
            />
            Preparing the analysis grounding…
          </div>
        ) : null}

        {ats.status === "error" ? (
          <p role="alert" className="mt-4 text-sm text-amber-700">
            ATS analysis unavailable: {ats.error}. Suggestions fall back to your
            resume alone.
          </p>
        ) : null}

        {jobMatch && jobAts.status === "error" ? (
          <p role="alert" className="mt-2 text-sm text-amber-700">
            Job-specific analysis unavailable: {jobAts.error}
          </p>
        ) : null}

        {status && !status.fallback_available && (
          <p role="alert" className="mt-4 text-sm text-amber-700">
            {status.providers[0]?.note || "No provider is currently available."}
          </p>
        )}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={!runnable || running || needsJobBlocked || bulletBlocked}
          className="mt-4 w-full rounded-xl bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-indigo-600"
        >
          {running ? "Generating…" : "Generate suggestions"}
        </button>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <label
          htmlFor="copilot-free-request"
          className="text-xs font-medium uppercase tracking-wide text-slate-500"
        >
          Ask Copilot anything
        </label>
        <textarea
          id="copilot-free-request"
          value={freeRequest}
          onChange={(event) => {
            setFreeRequest(event.target.value);
            setFreeValidation(undefined);
            setClarification(null);
          }}
          placeholder="e.g. Make my summary more concise, or rewrite my first bullet to sound stronger"
          aria-label="Free-text request for Copilot"
          rows={3}
          className="mt-1 w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
        />
        <div className="mt-1 flex items-center justify-between gap-2">
          <p className="text-xs text-slate-400">
            Tips, rewrites, or review requests in your own words.
          </p>
          <span className="text-xs text-slate-400">
            {freeRequest.length}/{MAX_FREE_REQUEST_CHARS}
          </span>
        </div>
        {freeValidation ? (
          <p role="alert" className="mt-1 text-xs text-red-700">
            {freeValidation}
          </p>
        ) : null}
        {targets.length > 0 ? (
          <div className="mt-4">
            <label
              htmlFor="copilot-free-target"
              className="text-xs font-medium uppercase tracking-wide text-slate-500"
            >
              Act on (optional)
            </label>
            <select
              id="copilot-free-target"
              value={freeTargetRef}
              onChange={(event) => {
                setFreeTargetRef(event.target.value);
                setFreeValidation(undefined);
              }}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
            >
              <option key="__free_placeholder__" value="">
                Let Copilot detect the target
              </option>
              {targets.map((target) => (
                <option key={`free:${target.ref}`} value={target.ref}>
                  {target.label.length > 90
                    ? `${target.label.slice(0, 90)}…`
                    : target.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-slate-400">
              Copilot usually spots the text your request is about. Pick an
              explicit target only when you want to override that and force the
              edit onto a specific section.
            </p>
          </div>
        ) : null}
        <button
          type="button"
          onClick={handleAskCopilot}
          disabled={running}
          className="mt-4 w-full rounded-xl bg-slate-900 px-6 py-3 text-base font-semibold text-white shadow-sm transition-colors hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-slate-900"
        >
          {running ? "Generating…" : "Ask Copilot"}
        </button>
      </div>

      {needsJobBlocked ? (
        <div
          role="alert"
          className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600"
        >
          Job alignment requires a job description. Run a match first, or pick
          a different operation.
        </div>
      ) : null}

      {error ? (
        <div
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {error}
        </div>
      ) : null}

      {clarification ? (
        <ClarificationCard
          clarification={clarification}
          onSelect={handleClarify}
        />
      ) : suggestions.length > 0 ? (
        <OutputStage
          explanation={explanation}
          suggestions={suggestions}
          providerLabel={providerLabel}
          fallbackUsed={fallbackUsed}
          version={version}
          appliedEdits={appliedEdits}
          dismissedEdits={dismissedEdits}
          onApplyEdit={handleApplyEdit}
          onRevertEdit={handleRevertEdit}
          onDismissEdit={handleDismissEdit}
        />
      ) : null}
    </section>
  );
}

/**
 * Target-clarification card: shown when a free-form edit request is ambiguous
 * (e.g. several projects share the named technology). Each option resubmits
 * the SAME request with that target as an explicit target_ref, feeding the
 * normal fact-safe EditProposal flow. The Copilot never guesses a project.
 */
function ClarificationCard({
  clarification,
  onSelect,
}: {
  clarification: CopilotClarification;
  onSelect: (target: CopilotEditTarget) => void;
}) {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 shadow-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="amber">Which target?</Badge>
        <p className="text-sm font-semibold text-amber-900">{clarification.question}</p>
      </div>
      {clarification.reason ? (
        <p className="mt-2 text-xs text-amber-800">{clarification.reason}</p>
      ) : null}
      <ul className="mt-3 space-y-2">
        {clarification.options.map((option, index) => (
          <li key={`${option.target.path}:${index}`}>
            <button
              type="button"
              onClick={() => onSelect(option.target)}
              className="w-full rounded-lg border border-amber-300 bg-white px-4 py-2 text-left text-sm text-slate-800 transition-colors hover:border-amber-400 hover:bg-amber-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:ring-offset-2"
            >
              {option.label}
            </button>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-amber-800">
        Choosing a project rewrites only that project&apos;s description as a
        fact-safe proposal you can review before applying. Nothing is changed
        until you apply it.
      </p>
    </div>
  );
}

function OutputStage({
  explanation,
  suggestions,
  providerLabel,
  fallbackUsed,
  version,
  appliedEdits,
  dismissedEdits,
  onApplyEdit,
  onRevertEdit,
  onDismissEdit,
}: {
  explanation: string;
  suggestions: CopilotSuggestion[];
  providerLabel: string;
  fallbackUsed: boolean;
  version: string;
  appliedEdits: Record<string, AppliedEditRecord>;
  dismissedEdits: Set<string>;
  onApplyEdit: (suggestion: CopilotSuggestion) => void;
  onRevertEdit: (suggestion: CopilotSuggestion) => void;
  onDismissEdit: (id: string) => void;
}) {
  const verifiedCount = suggestions.filter((item) => item.verification === "verified").length;
  const visible = suggestions.filter((item) => !dismissedEdits.has(item.id));
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-semibold text-slate-800">
            {explanation}
          </p>
          <span className="text-xs text-slate-500">
            {providerLabel}
            {fallbackUsed ? " · fallback" : ""} · {version}
          </span>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {verifiedCount} of {suggestions.length} suggestions verified against
          the resume. Review everything before changing your resume. Nothing is
          applied until you choose to apply it, and edits are reversible.
        </p>
      </div>

      <ul className="space-y-4">
        {visible.map((suggestion) => (
          <SuggestionCard
            key={suggestion.id}
            suggestion={suggestion}
            appliedEdit={appliedEdits[suggestion.id]}
            onApplyEdit={() => onApplyEdit(suggestion)}
            onRevertEdit={() => onRevertEdit(suggestion)}
            onDismissEdit={() => onDismissEdit(suggestion.id)}
          />
        ))}
      </ul>
    </div>
  );
}

function SuggestionCard({
  suggestion,
  appliedEdit,
  onApplyEdit,
  onRevertEdit,
  onDismissEdit,
}: {
  suggestion: CopilotSuggestion;
  appliedEdit?: AppliedEditRecord;
  onApplyEdit: () => void;
  onRevertEdit: () => void;
  onDismissEdit: () => void;
}) {
  const changed = suggestion.original_text !== suggestion.suggested_text;
  const edit = suggestion.edit ?? null;
  return (
    <li className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="slate">{suggestion.category}</Badge>
          <Badge tone={verificationTone(suggestion.verification)}>
            {VERIFICATION_LABEL[suggestion.verification] ?? suggestion.verification}
          </Badge>
          {suggestion.requires_user_confirmation ? <Badge tone="amber">Review needed</Badge> : null}
          {suggestion.priority !== null && suggestion.priority !== undefined ? (
            <Badge tone="indigo">Priority {suggestion.priority}</Badge>
          ) : null}
        </div>
        {edit ? (
          <button
            type="button"
            onClick={onDismissEdit}
            className="shrink-0 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            Dismiss
          </button>
        ) : (
          <Badge tone="slate">Advice only</Badge>
        )}
      </div>

      {edit ? (
        <EditProposal
          edit={edit}
          applied={Boolean(appliedEdit)}
          onApply={onApplyEdit}
          onRevert={onRevertEdit}
        />
      ) : suggestion.suggested_text ? (
        <div className="mt-4 space-y-3">
          {suggestion.original_text ? (
            <TextBlock label="Original" text={suggestion.original_text} muted={!changed} />
          ) : null}
          <TextBlock label="Suggested" text={suggestion.suggested_text} accent={changed} />
        </div>
      ) : null}

      {suggestion.issue ? (
        <p className="mt-3 text-sm leading-relaxed text-slate-700">
          <span className="font-medium text-slate-800">Issue: </span>
          {suggestion.issue}
        </p>
      ) : null}

      <p className="mt-2 text-sm leading-relaxed text-slate-700">
        {suggestion.rationale}
      </p>

      {suggestion.recommendation ? (
        <p className="mt-2 rounded-lg bg-slate-50 px-3 py-2 text-sm leading-relaxed text-slate-700">
          <span className="font-medium text-slate-800">Recommendation: </span>
          {suggestion.recommendation}
          {suggestion.impact ? (
            <span className="text-slate-400"> (expected impact: {suggestion.impact})</span>
          ) : null}
        </p>
      ) : null}

      {suggestion.evidence.length > 0 ? (
        <ul className="mt-3 space-y-1.5">
          {suggestion.evidence.map((item, index) => (
            <li key={index} className="flex gap-2 text-xs text-slate-500">
              <span aria-hidden="true" className="mt-0.5 text-indigo-400">
                •
              </span>
              <span>
                <span className="font-medium text-slate-600">{item.statement}</span>{" "}
                <span className="text-slate-400">— {item.source}</span>
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function EditProposal({
  edit,
  applied,
  onApply,
  onRevert,
}: {
  edit: CopilotEditProposal;
  applied: boolean;
  onApply: () => void;
  onRevert: () => void;
}) {
  const [preview, setPreview] = useState(false);
  const requiresConfirmation =
    edit.status === "unverified" || edit.validation.requires_user_confirmation;
  const failedChecks = edit.validation.checks.filter((check) => !check.passed);

  return (
    <div className="mt-4 rounded-lg border border-indigo-100 bg-indigo-50/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={requiresConfirmation ? "amber" : "emerald"}>
            {requiresConfirmation ? "Needs your confirmation" : "Ready to apply"}
          </Badge>
          <span className="text-xs font-medium text-slate-600">
            {editTargetLabel(edit.target)}
          </span>
        </div>
        {applied ? <Badge tone="emerald">Applied</Badge> : null}
      </div>

      {edit.reason ? (
        <p className="mt-2 text-xs leading-relaxed text-slate-500">{edit.reason}</p>
      ) : null}

      <div className="mt-3 space-y-3">
        <TextBlock label="Current" text={edit.original_value ?? ""} muted />
        <TextBlock label="Proposed" text={edit.proposed_value ?? ""} accent />
      </div>

      <button
        type="button"
        onClick={() => setPreview((value) => !value)}
        aria-expanded={preview}
        className="mt-3 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
      >
        {preview ? "Hide checks" : "Preview checks"}
      </button>

      {preview ? (
        <ul className="mt-3 space-y-1.5">
          {edit.validation.checks.map((check) => (
            <li key={check.category} className="text-xs text-slate-600">
              <span
                className={`font-semibold ${
                  check.passed ? "text-emerald-700" : "text-amber-700"
                }`}
              >
                {check.passed ? "Passed" : "Failed"} · {check.category}
              </span>
              {check.detail ? (
                <span className="text-slate-500"> — {check.detail}</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      {failedChecks.length > 0 ? (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
          This change introduces information the fact checks could not trace to
          your resume. Apply it only if it is genuinely true.
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap gap-2">
        {applied ? (
          <button
            type="button"
            onClick={onRevert}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            Revert
          </button>
        ) : (
          <button
            type="button"
            onClick={onApply}
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            {requiresConfirmation ? "Apply anyway" : "Apply to resume"}
          </button>
        )}
      </div>

      {applied ? (
        <p className="mt-3 text-xs font-medium text-emerald-700">
          Applied in this session only — use Revert to undo. Nothing is saved.
        </p>
      ) : null}
    </div>
  );
}

function TextBlock({
  label,
  text,
  muted = false,
  accent = false,
}: {
  label: string;
  text: string;
  muted?: boolean;
  accent?: boolean;
}) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p
        className={`mt-1 whitespace-pre-wrap rounded-lg border px-3 py-2 text-sm leading-relaxed ${
          accent
            ? "border-indigo-200 bg-indigo-50/50 text-slate-800"
            : muted
              ? "border-slate-200 bg-white text-slate-400 line-through"
              : "border-slate-200 bg-white text-slate-700"
        }`}
      >
        {text}
      </p>
    </div>
  );
}

function Badge({ tone, children }: { tone: string; children: React.ReactNode }) {
  const tones: Record<string, string> = {
    slate: "bg-slate-100 text-slate-600",
    indigo: "bg-indigo-50 text-indigo-700",
    emerald: "bg-emerald-50 text-emerald-700",
    amber: "bg-amber-50 text-amber-800",
    red: "bg-red-50 text-red-700",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${tones[tone] ?? tones.slate}`}
    >
      {children}
    </span>
  );
}

function verificationTone(verification: string): string {
  switch (verification) {
    case "verified":
      return "emerald";
    case "inferred":
      return "indigo";
    case "unverified":
      return "amber";
    default:
      return "slate";
  }
}