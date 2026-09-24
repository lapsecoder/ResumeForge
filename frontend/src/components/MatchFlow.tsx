"use client";

import { useCallback, useRef, useState } from "react";

import { apiErrorMessage } from "@/lib/api";
import {
  matchHybrid,
  parseJobDescription,
  type HybridMatchResult,
  type JobDescription,
} from "@/lib/matcher";
import type { Resume } from "@/lib/resume";
import { validateResumeFile } from "@/lib/upload";

import { MatchResultView } from "./MatchResultView";

type MatchPhase = "idle" | "parsing" | "matching" | "done";

interface MatchFlowProps {
  resume: Resume;
  onJobFileChange?: (name: string | null) => void;
  onJobMatched?: (job: JobDescription, result: HybridMatchResult) => void;
  onJobCleared?: () => void;
}

/**
 * Optional matching step after a resume is parsed: upload a job description,
 * have it parsed, then run the transient hybrid match and show the result.
 */
export function MatchFlow({ resume, onJobFileChange, onJobMatched, onJobCleared }: MatchFlowProps) {
  const [phase, setPhase] = useState<MatchPhase>("idle");
  const [jobFile, setJobFile] = useState<File | null>(null);
  const [error, setError] = useState<string | undefined>(undefined);
  const [result, setResult] = useState<HybridMatchResult | null>(null);
  const submittedRef = useRef(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptJobFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const validation = validateResumeFile(file);
      if (!validation.ok) {
        setJobFile(null);
        onJobFileChange?.(null);
        onJobCleared?.();
        setError(validation.message);
        return;
      }
      setJobFile(file);
      onJobFileChange?.(file.name);
      setError(undefined);
    },
    [onJobFileChange, onJobCleared]
  );

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    acceptJobFile(event.target.files?.[0]);
    if (event.target) event.target.value = "";
  };

  const handleRun = useCallback(async () => {
    if (!jobFile || submittedRef.current) return;
    submittedRef.current = true;
    setPhase("parsing");
    setError(undefined);
    try {
      const job = await parseJobDescription(jobFile);
      setPhase("matching");
      const matched = await matchHybrid(resume, job);
      setResult(matched);
      setPhase("done");
      onJobMatched?.(job, matched);
    } catch (err) {
      setError(apiErrorMessage(err));
      setPhase("idle");
    } finally {
      submittedRef.current = false;
    }
  }, [jobFile, resume, onJobMatched]);

  const handleReset = useCallback(() => {
    setJobFile(null);
    setResult(null);
    setError(undefined);
    setPhase("idle");
    onJobFileChange?.(null);
    onJobCleared?.();
  }, [onJobFileChange, onJobCleared]);

  if (phase === "done" && result) {
    return (
      <div className="space-y-4">
        <MatchResultView result={result} />
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleReset}
            className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            Match against a different job
          </button>
        </div>
      </div>
    );
  }

  const busy = phase === "parsing" || phase === "matching";

  return (
    <section aria-labelledby="match-heading" className="space-y-4">
      <div>
        <h3
          id="match-heading"
          className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
        >
          Match against a job description
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Upload a PDF, DOCX, or TXT job description to see how it matches.
          Matching is transient and runs locally.
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        {jobFile ? (
          <div className="flex items-center justify-between gap-3">
            <p className="truncate text-sm text-slate-700">{jobFile.name}</p>
            <button
              type="button"
              onClick={() => {
                setJobFile(null);
                onJobFileChange?.(null);
                onJobCleared?.();
              }}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition-colors hover:bg-slate-50"
            >
              Replace
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-white px-6 py-8 text-center transition-colors hover:border-indigo-400 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.75}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-8 w-8 text-slate-400"
              aria-hidden="true"
            >
              <path d="M12 16V4m0 0 4 4m-4-4-4 4" />
              <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
            </svg>
            <span className="mt-3 text-sm font-medium text-slate-700">
              Choose a job description file
            </span>
            <span className="mt-1 text-xs text-slate-500">PDF, DOCX, TXT · up to 10 MB</span>
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleChange}
          className="sr-only"
          aria-label="Choose a job description file"
        />

        {error ? (
          <div
            role="alert"
            className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          >
            {error}
          </div>
        ) : null}

        {busy ? (
          <div
            role="status"
            aria-live="polite"
            className="mt-4 flex flex-col items-center justify-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-6 text-center"
          >
            <div
              className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600"
              aria-hidden="true"
            />
            <p className="text-sm font-medium text-slate-800">
              {phase === "parsing" ? "Parsing job description…" : "Running match…"}
            </p>
            <p className="text-xs text-slate-500">
              Match runs locally and is not stored.
            </p>
          </div>
        ) : jobFile ? (
          <button
            type="button"
            onClick={handleRun}
            disabled={busy}
            className="mt-4 w-full rounded-xl bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Run match
          </button>
        ) : null}
      </div>
    </section>
  );
}