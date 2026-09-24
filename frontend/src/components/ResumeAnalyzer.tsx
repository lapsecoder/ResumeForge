"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { apiErrorMessage, parseResume } from "@/lib/api";
import { analyzeRoleCompatibility } from "@/lib/roleAnalysis";
import type { RoleAnalysisResult } from "@/lib/roleAnalysis";
import type { RoleInfo } from "@/lib/roleAnalysis";
import { useHistory } from "@/lib/history";
import type { HybridMatchResult, JobDescription } from "@/lib/matcher";
import type { Application } from "@/lib/applications";
import type { Resume } from "@/lib/resume";
import { validateResumeFile } from "@/lib/upload";

import { AnalysisProgress } from "./AnalysisProgress";
import { ApplicationTracker } from "./ApplicationTracker";
import { DEFAULT_TEMPLATE_ID, type TemplateId } from "./templates";
import { CopilotPanel } from "./CopilotPanel";
import { FilePreviewCard } from "./FilePreviewCard";
import { MatchFlow } from "./MatchFlow";
import { PrivacyNote } from "./PrivacyNote";
import { ResumeBuilder } from "./ResumeBuilder";
import { ResultsView } from "./ResultsView";
import { RoleCompatibilityView } from "./RoleCompatibilityView";
import { SearchableRoleSelector } from "./SearchableRoleSelector";
import { UploadZone, type UploadZoneHandle } from "./UploadZone";

type View = "upload" | "parsing" | "results";
type ResultTab = "overview" | "builder" | "applications";

interface JobMatchData {
  job: JobDescription;
  result: HybridMatchResult;
}

export default function ResumeAnalyzer() {
  const [view, setView] = useState<View>("upload");
  const [file, setFile] = useState<File | null>(null);
  // The parsed resume is the reset target; the working copy (with bounded
  // undo/redo history) is what the builder, matching, and Copilot operate on.
  // It is never persisted, and refreshing the page discards it.
  const [originalResume, setOriginalResume] = useState<Resume | null>(null);
  const {
    present: workingResume,
    commit: commitWorkingResume,
    undo: undoWorkingResume,
    redo: redoWorkingResume,
    reset: resetWorkingResume,
    canUndo,
    canRedo,
  } = useHistory<Resume | null>(null);
  const [resultTab, setResultTab] = useState<ResultTab>("overview");
  const [applications, setApplications] = useState<Application[]>([]);
  const [error, setError] = useState<string | undefined>(undefined);
  const [, setMatchJobFile] = useState<string | null>(null);
  const [jobMatch, setJobMatch] = useState<JobMatchData | null>(null);
  const [templateId, setTemplateId] = useState<TemplateId>(DEFAULT_TEMPLATE_ID);
  const [roleTitle, setRoleTitle] = useState<string>("");
  const [selectedRole, setSelectedRole] = useState<RoleInfo | null>(null);
  const [roleResult, setRoleResult] = useState<RoleAnalysisResult | null>(null);
  const [roleLoading, setRoleLoading] = useState(false);

  const uploadZoneRef = useRef<UploadZoneHandle>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const submittedRef = useRef(false);

  useEffect(() => {
    if (view === "results") {
      resultsRef.current?.focus();
    }
  }, [view]);

  const acceptFile = useCallback((next: File) => {
    const validation = validateResumeFile(next);
    if (validation.ok) {
      setFile(next);
      setError(undefined);
    } else {
      setFile(null);
      setError(validation.message);
    }
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file || submittedRef.current) return;
    submittedRef.current = true;
    setView("parsing");
    setError(undefined);
    try {
      const result = await parseResume(file);
      setOriginalResume(result);
      resetWorkingResume(result);
      setResultTab("overview");
      setFile(null);
      setView("results");
    } catch (err) {
      setError(apiErrorMessage(err));
      setView("upload");
    } finally {
      submittedRef.current = false;
    }
  }, [file, resetWorkingResume]);

  const handleReset = useCallback(() => {
    setFile(null);
    setOriginalResume(null);
    resetWorkingResume(null);
    setResultTab("overview");
    setError(undefined);
    setMatchJobFile(null);
    setJobMatch(null);
    setApplications([]);
    setRoleTitle("");
    setSelectedRole(null);
    setRoleResult(null);
    setRoleLoading(false);
    setView("upload");
  }, [resetWorkingResume]);

  const handleResetChanges = useCallback(() => {
    if (originalResume) resetWorkingResume(originalResume);
  }, [originalResume, resetWorkingResume]);

  const handleAnalyzeRole = useCallback(async () => {
    if (!workingResume || !roleTitle.trim() || roleLoading) return;
    setRoleResult(null);
    setRoleLoading(true);
    setError(undefined);
    try {
      const result = await analyzeRoleCompatibility(roleTitle.trim(), workingResume);
      setRoleResult(result);
    } catch (err) {
      setError(apiErrorMessage(err));
      setRoleResult(null);
    } finally {
      setRoleLoading(false);
    }
  }, [workingResume, roleTitle, roleLoading]);

  const handleResetRole = useCallback(() => {
    setRoleResult(null);
    setRoleTitle("");
    setSelectedRole(null);
    setError(undefined);
  }, []);

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-10 sm:py-14">
      <header className="text-center">
        <p className="text-sm font-semibold uppercase tracking-widest text-indigo-600">
          ResumeForge
        </p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
          Analyze your resume
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-base text-slate-600">
          Upload a PDF, DOCX, or TXT resume and see the parsed structure in seconds.
        </p>
      </header>

      <main className="mt-8" aria-live="polite">
        {view !== "results" ? (
          <div className="space-y-4">
            <UploadZone
              ref={uploadZoneRef}
              onFileAccepted={acceptFile}
              label="Choose a resume file to analyze"
            />

            {error ? (
              <div
                role="alert"
                className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                {error}
              </div>
            ) : null}

            {view === "parsing" ? <AnalysisProgress /> : null}

            {view === "upload" && file ? (
              <>
                <FilePreviewCard
                  file={file}
                  key={file.name}
                  onRemove={() => {
                    setFile(null);
                    setError(undefined);
                  }}
                  onReplace={() => uploadZoneRef.current?.replaceFile()}
                />
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={!file || view !== "upload"}
                  className="w-full rounded-xl bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-indigo-600"
                >
                  Analyze resume
                </button>
              </>
            ) : null}
          </div>
        ) : (
          workingResume && (
            <div className="space-y-4">
              <div
                ref={resultsRef}
                tabIndex={-1}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm outline-none"
              >
                <span className="font-medium text-slate-800">Analysis complete.</span> Your
                resume was processed temporarily and nothing was stored.
              </div>
              <div
                role="tablist"
                aria-label="Resume view"
                className="flex gap-2 rounded-xl border border-slate-200 bg-white p-1.5 shadow-sm"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={resultTab === "overview"}
                  onClick={() => setResultTab("overview")}
                  className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                    resultTab === "overview"
                      ? "bg-indigo-600 text-white"
                      : "bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  Structured view
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={resultTab === "builder"}
                  onClick={() => setResultTab("builder")}
                  className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                    resultTab === "builder"
                      ? "bg-indigo-600 text-white"
                      : "bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  Resume builder
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={resultTab === "applications"}
                  onClick={() => setResultTab("applications")}
                  className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                    resultTab === "applications"
                      ? "bg-indigo-600 text-white"
                      : "bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  Applications
                </button>
              </div>

          {resultTab === "overview" ? (
            <ResultsView resume={workingResume} />
          ) : resultTab === "builder" ? (
            <ResumeBuilder
              resume={workingResume}
              canUndo={canUndo}
              canRedo={canRedo}
              onCommit={commitWorkingResume}
              onUndo={undoWorkingResume}
              onRedo={redoWorkingResume}
              onReset={handleResetChanges}
              templateId={templateId}
              onTemplateChange={setTemplateId}
            />
          ) : (
            <ApplicationTracker
              applications={applications}
              onChange={setApplications}
              prefill={{
                job: jobMatch?.job ?? null,
                match: jobMatch?.result ?? null,
                resumeTemplate: templateId,
              }}
            />
        )}

              <div className="border-t border-slate-200 pt-6">
                {roleResult ? (
                  <div className="space-y-4">
                    <RoleCompatibilityView result={roleResult} />
                    <button
                      type="button"
                      onClick={handleResetRole}
                      className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
                    >
                      ← Change Target Role
                    </button>
                  </div>
                ) : (
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-indigo-600">
                  Target job role
                </h3>
                <p className="mt-1 text-sm text-slate-500">
                  Type a role to analyze compatibility. No need for a job
                  description — it is optional.
                </p>
                <div className="mt-3">
                  <SearchableRoleSelector
                    value={roleTitle}
                    onChange={setRoleTitle}
                    onSelect={(role) => setSelectedRole(role)}
                    onClear={() => setSelectedRole(null)}
                    placeholder="Search for a job role..."
                    disabled={roleLoading}
                    error={error && roleResult === null ? error : undefined}
                  />
                </div>
                <button
                  type="button"
                  onClick={handleAnalyzeRole}
                  disabled={
                    !workingResume || !roleTitle.trim() || roleLoading
                  }
                  className="mt-3 w-full rounded-xl bg-indigo-600 px-6 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-indigo-600"
                >
                  {roleLoading ? "Analyzing…" : "Analyze role"}
                </button>
              </div>
            )}
          </div>
              <div className="border-t border-slate-200 pt-6">
                <MatchFlow
                  resume={workingResume}
                  onJobFileChange={setMatchJobFile}
                  onJobMatched={(job, result) => setJobMatch({ job, result })}
                  onJobCleared={() => setJobMatch(null)}
                />
              </div>
              <div className="border-t border-slate-200 pt-6">
                <CopilotPanel
                  resume={workingResume}
                  jobMatch={jobMatch}
                  onResumeChange={commitWorkingResume}
                />
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="w-full rounded-xl border border-slate-300 bg-white px-6 py-3 text-base font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
              >
                Analyze another resume
              </button>
            </div>
          )
        )}
      </main>

      <PrivacyNote />
    </div>
  );
}