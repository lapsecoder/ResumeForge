"use client";

import { useEffect, useState } from "react";

const STEPS = ["Uploading resume…", "Extracting resume…", "Structuring resume…"] as const;

export function AnalysisProgress() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setIndex((current) => (current + 1) % STEPS.length);
    }, 1600);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex w-full flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white px-6 py-14 text-center shadow-sm"
    >
      <div
        className="h-10 w-10 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600"
        aria-hidden="true"
      />
      <p className="mt-5 text-base font-medium text-slate-800">{STEPS[index]}</p>
      <p className="mt-1 text-sm text-slate-500">
        This usually takes a few seconds. Your resume is processed temporarily and not stored.
      </p>
    </div>
  );
}