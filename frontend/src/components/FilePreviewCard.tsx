"use client";

import { formatFileSize, type AllowedExtension } from "@/lib/upload";

interface FilePreviewCardProps {
  file: File;
  onRemove: () => void;
  onReplace: () => void;
}

export function FilePreviewCard({ file, onRemove, onReplace }: FilePreviewCardProps) {
  const extension = (/\.([^.]+)$/.exec(file.name)?.[1] ?? "").toLowerCase() as
    | AllowedExtension
    | "";

  return (
    <div
      role="region"
      aria-label="Selected file"
      className="flex w-full items-center gap-3 rounded-xl border border-slate-200 bg-white p-3 shadow-sm"
    >
      <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-xs font-bold uppercase tracking-wide text-indigo-700">
        {extension}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-slate-800">{file.name}</p>
        <p className="text-xs text-slate-500">{formatFileSize(file.size)}</p>
      </div>
      <button
        type="button"
        onClick={onReplace}
        className="inline-flex items-center rounded-lg px-3 py-1.5 text-sm font-medium text-indigo-600 transition-colors hover:bg-indigo-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        Replace
      </button>
      <button
        type="button"
        onClick={onRemove}
        aria-label="Remove selected file"
        className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          className="h-4 w-4"
          aria-hidden="true"
          focusable="false"
        >
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}