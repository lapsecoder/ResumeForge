"use client";

import {
  forwardRef,
  useImperativeHandle,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";

import { ACCEPT_ATTR, ALLOWED_EXTENSIONS_LABEL, MAX_FILE_SIZE_MB } from "@/lib/upload";

export interface UploadZoneHandle {
  replaceFile: () => void;
}

interface UploadZoneProps {
  onFileAccepted: (file: File) => void;
  label: string;
}

/**
 * Click-to-browse, drag-and-drop, keyboard-accessible upload control.
 *
 * The dropzone is a <label> around a visually-hidden <input type="file">:
 * clicking it opens the picker natively, and the hidden input remains
 * focusable so keyboard users can activate it with Enter/Space.
 */
export const UploadZone = forwardRef<UploadZoneHandle, UploadZoneProps>(
  function UploadZone({ onFileAccepted, label }, ref) {
    const [isDragging, setIsDragging] = useState(false);
    const dragDepth = useRef(0);
    const labelRef = useRef<HTMLLabelElement>(null);

    useImperativeHandle(ref, () => ({
      replaceFile: () => labelRef.current?.click(),
    }));

    const handleFiles = (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onFileAccepted(files[0] as File);
    };

    const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
      handleFiles(event.target.files);
      // Allow re-selecting the same file (e.g. replacing the current one).
      event.target.value = "";
    };

    const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      dragDepth.current = 0;
      setIsDragging(false);
      if (event.dataTransfer) {
        handleFiles(event.dataTransfer.files);
      }
    };

    const handleDragEnter = (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      dragDepth.current += 1;
      setIsDragging(true);
    };

    const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    };

    const handleDragLeave = (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      dragDepth.current -= 1;
      if (dragDepth.current <= 0) {
        dragDepth.current = 0;
        setIsDragging(false);
      }
    };

    return (
      <label
        ref={labelRef}
        htmlFor="resume-file-input"
        data-testid="upload-zone"
        onDrop={handleDrop}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={[
          "group flex w-full cursor-pointer flex-col items-center justify-center",
          "rounded-2xl border-2 border-dashed px-6 py-12 text-center",
          "transition-colors focus-within:outline-none focus-within:ring-2",
          "focus-within:ring-indigo-500 focus-within:ring-offset-2",
          isDragging
            ? "border-indigo-500 bg-indigo-50"
            : "border-slate-300 bg-white hover:border-indigo-400 hover:bg-slate-50",
        ].join(" ")}
      >
        <input
          id="resume-file-input"
          type="file"
          accept={ACCEPT_ATTR}
          className="sr-only"
          onChange={handleChange}
          aria-label={label}
        />
        <UploadIcon className="h-10 w-10 text-slate-400 group-hover:text-indigo-500" />
        <p className="mt-4 text-base font-medium text-slate-700">
          Drag and drop your resume here
        </p>
        <p className="mt-1 text-sm text-slate-500">or</p>
        <span className="mt-2 inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700">
          Browse for a file
        </span>
        <p className="mt-4 text-xs text-slate-500">
          {ALLOWED_EXTENSIONS_LABEL} · up to {MAX_FILE_SIZE_MB} MB
        </p>
      </label>
    );
  }
);

export function UploadIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 16V4m0 0 4 4m-4-4-4 4" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </svg>
  );
}