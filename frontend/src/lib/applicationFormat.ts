/**
 * Phase 7E — small display formatters for the tracker and dashboard. Kept
 * separate so the meaning and units of each score stay consistent everywhere.
 */

export function formatScore(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value)} / 100`
    : "Not available";
}

/** Job-Specific ATS Coverage is stored as a 0-1 ratio. */
export function formatCoverage(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value * 100)}%`
    : "Not available";
}

export function formatDate(value: string | null | undefined): string {
  return value && value.trim() ? value : "Not set";
}

export function formatOptionalText(value: string | null | undefined, fallback = "Not set"): string {
  return value && value.trim() ? value : fallback;
}
