import { describe, expect, it } from "vitest";

import {
  formatCoverage,
  formatDate,
  formatOptionalText,
  formatScore,
} from "./applicationFormat";

describe("formatScore", () => {
  it("rounds a score to a /100 rendering", () => {
    expect(formatScore(81.6)).toBe("82 / 100");
    expect(formatScore(0)).toBe("0 / 100");
    expect(formatScore(100)).toBe("100 / 100");
  });

  it("falls back for missing or non-finite values", () => {
    expect(formatScore(null)).toBe("Not available");
    expect(formatScore(undefined)).toBe("Not available");
    expect(formatScore(Number.NaN)).toBe("Not available");
    expect(formatScore(Number.POSITIVE_INFINITY)).toBe("Not available");
  });
});

describe("formatCoverage", () => {
  it("renders a 0-1 ratio as a percentage", () => {
    expect(formatCoverage(0.864)).toBe("86%");
    expect(formatCoverage(0)).toBe("0%");
    expect(formatCoverage(1)).toBe("100%");
  });

  it("falls back for missing or non-finite values", () => {
    expect(formatCoverage(null)).toBe("Not available");
    expect(formatCoverage(undefined)).toBe("Not available");
  });
});

describe("formatDate / formatOptionalText", () => {
  it("renders dates with a fallback", () => {
    expect(formatDate("2026-09-01")).toBe("2026-09-01");
    expect(formatDate(null)).toBe("Not set");
    expect(formatDate("   ")).toBe("Not set");
  });

  it("renders optional text with a custom fallback", () => {
    expect(formatOptionalText("Berlin")).toBe("Berlin");
    expect(formatOptionalText(null, "Unknown")).toBe("Unknown");
    expect(formatOptionalText("", "Unknown")).toBe("Unknown");
  });
});