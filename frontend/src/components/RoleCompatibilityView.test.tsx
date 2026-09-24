import type { RoleAnalysisResult } from "@/lib/roleAnalysis";
import type { Resume } from "@/lib/resume";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RoleCompatibilityView } from "@/components/RoleCompatibilityView";
import { analyzeRoleCompatibility } from "@/lib/roleAnalysis";

const mockResume: Resume = {
  contact: { name: "Jane Doe", email: "jane@example.com" },
  summary: "Software engineer with 4 years of experience.",
  experience: [
    {
      company: "Acme",
      title: "Engineer",
      start_date: "2020",
      end_date: "2024",
    },
  ],
  education: [
    { institution: "MIT", degree: "B.Tech", field: "Computer Science" },
  ],
  skills: {
    technical: ["Python", "JavaScript", "Git", "SQL"],
    soft: ["Communication"],
    tools: [],
    languages: [],
    all: ["Python", "JavaScript", "Git", "SQL", "Communication"],
  },
  projects: [],
  certifications: [],
  custom_sections: [],
  metadata: {
    word_count: 100,
    file_type: "pdf",
    overall_confidence: "high",
  },
};

const supportedResult: RoleAnalysisResult = {
  role_title: "Software Engineer",
  profile: {
    title: "Software Engineer",
    aliases: ["software developer", "developer", "swe"],
    supported: true,
  },
  compatibility: {
    overall: 85,
    skills: 90,
    experience: 80,
  },
  skills: {
    have: ["Python", "JavaScript", "Git", "SQL"],
    missing: ["Docker", "AWS"],
    matched_count: 4,
    required_total: 4,
    preferred_total: 2,
  },
  experience: {
    candidate_years: 4,
    required_years: 2,
    required_level: "bachelor",
    gap_years: 0,
    met: true,
    resume_experience_count: 1,
  },
  education: {
    required_level: "bachelor",
    candidate_level: "bachelor",
    met: true,
  },
  matched_requirements: ["Required skill matched: Python", "Required skill matched: JavaScript"],
  missing_required: ["Docker", "AWS"],
  strengths: ["Matches 2 of 2 required skills."],
  gaps: ["Missing required skill: Docker", "Missing required skill: AWS"],
  recommendations: [
    "Add the following skills to your resume: Docker, AWS.",
    "Your ~4 years of experience meets the ~2 years requirement.",
  ],
};

const unknownResult: RoleAnalysisResult = {
  role_title: "Quantum Physicist",
  profile: {
    title: "Quantum Physicist",
    aliases: [],
    supported: false,
  },
  compatibility: { overall: null },
  skills: {
    have: [],
    missing: [],
    matched_count: 0,
    required_total: 0,
    preferred_total: 0,
  },
  experience: {
    candidate_years: null,
    required_years: null,
    required_level: null,
    gap_years: null,
    met: null,
    resume_experience_count: 0,
  },
  education: {
    required_level: null,
    candidate_level: null,
    met: null,
  },
  matched_requirements: [],
  missing_required: [],
  strengths: [],
  gaps: [],
  recommendations: [
    "The role 'Quantum Physicist' is not in the local profile database. Try a different role, or upload a specific job description for targeted analysis.",
  ],
};

describe("RoleCompatibilityView", () => {
  it("renders supported role with scores", () => {
    render(<RoleCompatibilityView result={supportedResult} />);
    expect(screen.getByText("Software Engineer")).toBeInTheDocument();
    expect(screen.getByText("Role-level compatibility")).toBeInTheDocument();
    expect(screen.getByText("85 / 100")).toBeInTheDocument();
  });

  it("renders skills have and missing", () => {
    render(<RoleCompatibilityView result={supportedResult} />);
    expect(screen.getByText("Python")).toBeInTheDocument();
    expect(screen.getByText("JavaScript")).toBeInTheDocument();
    expect(screen.getByText("Docker")).toBeInTheDocument();
    expect(screen.getByText("AWS")).toBeInTheDocument();
  });

  it("renders experience comparison", () => {
    render(<RoleCompatibilityView result={supportedResult} />);
    expect(screen.getByRole("heading", { name: "Experience" })).toBeInTheDocument();
    expect(screen.getByText("~4 years")).toBeInTheDocument();
    expect(screen.getByText("Meets requirement")).toBeInTheDocument();
  });

  it("renders recommendations", () => {
    render(<RoleCompatibilityView result={supportedResult} />);
    expect(
      screen.getByText(/Add the following skills to your resume/),
    ).toBeInTheDocument();
  });

  it("renders strengths", () => {
    render(<RoleCompatibilityView result={supportedResult} />);
    expect(screen.getByText(/Matches 2 of 2 required skills/)).toBeInTheDocument();
  });

  it("renders unknown role fallback", () => {
    render(<RoleCompatibilityView result={unknownResult} />);
    expect(screen.getByText("Quantum Physicist")).toBeInTheDocument();
    expect(
      screen.getByText(/not in the local profile database/),
    ).toBeInTheDocument();
    expect(screen.getByText("No local profile available")).toBeInTheDocument();
  });

  it("shows alias list when available", () => {
    render(<RoleCompatibilityView result={supportedResult} />);
    expect(
      screen.getByText(/Also known as: software developer, developer, swe/),
    ).toBeInTheDocument();
  });
});

describe("analyzeRoleCompatibility", () => {
  it("is defined and is a function", () => {
    expect(typeof analyzeRoleCompatibility).toBe("function");
  });

  it("calls the correct endpoint with correct payload", async () => {
    const mockFetch = vi.fn();
    const fakeResponse: RoleAnalysisResult = {
      ...unknownResult,
      role_title: "Software Engineer",
    };
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => fakeResponse,
    });
    globalThis.fetch = mockFetch;

    await analyzeRoleCompatibility("Software Engineer", mockResume as unknown as Resume);

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/roles/analyze"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role_title: "Software Engineer",
          resume: mockResume,
        }),
      }),
    );
  });
});