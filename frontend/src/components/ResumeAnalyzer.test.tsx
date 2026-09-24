import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { Resume } from "@/lib/resume";

import ResumeAnalyzer from "./ResumeAnalyzer";

const { parseResumeMock } = vi.hoisted(() => ({ parseResumeMock: vi.fn() }));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, parseResume: parseResumeMock };
});

const SAMPLE_RESUME: Resume = {
  contact: { name: "Ada Lovelace", email: "ada@example.com" },
  summary: "First programmer.",
  skills: { technical: ["Python", "React"] },
  experience: [
    {
      company: "Analytical Engines",
      title: "Engineer",
      start_date: "2020",
      end_date: "Present",
      achievements: ["Built the difference engine"],
    },
  ],
  metadata: {
    overall_confidence: "high",
    section_confidence: [{ section: "skills", level: "high" }],
  },
};

const SPARSE_RESUME: Resume = {
  contact: { name: "Only Name" },
  metadata: { overall_confidence: "low" },
};

function makePdfFile(contents = "synthetic resume", name = "resume.pdf"): File {
  return new File([contents], name, { type: "application/pdf" });
}

describe("ResumeAnalyzer", () => {
  beforeEach(() => {
    parseResumeMock.mockReset();
  });

  it("renders branding, heading, and privacy notice", () => {
    render(<ResumeAnalyzer />);
    expect(screen.getByRole("heading", { name: /analyze your resume/i })).toBeInTheDocument();
    expect(screen.getByText("ResumeForge")).toBeInTheDocument();
    expect(screen.getByText(/not permanently stored/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/choose a resume file/i)).toBeInTheDocument();
  });

  it("selects a file and analyzes it, rendering results", async () => {
    parseResumeMock.mockResolvedValue(SAMPLE_RESUME);
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    expect(screen.getByText("resume.pdf")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /analyze resume/i }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /resume overview/i })).toBeInTheDocument();
    });
    expect(screen.getByText("Ada Lovelace")).toBeInTheDocument();
    expect(screen.getByText("ada@example.com")).toBeInTheDocument();
    expect(screen.getByText("Python")).toBeInTheDocument();
    expect(screen.getByText("Analytical Engines")).toBeInTheDocument();
    expect(screen.getByText(/heuristic parsing confidence/i)).toBeInTheDocument();
    expect(parseResumeMock).toHaveBeenCalledTimes(1);
  });

  it("accepts a dropped file", async () => {
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    fireEvent.drop(screen.getByTestId("upload-zone"), {
      dataTransfer: { files: [makePdfFile("drop", "dropped.pdf")] },
    });

    expect(screen.getByText("dropped.pdf")).toBeInTheDocument();

    parseResumeMock.mockResolvedValue(SAMPLE_RESUME);
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));
    await waitFor(() => {
      expect(screen.getByText(/analysis complete/i)).toBeInTheDocument();
    });
  });

  it("rejects an unsupported extension with a message", async () => {
    render(<ResumeAnalyzer />);

    fireEvent.change(screen.getByLabelText(/choose a resume file/i), {
      target: { files: [new File(["x"], "a.exe")] },
    });
    expect(screen.getByRole("alert")).toHaveTextContent(/unsupported file type/i);
    expect(queryFilePreview()).toBeNull();
  });

  it("rejects an oversized file", async () => {
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    const bytes = new Uint8Array(11 * 1024 * 1024);
    await user.upload(
      screen.getByLabelText(/choose a resume file/i),
      new File([bytes], "big.pdf")
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/larger than 10 MB/i);
  });

  it("rejects an empty file", async () => {
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), new File([], "empty.pdf"));
    expect(screen.getByRole("alert")).toHaveTextContent(/empty/i);
  });

  it("shows a loading state while analyzing", async () => {
    let resolveResume: (value: Resume) => void = () => {};
    parseResumeMock.mockImplementation(
      () => new Promise<Resume>((resolve) => (resolveResume = resolve))
    );
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));

    expect(screen.getByText(/uploading resume/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /analyze resume/i })).toBeNull();

    resolveResume(SAMPLE_RESUME);
    await waitFor(() => {
      expect(screen.getByText(/analysis complete/i)).toBeInTheDocument();
    });
  });

  it("displays a friendly message for a backend parsing error", async () => {
    parseResumeMock.mockRejectedValue(
      new ApiError("malformed_file", "The file appears to be malformed or corrupted.")
    );
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/malformed or corrupted/i);
    });
  });

  it("displays a friendly message for a network failure", async () => {
    parseResumeMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/reach the resumeforge backend/i);
    });
    expect(screen.getByText("resume.pdf")).toBeInTheDocument();
  });

  it("allows removing the selected file", async () => {
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    expect(screen.getByText("resume.pdf")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove selected file/i }));
    expect(queryFilePreview()).toBeNull();
    expect(screen.queryByRole("button", { name: /analyze resume/i })).toBeNull();
  });

  it("resets to the upload state after analysis", async () => {
    parseResumeMock.mockResolvedValue(SAMPLE_RESUME);
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));
    await waitFor(() => {
      expect(screen.getByText(/analysis complete/i)).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /analyze another resume/i }));

    expect(screen.getByText(/drag and drop your resume here/i)).toBeInTheDocument();
    expect(queryFilePreview()).toBeNull();
    expect(screen.queryByText(/analysis complete/i)).toBeNull();
  });

  it("does not render empty optional sections", async () => {
    parseResumeMock.mockResolvedValue(SPARSE_RESUME);
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));

    await waitFor(() => {
      expect(screen.getByText("Only Name")).toBeInTheDocument();
    });
    expect(screen.queryByText("Experience")).not.toBeInTheDocument();
    expect(screen.queryByText("Summary")).not.toBeInTheDocument();
    expect(screen.queryByText("Skills")).not.toBeInTheDocument();
  });

  it("opens the resume builder for the working resume", async () => {
    parseResumeMock.mockResolvedValue(SAMPLE_RESUME);
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /resume overview/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("tab", { name: "Resume builder" }));
    expect(screen.getByRole("heading", { name: "Resume builder" })).toBeInTheDocument();
    expect(screen.getByTestId("resume-preview")).toBeInTheDocument();
    expect(screen.getByLabelText("Full name")).toHaveValue("Ada Lovelace");
  });

  it("shares builder edits with the structured view and can reset them", async () => {
    parseResumeMock.mockResolvedValue(SAMPLE_RESUME);
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(screen.getByLabelText(/choose a resume file/i), makePdfFile());
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /resume overview/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("tab", { name: "Resume builder" }));
    const nameInput = screen.getByLabelText("Full name");
    await user.clear(nameInput);
    await user.type(nameInput, "Ada L.");

    await user.click(screen.getByRole("tab", { name: "Structured view" }));
    expect(screen.getByText("Ada L.")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Resume builder" }));
    await user.click(screen.getByRole("button", { name: "Reset changes" }));
    await user.click(screen.getByRole("tab", { name: "Structured view" }));
    expect(screen.getByText("Ada Lovelace")).toBeInTheDocument();
  });

  it("never logs resume content to the console during a run", async () => {
    parseResumeMock.mockResolvedValue(SAMPLE_RESUME);
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const user = userEvent.setup();
    render(<ResumeAnalyzer />);

    await user.upload(
      screen.getByLabelText(/choose a resume file/i),
      makePdfFile("PII: secret street, 555-0199")
    );
    await user.click(screen.getByRole("button", { name: /analyze resume/i }));
    await waitFor(() => {
      expect(screen.getByText(/analysis complete/i)).toBeInTheDocument();
    });

    const logText = errorLog.mock.calls.concat(logSpy.mock.calls).join("|");
    expect(logText).not.toContain("555-0199");
    expect(logText).not.toContain("PII:");
  });
});

function queryFilePreview() {
  return screen.queryByText(/\.(pdf|docx|txt)$/);
}