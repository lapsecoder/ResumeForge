import { fireEvent, render, screen, within } from "@testing-library/react";
import { useRef, useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useHistory } from "@/lib/history";
import type { Resume } from "@/lib/resume";
import type { TemplateId } from "./templates";

import { ResumeBuilder } from "./ResumeBuilder";

function makeResume(): Resume {
  return {
    contact: {
      name: "Ada Lovelace",
      email: "ada@example.com",
      phone: "+44 20 7946 0958",
      location: "London, UK",
      linkedin: "linkedin.com/in/ada",
    },
    summary: "Mathematician and first programmer.",
    experience: [
      {
        company: "Analytical Engines Ltd",
        title: "Lead Engineer",
        achievements: ["Published extensive notes"],
      },
    ],
    education: [{ institution: "Home Academy", degree: "BSc", field: "Mathematics" }],
    skills: {
      technical: ["Algorithms"],
      soft: [],
      tools: [],
      languages: [],
    },
    projects: [{ name: "Note G", description: "First published algorithm." }],
    certifications: [{ name: "Fellow", issuer: "Royal Society" }],
    custom_sections: [{ heading: "Awards", content: ["Best notes"] }],
    metadata: { overall_confidence: "high", word_count: 42 },
  };
}

function twoExperiences(): Resume {
  const resume = makeResume();
  resume.experience = [
    { company: "Alpha Co", title: "Alpha Role", achievements: [] },
    { company: "Beta Co", title: "Beta Role", achievements: [] },
  ];
  return resume;
}

function Harness({ initial }: { initial: Resume }) {
  const history = useHistory<Resume>(initial);
  const original = useRef(initial);
  const [templateId, setTemplateId] = useState<TemplateId>("classic");
  return (
    <ResumeBuilder
      resume={history.present}
      canUndo={history.canUndo}
      canRedo={history.canRedo}
      onCommit={history.commit}
      onUndo={history.undo}
      onRedo={history.redo}
      onReset={() => history.reset(original.current)}
      templateId={templateId}
      onTemplateChange={setTemplateId}
    />
  );
}

function preview() {
  return within(screen.getByTestId("resume-preview"));
}

const getBlob = vi.fn().mockResolvedValue(new Blob(["pdf-bytes"]));
const createPdf = vi.fn().mockReturnValue({ getBlob });
const addVirtualFileSystem = vi.fn();

const pdfMakeMock = { createPdf, addVirtualFileSystem };

vi.mock("pdfmake", () => ({
  ...pdfMakeMock,
  default: pdfMakeMock,
}));

vi.mock("pdfmake/build/vfs_fonts", () => ({
  default: { "Roboto-Regular.ttf": "BASE64_FONT" },
}));

// Mock createPortal to avoid jsdom portal issues
vi.mock("react-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-dom")>();
  return {
    ...actual,
    createPortal: (children: React.ReactNode) => children,
  };
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = "";
});

describe("ResumeBuilder", () => {
  it("renders the builder toolbar and live preview", () => {
    render(<Harness initial={makeResume()} />);
    expect(screen.getByRole("heading", { name: "Resume builder" })).toBeInTheDocument();
    expect(screen.getByLabelText("Template")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Redo" })).toBeDisabled();
    expect(preview().getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeInTheDocument();
  });

  it("updates the preview when a field is edited", () => {
    render(<Harness initial={makeResume()} />);
    fireEvent.change(screen.getByLabelText("Full name"), { target: { value: "Ada L." } });
    expect(preview().getByRole("heading", { level: 1, name: "Ada L." })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
  });

  it("undoes and redoes edits", () => {
    render(<Harness initial={makeResume()} />);
    fireEvent.change(screen.getByLabelText("Full name"), { target: { value: "Ada L." } });
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(preview().getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Redo" }));
    expect(preview().getByRole("heading", { level: 1, name: "Ada L." })).toBeInTheDocument();
  });

  it("adds and removes entries", () => {
    render(<Harness initial={makeResume()} />);
    expect(screen.getAllByLabelText("Company")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Add experience" }));
    expect(screen.getAllByLabelText("Company")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Remove experience 2" }));
    expect(screen.getAllByLabelText("Company")).toHaveLength(1);
  });

  it("reorders entries", () => {
    render(<Harness initial={twoExperiences()} />);
    const textBefore = screen.getByTestId("resume-preview").textContent ?? "";
    expect(textBefore.indexOf("Alpha Role")).toBeLessThan(textBefore.indexOf("Beta Role"));

    fireEvent.click(screen.getByRole("button", { name: "Move experience 2 up" }));
    const textAfter = screen.getByTestId("resume-preview").textContent ?? "";
    expect(textAfter.indexOf("Beta Role")).toBeLessThan(textAfter.indexOf("Alpha Role"));
  });

  it("resets changes back to the original resume", () => {
    render(<Harness initial={makeResume()} />);
    fireEvent.change(screen.getByLabelText("Full name"), { target: { value: "Changed" } });
    expect(preview().getByRole("heading", { level: 1, name: "Changed" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset changes" }));
    expect(preview().getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
  });

  it("blocks export while validation errors exist", () => {
    render(<Harness initial={makeResume()} />);
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "not-an-email" } });
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export PDF" })).toBeDisabled();
  });

  it("generates PDF with a safe filename on export", async () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click");
    render(<Harness initial={makeResume()} />);
    fireEvent.click(screen.getByRole("button", { name: "Export PDF" }));

    await vi.waitFor(() => {
      expect(createPdf).toHaveBeenCalled();
    });
    expect(getBlob).toHaveBeenCalled();
    expect(addVirtualFileSystem).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalledTimes(1);
    const anchor = clickSpy.mock.instances[0] as HTMLAnchorElement;
    expect(anchor.download).toBe("Ada_Lovelace_Resume.pdf");
    expect(screen.getByRole("status").textContent).toContain("Ada_Lovelace_Resume.pdf");
    clickSpy.mockRestore();
  });

  it("switches templates without changing content", () => {
    render(<Harness initial={makeResume()} />);
    fireEvent.change(screen.getByLabelText("Template"), { target: { value: "modern" } });
    expect(preview().getByRole("heading", { name: "Profile" })).toBeInTheDocument();
    expect(preview().getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeInTheDocument();
    expect((screen.getByLabelText("Full name") as HTMLInputElement).value).toBe("Ada Lovelace");
  });

  it("never persists or uploads resume data while editing", () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const fetchSpy = vi.fn();
    const originalFetch = globalThis.fetch;
    Reflect.set(globalThis, "fetch", fetchSpy);
    try {
      render(<Harness initial={makeResume()} />);
      fireEvent.change(screen.getByLabelText("Full name"), { target: { value: "Ada L." } });
      fireEvent.click(screen.getByRole("button", { name: "Add experience" }));
      expect(setItem).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      setItem.mockRestore();
      Reflect.set(globalThis, "fetch", originalFetch);
    }
  });
});