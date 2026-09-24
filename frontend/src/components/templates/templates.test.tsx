import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DEFAULT_TEMPLATE_ID, getTemplate, isTemplateId, TEMPLATES } from "./index";
import type { Resume } from "@/lib/resume";

function makeResume(): Resume {
  return {
    contact: {
      name: "Ada Lovelace",
      email: "ada@example.com",
      phone: "+44 20 7946 0958",
      location: "London, UK",
      linkedin: "linkedin.com/in/ada",
      website: "https://ada.dev",
    },
    summary: "Mathematician and first programmer.",
    experience: [
      {
        company: "Analytical Engines Ltd",
        title: "Lead Engineer",
        location: "London",
        start_date: "1842",
        end_date: "1843",
        description: "Designed the first algorithm.",
        achievements: ["Published extensive notes", "Bridged maths and machines"],
      },
    ],
    education: [
      { institution: "Home Academy", degree: "BSc", field: "Mathematics", details: ["Self-directed"] },
    ],
    skills: {
      technical: ["Algorithms", "Analysis"],
      soft: ["Writing"],
      tools: ["Difference Engine"],
      languages: ["English", "French"],
    },
    projects: [
      { name: "Note G", description: "First published algorithm.", technologies: ["Bernoulli numbers"] },
    ],
    certifications: [{ name: "Fellow", issuer: "Royal Society", date: "1843" }],
    custom_sections: [{ heading: "Awards", content: ["Best notes"] }],
    metadata: { overall_confidence: "high", word_count: 42 },
  };
}

describe("template registry", () => {
  it("exposes at least three distinct templates and defaults to classic", () => {
    expect(TEMPLATES.length).toBeGreaterThanOrEqual(3);
    expect(new Set(TEMPLATES.map((t) => t.id)).size).toBe(TEMPLATES.length);
    expect(DEFAULT_TEMPLATE_ID).toBe("classic");
    expect(isTemplateId("modern")).toBe(true);
    expect(isTemplateId("__proto__")).toBe(false);
    expect(isTemplateId(42)).toBe(false);
  });

  it("falls back to the default template for unknown ids", () => {
    expect(getTemplate("noseuchtemplate").id).toBe(DEFAULT_TEMPLATE_ID);
    expect(getTemplate(null).id).toBe(DEFAULT_TEMPLATE_ID);
    expect(getTemplate("compact").id).toBe("compact");
  });
});

describe("templates render the same canonical resume", () => {
  for (const template of TEMPLATES) {
    it(`${template.id}: renders core content`, () => {
      const Template = template.component;
      const { container } = render(<Template resume={makeResume()} />);

      expect(screen.getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeInTheDocument();
      expect(within(container).getByText("Mathematician and first programmer.")).toBeInTheDocument();
      expect(within(container).getByText("Lead Engineer")).toBeInTheDocument();
      expect(within(container).getByText("Analytical Engines Ltd", { exact: false })).toBeInTheDocument();
      expect(within(container).getByText("Home Academy")).toBeInTheDocument();
      expect(within(container).getByText("Note G")).toBeInTheDocument();
      expect(within(container).getByText(/Algorithms/)).toBeInTheDocument();
      expect(within(container).getByText("Awards")).toBeInTheDocument();
      expect(within(container).getByText("Best notes")).toBeInTheDocument();
    });
  }

  it("uses layout-specific section headings while preserving the summary text", () => {
    const Classic = TEMPLATES.find((t) => t.id === "classic")!.component;
    const Modern = TEMPLATES.find((t) => t.id === "modern")!.component;

    const classic = render(<Classic resume={makeResume()} />);
    expect(
      within(classic.container).getByRole("heading", { name: "Summary" })
    ).toBeInTheDocument();
    classic.unmount();

    const modern = render(<Modern resume={makeResume()} />);
    expect(within(modern.container).getByRole("heading", { name: "Profile" })).toBeInTheDocument();
    expect(
      within(modern.container).getByText("Mathematician and first programmer.")
    ).toBeInTheDocument();
    modern.unmount();
  });

  it("produces different DOM per template for identical input", () => {
    const outputs = TEMPLATES.map((template) => {
      const Template = template.component;
      const { container, unmount } = render(<Template resume={makeResume()} />);
      const html = container.innerHTML;
      unmount();
      return html;
    });
    expect(new Set(outputs).size).toBe(TEMPLATES.length);
  });

  it("omits empty section headings", () => {
    const sparse: Resume = {
      contact: { name: "Only Name" },
      metadata: { overall_confidence: "low" },
    };
    for (const template of TEMPLATES) {
      const Template = template.component;
      const { container, unmount } = render(<Template resume={sparse} />);
      for (const title of ["Experience", "Education", "Projects", "Skills", "Certifications"]) {
        expect(within(container).queryByRole("heading", { name: title })).toBeNull();
      }
      expect(within(container).getByRole("heading", { level: 1, name: "Only Name" })).toBeInTheDocument();
      unmount();
    }
  });

  it("renders long content without truncation", () => {
    const long = "L".repeat(1200);
    const resume: Resume = {
      contact: { name: "Ada" },
      experience: [{ company: "Co", title: "Role", achievements: [long] }],
      metadata: { overall_confidence: "high" },
    };
    for (const template of TEMPLATES) {
      const Template = template.component;
      const { container, unmount } = render(<Template resume={resume} />);
      expect(within(container).getByText(long)).toBeInTheDocument();
      unmount();
    }
  });
});

describe("templates handle hostile resume content safely", () => {
  it("escapes HTML instead of executing it", () => {
    const resume: Resume = {
      contact: { name: '<script>alert("x")</script>' },
      summary: '<img src=x onerror="alert(1)">',
      metadata: { overall_confidence: "low" },
    };
    for (const template of TEMPLATES) {
      const Template = template.component;
      const { container, unmount } = render(<Template resume={resume} />);
      expect(container.querySelector("script")).toBeNull();
      expect(container.querySelector("img")).toBeNull();
      expect(
        within(container).getByRole("heading", { level: 1, name: '<script>alert("x")</script>' })
      ).toBeInTheDocument();
      unmount();
    }
  });

  it("never emits unsafe href schemes", () => {
    const resume: Resume = {
      contact: {
        name: "Ada",
        email: "not-an-email",
        website: "javascript:alert(1)",
        linkedin: "data:text/html;base64,PHNjcmlwdD4=",
        github: "vbscript:msgbox(1)",
      },
      metadata: { overall_confidence: "low" },
    };
    for (const template of TEMPLATES) {
      const Template = template.component;
      const { container, unmount } = render(<Template resume={resume} />);
      expect(container.querySelectorAll("a[href]").length).toBe(0);
      expect(within(container).getByText("javascript:alert(1)")).toBeInTheDocument();
      expect(within(container).getByText("not-an-email")).toBeInTheDocument();
      unmount();
    }
  });

  it("upgrades bare domains to https links", () => {
    const Classic = TEMPLATES.find((t) => t.id === "classic")!.component;
    const { container } = render(<Classic resume={makeResume()} />);
    const link = within(container).getByRole("link", { name: "linkedin.com/in/ada" });
    expect(link).toHaveAttribute("href", "https://linkedin.com/in/ada");
    const email = within(container).getByRole("link", { name: "ada@example.com" });
    expect(email).toHaveAttribute("href", "mailto:ada@example.com");
  });
});
