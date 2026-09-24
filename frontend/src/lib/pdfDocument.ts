/**
 * Structured PDF document definitions for the three resume templates.
 *
 * These builders translate the canonical `Resume` object and a template id
 * into a pdfmake document definition (A4, real selectable text). This is a
 * pure layer: no DOM, no canvas, no browser print API. Page breaks are left
 * to pdfmake's automatic flow, so long sections wrap and continue onto new
 * pages instead of being silently clipped. Section headings carry a
 * `headlineLevel` and a `pageBreakBefore` guard keeps them from being
 * orphaned at the bottom of a page.
 */

import type {
  Content,
  ContentCanvas,
  ContentText,
  TDocumentDefinitions,
} from "pdfmake/interfaces";

import type { TemplateId } from "@/components/templates";
import {
  contactItems,
  dateRange,
  isBlank,
  type ContactItem,
} from "@/components/templates/shared";
import type {
  Certification,
  CustomSection,
  Education,
  Project,
  Resume,
  WorkExperience,
} from "./resume";

export const PDF_PAGE_SIZE = "A4";

/** [left, top, right, bottom] in pt (≈ 14.8 mm sides, 17 mm bottom). */
export const PDF_PAGE_MARGINS: readonly [number, number, number, number] = [
  42, 40, 42, 48,
];

/** A4 page width in points (matches pdfmake/pdfkit's A4). */
export const A4_WIDTH_PT = 595.28;
export const PDF_CONTENT_WIDTH = A4_WIDTH_PT - PDF_PAGE_MARGINS[0] - PDF_PAGE_MARGINS[2];

/** Classic template's narrow label column (84 px ≈ 63 pt). */
export const COMPACT_LABEL_WIDTH = 63;

const INK = "#0f172a";
const BODY = "#334155";
const MUTED = "#64748b";
const ACCENT = "#6366f1";
const ACCENT_DARK = "#4f46e5";
const ACCENT_SOFT = "#818cf8";
const RULE = "#cbd5e1";
const RULE_STRONG = "#94a3b8";

function anyText(...values: (string | null | undefined)[]): boolean {
  return values.some((value) => !isBlank(value));
}

function nonBlank(value: string | null | undefined): string | null {
  return isBlank(value) ? null : (value as string);
}

function joinParts(...values: (string | null | undefined)[]): string {
  return values.filter((value): value is string => !isBlank(value)).join(" · ");
}

function divider(width: number, color: string, thickness = 0.7): ContentCanvas {
  return {
    canvas: [
      {
        type: "line",
        x1: 0,
        y1: 0,
        x2: width,
        y2: 0,
        lineWidth: thickness,
        lineColor: color,
      },
    ],
  };
}

function withHeading(
  title: string,
  headingStyle: string,
  headingMargin: [number, number, number, number],
  ruleColor: string,
  ruleMargin: [number, number, number, number],
  body: Content[]
): Content[] {
  return [
    {
      text: title.toUpperCase(),
      style: headingStyle,
      headlineLevel: 1,
      margin: headingMargin,
    },
    { ...divider(PDF_CONTENT_WIDTH, ruleColor), margin: ruleMargin },
    ...body,
  ];
}

function initialMargin(
  firstSection: boolean,
  compact: boolean
): [number, number, number, number] {
  return [0, firstSection ? 4 : compact ? 8 : 12, 0, 2];
}

function bulletList(
  items: string[],
  style: string,
  markerColor: string,
  marker: "disc" | "square"
): Content {
  return {
    ul: items.map((item) => ({
      text: item,
      style,
      margin: [0, 0, 0, 1],
    })),
    type: marker,
    markerColor,
    margin: [6, 0, 0, 0],
  };
}

interface SkillGroup {
  label: string;
  items: string[];
  compactLabel: string;
}

function skillGroups(resume: Resume): SkillGroup[] {
  const skills = resume.skills;
  const raw = [
    { label: "Technical Skills", compactLabel: "Technical", group: skills?.technical },
    { label: "Tools", compactLabel: "Tools", group: skills?.tools },
    { label: "Languages", compactLabel: "Languages", group: skills?.languages },
    { label: "Soft Skills", compactLabel: "Soft", group: skills?.soft },
  ] as const;
  const groups: SkillGroup[] = [];
  for (const { label, compactLabel, group } of raw) {
    const items = (group ?? []).filter((item): item is string => !isBlank(item));
    if (items.length > 0) groups.push({ label, compactLabel, items });
  }
  return groups;
}

interface FilteredResume {
  name: string | null;
  items: ContactItem[];
  summary: string | null;
  experience: WorkExperience[];
  education: Education[];
  projects: Project[];
  certifications: Certification[];
  custom: CustomSection[];
  skills: SkillGroup[];
}

function filterResume(resume: Resume): FilteredResume {
  const experience = (resume.experience ?? []).filter((entry) =>
    anyText(entry.title, entry.company, entry.description, ...(entry.achievements ?? []))
  );
  const education = (resume.education ?? []).filter((entry) =>
    anyText(entry.institution, entry.degree, entry.field, ...(entry.details ?? []))
  );
  const projects = (resume.projects ?? []).filter((entry) =>
    anyText(entry.name, entry.description, ...(entry.technologies ?? []))
  );
  const certifications = (resume.certifications ?? []).filter((entry) =>
    anyText(entry.name, entry.issuer)
  );
  const custom = (resume.custom_sections ?? []).filter((entry) =>
    anyText(entry.heading, ...(entry.content ?? []))
  );
  return {
    name: nonBlank(resume.contact?.name),
    items: contactItems(resume.contact),
    summary: nonBlank(resume.summary),
    experience,
    education,
    projects,
    certifications,
    custom,
    skills: skillGroups(resume),
  };
}

function contactLine(items: ContactItem[]): string | null {
  if (items.length === 0) return null;
  return items.map((item) => item.value).join(" · ");
}

/* ------------------------------------------------------------------ */
/* Classic                                                             */
/* ------------------------------------------------------------------ */

function classicEntryMargin(): [number, number, number, number] {
  return [0, 0, 0, 10];
}

function classicExperienceEntry(entry: WorkExperience): Content {
  const range = dateRange(entry.start_date, entry.end_date);
  const title = nonBlank(entry.title) ?? nonBlank(entry.company) ?? "Experience";
  const company = nonBlank(entry.title) ? nonBlank(entry.company) : null;
  const bullets = (entry.achievements ?? []).filter((item) => !isBlank(item));
  const stack: Content[] = [
    {
      columns: [
        { text: title, style: "classicEntryTitle", width: "*" },
        range
          ? { text: range, style: "classicRange", width: "auto", alignment: "right" }
          : { text: "", width: "auto" },
      ],
      margin: [0, 0, 0, 1],
    },
  ];
  const subtitle = joinParts(company, entry.location);
  if (subtitle) {
    stack.push({ text: subtitle, style: "classicSubtitle", margin: [0, 0, 0, 2] });
  }
  if (!isBlank(entry.description)) {
    stack.push({
      text: entry.description as string,
      style: "classicBody",
      margin: [0, 0, 0, bullets.length > 0 ? 2 : 0],
    });
  }
  if (bullets.length > 0) {
    stack.push(bulletList(bullets, "classicBody", MUTED, "disc"));
  }
  return { unbreakable: true, stack, margin: classicEntryMargin() };
}

function classicEducationEntry(entry: Education): Content {
  const range = dateRange(entry.start_date, entry.end_date);
  const degree = joinParts(entry.degree, entry.field);
  const details = (entry.details ?? []).filter((item) => !isBlank(item));
  const stack: Content[] = [
    {
      columns: [
        { text: entry.institution, style: "classicEntryTitle", width: "*" },
        range
          ? { text: range, style: "classicRange", width: "auto", alignment: "right" }
          : { text: "", width: "auto" },
      ],
      margin: [0, 0, 0, 1],
    },
  ];
  const subtitle = joinParts(degree, entry.location);
  if (subtitle) {
    stack.push({ text: subtitle, style: "classicSubtitle", margin: [0, 0, 0, entry.details ? 2 : 0] });
  }
  if (details.length > 0) {
    stack.push(bulletList(details, "classicBody", MUTED, "disc"));
  }
  return { unbreakable: true, stack, margin: classicEntryMargin() };
}

function classicProjectEntry(entry: Project): Content {
  const tech = (entry.technologies ?? []).filter((item) => !isBlank(item));
  const stack: Content[] = [
    { text: entry.name, style: "classicEntryTitle", margin: [0, 0, 0, 1] },
  ];
  if (!isBlank(entry.description)) {
    stack.push({ text: entry.description as string, style: "classicBody", margin: [0, 0, 0, tech.length > 0 ? 1 : 0] });
  }
  if (tech.length > 0) {
    stack.push({ text: tech.join(", "), style: "classicSubtitle" });
  }
  return { unbreakable: true, stack, margin: classicEntryMargin() };
}

function classicCertificationEntry(entry: Certification): Content {
  const meta = joinParts(entry.issuer, entry.date);
  const stack: Content[] = [{ text: entry.name, style: "classicEntryTitle" }];
  if (meta) {
    stack.push({ text: meta, style: "classicSubtitle", margin: [0, 1, 0, 0] });
  }
  return { unbreakable: true, stack, margin: [0, 0, 0, 6] };
}

function buildClassic(resume: Resume): Content[] {
  const data = filterResume(resume);
  const content: Content[] = [];

  if (data.name || data.items.length > 0) {
    const header: Content[] = [];
    if (data.name) {
      header.push({ text: data.name, style: "classicName" });
    }
    const line = contactLine(data.items);
    if (line) {
      header.push({ text: line, style: "classicContact", margin: [0, 2, 0, 0] });
    }
    content.push({ unbreakable: true, stack: header });
  }

  let firstSection = true;

  if (data.summary) {
    content.push(
      ...withHeading(
        "Summary",
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 8],
        [{ text: data.summary, style: "classicBody" }]
      )
    );
    firstSection = false;
  }

  if (data.experience.length > 0) {
    content.push(
      ...withHeading(
        "Experience",
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 4],
        data.experience.map(classicExperienceEntry)
      )
    );
    firstSection = false;
  }

  if (data.education.length > 0) {
    content.push(
      ...withHeading(
        "Education",
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 4],
        data.education.map(classicEducationEntry)
      )
    );
    firstSection = false;
  }

  if (data.projects.length > 0) {
    content.push(
      ...withHeading(
        "Projects",
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 4],
        data.projects.map(classicProjectEntry)
      )
    );
    firstSection = false;
  }

  if (data.skills.length > 0) {
    content.push(
      ...withHeading(
        "Skills",
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 4],
        data.skills.map((group) => ({
          text: [{ text: `${group.label}: `, bold: true, color: INK }, group.items.join(", ")],
          style: "classicBody",
          margin: [0, 0, 0, 2],
        }))
      )
    );
    firstSection = false;
  }

  if (data.certifications.length > 0) {
    content.push(
      ...withHeading(
        "Certifications",
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 4],
        data.certifications.map(classicCertificationEntry)
      )
    );
    firstSection = false;
  }

  for (const entry of data.custom) {
    const lines = (entry.content ?? []).filter((line) => !isBlank(line));
    content.push(
      ...withHeading(
        isBlank(entry.heading) ? "Additional" : (entry.heading as string),
        "classicHeading",
        initialMargin(firstSection, false),
        RULE,
        [0, 0, 0, 4],
        lines.length > 0 ? [bulletList(lines, "classicBody", MUTED, "disc")] : []
      )
    );
    firstSection = false;
  }

  return content;
}

/* ------------------------------------------------------------------ */
/* Modern                                                              */
/* ------------------------------------------------------------------ */

function modernAccentHeading(
  title: string,
  firstSection: boolean
): Content {
  return {
    columns: [
      {
        width: 4,
        canvas: [
          { type: "rect", x: 0, y: 1.6, w: 4, h: 9.6, color: ACCENT },
        ],
      },
      {
        width: "*",
        text: title.toUpperCase(),
        style: "modernHeading",
        headlineLevel: 1,
        marginLeft: 8,
        marginTop: 0.5,
      },
    ],
    columnGap: 0,
    margin: [0, firstSection ? 4 : 14, 0, 6],
  };
}

function modernExperienceEntry(entry: WorkExperience): Content {
  const range = dateRange(entry.start_date, entry.end_date);
  const title = nonBlank(entry.title) ?? nonBlank(entry.company) ?? "Experience";
  const company = nonBlank(entry.title) ? nonBlank(entry.company) : null;
  const bullets = (entry.achievements ?? []).filter((item) => !isBlank(item));
  const stack: Content[] = [
    {
      columns: [
        { text: title, style: "modernEntryTitle", width: "*" },
        range
          ? { text: range, style: "modernRange", width: "auto", alignment: "right" }
          : { text: "", width: "auto" },
      ],
      margin: [0, 0, 0, 1],
    },
  ];
  const subtitle = joinParts(company, entry.location);
  if (subtitle) {
    stack.push({ text: subtitle, style: "modernSubtitle", margin: [0, 0, 0, 2] });
  }
  if (!isBlank(entry.description)) {
    stack.push({
      text: entry.description as string,
      style: "modernBody",
      margin: [0, 0, 0, bullets.length > 0 ? 2 : 0],
    });
  }
  if (bullets.length > 0) {
    stack.push(bulletList(bullets, "modernBody", ACCENT_SOFT, "square"));
  }
  return { unbreakable: true, stack, margin: [0, 0, 0, 12] };
}

function modernEduProjectEntry(
  main: string,
  subtitle: string | null,
  details: string[]
): Content {
  const stack: Content[] = [{ text: main, style: "modernEntryTitle", margin: [0, 0, 0, 1] }];
  if (subtitle) {
    stack.push({ text: subtitle, style: "modernSubtitle", margin: [0, 0, 0, details.length > 0 ? 1 : 0] });
  }
  if (details.length > 0) {
    stack.push(bulletList(details, "modernBody", ACCENT_SOFT, "square"));
  }
  return { unbreakable: true, stack, margin: [0, 0, 0, 10] };
}

function buildModern(resume: Resume): Content[] {
  const data = filterResume(resume);
  const content: Content[] = [];

  const leftColumn: Content[] = [];
  if (data.name) {
    leftColumn.push({ text: data.name, style: "modernName" });
  }
  if (!isBlank(resume.contact?.location)) {
    leftColumn.push({
      text: resume.contact?.location as string,
      style: "modernSubtitle",
      margin: [0, 1, 0, 0],
    });
  }
  const rightColumn: Content[] =
    data.items.map((item) => ({
      text: item.value,
      style: "modernContact",
      alignment: "right",
      margin: [0, 0, 0, 0],
    }));

  content.push({
    unbreakable: true,
    stack: [
      {
        columns: [
          { width: "*", stack: leftColumn },
          rightColumn.length > 0
            ? { width: "*", stack: rightColumn, alignment: "right" }
            : { width: "*", text: "" },
        ],
        columnGap: 24,
        margin: [0, 0, 0, 8],
      },
      { ...divider(PDF_CONTENT_WIDTH, ACCENT, 2), margin: [0, 0, 0, 0] },
    ],
  });

  let firstSection = true;

  if (data.summary) {
    content.push({
      stack: [
        modernAccentHeading("Profile", firstSection),
        { text: data.summary, style: "modernBody", marginLeft: 12 },
      ],
    });
    firstSection = false;
  }

  if (data.experience.length > 0) {
    content.push({
      stack: [
        modernAccentHeading("Experience", firstSection),
        ...data.experience.map(modernExperienceEntry),
      ],
    });
    firstSection = false;
  }

  if (data.projects.length > 0) {
    content.push({
      stack: [
        modernAccentHeading("Projects", firstSection),
        ...data.projects.map((entry) => {
          const tech = (entry.technologies ?? []).filter((item) => !isBlank(item));
          return modernEduProjectEntry(
            entry.name,
            tech.length > 0 ? tech.join(", ") : null,
            []
          );
        }),
      ],
    });
    firstSection = false;
  }

  if (data.skills.length > 0) {
    const lines: Content[] = data.skills.map((group): Content => ({
      columns: [
        { text: group.label, style: "modernSkillLabel", width: "auto", marginRight: 6 },
        { text: group.items.join(" · "), style: "modernBody", width: "*" },
      ],
      margin: [0, 0, 0, 3],
      unbreakable: true,
    }));
    const left = lines.filter((_, index) => index % 2 === 0);
    const right = lines.filter((_, index) => index % 2 === 1);
    content.push({
      stack: [
        modernAccentHeading("Skills", firstSection),
        {
          columns: [
            { width: "*", stack: left },
            right.length > 0 ? { width: "*", stack: right } : { width: "*", text: "" },
          ],
          columnGap: 18,
        },
      ],
    });
    firstSection = false;
  }

  if (data.education.length > 0) {
    content.push({
      stack: [
        modernAccentHeading("Education", firstSection),
        ...data.education.map((entry) => {
          const range = dateRange(entry.start_date, entry.end_date);
          const degree = joinParts(entry.degree, entry.field);
          const details = (entry.details ?? []).filter((item) => !isBlank(item));
          return modernEduProjectEntry(
            entry.institution,
            joinParts(degree, entry.location, range),
            details
          );
        }),
      ],
    });
    firstSection = false;
  }

  if (data.certifications.length > 0) {
    content.push({
      stack: [
        modernAccentHeading("Certifications", firstSection),
        ...data.certifications.map((entry): Content => {
          const meta = joinParts(entry.issuer, entry.date);
          return {
            unbreakable: true,
            columns: [
              { text: entry.name, style: "modernEntryTitle", width: "*" },
              meta
                ? { text: meta, style: "modernSubtitle", width: "auto", alignment: "right" }
                : { text: "", width: "auto" },
            ],
            margin: [0, 0, 0, 6],
          };
        }),
      ],
    });
    firstSection = false;
  }

  for (const entry of data.custom) {
    const lines = (entry.content ?? []).filter((line) => !isBlank(line));
    content.push({
      stack: [
        modernAccentHeading(isBlank(entry.heading) ? "Additional" : (entry.heading as string), firstSection),
        lines.length > 0 ? bulletList(lines, "modernBody", ACCENT_SOFT, "square") : { text: "", style: "modernBody" },
      ],
    });
    firstSection = false;
  }

  return content;
}

/* ------------------------------------------------------------------ */
/* Compact                                                             */
/* ------------------------------------------------------------------ */

function compactExperienceEntry(entry: WorkExperience): Content {
  const range = dateRange(entry.start_date, entry.end_date);
  const title = nonBlank(entry.title) ?? nonBlank(entry.company) ?? "Experience";
  const parts: ContentText[] = [];
  if (!isBlank(entry.title) && !isBlank(entry.company)) {
    parts.push({ text: " · ", style: "compactBody" });
    parts.push({ text: nonBlank(entry.company) as string, style: "compactMuted" });
  }
  if (!isBlank(entry.location)) {
    parts.push({ text: " · ", style: "compactMuted" });
    parts.push({ text: nonBlank(entry.location) as string, style: "compactMuted" });
  }
  if (range) {
    parts.push({ text: ` (${range})`, style: "compactMuted" });
  }
  const bullets = (entry.achievements ?? []).filter((item) => !isBlank(item));
  const stack: Content[] = [
    { text: [{ text: title, bold: true, color: INK }, ...parts], style: "compactEntry", margin: [0, 0, 0, 1] },
  ];
  if (!isBlank(entry.description)) {
    stack.push({ text: entry.description as string, style: "compactBody", margin: [0, 0, 0, bullets.length > 0 ? 1 : 0] });
  }
  if (bullets.length > 0) {
    stack.push(bulletList(bullets, "compactBody", MUTED, "disc"));
  }
  return { unbreakable: true, stack, margin: [0, 0, 0, 6] };
}

function compactEducationEntry(entry: Education): Content {
  const range = dateRange(entry.start_date, entry.end_date);
  const degree = joinParts(entry.degree, entry.field);
  const details = (entry.details ?? []).filter((item) => !isBlank(item));
  const trailing: ContentText[] = [];
  if (degree) {
    trailing.push({ text: ` · ${degree}`, style: "compactMuted" });
  }
  if (range) {
    trailing.push({ text: ` (${range})`, style: "compactMuted" });
  }
  const stack: Content[] = [
    {
      text: [{ text: entry.institution, bold: true, color: INK }, ...trailing],
      style: "compactEntry",
      margin: [0, 0, 0, details.length > 0 ? 1 : 0],
    },
  ];
  if (details.length > 0) {
    stack.push({ text: details.join("; "), style: "compactBody" });
  }
  return { unbreakable: true, stack, margin: [0, 0, 0, 6] };
}

function compactProjectEntry(entry: Project): Content {
  const tech = (entry.technologies ?? []).filter((item) => !isBlank(item));
  const stack: Content[] = [
    {
      text: [
        { text: entry.name, bold: true, color: INK },
        tech.length > 0 ? { text: ` — ${tech.join(", ")}`, style: "compactMuted" } : { text: "", style: "compactMuted" },
      ],
      style: "compactEntry",
      margin: [0, 0, 0, !isBlank(entry.description) ? 1 : 0],
    },
  ];
  if (!isBlank(entry.description)) {
    stack.push({ text: entry.description as string, style: "compactBody" });
  }
  return { unbreakable: true, stack, margin: [0, 0, 0, 5] };
}

function compactCertificationEntry(entry: Certification): Content {
  const meta = joinParts(entry.issuer, entry.date);
  return {
    unbreakable: true,
    text: [
      { text: entry.name, bold: true, color: INK },
      meta ? { text: ` — ${meta}`, style: "compactMuted" } : { text: "", style: "compactMuted" },
    ],
    style: "compactEntry",
    margin: [0, 0, 0, 3],
  };
}

function buildCompact(resume: Resume): Content[] {
  const data = filterResume(resume);
  const content: Content[] = [];

  if (data.name || data.items.length > 0) {
    const header: Content[] = [];
    if (data.name) {
      header.push({ text: data.name, style: "compactName" });
    }
    const line = contactLine(data.items);
    if (line) {
      header.push({ text: line, style: "compactContact", margin: [0, 1, 0, 0] });
    }
    content.push({
      unbreakable: true,
      stack: header,
      margin: [0, 0, 0, 6],
    });
    content.push({ margin: [0, 0, 0, 4], ...divider(PDF_CONTENT_WIDTH, RULE_STRONG, 1.2) });
  }

  const section = (
    title: string,
    isFirst: boolean,
    body: Content[],
    spacing: number
  ): Content => {
    return {
      columns: [
        {
          width: COMPACT_LABEL_WIDTH,
          text: title.toUpperCase(),
          style: "compactHeading",
          headlineLevel: 1,
          margin: [0, 2, 0, 0],
        },
        {
          width: "*",
          stack: body,
          margin: [0, isFirst ? 0 : spacing, 0, 0],
        },
      ],
      columnGap: 8,
    };
  };

  let first = true;

  if (data.summary) {
    content.push(section("Summary", true, [{ text: data.summary, style: "compactBody" }], 6));
    first = false;
  }
  content.push(
    ...buildCompactSections(
      data,
      section,
      first
    )
  );
  return content;
}

function buildCompactSections(
  data: FilteredResume,
  section: (t: string, f: boolean, body: Content[], spacing: number) => Content,
  first: boolean
): Content[] {
  const out: Content[] = [];

  if (data.experience.length > 0) {
    out.push(section("Experience", first, data.experience.map(compactExperienceEntry), 4));
    first = false;
  }
  if (data.projects.length > 0) {
    out.push(section("Projects", first, data.projects.map(compactProjectEntry), 4));
    first = false;
  }
  if (data.skills.length > 0) {
    out.push(
      section(
        "Skills",
        first,
        data.skills.map((group) => ({
          text: [{ text: `${group.compactLabel}: `, bold: true, color: INK }, group.items.join(", ")],
          style: "compactBody",
          margin: [0, 0, 0, 1],
        })),
        4
      )
    );
    first = false;
  }
  if (data.education.length > 0) {
    out.push(section("Education", first, data.education.map(compactEducationEntry), 4));
    first = false;
  }
  if (data.certifications.length > 0) {
    out.push(section("Certifications", first, data.certifications.map(compactCertificationEntry), 4));
    first = false;
  }
  for (const entry of data.custom) {
    const lines = (entry.content ?? []).filter((line) => !isBlank(line));
    out.push(
      section(
        isBlank(entry.heading) ? "More" : (entry.heading as string),
        first,
        lines.length > 0
          ? [{ text: lines.join("; "), style: "compactBody" }]
          : [{ text: "", style: "compactBody" }],
        4
      )
    );
    first = false;
  }

  return out;
}

/* ------------------------------------------------------------------ */
/* Document                                                            */
/* ------------------------------------------------------------------ */

const CLASSIC_STYLES: TDocumentDefinitions["styles"] = {
  classicName: { fontSize: 20, bold: true, alignment: "center", color: INK },
  classicContact: { fontSize: 9.5, color: MUTED, alignment: "center" },
  classicHeading: {
    fontSize: 10,
    bold: true,
    characterSpacing: 1.8,
    color: BODY,
  },
  classicEntryTitle: { fontSize: 11, bold: true, color: INK },
  classicRange: { fontSize: 9.5, color: MUTED },
  classicSubtitle: { fontSize: 10, color: MUTED },
  classicBody: { fontSize: 10, color: BODY, lineHeight: 1.2 },
};

const MODERN_STYLES: TDocumentDefinitions["styles"] = {
  modernName: { fontSize: 21, bold: true, color: INK },
  modernContact: { fontSize: 9.5, color: MUTED },
  modernHeading: {
    fontSize: 10.5,
    bold: true,
    characterSpacing: 1.6,
    color: ACCENT_DARK,
  },
  modernEntryTitle: { fontSize: 11.5, bold: true, color: INK },
  modernRange: {
    fontSize: 9.5,
    color: ACCENT_DARK,
    characterSpacing: 0.4,
  },
  modernSubtitle: { fontSize: 10, color: MUTED },
  modernBody: { fontSize: 10, color: BODY, lineHeight: 1.35 },
  modernSkillLabel: { fontSize: 10, bold: true, color: INK },
};

const COMPACT_STYLES: TDocumentDefinitions["styles"] = {
  compactName: { fontSize: 16, bold: true, color: INK },
  compactContact: { fontSize: 9, color: MUTED },
  compactHeading: {
    fontSize: 8.5,
    bold: true,
    characterSpacing: 1,
    color: MUTED,
  },
  compactEntry: { fontSize: 10, color: INK, lineHeight: 1.15 },
  compactBody: { fontSize: 9.5, color: BODY, lineHeight: 1.15 },
  compactMuted: { fontSize: 9.5, color: MUTED, lineHeight: 1.15 },
};

/** Keep section headings with the following content instead of ending up alone. */
function headingPageBreak(currentNode: { headlineLevel?: number; startPosition: { verticalRatio: number } }): boolean {
  if (typeof currentNode.headlineLevel !== "number" || currentNode.headlineLevel <= 0) {
    return false;
  }
  return currentNode.startPosition.verticalRatio > 0.72;
}

export function buildResumePdfDocument(
  resume: Resume,
  templateId: TemplateId
): TDocumentDefinitions {
  const base: TDocumentDefinitions = {
    pageSize: PDF_PAGE_SIZE,
    pageMargins: [...PDF_PAGE_MARGINS],
    compress: true,
    content: [],
    defaultStyle: {
      font: "Roboto",
      fontSize: 10,
      color: BODY,
      lineHeight: 1.2,
    },
    pageBreakBefore: headingPageBreak,
    info: {
      title: `${nonBlank(resume.contact?.name) ?? "Resume"}`,
      subject: "Resume",
      creator: "ResumeForge",
    },
  };

  if (templateId === "modern") {
    base.styles = MODERN_STYLES;
    base.content = buildModern(resume);
  } else if (templateId === "compact") {
    base.styles = COMPACT_STYLES;
    base.content = buildCompact(resume);
  } else {
    base.styles = CLASSIC_STYLES;
    base.content = buildClassic(resume);
  }

  return base;
}