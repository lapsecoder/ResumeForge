"use client";

import type { ReactNode } from "react";

import type {
  Certification,
  CustomSection,
  Education,
  Project,
  Resume,
  WorkExperience,
} from "@/lib/resume";

interface ResultsViewProps {
  resume: Resume;
}

export function ResultsView({ resume }: ResultsViewProps) {
  const { contact, summary, experience, education, skills, projects, certifications } = resume;
  const customSections: CustomSection[] = resume.custom_sections ?? [];

  const hasContact = Boolean(
    contact &&
      (contact.name || contact.email || contact.phone || contact.location ||
        contact.linkedin || contact.github || contact.website)
  );

  return (
    <div className="space-y-6">
      {hasContact && (
        <Section title="Resume overview">
          <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
            <Field label="Name" value={contact?.name} />
            <Field label="Email" value={contact?.email} />
            <Field label="Phone" value={contact?.phone} />
            <Field label="Location" value={contact?.location} />
            <Field label="LinkedIn" value={contact?.linkedin} isLink />
            <Field label="GitHub" value={contact?.github} isLink />
            <Field label="Website" value={contact?.website} isLink />
          </dl>
        </Section>
      )}

      {summary ? (
        <Section title="Summary">
          <p className="text-sm leading-relaxed text-slate-700">{summary}</p>
        </Section>
      ) : null}

      <ExperienceList entries={experience} />
      <EducationList entries={education} />
      <SkillsSection
        technical={skills?.technical ?? []}
        soft={skills?.soft ?? []}
        tools={skills?.tools ?? []}
        languages={skills?.languages ?? []}
      />
      <ProjectList entries={projects} />
      <CertificationList entries={certifications} />
      <CustomSectionList entries={customSections} />
      <ConfidenceNote resume={resume} />
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={`section-${title.toLowerCase().replace(/\s+/g, "-")}`}>
      <h3
        id={`section-${title.toLowerCase().replace(/\s+/g, "-")}`}
        className="text-sm font-semibold uppercase tracking-wide text-indigo-600"
      >
        {title}
      </h3>
      <div className="mt-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        {children}
      </div>
    </section>
  );
}

function Field({
  label,
  value,
  isLink = false,
}: {
  label: string;
  value: string | null | undefined;
  isLink?: boolean;
}) {
  if (!value) return null;
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="mt-0.5 break-words text-sm text-slate-800">
        {isLink && /^https?:\/\//i.test(value) ? (
          <a
            href={value}
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 underline decoration-indigo-200 underline-offset-2 hover:text-indigo-700"
          >
            {value}
          </a>
        ) : (
          value
        )}
      </dd>
    </div>
  );
}

function ExperienceList({ entries }: { entries: WorkExperience[] | undefined }) {
  if (!entries || entries.length === 0) return null;
  return (
    <Section title="Experience">
      <ul className="space-y-5">
        {entries.map((entry, index) => (
          <li key={`${entry.company}-${index}`} className="space-y-1.5">
            <div>
              <p className="text-sm font-semibold text-slate-800">
                {entry.title || entry.company}
              </p>
              <p className="text-sm text-slate-500">
                {entry.company !== entry.title && entry.company ? entry.company : null}
                {entry.location ? ` · ${entry.location}` : ""}
              </p>
            </div>
            {(entry.start_date || entry.end_date) && (
              <p className="text-xs font-medium text-slate-400">
                {[entry.start_date, entry.end_date].filter(Boolean).join(" – ")}
              </p>
            )}
            {entry.description ? (
              <p className="text-sm leading-relaxed text-slate-700">{entry.description}</p>
            ) : null}
            {entry.achievements && entry.achievements.length > 0 ? (
              <ul className="space-y-1 pt-1">
                {entry.achievements.map((achievement, i) => (
                  <li key={i} className="flex gap-2 text-sm text-slate-700">
                    <span aria-hidden="true" className="mt-0.5 text-indigo-400">
                      •
                    </span>
                    <span>{achievement}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
    </Section>
  );
}

function EducationList({ entries }: { entries: Education[] | undefined }) {
  if (!entries || entries.length === 0) return null;
  return (
    <Section title="Education">
      <ul className="space-y-4">
        {entries.map((entry, index) => (
          <li key={`${entry.institution}-${index}`} className="space-y-1">
            <p className="text-sm font-semibold text-slate-800">
              {[entry.degree, entry.field].filter(Boolean).join(", ")}
            </p>
            <p className="text-sm text-slate-600">{entry.institution}</p>
            {(entry.start_date || entry.end_date) && (
              <p className="text-xs font-medium text-slate-400">
                {[entry.start_date, entry.end_date].filter(Boolean).join(" – ")}
              </p>
            )}
            {entry.details && entry.details.length > 0 ? (
              <ul className="space-y-1 pt-1">
                {entry.details.map((detail, i) => (
                  <li key={i} className="flex gap-2 text-sm text-slate-700">
                    <span aria-hidden="true" className="mt-0.5 text-indigo-400">
                      •
                    </span>
                    <span>{detail}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
    </Section>
  );
}

function SkillsSection({
  technical,
  soft,
  tools,
  languages,
}: {
  technical: string[];
  soft: string[];
  tools: string[];
  languages: string[];
}) {
  if (technical.length === 0 && soft.length === 0 && tools.length === 0 && languages.length === 0) {
    return null;
  }
  return (
    <Section title="Skills">
      <div className="space-y-4">
        <SkillGroup label="Technical skills" items={technical} />
        <SkillGroup label="Soft skills" items={soft} />
        <SkillGroup label="Tools" items={tools} />
        <SkillGroup label="Languages" items={languages} />
      </div>
    </Section>
  );
}

function SkillGroup({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <ul className="mt-2 flex flex-wrap gap-2">
        {items.map((item) => (
          <li
            key={item}
            className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
          >
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ProjectList({ entries }: { entries: Project[] | undefined }) {
  if (!entries || entries.length === 0) return null;
  return (
    <Section title="Projects">
      <ul className="space-y-5">
        {entries.map((entry, index) => (
          <li key={`${entry.name}-${index}`} className="space-y-1.5">
            <p className="text-sm font-semibold text-slate-800">
              {entry.url && /^https?:\/\//i.test(entry.url) ? (
                <a
                  href={entry.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-indigo-600 underline decoration-indigo-200 underline-offset-2 hover:text-indigo-700"
                >
                  {entry.name}
                </a>
              ) : (
                entry.name
              )}
            </p>
            {entry.description ? (
              <p className="text-sm leading-relaxed text-slate-700">{entry.description}</p>
            ) : null}
            {entry.technologies && entry.technologies.length > 0 ? (
              <ul className="flex flex-wrap gap-2">
                {entry.technologies.map((tech) => (
                  <li
                    key={tech}
                    className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600"
                  >
                    {tech}
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
    </Section>
  );
}

function CertificationList({ entries }: { entries: Certification[] | undefined }) {
  if (!entries || entries.length === 0) return null;
  return (
    <Section title="Certifications">
      <ul className="space-y-3">
        {entries.map((entry, index) => (
          <li key={`${entry.name}-${index}`}>
            <p className="text-sm font-semibold text-slate-800">
              {entry.url && /^https?:\/\//i.test(entry.url) ? (
                <a
                  href={entry.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-indigo-600 underline decoration-indigo-200 underline-offset-2 hover:text-indigo-700"
                >
                  {entry.name}
                </a>
              ) : (
                entry.name
              )}
            </p>
            {(entry.issuer || entry.date) && (
              <p className="text-sm text-slate-500">
                {[entry.issuer, entry.date].filter(Boolean).join(" · ")}
              </p>
            )}
          </li>
        ))}
      </ul>
    </Section>
  );
}

function CustomSectionList({ entries }: { entries: CustomSection[] }) {
  if (entries.length === 0) return null;
  return (
    <Section title="Other sections">
      <ul className="space-y-4">
        {entries.map((entry) => (
          <li key={entry.heading}>
            <p className="text-sm font-semibold text-slate-800">{entry.heading}</p>
            {entry.content && entry.content.length > 0 ? (
              <ul className="mt-1 space-y-0.5">
                {entry.content.map((line, i) => (
                  <li key={i} className="text-sm text-slate-700">
                    {line}
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
    </Section>
  );
}

function ConfidenceNote({ resume }: { resume: Resume }) {
  const overall = resume.metadata?.overall_confidence;
  const sections = resume.metadata?.section_confidence ?? [];
  if (!overall && sections.length === 0) return null;

  const levelStyle =
    overall === "high"
      ? "bg-emerald-50 text-emerald-700"
      : overall === "medium"
        ? "bg-amber-50 text-amber-700"
        : "bg-slate-100 text-slate-600";

  return (
    <section aria-label="Parsing confidence" className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Heuristic parsing confidence
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
        {overall ? (
          <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${levelStyle}`}>
            Overall: {overall}
          </span>
        ) : null}
        {sections.map((section) => (
          <span
            key={section.section}
            className="inline-flex rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-600"
          >
            {section.section}: {section.level}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Rough quality signal from the parser rules — not a calibrated probability.
      </p>
    </section>
  );
}