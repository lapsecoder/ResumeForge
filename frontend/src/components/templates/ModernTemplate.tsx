import type { ReactNode } from "react";

import type {
  Certification,
  CustomSection,
  Education,
  Project,
  Resume,
  WorkExperience,
} from "@/lib/resume";

import { contactItems, dateRange, isBlank } from "./shared";

function anyText(...values: (string | null | undefined)[]): boolean {
  return values.some((value) => !isBlank(value));
}

function ModernSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="resume-section mt-6">
      <h2 className="resume-heading border-l-4 border-indigo-500 pl-2 text-[10.5pt] font-semibold uppercase tracking-[0.16em] text-indigo-700">
        {title}
      </h2>
      <div className="mt-3 pl-3">{children}</div>
    </section>
  );
}

function ExperienceBlock({ entry }: { entry: WorkExperience }) {
  const range = dateRange(entry.start_date, entry.end_date);
  const bullets = (entry.achievements ?? []).filter((item) => !isBlank(item));
  return (
    <li className="resume-entry">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <p className="text-[11.5pt] font-semibold text-slate-900">
          {!isBlank(entry.title) ? entry.title : entry.company}
        </p>
        {range ? (
          <p className="text-[9.5pt] font-medium uppercase tracking-wide text-indigo-600">
            {range}
          </p>
        ) : null}
      </div>
      <p className="text-[10pt] font-medium text-slate-500">
        {[!isBlank(entry.title) ? entry.company : "", entry.location]
          .filter((part) => !isBlank(part))
          .join(" · ")}
      </p>
      {!isBlank(entry.description) ? (
        <p className="mt-1 text-[10pt] leading-relaxed text-slate-700">{entry.description}</p>
      ) : null}
      {bullets.length > 0 ? (
        <ul className="mt-1.5 space-y-1">
          {bullets.map((achievement, index) => (
            <li key={index} className="flex gap-2 text-[10pt] leading-relaxed text-slate-700">
              <span aria-hidden="true" className="mt-[3px] text-indigo-400">
                ▪
              </span>
              <span>{achievement}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function EducationBlock({ entry }: { entry: Education }) {
  const range = dateRange(entry.start_date, entry.end_date);
  const degree = [entry.degree, entry.field].filter((part) => !isBlank(part)).join(", ");
  const details = (entry.details ?? []).filter((item) => !isBlank(item));
  return (
    <li className="resume-entry">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <p className="text-[11pt] font-semibold text-slate-900">{entry.institution}</p>
        {range ? <p className="text-[9.5pt] text-slate-500">{range}</p> : null}
      </div>
      {degree || !isBlank(entry.location) ? (
        <p className="text-[10pt] text-slate-500">
          {[degree, entry.location].filter((part) => !isBlank(part)).join(" · ")}
        </p>
      ) : null}
      {details.length > 0 ? (
        <ul className="mt-1 space-y-0.5">
          {details.map((detail, index) => (
            <li key={index} className="text-[10pt] leading-relaxed text-slate-700">
              {detail}
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function ProjectBlock({ entry }: { entry: Project }) {
  const technologies = (entry.technologies ?? []).filter((item) => !isBlank(item));
  return (
    <li className="resume-entry">
      <p className="text-[11pt] font-semibold text-slate-900">{entry.name}</p>
      {!isBlank(entry.description) ? (
        <p className="text-[10pt] leading-relaxed text-slate-700">{entry.description}</p>
      ) : null}
      {technologies.length > 0 ? (
        <p className="mt-1 flex flex-wrap gap-1.5">
          {technologies.map((tech, index) => (
            <span
              key={index}
              className="rounded bg-slate-100 px-1.5 py-0.5 text-[9pt] text-slate-600"
            >
              {tech}
            </span>
          ))}
        </p>
      ) : null}
    </li>
  );
}

function CertificationBlock({ entry }: { entry: Certification }) {
  const meta = [entry.issuer, entry.date].filter((part) => !isBlank(part)).join(" · ");
  return (
    <li className="resume-entry flex flex-wrap items-baseline justify-between gap-x-4">
      <p className="text-[10.5pt] font-semibold text-slate-900">{entry.name}</p>
      {meta ? <p className="text-[9.5pt] text-slate-500">{meta}</p> : null}
    </li>
  );
}

function CustomBlock({ entry }: { entry: CustomSection }) {
  const lines = (entry.content ?? []).filter((line) => !isBlank(line));
  return (
    <div className="resume-entry">
      {lines.length > 0 ? (
        <ul className="space-y-0.5">
          {lines.map((line, index) => (
            <li key={index} className="text-[10pt] leading-relaxed text-slate-700">
              {line}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function ModernTemplate({ resume }: { resume: Resume }) {
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
  const skills = resume.skills;
  const skillGroups = [
    ["Technical", skills?.technical],
    ["Tools", skills?.tools],
    ["Languages", skills?.languages],
    ["Soft", skills?.soft],
  ] as const;
  const hasSkills = skillGroups.some(([, items]) =>
    (items ?? []).some((item) => !isBlank(item))
  );
  const items = contactItems(resume.contact);
  const name = resume.contact?.name;

  return (
    <article className="text-slate-800">
      <header className="border-b-2 border-indigo-500 pb-3">
        <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
          <div>
            {!isBlank(name) ? (
              <h1 className="text-[21pt] font-bold leading-tight tracking-tight text-slate-900">
                {name}
              </h1>
            ) : null}
            {!isBlank(resume.contact?.location) ? (
              <p className="text-[10pt] text-slate-500">{resume.contact?.location}</p>
            ) : null}
          </div>
          {items.length > 0 ? (
            <ul className="text-right text-[9.5pt] leading-relaxed text-slate-600">
              {items.map((item) => (
                <li key={item.key}>
                  {item.href ? (
                    <a href={item.href} className="text-indigo-700 underline-offset-2">
                      {item.value}
                    </a>
                  ) : (
                    item.value
                  )}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </header>

      {!isBlank(resume.summary) ? (
        <ModernSection title="Profile">
          <p className="text-[10pt] leading-relaxed text-slate-700">{resume.summary}</p>
        </ModernSection>
      ) : null}

      {experience.length > 0 ? (
        <ModernSection title="Experience">
          <ul className="space-y-3.5">
            {experience.map((entry, index) => (
              <ExperienceBlock key={index} entry={entry} />
            ))}
          </ul>
        </ModernSection>
      ) : null}

      {projects.length > 0 ? (
        <ModernSection title="Projects">
          <ul className="space-y-3">
            {projects.map((entry, index) => (
              <ProjectBlock key={index} entry={entry} />
            ))}
          </ul>
        </ModernSection>
      ) : null}

      {hasSkills ? (
        <ModernSection title="Skills">
          <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
            {skillGroups.map(([label, group]) => {
              const visible = (group ?? []).filter((item) => !isBlank(item));
              if (visible.length === 0) return null;
              return (
                <div key={label} className="flex gap-2 text-[10pt]">
                  <dt className="shrink-0 font-semibold text-slate-900">{label}</dt>
                  <dd className="text-slate-700">{visible.join(" · ")}</dd>
                </div>
              );
            })}
          </dl>
        </ModernSection>
      ) : null}

      {education.length > 0 ? (
        <ModernSection title="Education">
          <ul className="space-y-3">
            {education.map((entry, index) => (
              <EducationBlock key={index} entry={entry} />
            ))}
          </ul>
        </ModernSection>
      ) : null}

      {certifications.length > 0 ? (
        <ModernSection title="Certifications">
          <ul className="space-y-2">
            {certifications.map((entry, index) => (
              <CertificationBlock key={index} entry={entry} />
            ))}
          </ul>
        </ModernSection>
      ) : null}

      {custom.map((entry, index) => (
        <ModernSection
          key={index}
          title={isBlank(entry.heading) ? "Additional" : entry.heading}
        >
          <CustomBlock entry={entry} />
        </ModernSection>
      ))}
    </article>
  );
}
