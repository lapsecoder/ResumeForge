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

function ClassicSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="resume-section mt-5">
      <h2 className="resume-heading border-b border-slate-300 pb-1 text-[10pt] font-bold uppercase tracking-[0.18em] text-slate-700">
        {title}
      </h2>
      <div className="mt-2.5">{children}</div>
    </section>
  );
}

function ExperienceBlock({ entry }: { entry: WorkExperience }) {
  const range = dateRange(entry.start_date, entry.end_date);
  return (
    <li className="resume-entry">
      <div className="flex items-baseline justify-between gap-4">
        <p className="text-[11pt] font-semibold text-slate-900">
          {!isBlank(entry.title) ? entry.title : entry.company}
        </p>
        {range ? <p className="shrink-0 text-[9.5pt] text-slate-500">{range}</p> : null}
      </div>
      <p className="text-[10pt] text-slate-600">
        {[!isBlank(entry.title) ? entry.company : "", entry.location]
          .filter((part) => !isBlank(part))
          .join(" · ")}
      </p>
      {!isBlank(entry.description) ? (
        <p className="mt-1 text-[10pt] leading-snug text-slate-700">{entry.description}</p>
      ) : null}
      {entry.achievements?.filter((a) => !isBlank(a)).length ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-5">
          {entry.achievements.filter((a) => !isBlank(a)).map((achievement, index) => (
            <li key={index} className="text-[10pt] leading-snug text-slate-700">
              {achievement}
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
  return (
    <li className="resume-entry">
      <div className="flex items-baseline justify-between gap-4">
        <p className="text-[11pt] font-semibold text-slate-900">{entry.institution}</p>
        {range ? <p className="shrink-0 text-[9.5pt] text-slate-500">{range}</p> : null}
      </div>
      <p className="text-[10pt] text-slate-600">
        {[degree, entry.location].filter((part) => !isBlank(part)).join(" · ")}
      </p>
      {entry.details?.filter((d) => !isBlank(d)).length ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-5">
          {entry.details.filter((d) => !isBlank(d)).map((detail, index) => (
            <li key={index} className="text-[10pt] leading-snug text-slate-700">
              {detail}
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function ProjectBlock({ entry }: { entry: Project }) {
  return (
    <li className="resume-entry">
      <p className="text-[11pt] font-semibold text-slate-900">{entry.name}</p>
      {!isBlank(entry.description) ? (
        <p className="text-[10pt] leading-snug text-slate-700">{entry.description}</p>
      ) : null}
      {entry.technologies?.filter((t) => !isBlank(t)).length ? (
        <p className="text-[9.5pt] text-slate-500">
          {entry.technologies.filter((t) => !isBlank(t)).join(", ")}
        </p>
      ) : null}
    </li>
  );
}

function CertificationBlock({ entry }: { entry: Certification }) {
  const meta = [entry.issuer, entry.date].filter((part) => !isBlank(part)).join(" · ");
  return (
    <li className="resume-entry">
      <p className="text-[11pt] font-semibold text-slate-900">{entry.name}</p>
      {meta ? <p className="text-[10pt] text-slate-500">{meta}</p> : null}
    </li>
  );
}

export function ClassicTemplate({ resume }: { resume: Resume }) {
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
    ["Technical Skills", skills?.technical],
    ["Tools", skills?.tools],
    ["Languages", skills?.languages],
    ["Soft Skills", skills?.soft],
  ] as const;
  const hasSkills = skillGroups.some(([, items]) =>
    (items ?? []).some((item) => !isBlank(item))
  );
  const items = contactItems(resume.contact);
  const name = resume.contact?.name;

  return (
    <article className="text-slate-800">
      {!isBlank(name) || items.length > 0 ? (
        <header className="text-center">
          {!isBlank(name) ? (
            <h1 className="text-[20pt] font-bold leading-tight tracking-tight text-slate-900">
              {name}
            </h1>
          ) : null}
          {items.length > 0 ? (
            <p className="mt-1 text-[9.5pt] text-slate-600">
              {items.map((item, index) => (
                <span key={item.key}>
                  {index > 0 ? <span className="text-slate-400"> · </span> : null}
                  {item.href ? (
                    <a href={item.href} className="text-slate-700 underline-offset-2">
                      {item.value}
                    </a>
                  ) : (
                    item.value
                  )}
                </span>
              ))}
            </p>
          ) : null}
        </header>
      ) : null}

      {!isBlank(resume.summary) ? (
        <ClassicSection title="Summary">
          <p className="text-[10pt] leading-snug text-slate-700">{resume.summary}</p>
        </ClassicSection>
      ) : null}

      {experience.length > 0 ? (
        <ClassicSection title="Experience">
          <ul className="space-y-3">
            {experience.map((entry, index) => (
              <ExperienceBlock key={index} entry={entry} />
            ))}
          </ul>
        </ClassicSection>
      ) : null}

      {education.length > 0 ? (
        <ClassicSection title="Education">
          <ul className="space-y-3">
            {education.map((entry, index) => (
              <EducationBlock key={index} entry={entry} />
            ))}
          </ul>
        </ClassicSection>
      ) : null}

      {projects.length > 0 ? (
        <ClassicSection title="Projects">
          <ul className="space-y-3">
            {projects.map((entry, index) => (
              <ProjectBlock key={index} entry={entry} />
            ))}
          </ul>
        </ClassicSection>
      ) : null}

      {hasSkills ? (
        <ClassicSection title="Skills">
          <ul className="space-y-1">
            {skillGroups.map(([label, group]) => {
              const visible = (group ?? []).filter((item) => !isBlank(item));
              if (visible.length === 0) return null;
              return (
                <li key={label} className="text-[10pt] leading-snug text-slate-700">
                  <span className="font-semibold text-slate-900">{label}: </span>
                  {visible.join(", ")}
                </li>
              );
            })}
          </ul>
        </ClassicSection>
      ) : null}

      {certifications.length > 0 ? (
        <ClassicSection title="Certifications">
          <ul className="space-y-2">
            {certifications.map((entry, index) => (
              <CertificationBlock key={index} entry={entry} />
            ))}
          </ul>
        </ClassicSection>
      ) : null}

      {custom.map((entry, index) => (
        <ClassicSection key={index} title={isBlank(entry.heading) ? "Additional" : entry.heading}>
          <CustomSectionLines entry={entry} />
        </ClassicSection>
      ))}
    </article>
  );
}

function CustomSectionLines({ entry }: { entry: CustomSection }) {
  const lines = (entry.content ?? []).filter((line) => !isBlank(line));
  if (lines.length === 0) return null;
  return (
    <ul className="list-disc space-y-0.5 pl-5">
      {lines.map((line, index) => (
        <li key={index} className="text-[10pt] leading-snug text-slate-700">
          {line}
        </li>
      ))}
    </ul>
  );
}
