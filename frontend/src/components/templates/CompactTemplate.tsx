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

function CompactSection({
  title,
  children,
  first = false,
}: {
  title: string;
  children: ReactNode;
  first?: boolean;
}) {
  return (
    <section
      className={`resume-section grid grid-cols-[84px_1fr] gap-x-3 ${first ? "mt-0" : "mt-2.5"}`}
    >
      <h2 className="resume-heading pt-[1px] text-[8.5pt] font-bold uppercase tracking-[0.12em] text-slate-500">
        {title}
      </h2>
      <div className="resume-section-body">{children}</div>
    </section>
  );
}

function ExperienceBlock({ entry }: { entry: WorkExperience }) {
  const range = dateRange(entry.start_date, entry.end_date);
  const bullets = (entry.achievements ?? []).filter((item) => !isBlank(item));
  return (
    <li className="resume-entry">
      <p className="text-[10pt] leading-snug text-slate-900">
        <span className="font-semibold">
          {!isBlank(entry.title) ? entry.title : entry.company}
        </span>
        {!isBlank(entry.title) && !isBlank(entry.company) ? (
          <span className="text-slate-600"> · {entry.company}</span>
        ) : null}
        {!isBlank(entry.location) ? (
          <span className="text-slate-500"> · {entry.location}</span>
        ) : null}
        {range ? <span className="text-slate-500"> ({range})</span> : null}
      </p>
      {!isBlank(entry.description) ? (
        <p className="text-[9.5pt] leading-snug text-slate-700">{entry.description}</p>
      ) : null}
      {bullets.length > 0 ? (
        <ul className="mt-0.5 list-disc space-y-0 pl-4">
          {bullets.map((achievement, index) => (
            <li key={index} className="text-[9.5pt] leading-snug text-slate-700">
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
  const details = (entry.details ?? []).filter((item) => !isBlank(item));
  return (
    <li className="resume-entry">
      <p className="text-[10pt] leading-snug text-slate-900">
        <span className="font-semibold">{entry.institution}</span>
        {degree ? <span className="text-slate-600"> · {degree}</span> : null}
        {range ? <span className="text-slate-500"> ({range})</span> : null}
      </p>
      {details.length > 0 ? (
        <p className="text-[9.5pt] leading-snug text-slate-700">{details.join("; ")}</p>
      ) : null}
    </li>
  );
}

function ProjectBlock({ entry }: { entry: Project }) {
  const technologies = (entry.technologies ?? []).filter((item) => !isBlank(item));
  return (
    <li className="resume-entry">
      <p className="text-[10pt] leading-snug text-slate-900">
        <span className="font-semibold">{entry.name}</span>
        {technologies.length > 0 ? (
          <span className="text-slate-500"> — {technologies.join(", ")}</span>
        ) : null}
      </p>
      {!isBlank(entry.description) ? (
        <p className="text-[9.5pt] leading-snug text-slate-700">{entry.description}</p>
      ) : null}
    </li>
  );
}

function CertificationBlock({ entry }: { entry: Certification }) {
  const meta = [entry.issuer, entry.date].filter((part) => !isBlank(part)).join(" · ");
  return (
    <li className="resume-entry text-[9.5pt] leading-snug text-slate-800">
      <span className="font-semibold">{entry.name}</span>
      {meta ? <span className="text-slate-500"> — {meta}</span> : null}
    </li>
  );
}

function CustomBlock({ entry }: { entry: CustomSection }) {
  const lines = (entry.content ?? []).filter((line) => !isBlank(line));
  return (
    <div className="resume-entry text-[9.5pt] leading-snug text-slate-700">
      {lines.join("; ")}
    </div>
  );
}

export function CompactTemplate({ resume }: { resume: Resume }) {
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
      {!isBlank(name) || items.length > 0 ? (
        <header className="border-b border-slate-400 pb-1.5">
          {!isBlank(name) ? (
            <h1 className="text-[16pt] font-bold leading-tight tracking-tight text-slate-900">
              {name}
            </h1>
          ) : null}
          {items.length > 0 ? (
            <p className="text-[9pt] leading-snug text-slate-600">
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
        <CompactSection title="Summary" first>
          <p className="text-[9.5pt] leading-snug text-slate-700">{resume.summary}</p>
        </CompactSection>
      ) : null}

      {experience.length > 0 ? (
        <CompactSection title="Experience" first={isBlank(resume.summary)}>
          <ul className="space-y-2">
            {experience.map((entry, index) => (
              <ExperienceBlock key={index} entry={entry} />
            ))}
          </ul>
        </CompactSection>
      ) : null}

      {projects.length > 0 ? (
        <CompactSection title="Projects" first={isBlank(resume.summary) && experience.length === 0}>
          <ul className="space-y-1.5">
            {projects.map((entry, index) => (
              <ProjectBlock key={index} entry={entry} />
            ))}
          </ul>
        </CompactSection>
      ) : null}

      {hasSkills ? (
        <CompactSection
          title="Skills"
          first={isBlank(resume.summary) && experience.length === 0 && projects.length === 0}
        >
          <ul className="space-y-0.5">
            {skillGroups.map(([label, group]) => {
              const visible = (group ?? []).filter((item) => !isBlank(item));
              if (visible.length === 0) return null;
              return (
                <li key={label} className="text-[9.5pt] leading-snug text-slate-700">
                  <span className="font-semibold text-slate-900">{label}: </span>
                  {visible.join(", ")}
                </li>
              );
            })}
          </ul>
        </CompactSection>
      ) : null}

      {education.length > 0 ? (
        <CompactSection
          title="Education"
          first={
            isBlank(resume.summary) &&
            experience.length === 0 &&
            projects.length === 0 &&
            !hasSkills
          }
        >
          <ul className="space-y-1.5">
            {education.map((entry, index) => (
              <EducationBlock key={index} entry={entry} />
            ))}
          </ul>
        </CompactSection>
      ) : null}

      {certifications.length > 0 ? (
        <CompactSection title="Certifications" first={false}>
          <ul className="space-y-0.5">
            {certifications.map((entry, index) => (
              <CertificationBlock key={index} entry={entry} />
            ))}
          </ul>
        </CompactSection>
      ) : null}

      {custom.map((entry, index) => (
        <CompactSection
          key={index}
          title={isBlank(entry.heading) ? "More" : entry.heading}
          first={false}
        >
          <CustomBlock entry={entry} />
        </CompactSection>
      ))}
    </article>
  );
}
