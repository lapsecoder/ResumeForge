"use client";

import { useCallback, useState, type ReactNode } from "react";

import {
  addEntry,
  addEntryListItem,
  addSkill,
  CONTACT_FIELD_KEYS,
  CONTACT_FIELD_LABELS,
  moveEntry,
  moveEntryListItem,
  moveSkill,
  removeEntry,
  removeEntryListItem,
  removeSkill,
  SKILL_GROUP_LABELS,
  SKILL_GROUP_KEYS,
  updateContactField,
  updateEntryField,
  updateEntryListItem,
  updateSkill,
  updateSummary,
  type ArraySectionKey,
  type ContactFieldKey,
  type SkillGroupKey,
} from "@/lib/builder";
import type { HistoryCommitOptions } from "@/lib/history";
import type { Resume } from "@/lib/resume";

export type CommitResume = (resume: Resume, options?: HistoryCommitOptions) => void;

interface BuilderEditorProps {
  resume: Resume;
  onChange: CommitResume;
}

const inputClass =
  "mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1";

const labelClass = "text-xs font-medium uppercase tracking-wide text-slate-500";

const iconButtonClass =
  "rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-semibold text-slate-600 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-40";

export function BuilderEditor({ resume, onChange }: BuilderEditorProps) {
  const commit = useCallback(
    (next: Resume | null, mergeKey?: string) => {
      if (!next) return;
      onChange(next, mergeKey ? { mergeKey } : undefined);
    },
    [onChange]
  );

  const skills = resume.skills ?? {};

  return (
    <div className="space-y-6">
      <SectionBlock title="Contact">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {CONTACT_FIELD_KEYS.map((field) => (
            <ContactField
              key={field}
              field={field}
              value={resume.contact?.[field] ?? ""}
              onChange={(value) => commit(updateContactField(resume, field, value), `contact.${field}`)}
            />
          ))}
        </div>
      </SectionBlock>

      <SectionBlock title="Summary">
        <ListField
          id="summary"
          label="Professional summary"
          value={resume.summary ?? ""}
          multiline
          onChange={(value) => commit(updateSummary(resume, value), "summary")}
        />
      </SectionBlock>

      <SectionBlock
        title="Experience"
        onAdd={() => commit(addEntry(resume, "experience"))}
        addLabel="Add experience"
      >
        <EntryList
          section="experience"
          entries={resume.experience ?? []}
          render={(index) => (
            <ExperienceFields resume={resume} index={index} commit={commit} />
          )}
          onMove={(index, direction) => commit(moveEntry(resume, "experience", index, direction))}
          onRemove={(index) => commit(removeEntry(resume, "experience", index))}
          noun="experience"
        />
      </SectionBlock>

      <SectionBlock
        title="Education"
        onAdd={() => commit(addEntry(resume, "education"))}
        addLabel="Add education"
      >
        <EntryList
          section="education"
          entries={resume.education ?? []}
          render={(index) => <EducationFields resume={resume} index={index} commit={commit} />}
          onMove={(index, direction) => commit(moveEntry(resume, "education", index, direction))}
          onRemove={(index) => commit(removeEntry(resume, "education", index))}
          noun="education"
        />
      </SectionBlock>

      <SectionBlock
        title="Skills"
      >
        <div className="space-y-4">
          {SKILL_GROUP_KEYS.map((group) => (
            <SkillsGroup
              key={group}
              group={group}
              items={skills[group] ?? []}
              onAdd={(value) => commit(addSkill(resume, group, value))}
              onUpdate={(index, value) =>
                commit(updateSkill(resume, group, index, value), `skills.${group}.${index}`)
              }
              onRemove={(index) => commit(removeSkill(resume, group, index))}
              onMove={(index, direction) => commit(moveSkill(resume, group, index, direction))}
            />
          ))}
        </div>
      </SectionBlock>

      <SectionBlock
        title="Projects"
        onAdd={() => commit(addEntry(resume, "projects"))}
        addLabel="Add project"
      >
        <EntryList
          section="projects"
          entries={resume.projects ?? []}
          render={(index) => <ProjectFields resume={resume} index={index} commit={commit} />}
          onMove={(index, direction) => commit(moveEntry(resume, "projects", index, direction))}
          onRemove={(index) => commit(removeEntry(resume, "projects", index))}
          noun="project"
        />
      </SectionBlock>

      <SectionBlock
        title="Certifications"
        onAdd={() => commit(addEntry(resume, "certifications"))}
        addLabel="Add certification"
      >
        <EntryList
          section="certifications"
          entries={resume.certifications ?? []}
          render={(index) => <CertificationFields resume={resume} index={index} commit={commit} />}
          onMove={(index, direction) =>
            commit(moveEntry(resume, "certifications", index, direction))
          }
          onRemove={(index) => commit(removeEntry(resume, "certifications", index))}
          noun="certification"
        />
      </SectionBlock>

      <SectionBlock
        title="Custom sections"
        onAdd={() => commit(addEntry(resume, "custom_sections"))}
        addLabel="Add custom section"
      >
        <EntryList
          section="custom_sections"
          entries={resume.custom_sections ?? []}
          render={(index) => <CustomSectionFields resume={resume} index={index} commit={commit} />}
          onMove={(index, direction) =>
            commit(moveEntry(resume, "custom_sections", index, direction))
          }
          onRemove={(index) => commit(removeEntry(resume, "custom_sections", index))}
          noun="custom section"
        />
      </SectionBlock>
    </div>
  );
}

function SectionBlock({
  title,
  children,
  onAdd,
  addLabel,
}: {
  title: string;
  children: ReactNode;
  onAdd?: () => void;
  addLabel?: string;
}) {
  return (
    <section aria-label={title} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-indigo-600">{title}</h3>
        {onAdd ? (
          <button type="button" onClick={onAdd} className={iconButtonClass}>
            {addLabel ?? `Add ${title.toLowerCase()}`}
          </button>
        ) : null}
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function ListField({
  id,
  label,
  value,
  onChange,
  multiline = false,
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  multiline?: boolean;
  placeholder?: string;
}) {
  return (
    <div>
      <label htmlFor={id} className={labelClass}>
        {label}
      </label>
      {multiline ? (
        <textarea
          id={id}
          value={value}
          rows={3}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
          className={inputClass}
        />
      ) : (
        <input
          id={id}
          type="text"
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
          className={inputClass}
        />
      )}
    </div>
  );
}

function ContactField({
  field,
  value,
  onChange,
}: {
  field: ContactFieldKey;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <ListField
      id={`contact-${field}`}
      label={CONTACT_FIELD_LABELS[field]}
      value={value}
      onChange={onChange}
    />
  );
}

function EntryList({
  entries,
  render,
  onMove,
  onRemove,
  noun,
}: {
  section: ArraySectionKey;
  entries: unknown[];
  render: (index: number) => ReactNode;
  onMove: (index: number, direction: -1 | 1) => void;
  onRemove: (index: number) => void;
  noun: string;
}) {
  if (entries.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-center text-sm text-slate-500">
        No {noun} added yet.
      </p>
    );
  }
  return (
    <ul className="space-y-4">
      {entries.map((_, index) => (
        <li key={index} className="rounded-lg border border-slate-200 bg-slate-50/60 p-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              {noun} {index + 1}
            </span>
            <EntryActions
              noun={noun}
              index={index}
              count={entries.length}
              onMove={onMove}
              onRemove={onRemove}
            />
          </div>
          <div className="space-y-3">{render(index)}</div>
        </li>
      ))}
    </ul>
  );
}

function EntryActions({
  noun,
  index,
  count,
  onMove,
  onRemove,
}: {
  noun: string;
  index: number;
  count: number;
  onMove: (index: number, direction: -1 | 1) => void;
  onRemove: (index: number) => void;
}) {
  return (
    <span className="flex gap-1">
      <button
        type="button"
        className={iconButtonClass}
        onClick={() => onMove(index, -1)}
        disabled={index === 0}
        aria-label={`Move ${noun} ${index + 1} up`}
      >
        ↑
      </button>
      <button
        type="button"
        className={iconButtonClass}
        onClick={() => onMove(index, 1)}
        disabled={index === count - 1}
        aria-label={`Move ${noun} ${index + 1} down`}
      >
        ↓
      </button>
      <button
        type="button"
        className={iconButtonClass}
        onClick={() => onRemove(index)}
        aria-label={`Remove ${noun} ${index + 1}`}
      >
        Remove
      </button>
    </span>
  );
}

function TextItemList({
  section,
  index,
  field,
  label,
  items,
  commit,
  resume,
  addLabel,
}: {
  section: ArraySectionKey;
  index: number;
  field: string;
  label: string;
  items: string[];
  commit: (next: Resume | null, mergeKey?: string) => void;
  resume: Resume;
  addLabel: string;
}) {
  return (
    <div>
      <p className={labelClass}>{label}</p>
      <ul className="mt-1 space-y-2">
        {items.map((item, itemIndex) => (
          <li key={itemIndex} className="flex items-start gap-2">
            <div className="flex-1">
              <ListField
                id={`${section}-${index}-${field}-${itemIndex}`}
                label={`${label} ${itemIndex + 1}`}
                value={item}
                onChange={(value) =>
                  commit(
                    updateEntryListItem(resume, section, index, field, itemIndex, value),
                    `${section}.${index}.${field}.${itemIndex}`
                  )
                }
              />
            </div>
            <span className="mt-6 flex gap-1">
              <button
                type="button"
                className={iconButtonClass}
                onClick={() => commit(moveEntryListItem(resume, section, index, field, itemIndex, -1))}
                disabled={itemIndex === 0}
                aria-label={`Move ${label} ${itemIndex + 1} up`}
              >
                ↑
              </button>
              <button
                type="button"
                className={iconButtonClass}
                onClick={() => commit(moveEntryListItem(resume, section, index, field, itemIndex, 1))}
                disabled={itemIndex === items.length - 1}
                aria-label={`Move ${label} ${itemIndex + 1} down`}
              >
                ↓
              </button>
              <button
                type="button"
                className={iconButtonClass}
                onClick={() => commit(removeEntryListItem(resume, section, index, field, itemIndex))}
                aria-label={`Remove ${label} ${itemIndex + 1}`}
              >
                ×
              </button>
            </span>
          </li>
        ))}
      </ul>
      <button
        type="button"
        className={`${iconButtonClass} mt-2`}
        onClick={() => commit(addEntryListItem(resume, section, index, field))}
      >
        {addLabel}
      </button>
    </div>
  );
}

type Commit = (next: Resume | null, mergeKey?: string) => void;

function ExperienceFields({ resume, index, commit }: { resume: Resume; index: number; commit: Commit }) {
  const entry = resume.experience?.[index];
  if (!entry) return null;
  const field = (key: string) => (value: string) =>
    commit(updateEntryField(resume, "experience", index, key, value), `experience.${index}.${key}`);
  return (
    <>
      <ListField id={`experience-${index}-title`} label="Job title" value={entry.title ?? ""} onChange={field("title")} />
      <ListField id={`experience-${index}-company`} label="Company" value={entry.company ?? ""} onChange={field("company")} />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <ListField id={`experience-${index}-location`} label="Location" value={entry.location ?? ""} onChange={field("location")} />
        <ListField id={`experience-${index}-start`} label="Start date" value={entry.start_date ?? ""} onChange={field("start_date")} />
        <ListField id={`experience-${index}-end`} label="End date" value={entry.end_date ?? ""} onChange={field("end_date")} />
      </div>
      <ListField id={`experience-${index}-description`} label="Description" value={entry.description ?? ""} onChange={field("description")} multiline />
      <TextItemList
        section="experience"
        index={index}
        field="achievements"
        label="Achievements"
        items={entry.achievements ?? []}
        commit={commit}
        resume={resume}
        addLabel="Add achievement"
      />
    </>
  );
}

function EducationFields({ resume, index, commit }: { resume: Resume; index: number; commit: Commit }) {
  const entry = resume.education?.[index];
  if (!entry) return null;
  const field = (key: string) => (value: string) =>
    commit(updateEntryField(resume, "education", index, key, value), `education.${index}.${key}`);
  return (
    <>
      <ListField id={`education-${index}-institution`} label="Institution" value={entry.institution ?? ""} onChange={field("institution")} />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <ListField id={`education-${index}-degree`} label="Degree" value={entry.degree ?? ""} onChange={field("degree")} />
        <ListField id={`education-${index}-field`} label="Field of study" value={entry.field ?? ""} onChange={field("field")} />
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <ListField id={`education-${index}-location`} label="Location" value={entry.location ?? ""} onChange={field("location")} />
        <ListField id={`education-${index}-start`} label="Start date" value={entry.start_date ?? ""} onChange={field("start_date")} />
        <ListField id={`education-${index}-end`} label="End date" value={entry.end_date ?? ""} onChange={field("end_date")} />
      </div>
      <TextItemList
        section="education"
        index={index}
        field="details"
        label="Details"
        items={entry.details ?? []}
        commit={commit}
        resume={resume}
        addLabel="Add detail"
      />
    </>
  );
}

function ProjectFields({ resume, index, commit }: { resume: Resume; index: number; commit: Commit }) {
  const entry = resume.projects?.[index];
  if (!entry) return null;
  const field = (key: string) => (value: string) =>
    commit(updateEntryField(resume, "projects", index, key, value), `projects.${index}.${key}`);
  return (
    <>
      <ListField id={`projects-${index}-name`} label="Project name" value={entry.name ?? ""} onChange={field("name")} />
      <ListField id={`projects-${index}-url`} label="Project URL" value={entry.url ?? ""} onChange={field("url")} />
      <ListField id={`projects-${index}-description`} label="Description" value={entry.description ?? ""} onChange={field("description")} multiline />
      <TextItemList
        section="projects"
        index={index}
        field="technologies"
        label="Technologies"
        items={entry.technologies ?? []}
        commit={commit}
        resume={resume}
        addLabel="Add technology"
      />
    </>
  );
}

function CertificationFields({ resume, index, commit }: { resume: Resume; index: number; commit: Commit }) {
  const entry = resume.certifications?.[index];
  if (!entry) return null;
  const field = (key: string) => (value: string) =>
    commit(updateEntryField(resume, "certifications", index, key, value), `certifications.${index}.${key}`);
  return (
    <>
      <ListField id={`certifications-${index}-name`} label="Certification name" value={entry.name ?? ""} onChange={field("name")} />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <ListField id={`certifications-${index}-issuer`} label="Issuer" value={entry.issuer ?? ""} onChange={field("issuer")} />
        <ListField id={`certifications-${index}-date`} label="Date" value={entry.date ?? ""} onChange={field("date")} />
      </div>
      <ListField id={`certifications-${index}-url`} label="Credential URL" value={entry.url ?? ""} onChange={field("url")} />
    </>
  );
}

function CustomSectionFields({ resume, index, commit }: { resume: Resume; index: number; commit: Commit }) {
  const entry = resume.custom_sections?.[index];
  if (!entry) return null;
  return (
    <>
      <ListField
        id={`custom-${index}-heading`}
        label="Heading"
        value={entry.heading ?? ""}
        onChange={(value) =>
          commit(updateEntryField(resume, "custom_sections", index, "heading", value), `custom_sections.${index}.heading`)
        }
      />
      <TextItemList
        section="custom_sections"
        index={index}
        field="content"
        label="Lines"
        items={entry.content ?? []}
        commit={commit}
        resume={resume}
        addLabel="Add line"
      />
    </>
  );
}

function SkillsGroup({
  group,
  items,
  onAdd,
  onUpdate,
  onRemove,
  onMove,
}: {
  group: SkillGroupKey;
  items: string[];
  onAdd: (value: string) => void;
  onUpdate: (index: number, value: string) => void;
  onRemove: (index: number) => void;
  onMove: (index: number, direction: -1 | 1) => void;
}) {
  const [draft, setDraft] = useState("");
  const label = SKILL_GROUP_LABELS[group];
  const submit = () => {
    if (!draft.trim()) return;
    onAdd(draft);
    setDraft("");
  };
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-3">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</h4>
      {items.length === 0 ? (
        <p className="mt-1 text-sm text-slate-400">No {label.toLowerCase()} yet.</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {items.map((item, index) => (
            <li key={index} className="flex items-center gap-2">
              <div className="flex-1">
                <ListField
                  id={`skill-${group}-${index}`}
                  label={`${label} ${index + 1}`}
                  value={item}
                  onChange={(value) => onUpdate(index, value)}
                />
              </div>
              <span className="flex gap-1 pt-5">
                <button type="button" className={iconButtonClass} onClick={() => onMove(index, -1)} disabled={index === 0} aria-label={`Move ${label} ${index + 1} up`}>
                  ↑
                </button>
                <button type="button" className={iconButtonClass} onClick={() => onMove(index, 1)} disabled={index === items.length - 1} aria-label={`Move ${label} ${index + 1} down`}>
                  ↓
                </button>
                <button type="button" className={iconButtonClass} onClick={() => onRemove(index)} aria-label={`Remove ${label} ${index + 1}`}>
                  ×
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-2 flex items-end gap-2">
        <div className="flex-1">
          <ListField
            id={`skill-${group}-new`}
            label={`Add ${label.toLowerCase()}`}
            value={draft}
            placeholder="Type a skill and press Add"
            onChange={setDraft}
          />
        </div>
        <button type="button" className={iconButtonClass} onClick={submit} disabled={!draft.trim()}>
          Add
        </button>
      </div>
    </div>
  );
}
