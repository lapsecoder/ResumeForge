/**
 * Frontend TypeScript types mirroring the backend's transient `Resume` schema
 * (app/parsing/schemas.py) as returned by POST /api/v1/resumes/parse.
 *
 * Field names are snake_case because the backend serializes with
 * `model_dump()` (field names, not aliases).
 */

export type ConfidenceLevel = "high" | "medium" | "low";

export interface ContactInfo {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
  location?: string | null;
  linkedin?: string | null;
  github?: string | null;
  website?: string | null;
}

export interface WorkExperience {
  company: string;
  title?: string;
  location?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  description?: string;
  achievements?: string[];
  skills_mentioned?: string[];
}

export interface Education {
  institution: string;
  degree?: string | null;
  field?: string | null;
  location?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  details?: string[];
}

export interface SkillSet {
  technical?: string[];
  soft?: string[];
  tools?: string[];
  languages?: string[];
  all?: string[];
}

export interface Project {
  name: string;
  description?: string;
  technologies?: string[];
  url?: string | null;
}

export interface Certification {
  name: string;
  issuer?: string;
  date?: string | null;
  url?: string | null;
}

export interface CustomSection {
  heading: string;
  content?: string[];
}

export interface SectionConfidence {
  section: string;
  level: ConfidenceLevel;
}

export interface ResumeMetadata {
  word_count?: number;
  file_type?: string;
  overall_confidence: ConfidenceLevel;
  section_confidence?: SectionConfidence[];
}

export interface Resume {
  contact?: ContactInfo;
  summary?: string | null;
  experience?: WorkExperience[];
  education?: Education[];
  skills?: SkillSet;
  projects?: Project[];
  certifications?: Certification[];
  custom_sections?: CustomSection[];
  metadata: ResumeMetadata;
}