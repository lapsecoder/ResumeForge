# ResumeForge — Data Model & Architecture Design

> Status: Design document. This defines the data architecture for ResumeForge
> under the hard constraint that user materials are **not permanently stored**.

---

## Table of contents

1. [Architectural principles](#1-architectural-principles)
2. [User-data lifecycle](#2-user-data-lifecycle)
3. [Resume processing lifecycle](#3-resume-processing-lifecycle)
4. [Job-description lifecycle](#4-job-description-lifecycle)
5. [Proposed transient data structures](#5-proposed-transient-data-structures)
6. [Analysis data lifecycle](#6-analysis-data-lifecycle)
7. [Embedding/vector strategy](#7-embeddingvector-strategy)
8. [PostgreSQL role](#8-postgresql-role)
9. [Privacy/security design](#9-privacysecurity-design)
10. [Self-hosting model](#10-self-hosting-model)
11. [Future opt-in persistence](#11-future-opt-in-persistence)
12. [Recommended architecture](#12-recommended-architecture)
13. [Example request lifecycle](#13-example-request-lifecycle)
14. [What ResumeForge explicitly does NOT store](#14-what-resumeforge-explicitly-does-not-store)

---

## 1. Architectural principles

| Principle | Implication |
| --- | --- |
| **Privacy-first** | User materials are processed transiently and never persisted by default. |
| **Stateless backend** | Each request is self-contained. No server-side session state tied to user identity. |
| **No accounts** | No login, no JWT, no user tables. Anyone can use ResumeForge without authentication. |
| **₹0 cost** | No paid storage, APIs, databases, or AI models. Everything is local/self-hosted. |
| **Data minimisation** | Collect only what is needed to fulfil the current request. Delete everything else immediately. |
| **Deterministic-first** | Parsing uses libraries + rules. LLM is used only where generation is required. |
| **Local-first** | All processing happens on the user's machine or self-hosted backend. Data never leaves unless the user explicitly chooses an external integration. |

**Core rule:** If a piece of data is derived from user-uploaded or user-pasted
content, it must be transient by default. Persistence requires explicit future
design and user consent.

---

## 2. User-data lifecycle

Every user-provided material follows the same default lifecycle:

```
User provides material (upload / paste / type)
  → backend receives it in-memory
  → parsing / analysis / ML runs
  → results are returned to the user (HTTP response / SSE stream)
  → temporary data is discarded (goes out of scope / garbage collected)
  → nothing remains on the server
```

**Key properties of this lifecycle:**

- No user material touches persistent storage (disk, database, or object store).
- Temporary files on disk are **not used by default**. If a library requires
  disk I/O (e.g., certain PDF parsers), files are written to a volatile
  temporary directory and deleted immediately after processing — within the
  same request handler or worker task, with guaranteed cleanup in a `finally`
  block.
- In-memory Python objects (bytes, strings, Pydantic models) are the primary
  working representation. They are garbage-collected when the request or task
  completes.
- No user content is stored in logs, error messages, or analytics.

**If processing fails:**

- Any temporary files are deleted in a `finally` block.
- Any in-memory data is released when the Python objects go out of scope.
- The error response is returned to the user without including raw user content.
- No partial data persists on the server.

---

## 3. Resume processing lifecycle

### 3.1 Inputs

| Input | Source | Required |
| --- | --- | --- |
| Resume file (PDF or DOCX) | `POST /api/v1/resumes/analyze` multipart upload | One of file or text required |
| Resume text | `POST /api/v1/resumes/analyze` JSON body field | One of file or text required |

### 3.2 Processing pipeline

```
1. Receive file bytes (or text) in request handler — in-memory.
2. Validate: file extension, MIME type, size (≤ 10 MB), encoding.
3. Parse into raw text:
   - PDF  → pdfplumber / PyMuPDF → extracted text
   - DOCX → python-docx → extracted text
   - Pasted text → used directly
4. Segment raw text into sections (header, experience, education, skills, etc.)
   using rule-based heuristics + NLP (spaCy / custom).
5. Map segments to structured resume schema (Pydantic model — in-memory).
6. Generate analysis: ATS score, keyword coverage, skill extraction,
   strengths, weaknesses, recommendations — all computed transiently.
7. Generate embeddings from structured data using
   sentence-transformers — in-memory, transient (needed only by the semantic
   matching endpoint).
8. (If enabled) Compute semantic similarity and the hybrid
   (deterministic + semantic) match against a job description — all in-memory,
   transient (§5.5, §5.6).
9. Return structured result as JSON in the HTTP response.
10. All temporary data (file bytes, extracted text, structured model,
    analysis, embeddings) is discarded.
```

### 3.3 Where temporary data exists

| Data | Location | Lifetime |
| --- | --- | --- |
| Uploaded file bytes | Request handler memory (Python `bytes`) | Duration of request |
| Extracted raw text | Local variable in parser function | Duration of request |
| Structured resume object | Pydantic model instance | Duration of request |
| Analysis results | Pydantic model instance | Duration of request (returned in response) |
| Embeddings | NumPy array / Python list | Duration of request |
| Temp file on disk (if any parser requires it) | `tempfile.NamedTemporaryFile` in OS temp dir | Deleted in `finally` block |

### 3.4 What is persisted

**Nothing.** By default, no resume file, no extracted text, no structured
data, no analysis, and no embeddings are written to disk, database, or
object storage.

The only artefact is the **HTTP response** returned to the user's browser,
which the user can then choose to save locally.

### 3.5 Cleanup on failure

```python
async def analyze_resume(file: UploadFile | None, text: str | None):
    tmp_path: Path | None = None
    try:
        if file:
            tmp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
            tmp_path.write_bytes(await file.read())
            raw_text = parse_file(tmp_path)
        else:
            raw_text = text

        structured = segment_and_structure(raw_text)
        analysis = compute_analysis(structured)
        return ResumeAnalysisResponse(structured=structured, analysis=analysis)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        # All local variables (raw_text, structured, analysis)
        # go out of scope and are garbage-collected.
```

---

## 4. Job-description lifecycle

### 4.1 Inputs

| Input | Source | Required |
| --- | --- | --- |
| Job description text | `POST /api/v1/jobs/analyze` JSON body | Required |
| JD document (PDF/DOCX) | `POST /api/v1/jobs/analyze` multipart upload | Optional alternative |

### 4.2 Processing pipeline

```
1. Receive text (or file bytes) in request handler.
2. Validate content (non-empty, reasonable length ≤ 50 KB of text).
3. If file: parse to text (same pipeline as resume).
4. Segment JD into sections: role description, requirements, nice-to-haves,
   company info, benefits.
5. Extract: required skills, preferred skills, experience level, keywords.
6. Generate analysis: keyword density, skill coverage assessment.
7. (If enabled) Generate embeddings for semantic matching — in-memory.
8. Return structured JD analysis as JSON.
9. All temporary data discarded.
```

### 4.3 Persistence

**Nothing.** Same as resumes. The JD text, structured analysis, and
embeddings exist only for the duration of the request.

---

## 5. Proposed transient data structures

All data structures below are **in-memory Pydantic models**. They are not
database tables. They are not written to disk. They exist only during request
processing.

### 5.1 Resume structures

```python
from pydantic import BaseModel
from enum import Enum

class Resume(BaseModel):
    contact: ContactInfo
    summary: str | None
    experience: list[WorkExperience]
    education: list[Education]
    skills: SkillSet
    projects: list[Project]
    certifications: list[Certification]
    achievements: list[str]
    custom_sections: list[CustomSection]
    raw_text: str                       # extracted text, transient
    metadata: ResumeMetadata            # word count, page count, parse confidence

class ContactInfo(BaseModel):
    name: str | None
    email: str | None
    phone: str | None
    location: str | None
    linkedin: str | None
    website: str | None
    github: str | None

class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str | None              # free-form or ISO
    end_date: str | None                # "present" if current
    location: str | None
    bullets: list[str]
    skills_mentioned: list[str]         # skills detected in this role

class Education(BaseModel):
    institution: str
    degree: str
    field: str | None
    start_date: str | None
    end_date: str | None
    gpa: str | None
    highlights: list[str]

class SkillSet(BaseModel):
    technical: list[str]                # e.g. Python, React, PostgreSQL
    soft: list[str]                     # e.g. leadership, communication
    tools: list[str]                    # e.g. Git, Docker, Figma
    languages: list[str]                # spoken languages
    all: list[str]                      # flattened for convenience

class Project(BaseModel):
    name: str
    description: str
    technologies: list[str]
    url: str | None

class Certification(BaseModel):
    name: str
    issuer: str
    date: str | None
    url: str | None

class CustomSection(BaseModel):
    title: str
    content: list[str]

class ResumeMetadata(BaseModel):
    word_count: int
    parse_confidence: float             # 0.0–1.0, how confident the parser is
    file_type: str                      # "pdf" | "docx" | "text"
```

### 5.2 Job-description structures

Implemented in `backend/app/job_parsing/schemas.py` (Phase 4). Parsing is
**deterministic and conservative**: only explicit labels and clear patterns are
captured; missing or ambiguous information stays `None`/empty rather than
being invented ("precision over aggressive inference"). Strings are handed back
verbatim — no normalisation of job titles, company names, or location names
(only section-heading and whitespace normalisation happens internally).

```python
class Salary(BaseModel):
    text: str                    # original compensation wording, verbatim
    currency: str | None         # symbol/ISO code only if explicit ("₹", "$", "INR", …)
    period: str | None           # "hourly" | "monthly" | "annual"
    range_text: str | None       # normalised range, e.g. "10L–15L", "100k–130k", "60000–80000"
    # no estimate, no conversion, no currency arithmetic — the range is preserved as text

class JobDescription(BaseModel):
    title: str | None                  # explicit label ("Role:"/…), else leading strong heading
    company: str | None                # ONLY from an explicit label; never guessed
    location: str | None               # explicit label; None when the value is remote-like
    employment_type: str | None        # Full-time | Part-time | Contract | Internship | Temporary | Freelance
    remote_type: str | None            # Fully remote | Remote | Hybrid | On-site | Work from home | Office-based
    summary: str | None                # summary section, else derived "<title> at <company>"
    responsibilities: list[str]
    required_skills: list[str]
    preferred_skills: list[str]
    qualifications: list[str]          # non-skill, non-degree, non-experience requirements
    experience_requirements: list[str] # "3+ years in X" / "Senior level" statements
    education_requirements: list[str]  # degree/"Bachelor's …" statements
    certifications: list[str]
    nice_to_have: list[str]            # preferred items that aren't certs or short skills
    benefits: list[str]
    salary: list[Salary]
    custom_sections: list[CustomSection]  # headings outside the known set (after the first known one)
    metadata: JobMetadata

class JobMetadata(BaseModel):
    word_count: int
    file_type: FileTypeEnum             # pdf | docx | txt
    overall_confidence: ConfidenceLevel # heuristic HIGH/MEDIUM/LOW, NOT a calibrated probability
    section_confidence: list[SectionConfidence]
```

Earlier design ideas in this document (`description_text`, `experience_level`,
`keywords`) were deliberately dropped during implementation — the final model
captures structured, user-visible fields only.

### 5.3 Analysis structures

```python
class ResumeAnalysis(BaseModel):
    ats_score: ATSScore
    keyword_coverage: KeywordCoverage
    skill_analysis: SkillAnalysis
    quality_assessment: QualityAssessment
    recommendations: list[Recommendation]

class ATSScore(BaseModel):
    overall: float                      # 0–100
    section_scores: dict[str, float]    # per-section scores
    issues: list[str]                   # detected ATS-unfriendly patterns

class KeywordCoverage(BaseModel):
    matched: list[str]
    missing: list[str]
    coverage_ratio: float               # 0.0–1.0

class SkillAnalysis(BaseModel):
    demonstrated: list[str]
    missing_for_role: list[str]         # only when matched against a JD
    additional: list[str]               # skills found beyond JD requirements

class QualityAssessment(BaseModel):
    format_score: float                 # 0–100
    content_score: float                # 0–100
    length_assessment: str              # "too short" | "appropriate" | "too long"
    strengths: list[str]
    weaknesses: list[str]

class Recommendation(BaseModel):
    category: str                       # "skills" | "format" | "content" | "keywords"
    priority: str                       # "high" | "medium" | "low"
    message: str
    suggestion: str
```

### 5.4 Matching structures

Implemented in `backend/app/matching/schemas.py` (Phase 5A). Matching is a
**deterministic baseline**: transient, explainable, and derived from the two
structured models above. The earlier aspirational design (`semantic_score`,
`keyword_overlap_score` mixing both signals in one model) is deliberately NOT
built this way: semantic similarity is implemented as a **separate, disjoint
transient result** (see §5.5) and composed later by the **hybrid engine**
(see §5.6) without rewriting either signal.

```python
class SkillMatch(BaseModel):
    score: float | None                 # composite 0-100, or None (no skill requirements)
    matched_required: list[str]
    matched_preferred: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    matched: list[str]                  # required then preferred, deduplicated
    total_required: int
    total_preferred: int
    required_coverage: float | None     # 0.0-1.0, or None when not required
    preferred_coverage: float | None
    notes: list[str]

class ExperienceMatch(BaseModel):
    score: float | None                 # 0-100, None when not required/unverifiable
    candidate_years: float | None       # None = insufficient dates, NOT zero
    required_years: float | None
    required_max_years: float | None
    required_level: str | None          # entry | junior | mid | senior
    roles_total: int
    roles_measured: int
    assessment: str

class EducationMatch(BaseModel):
    score: float | None
    required_level: str | None          # doctorate | master | bachelor | diploma
    candidate_level: str | None
    matched: bool | None
    assessment: str

class QualificationMatch(BaseModel):
    score: float | None
    matched: list[str]                  # deterministically verified
    unmet: list[str]                    # explicit requirement absent from resume
    unknown: list[str]                  # evidence limitation, never penalised
    total: int
    assessment: str

class MatchMetadata(BaseModel):
    method: str                         # "deterministic-baseline"
    label: str                          # "Baseline Match Score"
    version: str
    weights: dict[str, float]           # 50/15/20/10/5 defaults
    applied_weights: dict[str, float]   # after redistribution (absent criteria excluded)
    reference_date: date                # "Present" experience resolves against this
    note: str                           # semantics deferred to a later phase

class MatchResult(BaseModel):
    overall_score: float | None
    skill_match: SkillMatch
    experience_match: ExperienceMatch
    education_match: EducationMatch
    qualification_match: QualificationMatch
    strengths: list[str]
    gaps: list[str]
    matched_requirements: list[str]
    unmet_requirements: list[str]
    metadata: MatchMetadata
```

Scoring rules:

- Weights: required skills **50**, preferred skills **15**, experience **20**,
  education **10**, qualifications **5**.
- `overall_score` = Σ(applied weight × component score) / Σ(applied weight)
  over **evaluated** components only. A criterion the job never defines is
  excluded (weight redistributed), never penalised.
- `overall_score` is `None` when the job defines **no** evaluable requirements.
- The score is a heuristic triage signal, **not** a hiring probability, ATS
  score, or recruiter decision.
- Experience is computed only from structured date ranges (`MM/YYYY`,
  `Month YYYY`, `YYYY`, `Present`/`Current`); insufficient data is reported as
  *unknown*, never as zero.
- Education uses an explicit degree hierarchy
  (doctorate > master > bachelor > diploma) so `Bachelor's ↔ B.Tech ↔ B.E.`
  and `Master's ↔ M.Tech ↔ MCA` are recognised without guessing.
- Qualification matching has three buckets — `matched` / `unmet` / `unknown` —
  and "not found in resume" is **not** treated as "candidate does not have it".
- All explanations (`strengths`, `gaps`, matched/unmet requirements) are
  derived deterministically from the calculation; no invented evidence.

### 5.5 Semantic matching structures

Implemented in `backend/app/semantic_matching/schemas.py` (Phase 5B). The
semantic layer is a **local, transient, sentence-embedding** result, disjoint
from the deterministic baseline. It shares the exact same input shapes as
`/api/v1/matching/score` — a parsed `Resume` + `JobDescription`.

```python
class SemanticMetadata(BaseModel):
    model: str                          # e.g. "sentence-transformers/all-MiniLM-L6-v2"
    device: str                         # "cuda" | "cpu" | "not-loaded"

class SemanticItem(BaseModel):          # a comparable text unit
    name: str                           # deterministic id: "skill:0", "required-skill:0", …
    text: str                           # PII-free unit text (PII never embedded/logged)

class SemanticMatchItem(BaseModel):     # one unit-to-unit comparison
    resume_item: SemanticItem
    job_item: SemanticItem
    similarity: float                   # normalized (cosine+1)/2 in [0, 1]
    level: str                          # "high" | "moderate" | "low"
    category: str                       # summary | skill | experience | responsibility |
                                        # project | qualification | …

class SemanticCategoryScore(BaseModel):
    category: str
    similarity: float | None            # None when either side has no comparable
                                        # units (absent sections are not penalised)

class SemanticMatchResult(BaseModel):
    overall_similarity: float | None    # mean over present categories, 0–1
    overall_cosine: float | None        # raw cosine form of the same number
    categories: list[SemanticCategoryScore]
    matched_semantic_items: list[SemanticMatchItem]  # level "high"
    related_items: list[SemanticMatchItem]           # level "moderate"
    low_similarity_items: list[SemanticMatchItem]    # level "low"
    note: str                           # honest caveat: similarity ≠ possession
    metadata: SemanticMetadata
```

Semantics:

- Similarity is **relatedness**, not a probability and not proof of skill
  possession; the deterministic baseline remains the possession check.
- Category similarity = mean over job units (responsibilities, required
  skills, …) of their best matching resume unit; per-category `None` is
  excluded from the overall average, never penalised.
- Buckets follow configurable thresholds (`high` ≥ 0.65, `moderate` ≥ 0.40 by
  default).
- **Transient by design**: the result, the vectors, and the unit texts exist
  only for the request lifetime. Nothing here is ever persisted or logged.

### 5.6 Hybrid matching structures

Implemented in `backend/app/hybrid_matching/schemas.py` (Phase 5C). The hybrid
engine composes the authoritative deterministic `MatchResult` (§5.4) with the
supporting `SemanticMatchResult` (§5.5) into a single transient, explainable
result. The two source results stay embedded inside it, untouched; the hybrid
layer only adds the score, its metadata, and rule-generated insights.

```python
class SemanticAvailability(BaseModel):
    available: bool                  # True when semantic evidence contributed
    status: str                      # available | no_content | model_unavailable |
                                     # inference_failed | error
    note: str = ""                   # non-sensitive reason when it did not

class ComponentScores(BaseModel):
    deterministic_overall: float | None   # 0-100
    semantic_overall: float | None        # 0-1
    hybrid_overall: float | None          # 0-100
    deterministic_skills: float | None    # 0-100
    semantic_skills: float | None         # 0-1
    deterministic_experience: float | None  # 0-100
    semantic_experience: float | None       # 0-1
    deterministic_education: float | None   # 0-100
    semantic_qualification: float | None    # 0-1
    # None = not evaluable, never a manufactured zero; native scales preserved.

class SemanticInsight(BaseModel):
    category: str                  # skill | summary | experience | responsibility |
                                   # project | qualification | overall
    evidence_level: str            # high | moderate | low
    similarity: float | None       # normalized [0, 1] unless stated
    statement: str                 # fixed rule template, never LLM prose

class HybridMetadata(BaseModel):
    method: str                    # "hybrid-match"
    label: str                     # "Hybrid Match Score"
    version: str                   # "5c-hybrid-1.0"
    mode: str                      # hybrid | deterministic-only | semantic-only |
                                   # no-evidence
    weights: dict[str, float]      # {"deterministic": 0.70, "semantic": 0.30}
    evidence_quality: str          # high | medium | limited (amount of structure,
                                   # NOT statistical confidence)
    semantic_availability: SemanticAvailability
    semantic_metadata: SemanticMatchMetadata | None
    reference_date: date
    note: str                      # explicit "heuristic relevance score, not a
                                   # hiring/ATS/employment probability" caveat

class HybridMatchResult(BaseModel):
    overall_score: float | None    # 0-100, or None (semantic-only / no evidence)
    deterministic: MatchResult     # authoritative, unchanged
    semantic: SemanticMatchResult | None  # None when model unavailable/failed
    component_scores: ComponentScores
    matched_requirements: list[str]   # deterministic explicit matches only
    missing_required: list[str]       # deterministic required skills not verified
    strengths: list[str]              # deterministic strengths
    gaps: list[str]                   # deterministic gaps
    semantic_insights: list[SemanticInsight]
    metadata: HybridMetadata
```

Semantics:

- **Blend (both meaningful):** `hybrid = 0.70·deterministic + 0.30·semantic·100`.
- **deterministic-only:** model missing/failed/no comparable content — hybrid
  simply equals the deterministic result; the failure is surfaced in
  `semantic_availability` (200, never a hard error).
- **semantic-only / no-evidence:** `overall_score = None` with the mode stated —
  relatedness alone never manufactures a match score.
- **Hard-requirement authority:** the deterministic layer remains the only
  source of required-skill verification. Semantic insights for a missing
  required/preferred skill say it is *"not explicitly verified … does not
  establish possession"*; relatedness never flips a missing requirement into a
  match.
- **Insights are fixed templates** (no LLM); they describe relatedness and its
  level, and are absent when semantic evidence is absent (no
  absence-of-evidence reasoning).
- `evidence_quality` is about **how much structure** supported the result, not
  a confidence interval.
- **Transient and PII-free**: nothing is persisted or logged; the response
  exposes only scalar similarities (never embedding vectors).

### 5.7 Embedding structures

```python
class EmbeddingVector(BaseModel):
    vector: list[float]                 # the embedding itself
    model_name: str                     # e.g. "all-MiniLM-L6-v2"
    dimensions: int                     # e.g. 384
    source_text_hash: str               # hash of the text that produced this
```

Embedding vectors are computed, used for similarity, and discarded. They are
not stored in any database or file.

---

## 6. Analysis data lifecycle

### 6.1 Types of analysis

| Analysis type | Inputs | Output | Storage |
| --- | --- | --- | --- |
| ATS scoring | Structured resume | `ATSScore` | Transient (in response) |
| Keyword coverage | Structured resume + optional JD | `KeywordCoverage` | Transient (in response) |
| Skill extraction | Raw text / structured resume | `SkillAnalysis` | Transient (in response) |
| Quality assessment | Structured resume | `QualityAssessment` | Transient (in response) |
| Recommendations | All of the above | `list[Recommendation]` | Transient (in response) |
| Semantic similarity | Resume embedding + JD embedding | `float` (cosine) | Transient (in response) |
| Skill gap analysis | Resume skills + JD skills | `SkillAnalysis` | Transient (in response) |
| Full match | Resume + JD (all of the above) | `MatchResult` / `HybridMatchResult` | Transient (in response) |
| Hybrid match | `MatchResult` + `SemanticMatchResult` | `HybridMatchResult` (§5.6) | Transient (in response) |

### 6.2 What should be cached temporarily

**Nothing by default.** All analysis is computed on-demand per request.

If performance profiling later reveals that certain computations are
repeatedly expensive, a short-lived in-process cache (e.g., `functools.lru_cache`
with TTL or a simple dict with expiry) could be introduced — but this is
an optimisation detail, not a persistence decision.

Redis caching is **not needed** for the stateless architecture. Redis's role
in this stack is limited to job queue management (Celery/ARQ broker), not
data caching.

### 6.3 Persistence

**Nothing is persisted.** Analysis results are returned in the HTTP response.
If the user wants to keep them, they save them on their own device.

---

## 7. Embedding/vector strategy

### 7.1 How embeddings are used

1. A resume is parsed → structured text → sentence-transformers generates
   an embedding vector (e.g., 384-dimensional float array).
2. A job description is parsed → structured text → sentence-transformers
   generates an embedding vector.
3. Cosine similarity is computed between the two vectors (normalised to
   `(cosine + 1) / 2`) → per-category and item-level `semantic_score`.
   Delivered by `POST /api/v1/matching/semantic` (Phase 5B).
4. A **hybrid engine** (`POST /api/v1/matching/hybrid`, Phase 5C) composes the
   deterministic `MatchResult` with this semantic score
   (`0.70·deterministic + 0.30·semantic`) into one transient, explainable
   `HybridMatchResult`, with the deterministic layer authoritative for explicit
   requirements.
5. Both vectors are discarded.

For multi-unit matching (summary/skills/experience/projects ↔
summary/responsibilities/required skills/qualifications), the endpoint embeds
**semantic units** rather than one blob per side, then compares per category —
see `docs/architecture.md → Local Semantic Matching (Phase 5B)`.

### 7.2 Are embeddings user data?

**Yes.** Embeddings are derived from user content and can, in principle, be
used to reconstruct information about the user's resume or job description.
They must be treated as transient user data and not persisted.

### 7.3 Should pgvector store embeddings?

**No — not in the current architecture.**

pgvector is useful when you need to:
- Persist embeddings for later similarity search across a corpus.
- Run nearest-neighbour queries against stored vectors.

In ResumeForge's stateless architecture:
- Embeddings are computed in-memory and used immediately.
- There is no stored corpus to search against.
- Embeddings are discarded after use.

Therefore, **pgvector is not required for the current product.** The `vector`
extension can remain enabled in PostgreSQL for future use, but no embedding
columns or vector tables are created.

### 7.4 When pgvector would be useful

pgvector becomes relevant **only** if a future opt-in feature requires
persistent semantic search:

- Optional accounts with saved resume/job libraries.
- A self-hosted job matching index.
- Persistent skill-gap tracking across multiple sessions.

These are future possibilities, not current requirements.

### 7.5 Embedding model

- **Model:** `all-MiniLM-L6-v2` (from sentence-transformers).
- **Dimensions:** 384.
- **Runtime:** Imported lazily on the first request that needs embeddings; the
  model is then loaded **once per process** (device `cuda` if available, else
  `cpu`) and reused in-memory for subsequent requests. Weights are downloaded
  once into the local Hugging Face cache — never from a paid source.
- **PII note:** Embeddings are computed from extracted text. Contact
  information is stripped and unit texts are curated before embedding to
  minimise information leakage — even though embeddings are transient.

---

## 8. PostgreSQL role

### 8.1 What PostgreSQL should NOT store

| Category | Reason |
| --- | --- |
| User resume files | Transient by design — deleted after processing |
| Extracted resume text | Transient — derived from uploaded files |
| Structured resume data | Transient — computed per request |
| User-pasted job descriptions | Transient — deleted after processing |
| Job description analysis | Transient — computed per request |
| Embeddings from user content | Transient — derived from user data |
| User contact information | Transient — extracted from resumes |
| Analysis results | Transient — returned in HTTP response |

### 8.2 What PostgreSQL SHOULD store (if used at all)

| Category | Purpose | Lifetime |
| --- | --- | --- |
| Application configuration | Feature flags, system settings, LLM model preferences | Persistent |
| Anonymous operational metadata | Processing statistics (counts, latencies, error rates — no PII) | Persistent, aggregated |
| Reference data | Skills taxonomy, industry keywords, ATS rule definitions | Persistent |
| Rate-limiting state | If rate limiting is implemented server-side | Ephemeral (short TTL) |

### 8.3 Honest assessment: does the current architecture need PostgreSQL?

**For the core product as designed here — no.**

If ResumeForge is strictly a stateless processing tool that:
- receives input,
- processes it,
- returns results,
- discards everything,

then PostgreSQL is not required. The backend could run with **only FastAPI +
in-memory processing** and no database at all.

**However, PostgreSQL is retained in the stack for these reasons:**

1. **Future opt-in features** (accounts, saved resumes, application tracking)
   will need relational persistence. Having PostgreSQL ready avoids a
   migration pain later.
2. **Operational metadata** (processing counts, health metrics, system
   config) is genuinely useful for a self-hosted tool.
3. **Reference data** (skills taxonomy, keyword lists) benefits from
   relational storage and versioning.
4. **Self-hosters** may want optional persistence. PostgreSQL being in
   docker-compose already supports that.

### 8.4 PostgreSQL configuration

```yaml
# docker-compose.yml already defines:
db:
  image: pgvector/pgvector:pg16
  # This is fine — pgvector is available if needed later,
  # but no vector columns are created now.
```

The initial migration enables the `vector` extension. This is harmless and
forward-looking. No application tables are created until persistence is
genuinely needed.

### 8.5 When PostgreSQL becomes essential

PostgreSQL becomes essential when any of these future features are implemented:

- Optional user accounts (requires users table, sessions).
- Optional saved resumes (requires resumes table with owner scoping).
- Optional application tracking (requires applications, companies, statuses).
- System configuration that must survive restarts.
- Aggregated analytics that must survive restarts.

---

## 9. Privacy/security design

### 9.1 Temporary file handling

```
Policy: No persistent files. Disk I/O only when a parser library requires it.

- Files are written to OS temp directory (Python tempfile module).
- Files are deleted in a `finally` block, guaranteed even on exceptions.
- File paths are randomised (UUID-based) to prevent path traversal.
- File permissions are restricted (mode 0o600 on Unix).
- File size limit: 10 MB for resumes, 50 KB for pasted text.
- Allowed extensions: .pdf, .docx only. MIME types validated.
```

### 9.2 Deletion guarantees

| Scenario | What happens |
| --- | --- |
| Request succeeds | Response sent → all variables out of scope → GC reclaims memory. Temp files deleted in `finally`. |
| Request fails / exception | `finally` block runs → temp files deleted. Variables out of scope → GC. No partial data on disk. |
| Worker crashes | Temp files remain in OS temp dir until OS cleanup. In-memory data lost. Acceptable for transient data. |
| Server restarts | All in-memory data lost. Temp files remain in OS temp dir until OS cleanup. Acceptable. |

### 9.3 File size and type limits

| Constraint | Value |
| --- | --- |
| Max resume file size | 10 MB |
| Max pasted resume text | 50 KB |
| Max job description text | 50 KB |
| Allowed resume extensions | `.pdf`, `.docx` |
| Allowed JD extensions | `.pdf`, `.docx`, or raw text |
| MIME validation | Yes (PDF: `application/pdf`, DOCX: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`) |

### 9.4 Path traversal protection

- Temp file names are generated using `uuid.uuid4()`, never derived from
  user-provided filenames.
- User-provided filenames are stored in `UploadFile.filename` but never used
  for filesystem operations.
- No user-supplied path segments are joined with server paths.

### 9.5 Isolation between simultaneous requests

- Each request handler operates in its own async context with local variables.
- Temp files use unique UUID-based names — no collisions.
- In-memory objects (Pydantic models, embedding arrays) are request-scoped.
- No shared mutable state between requests.

### 9.6 Logging

| Principle | Implementation |
| --- | --- |
| Never log PII | Contact info, email, phone, name are never included in log messages. |
| Never log raw resume text | Extracted text is not logged. |
| Never log filenames | User-provided filenames are not logged (they could reveal PII). |
| Never log embeddings | Transient data, not logged. |
| Log request metadata | `request_id`, processing duration, file size (not content), success/error status. |
| Structured logging | JSON format with correlation IDs. |

### 9.7 LLM prompt safety

When ResumeForge uses Ollama for generation/summarisation:

- Contact information is stripped from text before sending to LLM.
- Prompts are constructed in-memory and not persisted.
- LLM responses are processed in-memory and returned to the user.
- No prompts or responses are stored in any log, file, or database.
- Ollama runs locally — data never leaves the user's machine.

### 9.8 Embedding safety

- Embeddings are computed in-memory.
- They are never written to disk, database, or transmitted externally.
- They are not logged.
- They are garbage-collected after use.
- Contact information IS stripped before embedding (the semantic layer builds
  PII-free text units — contact fields, personal URLs, and spoken languages
  are excluded from embeddings).

---

## 10. Self-hosting model

### 10.1 Requirements

A self-hosted user needs:

| Component | Required | Purpose |
| --- | --- | --- |
| Python 3.11+ | Yes | Backend runtime |
| Node.js 18+ | Yes | Frontend build/runtime |
| Docker + Docker Compose | Optional (recommended) | PostgreSQL, Redis, Ollama |
| PostgreSQL | Optional | Only if using system config or future persistent features |
| Redis | Optional | Only if using background job queue |
| Ollama | Optional | Only if using AI generation features |
| MinIO / S3 | **Not required** | File storage is not used in the stateless architecture |
| No account | **Confirmed** | No login, no registration, no API key |
| No paid service | **Confirmed** | Everything is free and local |

### 10.2 Minimal self-hosted setup

The absolute minimum for a working ResumeForge instance:

```bash
# Backend (processes resumes in-memory, returns results)
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
uvicorn app.main:app --reload

# Frontend (serves the UI)
cd frontend
npm install
npm run dev
```

No Docker required. No database required. No object storage. The backend
receives uploads, processes them in-memory, returns results.

### 10.3 Full self-hosted setup (with all features)

```bash
docker compose up -d        # PostgreSQL + Redis + Ollama
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev
```

### 10.4 What self-hosted users are NOT required to pay for

- No ResumeForge account
- No cloud database
- No cloud storage
- No API keys
- No paid AI model access
- No domain / TLS (for local dev)

---

## 11. Future opt-in persistence

These features are **not current**. They are possible future additions that
would introduce persistence **only with explicit user consent**.

### 11.1 Optional accounts

```
If added in the future:
- User creates an account (email + password, or passwordless).
- Account enables: saved resumes, application tracking, cloud sync.
- Without an account: full functionality remains — stateless processing.
- Account creation is a gate, not a prerequisite.
```

Database tables required: `users`, `sessions` (or equivalent).

### 11.2 Optional saved resumes

```
If added in the future:
- User explicitly chooses to "save" a resume after analysis.
- Saved resumes are stored in PostgreSQL, scoped to the user's account.
- User can have multiple saved resumes with version history.
- Without saving: resume is processed transiently, as today.
```

Database tables required: `resumes`, `resume_versions`, `resume_sections`.

### 11.3 Optional cloud sync

```
If added in the future:
- User's saved resumes are synced to PostgreSQL server-side.
- Enables multi-device access.
- User must have an account and explicitly opt in.
- Local-first model: local copy is always primary, cloud is a backup.
```

### 11.4 Optional application tracking

```
If added in the future:
- User tracks job applications: company, role, status, dates, notes.
- Stored in PostgreSQL, scoped to user account.
- Optional: link application to a saved resume.
- Requires account.
```

Database tables required: `companies`, `applications`, `application_statuses`.

### 11.5 How these features integrate without redesign

The stateless architecture is designed to be **extended**, not replaced:

1. Account layer added as middleware (auth check on specific routes).
2. Persistence layer added as repository pattern (data access behind interfaces).
3. Stateless endpoints remain stateless — they don't break.
4. New persistent endpoints are added alongside, gated by auth.
5. No existing transient processing logic changes.

---

## 12. Recommended architecture

### 12.1 Alternatives considered

| Option | Description | Pros | Cons |
| --- | --- | --- | --- |
| **A. Permanent server storage** | Store all user files and data on the server permanently | Enables history, multi-device | Violates privacy-first principle. Requires accounts, storage, backup. Expensive. |
| **B. Local browser storage** | Store everything in browser localStorage/IndexedDB | Zero server storage. Privacy. | Data lost on browser clear. No server-side processing possible. Limited capacity. |
| **C. Stateless backend processing** | Process in-memory, return results, discard everything | Simple. Private. No accounts. No storage cost. Works self-hosted. | No history. No multi-device. User must save their own results. |
| **D. Hybrid** | Stateless by default, optional persistence with accounts | Best of both worlds | More complex. Requires auth foundation. |

### 12.2 Recommendation: Option C (Stateless) as the default

**Option C — Stateless backend processing** is the recommended architecture
for the current version of ResumeForge.

**Rationale:**

- Simplest to implement and maintain.
- Strongest privacy guarantee — no user data is ever stored on the server.
- No authentication complexity.
- Zero storage cost.
- Works perfectly for self-hosted single-user deployments.
- Aligned with the ₹0 cost constraint.
- Aligned with the privacy-first principle.

**The frontend is responsible for:**
- Displaying results to the user.
- Optionally allowing the user to download/save results as JSON or PDF.
- If the user wants history, they save files on their own device.

**Option D (Hybrid) is the recommended future evolution** — adding optional
persistence via accounts for users who want it, without breaking the
stateless default.

### 12.3 Why NOT store anything in the default deployment

| Concern | Why storage is avoided |
| ---| ---|
| Privacy | User resumes contain highly sensitive PII. Storing them creates liability. |
| Legal | Storing PII without consent may violate GDPR/privacy laws. |
| Security | Stored PII is a target for breaches. Not storing it eliminates the risk. |
| Cost | Storage costs money (disk, backup, redundancy). Violates ₹0 constraint. |
| Complexity | Storage requires cleanup, backup, retention policies, access control. |
| Self-hosting | Self-hosters shouldn't need to manage database backups for a processing tool. |

---

## 13. Example request lifecycle

### 13.1 Resume upload and analysis (full walk-through)

```
User clicks "Upload Resume" in browser → selects resume.pdf (3.2 MB)

1. FRONTEND sends: POST /api/v1/resumes/analyze
   Content-Type: multipart/form-data
   Body: file=resume.pdf

2. FASTAPI HANDLER receives request:
   - Assigns request_id = "req-abc-123"
   - Logs: {"request_id": "req-abc-123", "event": "resume_upload_received", "size": 3355443}
   - Validates: extension=.pdf, MIME=application/pdf, size=3.2MB ≤ 10MB ✓

3. PARSER (in-memory):
   - Writes file bytes to tempfile: /tmp/resume-a1b2c3d4.pdf  (UUID-based name)
   - Uses pdfplumber to extract text → raw_text (4,200 characters)
   - Deletes temp file in finally block: /tmp/resume-a1b2c3d4.pdf removed

4. SEGMENTER (in-memory):
   - Parses raw_text into sections using spaCy + rules
   - Produces structured Resume Pydantic model:
     - contact: {name: "Jane Doe", email: "jane@example.com", ...}
     - experience: [3 WorkExperience entries]
     - education: [1 Education entry]
     - skills: {technical: ["Python", "React", ...], ...}

5. ANALYSIS ENGINE (in-memory):
   - ATS scoring: overall=78, issues=["Missing summary section", ...]
   - Keyword coverage: matched=15, missing=5, ratio=0.75
   - Skill analysis: demonstrated=[...], weaknesses=["No cloud experience"]
   - Recommendations: [5 Recommendation objects]

6. (If matching endpoint) EMBEDDING + MATCHING (in-memory):
   - sentence-transformers encodes resume text → [0.12, -0.34, ...] (384 dims)
   - sentence-transformers encodes JD text → [0.15, -0.31, ...] (384 dims)
   - cosine_similarity = 0.82 → semantic_score = 82
   - Both vectors go out of scope → garbage collected

7. RESPONSE:
   - HTTP 200 with JSON body containing structured resume + analysis
   - Logged: {"request_id": "req-abc-123", "event": "analysis_complete", "duration_ms": 340}

8. CLEANUP:
   - All local variables (raw_text, Resume model, analysis, temp file)
     go out of scope → Python GC reclaims memory
   - Temp file already deleted in step 3's finally block
   - Nothing persists on the server

9. FRONTEND receives response → displays analysis to user
   - User can click "Download Results" to save JSON locally
   - User can click "Export PDF" → POST /api/v1/exports/resume
     → server generates PDF → returns binary → user downloads → nothing stored
```

### 13.2 No data remains on the server after step 8.

---

## 14. What ResumeForge explicitly does NOT store

This section is intentionally exhaustive. If a data type is listed here,
ResumeForge does **not** persist it — not in a database, not in a file,
not in object storage, not in logs.

| Data type | Stored? | Reason |
| --- | --- | --- |
| Uploaded resume file (PDF/DOCX) | No | Transient — parsed and discarded |
| Extracted resume text | No | Transient — derived from upload |
| Structured resume data | No | Computed per request, returned in response |
| User contact information | No | Extracted from resume, transient |
| User-pasted job description text | No | Transient — processed and discarded |
| Job description analysis | No | Computed per request, returned in response |
| Resume-JD match results | No | Computed per request, returned in response |
| Embedding vectors (resume) | No | Computed in-memory, used, discarded |
| Embedding vectors (JD) | No | Computed in-memory, used, discarded |
| AI-generated suggestions | No | Generated in-memory, returned in response |
| LLM prompts containing user data | No | Not persisted anywhere |
| User filenames | No | Not logged, not stored |
| PII in logs | No | Never logged |
| Session state | No | Stateless — no sessions |
| User preferences | No | Stateless — no accounts |
| Processing history | No | No records of past requests |
| Analytics tied to user identity | No | No identity to tie to |

**The only things that persist across requests are:**

- Application configuration (feature flags, system settings).
- Anonymous aggregate operational metrics (total requests processed, error
  rates — no PII, no user content).
- Reference data (skills taxonomy, keyword lists) if explicitly configured.
- Docker volumes for PostgreSQL and Redis (operational data, not user data).

---

## Summary

ResumeForge is designed as a **stateless, privacy-first processing tool**.
User materials — resumes, job descriptions, extracted text, structured data,
analysis results, embeddings — are all transient. They exist only in memory
during request processing and are discarded immediately after results are
returned.

**PostgreSQL** is retained in the stack for future optional features (accounts,
saved resumes, application tracking) and for system configuration, but is **not
required** for the core product. No application tables are created until
persistence is genuinely needed.

**pgvector** is available but not used in the current architecture, since
embeddings are computed and consumed in-memory without persistence.

**The recommended architecture is stateless backend processing (Option C)**:
the simplest, most private, most self-hostable design that satisfies all
constraints. Future opt-in persistence (Option D: Hybrid) can be layered on
top without redesigning the core.

The user's browser is the only place where results persist — if the user
chooses to save them. ResumeForge's server forgetts everything by default.
