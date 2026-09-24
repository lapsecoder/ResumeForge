# ResumeForge — Technical Architecture

> Status: Foundation + ingestion, deterministic resume parsing, frontend
> upload→parse UI, deterministic job-description parsing, a deterministic
> resume↔job matching baseline, a local semantic (embedding) matching layer,
> a hybrid (deterministic + semantic) matching engine, and a JD-free ATS
> Readiness analysis are implemented.
> This document captures the
> target architecture; implementation proceeds phase-by-phase in the roadmap.

## Overview

ResumeForge is a client-server platform with two decoupled services:

- **Frontend** — Next.js (App Router), TypeScript, Tailwind CSS.
- **Backend** — Python FastAPI (async), Pydantic v2, SQLAlchemy 2.0.

They communicate over versioned REST (`/api/v1`) plus SSE for job progress.
All long-running work (parsing, embedding, matching, LLM) runs on background
workers (Celery/ARQ + Redis), never blocking the API.

## Accounts and privacy (decisions)

- **No accounts required in the current version.** ResumeForge is
  **account-free by default** — there is no login, registration, password,
  email verification, JWT, or refresh-token flow. Anyone can use the product
  without signing up.
- **Local-first / privacy-first.** User data is stored and processed on the
  user's own machine or the local self-hosted backend wherever practical.
  Data minimization and privacy are treated as design constraints.
- **Optional accounts may be added later** only if a concrete need for cloud
  synchronization or persistent server-side user data emerges. No
  authentication foundation is built speculatively.
- **PostgreSQL + pgvector are retained** as the persistence foundation for
  future core application data and for semantic (embedding) search. The exact
  persistence model for resumes, jobs, analyses, and related entities is not
  yet finalized and is intentionally not being created prematurely.

## Key technical decisions (locked)

See the numbered list below — these are the decisions that shape the whole
system and MUST be preserved by future work.

### 1. Two decoupled services + gateway
Next.js and FastAPI are independently deployable, fronted by an Nginx gateway
for TLS, rate limiting, and static serving.

### 2. Async-first job pipeline
Uploads and analysis enqueue background jobs with a status table; the frontend
polls or subscribes via SSE. The API layer stays fast and non-blocking.

### 3. PostgreSQL as source of truth
Relational data, JSONB for flexible payloads, full-text, and pgvector for
embeddings. No redundant databases.

### 4. UUID PKs + soft deletes + timestamps
All entities use opaque UUID primary keys, `created_at`/`updated_at`, and soft
delete (`deleted_at`) where relevant.

### 5. Privacy and data isolation
Account-free and local-first by default. Where data is stored server-side
(future phases), scope and isolate records by the originating owner; row-level
security (RLS) is available as a defense-in-depth backup. No authentication is
required in the current version.

### 6. No authentication in the current version
No login/register, passwords, JWT, or refresh tokens. If accounts are needed
later, a clean auth layer can be added without disturbing the data foundation.

### 7. Deterministic-first, LLM-where-needed
Parsing/segmentation uses libraries + rules + embeddings; LLM (Ollama) only for
generation/summarization.

### 8. Provider-agnostic LLM abstraction
Swappable providers, versioned prompts, structured-output validation, retries,
token budgets, and rule-based fallbacks.

### 9. Local embeddings
sentence-transformers → pgvector. Resume PII never leaves our infrastructure
for similarity.

### 10. Hybrid semantic matching
Cosine similarity of resume↔JD vectors combined with weighted skill/keyword
overlap, producing explainable section-level breakdowns.

> **Implemented now (Phase 5A + 5B + 5C):** the weighted skill/keyword side is
> the **deterministic baseline** (`POST /api/v1/matching/score`, no embeddings),
> and the **semantic side** is a separate local, transient sentence-embedding
> layer (`POST /api/v1/matching/semantic`, sentence-transformers, no
> persistence). A **hybrid engine** (`POST /api/v1/matching/hybrid`) blends the
> two into a single explainable, transient score (70% deterministic / 30%
> semantic when both are meaningful) without rewriting either signal. See the
> "Hybrid Matching (Phase 5C)" section below.

### 11. Symmetric parsing pipelines
Resume and JD share a parser core: upload → validate → store → background
parse → structured schema.

### 12. Resume version history
Each re-parse creates a new `resume_version`, enabling rollback and diffing.

### 13. S3-compatible storage with presigned URLs
Private buckets with opaque per-resource keys, server-side validation, virus
scanning, short-TTL presigned download links only. Bucket/object ownership is
tracked by the owning entity once the persistence model is defined (may be a
client/device/local owner; never a hardcoded authenticated user in the current
account-free version).

### 14. Server-side PDF export
Templates render to PDF on workers into `exports/`, delivered via presigned
URLs — off the API thread, consistent output.

### 15. Unified error contract
`{ error: { code, message, details, request_id } }`, global exception handler,
idempotency keys, job status/retry tracking.

### 16. Three-tier testing
pytest (backend unit/integration), Vitest + Testing Library (frontend),
Playwright (E2E), gated in CI.

### 17. Fully containerized + CI/CD
Docker for all services, GitHub Actions gates, Alembic migrations as a job.

### 18. Observability
Prometheus/Grafana metrics, structured JSON logging, error tracing with
request_id correlation.

### 19. Phased delivery
Foundations → ingestion → analysis/matching → AI → builder → applications →
hardening → production. Accounts/authentication are deferred and only added if
a concrete need arises.

## Ingestion (implemented)

Resume ingestion is **stateless and privacy-first** — user materials are
processed transiently and never permanently stored. See
[`docs/data-model.md`](data-model.md) for the full data lifecycle.

### Lifecycle

```
upload → validate → temporary processing → extract text
→ normalise → return result → delete temporary data
```

- **Supported formats:** PDF (text-based, multi-page, page count returned),
  DOCX (paragraphs + basic tables), and plain UTF-8 text.
- **File-size limit:** 10 MB. Empty and content-less files are rejected.
- **Temporary files:** PDF/DOCX parsing uses a unique UUID-based temp file
  (never the user's filename), removed in a `finally` block on success,
  failure, or unexpected exception. `.txt` is processed entirely in memory.
- **No permanent storage:** nothing is written to disk, PostgreSQL, or object
  storage. No resume contents, extracted text, filenames, or PII are logged.
- **OCR not yet supported:** scanned/image-only PDFs return an `ocr_required`
  status rather than a silently empty result.

### Endpoint

- `POST /api/v1/resumes/extract` — multipart file upload → normalised text +
  metadata. Errors use a structured `{ error: { code, message, request_id } }`
  payload and never expose tracebacks or filesystem paths.

### Key technical decision update

Item 11 ("Symmetric parsing pipelines") originally described
`upload → validate → store → background parse → structured schema`. Given the
**stateless** constraint, the current pipeline is:
`upload → validate → extract → normalise` with **no store step** and **no
permanent application storage**. Background parsing/structured schema and any
persistence remain future phases.

## Deterministic Parsing (Phase 3B)

Structured resume extraction runs **synchronously** on top of the ingestion
pipeline as a deterministic, local, rule-based pass — no LLM, no embeddings, no
background worker, and no persistence of the parsed model.

```
upload → validate → extract → normalise → parse (rules + regex)
→ transient Resume model → response → cleanup
```

### Endpoint

- `POST /api/v1/resumes/parse` — same multipart input as `/extract`, returns a
  transient structured `Resume` (contact, experience, education, skills,
  projects, certifications, custom sections, metadata) plus per-section and
  overall heuristic `confidence`. Errors use the same structured
  `{ error: { code, message, request_id } }` payload.
- The parsed model exists only for the lifetime of the request; nothing is
  written to disk, PostgreSQL, or object storage.

### Determinism and privacy

- Output depends only on the normalised text — deterministic and reproducible,
  favoring **precision over aggressive inference**. Missing values stay
  `Optional`/empty; dates are preserved as strings.
- Plain-text resumes parse entirely in memory. PDF/DOCX still use the ingest
  temporary-file lifecycle (UUID name, `finally` cleanup).
- No resume content, contact details, filenames, or PII are logged at any
  stage. User material is not sent anywhere external.

### Module layout

```
app/parsing/
├── schemas.py    # transient Resume Pydantic models (aligned with data-model.md)
├── heuristics.py # regex helpers: contact info, URLs, dates, bullets
├── sections.py   # heading-alias detection and section splitting
└── parser.py     # parse_resume() orchestration + section parsers
```

## Frontend Upload → Parse → Results (Phase 3C)

The Next.js frontend connects the deterministic pipeline to the browser:

```
Browser (React state only)
  → POST /api/v1/resumes/parse (multipart)
  → FastAPI: validate → temporarily extract → normalise → parse (rules + regex)
  → structured Resume JSON
  → frontend renders present sections + heuristic confidence
→ page/session ends: everything is gone
```

- **`NEXT_PUBLIC_API_BASE_URL`** configures the backend URL (see
  `frontend/.env.example`). No base URL is hardcoded into components.
- **Typed client** in `frontend/src/lib/api.ts` (`parseResume(file)`) with a
  small `lib/resume.ts` type mirror of the backend schema (snake_case field
  names to match `model_dump()` output).
- **CORS:** `app/main.py` adds a CORSMiddleware allow-list for local
  development origins (`http://localhost:3000`, `http://127.0.0.1:3000`),
  configurable via `CORS_ORIGINS`.
- **Client-side validation** (extension/size/empty) is UX only; the backend
  remains the security boundary.
- **Privacy:** parsed results live only in React state; the frontend uses no
  localStorage/sessionStorage/IndexedDB and never caches or logs resume
  content. The backend transient lifecycle is unchanged.
- **Tests:** Vitest + Testing Library (added because no frontend test framework
  existed) — synthetic fixtures only, with assertions that resume content is
  never logged.

## Job Description Ingestion + Parsing (Phase 4)

Job-description ingestion runs **synchronously** with the exact same transient
lifecycle as the resume pipeline. Upload → validate → extract → normalise→ reuses
the Phase 3A ingestion/extraction service (`extract_resume_text`); no extraction
code is duplicated.

```
upload → validate → extract → normalise → parse JD (rules + regex)
→ transient JobDescription model → response → cleanup
```

### Endpoint

- `POST /api/v1/jobs/parse` — multipart PDF/DOCX/TXT, returns a transient
  structured `JobDescription` (title, company, location, employment type,
  remote type, summary, responsibilities, required/preferred skills,
  qualifications, experience/education requirements, certifications,
  nice-to-have, benefits, salary statements, custom sections, metadata with
  per-section and overall heuristic confidence). Errors use the same
  `{ error: { code, message, request_id } }` contract as resumes.

### Parsing model

- **Deterministic and conservative** — "precision over aggressive inference".
  Company, employment, and location are captured only from explicit labels;
  they are never guessed. Headings unknown to the alias set become custom
  sections, but only *after* the first recognised heading is seen, so the
  preamble (title/company/location lines) is never misread as a heading.
- **Salary:** original wording is preserved verbatim in `text`; only an
  explicit currency symbol/ISO code, an explicit period
  (`hourly`/`monthly`/`annual`), and a normalised `range_text` are attached.
  No currency conversion and no estimation ever happen. Salary *sections* and
  *labelled statements* both feed this field.
- Requirements items are classified into education (degree), experience
  ("5+ years", "Senior level"), certifications, short skill lists, and
  qualifications; preferred items route to `preferred_skills` /
  `certifications` / `nice_to_have`.
- `overall_confidence` and `section_confidence` are heuristics (HIGH/MEDIUM/LOW),
  **not** calibrated probabilities.

### Privacy and persistence

- Nothing is persisted — no PostgreSQL, no object storage, no files. The model
  exists only for the lifetime of the request (same guarantee as resumes).
- `app/job_parsing` never imports the DB layer, never creates files, never uses
  filenames as paths, and never logs JD content or salary figures.

### Module layout

```
app/job_parsing/
├── schemas.py    # transient JobDescription/Salary/JobMetadata models
├── heuristics.py # regex helpers: labels, employment/remote, salary, requirements
├── sections.py   # JD heading-alias detection and section splitting
└── parser.py     # parse_job_description() orchestration + section parsers
app/api/v1/jobs.py  # POST /api/v1/jobs/parse
```

## Deterministic Matching (Phase 5A)

Resume↔job matching currently runs as a **deterministic, transient baseline**
— rules only, no LLM, no embeddings, no background worker, no persistence.

```
POST /api/v1/matching/score
  { resume: <Resume>, job: <JobDescription> }   # already-parsed structured models
  → normalise → skill/experience/education/qualification signals
  → weighted overall score (weights redistributed over evaluated criteria)
  → explainable strengths/gaps/matched+unmet requirements
  → response → everything discarded
```

### Endpoint

- `POST /api/v1/matching/score` — JSON body with a previously parsed `Resume`
  and `JobDescription` (both `metadata`-bearing transient models). Returns the
  transient `MatchResult`. No files are uploaded here; matching is a pure
  function over structured data.
- Validation errors are normalised by a global `RequestValidationError` handler
  into the standard `{ error: { code: "validation_error", message, request_id } }`
  contract. Error messages are built from field locations only and **never echo
  the offending input values** (PII-safe).

### Scoring model

| Component | Weight | Source |
| --- | --- | --- |
| Required skills | 50 | coverage of `job.required_skills` |
| Preferred skills | 15 | coverage of `job.preferred_skills` |
| Experience | 20 | candidate years vs required years |
| Education | 10 | degree-level requirement vs candidate level |
| Qualifications | 5 | certified/unmet/unknown qualification buckets |

The overall score is a weighted mean over the **evaluated** components only;
absent criteria redistribute their weight instead of penalising the candidate.
It is documented as a heuristic "Baseline Match Score", explicitly **not** a
probability of hire, an ATS score, or a recruiter decision.

### Consistency with the target architecture

Key decision #10 called for hybrid (semantic + overlap) matching. Phase 5A
implemented the deterministic overlap layer; Phase 5B added the disjoint local
semantic layer; Phase 5C composes them into a transient hybrid score (see
below). The matcher is layered
(`normalizer → per-signal matchers → scorer → service`, and separately
`semantic_matching/service.py`, with `hybrid_matching/` composing the two) so
the two signals were blended without rewriting either. pgvector storage of
embeddings and semantic _search_ remain **deferred**.

### Privacy and persistence

- Pure in-memory computation: no PostgreSQL, no object storage, no files, no
  external calls. Request and result exist only for the lifetime of the call.
- `app/matching` never imports the DB layer and never touches the filesystem.
- Inputs are the structured models only — no raw resume/JD text enters the
  matching path, and no resume/JD content or PII is ever logged.
- Existing `/resumes/extract`, `/resumes/parse`, and `/jobs/parse` contracts are
  unchanged.

### Module layout

```
app/matching/
├── __init__.py           # exports match_resume_to_job
├── schemas.py            # transient MatchResult + per-signal + metadata models
├── normalizer.py         # conservative skill/alias/stopword normalisation
├── skill_matcher.py      # required/preferred skill coverage
├── experience_matcher.py # structured-date experience estimation
├── education_matcher.py  # degree-level classification (doctorate…diploma)
├── qualification_matcher.py  # matched/unmet/unknown qualification buckets
├── scorer.py             # weighted overall + deterministic explanations
└── service.py            # match_resume_to_job() orchestration
app/api/v1/matching.py    # POST /api/v1/matching/score
```

## Local Semantic Matching (Phase 5B)

Semantic similarity between a resume and a job description is computed with a
**small local sentence-embedding model** — no cloud inference, no API key, no
persistence, and it never touches the deterministic baseline.

```
POST /api/v1/matching/semantic
  { resume: <Resume>, job: <JobDescription> }
  → build PII-free "semantic units" (summary/skills/experience/projects/
     education/certifications ↔ summary/responsibilities/skills/qualifications/
     experience+education requirements)
  → embed both sides locally (sentence-transformers, lazy-load, cached model)
  → per-category cosine similarity (mean over job units of their best resume
     unit) + item-level comparisons bucketed high / moderate / low
  → transient SemanticMatchResult → everything discarded
```

### Endpoint

- `POST /api/v1/matching/semantic` — same input shape as `/score` (parsed
  `Resume` + `JobDescription`), returns the transient `SemanticMatchResult`:
  `overall_similarity`/`overall_cosine`, six per-category similarities
  (summary, skill, experience, responsibility, project, qualification),
  item-level buckets (`matched_semantic_items` / `related_items` /
  `low_similarity_items`), a plain-language note, and non-sensitive model
  metadata.
- Categories whose side has no comparable units are reported `None` and are
  simply excluded from the overall average (no penalty for absent sections).
- Errors: 503 `semantic_model_unavailable` when the local model cannot load,
  500 `semantic_inference_failed` on inference/vector errors, otherwise the
  standard 422 validation contract. Errors never echo resume/JD content.

### Model, device and lifecycle

- Default model `sentence-transformers/all-MiniLM-L6-v2` (384-dim, Apache-2.0,
  ~90 MB fp32, no API key). Overridable via `SEMANTIC_MODEL_NAME` /
  `SEMANTIC_MODEL_DIMENSION` (see `backend/.env.example` values in
  `backend/app/semantic_matching/config.py`).
- Weights are fetched once into the machine's Hugging Face cache; the repo
  contains no model files.
- The library is imported **lazily** on first embedding call; the model is
  loaded **once per process** and kept in RAM/VRAM for reuse (device never
  reloads per request). Device selection: CUDA if available, else CPU.
  Thresholds (`SEMANTIC_HIGH_SIMILARITY_THRESHOLD` default 0.65,
  `SEMANTIC_MODERATE_SIMILARITY_THRESHOLD` default 0.40) drive the buckets.

### Semantics of the numbers

- Normalized similarity = `(cosine + 1) / 2`, kept distinct from the raw
  cosine; values are **not probabilities**.
- Semantic similarity is evidence of **relatedness**, not proof of skill
  possession or requirement satisfaction — the deterministic baseline remains
  the possession check. Callers receive this caveat in the result `note`.

### Model, similarity and text-building internals

- `app/semantic_matching/similarity.py` implements cosine in pure Python:
  deterministic, numerically stable, bounded to [-1, 1], zero vectors → 0.0,
  never NaN, no division by zero (and no numpy in the base suite).
- `app/semantic_matching/text_builder.py` builds independent, order-stable
  units and **excludes PII** (name, email, phone, location, personal/linkedin
  URLs). Spoken languages from the dedicated `languages` bucket are handled by
  the deterministic matcher, not by semantic units; the flattened `all` skill
  bucket is included.

### Privacy and persistence

- **Nothing is persisted**: no resume/JD text, no embedding vectors, no
  semantic results — not to disk, PostgreSQL/pgvector, object storage, cache,
  or anything else. Results and vectors exist only for the request lifetime.
- The only disk artifact is the **public model-weight cache**, which contains
  no user data.
- No external service is contacted during inference. Only non-sensitive
  technical metadata is logged (model name + device on load); never request
  bodies, unit texts, PII, or full embeddings.
- `app/semantic_matching` never imports the DB layer, never touches the
  filesystem, and never emits network calls.

### Module layout

```
app/semantic_matching/
├── __init__.py        # exports compute_semantic_match + SemanticMatchResult
├── config.py          # SemanticSettings (SEMANTIC_* env), thresholds, model id
├── model.py           # EmbeddingProvider, LocalSentenceTransformerProvider,
│                      #   lazy load + device pick, SemanticError hierarchy
├── embedder.py        # batch-embeds SemanticUnits via a provider
├── similarity.py      # cosine / normalized similarity / level (pure Python)
├── text_builder.py    # deterministic PII-free unit building (resume + JD)
├── schemas.py         # transient SemanticMatchResult / metadata / item models
└── service.py         # compute_semantic_match() orchestration
app/api/v1/matching.py # POST /api/v1/matching/semantic
```

Tests: pure-math cosine (A), text building / PII (B), lazy provider (C),
controlled-vector service behaviour (D), API (E), security matrices (F), plus a
manual **integration** test (`tests/test_semantic_integration.py`,
`-m integration`) that runs the real model. The base suite is fully offline and
needs neither the model library nor the ~90 MB download.

## Hybrid Matching (Phase 5C)

`POST /api/v1/matching/hybrid` composes the authoritative deterministic
baseline (5A) with the supporting local semantic layer (5B) into a single
**transient, explainable Hybrid Match Score**, without rewriting either
signal. It is deliberate that the engine produces **one score plus rich
evidence**, never a black box.

```
POST /api/v1/matching/hybrid
  { resume: <Resume>, job: <JobDescription> }
  → deterministic MatchResult (5A, unchanged, authoritative)
  → semantic SemanticMatchResult (5B, unchanged, supporting)  ── degraded if model unavailable
  → 0.70·deterministic + 0.30·semantic  (weighted blend, only when both are meaningful)
  → rule-based semantic insights + deterministic strengths/gaps/matched/missing
  → transient HybridMatchResult → everything discarded
```

### Blend formula and fallbacks

- When **both** signals are meaningful:
  `hybrid = round(0.70 · deterministic_overall + 0.30 · semantic_overall · 100, 1)`.
- `deterministic-only` — semantic model missing/failed/no comparable content:
  `hybrid = deterministic` (the deterministic signal is never a casualty of the
  model). No 503/500: the endpoint returns 200 with the semantic availability
  marked.
- `semantic-only` — deterministic component not evaluable (e.g. an empty JD):
  the top-level score is reported `None`, together with an explicit mode, rather
  than manufacturing a match score out of relatedness alone.
- `no-evidence` — nothing meaningful on either side: overall `None`.
- No NaN, no division by zero, no manufactured zeros: unavailable component
  scores stay `None`.

### Hard-requirement safety

The **deterministic layer remains authoritative for explicit requirements**.
Semantic-relatedness insights never flip a missing required skill into a match:
a missing required skill keeps appearing in `missing_required`/`deterministic`
unmet lists, and any supporting insight says the skill is *"not explicitly
verified … does not establish possession"*. Semantic evidence can only
**supplement** context-level matching (a wording-driven insight with an
explicit caveat), never override an exact, hard requirement. Matching is
alias/normalization-consistent in both layers, so an explicit synonym match is
reported as a deterministic match, not as a semantic guess.

### Semantic insights

`semantic_insights[]` are **rule-generated, deterministic template
statements** — never LLM prose, never a score of their own. They cover:

- item-level relatedness for required/preferred skills that are **not**
  explicitly verified (with the possession caveat above);
- category-level relatedness (summary, experience, responsibility, project,
  qualification) against the high (≥0.65) / moderate (≥0.40) thresholds;
- an overall relatedness statement when it crossed the reporting bar.

Every insight carries its `evidence_level` and normalized similarity. `missing`
semantic evidence produces **no** insight; there is no "absence of evidence
implies absence" reasoning anywhere.

### Evidence quality

`metadata.evidence_quality` describes **how much structured information
supported the result** — it is explicitly *not* a statistical confidence in the
score: `high` (hybrid blend, ≥2 evaluable deterministic components, semantic
present), `medium` (otherwise), or `limited` (no meaningful evidence on either
side).

### What the score is not

`metadata.label = "Hybrid Match Score"` with a fixed note: the score is a
**heuristic relevance score, not a hiring probability, an ATS probability, an
employment prediction, or a recruiter decision**. Deterministic evidence
remains authoritative for explicit requirements; semantic similarity only
supplements context-level matching.

### Privacy and persistence

- Everything is computed in memory for the lifetime of the request. **Nothing
  is persisted or logged**: no resume/JD content, no PII, no embedding vectors,
  no hybrid results. The composite adds no network, storage, cache, or LLM
  dependency on top of 5A/5B.
- Degraded (model-unavailable) paths log only a static message — never content,
  vectors, or model internals.
- The response never contains raw embeddings; `component_scores` exposes only
  scalar similarities on their native scales (deterministic 0–100, semantic
  0–1).

### Module layout

```
app/hybrid_matching/
├── __init__.py   # exports compute_hybrid_match + HybridMatchResult
├── schemas.py    # HybridMatchResult / ComponentScores / SemanticInsight /
│                 #   HybridMetadata / SemanticAvailability (+ literals)
├── scoring.py    # 70/30 blend, fallbacks, evidence-quality rule (pure math)
├── insights.py   # deterministic, rule-generated semantic insight statements
└── service.py    # compute_hybrid_match() orchestration + graceful degradation
app/api/v1/matching.py  # POST /api/v1/matching/hybrid
```

Tests (`tests/test_hybrid_*.py`): scoring math (pure formula/fallbacks),
controlled-vector service behaviour (blend + degradation + hard-requirement
safety + insights), API (contract, validation, graceful degradation, no
embedding exposure), security (no persistence/network/LLM, no vector fields, no
content logging), and a 7-scenario synthetic evaluation. The whole suite runs
offline with fake providers.

## ATS Readiness Analysis (Phase 6A)

`POST /api/v1/resumes/ats-analysis` accepts a **single** parsed `Resume` (no
job description) and returns a transient, deterministic, explainable **ATS
Readiness Score**. It is implemented as a pure Python package
(`app/ats_analysis/`) with a deliberate surface:

- **Scope:** resume quality only — contact completeness, section structure,
  experience depth, bullet style, quantified evidence, skills coverage,
  education projects/certifications, date consistency, and parsing quality.
  It does **not** judge job fit, keywords, roles, or "chance of passing an ATS".
- **Transient & private:** nothing is persisted, transmitted, or logged; no
  LLM, no embeddings, no network, no parallel execution, no DB access, no
  filesystem use — the engine is pure `re`/`dataclass` Python over the existing
  `Resume` schema. Security tests lock this in.
- **Deterministic and explainable:** the overall score is a weighted average of
  applicable category scores; non-applicable weights redistribute. Every
  deduction maps to a finding with `category`, `severity`, `rule_id`,
  `explanation`, `recommendation`, and `impact` — no black boxes.
- **Honest unknown handling:** optional sections (summary, projects, certs) and
  unavailable dates/parsing metadata are treated as "not assessed" info
  findings, never as invalid or as a penalty.

Big-O note: the whole analysis is single-pass over the resume's entries/lines
(no quadratic work except a tiny overlap check over experience date ranges for
chronological consistency), so it stays trivial even for large resumes.

### Module layout

```
app/ats_analysis/
├── __init__.py   # exports analyze_resume + ATSReadinessResult
├── schemas.py    # Severity / Finding / CategoryScore / AnalysisMetadata /
│                 #   ATSReadinessResult
├── rules.py      # CATEGORY_ORDER, WEIGHTS, labels, disclaimer, version
├── heuristics.py # ACTION_VERBS, quantifiers, vagueness, date parsing, lines
├── analyzers.py  # 10 category analysers emitting findings (all ats.* rules)
├── scorer.py     # weighted average (with redistribution) + score labels
└── service.py    # analyze_resume() orchestration (immutable resume)
app/api/v1/resumes.py      # POST /api/v1/resumes/ats-analysis
```

Category weights: contact 10 · structure 15 · experience 20 · bullets 15 ·
quantified 10 · skills 10 · education 5 · projects/certs 5 · dates 5 ·
parsing 5.

### Tests

- `tests/test_ats_analysis.py` — unit tests per category (contact, structure,
  experience, bullets, quantified, skills, education, projects/certs, dates,
  parsing), the overall score and boundary labels, and score-bounds.
- `tests/test_ats_analysis_api.py` — endpoint contract, validation (422,
  PII-safe errors), uniqueness, determinism over the wire.
- `tests/test_ats_analysis_security.py` — proves the engine cannot create
  temp/files, cannot import DB/network/ML/LLM layers, and cannot log resume
  content or PII.
- `tests/test_ats_analysis_evaluation.py` — 5 synthetic scenarios (strong,
  sparse, student, experienced, project-heavy) with expected exact scores plus
  cross-cutting invariants (bounds, redistribution, deduction traceability,
  determinism).

## Database (PostgreSQL + pgvector)

The database foundation is in place: PostgreSQL + pgvector with the `vector`
extension enabled via the initial Alembic migration, the async SQLAlchemy
engine/session layer, and declarative `Base`. No application tables or
embedding columns exist yet. Application tables are added in later phases once
their persistence model is decided.

### pgvector

The `vector` extension is enabled through the initial Alembic migration, but
no embedding columns exist yet. pgvector will be used in the semantic matching
phase to store resume and job-description embeddings (produced locally by
sentence-transformers) and run similarity searches directly in PostgreSQL —
keeping all data, including embeddings, in the free self-hosted database.

## Deployment targets (deferred)

- Frontend: Vercel free (Hobby).
- Backend/DB/storage: genuinely free — self-hosted (Docker) or free managed
  tiers only. Never incur cost.

See the roadmap in `README.md` for sequencing.
