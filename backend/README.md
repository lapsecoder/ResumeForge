# ResumeForge Backend

Python FastAPI service for the ResumeForge platform.

## Status

Database foundation stage plus resume ingestion, deterministic parsing, a
deterministic matching baseline, and JD-free ATS Readiness analysis. Exposes:

- `GET /api/v1/health` → `{"status": "ok"}` (no database required)
- `POST /api/v1/resumes/extract` → validated, normalised resume text (transient)
- `POST /api/v1/resumes/parse` → transient structured `Resume` model (transient)
- `POST /api/v1/jobs/parse` → transient structured `JobDescription` model (transient)
- `POST /api/v1/matching/score` → transient deterministic Baseline Match Score (transient)
- `POST /api/v1/matching/semantic` → transient local sentence-embedding Semantic Match (transient)
- `POST /api/v1/matching/hybrid` → transient explainable Hybrid Match Score (deterministic + semantic, transient)
- `POST /api/v1/resumes/ats-analysis` → transient ATS Readiness / resume-quality analysis (JD-free)

**Account-free by default**: no authentication, users table, or application
features (AI, accounts) yet. Optional accounts may be added later only if a
concrete need appears.

## Resume Ingestion

`POST /api/v1/resumes/extract` accepts a multipart file upload and returns
extracted, normalised text. **All processing is stateless and transient** —
no resume file, extracted text, or derived data is ever persisted.

### Supported formats

| Format | Extension | Notes |
| ------ | --------- | ----- |
| PDF    | `.pdf`    | Text-based PDFs; multi-page preserved; page count returned |
| DOCX   | `.docx`   | Paragraphs and basic tables preserved |
| Text   | `.txt`    | UTF-8 (falls back to Latin-1 with a warning) |

### File-size limit

- Maximum upload size: **10 MB**.
- Empty files and content-less files are rejected.

### Stateless processing

The full lifecycle is:

```
upload → validate → temporary processing → extract text
→ normalise → return result → delete temporary data
```

- Uploaded bytes are read into memory.
- PDF/DOCX parsing uses a unique temporary file (UUID-based name, never the
  user's filename) that is **deleted in a `finally` block** on success,
  failure, or any unexpected exception.
- Validated `.txt` uploads are processed entirely in memory (no temp file).
- Nothing is written to disk, PostgreSQL, or object storage.
- No resume contents, extracted text, or personal information is logged.

### Cleanup guarantees

Temporary files are always removed, including on parser failure, validation
failure after temp creation, and unexpected exceptions. OCR is **not yet
supported** — scanned/image-only PDFs return an `ocr_required` status instead
of silently producing an empty resume.

### API response

A successful extraction returns the extracted text plus metadata:

```json
{
  "extracted_text": "...",
  "file_type": "pdf",
  "page_count": 2,
  "character_count": 1200,
  "word_count": 200,
  "status": "success",
  "warnings": []
}
```

Errors return a structured `{ "error": { code, message, request_id } }`
payload without exposing tracebacks or filesystem paths.

## Deterministic Resume Parsing (Phase 3B)

`POST /api/v1/resumes/parse` accepts the same file formats as `/extract`, runs
the ingestion pipeline, then converts the normalised text into a structured
transient `Resume` Pydantic model.

### Phase 3B pipeline

```
upload → validation → temporary extraction → normalization
→ deterministic parsing → transient structured Resume
→ response → cleanup
```

- Parsing is **deterministic and local**: rules + regex only, no LLM, no
  embeddings, no external calls.
- **Structured resume data is NOT persisted.** The uploaded file, extracted
  text, and parsed model exist only in memory for the lifetime of the request
  and are discarded afterwards (temp files through the ingestion lifecycle).
- No resume content, contact details, or filenames are logged.
- Unknown headings are retained as `custom_sections` rather than discarded.

### Parser capabilities

- **Contact** — name (conservative, from the header), email, phone,
  LinkedIn, GitHub, and portfolio/website URL.
- **Summary**, **Experience** (title, company, date ranges incl.
  `Present`/`Current`, bullet achievements, description), **Education**
  (degree, institution, dates, details), **Skills** (comma-, pipe-, bullet-,
  and line-separated), **Projects** (name, description, technologies, URL),
  **Certifications** (name, date, URL), and **Custom sections**.
- **Missing information is left Optional/empty**, never invented. Dates are
  preserved as strings; "Present" is the end-date marker for current roles.
- **Date formats**: `MM/YYYY`, `Month YYYY`, `YYYY`, and `Present`/`current`.

### Confidence semantics

Each section and the overall parse carry a heuristic `confidence` level —
`high`, `medium`, or `low` — described in `app/parsing/parser.py`. These are
**not** calibrated probabilities; they signal how clearly the deterministic
rules were able to structure a section.

- `high` → a section clearly contained structured, parseable entries.
- `medium` → the section was present but only partially structured.
- `low` → absent, empty, or unstructured.

### Known limitations

- Deterministic heuristics deliberately favor **precision over aggressive
  inference**, so some valid resumes yield partially structured output.
- Experience entries without date ranges may merge with a neighbouring entry.
- Locations and `skills_mentioned` per role are **not** inferred.
- Skill set is flattened: everything from a skills section is placed in
  `skills.all`/`technical`; no soft-skill/tool/language taxonomy is applied.
- Name extraction is best-effort; an unremarkable first line (e.g. a short
  job title) may be mistaken for a name.
- Curriculum sections using unusual layouts (multi-column, images) parse only
  as well as the extracted text allows.

### Parsing module layout

```
app/parsing/
├── schemas.py    # transient Pydantic models (Resume, sub-models, metadata)
├── heuristics.py # regex helpers: contact, URLs, dates, bullets
├── sections.py   # heading-alias detection and section splitting
└── parser.py     # parse_resume() orchestration + section parsers
```

## Job-Description Parsing (Phase 4)

`POST /api/v1/jobs/parse` accepts the same file formats as `/resumes/extract`
(PDF/DOCX/TXT), reuses the ingestion pipeline for validation/extraction (no
duplicated extraction code), then parses the normalised text into a structured,
**transient** `JobDescription` Pydantic model.

```
upload → validation → temporary extraction → normalization
→ deterministic JD parsing → transient JobDescription
→ response → cleanup
```

- Parsing is **deterministic and local**: rules + regex only, no LLM, no
  embeddings, no external calls.
- **Nothing is persisted**, exactly like resumes: no PostgreSQL, no object
  storage, no files. The uploaded file, extracted text, and parsed model exist
  only for the lifetime of the request.
- No JD content or salary figures are ever logged.
- `app/job_parsing` never imports the database layer, never creates files, and
  never uses user filenames as paths.

### Parser capabilities

- **Role metadata** — title (explicit `Role:`/`Position:`/… label, else the
  leading strong heading); company, employment type, remote type, and location
  **only from explicit labels** (never guessed); remote-like values are not
  treated as locations.
- **Content** — summary (section or derived `"<title> at <company>"`),
  responsibilities, required/preferred skills (comma-, pipe-, semicolon-, and
  bullet-separated), qualifications, experience requirements (e.g.
  `5+ years of Python`, `Senior level`), education requirements (degree terms),
  certifications, nice-to-have, benefits, and custom sections (unknown
  headings after the first known heading).
- **Salary** — original wording preserved verbatim; an explicit currency,
  period (`hourly`/`monthly`/`annual`), and a normalised `range_text` are
  attached when present. **No currency conversion and no estimation.**
- **Missing information is left `None`/empty**, never invented.

### Confidence semantics

Same heuristic model as resume parsing: per-section and overall `high` /
`medium` / `low` levels that signal how clearly the deterministic rules could
structure the document — **not** calibrated probabilities.

### Known limitations

- Deterministic heuristics favor **precision over aggressive inference**.
- Company/employment/location are only recognised from explicit labels;
  values embedded in prose are not inferred.
- Salary figures embedded inside prose sentences are not extracted (only
  labelled statements and salary-section lines).
- Custom headings are only recognised after the first known section heading;
  pre-heading custom blocks fold into the preamble.

### Parsing module layout

```
app/job_parsing/
├── schemas.py    # transient Pydantic models (JobDescription, Salary, metadata)
├── heuristics.py # regex helpers: labels, employment/remote, salary, requirements
├── sections.py   # JD heading-alias detection and section splitting
└── parser.py     # parse_job_description() orchestration + section parsers
app/api/v1/jobs.py   # POST /api/v1/jobs/parse
```

## Deterministic Matching (Phase 5A)

`POST /api/v1/matching/score` accepts a fully parsed, transient `Resume` and
`JobDescription` as JSON and returns a deterministic, explainable
**Baseline Match Score** — computed entirely in memory, no persistence, no
external calls.

> This is a **deterministic baseline**. Semantic similarity is provided by the
> separate local semantic layer (`POST /api/v1/matching/semantic`, Phase 5B),
> and the two signals are blended by the hybrid engine
> (`POST /api/v1/matching/hybrid`, Phase 5C) without rewriting either one.

```
POST /api/v1/matching/score
  { resume: <Resume>, job: <JobDescription> }
  → deterministic signal computation → weighted overall score
  → explainable strengths/gaps/requirements → response
```

### How the score is computed

| Component | Weight | Evaluated from |
| --- | --- | --- |
| Required skills | 50 | coverage of `job.required_skills` against the resume's normalised skill set |
| Preferred skills | 15 | coverage of `job.preferred_skills` |
| Experience | 20 | candidate years (structured date ranges only) vs requested years |
| Education | 10 | recognised degree-level requirement vs highest resume degree level |
| Qualifications | 5 | certifications & qualification claims verified against the resume |

- Components the job does **not** define are excluded and the remaining weights
  are redistributed proportionally — a candidate is never penalised for a
  criterion the job never asked for.
- `overall_score` is `None` when the job defines **no** evaluable requirements.
- The result is **not** a hiring probability, an ATS score, or a recruiter
  decision — it is a deterministic triage heuristic with fully traceable
  rationale.

### Deterministic guarantees

- **Normalisation** is conservative: case/whitespace/Unicode-insensitive
  comparison that preserves meaningful punctuation (`C++`, `C#`, `.NET`,
  `Node.js`, `React.js`). Only a small documented alias set is applied
  (`js↔javascript`, `ts↔typescript`, `py↔python`, `postgres↔postgresql`,
  `golang↔go`, `k8s↔kubernetes`). No aggressive equivalences
  (e.g. *machine learning ≠ deep learning*).
- **Experience** uses only structured start/end dates (`MM/YYYY`, `Month YYYY`,
  `YYYY`, `Present`/`Current`). "Present" resolves against a recorded
  `reference_date`. Entries without two dates are excluded and surfaced;
  insufficient data is reported as **unknown, never zero**.
- **Education** classifies degrees into a hierarchy (doctorate > master >
  bachelor > diploma) so `Bachelor's ↔ B.Tech ↔ B.E.` and
  `Master's ↔ M.Tech ↔ MCA` work without guessing semantics.
- **Qualifications** keep three explicit buckets: `matched`, `unmet`
  (explicitly required and absent), and `unknown` (evidence limitation:
  "not found in resume" is **not** treated as "candidate lacks it"). Unknown
  items never reduce the score.
- Every `strength`, `gap`, `matched_requirement`, and `unmet_requirement` is
  derived from the calculation — no invented evidence.

### Privacy and persistence

- Nothing is persisted and nothing leaves the process: no PostgreSQL, no
  object storage, no files.
- `POST /api/v1/matching/score` accepts structured JSON only — previously
  parsed models are matched, never re-extracted.
- Validation errors return the same `{ error: { code, message, request_id } }`
  contract; error messages **never echo the input**, and request/response
  content is never logged.
- `app/matching` never imports the DB layer and never touches the filesystem.

### Matching module layout

```
app/matching/
├── schemas.py                # transient MatchResult + sub-matches + metadata
├── normalizer.py             # conservative skill/phrase/stopword normalisation
├── skill_matcher.py          # required/preferred skill coverage
├── experience_matcher.py     # structured-date experience estimation
├── education_matcher.py      # degree-level classification + comparison
├── qualification_matcher.py  # 3-bucket qualification verification
├── scorer.py                 # weighted overall score + explanations
└── service.py                # match_resume_to_job() orchestration
app/api/v1/matching.py        # POST /api/v1/matching/score
```

### Known limitations

- The matcher only sees *structured* fields, so experience with missing dates,
  generic prose qualifications, and degrees it cannot classify all land in the
  explicit "unknown / insufficient data" buckets rather than being guessed.
- Seniority-level requirements (`Senior level`) are recognised but not
  quantitatively scored.
- Skills outside the documented alias map are matched literally — synonyms
  (e.g. *machine learning* vs *deep learning*) are treated as distinct.
- Semantic relatedness is covered by the separate semantic layer; it does not
  influence this baseline score.

## Semantic Matching (Phase 5B)

`POST /api/v1/matching/semantic` accepts the **same** parsed, transient
`Resume` + `JobDescription` JSON as `/score` and returns a **local
sentence-embedding** semantic match — no cloud inference, no API key, no
persistence.

```
POST /api/v1/matching/semantic
  { resume: <Resume>, job: <JobDescription> }
  → build PII-free text units for both sides
  → embed locally (sentence-transformers, lazy-loaded, cached per process)
  → per-category + item-level normalized cosine similarity
  → transient SemanticMatchResult → nothing retained
```

### Model, device, lifecycle

- Default model **`sentence-transformers/all-MiniLM-L6-v2`** (384-dim,
  Apache-2.0, ~90 MB, no API key), overridable via `SEMANTIC_MODEL_NAME` /
  `SEMANTIC_MODEL_DIMENSION` in `.env`.
- Weights are downloaded once into the local Hugging Face cache; the repo
  contains no model files.
- The library is imported lazily on the first request; the model loads **once
  per process** (`cuda` if available, else `cpu`) and is reused. Thresholds
  (`SEMANTIC_HIGH_SIMILARITY_THRESHOLD` default `0.65`,
  `SEMANTIC_MODERATE_SIMILARITY_THRESHOLD` default `0.40`) drive the result
  buckets (`matched_semantic_items` / `related_items` / `low_similarity_items`).
- The base test suite runs fully offline with a **fake provider**; the real
  model is exercised by `tests/test_semantic_integration.py` (`-m integration`,
  skips cleanly when the library/DLL is unavailable).

### What the numbers mean

- Similarity = normalized `(cosine + 1) / 2`, in `[0, 1]`, reported alongside
  raw `overall_cosine`. Values are **not probabilities** and not possession
  proof — relatedness is not "has the skill". The result `note` says so
  explicitly, and the deterministic baseline remains the possession check.
- Categories whose side has no comparable units are `None` and excluded from
  the overall average (absent sections are never penalised).
- Categories: summary, skill, experience, responsibility, project,
  qualification. Text units strip all PII (contact, personal URLs, spoken
  languages); only non-sensitive model+device metadata is logged.

### Privacy and persistence

- **Nothing is persisted or transmitted**: no text, no embeddings, no results;
  unit texts and vectors exist only for the request lifetime.
- No external service is contacted during inference (embedding runs locally).
- Error contract is unchanged (`{ error: { code, message, request_id } }`);
  503 `semantic_model_unavailable` when the model cannot load. Content is never
  echoed and never logged.

### Semantic module layout

```
app/semantic_matching/
├── __init__.py        # exports compute_semantic_match + SemanticMatchResult
├── config.py          # SemanticSettings (SEMANTIC_* env), thresholds
├── model.py           # EmbeddingProvider + LocalSentenceTransformerProvider
│                      #   lazy load, device pick, SemanticError hierarchy
├── embedder.py        # batch-embeds text units via a provider
├── similarity.py      # pure-Python cosine / normalized similarity / level
├── text_builder.py    # deterministic PII-free unit building (resume + JD)
├── schemas.py         # transient SemanticMatchResult / metadata / item models
└── service.py         # compute_semantic_match() orchestration
app/api/v1/matching.py # POST /api/v1/matching/semantic
```

### Known limitations

- Semantic similarity is a **relatedness signal**, not a requirement-evidence
  check — use `/score` (or `/hybrid`, which keeps deterministic authority).
- Embeddings are computed locally; the **real-model integration test cannot
  run on this machine** (see "Real-model verification" below).
- Spoken languages are handled by the deterministic matcher, not the semantic
  layer.
- First request downloads ~90 MB of public weights if not cached.

### Real-model verification (one-time)

```bash
# Python 3.13 note: numpy has no 3.13 wheel for the pinned-era versions; build
# toolchain needed, or run this on Python ≤ 3.12 where wheels exist.
pip install -e ".[dev]"          # installs sentence-transformers + torch-cpu
uvicorn app.main:app --reload
# then POST a parsed resume+JD to /api/v1/matching/semantic and check
# metadata: { model, device: "cuda"|"cpu" }
pytest -m integration -q         # real-model run, skips if blocked
```

On this development machine the OS-level Windows Application Control policy
blocks numpy's native `numpy.random._sfc64` DLL, so the real model cannot load
here; the integration test skips cleanly. No code path degrades — only the
offline fake-provider tests cover the service/API on this machine.

## Hybrid Matching (Phase 5C)

`POST /api/v1/matching/hybrid` accepts the **same** parsed, transient
`Resume` + `JobDescription` JSON as `/score` and `/semantic`, computes both
signals, and returns a single **transient, explainable Hybrid Match Score** —
deterministic evidence remains authoritative for explicit requirements while
semantic similarity supplements context-level matching.

```
POST /api/v1/matching/hybrid
  { resume: <Resume>, job: <JobDescription> }
  → deterministic MatchResult (5A, unchanged)
  → semantic SemanticMatchResult (5B, unchanged)  ── degrades if model unavailable
  → 0.70·deterministic + 0.30·semantic  (only when both are meaningful)
  → rule-based semantic insights + deterministic strengths/gaps/matches
  → transient HybridMatchResult → nothing retained
```

### Score and fallbacks

- **Both meaningful:** `hybrid = round(0.70·deterministic + 0.30·semantic·100, 1)`.
- **deterministic-only:** the semantic model is missing, fails inference, or has
  no comparable content → `overall_score` equals the deterministic score and the
  metadata reports `mode="deterministic-only"` plus the semantic availability
  reason (200, never a hard 503/500 — the deterministic signal is never a
  casualty of the model).
- **semantic-only / no-evidence:** the overall score is `None` with the mode
  stated; relatedness alone never manufactures a match score.
- No NaN and no manufactured zeros — unavailable components stay `None`.

### Hard-requirement safety and insights

- The **deterministic layer is the only authority** for required-skill
  verification. A missing required skill stays in `missing_required`/
  `deterministic` unmet lists; supporting insights say it is *"not explicitly
  verified … does not establish possession"*. Relatedness never flips a missing
  requirement into a match.
- `semantic_insights[]` are **fixed template statements** (no LLM) describing
  relatedness levels for unverified required/preferred skills and for
  summary/experience/responsibility/project/qualification categories. No
  semantic evidence → no insight (never "absence of evidence = absence").
- `metadata.evidence_quality` (`high`/`medium`/`limited`) describes how much
  structured information supported the result — **not** a statistical
  confidence.
- `metadata.label = "Hybrid Match Score"` with an explicit note: a heuristic
  relevance score, **not** a hiring probability, ATS probability, employment
  prediction, or recruiter decision.

### Privacy and persistence

- **Nothing is persisted or transmitted**: no text, no PII, no embeddings, no
  results. The composite adds no network, storage, cache, or LLM dependency on
  top of 5A/5B; degraded paths log only static messages.
- The response never exposes raw embedding vectors — only scalars on their
  native scales (`component_scores`: deterministic 0–100, semantic 0–1).

### Hybrid module layout

```
app/hybrid_matching/
├── __init__.py   # exports compute_hybrid_match + HybridMatchResult
├── schemas.py    # HybridMatchResult / ComponentScores / SemanticInsight /
│                 #   HybridMetadata / SemanticAvailability (+ literals)
├── scoring.py    # 70/30 blend + fallbacks + evidence-quality rule (pure math)
├── insights.py   # deterministic, rule-generated semantic insight statements
└── service.py    # compute_hybrid_match() orchestration + graceful degradation
app/api/v1/matching.py  # POST /api/v1/matching/hybrid
```

### Known limitations

- The semantic (30%) side is a relatedness signal, not a requirement check;
  for hiring triage always read `missing_required`/`deterministic` first.
- `semantic-only` results deliberately carry an `overall_score: null` — there
  is no hybrid score without deterministic evidence.
- Real-model behaviour for the semantic side is covered by the offline suite
  plus the manual integration test described above.

## ATS Readiness Analysis (Phase 6A)

`POST /api/v1/resumes/ats-analysis` accepts a **single** parsed, transient
`Resume` (same model as `/resumes/parse`) — **no job description** — and
returns a deterministic, explainable **ATS Readiness Score** plus per-category
scores and per-rule findings. Everything is computed in memory and nothing is
persisted, transmitted, logged, or shared with any external service.

```
POST /api/v1/resumes/ats-analysis
  { resume: <Resume> }
  → overall_score 0–100 + score_label (Weak / Needs Improvement / Good / Strong)
  → category_scores[] with applied weight per category
  → findings[] — every deduction traceable to a rule_id
  → metadata (.method "ats-readiness-heuristic", .version, weights, disclaimer)
```

### Categories and weights

The overall score is the **(weighted average of the applicable category
scores)**; when a category is not applicable its weight is redistributed to the
other applicable categories — a resume is never penalised for a section type it
does not need (e.g. no work experience ⇒ Experience/Bullets/Bullet-craft are
skipped, their weight splits across the remaining categories).

| Category | Default weight | Rules | Examples of rule_ids |
|---|---|---|---|
| Contact details | 10 | `ats.contact.*` | `missing_email` (-30), `missing_phone` (-8), `missing_links` (-15) |
| Section structure | 15 | `ats.structure.*` | `missing_summary`, `missing_experience`, `order_unavailable`, `wrong_order` |
| Experience depth | 20 | `ats.experience.*` | `missing`, `no_dates`, `short_entry`, `few_bullets` |
| Achievement bullets | 15 | `ats.bullets.*` | `short`, `vague`, `repetitive_start`, `not_first_person` |
| Quantified outcomes | 10 | `ats.quantified.*` | `none`, `low`, `moderate`, `good` |
| Skills coverage | 10 | `ats.skills.*` | `missing`, `thin`, `excessive_phrases` |
| Education quality | 5 | `ats.education.*` | `institution`, `degree`, `field`, `dates` |
| Projects & certifications | 5 | `ats.projects.*`, `ats.certs.*` | `thin`, `no_description`, `date` |
| Date consistency | 5 | `ats.dates.*` | `end_before_start`, `overlap`, `order`, `unverified` |
| Parsing quality | 5 | `ats.parsing.*` | `low_confidence`, `format_not_assessed`, `sparse_content` |

### What it measures (and what it does NOT)

- **Deterministic and explainable.** Every deduction is a named rule with a
  fixed impact; the score is reproducible for the same resume.
- **JD-free by design.** It does not judge keyword fit, job requirements, or
  role suitability, and it never tells the user a "probability of passing an
  ATS" or of being hired — the disclaimer in `metadata` states this explicitly.
- **Never penalises honesty.** Missing dates, unknown parsing status, and
  absent optional sections (summary, projects, certifications) are treated as
  "unknown / not assessed" info findings, never as invalid or as deductions —
  except where the category itself is core (e.g. work experience dates).
- **No LLM, no embeddings, no files, no DB.** `app/ats_analysis` is pure Python
  (stdlib `re`/`dataclasses` only) and has zero runtime dependencies beyond the
  existing `Resume` schema; security tests assert it cannot touch the DB,
  network, filesystem, or logging of resume content/PII.

### Manual verification (optional)

```bash
cd backend
# Start the API, then hit the endpoint with a resume you parsed earlier:
curl -s http://localhost:8000/api/v1/resumes/ats-analysis \
  -H "Content-Type: application/json" \
  -d '{"resume": {}}'
```

or via the OpenAPI docs at http://localhost:8000/docs (look for
`POST /api/v1/resumes/ats-analysis`, model `ATSRequestBody`).

## Setup (development)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Verify at http://localhost:8000/api/v1/health

### Browser access (CORS)

To let the Next.js frontend call the API from the browser, CORS is enabled for
the local development origins via `CORS_ORIGINS` (comma-separated, default
`http://localhost:3000,http://127.0.0.1:3000`). The frontend point this at the
backend using `NEXT_PUBLIC_API_BASE_URL` (see `frontend/.env.example`).

## Database (PostgreSQL + pgvector)

The backend targets **PostgreSQL + pgvector**. All credentials come from
environment variables — never hardcoded.

- Connection string: `DATABASE_URL` (async driver), e.g.
  `postgresql+asyncpg://user:pass@localhost:5432/resumeforge`
- Start the database locally with the repo's Docker Compose (PostgreSQL +
  pgvector image):

```bash
docker compose up -d db      # starts only the PostgreSQL + pgvector service
```

Copy `.env.example` → `.env` and confirm `DATABASE_URL` matches your instance.

> **Why pgvector?** pgvector enables storing vector embeddings directly in
> PostgreSQL. Semantic resume↔job matching is already implemented **transiently**
> with local sentence-transformers (no database involved); pgvector is reserved
> for a future phase that needs *persistent* semantic search across a stored
> corpus (e.g. saved resumes / saved jobs) — all in one free, self-hosted
> database.

PostgreSQL + pgvector are **retained for future core application persistence
and semantic search**. No application tables exist yet; the precise persistence
model for resumes, jobs, analyses, and related entities is not finalized and is
intentionally not being created prematurely.

### Database module layout

```
app/
├── core/config.py   # pydantic-settings: DATABASE_URL + pool settings from env
├── db/base.py       # declarative Base + TimestampMixin
└── db/session.py    # async engine, session factory, FastAPI session dependency
```

### Migrations (Alembic)

Alembic is fully configured. The database URL is sourced from the same
`DATABASE_URL` environment variable (`app/core/config.py`) — not hardcoded in
`alembic.ini`.

```bash
# Apply all migrations (requires a running PostgreSQL)
alembic upgrade head

# Generate a new migration after model changes
alembic revision --autogenerate -m "description"

# Roll back the last batch
alembic downgrade -1
```

The single initial migration enables the `vector` extension. It is designed to
upgrade and downgrade cleanly. Application tables are added in later phases.

### Health check

`GET /api/v1/health` → `{"status": "ok"}` (does not require a database).

## Tests

```bash
pytest                    # unit + schema/migration + ingestion + analysis tests (offline)
pytest -m integration     # real-model semantic test (skips cleanly if blocked)
ruff check app tests alembic   # lint
mypy app                  # type check
```

Live-database tests (connectivity, pgvector extension presence) are skipped
automatically when PostgreSQL is not reachable.

## Layout (current)

```
app/
├── api/          # FastAPI routers (v1) — includes resumes + jobs endpoints
├── core/         # config, security, dependencies
├── db/           # SQLAlchemy engine, sessions, Alembic migrations
├── ingestion/    # secure file ingestion: validators, extractors, service
│   ├── validators.py     # extension/type/size/empty validation
│   ├── normalizer.py     # deterministic text normalisation
│   ├── extractors/       # pdf.py, docx.py, text.py
│   ├── schemas.py        # transient Pydantic result models
│   └── service.py        # orchestration (validate → extract → normalise)
├── parsing/      # deterministic resume parser (Phase 3B)
│   ├── schemas.py        # transient Resume Pydantic models
│   ├── heuristics.py     # regex helpers (contact, URLs, dates, bullets)
│   ├── sections.py       # heading aliases and section splitting
│   └── parser.py         # parse_resume() orchestration
├── job_parsing/  # deterministic job-description parser (Phase 4)
│   ├── schemas.py        # transient JobDescription Pydantic models
│   ├── heuristics.py     # regex helpers (labels, salary, requirements)
│   ├── sections.py       # JD heading aliases and section splitting
│   └── parser.py         # parse_job_description() orchestration
├── matching/     # deterministic resume↔JD matching baseline (Phase 5A)
│   ├── schemas.py        # transient MatchResult + metadata
│   ├── normalizer.py     # conservative skill/alias normalisation
│   ├── skill_matcher.py  # required/preferred skill coverage
│   ├── experience_matcher.py# structured-date experience estimation
│   ├── education_matcher.py # degree-level classification
│   ├── qualification_matcher.py  # 3-bucket qualification verification
│   ├── scorer.py         # weighted overall score + explanations
│   └── service.py        # match_resume_to_job() orchestration
├── ats_analysis/  # ATS Readiness / resume-quality analysis (Phase 6A)
│   ├── schemas.py        # transient ATSReadinessResult + findings
│   ├── rules.py          # category keys, weights, labels, disclaimer
│   ├── heuristics.py     # action verbs, quantifiers, vague/date helpers
│   ├── analyzers.py      # 10 category analysers (all rule_ids)
│   ├── scorer.py         # weighted average + score labels
│   └── service.py        # analyze_resume() orchestration (pure, transient)
└── semantic_matching/  # local sentence-embedding matching (Phase 5B)
    ├── config.py         # SemanticSettings (SEMANTIC_* env), thresholds
    ├── model.py          # lazy-load LocalSentenceTransformerProvider + errors
    ├── embedder.py       # batch-embeds text units via a provider
    ├── similarity.py     # pure-Python cosine / normalized similarity
    ├── text_builder.py   # deterministic PII-free unit building
    ├── schemas.py        # transient SemanticMatchResult models
    └── service.py        # compute_semantic_match() orchestration
app/hybrid_matching/   # transient hybrid (deterministic + semantic) engine (Phase 5C)
├── schemas.py         # HybridMatchResult / ComponentScores / metadata / insights
├── scoring.py         # 70/30 blend + fallbacks + evidence-quality rule
├── insights.py        # deterministic, rule-generated semantic insight statements
└── service.py         # compute_hybrid_match() orchestration + degradation
```

## Layout (target)

```
app/
├── api/          # FastAPI routers (v1)
├── core/         # config, security, dependencies
├── db/           # SQLAlchemy engine, sessions, Alembic migrations
├── ingestion/    # secure file ingestion (implemented)
├── models/       # ORM models
├── schemas/      # Pydantic schemas
├── services/     # business logic
├── repositories/ # data access
├── parsers/      # pdf/docx/ocr
├── matching/     # deterministic baseline (implemented, Phase 5A)
├── semantic_matching/  # local sentence-embedding matching (implemented, Phase 5B)
├── hybrid_matching/    # transient blend of both signals (implemented, Phase 5C)
├── ats_analysis/       # JD-free ATS Readiness analysis (implemented, Phase 6A)
├── llm/          # providers, prompts
├── storage/      # S3, presigned URLs
└── workers/      # background tasks
```

The remaining modules are added in later phases.
