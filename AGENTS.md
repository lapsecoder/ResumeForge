# AGENTS.md — ResumeForge

Guidance for AI agents (and human engineers) working on this repository.
Read this file first. Follow these rules unless a specific task explicitly
overrides them.

## What this project is

ResumeForge is a production-quality, AI-powered resume and career platform.
Users can upload resumes (PDF/DOCX), get them parsed into structured data,
receive ATS/quality analysis, match resumes against job descriptions,
identify skill gaps, and get AI-powered improvement suggestions and career
recommendations.

## ₹0 cost constraint (HARD REQUIREMENT)

The entire project MUST operate at zero monetary cost.

- NO paid APIs, paid SaaS, paid databases, paid storage, or paid AI services.
- NO paid AI model APIs (e.g. OpenAI, Anthropic, Google) — even with free
  tiers, treat these as non-permitted unless the cost is literally ₹0 forever.
- Prefer open-source, local, and genuinely free solutions.
- Approved tooling: PostgreSQL + pgvector, Redis, S3-compatible/MinIO object
  storage, sentence-transformers (local embeddings), Ollama (local LLM),
  open-source Python NLP libraries (pdfplumber, PyMuPDF, python-docx, spaCy).
- Any new dependency MUST be free (license + runtime cost). Flag anything that
  could incur cost in a PR description.
- Never add a dependency or service secretly. Call out cost implications.

## Repository structure

```
frontend/   Next.js + TypeScript + Tailwind CSS application
backend/    Python FastAPI application (app/ package, pyproject.toml)
tests/      Cross-cutting / shared test assets
docs/       Architecture and decision documentation
```

## Architecture principles

- Two decoupled services: Next.js frontend and FastAPI backend.
- Communication over versioned REST (`/api/v1`) plus SSE for job progress.
- Async-first: long-running work (parsing, embeddings, matching, LLM) runs on
  background workers, never blocking the API.
- PostgreSQL + pgvector is the persistence foundation (relational data, JSONB,
  embeddings) — retained for future core application data and semantic search.
- Account-free by default: no login, registration, passwords, JWT, or refresh
  tokens in the current version. Optional accounts may be considered later
  only if a concrete need for cloud sync or persistent server-side user data
  appears.
- Local-first / privacy-first: process and store user data on the user's own
  machine or the local self-hosted backend wherever practical; minimize data.
- Deterministic-first, LLM-where-needed. Parsing/segmentation uses libraries +
  rules + embeddings; LLM (Ollama) is used only for generation/summarization.
- Security and privacy are non-negotiable (see Security).

## Coding standards

### General
- Minimal, clean, well-named code. Do not over-engineer.
- No unnecessary comments. Explain WHY, not WHAT.
- Follow existing conventions in the file you touch.
- Never commit secrets, keys, or real user data.
- Keep the foundation minimal; add features deliberately per the roadmap.

### Backend (Python / FastAPI)
- Python 3.11+ (repo currently targets a modern 3.x). Use type hints everywhere.
- FastAPI + Pydantic v2 for validation. Layered modules:
  `api / core / db / models / schemas / services / repositories / parsers /
  matching / llm / storage / workers`.
- Use SQLAlchemy 2.0 (async) and Alembic for migrations.
- Format with `ruff`; type-check with mypy where practical.

### Frontend (Next.js / TypeScript)
- App Router, TypeScript (strict), Tailwind CSS.
- React Query (TanStack Query) for server state.
- Strict typing — shared OpenAPI types drive the API client.
- Run lint and typecheck before finishing a change.

### Tests
- Backend: pytest (unit + integration).
- Frontend: Vitest + Testing Library.
- E2E: Playwright.
- Preserve/grow the sample PDF/DOCX fixtures in `tests/`.

## Security requirements (MANDATORY)

- Validate and sanitize ALL inputs (Pydantic, file type/size checks); never
  trust client input.
- Privacy-first / data minimization: collect and retain the minimum data
  needed; process and store user data locally wherever practical.
- PII minimization: strip contact details before sending data to external LLM
  providers; keep embeddings local.
- File uploads: extension/MIME whitelist, size limits, virus scan, presigned
  URLs only — never expose raw keys.
- Data isolation: scope server-side records by their owning entity; use RLS as
  a backup.
- No secrets in code or config — use environment variables / secret manager.
- Structured logging with request_id correlation; never log PII.
- NOTE: accounts, password hashing, JWT, and refresh tokens are deferred — the
  current version is account-free. Do not add authentication foundation until
  it is explicitly requested.

## ₹0-aligned technical decisions (locked)

- Database: PostgreSQL + pgvector (self-hosted or free managed tier only).
- Object storage: S3-compatible (MinIO locally / free tier only).
- Embeddings: local sentence-transformers (no external API).
- LLM generation: Ollama (local). No paid model API.
- Everything runs in Docker; `docker-compose.yml` orchestrates dev services.

## Workflow for AI agents

1. Read AGENTS.md and the relevant `README.md` / `docs/` first.
2. Inspect the current directory before creating or modifying files.
3. Do not implement features out of scope for the current task.
4. Do not install unnecessary dependencies.
5. Run lint/typecheck/tests where relevant before concluding.
6. Report files created, dependencies added, and assumptions made.

## Deferred (explicitly out of scope until later phases)

- User accounts and authentication (login/register, passwords, JWT, refresh
  tokens) are deferred and optional — only added if a concrete need for cloud
  sync or persistent server-side user data appears. Do not build an auth
  foundation speculatively.
- Resume parsing, AI, semantic matching, and remaining database models are NOT
  yet implemented; the persistence model for resumes/jobs/analyses is not
  finalized. Do not create speculative application tables.
- UI pages and application API endpoints are NOT yet implemented.
- Production deployment is NOT part of current tasks. Deployment targets
  Vercel free (frontend) and free/self-hosted infrastructure (backend/DB).
