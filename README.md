# ResumeForge

A production-quality, AI-powered resume and career platform — **built to run at
₹0 cost**.

ResumeForge helps users upload resumes (PDF/DOCX), have them parsed into
structured data, get ATS/quality analysis, match resumes against job
descriptions, discover skill gaps, and receive AI-powered improvement
suggestions and career recommendations — all using free, open-source, and local
tooling.

## Accounts & privacy

- **Account-free by default.** The current version has no login, registration,
  passwords, or accounts — anyone can use ResumeForge without signing up.
- **Local-first / privacy-first.** Data is processed and stored on the user's
  own machine or the local self-hosted backend wherever practical, with minimal
  data retained.
- **Optional accounts may come later**, only if a concrete need for cloud
  sync or persistent server-side user data arises. No authentication is built
  speculatively.
- **PostgreSQL + pgvector** are retained as the persistence foundation for
  future core application data and semantic (embedding) search.

## Main planned features

1. Resume upload (PDF and DOCX)
2. Resume parsing and structured data extraction
3. Multiple resumes + version history
4. Resume quality and ATS analysis
5. Job description upload/paste and analysis
6. Semantic resume-to-job matching (deterministic baseline implemented; local
   semantic embedding layer implemented; hybrid blend of the two implemented)
7. Skill gap analysis
8. AI-powered resume improvement
9. AI career recommendations and learning roadmaps
10. Resume builder/editor with professional templates
11. PDF resume export
12. Job application tracking
13. Career analytics/dashboard
14. Secure, local-first storage of user data

> **Status:** The foundation, database foundation, resume ingestion,
> deterministic resume parsing, frontend upload→parse UI, deterministic
> job-description parsing, a deterministic resume↔job matching baseline, a
> local (sentence-embedding) semantic matching layer, **and a hybrid
> (deterministic + semantic) matching engine** are in place. AI generation,
> application database models, accounts, and the remaining features are NOT
> yet implemented — see the roadmap below.

## Technology stack

| Layer | Technology |
| ----- | ---------- |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS |
| Backend | Python, FastAPI, Pydantic v2 |
| Database | PostgreSQL + pgvector |
| AI / NLP | open-source Python NLP, sentence-transformers, Ollama (local LLM) |
| Infra | Docker, docker-compose |
| Tests | pytest, Vitest, Playwright |

All components are genuinely free: no paid SaaS, paid APIs, or paid AI models.

## High-level architecture

ResumeForge is built as **two decoupled services**:

```
frontend (Next.js)  <─ REST /api/v1 + SSE ─>  backend (FastAPI)
                                                    │
                                            PostgreSQL + pgvector
                                            Redis (queue/cache)
                                            S3-compatible storage (MinIO)
                                            background workers
```

- The **frontend** is a Next.js + TypeScript + Tailwind application.
- The **backend** is an async FastAPI service.
- Long-running work (parsing, embeddings, matching, LLM) runs on background
  workers, never blocking the API.

See [`docs/architecture.md`](docs/architecture.md) for the detailed technical
architecture.

## Development setup

Prerequisites:

- Node.js 18+ and npm
- Python 3.11+
- Docker + Docker Compose (for PostgreSQL, Redis, MinIO)

Clone and start the services:

```bash
# 1. Backend
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload        # → http://localhost:8000

# 2. Frontend
cd frontend
npm install
cp .env.example .env.local           # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev                          # → http://localhost:3000
```

> The frontend calls the backend at `NEXT_PUBLIC_API_BASE_URL` (default
> `http://localhost:8000`). The backend allows browser requests from
> `http://localhost:3000` / `http://127.0.0.1:3000` via CORS
> (`CORS_ORIGINS`, comma-separated).

Optional (infrastructure via Docker):

```bash
docker compose up -d db redis minio
```

## Project structure

```
ResumeForge/
├── AGENTS.md          # Rules for AI agents / engineers
├── README.md
├── docs/              # Architecture and decision documentation
├── tests/             # Cross-cutting / shared test assets (fixtures)
├── backend/           # Python FastAPI application
│   ├── app/           # application package
│   └── pyproject.toml
└── frontend/          # Next.js application
```

## Roadmap (development phases)

1. **Foundation** (this stage): monorepo, config, empty runnable skeleton.
2. Database foundation (PostgreSQL + pgvector, async SQLAlchemy, Alembic).
3. Resume ingestion, deterministic parsing, and the frontend upload → parse
   → results flow (implemented end-to-end, transient/privacy-first).
4. Job-description ingestion + structured parsing (deterministic, transient,
   `POST /api/v1/jobs/parse`).
5. Deterministic resume↔job matching baseline (transient,
   `POST /api/v1/matching/score`).
6. **Semantic matching — local embeddings (Phase 5B):** implemented, transient,
   fully local sentence-transformers (`POST /api/v1/matching/semantic`),
   disjoint from the baseline.
7. **Hybrid matching (Phase 5C):** implemented, transient, explainable blend of
   the deterministic baseline (70%) + local semantic layer (30%)
   (`POST /api/v1/matching/hybrid`); deterministic evidence stays authoritative
   for explicit requirements.
8. ATS & quality analysis.
9. AI generation features.
10. Resume builder, templates, PDF export.
11. Application tracking & analytics.
12. Hardening, security, production deployment.

> Accounts/authentication are deferred and optional; they will be added only
> if a concrete need for cloud sync or server-side user data appears.

## License

Proprietary / to be decided. See maintainers before reuse.
