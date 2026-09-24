# ResumeForge Frontend

Next.js (App Router) + TypeScript + Tailwind CSS application. The current page
is a privacy-first **resume analyzer**: upload a PDF/DOCX/TXT resume, parse it
through the FastAPI backend, and view the structured result.

## Environment variables

Create `.env.local` from the example:

```bash
cp .env.example .env.local
```

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `NEXT_PUBLIC_API_BASE_URL` | Base URL of the FastAPI backend | `http://localhost:8000` |

The frontend calls `POST {NEXT_PUBLIC_API_BASE_URL}/api/v1/resumes/parse`.

## Getting Started

```bash
npm install
npm run dev        # http://localhost:3000
```

The backend must be running (see `../backend/README.md`). It allows browser
requests from `http://localhost:3000` / `http://127.0.0.1:3000` via CORS
(configure with `CORS_ORIGINS`).

## End-to-end flow

```
Browser (React state only)
  → POST /api/v1/resumes/parse (multipart file)
  → FastAPI: validate → temporarily extract → normalise
  → deterministic parse to structured Resume (rules + regex, no LLM)
  → structured JSON response
  → frontend renders parsed sections (present fields only)
→ page/session ends: everything is gone
```

## Privacy / storage behaviour

- **Resume content is not persisted anywhere.** The uploaded file, extracted
  text, and parsed model exist only in memory for the duration of the request.
- The frontend keeps the parsed result in React state only — no
  localStorage, sessionStorage, IndexedDB, or caching of resume contents.
  Uploaded file bytes are held in component state for the session and cleared
  on reset/new analysis.
- No resume contents, filenames, or contact details are logged (frontend or
  backend). Errors surface as human-readable messages, never tracebacks or
  filesystem paths.
- Client-side file validation (extension, size, empty) is UX-only; the backend
  re-validates everything and is the security boundary.

## UI

- Drag-and-drop upload zone (also keyboard-accessible: the browse input stays
  focusable) supporting PDF, DOCX, TXT up to 10 MB.
- Selected-file preview with remove/replace.
- Indeterminate progress while parsing (uploading → extracting → structuring).
- Structured results view — only fields the parser actually found are shown,
  including a clearly-labelled **heuristic parsing confidence** (rough quality
  signal, not a calibrated probability).
- "Analyze another resume" resets the transient state.

## Tests

Vitest + Testing Library (jsdom). No test framework existed, so this minimal,
free setup was added.

```bash
npm run lint        # eslint (eslint-config-next)
npm run typecheck   # tsc --noEmit
npm run build       # next build
npm run test        # vitest run
```

Test fixtures use synthetic data only — never real resumes or PII.