# Phase 7D - Resume Builder, Templates & PDF Export

## 1. Goals

Phases 7A-7C made the parsed resume inspectable and let the Copilot propose
reversible, in-memory edits. Phase 7D makes the resume **editable** and
**exportable**:

- a structured **Resume Builder** (manual CRUD over the canonical `Resume`);
- live **preview** with at least three genuinely different templates;
- **real PDF export** using the browser, with no server round-trip and no paid
  service;
- bounded, in-memory **undo / redo** and **reset to parsed**;
- **validation** before export (`errors` block, `warnings` do not);
- **Copilot** proposals applied against the *current* working resume and visible
  in the builder.

The design rule for the phase is:

> One canonical resume, edited by the user or proposed by the AI, rendered by
> interchangeable templates, and exported locally. Everything is transient: no
> accounts, no storage, no upload.

## 2. Canonical Resume model

The builder does not introduce a second representation. It edits the exact same
structure that the parser produces (`backend/app/parsing/schemas.py`) and that
the frontend already consumes (`frontend/src/lib/resume.ts`, snake_case mirror):

| Field              | Shape |
| ------------------ | ----- |
| `contact`          | `{ name, email, phone, location, linkedin, github, website }` |
| `summary`          | `string \| null` |
| `experience[]`     | `{ company, title, location, start_date, end_date, description, achievements[], skills_mentioned[]? }` |
| `education[]`      | `{ institution, degree, field, location, start_date, end_date, details[] }` |
| `skills`           | `{ technical[], soft[], tools[], languages[], all[] }` |
| `projects[]`       | `{ name, description, technologies[], url }` |
| `certifications[]` | `{ name, issuer, date, url }` |
| `custom_sections[]`| `{ heading, content[] }` |
| `metadata`         | `{ overall_confidence, word_count, file_type?, section_confidence? }` |

Two fields are **derived**, never authored:

- `skills.all` is the de-duplicated (case-insensitive) flattened union of
  `technical → tools → languages → soft`.
- `metadata.word_count` is recomputed from the visible text. The backend
  requires `word_count` when validating Copilot edit targets, so keeping it
  current is what lets a Copilot proposal run against edited content.

`normalizeResume()` (`frontend/src/lib/builder.ts`) is the single place that
recomputes these fields and repairs missing/malformed optional collections. It
never invents user-authored content.

## 3. Builder state architecture

```
ResumeAnalyzer
  ├─ originalResume        (the parsed resume, kept for "Reset changes")
  └─ useHistory<Resume>    (in-memory undo/redo, bounded)
        │  present = workingResume
        ├─ ResultsView            (structured, read-only)
        ├─ ResumeBuilder          (manual edits → commit)
        │     └─ TemplatePreview  (live preview + print-only copy)
        └─ CopilotPanel           (proposals → onResumeChange = commit)
```

- There is **one** working-resume state, owned by `ResumeAnalyzer`. The builder
  and the Copilot panel both write to it through a single `commit` function, so
  they can never drift apart.
- The builder is **controlled**: it receives `resume` and calls
  `onCommit(next, { mergeKey })`. It holds only UI-local state (selected
  template, mobile tab, draft skill input, export status message).
- The builder is mounted only while the "Resume builder" tab is active, so the
  structured view and the builder never render two copies of the resume at once.

## 4. Supported fields

Every editor control maps to exactly one allowlisted field:

- **Contact**: `name, email, phone, location, linkedin, github, website`
  (empty string clears the field to `null`).
- **Summary**: `summary` (multiline, cleared when blank).
- **Experience**: `title, company, location, start_date, end_date, description`
  plus the `achievements[]` list.
- **Education**: `institution, degree, field, location, start_date, end_date`
  plus the `details[]` list.
- **Skills**: items in `technical, soft, tools, languages` (add / edit / remove /
  move; duplicate and blank entries rejected).
- **Projects**: `name, description, url` plus the `technologies[]` list.
- **Certifications**: `name, issuer, date, url`.
- **Custom sections**: `heading` plus the `content[]` list.

## 5. CRUD operations

All mutations live in `frontend/src/lib/builder.ts` as **pure functions** that
return a new `Resume` or `null`, and never mutate their input:

| Operation | Functions |
| --------- | --------- |
| Scalar edit | `updateContactField`, `updateSummary`, `updateEntryField`, `updateEntryListItem`, `updateSkill` |
| Add | `addEntry`, `addEntryListItem`, `addSkill` |
| Remove | `removeEntry`, `removeEntryListItem`, `removeSkill` |
| Reorder | `moveEntry`, `moveEntryListItem`, `moveSkill` |

Guarantees enforced by every function:

- **Allowlists only.** Section keys, scalar fields, list fields, contact fields,
  and skill groups are checked against fixed allowlists. There is no arbitrary
  object-path API, so no field outside the lists can ever be written.
- **Prototype-pollution guard.** `__proto__`, `constructor`, and `prototype`
  are rejected as field names.
- **Index guards.** Indexes must be non-negative integers within range;
  fractional, negative, and out-of-range indexes return `null`.
- **Bounds.** `MAX_FIELD_CHARS = 5000` per field and `MAX_LIST_ITEMS = 200` per
  list; oversized input is rejected (not silently truncated).
- **Sanitization.** C0 control characters (except tab/newline) are stripped.
- The result is always normalized, so it is a structurally valid `Resume`.

The editor renders add/remove/move buttons with accessible names
(`Move experience 2 up`, `Remove experience 1`, …) and is keyboard reachable.

## 6. Undo / redo

`frontend/src/lib/history.ts` implements a bounded, in-memory history:

- `HISTORY_LIMIT = 50`; older steps are dropped from the front of `past`.
- `commit(present, { mergeKey })`: consecutive commits with the *same* mergeKey
  collapse into one undo step, so typing in a field does not flood the stack.
  A different/absent mergeKey starts a new step. Committing clears `future`.
- `undo()` / `redo()` move the present across `past` / `future`.
- `reset(present)` discards all history (used by "Reset changes").
- History is React state only. It is **never** written to `localStorage`,
  `sessionStorage`, `IndexedDB`, cookies, or the network; a page refresh
  discards it.

Copilot-applied edits and manual edits share the same `commit`, so an AI edit
becomes an ordinary, reversible working-state change. "Reset changes" restores
the parsed `originalResume`, which also discards any applied Copilot edits.

## 7. Validation

`frontend/src/lib/builderValidation.ts` exposes
`validateResume(resume): { errors, warnings }`. It treats the working object as
untrusted and checks:

- **Structure**: `metadata`, `contact`, each array section, `skills`, and each
  skill group must have the right container type; entries must be objects.
- **Types**: every string field must actually be a string; list items must be
  strings.
- **Lengths**: fields over `MAX_FIELD_CHARS` are errors; fields over
  `WARN_FIELD_CHARS = 1000` are warnings.
- **Counts**: lists over `MAX_LIST_ITEMS` are errors.
- **Semantics**: email shape, unsafe URL schemes (`javascript:`, `data:`,
  `vbscript:`, `file:`), and malformed URLs.
- **Completeness**: an entirely empty resume is an error; missing
  `company`/`institution`/`name`/`heading` on an entry are warnings.

`errors` disable the **Export PDF** button; `warnings` are shown but do not
block. Optional fields are never required, so a sparse resume still exports.

## 8. Template architecture

Templates are pure presentational components with the same contract:

```ts
type ResumeTemplate = {
  id: TemplateId;              // "classic" | "modern" | "compact"
  name: string;
  description: string;
  component: (props: { resume: Resume }) => JSX.Element;
};
```

- `TEMPLATES` is a fixed registry (`frontend/src/components/templates/index.ts`).
- `getTemplate(id)` resolves an id and **falls back to `classic`** for unknown
  or hostile ids, so a bad template id can never break rendering.
- Switching templates changes only which component renders. It never touches the
  resume: the same canonical object is passed to every template.
- Templates are deterministic. No AI is involved in layout, and there is no
  LLM-generated HTML/CSS.

Shared helpers (`templates/shared.ts`) centralize safe rendering:
`contactItems`, `safeProfileHref` (only `http(s)` or bare-domain → `https`),
`safeEmailHref` (valid email → `mailto`), `dateRange`, `isBlank`. Any value that
does not resolve to a safe href is rendered as **inert text**, never as a link.

## 9. Template descriptions

| Id | Name | Layout |
| -- | ---- | ------ |
| `classic` | Classic | Single column, centered header, uppercase rules. The ATS-friendly default. |
| `modern` | Modern | Left accent bars on indigo headings, right-aligned contact block, tag-style technologies. |
| `compact` | Compact | Dense two-column `label | content` grid (84px label column) that fits more on one page. |

All three render the same sections, filter out blank entries, and omit empty
section headings entirely. No photos, graphics, progress bars, star ratings, or
fabricated percentages appear in any template.

## 10. A4 layout

Print styling lives in `frontend/src/app/globals.css`:

- `@page { size: A4; margin: 14mm }`.
- `.resume-paper` is `210mm` wide with a `297mm` min-height and `14mm 15mm`
  padding, so the on-screen preview matches the printed page.
- `.resume-entry` uses `break-inside: avoid` and `.resume-heading` uses
  `break-after: avoid` to reduce awkward page breaks.
- `print-color-adjust: exact` keeps accent colors (e.g. the Modern border)
  in the PDF.

## 11. PDF export architecture

Export is **browser-native and frontend-only** (`frontend/src/lib/pdf.ts`):

1. The resume is rendered twice: once in the on-screen preview, and once into a
   body-level, print-only root (`.resume-print-root`) created with
   `createPortal`. On screen that root is `display: none`.
2. In print, `body > *` is hidden and `body > .resume-print-root` is shown. The
   browser therefore prints only the resume, avoiding clipping from scroll
   containers and blank trailing pages.
3. `exportResumePdf(resume)` sets `document.title` to the suggested filename,
   calls `window.print()`, and restores the previous title in a `finally` block.
   Browsers use the document title as the default "Save as PDF" filename.

**Filename safety**: `sanitizeFilenameBase` normalizes Unicode (NFKD), strips
accents, replaces anything outside `[A-Za-z0-9 ._-]`, collapses `..`/underscores,
strips leading/trailing non-alphanumerics, caps at 60 characters, and prefixes
Windows reserved device names (`CON`, `NUL`, `LPT1`, …) with `_`. The result is
`<Name>_Resume.pdf`, or `resume.pdf` when there is no name; it can never contain
a path separator or traversal.

**Why no backend generation?** A backend PDF service would need a renderer and
server-side temp files for content that is already fully present in the browser.
The browser's own print pipeline produces a real, text-based, selectable PDF
with zero dependencies, zero cost, and zero data movement. Nothing is uploaded
and the app writes no file to disk; the user chooses whether to save.

## 12. Copilot integration

- `CopilotPanel` receives `resume={workingResume}` and
  `onResumeChange={commitWorkingResume}`. Proposals are always generated against
  the *current* working resume, never the original parsed snapshot.
- Applying a proposal calls `commit`, so the change lands in the same history as
  manual edits: it appears immediately in the builder and preview, and can be
  undone with **Undo** or discarded with **Reset changes**.
- Copilot is **not** auto-invoked after manual edits. The user explicitly starts
  an analysis; 7A-7C behaviour and API contracts are unchanged.
- Manual edits refresh `metadata.word_count` (via `normalizeResume`), which the
  backend needs to validate Copilot edit targets.

## 13. ATS / matching integration

- The builder only edits the resume; it does not score anything. ATS analysis
  and job matching remain the existing endpoints and components
  (`ResultsView`, `MatchFlow`, `backend/app/ats`, `backend/app/matching`).
- `MatchFlow` and `CopilotPanel` consume the same `workingResume`, so downstream
  analysis automatically reflects builder edits without any new contract.
- No duplicate ATS or matching logic was added to the builder.

## 14. Privacy model

- **Account-free and stateless.** No login, no auth, no user records.
- The working resume lives only in React state for the current browser session.
- **No persistence** of resume or job-description content: no database, no
  object storage, no `localStorage`/`sessionStorage`/`IndexedDB`, no cookies.
  A refresh discards everything; re-analysis is required (acceptable by design).
- **No uploads during editing or export.** Export is entirely client-side.
- Resume content, PII, and generated documents are never logged.

## 15. Security model

- **Input validation**: every mutation validates section keys, field names,
  indexes, and value types/lengths against allowlists before writing.
- **Prototype-pollution defence**: `__proto__`, `constructor`, and `prototype`
  are rejected; `Object.prototype` is never touched. `normalizeResume` tolerates
  tampered input without throwing.
- **XSS defence**: React escapes all text nodes. Templates never use
  `dangerouslySetInnerHTML`; hostile strings such as `<script>` render as
  literal text.
- **Link safety**: only `http(s)`/`mailto` hrefs are emitted; `javascript:`,
  `data:`, `vbscript:`, and `file:` values render as inert text.
- **Template-id safety**: unknown ids fall back to the classic template.
- **Export safety**: filenames are sanitized (no separators/traversal/reserved
  names), and export is blocked while validation errors exist.
- **No backend attack surface added**: Phase 7D touches no backend code and adds
  no endpoint or temporary file.

## 16. Tests and verification

New frontend tests (frontend uses Vitest + Testing Library):

| File | Covers |
| ---- | ------ |
| `src/lib/history.test.ts` | commit/undo/redo, redo clearing, no-op, mergeKey coalescing, limit, reset |
| `src/lib/builder.test.ts` | field/summary/entry edits, list items, skills, immutability, normalization, empty resume, Copilot `applyEdit` → working resume |
| `src/lib/builderValidation.test.ts` | valid/sparse/empty, email, unsafe scheme, long/over-long, malformed sections/entries, wrong types, item counts |
| `src/lib/pdf.test.ts` | filename sanitization (traversal, reserved names, accent, length), fallback, title set/restore, print failure |
| `src/lib/builderSecurity.test.ts` | prototype pollution, invalid indexes, oversized values, wrong-section writes, hostile text stored inert, tampered resume, no storage/network |
| `src/components/templates/templates.test.tsx` | registry + fallback, all templates render shared content, layout-specific headings, distinct DOM, empty sections omitted, long content, HTML escaping, unsafe hrefs, bare-domain upgrade |
| `src/components/ResumeBuilder.test.tsx` | toolbar + preview, live edit, undo/redo, add/remove, reorder, reset, export blocking, print + filename, template switch without content change, no persistence/network |
| `src/components/ResumeAnalyzer.test.tsx` (extended) | builder tab mounts for the working resume; builder edits shared with the structured view and can be reset |

Results:

- Frontend: **191 passed / 191** across **17 files** (was 119 / 10; +72 tests,
  +7 files), plus 2 extended `ResumeAnalyzer` cases.
- `tsc --noEmit`: clean.
- ESLint: clean.
- `next build`: succeeds.
- Backend: unchanged by Phase 7D; last full run for reference was
  **1148 passed, 2 skipped**.

## 17. Known limitations

- Export relies on the browser print dialog ("Save as PDF"). The suggested
  filename is applied via `document.title`; some browsers may still prompt the
  user to confirm the name.
- Print fidelity depends on the browser/OS print engine; exact page breaks vary
  slightly between engines.
- Section **order** is fixed per template; users can reorder entries within a
  section but not move whole sections.
- Templates are a fixed set; there is no custom template editor.
- History is bounded (50 steps) and session-only by design.
- Very long resumes may span multiple pages; the app does not warn about page
  count.

## 18. Manual verification

1. Analyze a PDF resume and confirm the results screen appears.
2. Switch to the **Resume builder** tab; confirm the editor (left) and live
   preview (right) render the parsed content.
3. Edit the name, company, a bullet, and a skill; confirm the preview updates.
4. Add and remove an experience entry, then move one up/down.
5. Use **Undo**, **Redo**, and **Reset changes**; confirm the preview follows.
6. Switch between Classic / Modern / Compact; confirm content is unchanged and
   only the layout differs.
7. Make the email invalid; confirm the export button disables and an error is
   shown. Fix it; confirm export re-enables.
8. Click **Export PDF**; confirm the print dialog opens with a filename like
   `Ada_Lovelace_Resume.pdf` and that "Save as PDF" yields a text-based PDF.
9. Apply a Copilot proposal; confirm it appears in the builder and can be undone
   or reset.
10. Refresh the page; confirm everything is discarded (no stored data).
