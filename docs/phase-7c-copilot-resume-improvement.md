# Phase 7C - Copilot Resume Improvement & Controlled Apply

## 1. Goals

Phase 7B turned Copilot suggestions into *grounded* suggestions. Phase 7C
turns the replacement-text subset of those suggestions into **structured edit
proposals** that a user can preview, apply, revert, or dismiss against a
transient, in-memory copy of their resume.

The design rule for the whole phase is:

> The AI proposes, the validator checks, the user decides, the frontend applies
> in memory, and the user can revert. The model never edits anything by itself.

Concretely, Phase 7C delivers:

- a deterministic edit-proposal layer on top of the existing 7B pipeline;
- an explicit editable-field **allowlist** with structured, server-resolved
  targets (the model never supplies a path);
- per-category **fact-preservation checks** (metrics, dates, skills, and all
  remaining distinctive wording);
- **explicit apply / revert / dismiss** behaviour in the UI, with no autosave;
- additive, backward-compatible API changes (no new endpoint).

## 2. Architecture

The end-to-end flow is:

```
Resume + JobDescription
        │
        ▼
Copilot operation (improve_summary | improve_bullet | ...)
        │
        ▼
Provider chain: LocalOllamaProvider → DeterministicFallbackProvider
        │
        ▼
7B factual validation (hallucination guard, evidence sanitation)
        │
        ▼
7C edit proposal (allowlist target + fact-preservation checks)   ← backend stops here
        │
        ▼
USER previews → USER applies → frontend updates in-memory resume
        │
        ▼
USER may revert → frontend restores the previous value
```

- Edit proposals are attached **centrally** in `CopilotService.suggest()`
  (`backend/app/copilot/service.py`) after the provider returns, so both the LLM
  and the deterministic fallback always pass through the same allowlist and the
  same checks.
- The backend never mutates a resume and never stores one. `apply_edit` in
  `backend/app/copilot/editing.py` is a pure function used by tests and mirrored
  on the frontend; the API only ever *proposes*.
- No new endpoint was added. `POST /api/v1/copilot/suggest` gained a nullable
  `edit` object on each suggestion. Phase 7B clients keep working unchanged.

## 3. Supported operations

| Operation           | Edit proposal? | Notes |
| ------------------- | -------------- | ----- |
| `improve_summary`   | Yes (target `summary`) | Replacement text for the professional summary. |
| `improve_bullet`    | Yes (target from `target_ref`) | Replacement text for a selected achievement/description. |
| `identify_priorities` | No | Advisory guidance only. |
| `explain_finding`   | No | Advisory explanation only. |
| `job_alignment`     | No | Advisory alignment guidance only. |

Only `improve_summary` and `improve_bullet` can emit edits. For
`improve_bullet`, an edit is only produced when the client supplied a
`target_ref` that resolves to an allowlisted resume field. Pasted custom text
(`target_text` with no `target_ref`) is improved but yields **no** edit, because
it is not part of the resume.

At most **one** edit proposal is emitted per distinct target path (first wins).
This guarantees the UI can never present two conflicting replacements for the
same field.

## 4. The edit proposal model

Added to `backend/app/copilot/schemas.py`:

- `EditStatus` — `proposed | applied | reverted | dismissed | unverified`.
  The backend only ever emits `proposed` (safe to apply) or `unverified`
  (needs explicit confirmation). `applied` / `reverted` / `dismissed` are
  UI-only transitions on the in-memory copy.
- `EditCheckCategory` — `metric | date | skill | source_term`.
- `CopilotEditTarget` — `path` (readable) plus structured `section`, `index`,
  `field`, `sub_index`. The frontend applies edits from the structured fields,
  never by parsing path strings.
- `EditCheck` — one check: `category`, `passed`, PII-safe `detail`.
- `CopilotEditValidation` — `status`, `requires_user_confirmation`, `checks[]`,
  plus an `all_passed` convenience property.
- `CopilotEditProposal` — `edit_id`, `operation`, `target`, `original_value`,
  `proposed_value`, `reason`, `evidence`, `validation`, `status`.
- `CopilotSuggestion.edit: CopilotEditProposal | None` — the additive field.

`edit_id` is derived from the suggestion id (`edit_<suggestion.id>`, e.g.
`edit_sug_1`). `original_value` is always **resolved from the resume
server-side**; it is never taken from model output.

## 5. Target-path allowlist

The allowlist is hard-coded in `backend/app/copilot/editing.py` and mirrored in
`frontend/src/lib/edit.ts`.

**Editable**

- `summary`
- `experience[i].title`
- `experience[i].company`
- `experience[i].description`
- `experience[i].achievements[j]` (alias `experience[i].bullets[j]`)
- `projects[i].description`
- `projects[i].technologies` (comma-separated list)

**Never editable**

- contact fields (`name`, `email`, `phone`, `linkedin`, `github`, `website`);
- `skills.*`, `education[i].*`, `certifications[i].*`;
- `projects[i].name`, `experience[i].start_date` / `end_date` /
  `skills_mentioned`;
- `custom_sections[i].content[j]`, `metadata.*`;
- arbitrary JSON paths, server configuration, and filesystem paths.

The path grammar is intentionally narrow
(`^(summary|experience|projects)(?:\[(\d+)\])?(?:\.([a-z_]+)(?:\[(\d+)\])?)?$`).
Indices are non-negative integers; malformed indices, unknown fields, unknown
sections, and any extra segments are rejected. A tampered target whose
structured fields disagree with its `path` is rejected by `is_editable_target`.

## 6. Factual validation

`validate_edit_value()` runs four deterministic checks against the resume fact
corpus (and, where relevant, the target's original text):

| Category      | What it catches | Method |
| ------------- | --------------- | ------ |
| `metric`      | New/unsupported numbers (comma-normalised) | `canonical_numbers(proposed) − canonical_numbers(corpus)`, excluding 4-digit years |
| `date`        | New years or month/present words | `_date_tokens(proposed) − _date_tokens(corpus)` |
| `skill`       | JD skills the resume does not support | `unsupported_job_skills()` (same logic as 7B `_check_skills`) |
| `source_term` | New distinctive wording: employers, titles, organisations, projects, schools, certifications, locations | `significant_term_tokens(proposed) − significant_term_tokens(corpus)`, ignoring tokens containing digits |

If any check fails, or the underlying suggestion was already `unverified`, the
proposal status becomes `unverified` with `requires_user_confirmation = true`.
Otherwise the proposal inherits the suggestion's `verified` / `inferred` level
and is marked `proposed`.

The check details never echo resume content; they name the category and, for
skills, the JD label only. The exposed `source_term` wording is generic.

These checks reuse the exact 7B primitives via additive, read-only wrappers in
`validation.py` (`resume_fact_corpus`, `canonical_numbers`,
`significant_term_tokens`, `unsupported_job_skills`), so the edit layer cannot
silently diverge from the suggestion validator.

## 7. Apply / revert behaviour

- **Apply** is explicit and user-initiated. Clicking *Apply to resume* (or
  *Apply anyway* for `unverified` proposals) calls the frontend `applyEdit`,
  which deep-copies the resume, writes the value, and hands the new object to
  the caller. The input object is never mutated.
- **Revert** restores the recorded `original_value` at the same target. The
  original value is captured at apply time.
- **Dismiss** removes the proposal from the list without touching the resume.
- **Preview checks** reveals the four deterministic check results so the user
  can see *why* a proposal is (or is not) safe to apply.
- Edits and their applied/reverted state live only in React state. There is no
  autosave, no `localStorage` / `sessionStorage` / `IndexedDB`, and no server
  round-trip. Refreshing the page discards edits along with the analysis.
- Applying an out-of-range or stale target is refused by `applyEdit` and
  surfaces a non-destructive error instead of corrupting the resume.

## 8. Frontend flow

- `frontend/src/lib/copilot.ts` gained the Phase 7C TypeScript types and the
  optional `edit` field on `CopilotSuggestion`.
- `frontend/src/lib/edit.ts` mirrors the allowlist and provides
  `parseEditPath`, `isEditableTarget`, `applyEdit`, and `editTargetLabel`,
  bounded by `MAX_EDIT_VALUE_CHARS = 4000`.
- `frontend/src/components/CopilotPanel.tsx` renders an edit block for
  suggestions that carry an `edit`: the target label, the current and proposed
  values, the validation badge, the collapsible check list, and the
  Apply / Revert / Dismiss controls. Suggestions **without** an `edit` keep the
  previous in-session Apply/Undo visual toggle.
- `frontend/src/components/ResumeAnalyzer.tsx` now owns a `workingResume`
  state (initialised from the parsed resume). `ResultsView`, `MatchFlow`, and
  `CopilotPanel` all read it, and `CopilotPanel` writes back through
  `onResumeChange`. Reset clears it.

## 9. Ollama integration

- The primary provider remains the local Ollama provider
  (`LocalOllamaProvider`), loopback-only. No cloud or paid model API is used.
- Edit proposals are derived from whatever provider answered, so no
  Ollama-specific code was added in Phase 7C.
- A bounded live smoke test (see §16) confirmed that the local model
  (`qwen2.5-coder:7b`) returned a summary rewrite, that a proposal was attached
  with target `summary`, and that the 7B guard's conservative new-wording rule
  correctly marked it `unverified` pending confirmation.

## 10. Deterministic fallback

- `DeterministicFallbackProvider` continues to answer offline with no model.
- Its safe rewrites (leading first-person stripping, capitalisation, whitespace
  trimming) preserve every token, so they pass all four checks and produce
  `proposed` (verified) edits with no confirmation required.
- The API tests and the security tests all run against a deterministic-only
  service, proving the edit path works with no network and no LLM.

## 11. Privacy model

- Account-free and stateless. Nothing about the resume, the job, the
  suggestions, or the edits is persisted or logged.
- Edit proposals round-trip to the client and live only in the browser's
  in-memory component state.
- Check details are PII-safe and never echo contact details.
- A response-level test asserts that contact PII (name/email/phone) is never
  present in the serialized `/suggest` response, including when an edit block
  is attached.

## 12. Security model

Tested threats (`tests/test_copilot_edit_security.py`, 102 cases):

- **Arbitrary target-path injection** — protected refs and hostile strings
  (`contact.email`, `../../etc/passwd`, `C:\Windows\...`, `summary; rm -rf /`,
  `$(whoami)` / backticks, `server.config`, `env.SECRET_KEY`,
  `resume.__class__`, NUL bytes, newlines) never parse, never build a proposal,
  and never produce an edit through the service.
- **Protected fields** — 17 protected refs never yield an edit.
- **Invalid indexes** — out-of-range targets are refused at build and apply.
- **Malformed / tampered proposals** — structured targets that disagree with
  their path, or that claim a protected `section`, are rejected.
- **Unsupported operations** — `identify_priorities`, `explain_finding`, and
  `job_alignment` never emit edits.
- **Hallucinated facts** — invented metrics, dates, employers, and job skills
  are downgraded to `unverified`, never `proposed`.
- **Prompt injection in the resume/JD** — injected instructions do not create
  targets; any attached edit still has an allowlisted target with no `contact`
  or traversal segment.
- **Oversized input/output** — suggested values are capped at
  `max_edit_value_chars` (4000); `applyEdit` refuses values over the bound.
- **Deterministic fallback** — edits are produced and applied with no LLM.

## 13. Testing

**Backend (new)**

- `tests/test_copilot_edits.py` — **66** tests: allowlist parsing (accepted /
  rejected parametrised), alias canonicalisation, target resolution, pure
  apply/revert semantics, all four fact checks, proposal construction, and
  duplicate-target de-duplication.
- `tests/test_copilot_edit_security.py` — **102** tests: the threat matrix
  above.
- `tests/test_copilot_edit_api.py` — **9** tests: the additive API contract,
  7B backward compatibility, the `7c-copilot-1.0` version, and PII absence.

**Frontend (new)**

- `src/lib/edit.test.ts` — **41** tests: allowlist parsing, tamper rejection,
  apply semantics (summary, bullet, project description, technologies),
  immutability, range/bound refusal, and target labels.
- `src/components/CopilotPanelEdit.test.tsx` — **6** tests: apply updates the
  working resume and offers Revert, revert restores the original, dismiss
  changes nothing, preview reveals the checks, unverified proposals are
  labelled and warned, and an out-of-range target shows an error instead of
  applying.

## 14. Known limitations

- The fact checks are deterministic and conservative. A paraphrase that
  introduces genuinely new *wording* is flagged `unverified` even when it is
  true, because the system cannot prove it from the resume alone. The user
  remains the final authority.
- Edit proposals are text-only. Skills, education, certifications, dates, and
  project names are intentionally not editable.
- `improve_bullet` only produces an edit when the text already exists in the
  resume (via `target_ref`); pasted text is guidance only.
- Applying an edit does not silently re-run downstream analyses beyond what the
  component naturally recomputes; the working copy is the user's current view.
- Nothing survives a page refresh. This is intentional and part of the privacy
  model.

## 15. Rejected hallucination examples

Verified by the unit and security suites (fictional data):

| Proposed value | Original / context | Result |
| -------------- | ------------------ | ------ |
| `Reduced deployment time by 45%.` | resume says `30%` | `metric` fails → `unverified` |
| `Built data processing systems at Google.` | no `Google` in resume | `source_term` fails → `unverified` |
| `Built data processing systems in 2023.` | no `2023` in resume | `date` fails → `unverified` |
| `Software engineer experienced with Kubernetes.` | JD requires Kubernetes; resume lacks it | `skill` fails → `unverified` |
| `Experienced software engineer with expertise in Python and PostgreSQL.` (live Ollama) | resume has `experience`, not `experienced`/`expertise` | `source_term` fails → `unverified`, confirmation required |

In every case the edit is still offered, but clearly labelled and gated behind
explicit confirmation — it is never silently applied.

## 16. Verification results

All commands were run in this repository after the Phase 7C changes.

**Backend**

- `pytest` (full): **1148 passed, 2 skipped** (skips are live-database tests;
  no database is running). The Copilot subset alone is **269 passed**.
- New Phase 7C tests: **66 + 102 + 9 = 177**.
- `ruff check app tests/test_copilot_edits.py tests/test_copilot_edit_security.py
  tests/test_copilot_edit_api.py`: clean.
- `mypy app`: **Success: no issues found in 83 source files**.

**Frontend**

- `vitest run`: **119 passed** across **10 files** (new Phase 7C tests: 41 + 6
  = 47).
- `tsc --noEmit`: clean.
- `eslint`: clean.
- `next build`: clean (static `/` and `/_not-found`).

**Live local Ollama smoke test (bounded)**

`CopilotService` with the real provider chain,
`make_resume(summary="I am a software engineer with Python and PostgreSQL
experience.")`, `improve_summary`:

```
provider=local-ollama model=qwen2.5-coder:7b fallback=False
suggestion=sug_1 unverified edit={'path': 'summary', 'status': 'unverified',
  'proposed': 'Experienced software engineer with expertise in Python and
  PostgreSQL.', 'checks': [('metric', True), ('date', True), ('skill', True),
  ('source_term', False)]}
suggestion=sug_2 unverified edit=None
suggestion=sug_3 unverified edit=None
```

The local model produced the rewrite, the edit proposal targeted `summary`,
the four checks ran, and the untraceable new wording was surfaced as
`unverified` with confirmation required.

**Pre-existing, unrelated**

- `ruff check tests` reports **6 errors, all in `tests/test_phase_6g.py`**
  (import sorting, long lines, missing trailing newline). Phase 6G is
  untouched by Phase 7C and these were left as-is.

**Guarantees**

- No paid or cloud dependency was added; all generation stays on local
  loopback Ollama with a deterministic fallback.
- No resume, job, suggestion, or edit is persisted anywhere.
