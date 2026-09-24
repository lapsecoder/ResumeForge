# Phase 7B — Resume Copilot Operations

Date: 2026-09-17. Report on the Phase 7B implementation of the Resume
Copilot: the factual-validation ("hallucination guard") pipeline, the local
Ollama generative provider, and the Copilot UI. All claims are backed by code
and tests referenced inline. Phase 7B builds on the Phase 7A contract
(`docs/phase-7a-copilot-foundation.md`); nothing from 7A was reworked.

---

## 1. Executive summary

| Question | Answer | Section |
|---|---|---|
| What does 7B add? | A factual-validation layer between LLM output and the response, plus the full Copilot UI | §5, §12 |
| Who generates responses? | Local Ollama (`qwen2.5-coder:7b`) when reachable; deterministic fallback otherwise | §3, §6 |
| Does it invent facts? | No — any untraceable claim is demoted to `unverified` + user confirmation | §5 |
| Any content sent off the machine? | No — loopback-only, stdlib-only, PII stripped from the prompt | §4, §7, §9 |
| Cost? | ₹0 — urllib to a local Ollama, no new dependencies, no paid API | §3 |
| Anything persisted? | No — transient request/response; UI Apply/Undo is in-memory only | §10, §12 |
| Version | `7b-copilot-1.0` | §2 |

**Bottom line:** the Copilot is now a guardrailed local-LLM assistant: the
generative provider produces rewrites and explanations, and a deterministic
validation pipeline refuses to let unsupported facts through. The bespoke UI
in the app lets a user run any of the five controlled operations and review
evidence-labelled, verification-badged suggestions.

---

## 2. Scope & method

- **Inputs.** The Phase 7A operation contract (`app/copilot/operations.py`,
  `app/copilot/schemas.py`) is unchanged. Phase 7B adds:
  1. `app/copilot/validation.py` — deterministic factual validation for all
     LLM-produced suggestions.
  2. `app/copilot/prompting.py` — the only place prompts are built (system
     prompt, PII-stripped data blocks, strict output contract).
  3. Full `LocalOllamaProvider` run path wired to the validator
     (`app/copilot/providers/ollama.py`).
  4. Priority/issue/recommendation/impact fields on `CopilotSuggestion` and
     their deterministic population by `identify_priorities`.
  5. The Copilot UI (`frontend/src/components/CopilotPanel.tsx`) wired into
     the app at `frontend/src/components/ResumeAnalyzer.tsx:158`.
- **Non-goals.** Multi-turn chat, free-text prompting, remote/paid model APIs,
  persistence, auth, and any new runtime dependency. The operation contract
  remains "controlled": one request, one operation, bounded inputs.
- **Method & verification.** Everything is verified by the 92-test backend
  Copilot suite and the frontend panel/client suites, plus the full
  backend/frontend gates (see §13).

Implementation version is `7b-copilot-1.0`
(`COPILOT_IMPLEMENTATION_VERSION`, `app/copilot/config.py:26`).

---

## 3. Provider chain and the local Ollama provider

`CopilotService` (`app/copilot/service.py`) runs providers in fixed priority
order: `[LocalOllamaProvider, DeterministicFallbackProvider]`. All five
operations are generative-capable (every contract has
`fallback_available=True`), so a healthy Ollama answers every operation; any
`CopilotError` falls through to the deterministic provider, which is always
available.

`LocalOllamaProvider` (`app/copilot/providers/ollama.py`):

- **Availability.** `available()` probes `GET /api/tags` with a 2.5 s timeout
  and confirms the configured model is installed. It never raises, never
  downloads a model, and returns `False` when `COPILOT_LLM_ENABLED=false`.
- **Model.** Default `qwen2.5-coder:7b` (`DEFAULT_OLLAMA_MODEL`,
  `config.py:21`), verified against installed models. `qwen2.5-coder:7b`
  support was validated live (see §13).
- **Generation.** A single bounded `POST /api/chat` with `format="json"`,
  `stream=false`, and `options.num_predict` fixed server-side to
  `llm_max_tokens` (default 2048) inside a 120 s hard timeout. Clients cannot
  tune generation parameters (`ollama.py:112-119`).
- **Failure mapping.** `TimeoutError` → `ProviderTimeoutError`;
  `URLError`/`HTTPError` → `ProviderUnavailableError`; non-object/missing
  message/empty content/malformed JSON → `MalformedProviderOutputError`. All
  are `CopilotError` subclasses, so the service degrades to the fallback.
- **Zero dependencies.** stdlib `urllib` only; no new package.

The deterministic fallback (`app/copilot/providers/deterministic.py`) is
unchanged in nature from 7A but now also populates the 7B priority fields.

---

## 4. Structured output contract

The model is asked for exactly one JSON object
(`prompting.py:_OUTPUT_FORMAT_INSTRUCTIONS`):

```json
{
  "explanation": string,
  "suggestions": [
    {
      "category": string,
      "original_text": string,
      "suggested_text": string,
      "rationale": string,
      "verification": "verified|inferred|advisory|unverified",
      "requires_user_confirmation": boolean,
      "evidence": [{"kind": "fact|job_requirement|inference|suggestion|unverified",
                    "source": string, "statement": string, "reference": string}]
    }
  ]
}
```

`parse_llm_json` (`prompting.py:369`) strictly validates this shape: it must
contain `explanation` (string) and `suggestions` (list); every suggestion must
have string fields, a valid `category`, a valid `verification` literal, a
boolean `requires_user_confirmation`, and a list-of-objects `evidence` whose
`kind` is a valid literal. Any deviation raises `MalformedProviderOutputError`
→ deterministic fallback. Only the first `max_suggestions` (8) suggestions are
kept, each coerced into the `CopilotSuggestion` response contract
(`ollama.py:151-181`). No chain-of-thought is ever requested or surfaced.

---

## 5. Factual validation (hallucination guard)

`app/copilot/validation.py` runs inline after parsing every LLM response
(`ollama.py:184`). It is deliberately conservative and deterministic — it
never rejects and never hits a network. Design principle: instead of a
"perfect truth detector", any uncertain claim is downgraded to
`VerificationLevel.UNVERIFIED` with `requires_user_confirmation=true` plus a
`copilot.validation` evidence note, and the user decides.

The four checks:

1. **Metrics** (`_check_metrics`). Numbers/percentages in a
   `verified`/`inferred` rewrite must be traceable to the resume + target-text
   corpus (canonicalised via `_canonical_number`: commas stripped, trailing
   zeros trimmed). A rewrite introducing `1,000,000 users` when the corpus
   never says it is demoted.
2. **Skills** (`_check_skills`). A JD skill claimed by an authoritative
   rewrite must be supported by the resume's normalised skill set or resume
   text (whole-token matching, so `java` never satisfies `javascript`).
   The JD alone never supports a possession claim.
3. **New terms** (`_check_new_terms`). Significant tokens in a rewrite must
   exist in the resume+target corpus (stopwords stripped, trailing periods
   trimmed). This catches fabricated employers, titles, organisations,
   degrees, certifications, and dates written as words.
4. **Evidence sanitisation** (`_sanitize_evidence`). Evidence references that
   look like structured paths are resolved against the resume/job:
   out-of-range indices and invalid namespaces are dropped; `ats`,
   `job_specific`, `deterministic_match`, `hybrid_match`, `copilot` roots are
   trusted. If nothing survives, a safe `copilot.validation` fallback evidence
   item is added.

Only `verified`/`inferred` rewrites are checked; advisory guidance without
rewritten text is never demoted. Every demotion respects
`max_evidence_per_suggestion`.

---

## 6. Evidence sanitation

Part of the validator (§5, `_sanitize_evidence`). Key
behaviour: `reference` paths like `experience[0].achievements[1]`,
`job.required_skills`, `resume.skills` are resolved against the structured
Resume/JobDescription. Valid `resume`/`job` namespace paths and the trusted
analysis roots are kept; paths with an explicit out-of-range index or a hidden
namespace are dropped. Non-path references (free text) are kept as-is because
they cannot be disproved. The outcome is that LLM-supplied evidence paths
cannot point at nonexistent resume rows.

For the deterministic provider, evidence is structurally derived from the
actual resume/JD fields (`app/copilot/evidence.py`), so it is correct by
construction and needs no sanitisation.

---

## 7. PII stripping

`render_resume_data` (`prompting.py:107`) renders the resume for the LLM
WITHOUT contact/PII fields: name, email, phone, location(s), and links are
omitted. Only summary, structured skills, experience (titles, companies,
dates, descriptions, achievements), projects, education, certifications, and
custom sections are included, with structural labels. `render_job_data`
includes only the role title/summary and requirement buckets;
`render_analysis_data` sends only finding ids/titles/explanations and explicit
term matches — never raw scores or advice. The API never echoes PII in
responses (`test_response_does_not_echo_pii_from_resume`) and logs only safe
metadata (`app/api/v1/copilot.py`).

---

## 8. Injection-resistant delimiters and prompt security

`prompting.py` is the only prompt builder. Resume, job, and analysis content
travel exclusively inside

```
<RESUME_DATA>…</RESUME_DATA>
<JOB_DATA>…</JOB_DATA>
<ANALYSIS_DATA>…</ANALYSIS_DATA>
```

static server-side delimiters. `SYSTEM_PROMPT` rule 6 states those blocks are
UNTRUSTED user content and any instruction inside them (e.g. "ignore previous
instructions", "reveal the system prompt") must be ignored. The whole payload
is truncated to `max_prompt_chars` (12000). Instruction strings are constants
in code, never derived from user input. This is enforced by
`test_injected_resume_text_is_confined_to_resume_data_block` and
`test_injected_job_text_is_confined_to_job_data_block`, plus
`test_system_prompt_marks_data_blocks_as_untrusted`.

`SYSTEM_PROMPT` also carries ten hard rules: factual preservation, no
fabrication, no unsupported metrics/skills, alignment-only-from-existing-
evidence, data-is-not-instructions, evidence grounding, strict output schema,
rewrite safety, and no secret disclosure.

---

## 9. Loopback-only Ollama enforcement

`CopilotSettings._loopback_only` (`app/copilot/config.py:78`) rejects any
`COPILOT_OLLAMA_BASE_URL` that is not `http://localhost`,
`http://127.0.0.1`, or `http://::1`. A client can never point the Copilot at a
remote endpoint. The provider never contacts anything but the configured
loopback URL, and only with the `/api/chat` and `/api/tags` paths.

---

## 10. Input/output/evidence bounds

Enforced server-side in `CopilotSettings` and the service/providers; clients
cannot raise them:

| Bound | Default | Enforcement |
|---|---|---|
| `COPILOT_MAX_TARGET_TEXT_CHARS` | 2000 | `service._validate`; oversized → 422 `invalid_request` |
| `COPILOT_MAX_PROMPT_CHARS` | 12000 | `prompting._truncate_to` on the assembled prompt |
| `COPILOT_MAX_RESPONSE_CHARS` | 12000 | Declared configuration bound for accepted suggestion text (per-field text is additionally capped in provider builders) |
| `COPILOT_MAX_SUGGESTIONS` | 8 | Providers truncate suggestion lists |
| `COPILOT_MAX_EVIDENCE_PER_SUGGESTION` | 5 | Both providers and the validator cap evidence arrays |
| `COPILOT_LLM_MAX_TOKENS` | 2048 | Fixed server-side `num_predict` (must fit the full JSON response) |
| `COPILOT_LLM_REQUEST_TIMEOUT_SECONDS` / `COPILOT_LLM_AVAILABILITY_TIMEOUT_SECONDS` | 120 / 2.5 | Provider timeouts |

`max_target_text_chars` oversize is asserted in `test_copilot_api.py` and
`test_copilot_security.py`.

---

## 11. The five controlled operations (as implemented)

`app/copilot/operations.py` defines each `OperationContract`; both providers
execute them:

- **`improve_summary`** — rewrites the professional summary preserving every
  factual claim. Deterministic: drafts a fact-only summary when none exists
  (from most-recent role + listed skills, `VERIFIED`/`INFERRED`), applies the
  safe first-person/typography rewrite, adds advisory "too short /
  quantification" guidance, and surfaces resume skills that the job also asks
  for. Generative: the LLM rewrites with factual-preservation rules, and the
  validator checks the rewrite.
- **`improve_bullet`** — requires `target_ref` or `target_text`. Deterministic:
  safe rewrite (first-person strip, capitalisation, whitespace), then
  advisory checks for vague phrasing (`is_vague`), missing quantification, and
  non-action openings. Generative: same through the LLM, validated for
  traceable numbers/skills/terms.
- **`identify_priorities`** — ranks the highest-impact fixes. Deterministic:
  ranks findings from `analysis.ats_readiness` + `analysis.job_specific_ats`
  (severity → impact → rule id), maps finding categories to suggestion
  categories, adds job-missing-skill items (labelled `UNVERIFIED`, never a
  possession claim), and heuristic resume checks (missing summary, no
  quantitative lines, no action verbs, sparse word count). Populates the 7B
  priority fields: `priority` (1 = highest), `issue`, `recommendation`,
  `impact` (`high`/`medium`/`low` by the finding's impact weight).
- **`explain_finding`** — requires `finding_ref`. Deterministic: looks up the
  rule id in the supplied analyses and returns the finding's own
  explanation/recommendation, or a graceful "not found" advisory. Never
  predicts hiring outcomes.
- **`job_alignment`** — requires `job_description`. Deterministic: matched
  skills (resume ∩ job) are surfaced `VERIFIED` as alignment candidates; job
  skills missing from the resume produce `UNVERIFIED` guidance to add them
  only if genuinely true; never keyword-stuffing advice. All five operations
  are also available through the LLM with the same safety rules.

Every response carries the standard disclaimer and `ProviderMetadata`
(`provider`, `provider_label`, `model`, `fallback_used`, `version`, `note`).

---

## 12. Frontend CopilotPanel integration

`frontend/src/components/CopilotPanel.tsx` (wired at `ResumeAnalyzer.tsx:158`)
is the full Copilot UI:

- **Operation selector** — the five controlled operations; `job_alignment` is
  gated on a completed match.
- **Grounding preparation** — runs ATS readiness + job-specific ATS analysis
  locally first and passes the results as `CopilotAnalysisContext`, so
  `explain_finding`/`identify_priorities` can reference real finding ids.
- **Target selection** — `improve_bullet` offers structured targets
  (`experience[i].achievements[j]`, `experience[i].description`,
  `projects[i].description`, `summary`) or a custom bounded bullet;
  `explain_finding` offers a finding dropdown. `target_ref`/`target_text`/
  `finding_ref` are sent to the backend, which resolves and re-validates them.
- **Output stage** — renders explanation, provider label + `· fallback` badge
  + version, verification badges (`verified`/`inferred`/`advisory`/
  `unverified` + "Review needed"), priority/issue/recommendation/impact
  (7B fields), evidence list, and a verified-count summary.
- **Apply/Undo** — per-suggestion `Apply`/`Undo` toggle tracks a session-only
  in-memory `Set` of suggestion ids. It does not mutate the resume and
  nothing is saved or sent when toggled; the panel labels applied cards
  "Applied in this session only — nothing is saved or sent to a server."
  There is no server-side state.

`frontend/src/lib/copilot.ts` mirrors the backend contract as strict TS types
and provides `suggestCopilot()` / `getCopilotStatus()` (fetch-based,
`ApiError`-consistent).

---

## 13. Testing, verification, and live Ollama smoke test

### Backend

- Full pytest suite: **971 passed, 2 skipped** (the 2 skips are live-database
  tests; no database was running).
- Copilot suite (92 tests): `test_copilot_api.py` (10), `test_copilot_fallback.py` (17),
  `test_copilot_providers.py` (18), `test_copilot_schemas.py` (9),
  `test_copilot_security.py` (17), `test_copilot_validation.py` (21).
  Highlights:
  - Validation: `test_hallucinated_metric_is_demoted`, `test_fabricated_wording_is_demoted`,
    `test_new_employer_in_rewrite_is_demoted`, `test_unsupported_job_skill_claim_is_demoted`,
    `test_supported_job_skill_claim_is_allowed`, `test_out_of_range_evidence_path_is_dropped`,
    `test_valid_structured_evidence_path_is_kept`, `test_number_canonicalisation[...]`,
    `test_word_boundary_prevents_substring_false_positives`,
    `test_multiple_suggestions_are_validated_independently`.
  - Providers/fallback: `test_ollama_*` (availability, structured response,
    timeout/unreachable mapping, malformed rejection), `test_service_*`
    (chain preference, fallback on malformed/unavailable, status reporting),
    fallback fact-preservation and no-fabrication tests.
  - Security: `test_rendered_resume_excludes_contact_information`,
    `test_ollama_body_is_server_controlled`, `test_llm_payload_contains_no_pii`,
    injection confinement, no-fabricated-metrics/skills, PII non-echo.
- `ruff check` and `mypy` on `app/copilot`: clean.

### Frontend

- Vitest: `src/lib/copilot.test.ts` + `src/components/CopilotPanel.test.tsx`
  (17 tests) — client happy path/errors, 503 handling, panel rendering for all
  five operations, 7B priority-field rendering, Apply/Undo, job-gating, error
  states.
- `tsc --noEmit`, ESLint, and `next build`: clean.

### Live local Ollama smoke test (bounded)

- Environment: Ollama 0.33.3 running locally; `qwen2.5-coder:7b` installed;
  RTX 5050 8 GB.
- The provider already passes direct-verification checks:
  `ollama run qwen2.5-coder:7b "…OLLAMA_WORKS"` and HTTP `POST /api/generate`
  both succeed.
- Bounded `POST /api/chat` against `http://127.0.0.1:11434` with
  `format=json`, `num_predict=24`, and a 90 s timeout: **HTTP 200**,
  `model=qwen2.5-coder:7b`, content `{"answer": "OLLAMA_WORKS"}`.
- End-to-end through the real `LocalOllamaProvider.run()` (improve_bullet on a
  sample resume): **success**. `provider=local-ollama`, `model=qwen2.5-coder:7b`,
  3 suggestions. The factual guard demoted the model's invented
  "20% increase in system performance" claim to `unverified` +
  `requires_user_confirmation=true`, confirming the validator works against
  live output.
- **Smoke-test finding and fix.** The first live run raised
  `MalformedProviderOutputError`: with the original `llm_max_tokens=700`, the
  model returned `done_reason=length` and truncated JSON. The default was
  raised to `2048` (the response completed at `eval_count=824`, valid JSON)
  and the output instruction now requests at most 3 concise suggestions. After
  the fix the end-to-end call succeeds. This is the only behavioural change
  made to complete Phase 7B.
- The 92 backend Copilot tests never contact real Ollama (monkeypatched
  transport), so the suite is hermetic.

---

## 14. Security & privacy guarantees

- Loopback-only LLM endpoint (§9); no remote inference.
- Contract/no-chat input model; server-controlled generation parameters (§3).
- PII is stripped from the prompt (§7) and never emitted in responses or logs.
- Prompt-injection-resistant data blocks (§8).
- Factual-validation pipeline (§5) + evidence sanitation (§6).
- Input/output/evidence bounds enforced server-side (§10).
- No persistence anywhere: transient schemas, transient requests, in-memory
  Apply/Undo in the UI (§12, §16 of the 7A doc).
- Structured logging only of operation + error class + request_id; never
  content.

---

## 15. Known limitations

- **Conservative validator.** The new-term and metric checks can demote a
  legitimate reworded claim when the resume corpus lacks its wording; the
  user is shown why and can confirm. No "perfect truth detector" exists.
- **Deterministic cross-checks only rewrite-level claims.** Advisory and
  guidance suggestions (which add no rewritten text) are not run through the
  metric/skill/new-term checks — by design (§5).
- **`max_response_chars`** is declared as a configuration bound but is not a
  hard runtime cap on total rendered suggestion text; per-field text and
  suggestion/evidence counts are capped instead (§10).
- **Local model quality.** `qwen2.5-coder:7b` is a coding model used for
  resume-writing; outputs are structurally validated and factually demoted,
  so quality is bounded by the validator, not the model.
- **No chat, no persistence, no auth, no remote models** — unchanged from 7A
  and intentionally out of scope (repo roadmap).
- **Ollama cold start.** First call after Ollama loads the model into VRAM
  can approach the 120 s request timeout; `available()` uses a 2.5 s probe and
  reports `False` while the model is loading, routing to the deterministic
  fallback until Ollama exposes the model in `/api/tags`.
- **Generation length is bounded.** `llm_max_tokens` (2048) is a hard cap;
  extremely long inputs could still truncate the JSON and trigger the
  deterministic fallback. The output instruction asks for at most 3 concise
  suggestions to leave headroom. Raising the cap raises per-request latency on
  the local GPU.

---

## 16. Results

- Files created by Phase 7B (this phase adds to the Phase 7A base):
  `app/copilot/validation.py`, `app/copilot/prompting.py` (finalised),
  `app/copilot/config.py` (7B version + bounds),
  `app/copilot/schemas.py` (7B priority fields),
  `app/copilot/providers/ollama.py` (validator-wired run path),
  `app/copilot/providers/deterministic.py` (priority population),
  `tests/test_copilot_validation.py`, `tests/test_copilot_security.py`,
  `tests/test_copilot_schemas.py`, `tests/test_copilot_fallback.py`,
  `frontend/src/lib/copilot.ts` + `copilot.test.ts`, and
  `frontend/src/components/CopilotPanel.tsx` + `CopilotPanel.test.tsx`, plus
  this report.
- Added to finish the phase: this report; the missing bounds
  (`COPILOT_MAX_PROMPT_CHARS`, `COPILOT_MAX_RESPONSE_CHARS`,
  `COPILOT_MAX_EVIDENCE_PER_SUGGESTION`) in `.env.example`; and the live
  token-budget fix (`llm_max_tokens` 700 → 2048, concise 3-suggestion output
  instruction) in `app/copilot/config.py` and `app/copilot/prompting.py`.
- No new dependencies; no paid/cloud service introduced.
- Gates (run during this phase, on Windows/PowerShell):
  - Backend full pytest: **971 passed, 2 skipped** (the 2 skips are the
    live-database tests; no database was running). Copilot-only subset: 92.
  - Backend `ruff check app/copilot app/api/v1/copilot.py tests/test_copilot_*.py`:
    clean. `mypy app`: clean (82 files).
  - Frontend Vitest: **72 passed** (8 files), including the 17 Copilot
    client/panel tests. `tsc --noEmit`, ESLint, `next build`: clean.
  - One pre-existing, unrelated `ruff check` finding remains in
    `tests/test_phase_6g.py` (Phase 6G import order + line length + missing
    final newline); it is not Copilot code and was left untouched.
- Phase 7B is complete.