# Phase 6G — ML Integration Report

Date: 2026-09-15. Report on integrating the locally-hosted MiniLM semantic
model into the production ResumeForge matching flow: what model was chosen,
how it is presented to users, how it is versioned and disclosed, which trained
models were deliberately **not** integrated and why, and the guardrails that
keep the deterministic Phase-5A signal authoritative. All claims below are
backed by code, tests, and the earlier phase documents cited inline.

---

## 1. Executive summary

Phase 6D/6E/6F trained and evaluated custom matching models (Tiers 0–5). The
headline findings were that no trained tier is a trustworthy production signal
(6F §5, §9) and the highest-value, lowest-risk ML addition is a general-purpose
local sentence embedding used as a *supporting* relatedness signal.

| Question | Answer | Section |
|---|---|---|
| What ML goes into production? | `sentence-transformers/all-MiniLM-L6-v2`, hosted locally in the Hugging Face cache, driven by the existing `app/semantic_matching` layer | §3 |
| How is it combined with deterministic matching? | Fixed 70/30 deterministic/semantic hybrid (Phase 5C), unchanged | §3 |
| Does semantic matching override evidence? | No — it never converts absence into possession; `missing_required` stays deterministic-only | §4 |
| Which trained tiers go to production? | None of Tier 3/4/5 | §5 |
| What happens if the model is unavailable? | Degraded graceful path: result returns deterministic-only, `semantic_availability` explains why | §6 |
| How is the model versioned? | `model_version` (pinned model id), `implementation_version` (semantic layer schema logic), both surfaced in API + UI | §7 |
| How is the model disclosed to users? | Top-level `model_disclosure` on the hybrid result + rendered disclosure card; license/source/version shown | §8 |
| Is the semantic score a probability? | No — UI and API both state it is relatedness, not a hiring probability | §8, §9 |
| What did verification show? | Backend full suite green (879 passed, 2 skipped); frontend 55 tests green; typecheck, lint, build, ruff, mypy all clean | §10 |

**Bottom line:** production matching = the proven deterministic MatchResult plus
a local, disclosed, versioned semantic *relatedness* signal blended at the
locked 70/30 weights. No paid service, no hosted model API, zero monetary cost.

---

## 2. Scope & method

- **Inputs.** Phase 5A deterministic matching (`app/matching`), Phase 5C hybrid
  blending (`app/hybrid_matching`), the semantic matching layer built in
  `app/semantic_matching` (which loads MiniLM via `sentence-transformers` from
  the local Hugging Face cache), and the 6D/6E/6F evidence (`docs/phase-6d-*`,
  `docs/phase-6e-*`, `docs/phase-6f-evaluation-error-analysis.md`).
- **Non-goals.** No retraining, no dataset-produced model in the runtime path,
  no new database tables, no persistence of match results, no changes to the
  Phase 5A/6A/6B scoring formulas, and no starting of Phase 7 work.
- **Method.** (1) Audit the candidate signals already available in the repo;
  (2) select MiniLM through the existing semantic layer as the sole ML signal;
  (3) add versioning and disclosure metadata to the hybrid payload; (4) build
  the front-end matching flow; (5) add regressions at the last layer verified.
- **Verification.** Backend pytest (full suite + a dedicated 6G file), frontend
  Vitest + Testing Library for every new component/client, `tsc --noEmit`,
  ESLint, `next build`, ruff, and mypy for the changed backend modules.

---

## 3. The integrated signal: local MiniLM via `app/semantic_matching`

### 3.1 Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, a 384-dim,
  Apache-2.0-licensed, general-purpose sentence embedding model.
- **Runtime:** loaded by `LocalSentenceTransformerProvider`
  (`backend/app/semantic_matching/model.py`) through `sentence-transformers`
  from the **local** Hugging Face cache. No external inference service is ever
  contacted.
- **Why this model:** it is the general-purpose embedding model already used
  and load-tested by the semantic layer; it is free, offline, fast on CPU, and
  documents relatedness without claiming possession.

### 3.2 Blend

The production response is the existing **Hybrid Match Score** (Phase 5C):
`0.70 * deterministic + 0.30 * semantic`, weights locked and surfaced in
`HybridMetadata.weights`. This meets the ladder requirement "hybrid stays
70/30". The deterministic `MatchResult` remains the authoritative object in the
payload; `semantic` is a separate, optional companion object.

### 3.3 API surface (additive, backward-compatible)

- `POST /api/v1/jobs/parse` (JD parsing) — unchanged.
- `POST /api/v1/matching/hybrid` — response gains three additive metadata
  pieces (all optional/defaulted, nothing removed or re-typed):
  1. `HybridMetadata.model_disclosure` (string) — the disclosure saw in §8.
  2. `SemanticMatchMetadata.model_version`,
     `SemanticMatchMetadata.implementation_version` (strings).
  3. `SemanticMatchMetadata.model_source`,
     `SemanticMatchMetadata.model_license` (strings; license from
     `MODEL_LICENSE = "Apache-2.0"`).
- No response field is removed; nested deterministic/semantic structures are
  unchanged types, satisfying the "additive-only" constraint.

---

## 4. Guardrails implemented

These are enforced by code structure and by regressions in
`backend/tests/test_phase_6g.py`:

1. **Deterministic remains authoritative for explicit requirements.**
   `matched_requirements` and `missing_required` at the hybrid level are the
   deterministic lists only. Test: "an absent skill stays missing even when
   semantically similar".
2. **Semantic never implies possession.** All semantic copy says "relatedness",
   and the missing-requirements block in the UI states items there were *not*
   found; relatedness does not convert them.
3. **No hiring-probability language.** `HYBRID_NOTE` and every UI string about
   the score say it is a heuristic relevance score, not a hiring probability,
   ATS probability, or employment prediction. Regression tests assert none of
   the positive phrases ("probability of hire", "you're likely to get", etc.)
   appear in the UI.
4. **70/30 weights are locked and documented.** `test_phase_6g` asserts the
   weight keys/values and the 5c-hybrid version string.
5. **No raw embeddings, no vectors in responses.** `app/hybrid_matching`
   exposes only aggregate similarity values; the 6G tests walk every field of
   payloads and assert no embedding/vector key exists.
6. **Null ≠ zero.** When a semantic component is not evaluable it is `None`,
   never a manufactured 0.
7. **Transient only.** Matches live for the duration of the request; nothing is
   written to PostgreSQL. UI communicates "not stored".

---

## 5. Trained tiers (Tier 3/4/5) — deliberately NOT integrated

6F concluded none of the trained models are production-worthy. The decision to
keep them out is therefore an *evidence-based* decision, not an omission:

| Tier | What it is | 6F finding | 6G decision |
|---|---|---|---|
| 1/2 | TF-IDF text logistics | weak | not integrated |
| 3 | MiniLM + small head, dataset-trained | ~0.72 macro-F1, coverage-adjacent signal, duplicates the MiniLM embedding already used, needs the 6E feature-pipeline to even run | **not integrated** — Tier 3 would add a dataset-specific trained head on top of the exact embedding signal the runtime already provides, at model-fleet cost and no independent signal gain |
| 5 | full quasi-generative feature model | 0.95 macro-F1 but driven entirely by coverage features that re-encode the deterministic pipeline; runs only on synthetic dataset inputs | **not integrated** — it is a cover-rule reproducer, not an independent matcher; requires the 6E feature extractor and dataset-shaped inputs that production resumes/JDs don't provide |
| 4 | structured-feature classical model | collapses to chance (0.5009) without the rule features (6F §5) | **not integrated** |

Also, Tier 3–5 artifacts are persisted only as experiment artifacts
(`backend/ml/artifacts/phase_6e/`, `phase_6f/`) and are not part of the runtime
service image or any API response. Rationale is preserved in
`docs/phase-6f-evaluation-error-analysis.md` (§5) and this section is the
authoritative "why not" record.

---

## 6. Degraded path: semantic model unavailable

When the local model is missing, fails to load, or inference fails, the hybrid
result continues to return a full, deterministic-backed result:

- `semantic` is `null`;
- `HybridMetadata.mode` is `deterministic-only`;
- `HybridMetadata.semantic_availability` carries
  `{available: false, status: <reason>, note}`;
- the UI shows a "Semantic model unavailable" banner and hides the disclosure
  card, while the deterministic match score is still shown.

This is covered by backend tests (`test_phase_6g`) and a frontend test
(`MatchResultView` unavailable-model case).

---

## 7. Model versioning

- **`model_version`** on `SemanticMatchMetadata` identifies *which model
  artifact* produced the vectors. The real provider pins it to the model id
  (`sentence-transformers/all-MiniLM-L6-v2`). `ModelMetadata` gained a
  defaulted `model_version: str = ""` so third-party/fake providers remain
  backward-compatible.
- **`implementation_version`** on `SemanticMatchMetadata` identifies *which
  version of the semantic layer logic* (thresholds, normalization, note text)
  produced the aggregates. Current value: `"1.0"`, a module constant
  `SEMANTIC_IMPLEMENTATION_VERSION` in `app/semantic_matching/config.py`.
- **Fallback:** when a provider does not pin `model_version`, the semantic
  service falls back to the provider's `model_name` so the field is never
  empty in real responses.
- Both values are serialized into every semantic result and render in the UI
  disclosure card ("Model", "Model version", "License", "Device").

---

## 8. Model disclosure

- **API:** `HybridMetadata.model_disclosure` defaults to
  `HYBRID_MODEL_DISCLOSURE` (defined in `app/hybrid_matching/schemas.py`):
  *"The semantic signal is produced entirely locally by an open-source
  sentence-embedding model (Apache-2.0). No external, hosted, or paid AI model
  is consulted, and no content leaves the machine."*
- **UI:** the result view renders this disclosure, plus a metadata list
  (model name, model version, device, license). If the semantic signal is
  unavailable, no disclosure is shown (nothing semantic ran).
- **Rationale:** meets the ladder requirements to document the ML model
  (§13 model disclosure, §14 model versioning) and supports the ₹0 / local-first
  story: the user can verify no external model is involved.

---

## 9. Frontend matching flow

New UI built this phase (there was no frontend for matching before):

- `frontend/src/lib/matcher.ts` — typed API client for JD parse +
  hybrid match (`parseJobDescription`, `matchHybrid`), plus result types and
  type guards. Errors map to user-facing messages; content is never logged.
- `frontend/src/components/MatchFlow.tsx` — "Match against a job description"
  section: choose a PDF/DOCX/TXT (≤10 MB, shared `validateResumeFile`), upload,
  parse, match, run/reset. Neutral copy; states loading text like "Match runs
  locally and is not stored."
- `frontend/src/components/MatchResultView.tsx` — renders:
  - the Hybrid Match Score with its 70/30 component breakdown
    (deterministic evidence-based vs semantic relatedness, in native scales);
  - the guardrail line "Relatedness is not a hiring probability and does not
    prove a candidate owns a skill. Deterministic evidence stays authoritative
    for explicit requirements.";
  - the semantic disclosure card (model, version, device, license, disclosure);
  - explicit vs not-explicitly-verified requirement lists;
  - the degraded-model banner when `semantic` is null.
- `frontend/src/components/ResumeAnalyzer.tsx` — wires `MatchFlow` into the
  results view (visible after a resume has been analyzed); `handleReset` clears
  any in-flight match state.

No new dependency was added on either side.

---

## 10. Verification & gates

### Backend
- New file `backend/tests/test_phase_6g.py` (13 tests): model-version and
  implementation-version presence and fallback; model-license/source values;
  disclosure present with default text; disclosure survives the degraded path;
  "absent skill stays missing" guardrail; hybrid stays 70/30 (`weights`,
  version string); component-scores layout (native scales, None semantics);
  payload walks asserting no vector/embedding fields in hybrid or semantic
  responses; additive-schema checks (new fields defaulted).
- Full suite: **879 passed, 2 skipped** (previously 866). `ruff check app`:
  clean. `mypy` on the changed modules (config, schemas, model, service,
  hybrid/schemas): clean.

### Frontend
- `frontend/src/lib/matcher.test.ts` — client contract, network/malformed/
  non-OK errors, degraded semantic payload acceptance, no-logging-on-success.
- `frontend/src/components/MatchResultView.test.tsx` (8 tests) and
  `frontend/src/components/MatchFlow.test.tsx` (6 tests) — full-flow render,
  guardrail wording, disclosure + metadata, unavailable-model path,
  error/alert, reset.
- Full suite: **55 tests passed**. `tsc --noEmit`: clean. ESLint on all new/
  changed files: clean. `next build`: clean production build.

### Manual smoke (transient matching flow)
- Upload a resume → analysis UI → results view shows the match section →
  choose a JD → parse → hybrid → result card renders scores, guardrail,
  disclosure, and requirement lists; "Run match" re-enables only after a new
  file is chosen; degraded path tested by disabling the model.

---

## 11. ₹0 / privacy compliance

- No paid API, hosted AI, external inference, or storage; MiniLM loads from the
  local Hugging Face cache (Apache-2.0).
- No dependency added; `docs/data-model.md` and `architecture.md` unchanged in
  spirit.
- Content stays on the machine during matching; nothing is persisted; PII is
  not transmitted anywhere (there is no external sink at all).
- The disclosure text is the user-facing guarantee of the above.

---

## 12. References

- `docs/phase-6d-ml-experiment-design.md`, `docs/phase-6e-ml-training-report.md`,
  `docs/phase-6f-evaluation-error-analysis.md` — the evidence this phase acts on.
- `docs/data-model.md` — transient schema conventions for matching results.
- Backend modules: `app/semantic_matching/{config,model,schemas,service}.py`,
  `app/hybrid_matching/schemas.py`.
- Frontend: `frontend/src/lib/matcher.ts`, `frontend/src/components/
  {MatchFlow,MatchResultView,ResumeAnalyzer}.tsx`.
- Tests: `backend/tests/test_phase_6g.py`, `frontend/src/lib/matcher.test.ts`,
  `frontend/src/components/{MatchFlow,MatchResultView}.test.tsx`.