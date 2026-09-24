# Phase 6B + 6C — Two-Track Final Report

Date: 2026-09-13. Track A (Phase 6B) is a runtime feature: deterministic
Job-Specific ATS Coverage analysis. Track B (Phase 6C) is an audit-only
foundation for the HF synthetic candidate-matching dataset. Everything below
was executed on this machine against the pinned dataset snapshot.

---

## Track A — Job-Specific ATS Coverage (Phase 6B)

### What was built

New package `backend/app/ats_analysis/job_specific/` plus one endpoint:

| File | Role |
|---|---|
| `schemas.py` | `MatchType` (exact/normalized/alias/phrase/absent), `TermOrigin` (required/preferred/phrase/skill_only), `TermMatch`, `CoverageTotals`, `CoverageReport`, `JobSpecificATSResult` |
| `rules.py` | Weights required 50 / preferred 20 / phrases 15 / evidence 15, caps, phrase window limits, `METHOD_NAME=job-specific-ats-coverage-heuristic`, `VERSION=6b-ats-1.0`, disclaimer, score labels |
| `heuristics.py` | Token-preserving presence, meaningful-token phrase matching, term resolution priority exact→alias→normalized→phrase→absent, evidence / skills-only location tracking |
| `scorer.py` | `compute_overall` with weight redistribution for N/A categories, label bounds Weak <50 / Needs Improvement 50–64.9 / Good 65–79.9 / Strong ≥80 |
| `analyzers.py` | `analyze_job_specific` producing category scores, coverage fractions, term matches, and findings |
| `service.py` | `analyze_job_specific_ats` result assembly, deterministic ordering |

Endpoint: `POST /api/v1/resumes/job-specific-ats` in
`backend/app/api/v1/resumes.py`, accepting `{resume, job_description}` and
returning the full coverage report.

Supporting change: `backend/app/matching/normalizer.py` gained `STOPWORDS`,
`normalized_base()`, and `alias_target()` (existing behaviour preserved — the
pre-existing 6A suite still passes).

### Design decisions (locked)

- **Transient-only**: resume + job parse in memory; no persistence, no
  logging of content, no database, no network, no embeddings, no LLM.
- **Deterministic**: same inputs → byte-identical output (asserted by tests).
- **Token discipline**: presence tokens keep punctuation, so `C`≠`C++`,
  `.NET`≠`net`, `Node.js`≠`Node`/`nodejs`, `Java`≠`JavaScript` — searches
  never collide because of substrings.
- **Phrase matching is alignment-robust**: stopwords are skipped on both sides
  (`design data pipelines analytics` matches "...for analytics dashboards").
- **Never infers possession**: an absent required term lowers the coverage
  score and is reported as missing; it is not interpreted as "hiring risk".
- Findings never assert hiring or ATS-pass probability; a disclaimer heads
  every result.

### Evidence of correctness

- 4 test files, 27 scenarios with hand-computed expectations:
  - `test_job_specific_ats.py` — matching types, token discipline,
    normalisation, buckets, evidence, scoring, output ordering.
  - `test_job_specific_ats_api.py` — 200 shape, bounds, distinct method name,
    determinism, 422 safety, health unchanged.
  - `test_job_specific_ats_security.py` — no temp files, no DB imports, no
    network/embedding/LLM imports, no content/PII logging, no persistence.
  - `test_job_specific_ats_evaluation.py` — scenario anchors: perfect
    alignment 100.0; skills-only 50.0; one missing required 70.0; empty job
    → None with `ats.job.limited_terminology`; Java-vs-JS 0.0.
- Full backend suite: **738 passed, 2 skipped**.

---

## Track B — Candidate-Matching Dataset Audit (Phase 6C)

### Dataset facts (verified)

- `michaelozon/candidate-matching-synthetic` at revision
  `178ab864dcad9910c5670d43e4bdbbb901a11f18`.
- Frames: **10,000 resumes** × 9 cols; **2,500 jobs** × 9 cols; **2,500 match
  rows** each with exactly **30 relevant resumes** → **75,000 pairs**;
  9,971 of 10,000 resumes are ever labeled (29 never appear).
- `datasets.load_dataset` only exposes the `resumes` split; `data_files=`
  with mixed schemas raises a `datasets.table.CastError`. The supported loader
  is a pinned `hf_hub_download` + `pandas.read_parquet` (implemented in
  `ml/loader.py`, offline-first).
- Generation rule (reconstructed from the notebook, SEED 42): must-have skill
  coverage ≥ 60%, random sample capped at 30.

### Audit findings (all from the run artefacts)

- **No binary label field** — labels live as per-job `relevant_resume_ids`.
- **Rule-exactness**: 100.0% of published labels are reachable by the
  reconstructed rule (labels ⊆ rule candidates), but the 30-cap discards
  **96.5% of candidate mass** (mean ≈866 candidates/job) — so labels are
  complete for the positive side and heavily truncated on negatives.
- **Language is template-heavy**: 100.0% of summaries match
  `"{role} with {years} years of experience in {industry}"`; the 30,000
  experience bullets are drawn from only **3 distinct strings**; resume and
  job vocabularies are identical (24 shared roles, 10 shared industries,
  73 identical skill vocabulary items, 100% must-have coverage by resume
  skills).
- **Quality**: zero empty fields, zero negative years, zero duplicate job
  ids, zero rows referencing missing jobs/resumes, zero PII-like strings
  detected (synthetic data is clean).
- **No repository split** exists.

### Workspace delivered (`backend/ml/`)

`loader.py`, `audit/{statistics,target,leakage,splits,quality,report}`,
`models/baselines.py` (contracts only), `scripts/run_dataset_audit.py`,
plus generated artefacts under `ml/artifacts/`:
`audit_report.md`, `audit_report.json`, and 4 EDA PNGs (top skills, seniority
supply vs demand, resume exposure).

Run: `.venv\Scripts\python -m ml.scripts.run_dataset_audit` from `backend/`.
Obligatory privacy property (tested): every audit output is aggregate only —
markers/emails/resume-ids inserted into inputs never surface in any result.

### Baselines defined for Track C (not trained)

A uniform-random, B exact role match, C must-have coverage (the generation
rule), D deterministic TF-based overlap, E local sentence-embedding cosine.

### Split-strategy recommendation

No split exists in the repo; recommended split keeps every job's 30 labels
together (job-grouped), dedupes near-duplicate template resumes into single
splits, and treats shared skill vocabulary as a feature, not leakage.

---

## Quality gates (all run this session)

| Gate | Result |
|---|---|
| `pytest` (backend, full) | **738 passed, 2 skipped** |
| `ruff check app ml tests` | **All checks passed** |
| `mypy app ml` (strict) | **Success, 86 source files** |
| Audit run (real dataset) | artefacts regenerated cleanly |

Tooling note: `pyproject.toml` mypy `python_version` moved 3.11 → 3.13 to
match the venv runtime (3.13.15) — numpy/matplotlib ship stubs that only
parse on ≥3.12. No gate failure remains, new-format parity holds, and the
`docstring-attr` cov of prior phases was preserved.

## Environment constraint

`git` is not installed and `D:\ResumeForge` is not a repository, so the
final `git diff` inspection gate could not run. All file changes are listed
in the per-phase work logs; nothing was staged or committed.

## Files created/modified

- Created: `backend/app/ats_analysis/job_specific/*`
- Created: `backend/tests/test_job_specific_ats.py`,
  `test_job_specific_ats_api.py`, `test_job_specific_ats_security.py`,
  `test_job_specific_ats_evaluation.py`
- Modified: `backend/app/matching/normalizer.py`,
  `backend/app/api/v1/resumes.py`
- Created: `backend/ml/**` (workspace + artifacts)
- Created: `backend/tests/test_ml_loader.py`, `test_ml_audit.py`,
  `test_ml_security.py`
- Modified: `backend/pyproject.toml` (ml extras, network marker, mypy
  overrides + 3.13), `D:\ResumeForge\.gitignore` (ML artifacts/cache)

No paid service, API, or dependency was added (all additions are free/open:
pandas, datasets, matplotlib, huggingface_hub — already installed). Track B
introduces no application behavior.