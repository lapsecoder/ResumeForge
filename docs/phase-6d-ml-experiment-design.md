# Phase 6D — ML Experiment Design & Dataset Strategy

Status: **design + contracts**, no training. The experiment harness contract is
pinned so Phase 6E can implement fitting against stable signatures. Nothing in
this phase trains a model, fine-tunes a transformer, produces binaries, or is
wired into the FastAPI application.

Dataset and audit ground truth: `docs/phase-6b-6c-report.md`.

---

## 1. Problem definition (task statement)

**Task.** Supervised binary relevance: a model receives one `(job, resume)`
pair and predicts whether the resume is *relevant* for that job.

- Input: a job description and a resume, both as structured attributes and
  raw text.
- Output: `relevant ∈ {0, 1}`.
- Interpretation: *fit for an opening*, **not** hiring probability, not ATS
  pass probability, not a match score, not a ranking. Phase 5/6 already own
  explainable scores; the model is a separate, measured hypothesis.

**Why classification and not regression.** The dataset carries no continuous
grade; it carries a binary membership (`relevant` vs not). Any continuous
target would have to be derived from the coverage rule — a target-derived
construct that would (a) bake the generation rule into supervision and
(b) reduce to a classification anyway at evaluation.

**Why binary membership and not a multi-class "degree of fit".** The published
labels are binary per pair. Generalizing them into degrees would be inventing
labels, which Phase 6D forbids.

## 2. Source dataset and target construction

Single source: `michaelozon/candidate-matching-synthetic` at fixed revision
`178ab864dcad9910c5670d43e4bdbbb901a11f18` (pinned in `ml/config.py`, loaded
by `ml/loader.py`). Audited structure (Phase 6C):

| Table | Rows | Content |
|---|---|---|
| `resumes` | 10,000 | resume_id, role, seniority, years_experience, industry, skills, education, summary, experience_bullets |
| `jobs` | 2,500 | job_id, job_title, seniority, industry, must_have_skills, nice_to_have_skills, description, responsibilities, requirements |
| `matches` | 2,500 | job_id → `relevant_resume_ids` (exactly 30/resume row, 75,000 positive pairs) |

Generation rule (verified): a resume is published as relevant when its
must-have coverage ≥ 0.6; the generator then random-samples (seed 42) up to
30. 100% of labels are reachable by that rule; the 30-cap discards ~96.5% of
rule-candidates (~866 candidates/job on average), so **the dataset is a capped
positive sample** — far from exhaustive.

**Target construction** (`ml/data/pairs.py`):

- Positive pair: `(job, resume)` where `resume ∈ relevant_resume_ids[job]`.
- Negative pair: per job, a seeded random sample of `negatives_per_job`
  resumes that are *not* in that job's positive list.
- **Sampled-negative semantics:** a negative means "not published as
  relevant", not "certified irrelevant". Because of the 30-cap, some sampled
  negatives genuinely satisfy the coverage rule. This is documented in code
  and in every report; no model can extract more signal than the cap admits.
- Default `negatives_per_job = 30` (1:1 balanced), seed 42, deterministic via
  `random.Random(f"{seed}::{job_id}")`.
- The full pair frame is built per split so negative sampling never leaks
  across partitions (each split samples its own negatives).
- `resume_allowlist` lets the strict both-group split keep each resume
  visible to exactly one partition.

## 3. Feature boundary (approved & excluded)

The registry in `ml/features/registry.py` is the single source of truth; a
`KeyError` or a programmatic misuse can never quietly feed a forbidden column.

### Approved families (trains models may consume)

| fid | What it is | Leakage class |
|---|---|---|
| `resume_text` | summary, experience_bullets, skills-as-text | none |
| `job_text` | description, responsibilities, requirements-as-text | none |
| `structured_resume` | role, seniority, years_experience, industry, education | none |
| `structured_job` | job_title, seniority, industry | none |
| `deterministic_match` | Phase-5 keyword overlap / semantic cosine / hybrid | quasi-generative |
| `ats_6a` | general ATS heuristics (grammar/impact/coverage style) | none |
| `ats_6b` | job-specific coverage/evidence (see exclusions below) | quasi-generative |
| `minilm` | local all-MiniLM-L6-v2 cosine + per-block cosines | none |

### Excluded (trained models must NOT consume — enforced by registry)

| fid | Reason |
|---|---|
| `identity` (`resume_id`, `job_id`) | primary keys → model memorises the label table |
| `raw_must_have` (raw `must_have_skills`, `nice_to_have_skills`) | these **are** the generation-rule inputs; feeding them exactly re-implements the labels |
| `generative_rule` (candidate flags, coverage≥0.6, capped membership) | target-derived formula |
| `provided_embeddings` (dataset-shipped arrays) | unknown provenance, not reproducible |

### Sub-field exclusions within an approved family

`ats_6b` may be used except `required_coverage` and `overall_coverage`, which
nearly reproduce the coverage rule directly. `preferred_coverage`,
`evidence_supported_*` stay (the generator's rule used only must-have skills).

### Leakage policy (summary)

1. **Identity / target** → never a feature.
2. **Generative-rule inputs** (`raw_must_have`) → never a raw feature. The
   *same content* may still reach the model naturally through `job_text` and
   `structured_job`.
3. **Target-derived** (`generative_rule`, the coverage threshold itself) →
   never a feature or a label transformation.
4. **Quasi-generative** (`deterministic_match`, `ats_6b`) → allowed but held
   apart in the ablation canary group (`CANARY_FAMILIES` in `builders.py`).
   The final report must state macro-F1 with and without this group so
   "matching skill" gains are separated from "rule reproduction".
5. **Provided embeddings** → never; local embeddings replace them with full
   provenance.

## 4. Preprocessing design

`ml/preprocessing/text.py` and `ml/preprocessing/structured.py` ship
deterministic, stateless, pure functions (NFKC, whitespace collapse,
technical-term-preserving lowercase tokenisation, seniority/industry/education
bucketing).

- `C++ → c++`, `.NET → .net`, `Node.js → node.js`, `aws-s3 → [aws, s3]`.
- Statelessness is enforced by design and tested (same input ⇒ same output;
  no module imports sklearn).
- **Fit-on-train only** applies to any learned representation stacked on top
  (TF-IDF vocabulary, scalers, encoders, projections). `builders.py` lists
  `FIT_ON_TRAIN` families; learned artifacts are bound inside a fitted
  `FittedTransformer` that never sees validation/test pairs at train time.

## 5. Feature engineering plan

Deterministic families are built with the same token discipline as Phase 5/6
writers so feature meaning is consistent end-to-end:

1. Text: per-family TF-IDF (fit-on-train) over `resume_text` / `job_text`.
2. Structured overlap: role, seniority, industry, degree matching (bool), years
   bucket; education level via `degree_level`.
3. Deterministic match: Phase-5 overlap/cosine/hybrid scores recomputed at
   inference.
4. ATS: Phase-6A category scores; Phase-6B allowed coverage/evidence
   sub-fields only.
5. Semantic: MiniLM cosine of resume vs job and per-section vectors.

Same-column guarantee: every transformer returns a plain feature table with no
`resume_id`/`job_id` columns; the registry + builder signature make identity
leakage structurally impossible.

## 6. Experiment tiers (models to compare)

Declared in `ml/experiments/config.py` (`TIERS`), run in 6E:

| Tier | Model | Feature families | Why |
|---|---|---|---|
| 0 | majority baseline | — | chance floor, must be beaten |
| 1 | TF-IDF + LogisticRegression | text | vocabulary overlap alone |
| 2 | TF-IDF + LinearSVM | text | margin view of the same signal |
| 3 | MiniLM + classical head | minilm | semantic-only |
| 4 | structured + classical head | structured + deterministic + ATS | ResumeForge features only |
| 5 | combined | all approved | reference config |

Tier 0 is mandatory in every comparison. Tiers exist to answer *"does adding a
feature family generalize, or just overfit to the rule?"* — not to chase the
best number.

## 7. Train / validation / test strategy

Three splits, all in `ml/splits.py`, all seeded and deterministic:

1. **Primary — `job_grouped`** (70/10/20, stratified by job_title+seniority).
   Every one of a job's 30 labels stays together. Tests **new jobs** (the
   deployment reality) while sharing the resume vocabulary. This is the
   headline number.
2. **Secondary — `strict_both`** (60/20/20). Held-out jobs AND their positive
   resumes are removed from training entirely (resume-in-partition rule:
   test > val > train). Tests **new jobs + new people**. Smaller/noisier;
   reported separately, never conflated with the primary.
3. **Diagnostic — `resume_grouped`** (70/10/20 by role+seniority). Holds out
   resumes to expose memorisation of pair co-occurrence.

Fixed seeds in config; the final test split is touched exactly once per tier.
`GitHub-style` no: the repo ships no repository split; we create it.

## 8. Metrics

Committed set in `ml/metrics.py` (pure, stdlib-only):

- **Primary:** `macro_f1` (both classes weighted equally — the only number
  used to choose a model).
- Secondary: `accuracy`, `balanced_accuracy`, per-class precision/recall/F1,
  confusion matrix (TP/FP/FN/TN).
- Diagnostic (probability-model only): ROC-AUC, PR-AUC — PR-AUC is the honest
  metric under sampled negatives where positives are a definable minority.

Every report includes the majority-baseline row. `ml/evaluation/contracts.py`
freezes the report shape so tiers are comparable by construction.

## 9. Hyperparameter tuning (leakage-safe, modest)

- Grids in `MODEST_GRID` (`ml/experiments/config.py`) — e.g. `C ∈ {0.1,1,10}`
  for linear models, `tfidf_max_features ∈ {500,2000}`, scaling on/off.
- Protocol: GroupKFold with groups = job, **within the train split only**.
  The test split is never consulted during tuning; the val split is the
  tie-breaker only if multiple configs are equal on CV.
- Seed fixed; any randomness in CV repeats from the config seed.

## 10. Compute plan (RTX 5050 Laptop, 8 GB VRAM / 16 GB RAM)

- MiniLM embeddings: GPU-accelerated (already verified working with CUDA
  torch), generated once per snapshot and cached under
  `ml/cache/` (gitignored). 10k resumes + 2.5k jobs is trivial for MiniLM on
  the 8 GB GPU.
- Classical models (LR / LinearSVM / ridge) on CPU; feature
  matrices 75k×k rows fit comfortably in RAM at the declared grid sizes.
- **No transformer fine-tuning** is planned or assumed; if 6E finds Tier 3/5
  unusable, the fallback is the classical stack, not a parameter sweep.
- Everything stays local: no external APIs, no paid compute.

## 11. Explainability plan

Separate channels, never conflated:

- Deterministic decisions (Phase 5 matching, Phase 6A quality, Phase 6B
  coverage/evidence) remain the *explanable* output of the product; they
  already ship their own evidence fields and disclaimers.
- 6E model outputs are research: a metric row, not a product claim. A model
  prediction will never be labelled as "ATS" or "match" in the UI, and no
  rule-vs-model attribution sentence will be produced from learned weights.
- Ablation canary (`CANARY_FAMILIES` above) quantifies how much of any gain
  walks back to the generation rule.

## 12. Synthetic-data limitations (explicit, recorded)

Verified in 6C and *must* be repeated in the 6E report header:

- Only 3 distinct experience-bullet strings across all 10,000 resumes;
  summaries 100% template. Text diversity is tiny ⇒ text-only models will
  overstate generalization beyond this synthetic universe.
- Identical role/industry/skill vocabularies between resumes and jobs
  (24 roles, 10 industries, 73 skills) ⇒ structured overlap can dominate
  trivially.
- Labels come from one fixed formula, so near-perfect macro-F1 is *expected*
  when overlap features are present — interpret relative to the canary
  ablation, not absolute.
- 29 of 10,000 resumes never appear as positive anywhere; models cannot
  possibly generalise relevance for them; this is a ceiling, not a bug to
  fix.

## 13. Second dataset investigation (optional — outcome: defer)

Searched for a legally usable external resume↔job benchmark. Closest matches
and why they were **not** adopted in 6D:

1. `build-small-hackathon/job-search-distill` (Apache-2.0) — task-identical
   `(resume, job)` fit evaluations (9.86k pairs). Rejected: redistributes
   real resumes *without* PII redaction (violates data-minimisation/privacy
   posture); labels are teacher-AI-generated and the card itself disclaims
   them as a validated rubric (invented labels); jobs derive from LinkedIn
   scraping with downstream-ToS responsibility.
2. `mithinsagar/exai-resumeintel-data` — 2,484 real resumes + 1M job
   postings, but **no pair relevance labels**; using it would require
   inventing the target.
3. `AzharAli05/Resume-Screening-Dataset` (MIT, 10.2k rows, has a decision
   label) — screening-style resume label, not a job-specific relevance pair;
   no documented link between JD content and the decision.

Verdict: no second dataset is wired in. Revisit only if 6E primary results
are dominated by the generative rule and a synthetic spike (new distribution,
documented, PII-free) becomes necessary. Any future adoption must clear the
same three gates this phase applies to the primary dataset: legal license,
no PII, no invented labels.

## 14. Phase 6E training contract (interfaces shipped here)

| Concern | Interface |
|---|---|
| Dataset | `ml/loader.py` → `DatasetBundle` (pinned snapshot) |
| Target / pairs | `ml/data/pairs.py` → `build_pairs(...)` |
| Splits | `ml/splits.py` → `job_group_split`, `resume_group_split`, `strict_both_split` → `SplitManifest` |
| Preprocessing | `ml/preprocessing/text.py`, `structured.py` (pure) |
| Feature transformers | `ml/features/builders.py` → `StatelessTransformer` / `TrainableTransformer` / `FittedTransformer` (fit-on-train) |
| Trainer | `ml/models/trainer.py` → `Trainer(Protocol)`, `train(plan)` (stub, raises `NotImplementedError`) |
| Evaluator | `ml/evaluation/contracts.py` → `Evaluator(Protocol)` → `EvaluationReport` |
| Experiment config | `ml/experiments/config.py` → `ExperimentConfig` (+ fingerprint), `TIERS`, `MODEST_GRID` |

`train(plan)` intentionally raises until 6E implements it; a test asserts this
so nobody mistakes 6D for a trained system.

## 15. Reproducibility & versioning

Every experiment is pinned by `ExperimentConfig`:

- dataset id + revision (fixed), `seed`, `negatives_per_job`,
  `split_strategy`, `split_sizes`, `tiers`.
- version fields: `preprocessing_version`, `feature_version`,
  `split_version`, `experiment_version` — bump the relevant one when its
  definition changes; the 16-hex `fingerprint()` then changes and old runs
  stop matching.
- Snapshot cached in the local HF hub; no network needed for the primary
  dataset after first fetch; tests never download.

## 16. Output artifact plan

- Working files live under `backend/ml/artifacts/` (audit reports, plots), and
  6E model artifacts will land there or under `ml/cache/`.
- `.gitignore` already covers `backend/ml/artifacts/`, `ml/data/`, `ml/cache/`,
  `*.parquet`, `*.npy`, and (new) `*.joblib`/`*.pkl`/`*.pt`/`*.bin` — model
  binaries never enter the repository.
- No user data ever enters the pipeline; the synthetic dataset is retained
  only in the git-ignored cache/artifacts.

## 17. Testing strategy (this phase)

Offline, no downloads, small synthetic fixtures:

- `tests/test_ml_data_pairs.py` — target definition, per-job negative
  sampling, determinism, allowlist, NaN/None robustness.
- `tests/test_ml_splits.py` — job-group integrity (a job's labels never
  split), both-group disjointness, stratum coverage, determinism.
- `tests/test_ml_experiment_design.py` — registry exclusions (identity,
  raw_must_have, generative_rule, provided_embeddings; `ats_6b` sub-field
  exclusions), quasi-generative canary tagging, preprocessing
  purity/technical tokens/NFKC, metric arithmetic, config fingerprint
  determinism, trainer stub raises, evaluator report validation.

This phase's contract tests are the calibration for 6E: if they stay green,
6E can implement against these signatures without re-designing the data
layer.

## 18. Scope guardrails (what this phase is NOT)

No training, no fine-tuning, no hyperparameter optimisation, no model
binaries, no FastAPI integration of ML, no frontend ML, no removal or
replacement of the deterministic Phase-5 matching or Phase-6A/6B ATS, no new
monetary-cost dependencies. `scikit-learn` was added only to the `ml`
optional-dependency set as a forward declaration for 6E and is imported
nowhere in 6D.

## 19. Files

New in 6D (backend):
- `ml/data/__init__.py`, `ml/data/pairs.py`
- `ml/metrics.py`
- `ml/splits.py`
- `ml/experiments/config.py`
- `ml/features/registry.py`, `ml/features/builders.py`
- `ml/preprocessing/text.py`, `ml/preprocessing/structured.py`
- `ml/models/trainer.py`
- `ml/evaluation/contracts.py`

Tests (new): `tests/test_ml_data_pairs.py`, `tests/test_ml_splits.py`,
`tests/test_ml_experiment_design.py`.

Modified: `pyproject.toml` (`ml` extra += scikit-learn), root `.gitignore`
(model artifact patterns). Updated: `ml/README.md`, this document.

## 20. Final report (Phase 6D, completed)

- **Task / target:** binary relevance per `(job, resume)` pair;
  positives = published `relevant_resume_ids`, negatives = seeded per-job
  sample (1:1 default, "not published" semantics). See §1–§2.
- **Features:** 8 approved families, 4 excluded (identity, raw_must_have,
  generative_rule, provided_embeddings) with leakage classes; `ats_6b`
  required/overall coverage excluded as quasi-generative. Registry is the
  enforced single source of truth. §3.
- **Splits (implemented, tested):** primary `job_grouped` 70/10/20
  stratified; secondary `strict_both` 60/20/20 (jobs + resumes disjoint);
  diagnostic `resume_grouped`. §7.
- **Metrics:** primary macro-F1; secondary accuracy / balanced-accuracy /
  per-class P/R/F1 / confusion; diagnostic ROC-AUC / PR-AUC §8.
- **Tuning:** leaked-safe GroupKFold (group = job) on the train split only,
  `MODEST_GRID` sizes, val as tie-breaker §9.
- **Reproducibility:** `ExperimentConfig` pins dataset+seed+negatives+split
  and preprocessing/feature/split/experiment versions; 16-hex fingerprint
  distinguishes runs §15.
- **Compute:** RTX 5050 (8 GB VRAM) MiniLM embeddings on GPU, classical
  models on CPU, no fine-tuning assumed §10.
- **Synthetic limitations:** recorded verbatim (§12); only 3 distinct bullet
  strings, 100% template summaries, identical role/industry/skill vocab.
- **Second dataset:** investigated; closest candidates rejected (PII,
  invented/teacher labels, no pair labels, scraping ToS); none wired in §13.
- **6E contracts:** loader → pairs → splits → preprocessing → feature
  transformers → trainer → evaluator; `train()` is an intentional
  `NotImplementedError` stub §14.
- **Files created:** `ml/data/{__init__,pairs}.py`, `ml/metrics.py`,
  `ml/splits.py`, `ml/experiments/config.py`, `ml/features/{registry,
  builders}.py`, `ml/preprocessing/{text,structured}.py`,
  `ml/models/trainer.py`, `ml/evaluation/contracts.py`;
  tests `tests/test_ml_data_pairs.py`, `test_ml_splits.py`,
  `test_ml_experiment_design.py`.
- **Modified:** `backend/pyproject.toml` (declared `scikit-learn>=1.4` in the
  `ml` extra only — imported nowhere in 6D), root `.gitignore` (model
  artifact extensions), `backend/ml/README.md`.
- **Quality gates:** full backend pytest **789 passed, 2 skipped** (was 738
  + 51 new); ruff **All checks passed**; mypy strict on `app ml`
  **Success** (97 source files, +11 new). Phase 5 / 6A / 6B / 6C suites all
  green within the full run.
- **Environment note:** no git repository exists at `D:\ResumeForge`, so a
  `git diff` change-inspection gate is impossible here; this report's file
  list is the change record.