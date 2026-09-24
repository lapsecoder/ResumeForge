# ML workspace (Track B data audit + Track C experiment design)

Track B (Phase 6C) audited the dataset; the audit is complete. Track C
(Phase 6D) fixed the supervised-experiment design and its contracts. Nothing
here trains models, and Phase 6D must not be confused with a trained system:
`ml/models/trainer.py::train` deliberately raises `NotImplementedError`.

Full design: `docs/phase-6d-ml-experiment-design.md`.

## Scope today

### Track B — dataset audit (complete)

| Module | Purpose |
|---|---|
| `loader.py` | Pinned, offline-first reader for the HF dataset raw parquet files |
| `audit/statistics.py` | Aggregate row/skill/seniority/industry statistics |
| `audit/target.py` | Target reconstruction + derivability analysis |
| `audit/leakage.py` | Template/boilerplate/vocab leakage diagnostics |
| `audit/splits.py` | Split-strategy recommendation (audit-time only) |
| `audit/quality.py` | Emptiness/PII-presence/integrity diagnostics (counts only) |
| `audit/report.py` | Markdown + JSON artefacts, EDA plots |
| `scripts/run_dataset_audit.py` | One-shot audit runner |
| `models/baselines.py` | Baseline A–E definitions (contracts only, not run) |

### Track C — experiment design contract (Phase 6D, not run yet)

| Module | Purpose |
|---|---|
| `data/pairs.py` | Target definition: flat (job_id, resume_id, relevant) pairs + seeded negative sampling |
| `splits.py` | Job-grouped, strict both-group, and resume-grouped splits (seeded, deterministic) |
| `metrics.py` | Pure metric definitions; primary = macro-F1 |
| `experiments/config.py` | `ExperimentConfig` + fingerprint, tiers 0–5, modest tuning grid |
| `features/registry.py` | Approved/excluded feature families with leakage classification |
| `features/builders.py` | Phase 6E transformer contracts (stateless / fit-on-train) |
| `preprocessing/text.py` | Deterministic NFKC + technical-token-preserving tokeniser |
| `preprocessing/structured.py` | Seniority/industry/education/role matching helpers |
| `models/trainer.py` | 6E Trainer contract; `train()` stub (raises) |
| `evaluation/contracts.py` | 6E Evaluator contract + committed `EvaluationReport` shape |

## Dataset

`michaelozon/candidate-matching-synthetic` @
`178ab864dcad9910c5670d43e4bdbbb901a11f18`

- 10,000 resumes, 2,500 jobs, 2,500 match rows (30 relevant resumes per job = 75,000 pairs).
- No explicit binary label; labels are the `relevant_resume_ids` lists.
- Generation rule (verified from the notebook): must-have coverage ≥ 60%,
  random sample capped at 30, random seed 42.
- `datasets.load_dataset` only reveals the `resumes` split, and
  `data_files=` mixing hits a `CastError`; reading the raw parquet files
  directly is the supported path.

## Running the audit

From `backend/`:

```bash
.venv\Scripts\python -m ml.scripts.run_dataset_audit
```

Uses the local HF hub cache by default. `--allow-download` fetches a pinned
snapshot if the cache is cold. `--skip-plots` skips matplotlib for CI.

Artefacts land in `ml/artifacts/` (git-ignored).

## Privacy

Aggregate counts and plots only. No resume/job text, no PII, no per-record
rows are emitted. Plots render counts, never content. Model artifacts
(`*.joblib`, `*.pkl`, `*.pt`, `*.bin`) are git-ignored; no user data enters
the pipeline.