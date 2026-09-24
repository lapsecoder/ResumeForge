# Phase 6E — ML Training Report

Date: 2026-09-15. This is the final report for Phase 6E: the supervised ML
training pipeline for candidate–job matching, executed on this machine against
the pinned synthetic dataset snapshot, plus a deterministic Phase-5A baseline
evaluation against the same ground truth.

All numbers below come from the run artifacts in
`backend/ml/artifacts/phase_6e/` (JSON/CSV) — no metric was hand-derived after
the fact. The artifact directory is the source of truth.

---

## 1. Executive summary

We trained 6 experiment tiers (0–5) under 3 split strategies
(18 experiments total) on the pinned candidate-matching dataset. Primary
metric is **macro-F1** (positive and negative classes weighted equally).

| Takeaway | Result |
|---|---|
| Chance floor (majority) | macro-F1 ≈ 0.33 on every split |
| Text-only (TF-IDF, LR/SVM) | macro-F1 ≈ 0.56 — vocabulary overlap is weak |
| Semantic-only (MiniLM) | macro-F1 ≈ 0.72 |
| Structured + deterministic + ATS | macro-F1 ≈ 0.95 on all three splits |
| Combined (text+semantic+structured) | ≈ 0.95 on job/resume grouped, **drops to 0.87 on strict both-group split** |
| Deterministic Phase-5A baseline (same ground truth) | macro-F1 ≈ 0.95 on job_grouped test |

The headline is **not** "we built a 95% matcher". It is:

- The structured + deterministic + ATS configuration reproduces the
  label-generation rule (must-have skill coverage ≥ 0.6) because the
  deterministic match features and structured overlaps encode exactly that
  signal (see §10 — shortcut-learning analysis).
- The deterministic Phase-5A matching pipeline — running *with no learning at
  all* — already reaches ≈ 0.95 macro-F1 on the same job_grouped test set.
  The trained structured model adds little over the deterministic matcher on
  this benchmark (0.9525 vs 0.9468).
- MiniLM semantic similarity alone lands at ≈ 0.72, well below the coverage
  rule, confirming that **word/embedding semantics ≠ the synthetic label
  rule**. This number is a property of *this* benchmark, not of resume
  matching in the real world.
- The strict both-group split is the honest cold-start test: combined dropped
  to **0.8657**, while the pure structured model held at 0.9523 — the text and
  semantic signals do not survive cold-start as well as the structured rule.

---

## 2. Dataset

| Property | Value |
|---|---|
| Dataset id | `michaelozon/candidate-matching-synthetic` |
| Revision (pinned) | `178ab864dcad9910c5670d43e4bdbbb901a11f18` |
| Resumes | 10,000 (9,971 ever labeled; 29 never appear) |
| Jobs | 2,500 |
| Match rows | 2,500 rows × exactly 30 relevant resumes = **75,000 positives** |
| Negatives (sampled) | **75,000** (30 non-relevant / job, seeded) |
| Total pairs | **150,000** |

Generation rule (verified in 6C): a resume is published as relevant when its
must-have skill coverage ≥ 0.6; the generator then random-samples (seed 42) up
to 30 candidates. The 30-cap discards ~96.5% of rule-candidates, so this is a
**capped positive sample**, not an exhaustive relevance set. Negative pairs are
sampled as "not published as relevant", not "certified irrelevant" — some
sampled negatives genuinely satisfy the rule.

---

## 3. Training environment

| Component | Version / detail |
|---|---|
| Python | 3.13.15 |
| scikit-learn | 1.9.0 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| sentence-transformers | 6.0.1 |
| torch | 2.8.0+cu128 |
| joblib | 1.6.0 |
| FastAPI | 0.141.1 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU (CUDA available; semantic device `cuda:0`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, dim 384, batch 128 |
| CPU fallback | semantic embedder falls back to CPU with a warning if CUDA is unavailable |

No paid API was used. Embeddings are fully local; the LLM is not involved in
any experiment — the harness is deterministic-first.

---

## 4. Feature families

Approved families consumed by the trained tiers (registry-enforced, see
`backend/ml/features/registry.py`; design in `phase-6d-ml-experiment-design.md`):

| fid | Content | Tier(s) |
|---|---|---|
| `resume_text` | summary, experience_bullets, skills-as-text → TF-IDF | 1, 2, 5 |
| `job_text` | description, responsibilities, requirements-as-text → TF-IDF | 1, 2, 5 |
| `minilm` | local MiniLM cosine (overall + per-section) | 3, 5 |
| `structured_resume` | role, seniority, years_experience, industry, education (bucketed overlap) | 4, 5 |
| `structured_job` | job_title, seniority, industry | 4, 5 |
| `deterministic_match` | Phase-5A keyword overlap / semantic cosine / hybrid — recomputed, **quasi-generative** | 4, 5 |
| `ats_6a` | general ATS heuristics (grammar/impact/style) | 4, 5 |
| `ats_6b` | job-specific coverage/evidence, **excluding** `required_coverage`/`overall_coverage` | 4, 5 |

Excluded from *all* trained models (enforced structurally):

- `identity` (`resume_id`, `job_id`), `raw_must_have`, `generative_rule`,
  `provided_embeddings`.
- Within `ats_6b`: the coverage fields that nearly reproduce the rule.

The `deterministic_match` and `ats_6b` families are held apart as
quasi-generative canaries: they are allowed so we can measure how much of the
"skill" is rule reproduction.

---

## 5. Split strategy

Derived in `ml/splits.py`, identical configuration for every tier
(seed 42, sizes 0.7/0.1/0.2). `job_grouped` is the primary split (used for
threshold derivation in §9); the other two are robustness checks that control
for cross-split leakage.

| Split | Train | Val | Test | Test jobs |
|---|---|---|---|---|
| `job_grouped` | 106,920 | 15,000 | 28,080 | 468 |
| `strict_both` | 6,173 | 23,476 | 120,351 | 2,500 |
| `resume_grouped` | 105,313 | 15,174 | 29,513 | 2,500 |

Note `strict_both`: both resume and job identities are train-held-out, and the
test partition is the bulk of the data (120,351 pairs) — the honest
cold-start arrangement. Training set shrinks to 6,173 pairs, which directly
explains the combined-model drop in §8/§11.

---

## 6. Experiments — full table (all 18)

Test-set metrics from `comparison_table.csv`. Acc = accuracy,
B-Acc = balanced accuracy, P/R/F1 = per-class, MF1 = macro-F1.

### job_grouped

| Tier | Model | Features | Acc | MF1 | B-Acc | Pos P/R/F1 | Neg P/R/F1 |
|---|---|---|---|---|---|---|---|
| 0 | majority | — | 0.5000 | 0.3333 | 0.5000 | 0/0/0 | 0.5/1.0/0.6667 |
| 1 | TF-IDF + LR | text | 0.5603 | 0.5601 | 0.5603 | 0.558/0.580/0.569 | 0.563/0.540/0.551 |
| 2 | TF-IDF + linear SVM | text | 0.5627 | 0.5625 | 0.5627 | 0.560/0.585/0.572 | 0.566/0.541/0.553 |
| 3 | MiniLM + head | semantic | 0.7195 | 0.7190 | 0.7195 | 0.741/0.676/0.707 | 0.702/0.763/0.731 |
| 4 | structured + head | struct+det+ats | 0.9526 | 0.9525 | 0.9526 | 0.914/1.0/0.955 | 1.0/0.905/0.950 |
| 5 | combined + head | all | 0.9530 | 0.9529 | 0.9530 | 0.914/1.0/0.955 | 1.0/0.906/0.951 |

### strict_both

| Tier | Model | Features | Acc | MF1 | B-Acc | Pos P/R/F1 | Neg P/R/F1 |
|---|---|---|---|---|---|---|---|
| 0 | majority | — | 0.4921 | 0.3298 | 0.5000 | 0/0/0 | 0.492/1.0/0.660 |
| 1 | TF-IDF + LR | text | 0.5293 | 0.5278 | 0.5285 | 0.534/0.576/0.554 | 0.524/0.481/0.502 |
| 2 | TF-IDF + linear SVM | text | 0.5232 | 0.5210 | 0.5223 | 0.528/0.582/0.554 | 0.517/0.463/0.489 |
| 3 | MiniLM + head | semantic | 0.7207 | 0.7203 | 0.7215 | 0.751/0.674/0.710 | 0.696/0.769/0.730 |
| 4 | structured + head | struct+det+ats | 0.9525 | 0.9523 | 0.9518 | 0.915/1.0/0.955 | 1.0/0.904/0.949 |
| 5 | combined + head | all | 0.8658 | 0.8657 | 0.8657 | 0.868/0.868/0.868 | 0.863/0.864/0.864 |

### resume_grouped

| Tier | Model | Features | Acc | MF1 | B-Acc | Pos P/R/F1 | Neg P/R/F1 |
|---|---|---|---|---|---|---|---|
| 0 | majority | — | 0.4991 | 0.3329 | 0.5000 | 0.499/1.0/0.666 | 0/0/0 |
| 1 | TF-IDF + LR | text | 0.5547 | 0.5547 | 0.5547 | 0.554/0.557/0.555 | 0.556/0.553/0.554 |
| 2 | TF-IDF + linear SVM | text | 0.5542 | 0.5542 | 0.5542 | 0.554/0.550/0.552 | 0.555/0.558/0.556 |
| 3 | MiniLM + head | semantic | 0.7179 | 0.7174 | 0.7178 | 0.736/0.678/0.706 | 0.703/0.757/0.729 |
| 4 | structured + head | struct+det+ats | 0.9512 | 0.9511 | 0.9513 | 0.911/1.0/0.953 | 1.0/0.903/0.949 |
| 5 | combined + head | all | 0.9508 | 0.9507 | 0.9509 | 0.911/0.999/0.953 | 0.999/0.903/0.948 |

---

## 7. Confusion matrices (job_grouped test, 14,040 pos / 14,040 neg)

| Tier | TP | FP | FN | TN |
|---|---|---|---|---|
| 0 majority | 0 | 0 | 14,040 | 14,040 |
| 1 TF-IDF + LR | 8,149 | 6,456 | 5,891 | 7,584 |
| 2 TF-IDF + SVM | 8,207 | 6,445 | 5,833 | 7,595 |
| 3 MiniLM | 9,490 | 3,326 | 4,550 | 10,714 |
| 4 structured | 14,040 | 1,330 | 0 | 12,710 |
| 5 combined | 14,034 | 1,314 | 6 | 12,726 |
| Deterministic Phase-5A | 14,040 | 1,491 | 0 | 12,549 |

The structured tiers and the deterministic baseline capture **100% of
positives** (FN=0). Their error is exclusively false positives — exactly what
the sampled-negative semantics predicts: negatives that satisfy the coverage
rule but were simply not published.

---

## 8. Model comparison

- **Tier 0** defines the floor: 0.33 macro-F1 on a 50/50 split everywhere.
- **Tiers 1–2**: TF-IDF text alone ≈ 0.56. Vocabulary overlap carries little
  signal; momentum exists but is marginal. LR and SVM are nearly
  interchangeable (0.5601 vs 0.5625).
- **Tier 3 (MiniLM)**: ≈ 0.72 consistently. The fastest gain per family and
  the most stable across splits. It has the best negative-class recall
  (≈0.76) of any tier — semantic similarity backstops "this is not a match"
  better than exact-vocabulary methods.
- **Tiers 4–5**: ≈ 0.95. Tasked for the split the model saturates positive
  recall; residual error is false positives. Tier 5 rarely beats Tier 4 on
  job/resume grouped (0.9529 vs 0.9525) and is **worse** on strict_both
  (0.8657 vs 0.9523) — extra signal is not free.
- **Deterministic Phase-5A** (no learning): 0.9468 — within ~0.6 points of the
  best trained model on job_grouped, with the same FN=0 profile.

Clear conclusion: **on this dataset, the coverage rule is the signal, and
every competitive configuration already contains it** — either via the
deterministic features or via the adjacent structured fields.

---

## 9. Deterministic Phase-5A baseline (Part 2 of scope)

To interpret the ML numbers honestly, the Phase-5A deterministic matcher
(`app/matching/service.match_resume_to_job`) was evaluated against the **same
ground truth** — the 6D/6E pairs and splits — with **no fabrication** of
missing fields.

### Adapter (honest mapping only)

| Parse field | Dataset source |
|---|---|
| `Resume.summary` | `resume.summary` |
| `Resume.skills.all` | `resume.skills` |
| `JobDescription.title` | `job_title` |
| `JobDescription.summary` | `description` |
| `JobDescription.responsibilities` | `responsibilities` |
| `JobDescription.required_skills` | `must_have_skills` |
| `JobDescription.preferred_skills` | `nice_to_have_skills` |
| metadata | word_count computed, file_type `txt`, confidence HIGH |

Explicitly **not** mapped (field does not exist in the dataset → left absent,
documented rather than fabricated): resume `experience` (no dates/companies),
resume `education` (no institution), job `experience_requirements`,
`education_requirements`, `certifications`, and generic `requirements` prose
(no structured quantification). Qualified effects:

- Experience matcher → not evaluated (needs date ranges).
- Education matcher → not evaluated (needs recognized degree labels).
- Qualification matcher → not evaluated (needs certifications).

The Phase-5A `overall_score` therefore reduces to the skill-coverage composite:
`required_skill (50) + preferred_skill (15)` of the 50/15/20/10/5 weights,
with absent weights redistributed. This is reported in
`deterministic_baseline_metrics.json`.

### Threshold

Phase-5A has no documented score→label threshold (grader output is a 0–100
score, not a classifier). We therefore **derived a threshold on the
job_grouped TRAIN split only** (grid-search over 0.1 steps maximizing
macro-F1), **froze it**, then evaluated exactly once on the job_grouped TEST
split — no test tuning. Frozen threshold: **42.4**.

### Results (job_grouped test, 14,040 pos / 14,040 neg)

| Metric | Value |
|---|---|
| Accuracy | 0.9469 |
| Macro-F1 | 0.9468 |
| Balanced accuracy | 0.9469 |
| Positive P / R / F1 | 0.9040 / 1.0 / 0.9496 |
| Negative P / R / F1 | 1.0 / 0.8938 / 0.9439 |
| Confusion | TP 14,040 · FP 1,491 · FN 0 · TN 12,549 |
| None-score fraction (train/test) | 0.0 / 0.0 |

The deterministic matcher reaches **0.9468 macro-F1 on the test job_grouped
split with zero learning**. FN=0 mirrors the trained structured tier exactly:
it maxes positive recall; every error is a false positive that satisfied the
coverage rule but was not published.

### Baseline comparison table (same split, same pairs)

| Baseline/Model | Score type | Test macro-F1 | Test accuracy |
|---|---|---|---|
| Majority baseline | supervised (chance floor) | 0.3333 | 0.5000 |
| Phase-5A deterministic | deterministic score → threshold | **0.9468** | **0.9469** |
| Tier 1 TF-IDF + LR | supervised ML | 0.5601 | 0.5603 |
| Tier 2 TF-IDF + SVM | supervised ML | 0.5625 | 0.5627 |
| Tier 3 MiniLM | supervised ML | 0.7190 | 0.7195 |
| Tier 4 structured | supervised ML | 0.9525 | 0.9526 |
| Tier 5 combined | supervised ML | 0.9529 | 0.9530 |

**The deterministic score and the supervised ML classifications are different
kinds of objects.** The deterministic number is a hardcoded heuristic
operating directly on must-have/nice-to-have skill coverage — effectively the
rule input. It is reported so the ML numbers can be judged against the
simplest possible comparator, not as a "production matcher" claim. Neither
row is evidence about real-world hiring accuracy.

---

## 10. Shortcut-learning analysis (REQUIRED section)

**Why Tier 4 is ~95% and yet suspicious — and why it is exactly what we expect
on this data.**

1. **The label is literally coverage ≥ 0.6.** The generator publishes a resume
   as relevant iff teacher must-have coverage ≥ 0.6. The `deterministic_match`
   and `ats_6b` features recompute the *same kind* of coverage from the text.
   `minilm` and text features see the same content through different lenses.
2. **Skills tokens are in plain sight.** `resume.skills` (approved family) is
   a token-level list of exactly the strings the generator used. A model that
   learns "skills present ↔ must-have satisfied" has simply re-learned the
   rule. Tier 4/5 positive recall = 1.0 and FN=0 is the fingerprint: no
   positive pair is ever missed because every positive literally satisfies the
   coverage rule.
3. **The deterministic baseline says it best: 0.9468 with no learning.** The
   trained structured model's 0.9525 is within 0.6 points of a rule-based
   comparator. The residual is the model learning to also dismiss many
   non-published-but-satisfying negatives.
4. **known shortcut candidates** (must be listed, per 6D):
   - `same_seniority`, `same_industry`, `same_role` structured overlaps —
     proxies for the generator's profile-templating, not hiring signal.
   - TF-IDF vocabulary overlap — similarity = shared template words.
   - Coverage/evidence fields (`ats_6b`) — the closer they are to
     `required_coverage`, the more they *are* the rule. The excluded
     `required_coverage`/`overall_coverage` sub-fields were removed precisely
     so the model cannot key directly on the generator's own coverage value.
5. **What is NOT explained by the rule:** the deterministic matcher and
   structured tier agree, but MiniLM (0.719) does not. Semantic similarity
   does not reconstruct the exact coverage threshold — it encodes meaning, not
   a 0.6 cutoff. That separation is the legitimate finding of this phase: the
   synthetic rule is learnable via lexical coverage and is not the same as
   semantic relevance.

Conclusion: Tier 4 "≈95% on synthetic" means **"the features reproduce the
generation rule"**, not "ResumeForge is 95% accurate at real matching". Any
claim written into the product must use the mini-negative framing: this is a
benchmark property, and on this benchmark the ceiling *is* the rule.

---

## 11. MiniLM interpretation (~72%)

Tier 3 reaches 0.7190 (job_grouped), 0.7203 (strict_both), 0.7174
(resume_grouped) — the most stable single-family signal, yet far below the
coverage rule. Interpretation:

- MiniLM is being asked "is the resume semantically aligned with the job at
  the block level" — the cosine is over a **synthetic, template-generated
  corpus**. The 0.72 plateau sits between vocabulary overlap (0.56) and the
  rule (0.95), so it captures *some* latent structure but not the coverage
  cutoff.
- Its negative-class recall advantage (≈0.76 vs ≈0.54 for TF-IDF) is the
  one trait a real production matcher would want: it is conservative about
  declaring a match.
- **This number must never be quoted as "real-world semantic matching
  quality".** It is a property of this synthetic benchmark and its shared
  vocabulary. It tells us local sentence transformers can *partially*
  reconstruct rule alignment, not that 72% of real resumes would ATS-match.

---

## 12. strict_both interpretation (Tier 5 drop to 0.8657)

The combined model is the only configuration whose strict_both score clearly
degrades (0.9523 → 0.8657, vs Tier 4's stable 0.9523). Claims supported by the
data:

1. **Cold-start / small train set.** strict_both trains on only 6,173 pairs
   while testing on 120,351. The combined model ships 1,267 features (TF-IDF
   + MiniLM + structured) — high capacity, tiny sample. Overfitting to the
   6,173 train pairs is the natural explanation.
2. **Distribution shift.** In strict_both, both the resume and the job in the
   test set were entirely unseen at train time. Text and semantic features
   must generalize to *new* identities/templates; the structured coverage
   features are identity-independent rule features and survive.
3. **Which signal survives?** The pure structured model keeps 0.9523 because
   the coverage rule features are invariant to identity. MiniLM (0.7203)
   keeps its plateau because it never relies on identity. The *combination*
   is the fragile one — the extra text/semantic capacity competes with the
   reliable signal on 6,173 samples.
4. **What we cannot claim:** that Tier 5 "matters less" in production, or
   that 0.8657 is a better estimate of real-world performance. Both statements
   are outside this benchmark's evidence.

---

## 13. Limitations (candid)

1. **Fully synthetic data.** Both resumes and JDs are template-generated; the
   vocabulary is shared between train and test structures (only refined by
   split strategy).
2. **Generated labels, not judged relevance.** Target = published-as-relevant
   from a coverage rule, capped at 30. Not an expert assessment.
3. **Candidate-cap bias.** The 30 cap discards ~96.5% of rule-candidates;
   positives are a *sample* of the rule, so no model can "certify irrelevance".
   FN=0 across structured/deterministic reflects the rule's own logic.
4. **Template summaries** with low bullet diversity → text/vocab features are
   partly measuring template similarity.
5. **Shared vocabulary** inflates literal-overlap baselines and MiniLM alike.
6. **Shortcut risk** documented in §10 — coverage/overlap features can equal
   the rule.
7. **Limited external validity.** Numbers are comparable *within* this
   benchmark only. No claim transfers to real resumes/JDs without a
   real-data evaluation.
8. **Deterministic baseline adaptation drops components.** Experience,
   education, qualification could not be evaluated because the dataset has no
   corresponding structured fields — the honest baseline is skill-coverage
   only.

---

## 14. Artifacts (backend/ml/artifacts/phase_6e/)

| Artifact | Contents |
|---|---|
| `comparison_table.csv` / `.json` | All 18 experiments, one row each (full metric set) |
| `metrics_tier{0..5}_{split}.json` | Per-experiment metrics, val+test, confusion, per-class |
| `model_tier{1..5}_{split}.joblib` | Trained estimators (serializable; Tier-0 majority = 2-byte stub, unpicklable by design, noted in logs) |
| `experiment_summary.json` | Fingerprint, best macro-F1, per-experiment summary |
| `dataset_metadata.json` | Dataset id/revision, frame sizes, pair counts |
| `split_metadata.json` | Exact train/val/test counts per split |
| `experiment_config.json` | Full reproducibility config |
| `semantic_metadata.json` | Embedder, dim, batch, device |
| `deterministic_baseline_metrics.json` | Phase-5A baseline: threshold, all-split metrics, per-class, threshold curve top-10, scoring note |
| `deterministic_baseline_confusion_matrix.json` | Test confusion counts |

No raw resumes or job descriptions are stored in artifacts (privacy/data
minimization).

---

## 15. Reproducibility

| Key | Value |
|---|---|
| Config fingerprint | `99d6d7b41ec502a9` |
| Seed | 42 |
| Dataset revision | `178ab864dcad9910c5670d43e4bdbbb901a11f18` |
| Preprocessing / feature / split / experiment versions | `1.0.0` ×4 |
| Negative sampling | `negatives_per_job = 30`, seeded per job |
| Primary metric | `macro_f1` |
| Split sizes | 0.7 / 0.1 / 0.2 |
| Deterministic threshold | 42.4 (frozen from job_grouped TRAIN only) |

Re-run: `cd backend && python -m ml.scripts.run_phase_6e` for the 18
experiments; `python -m ml.deterministic_baseline.evaluate` for the Phase-5A
baseline. Deterministic baseline asserts the expected job_grouped split sizes
(106,920 / 15,000 / 28,080) as a parity guard.

---

## 16. Phase 6F handoff questions (for the next phase, not resolved here)

1. Why do structured features dominate? Is it `deterministic_match`/`ats_6b`
   specifically, or also the plain `structured_*` overlaps? (Feature-importance
   ablation on Tier 4.)
2. Which individual features are responsible for the 0.95? (Permutation
   importances within Tier 4, and on strict_both.)
3. Why does the combined model survive job_grouped but not strict_both —
   confirm it is train-size-driven (repeat Tier 5 with fewer features on
   strict_both).
4. Exactly where does MiniLM fail vs the rule (which roles/seniority
   buckets)? Error concentration by `job_title`.
5. Where does the *combined* model fail on strict_both — is the degradation
   text, semantic, or both? Ablate: Tier 5 minus `resume_text`/`job_text`,
   Tier 5 minus `minilm`.
6. Separate genuine vs synthetic signal: estimate how much of Tier 4/5 is
   rule reproduction (holding out the deterministic/ATS families), as
   required by the leakage policy.
7. Decide whether any non-rule signal is worth engineering before touching
   real data, given FN=0 everywhere.

---

## 17. Bottom line

- On this benchmark the **label is the rule**, the strongest configuration is
  the one that reproduces it, and the deterministic Phase-5A baseline already
  reaches 0.9468 with **no learning**.
- **Do not** promote "95% match accuracy" as a product claim — it is
  benchmark-confounded by the generation rule.
- The production-worthy takeaway is a structural one: exact-coverage features
  saturate; **semantic signal (MiniLM) is a conservative, identity-independent
  complement**; and **cold-start (strict_both) damages high-capacity
  combined models** — all of which must shape any future real-data experiment
  (Phase 6F scope).