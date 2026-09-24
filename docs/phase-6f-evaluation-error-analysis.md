# Phase 6F — Evaluation & Error Analysis Report

Date: 2026-09-15. Analysis of the Phase 6E trained matching models: ablation
study, feature attributions, error concentration, robustness, leakage/sanity
checks, calibration, and split generalization. All numbers come from executed
run artifacts in `backend/ml/artifacts/phase_6f/` and the authoritative 6E
artifacts in `backend/ml/artifacts/phase_6e/` — nothing was hand-derived after
the fact.

---

## 1. Executive summary

Phase 6F asked: **what actually drives the ~0.95 macro-F1**, and **where do the
models fail**. Answers, all supported by the sections below:

| Question | Answer | Section |
|---|---|---|
| What drives Tier 4/5? | The quasi-generative coverage features carry essentially all the signal | §5, §6, §7 |
| Can the model be "shown up" for shortcut learning? | Yes, quantitatively: dropping the 8 quasi-generative features collapses Tier 4 to 0.6089 (still above text-only 0.56), dropping the 12 rule-derived candidates collapses it to 0.5009 (chance) | §5 |
| Are the plain structured overlaps real signal? | No — remove all 4 "suspicious" features → no change (0.9526 vs 0.9525) | §5 |
| Are the models robust? | Yes, but trivially: 4 text transforms produced 0 flips and 0.0 feature deltas because the features are lexical-exact, not text-derived | §13 |
| Where are errors? | Tier 4/5: strictly false positives (1,330 / 1,314 of 28,080), concentrated on pairs with high cosine but mismatched role/industry/seniority | §9 |
| Is the model useful without the rule family? | No. Tier 4 without quasi-generative features ≈ 0.61; this benchmark has no learnable signal beyond the coverage rule | §5 |
| Does combining text/semantic help? | Marginal +0.0004; harms cold-start (strict_both 0.9523 → 0.8657) | §4, §8 |

**Bottom line:** Tier 4/5 are not "95% matchers". They are coverage-rule
reproducers over deterministic match features. No current trained model is
suitable as the real-world matching signal; the deterministic Phase-5A pipeline
remains the defensible production path.

---

## 2. Scope & method

- Consumes the 18 Tier 0–5 models already trained and serialized in Phase 6E
  (config fingerprint `99d6d7b41ec502a9`, seed 42, dataset revision
  `178ab864dcad9910c5670d43e4bdbbb901a11f18`).
- **No 6E model was retrained.** The only new training this phase is the 8
  controlled Tier-4 ablation experiments (`6f-ablation-*`), each retrained on
  the frozen job_grouped train split and evaluated once on the frozen
  job_grouped test split (28,080 pairs).
- Analyses computed by `backend/ml/phase_6f/*` and orchestrated by
  `backend/ml/scripts/run_phase_6f.py`, which also re-executed five 6E models
  on job_grouped test to reproduce their recorded metrics (exact match — §14).
- All probing of mini-negative/resume-pair content is **derived/anonymized**:
  only role/seniority/experience/industry attribute strings from the public
  synthetic fields are used. No raw resume or JD text is stored in artifacts.

---

## 3. Feature families and the "quasi-generative" canary family

Registry-enforced families (design in `phase-6d-ml-experiment-design.md`):

| Family | Content | In tiers |
|---|---|---|
| `resume_text`, `job_text` | TF-IDF vocabulary | 1, 2, 5 |
| `minilm` | MiniLM cosine (overall + per-section) | 3, 5 |
| `structured` | role/seniority/experience/industry/education overlaps | 4, 5 |
| `deterministic_match` | Phase-5A keyword overlap / hybrid score (**quasi-generative**) | 4, 5 |
| `ats_6a` | general ATS heuristics (all coefficients 0 in analysis) | 4, 5 |
| `ats_6b` | job-specific coverage/evidence (**quasi-generative**, excludes `required_coverage`/`overall_coverage`) | 4, 5 |

Canary naming used in this report:

- **quasi-generative (8):** `keyword_overlap_required`, `keyword_overlap_preferred`,
  `keyword_overlap_combined`, `text_token_overlap`, `hybrid_score` (5 from
  `deterministic_match`) + `preferred_coverage`, `evidence_supported_required`,
  `evidence_supported_preferred` (3 from `ats_6b`).
- **suspicious structured (4):** `same_role`, `same_industry`, `same_seniority`,
  `years_experience`.

---

## 4. Ablation study (Tier 4, job_grouped test)

All 8 ablations retrain the Tier-4 model (structured features + classical head,
default config) with feature subsets excluded; the full model (A) is the 6E
Tier 4 baseline (22 features). Metrics from `experiment_summary.json` /
`ablation_results.csv`.

| Exp | Removed | n | macro-F1 | Acc |
|---|---|---|---|---|
| A (baseline) | none | 22 | **0.9525** | 0.9526 |
| B | suspicious structured (4) | 18 | **0.9526** | 0.9527 |
| C | suspicious (4) + quasi-generative (8) | 10 | **0.5009** | 0.5009 |
| D-same_role | `same_role` only | 21 | 0.9523 | 0.9524 |
| D-same_industry | `same_industry` only | 21 | 0.9526 | 0.9527 |
| D-same_seniority | `same_seniority` only | 21 | 0.9525 | 0.9526 |
| D-years_experience | `years_experience` only | 21 | 0.9526 | 0.9527 |
| E | quasi-generative only (8) | 14 | **0.6089** | 0.647 |

Reading:

- **Removing the 8 quasi-generative features (E) collapses macro-F1 from 0.9525
  → 0.6089.** What survives (0.61) is only slightly better than the TF-IDF
  text-only tiers (~0.56) and well below the MiniLM tier (~0.72): the *only*
  learnable signal in this benchmark is the coverage rule family.
- **Removing all 12 rule-derived candidates (C, leaving only the 10 non-rule
  structured/ATS features) → 0.5009, i.e. chance.** The non-rule features carry
  no label information.
- **Removing the 4 supposedly-suspicious structured features one-at-a-time
  (D-*) or together (B) changes nothing within noise** (0.9523–0.9526). They
  are inert, not shortcuts — the shortcut is the deterministic family.
- Conclusion: Tier 4's accuracy is **fully attributable to the quasi-generative
  features**. This confirms the Phase 6E hypothesis quantitatively.

---

## 5. Feature importances — Tier 4 (regression coefficients, standardized)

Source: `feature_importance.json` (logistic-regression style coefficients on
standardized features; abs = |coefficient|).

| Feature | coef | abs |
|---|---|---|
| `keyword_overlap_required` | +1.7876 | 1.7876 |
| `keyword_overlap_combined` | +1.3618 | 1.3618 |
| `hybrid_score` | +1.1970 | 1.1970 |
| `keyword_overlap_preferred` | −0.3169 | 0.3169 |
| `preferred_coverage` | −0.3169 | 0.3169 |
| `text_token_overlap` | +0.1228 | 0.1228 |
| `same_industry` | −0.0596 | 0.0596 |
| `same_role` | −0.0349 | 0.0349 |
| `evidence_supported_preferred` | +0.0323 | 0.0323 |
| `years_experience` | −0.0252 | 0.0252 |
| `evidence_supported_required` | +0.0168 | 0.0168 |
| `years_experience_bucket` | +0.0139 | 0.0139 |
| degree one-hots (5) | −0.0069 … +0.0032 | ≤0.0069 |
| `same_seniority` | −0.0006 | 0.0006 |
| `degree_phd`, `degree_diploma`, `action_verb_count`, `quantitative_count`, `section_completeness` | 0.0 | 0.0 |

Family sums (sum |coef|): `deterministic_match` **4.786** (5 features),
`ats_6b` 0.366 (3), `structured` 0.148 (11), `ats_6a` 0.0 (3).

The top three features — all in the deterministic family — are the same three
that the Phase-5A composite score is built from. `ats_6a` contributes nothing.

---

## 6. Feature importances — Tier 5 (combined model)

Source: `feature_importance_tier5.json`. 1,267 features. Top of the ranking:

| Feature | abs coef |
|---|---|
| `keyword_overlap_required` | 1.7796 |
| `keyword_overlap_combined` | 1.3434 |
| `hybrid_score` | 1.2007 |
| `minilm_cosine` | 0.3226 |
| `keyword_overlap_preferred` | 0.2973 |
| `preferred_coverage` | 0.2973 |
| `same_role` | 0.2337 |
| `same_industry` | 0.2251 |
| `text_token_overlap` | 0.2068 |
| top TF-IDF token (`resume__prospecting closing`, job side?) | 0.0554 |

Family sums: `deterministic_match` 4.828, `minilm` 0.323 (1 feature),
`structured` 0.530, `ats_6b` 0.321, `ats_6a` 0.0, text/`unknown` 13.46 across
1,244 tokens (mean 0.0108 — collectively small, individually negligible).

Tier 5 adds 1,245 text/semantic features yet the three deterministic features
still out-score the whole vocabulary sum. This is why Tier 5 ≈ Tier 4 on
job_grouped and **worse** on strict_both (§8).

---

## 7. Deterministic Phase-5A baseline vs the trained models

Source: `deterministic_baseline_analysis.json` (computed on the same 28,080
job_grouped test pairs).

| Quantity | Value |
|---|---|
| Phase-5A score ↔ label correlation | **0.8596** |
| Phase-5A score ↔ `keyword_overlap_required` | **0.9702** |
| Phase-5A score ↔ `hybrid_score` | **0.9981** |
| `keyword_overlap_required` ↔ label | 0.8721 |
| Positive score mean / std | 73.81 / 15.03 |
| Negative score mean / std | 9.53 / 22.45 |
| `overlap_required` ≥ 0.6 among positives | **1.0000** |
| `overlap_required` ≥ 0.6 among negatives | 0.0843 |

The Phase-5A deterministic score is essentially the `hybrid_score` feature
(r = 0.998) which is itself essentially the required-coverage rule (r = 0.97).
Positive-pair coverage ≥ 0.6 is **100%** — a direct restatement of the label
generation rule (coverage ≥ 0.6) from Phase 6D/6E. This is the mechanism behind
the classifiers' 95%.

---

## 8. Split generalization (from 6E authoritative artifacts)

Source: 6E `comparison_table.csv`, summarized across the three splits. Macro-F1
on each test split:

| Tier | job_grouped | strict_both | resume_grouped |
|---|---|---|---|
| 0 majority | 0.3333 | 0.3298 | 0.3329 |
| 1 TF-IDF LR | 0.5601 | 0.5278 | 0.5547 |
| 2 TF-IDF SVM | 0.5625 | 0.5210 | 0.5542 |
| 3 MiniLM | 0.7190 | 0.7203 | 0.7174 |
| 4 structured | 0.9525 | 0.9523 | 0.9511 |
| 5 combined | 0.9529 | **0.8657** | 0.9507 |

Tier 4 is stable across every split — expected, because its signal is the
identity-independent coverage rule. Tier 5 degrades sharply only on strict_both
(the cold-start arrangement: train 6,173 pairs / test 120,351, both identities
held out), consistent with Phase 6E §12: high-capacity text/semantic features
do not supply identity-independent signal, while the deterministic family does.

---

## 9. Error concentration (job_grouped test, 28,080 pairs)

Source: `error_summary.json`.

| Tier | TP | FP | FN | TN | macro-F1 |
|---|---|---|---|---|---|
| 3 MiniLM | 9,490 | 3,326 | 4,550 | 10,714 | 0.7190 |
| 4 structured | 14,040 | 1,330 | **0** | 12,710 | 0.9525 |
| 5 combined | 14,034 | 1,314 | 6 | 12,726 | 0.9529 |

Tiers 4/5 are near-ceiling on the rule: FN ≈ 0. Their error is **entirely false
positives** — negatives that do not satisfy the coverage rule yet receive a
match label. Given the sampled-negative semantics (negatives are "not published
as relevant", not "certified irrelevant"), a fraction of these are genuine
rule-satisfying pairs that happened not to be published — see Phase 6E §7.

---

## 10. False-positive attribute analysis (Tier 4)

Source: `error_summary.json`, Tier-4 FP bucket (1,330 pairs). Shared-attribute
fractions among FPs (heuristics for report structuring — not causal claims):

| Derived hypothesis | FP count | FP shown positive-share |
|---|---|---|
| `industry_mismatch` (resume industry ≠ job industry) | 1,201 | 90% |
| `seniority_mismatch` (resume seniority ≠ job seniority) | 893 | 67% |
| `role_mismatch` (role ≠ job title) | 875 | 66% |
| high cosine but label negative (`high_cosine_but_wrong`) | 1,257 | 95% |

The most common structure of a Tier-4 false positive is: *the coverage features
(or their proxies) fired, while role/industry/seniority say the pair is not a
fit.* Because matching never requires the role/industry to align, the classifier
has learned a decision rule (coverage ≥ threshold) that is *blind* to the
role/industry mismatch signals. The FP's are overwhelmingly pairs where the
resume shares skill vocabulary with the JD but the professional profile differs.

Tier-4 FN bucket is empty (recall = 1.0) and Tier-5 FN = 6 — Tier 5 corrects
only a handful of FP vs Tier 4 (1,314 vs 1,330) at the cost of 6 FN.

---

## 11. Class / role analysis

Source: `class_role_analysis.json` (test accuracy per `job_title`, per tier).
Mean accuracy across the 24 job titles:

| Tier | mean accuracy |
|---|---|
| 0 | 0.5000 |
| 1 | 0.5584 |
| 2 | 0.5612 |
| 3 | 0.7206 |
| 4 | 0.9526 |
| 5 | 0.9530 |

Tier-4/5 per-title accuracy is uniform (0.939–0.964) — no title is a systematic
failure pocket; the balanced split and coverage rule produce this uniformity.
Tier 3 variance is wider (0.597–0.838), confirming semantic similarity is
title-sensitive while the coverage rule is not.

---

## 12. Model agreement / divergence

Source: `model_comparison.json` (28,080 job_grouped pairs).

| Comparison | same | diff |
|---|---|---|
| Tier 3 vs Tier 4 | 20,648 | 7,432 |
| Tier 3 vs Tier 5 | 20,658 | 7,422 |
| Tier 4 vs Tier 5 | 28,002 | 78 |

Divergence where one is right and the other wrong:

| Case | n |
|---|---|
| Tier 3 right, Tier 4 wrong | 443 |
| Tier 3 right, Tier 5 wrong | 433 |
| Tier 4 right, Tier 3 wrong | 6,989 |
| Tier 4 right, Tier 5 wrong | 34 |
| Tier 5 right, Tier 3 wrong | 6,989 |
| Tier 5 right, Tier 4 wrong | 44 |

Tiers 4 and 5 disagree on only 78/28,080 pairs (0.28%). Tier 5 has a tiny
independent advantage (44 vs 34) — the text/semantic heads add almost nothing
over the deterministic family on job_grouped.

---

## 13. Robustness / transform invariance

Source: `robustness_results.json` (500 pair sample, Tier-4 model).

| Transform | flipped | flip_fraction | mean_abs_feature_delta |
|---|---|---|---|
| whitespace/case normalization | 0 | 0.0 | 0.0 |
| dedupe skills | 0 | 0.0 | 0.0 |
| reorder bullets | 0 | 0.0 | 0.0 |
| remove boilerplate phrases | 0 | 0.0 | 0.0 |

All flips = 0 and **feature deltas = 0.0 exactly**: the Tier-4 features are
computed from the normalized/structured skill lists and coverage counts, so the
applied presentational transforms do not touch the feature values at all.
Meaning: invariance here is an artifact of **feature construction**, not a
property of the classifier. This is a diagnostic, not evidence of generalization
(these transforms would not be free on real parsing noise).

---

## 14. Sanity / leakage checks (all pass)

Source: `sanity_checks.json` (assertions stop the run on failure; all passed).

| Check | Detail |
|---|---|
| no forbidden columns | 22 features, none of identity/generative/raw-embedding |
| no identity features | none found |
| ablation exclusions respected (×7) | none of the excluded names present in fitted matrices |
| row budgets | 150,000 pairs → train 106,920 / val 15,000 / test 28,080 |
| split disjointness (jobs, resumes) | no overlap across train/val/test for either identity (3 strict/resume checks) |
| vectorizer fit on train only | vocab 1,244, train tokens 314, test-exclusive tokens 0 |
| metric reproduction (6 tiers) | recorded vs reproduced macro-F1/acc/b-acc — **all diffs 0.0** |

The metric-reproduction checks re-evaluated the 6E tiers 0–5 on the frozen
job_grouped test with the harness and matched the recorded numbers exactly
(e.g. Tier 4 macro-F1 0.952528830234487 both times), so every new metric below
is measured on the identical evaluation path as 6E.

---

## 15. Calibration of probabilistic tiers

Source: `calibration_analysis.json` (job_grouped test, `predict_proba` for
tiers 3/4/5, 10 bins).

| Tier | Expected Calibration Error (ECE) |
|---|---|
| 3 | 0.0450 |
| 4 | 0.0453 |
| 5 | 0.0424 |

Tier-4 reliability bins (predicted mean → observed positive rate):

| Bin | n | pred mean | observed |
|---|---|---|---|
| 0.00–0.10 | 12,196 | 0.009 | 0.000 |
| 0.10–0.20 | 263 | 0.139 | 0.000 |
| 0.20–0.30 | 151 | 0.246 | 0.000 |
| 0.30–0.40 | 55 | 0.318 | 0.000 |
| 0.40–0.50 | 45 | 0.464 | 0.000 |
| 0.50–0.60 | 134 | 0.547 | 0.000 |
| 0.60–0.70 | 159 | 0.684 | 0.881 |
| 0.70–0.80 | 2,654 | 0.760 | 0.913 |
| 0.80–0.90 | 3,317 | 0.859 | 0.921 |
| 0.90–1.00 | 9,106 | 0.961 | 0.925 |

The mass sits at the two extremes and the curve is essentially a step at ~0.6 —
again a direct trace of the coverage threshold. **Diagnostic only:** with
balanced class weights, these scores are not probabilities of a real-world
match and must not be treated as such.

---

## 16. Limitations (candid)

1. Every headline number is a property of the **synthetic benchmark and its
   coverage-rule labels**, not of real resume matching.
2. Ablations re-train only Tier 4; Tier 5 and Tier 3 were not ablated. Tier-5
   behavior is inferred from feature importances + 6E splits.
3. Error "patterns" in §10 are derived-shared-attribute counts used to organize
   the report — they describe what the pairs look like, not a causal mechanism.
4. Robustness study used a 500-pair sample of formatted text transforms; real
   parsing noise (layout, OCR, synonyms) is not covered.
5. Calibration is diagnostic; no threshold or score is proposed for production.
6. `ats_6a` features have zero coefficients in both Tier 4 and Tier 5 — they
   are dead features on this benchmark, but that is a statement about *this*
   dataset, not about ATS heuristics in general.

---

## 17. Artifacts

`backend/ml/artifacts/phase_6f/`:

| Artifact | Contents |
|---|---|
| `ablation_results.csv`, `ablation_metrics.json` | 8 ablation experiments, comma / long-form |
| `experiment_summary.json` | Ablation rows + config fingerprint |
| `feature_importance.json`, `feature_importance_tier5.json` | Coefficients, rankings, family sums |
| `deterministic_baseline_analysis.json` | Phase-5A correlations + label distributions |
| `error_summary.json` | Per-tier confusion, per-bucket derived attributes & counts |
| `model_comparison.json` | Pairwise agreement/divergence |
| `class_role_analysis.json` | Accuracy per job title per tier |
| `robustness_results.json` | Transform-invariance probe |
| `sanity_checks.json` | All leakage/leakage-style assertions |
| `calibration_analysis.json` | Reliability curves + ECE, tiers 3/4/5 |
| `split_generalization.csv` | 6E metrics aggregated by tier × split |
| `confusion_tier{3,4,5}_job_grouped.json` | Confusion matrices |

No raw resume/JD text is stored. Re-run: `cd backend && python -m
ml.scripts.run_phase_6f`.

---

## 18. Recommendations

1. **Keep deterministic Phase-5A matching as the production path.** On this
   benchmark it already equals the trained structured model (0.9468 vs 0.9525)
   with zero learning and is fully explainable. The trained models add ~0.5
   points of benchmark-specific, quasi-generative signal — not worth the
   opacity, and not (yet) a fair estimate of real-world performance.
2. **Do not integrate Tier 4/5 into the product at this stage.** They are
   coverage-rule reproducers; shipping them would ship "must-have skill
   coverage ≥ 0.6" dressed as ML. Any real-data label source (parser output on
   genuinely varied resumes) would change the picture.
3. **If ML is ever re-attempted:** drop `ats_6a` (zero coefficients), drop or
   separately model `same_*`/`years_experience` (inert but deceptive), and
   probe Tier-5 strict_both collapse (§8) by ablating text vs minilm, as
   deferred in Phase 6E §16.
4. **Before any of that:** obtain evaluated real, varied resumes. No amount of
   analysis of this synthetic snapshot can certify external validity.

---

## 19. Bottom line

- Tier 4/5 accuracy is **fully explained by the quasi-generative coverage
  family**: removing it drops Tier 4 to 0.6089; removing it plus the inert
  structured overlaps drops it to chance (0.5009). The plain structured
  features are not shortcuts — they are dead weight.
- The deterministic Phase-5A score is the same coverage rule (r ≈ 0.998 with
  the `hybrid_score` feature, 100% of positives at ≥0.6) — the models learned
  the generation rule, exactly as the Phase 6E design warned.
- Errors are exclusively (Tier 4) or almost exclusively (Tier 5) false
  positives on pairs that share skill vocabulary but mismatch on
  role/industry/seniority — a known consequence of a coverage-only label.
- No current model is suitable for real-world matching. This phase's value is
  the **negative result**: on this benchmark, coverage-based matching is a
  solved, deterministic problem, and the ML component has been honestly
  quantified as redundant with it.