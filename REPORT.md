# Reproducible KION feature-search study

## Executive status

This repository contains a **methodology-validity study**, not evidence of production lift. The canonical KION result is `results/kion_50k_protocol_v4.json`: 50,000 selected users, 285,343 retained interactions, three seeds, and two expanding temporal folds. Its protocol-v4 candidate correction is authoritative; earlier 50k JSONs are provenance only (see Section 5). The run's stderr records 1:58:19 elapsed and 2,509,416 KiB peak RSS.

On the canonical result, `auto_gain` differs from `manual` by +0.00137 test AUC and +0.21927 sampled NDCG@10 averaged over six fold/seed cells. Decayed popularity remains stronger on sampled NDCG@10 (0.52138). These are offline, protocol-conditional observations—not causal, impression-level, or production-lift claims.

## 1. Research question and estimands

**Question.** Can auditable, template-generated temporal aggregate features match or improve a manually designed LightGBM feature set?

**Primary comparison.** `auto_gain` versus `manual`; the validation-selected winner is not the primary endpoint.

**KION event-level estimand.** ROC AUC for classifying high-watch versus lower-watch **observed user-item records** in each later chronological test fold. The reported summary is the unweighted mean of six fold/seed AUCs (three seeds × two folds); its SD is descriptive dispersion across those six cells, not a confidence interval.

**Target semantics.** The label is `watched_pct >= 50.0` (inclusive threshold, percentage units). Rows are observed, pair-aggregated watch records with day-level `last_watch_dt`, not impressions or a fully ordered event stream. Values below 50 and missing `watched_pct` (mapped to 0 by the loader) are label 0; missing is not a neutral or unobserved label. No-interaction user-item pairs are absent from the event-level classification population. Completion may be known only after the watch, so this is not a claim that the target was available at the query time.

**KION ranking estimand.** Query-weighted macro mean NDCG@10 over held-out positive high-watch records that pass historical-availability filtering, each ranked against 99 sampled negatives. Per-fold/seed user-cluster bootstrap intervals preserve each user's query count; the JSON reports separate intervals for each of the six cells, not a pooled interval over folds/seeds.

**Uncertainty artifact boundary.** `results/kion_50k_protocol_v4.json` was generated before the event-level AUC bootstrap was added. It contains `ranking_ndcg10_uncertainty` for sampled NDCG@10, but it does **not** contain `event_auc_uncertainty`; event-level AUC bootstrap uncertainty was not recomputed for this checked-in canonical run. New rolling outputs produced by the current code may contain `event_auc_uncertainty` (user-cluster bootstrap for pooled event-level AUC), but those are new outputs and must not be attributed to the canonical JSON.

**Candidate protocol.** Negatives are sampled uniformly without replacement from the 8,385-item selected-cohort catalog, restricted to items first observed strictly before the query timestamp and excluding items previously observed for that user. Future catalog items are excluded. A positive whose item was not historically available is skipped rather than ranked against an inconsistent candidate universe. The cap of 400 applies to attempted positive queries before historical-availability eligibility filtering; retained query counts are therefore lower. Each retained query has one held-out positive and 99 negatives (mean actual negatives 99.0). This synthetic candidate set is a sampled-discrimination diagnostic—not an impression log, retrieval evaluation, or production candidate set.

## 2. Validity contract

- Equal timestamps are atomic: features at time `t` use only rows with timestamp `< t`.
- Test rows are excluded from feature sources and causal popularity references.
- Candidate rows are scored from the same pre-test snapshot for every method.
- Feature selection and model configuration use train/select; validation is used for early stopping; test is reported once.
- Ranking point metrics are macro means over queries. User-cluster bootstrap intervals are per fold/seed and preserve query counts; do not interpret them as a pooled cross-seed/fold interval.
- User sampling retains every row for selected users, but does not establish user/item cold-start performance.

### Known threats

KION has no impression log, no exposure propensity, and only day-level timestamps. `watched_pct` may represent final pair-level completion rather than feedback available at a decision time; missing values are labeled 0. The candidate catalog is cohort-restricted and sampled, and positive queries with historically unavailable items are excluded. Popularity dominates the checked-in sampled ranking result, so candidate construction is a major determinant of the score. These limitations prevent causal, next-event, exposure-ranking, or online claims.

## 3. Methods and baselines

- `manual`: hand-designed user/item/pair/category temporal aggregates + LightGBM.
- `auto_gain`: auditable template enumeration followed by training-only LightGBM gain top-K selection.
- `auto_forward`: validation forward selection over the same template space.
- `decay`: manual features augmented with exponential-decay aggregates.
- `linear`: regularized logistic regression over manual features.
- References: cumulative popularity, decayed popularity, category popularity, and prior user-item positive-watch rate.

The search is not unconstrained AI-generated SQL. The restricted, inspectable search space is intentional: it makes every feature traceable and leakage-auditable.

## 4. Hypotheses and status

These are declared follow-up hypotheses, not preregistered hypotheses: they were formulated after development exploration.

- **H1:** `auto_gain` improves test AUC over `manual` by at least 0.01. **Status: not supported.** The mean paired fold/seed difference is +0.00137; the sample SD of the six paired differences is 0.01723. This is a descriptive summary, not an AUC confidence interval.
- **H2:** `auto_gain` improves sampled NDCG@10 over `manual` by at least 0.02. **Status: inconclusive as a confirmatory claim; an exploratory pattern is observed.** Canonical six-cell means are 0.34345 versus 0.12418 (difference +0.21927). All six fold/seed point differences are positive (range +0.16120 to +0.27985); each cell's paired user-cluster 95% CI for the difference is above +0.02 (envelope of lower bounds +0.12440 to +0.24242; upper bounds +0.19903 to +0.31959). The hypothesis was formulated after development exploration, and these are per-cell intervals—not a pooled or confirmatory cross-seed/fold analysis. Decayed popularity (0.52138) and cumulative popularity (0.49785) both exceed `auto_gain`.
- **H3:** decay features are more robust under temporal drift. **Status: not supported by this run.** Across the six fold/seed cells, the mean decay AUC (0.64959) and mean sampled NDCG@10 (0.11346) are below the corresponding manual means (0.65271 and 0.12418). This small two-fold study does not establish a general robustness property.
- **H4:** future rows do not affect prefix features. **Status: supported for the implemented invariant**, based on atomic timestamp/snapshot tests. This is not exposure causality.

No method is recommended for deployment.

## 5. Checked-in evidence

| Artifact | Status | Scope |
|---|---|---|
| `results/kion_50k_protocol_v4.json` | **canonical current evidence** | 50k selected users, three seeds, two folds; corrected historical-catalog candidates |
| `results/kion_50k_corrected.json` | **superseded; provenance only** | predates protocol-v4 candidate-availability correction; do not use for current claims |
| `results/kion_10k_corrected.json` | protocol validation only | 10k selected users, one seed, two folds; not the canonical estimate |
| `results/kion_50k_legacy_pre_correction.json` | historical only | predates candidate, semantic, and uncertainty corrections |
| `historical-results.md` | historical synthetic smoke only | not KION and not comparable to protocol v4 |

The following point summaries are recomputed from the six fold/seed cells in the canonical JSON. Parentheses show the sample SD across those six cells (descriptive dispersion, not a confidence interval); paired AUC delta SD is also descriptive.

| Metric | Manual | `auto_gain` | Reference |
|---|---:|---:|---:|
| Test ROC AUC | 0.65271 (0.01935) | 0.65408 (0.00412) | Paired difference: +0.00137 (SD 0.01723) |
| Sampled NDCG@10 | 0.12418 (0.00906) | 0.34345 (0.04473) | Cumulative popularity: 0.49785 (0.01787); decayed popularity: 0.52138 (0.01405) |

The canonical JSON records 2,187 fold×seed ranking queries (345–378 per cell) and per-cell skipped-positive counts (22–55; 213 summed over the six evaluation cells, not unique queries). It includes package versions, command, source hashes, source-tree manifest/hash, thread settings, protocol and split metadata, per-query metrics, and per-cell user-cluster intervals for query-weighted NDCG@10. It does not contain `event_auc_uncertainty`, and no event-level AUC bootstrap was recomputed for this artifact. Its immutable provenance snapshot is `metadata.source_tree.sha256 = c77188b69f3d65af97879d48dcecc1b5554fb9b780cbabdb6f374a865903fe13` and code revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c`.

## 6. Reproduction

```bash
cd recsys_corrected_artifact
.venv/bin/python -m pytest -q

.venv/bin/python compare_methods.py \
  --kion-interactions datasets/kion/interactions.csv \
  --kion-users datasets/kion/users.csv \
  --kion-items datasets/kion/items.csv \
  --kion-max-users 50000 \
  --evaluation rolling --folds 2 --seeds 42,43,44 \
  --forward-max-candidates 15 --forward-steps 3 --json \
  > results/kion_50k_reproduced.json
```

For a stronger follow-up, freeze the protocol first and run at least 50k/all users, three training seeds, at least 1,000 ranking queries per fold, and retain JSON, stderr, cost, input hashes, and environment metadata together.

## 7. Required follow-up experiment

Before claiming improvement, run a locked experiment with:

1. primary endpoint `auto_gain - manual` on temporal test AUC and query-weighted NDCG@10;
2. user-cluster CIs for AUC and ranking, plus seed dispersion;
3. candidate sensitivity: 10/50/99/499 negatives, uniform/popularity-weighted/recent sampling, cohort/full catalog;
4. target sensitivity at 25/50/75% watched completion;
5. user/item cold-start and popularity-decile slices;
6. prevalence, history-length, query-count, and test-window coverage;
7. retrieval recall, coverage, novelty, diversity, and latency;
8. an end-to-end fixed fixture with expected split and metric tolerances.

For logged-exposure claims, use MIND `official_train_dev` or another dataset with impressions. For causal claims, define exposure, treatment, outcome, time zero, propensity support, and an online/randomized bridge; KION alone cannot identify those quantities.

## 8. Evidence ledger

| Claim | Evidence | Applicability limit |
|---|---|---|
| KION contains users, items, day-level watch completion and metadata | KION repository and dataset paper cited in the project README | Public schema does not establish exposure or within-day ordering |
| Prefix features exclude future rows | `causal_features.py`, `history.py`, and causal/snapshot tests | Establishes an implementation invariant, not causal impact |
| Candidate metrics use historical unseen negatives | `candidate_eval.py` and candidate protocol metadata | Not a logged-impression ranking estimate |
| Popularity is a strong reference | Checked-in corrected development result | Local observation only; not universal |

## Final recommendation

Present this as a careful, transparent methodology study tied to the protocol-v4 JSON. The ranking result shows exploratory separation of `auto_gain` from `manual` on this particular sampled candidate set, while popularity references remain stronger. Do not generalize the sampled result to exposure ranking, causal impact, or production lift; the study does not evaluate logged impressions or a live retrieval system.
