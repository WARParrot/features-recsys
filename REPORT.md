# Reproducible KION feature-search study

## Executive status

### Latest completed evidence: 10k factorial follow-up

The current primary research evidence is the completed **10,000-user protocol-v6 factorial follow-up**, not the historical v4/v5 runs: 55,922 watch records, three seeds, three temporal folds, four variants, exact matched split/candidate fingerprints. All 12 cells (36 fold evaluations) passed validation. [Detailed report](audits/v6_10k_results.md), [result](results/kion_10k_v6_factorial.json), [lock](results/kion_10k_v6_factorial.lock.json).

| History | Grammar | Manual AUC | Auto gain AUC | Delta AUC | Manual sampled NDCG@10 | Auto sampled NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| Prequential | Base | .66936 | .66161 | -.00775 | .20991 | .26553 |
| Prequential | Expanded | .66936 | .68087 | +.01150 | .20991 | .26736 |
| Snapshot | Base | .66736 | .66029 | -.00707 | .20979 | .27413 |
| Snapshot | Expanded | .66736 | .68194 | +.01458 | .20979 | .32869 |

Each entry is the unweighted mean of nine fold×seed summaries, not a pooled estimate or CI. Decayed popularity sampled NDCG is .48542. Expanded-snapshot AUC deltas by temporal fold (mean over seeds) are +.03596 / +.00062 / +.00717: only the three fold0 paired conditional AUC CIs exclude zero. All nine sampled-NDCG delta CIs exceed +.02, but popularity is still stronger. The run supports a local expanded-feature ranking pattern, **not time-robust AUC superiority or deployment**. Whole-run wall time is 6:14:10, peak RSS ~1.07 GiB.

### Post-run validity correction (protocol v5)

The completed v5 run (`results/kion_50k_protocol_v5.json`, exit 0, 4:19:22, peak RSS 3,341,024 KiB) is preserved as **exploratory diagnostic evidence only**. Mean AUC was .62428 manual versus .65527 auto_gain, but this does not establish superiority: independent audit found an empty training-history prefix, all 187 history predictors constant zero in first-fold train, all six auto selection AUCs .5, and tied zero-gain feature selection. Refit on train+select introduced populated history only afterward. Bootstrap intervals cannot repair this deterministic design defect. See [audit](audits/v5_validity.md).

Current protocol v6 reserves a history-only warmup and constructs training examples at historical block snapshots. A locked 2×2 factorial design separates history protocol from expanded feature grammar on matched splits and candidates; see [experiment plan](EXPERIMENT_PLAN_V6.md). This is an exploratory follow-up on inspected periods, not a fresh confirmatory holdout. Test remains frozen over the full role whereas training/selection use block cadence: freshness horizon mismatch remains visible and must not be called online equivalence.

This repository contains a **methodology-validity study**, not evidence of production lift. The canonical historical protocol-v4 KION result is `results/kion_50k_protocol_v4.json`: 50,000 selected users, 285,343 retained interactions, three seeds, and two expanding temporal folds. Its protocol-v4 candidate correction is authoritative; earlier 50k JSONs are provenance only (see Section 5). The run's stderr records 1:58:19 elapsed and 2,509,416 KiB peak RSS.

On the canonical result, `auto_gain` differs from `manual` by +0.00137 test AUC and +0.21927 sampled NDCG@10 averaged over six fold/seed cells. Decayed popularity remains stronger on sampled NDCG@10 (0.52138). These are offline, protocol-conditional observations—not causal, impression-level, or production-lift claims.

### Completed protocol-v6 pilot

The locked 1,000-user factorial pilot completed all four cells with exact split/candidate matching. Mean manual versus auto-gain AUC: prequential/base .64282/.62401; prequential/expanded .64282/.61728; snapshot/base .62717/.61183; snapshot/expanded .62717/.62542. No cell supports mean auto superiority. Expanded snapshot sampled NDCG .23360 versus manual .22834 remains below decayed popularity .45345. See [pilot report](audits/v6_pilot_results.md) and [JSON](results/kion_1k_v6_factorial.json) for scope, failures retained, costs, and hypothesis decisions. This pilot does not establish larger-population or online conclusions.

## 1. Research question and estimands

**Question.** Can auditable, template-generated temporal aggregate features match or improve a manually designed LightGBM feature set?

**Primary comparison.** `auto_gain` versus `manual`; the validation-selected winner is not the primary endpoint.

**KION event-level estimand.** ROC AUC for classifying high-watch versus lower-watch **observed user-item records** in each later chronological test fold. The latest 10k factorial summary is the unweighted mean of nine fold/seed AUCs per variant (three seeds × three folds); historical v4 uses six. These are not pooled estimates; SD across repeated fold/seed cells is descriptive, not a confidence interval.

**Target semantics.** The label is `watched_pct >= 50.0` (inclusive threshold, percentage units). Rows are observed, pair-aggregated watch records with day-level `last_watch_dt`, not impressions or a fully ordered event stream. Values below 50 and missing `watched_pct` (mapped to 0 by the loader) are label 0; missing is not a neutral or unobserved label. No-interaction user-item pairs are absent from the event-level classification population. Completion may be known only after the watch, so this is not a claim that the target was available at the query time.

**KION ranking estimand.** Query-weighted macro mean NDCG@10 over held-out positive high-watch records that pass historical-availability filtering, each ranked against 99 sampled negatives. Per-fold/seed user-cluster bootstrap intervals preserve each user's query count; the latest factorial JSON reports separate intervals for each fold/seed within each variant (nine per variant), not a pooled interval over folds/seeds. Historical v4 has six cells.

**Uncertainty artifact boundary.** `results/kion_50k_protocol_v4.json` was generated before the event-level AUC bootstrap was added. It contains `ranking_ndcg10_uncertainty` for sampled NDCG@10, but it does **not** contain `event_auc_uncertainty`; event-level AUC bootstrap uncertainty was not recomputed for this checked-in canonical run. New rolling outputs produced by the current code may contain `event_auc_uncertainty` (user-cluster bootstrap for pooled event-level AUC), but those are new outputs and must not be attributed to the canonical JSON.

**Candidate protocol.** Negatives are sampled uniformly without replacement from the selected-cohort catalog (5,090 items in the latest 10k run; 8,385 in historical 50k v4), restricted to items first observed strictly before the query timestamp and excluding items previously observed for that user. Future catalog items are excluded. A positive whose item was not historically available is skipped rather than ranked against an inconsistent candidate universe. The cap of 400 applies to attempted positive queries before historical-availability eligibility filtering; retained query counts are therefore lower. Each retained query has one held-out positive and 99 negatives (mean actual negatives 99.0). This synthetic candidate set is a sampled-discrimination diagnostic—not an impression log, retrieval evaluation, or production candidate set.

## 2. Validity contract

Historical rolling protocol (version 5) supports `history_mode=role_frozen` (now known to degenerate training without warmup): for each train/select/valid/test role, feature sources are the strict timestamp prefix before the role start and remain frozen within that role. The legacy prequential behavior remains available explicitly with `--history-mode prequential`. Candidate sensitivity is configurable through `--candidate-n-negatives`, `--candidate-max-queries`, and `--candidate-sampling` (`uniform`, `popularity`, `recent`); `--candidate-catalog full` requires a validated static catalog with `item`, `item_cat`, and finite `first_seen`. These settings define new evaluations and are not retroactively applied to the checked-in protocol-v4 JSON.

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
- `auto_forward`: selection-interval forward search using a regularized linear probe over a bounded candidate budget; the separate `valid` role is reserved for LightGBM early stopping.
- `decay`: manual features augmented with exponential-decay aggregates.
- `linear`: regularized logistic regression over manual features.
- References: cumulative popularity, decayed popularity, category popularity, and prior user-item positive-watch rate.

The search is not unconstrained AI-generated SQL. The restricted, inspectable search space is intentional: it makes every feature traceable and leakage-auditable.

## 4. Hypotheses and status

### Updated decisions for the completed 10k factorial follow-up

- **H1, AUC gain >= .01 robust across time: inconclusive.** Expanded mean gains are +.01150 (prequential) and +.01458 (snapshot), but snapshot support is concentrated in fold0; six later-period conditional CIs include zero. Base means are negative. No aggregate/time-general uncertainty interval is available.
- **H2, sampled NDCG gain >= .02: locally supported for snapshot-expanded, exploratory only.** Mean gain +.11890; all nine per-cell conditional CI lower bounds exceed +.02. Decayed popularity .48542 still beats auto .32869; no logged-exposure claim follows.
- **Expanded grammar versus base: descriptively improved in both history modes on 10k.** Auto AUC differences +.01925 / +.02165. No direct expanded-minus-base paired CI or component ablation was computed.
- **H3, decay robustness: not established by this factorial design.** It does not isolate a decay component or establish temporal robustness.
- **H4, strict-prefix safety: supported as a tested implementation invariant**, not proof that KION final aggregate feedback was available at recommendation time.

The 1k pilot had negative mean auto-minus-manual AUC in all variants; the 10k result is not a universal success or a controlled causal sample-size comparison. See [full interpretation and costs](audits/v6_10k_results.md).

### Historical v4 hypothesis decisions (not updated or pooled)

These are declared follow-up hypotheses, not preregistered hypotheses: they were formulated after development exploration.

- **H1:** `auto_gain` improves test AUC over `manual` by at least 0.01. **Status: not supported.** The mean paired fold/seed difference is +0.00137; the sample SD of the six paired differences is 0.01723. This is a descriptive summary, not an AUC confidence interval.
- **H2:** `auto_gain` improves sampled NDCG@10 over `manual` by at least 0.02. **Status: inconclusive as a confirmatory claim; an exploratory pattern is observed.** Canonical six-cell means are 0.34345 versus 0.12418 (difference +0.21927). All six fold/seed point differences are positive (range +0.16120 to +0.27985); each cell's paired user-cluster 95% CI for the difference is above +0.02 (envelope of lower bounds +0.12440 to +0.24242; upper bounds +0.19903 to +0.31959). The hypothesis was formulated after development exploration, and these are per-cell intervals—not a pooled or confirmatory cross-seed/fold analysis. Decayed popularity (0.52138) and cumulative popularity (0.49785) both exceed `auto_gain`.
- **H3:** decay features are more robust under temporal drift. **Status: not supported by this run.** Across the six fold/seed cells, the mean decay AUC (0.64959) and mean sampled NDCG@10 (0.11346) are below the corresponding manual means (0.65271 and 0.12418). This small two-fold study does not establish a general robustness property.
- **H4:** future rows do not affect prefix features. **Status: supported for the implemented invariant**, based on atomic timestamp/snapshot tests. This is not exposure causality.

No method is recommended for deployment.

## 5. Checked-in evidence

| Artifact | Status | Scope |
|---|---|---|
| `results/kion_10k_v6_factorial.json` | **latest factorial follow-up** | 10k users, 3 seeds × 3 folds × 4 variants; completed, exact fingerprints and conditional AUC/ranking CIs |
| `results/kion_1k_v6_factorial.json` | exploratory pilot | 1k users, one seed, two folds; not generalizable or pooled with 10k |
| `results/kion_50k_protocol_v5.json` | diagnostic only | degenerate training-history selection; see audit |
| `results/kion_50k_protocol_v4.json` | **canonical historical v4 evidence** | 50k selected users, three seeds, two folds; corrected historical-catalog candidates |
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

### Latest 10k factorial follow-up

Restore source revision `388e48848099f352d05067e556fff882015a3c2e` in a separate checkout, install the recorded Python 3.11 dependency environment, and verify input hashes in the historical lock. On the original paths/runtime, `run_ablation.py --execute-lock results/kion_10k_v6_factorial.lock.json --output <new-path>` validates the source/runtime lock. On another host/path create a new lock, preserving all scientific settings below. Export the thread settings for **both** lock and execution. Do not overwrite historical artifacts.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python run_ablation.py --dry-run --lock results/10k_rerun.lock.json \
  --kion-interactions datasets/kion/interactions.csv \
  --kion-users datasets/kion/users.csv --kion-items datasets/kion/items.csv \
  --max-users 10000 --cohort-seed 42 --seeds 42,43,44 --folds 3 \
  --warmup-fraction .2 --training-blocks 4 --watch-threshold 50 \
  --forward-max-candidates 15 --forward-steps 3 \
  --candidate-n-negatives 99 --candidate-max-queries 400 \
  --candidate-sampling uniform --smoke-users 1000 --max-wall-seconds 172800
.venv/bin/python run_ablation.py --execute-lock results/10k_rerun.lock.json \
  --output results/10k_rerun.json
```

### Optional 50k current-code extension (not the completed 10k run)

The command below is a **new current-code evaluation**, neither a replay of v4 nor the factorial result. `prequential` with the same positive warmup is the alternative comparator; legacy boundaries require explicit zero warmup and degenerate gain selection now fails rather than returning arbitrary features. The KION adapter validates required IDs/dates separately from allowed missing feedback; source manifests exclude runtime/build outputs.

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

## 7. Remaining follow-up experiments

The 10k factorial follow-up now supplies paired AUC/ranking intervals, three temporal folds and seed repetitions. It does not close all research requirements. Before broader claims, complete:

1. fresh-period or second-dataset replication of `auto_gain - manual`, retaining AUC and sampled ranking as separate estimands;
2. temporal mechanism analysis, direct expanded-minus-base paired uncertainty and transform-family ablations; conditional AUC/ranking CIs already exist but do not capture retraining or temporal shift;
3. candidate sensitivity: 10/50/99/499 negatives, uniform/popularity-weighted/recent sampling, cohort/full catalog;
4. target sensitivity at 25/50/75% watched completion;
5. user/item cold-start and popularity-decile slices;
6. prevalence, history-length, query-count, and test-window coverage;
7. retrieval recall, coverage, novelty, diversity, and latency;
8. extend the existing end-to-end KION/snapshot fixtures to calibrated metric-tolerance fixtures where useful (basic pipeline and split invariants are already tested).

For logged-exposure claims, use MIND `official_train_dev` or another dataset with impressions. For causal claims, define exposure, treatment, outcome, time zero, propensity support, and an online/randomized bridge; KION alone cannot identify those quantities.

## 8. Evidence ledger

| Claim | Evidence | Applicability limit |
|---|---|---|
| KION local adapter/schema contains users, items, day-level watch completion and metadata | `kion_data.py`; local CSV schema/hashes; [upstream dataset repository](https://github.com/irsafilo/KION_DATASET) | Upstream paper not independently verified here; schema does not establish exposure or within-day ordering |
| Prefix features exclude future rows | `causal_features.py`, `history.py`, and causal/snapshot tests | Establishes an implementation invariant, not causal impact |
| Candidate metrics use historical unseen negatives | `candidate_eval.py` and candidate protocol metadata | Not a logged-impression ranking estimate |
| Popularity is a strong reference | `results/kion_10k_v6_factorial.json`: decayed popularity mean NDCG .48542 | Local sampled-discrimination observation only; not universal |

## Final recommendation

Present this as a transparent methodology study led by the completed 10k protocol-v6 factorial evidence; retain v4 as historical and v5 as a documented failure. Expanded features improve mean AUC locally, but the benefit is dominated by one period and is not time-robustly established. Snapshot-expanded improves sampled NDCG versus manual across all nine conditional intervals, while popularity remains stronger. The 1k pilot's different outcome and the remaining horizon/data limits prevent a universal superiority claim. Do not generalize the sampled result to exposure ranking, causal impact, or production lift; the study does not evaluate logged impressions or a live retrieval system.
