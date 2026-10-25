# Reproducible KION feature-search study

## Executive status

### Latest result: corrected 10k factorial run (v7)

v7 is the first complete 10k factorial run after the implementation fixes described below. It uses 10,000 selected users, 55,922 observed watch records, three seeds, three temporal folds, and the same four matched variants as v6. All 12 cells and 36 fold evaluations passed the result, cohort, split, and candidate checks.

| History | Features | Manual AUC | Auto gain AUC | Auto - manual | Manual NDCG@10 | Auto NDCG@10 | Decayed popularity |
|---|---|---:|---:|---:|---:|---:|---:|
| Prequential | Base | .66892 | .66163 | -.00728 | .20500 | .26546 | .48542 |
| Prequential | Expanded | .66892 | .68088 | +.01196 | .20500 | .26776 | .48542 |
| Snapshot | Base | .66629 | .66029 | -.00600 | .19988 | .27413 | .48542 |
| Snapshot | Expanded | .66629 | .68241 | +.01612 | .19988 | .31507 | .48542 |

Each value is the unweighted mean of nine fold/seed results. These are not pooled estimates. For snapshot-expanded, the mean AUC difference by temporal fold is +.03596, +.00063, and +.01176. Conditional intervals exclude zero in 6 of 9 fold-seed results, all in folds 0 and 2, so the improvement is not yet established as stable over time. The mean NDCG@10 difference is +.11519 and is positive in every conditional interval, but the smallest lower bound is only +.01487. The research threshold +.02, formulated after development for this analysis, is therefore not supported in every fold-seed result. Decayed popularity remains clearly stronger on this sampled ranking task (.48542 versus .31507).

This is offline evidence from observed watch records. It is not evidence about logged impressions, causal impact, or production lift.

The v7 result combines nine cells reused from the locked Linux run at revision `647f393` with three missing `snapshot + expanded` cells computed on Windows at revision `30065af`. Both environments report Python 3.11.15, the same package versions, data hashes, seeds, temporal boundaries, candidate protocol, and single-thread settings. The reused cells account for 4:04:20 of recorded cell time; the three Windows cells account for 17:05, or 17:42 including startup and preflight. These are mixed-host timings, not one hardware benchmark; Windows peak memory was not recorded.

### Historical v6 run

v6 is kept as a before-and-after reference. It was produced before the implementation fixes, so its numbers must not be presented as results from the current code.

| History | Features | Manual AUC | Auto gain AUC | Auto - manual | Manual NDCG@10 | Auto NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| Prequential | Base | .66936 | .66161 | -.00775 | .20991 | .26553 |
| Prequential | Expanded | .66936 | .68087 | +.01150 | .20991 | .26736 |
| Snapshot | Base | .66736 | .66029 | -.00707 | .20979 | .27413 |
| Snapshot | Expanded | .66736 | .68194 | +.01458 | .20979 | .32869 |

The v6 values are also unweighted means over nine fold/seed results. Its snapshot-expanded gain was concentrated in the earliest period, and decayed popularity still led the sampled NDCG comparison. See the [v6 report](audits/v6_10k_results.md) for the original analysis.


### Why there are several result versions

The older v5 run is retained as a diagnostic failure. Its first training fold had an empty history prefix, so all 187 history features were constant and the automatic selector had no meaningful signal to use. Its attractive-looking AUC difference is therefore not evidence of a model improvement; see the [v5 audit](audits/v5_validity.md).

v6 introduced a history-only warmup and timestamp-block snapshots, then compared history protocols and feature grammars on matched splits and candidates (see the [experiment plan](EXPERIMENT_PLAN_V6.md)). It is useful historical evidence, but it predates the implementation fixes used for v7. The test role remains frozen over its full horizon while training and selection use block cadence, so neither v6 nor v7 should be described as an online evaluation.

This repository is a methodology study, not a production-lift claim. The historical v4 KION result is retained for provenance and comparison; it should not be pooled with v6 or v7 because the protocols and artifacts differ.

### What changed between v6 and v7

Two implementation issues found after v6 were fixed before the v7 run:

- LightGBM early stopping now uses AUC as the only stopping metric.
- Forward selection removes train-constant candidates before applying its candidate budget.

The selector also validates `max_candidates`, and each new fold result records the effective candidate pool and its hashes. v7 is the first full 10k run using these fixes. Its `auto_gain` models use the corrected stopping policy; its `auto_forward` and descriptive `selected` results use the corrected candidate pool. The v6 JSON remains unchanged and should be treated as historical evidence.


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

### What the v7 run supports

- **AUC improvement of at least 0.01 across time: inconclusive.** Snapshot-expanded gains +.01612 on average, but the effect is much larger in the first fold. The later folds do not show a comparably reliable improvement. Prequential-expanded gains +.01196; both base variants are below manual.
- **Sampled NDCG@10 improvement of at least 0.02: not established as a threshold claim.** Snapshot-expanded is better than manual by +.11519 on average, and every conditional interval is positive. However, the smallest lower bound is +.01487. Decayed popularity still scores higher (.48542 versus .31507), and these are sampled candidates rather than logged impressions.
- **Expanded features versus base: a descriptive improvement.** `auto_gain` is higher with the expanded grammar by +.01925 in prequential mode and +.02212 in snapshot mode. The run did not include a separate uncertainty analysis for this contrast.
- **Decay robustness: not established.** This factorial design does not isolate the contribution of decay features or prove robustness under temporal drift.
- **Prefix safety: supported as an implementation invariant.** Tests show that future rows are excluded from prefix features. This does not prove that final KION watch completion was available at recommendation time.


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
| `results/kion_10k_v7_final.json` | **current post-correction factorial evidence** | 10k users, 3 seeds × 3 folds × 4 variants; all 12 cells complete, exact fingerprints and conditional AUC/ranking intervals |
| `results/kion_10k_v7_final.lock.json` | initial immutable v7 lock | Pre-run configuration, data/source/runtime provenance for the nine reused cells; continuation provenance is embedded in the final JSON |
| `results/kion_10k_v6_factorial.json` | historical pre-correction factorial follow-up | 10k users, 3 seeds × 3 folds × 4 variants; completed, exact fingerprints and conditional AUC/ranking CIs |
| `results/kion_1k_v6_factorial.json` | exploratory pilot | 1k users, one seed, two folds; not generalizable or pooled with 10k |
| `results/kion_50k_protocol_v5.json` | diagnostic only | degenerate training-history selection; see audit |
| `results/kion_50k_protocol_v4.json` | **canonical historical v4 evidence** | 50k selected users, three seeds, two folds; corrected historical-catalog candidates |
| `results/kion_50k_corrected.json` | **superseded; provenance only** | predates protocol-v4 candidate-availability correction; do not use for current claims |
| `results/kion_10k_corrected.json` | protocol validation only | 10k selected users, one seed, two folds; not the canonical estimate |
| `results/kion_50k_legacy_pre_correction.json` | historical only | predates candidate, semantic, and uncertainty corrections |
| `historical-results.md` | historical synthetic smoke only | not KION and not comparable to protocol v4 |

### Historical v4 reference summary

The following point summaries come from the six fold/seed cells in the historical canonical JSON. Parentheses show sample SD across those cells; they describe dispersion and are not confidence intervals. The paired AUC difference is also descriptive.

| Metric | Manual | `auto_gain` | Reference |
|---|---:|---:|---:|
| Test ROC AUC | 0.65271 (0.01935) | 0.65408 (0.00412) | Paired difference: +0.00137 (SD 0.01723) |
| Sampled NDCG@10 | 0.12418 (0.00906) | 0.34345 (0.04473) | Cumulative popularity: 0.49785 (0.01787); decayed popularity: 0.52138 (0.01405) |

The canonical JSON records 2,187 fold×seed ranking queries (345–378 per cell) and per-cell skipped-positive counts (22–55; 213 summed over the six evaluation cells, not unique queries). It includes package versions, command, source hashes, source-tree manifest/hash, thread settings, protocol and split metadata, per-query metrics, and per-cell user-cluster intervals for query-weighted NDCG@10. It does not contain `event_auc_uncertainty`, and no event-level AUC bootstrap was recomputed for this artifact. Its immutable provenance snapshot is `metadata.source_tree.sha256 = c77188b69f3d65af97879d48dcecc1b5554fb9b780cbabdb6f374a865903fe13` and code revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c`.

## 6. Reproduction

### v7 final artifact and continuation

The final artifact is a completed 12-cell result, not a single-host rerun. The initial lock is `results/kion_10k_v7_final.lock.json`; the nine cells listed in `continuation.reused_cells` were completed under that lock and the three missing `snapshot/expanded` cells were recomputed on Windows. The final JSON records both provenance snapshots, the checkpoint payload hash, host, timestamps, data cohort fingerprint, and per-cell costs.

To validate the checked-in artifact without retraining:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import run_ablation as ab
p = json.loads(Path("results/kion_10k_v7_final.json").read_text())
config = p["lock"]["config"]
assert p["status"] == "completed" and len(p["cells"]) == 12
assert all(cell["status"] == "completed" for cell in p["cells"])
assert p["lock"] == json.loads(Path("results/kion_10k_v7_final.lock.json").read_text())
assert ab.digest_json({k: v for k, v in p["lock"].items() if k != "lock_sha256"}) == p["lock"]["lock_sha256"]
cohort = p["cohort"]["fingerprint"]
assert all(cell["cohort_fingerprint"] == cohort for cell in p["cells"])
references = {}
for cell in p["cells"]:
    ab.validate_result(cell["result"], config)
    if cell["seed"] in references:
        ab.assert_comparable(references[cell["seed"]], cell["result"])
    else:
        references[cell["seed"]] = cell["result"]
print("v7 result contracts and per-seed comparability: PASS")
PY
```

A fresh full rerun is not required to use the artifact as evidence. A byte-identical scientific rerun requires the original locked environment and data hashes; a new rerun from current sources must use a new lock and output path.

### Historical v6 factorial follow-up

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

The completed v7 factorial follow-up supplies paired AUC/ranking intervals, three temporal folds and seed repetitions under the corrected implementation. It does not close all research requirements. Before broader claims, complete:

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
| Popularity is a strong reference | `results/kion_10k_v7_final.json`: decayed popularity mean NDCG .48542 | Local sampled-discrimination observation only; not universal |

## Final recommendation

Present this as a transparent methodology study led by the completed **post-correction v7** 10k factorial evidence; retain v6 as pre-correction historical evidence, v4 as historical, and v5 as a documented failure. Expanded features improve mean AUC locally in the corrected run, but the benefit is not established as time-robust across periods. Snapshot-expanded improves sampled NDCG versus manual across all nine conditional intervals, while decayed popularity remains stronger. The 1k pilot's different outcome and the remaining horizon/data limits prevent a universal superiority claim. Do not generalize the sampled result to exposure ranking, causal impact, or production lift; the study does not evaluate logged impressions or a live retrieval system.
