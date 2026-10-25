# Protocol-v6 pilot: independent final review

## Scope and decision

Read-only code/result review, except creation of this audit. Evidence: `run_ablation.py`, `rolling_eval.py`, `diagnostics.py`, warmup/split implementation, `EXPERIMENT_PLAN_V6.md`, and `results/archive/v6/kion_1k_v6_factorial.json`. No full model runs, commits, or subagents. Targeted tests: **89 passed** (`test_warmup_protocol`, `test_diagnostics`, `test_protocol_comparability`, `test_selection_validity`).

**Bottom line:** pilot execution/comparability and informative-history repair pass; recommendation-quality superiority does not. One explicit plan/implementation discrepancy should be acknowledged before interpreting the locked follow-up.

## Validity checks

- Artifact status **completed**, smoke **passed**; all four expected cells **completed**, seed 42, two folds each. No failed/missing cell represented as success.
- All four cohort fingerprints equal the top-level fingerprint `5b41a6a2829b83c3138f1812e8d6531cc7c5e5daef24ecf5b966de1131e19714`: **1,000 users, 5,505 events, 1,885 items**. Recorded source-file hashes match the reviewed files. Input files were not rehashed in this fast review.
- Independently asserted exact equality of **all role split fingerprints**, split metadata, fold seeds, and candidate fingerprints across cells. Warmup has 1,112 rows; train rows 455/1,164 and test rows 977/1,292. Candidate support is 275/287 queries, each with 99 negatives, from 400 attempted positives per fold; skipped queries must not be described as 400 evaluated queries.
- Training predictors are informative, not the old all-zero-history failure. Constant columns (folds 0/1 respectively): prequential/base **51/135, 51/135**; prequential/expanded **80/189, 80/189**; snapshot/base **53/135, 53/135**; snapshot/expanded **82/189, 82/189**. Of 32 inspected count predictors, 27 vary in prequential cells and 26 in snapshot cells. Some absent-event/window predictors legitimately remain constant. Role-prefix training history means are 9.44/6.48 events; these diagnostics deliberately describe a common prefix, not each cell's within-role history policy.
- Snapshot features use strict pre-block sources; test feedback is excluded. Test is one frozen block in both modes. Timestamp atomicity and positional alignment have targeted coverage. Source hashing, immutable configuration checking, finite primary AUC validation, candidate identity checks, and atomic artifact writes provide useful fail-closed safeguards; this is not proof against every data/availability assumption.

## Actionable finding / interpretation guardrail

**Medium — planned four training blocks are not always four.** `rolling_eval.py:_history_blocks` uses `cadence = n_groups // training_blocks` and then steps through all groups; a remainder can create an extra block. The pilot records **four train blocks in fold 0 but five in fold 1**, despite `training_blocks=4` and the plan's “four timestamp-atomic blocks.” This does not invalidate the observed cross-cell identity/history checks, but the configuration is effectively a cadence target rather than an exact block count. Before follow-up, explicitly accept/document that semantics or implement exact timestamp-group partitioning with a new source lock and rerun; do not silently relabel this artifact or change its metrics. Snapshot select/valid also use multiple blocks, whereas test remains frozen: horizon mismatch is a real limitation, not serving equivalence.

## Quality results

Arithmetic fold means below are descriptive, **not pooled AUCs or confidence intervals**. `Gain` means the declared primary `auto_gain` comparator, not a post-hoc choice of forward selection.

| History / grammar | Manual AUC | Gain AUC | Gain − manual | Manual NDCG@10 | Gain NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Prequential / base | .64282 | .62401 | −.01881 | .29922 | .18051 |
| Prequential / expanded | .64282 | .61728 | −.02554 | .29922 | .19541 |
| Snapshot / base | .62717 | .61183 | −.01534 | .22834 | .21942 |
| Snapshot / expanded | .62717 | .62542 | −.00175 | .22834 | .23360 |

Paired gain-minus-manual AUC effects and 95% user-cluster bootstrap intervals, fold 0 / fold 1:

- Prequential/base: −.01445 [−.04017, .01280] / −.02317 [−.04430, −.00396].
- Prequential/expanded: −.02875 [−.05313, −.00385] / −.02233 [−.04464, −.00102].
- Snapshot/base: −.01411 [−.04834, .01684] / −.01656 [−.04167, .01236].
- Snapshot/expanded: +.00195 [−.02545, .03177] / −.00545 [−.02819, .01695].

These intervals use 1,000 valid resamples, 317/374 test-user clusters, conditional on fitted scores/features and the observed population. They exclude retraining, selection and temporal-shift uncertainty; folds/seeds are not independent temporal replications. No paired cross-grammar AUC interval or factorial interaction interval is supplied, so do not infer those from overlapping marginal intervals.

Gain NDCG is significantly below manual conditionally in both prequential folds. Snapshot/base changes sign by fold (paired delta CIs [−.10991, −.02444] / [.00585, .09151]); snapshot/expanded is uncertain in both ([−.00354, .07010] / [−.05293, .00805]). Popularity NDCG is .46191/.44897 and decayed popularity .46698/.43992, materially above gain in every cell/fold. Expanded forward-selection AUC means (.64509 prequential, .63919 snapshot) cannot replace the primary endpoint; forward NDCG is mixed and lacks a consistent superiority result.

## Hypothesis verdicts and next decision

1. **Warmup/snapshot repair yields informative leakage-safe training history: supported locally** by actual varying counts, strict-source code paths and targeted tests. This does not recover true within-day feedback availability from KION pair aggregates.
2. **Primary: gain exceeds manual AUC by at least .01: not supported in this pilot in every cell.** All cell means are negative; expanded snapshot's small positive fold-0 estimate is below threshold and uncertain. Absence of support is not proof of equivalence; material gains remain inside some intervals.
3. **Expanded grammar improves quality over base: inconclusive overall.** Gain AUC changes are −.00673 prequential and +.01359 snapshot (descriptive means), with no cross-cell paired uncertainty. This is history-dependent evidence, not a general expansion win.
4. **Expanded snapshot provides a reliable sampled-ranking improvement / beats popularity: not supported.** Its small mean gain over manual is fold-fragile and uncertain; it remains far below popularity. No logged-exposure, causal, retrieval, online, or broader-population success is established.

Proceed only as a separately locked exploratory follow-up, preserving all cells and the declared endpoint irrespective of sign, after explicitly resolving/accepting block-count semantics. A plausible failure mechanism is sparse history plus expanded redundant/constant templates and a gain-only feature set that is not the manual-plus-feature intervention; this is a hypothesis, not demonstrated causation. Retain popularity and manual references and inspect fixed cold-history slices in the follow-up. Any later materially different manual-plus-regularized-template intervention needs its own prospective lock, not pilot-driven retuning of this one.
