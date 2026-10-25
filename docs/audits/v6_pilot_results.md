# Completed protocol-v6 factorial pilot

## Scope and lock

Evidence: `results/archive/v6/kion_1k_v6_factorial.json`, status `completed`; 1,000 fixed users, all 5,505 retained rows, cohort seed 42, model seed 42, two temporal folds. Initial 20% timestamp groups are history-only; training-block cadence target four (observed four blocks in fold0 and five in fold1 due to remainder groups; not exactly four). Same supervised role event-ID hashes and candidate fingerprints asserted across all four factorial cells. Candidate protocol: uniform 99 negatives, cap 400 attempted positives per fold, selected-cohort historical catalog.

The original 50-user tooling smoke failed because feature selection had no positive training gains. Preserved as `results/archive/v6/kion_1k_v6_failed_smoke.json`. No guard was weakened and no users were repeatedly resampled. A new explicit lock increased the tooling cohort to all 1,000 pilot users; the final artifact retains that lock and smoke results. This change responds to training support, not favorable test scores. Its smoke is not independent validation evidence and must not be pooled with the pilot.

## Results

Unweighted mean of two temporal-fold point estimates; one model seed. These means are not pooled metrics or confidence intervals.

| History | Grammar | Manual AUC | Auto gain AUC | Delta | Manual sampled NDCG@10 | Auto sampled NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| Prequential + same warmup | Base | .64282 | .62401 | -.01881 | .29922 | .18051 |
| Prequential + same warmup | Expanded | .64282 | .61728 | -.02554 | .29922 | .19541 |
| Snapshot blocks | Base | .62717 | .61183 | -.01534 | .22834 | .21942 |
| Snapshot blocks | Expanded | .62717 | .62542 | -.00175 | .22834 | .23360 |

Decay popularity sampled NDCG@10 is .45345 in every cell. Its unchanged result and exact candidate hashes provide useful consistency checks. Manual predictions/metrics are unchanged between grammars within each history mode. Expanded grammar changes auto AUC by -.00673 under prequential history and +.01359 under snapshot history. That interaction is exploratory, not a general improvement claim.

Snapshot-expanded fold AUC deltas are +.00195 and -.00545. Thus even the pointwise direction is not stable across these two periods.

## Hypothesis decisions

- Warmup/block training removes the structurally empty train prefix: **supported for implementation invariants**, by regression tests and retained per-role diagnostics. This is not proof of exposure availability in KION.
- Auto gain improves AUC by at least .01 over manual: **not supported in this pilot**; all four mean differences are negative.
- Expanded grammar universally improves the base grammar: **not supported**; signs differ across history modes.
- Auto gain beats the strongest popularity ranking baseline: **not supported**.
- Generalization to larger cohorts/seasons or online recommendation quality: **inconclusive/unmeasured**.

## Compute and uncertainty

Whole job including load and four-cell tooling gate: 1,684.12 seconds wall, 2,061.96 seconds CPU, peak RSS 840,256 KiB (~0.80 GiB). Main four-cell wall times: 237.94 / 255.97 / 359.42 / 385.62 seconds. RSS is process lifetime high-water mark, not independently attributable per method.

The artifact retains per-fold user-cluster conditional AUC intervals and ranking query-cluster intervals. Do not treat two folds/one seed as an adequately powered broad comparative study. The periods were previously inspected; this is exploratory, not a new untouched confirmatory holdout.

## Limitations and next locked run

Snapshot test remains frozen for the complete role; train/select/valid use shorter blocks. This freshness horizon mismatch is disclosed rather than called serving equivalence. Warmup and cohort changes mean v5-v6 metric changes cannot be attributed exclusively to one patch. KION remains pair-aggregated last-watch data, not impressions.

Next run fixed without metric-driven tuning: 10,000 complete-history users, seeds 42/43/44, three folds, same 20% warmup/block-cadence target four (remainder blocks explicitly accepted), four factorial cells, uniform99/cap400. Larger-run results must be retained separately, including failures and diagnostics; no success claim before completion.
