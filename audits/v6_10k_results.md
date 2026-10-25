# Completed 10k-user protocol-v6 factorial follow-up

## Evidence and validity checks

Primary artifact: [`results/kion_10k_v6_factorial.json`](../results/kion_10k_v6_factorial.json), status **completed**, 12/12 completed cells; fixed-cohort tooling smoke passed. Separate [configuration lock](../results/kion_10k_v6_factorial.lock.json) and [completion log](../results/kion_10k_v6_factorial.stderr.log) are retained. Source revision `388e48848099f352d05067e556fff882015a3c2e`, source SHA256 `f651ee341147ceed0117491a0557a02085d39d3de696ad6f7e08b351e2a3e055`, lock SHA256 `5ff59a75b37cd13d6110af9b5afe284f021d428b5449c21c1e96f2317388ff78`.

- 10,000 users, 55,922 retained observed watch records, 5,090 cohort items; complete recorded histories for the fixed user sample, sampling seed 42.
- Four factorial variants: prequential/snapshot × base/expanded grammar; model seeds 42/43/44; three temporal folds per variant/seed, **36 fold evaluations total**.
- Every completed result passed `run_ablation.validate_result`. Exact split event-ID fingerprints and candidate fingerprints matched across variants for each seed; no finite-endpoint or class-support failures.
- Common 20% timestamp-group warmup; snapshot cadence target four, not a guarantee of exactly four blocks; test remains frozen for the full role.
- Uniform 99 historical unseen negatives and cap400 attempted positives per fold. Each variant totals **3,287 fold×seed ranking queries**; these are repeated evaluations, not unique independent users/queries. Test row counts are 7,490 / 8,679 / 10,242 per fold, repeated across seeds and variants.

This follows the locked exploratory design in [EXPERIMENT_PLAN_V6.md](../EXPERIMENT_PLAN_V6.md). It is not preregistration on fresh uninspected data. The separate 1k pilot and v4/v5 runs must not be pooled with this result.

## Quality results

All values are unweighted means of nine fold×seed summaries per variant, **not pooled AUC, pooled ranking metrics, or confidence intervals**.

| History | Grammar | Manual AUC | Auto gain AUC | Paired delta | Manual sampled NDCG@10 | Auto sampled NDCG@10 | Paired delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Prequential | Base | .66936 | .66161 | -.00775 | .20991 | .26553 | +.05562 |
| Prequential | Expanded | .66936 | .68087 | +.01150 | .20991 | .26736 | +.05745 |
| Snapshot | Base | .66736 | .66029 | -.00707 | .20979 | .27413 | +.06434 |
| Snapshot | Expanded | .66736 | .68194 | +.01458 | .20979 | .32869 | +.11890 |

Decayed popularity NDCG@10 is **.48542** for every variant, still stronger than all auto-gain variants. Manual scores are exactly unchanged between feature grammars for the same history mode. Expanding grammar changes auto AUC by +.01925 under prequential and +.02165 under snapshot; this is stronger local evidence for the expanded grammar than the small pilot, but not attribution to any individual transform.

The 1k snapshot-expanded mean AUC delta was -.00175, whereas this 10k mean is +.01458. This sign change demonstrates sample sensitivity; it is not a controlled estimate of the causal effect of adding users because cohort, timestamp-group availability, supervised windows, and candidate catalog differ.

## Temporal stability and conditional uncertainty

Snapshot-expanded AUC point deltas are positive in all nine evaluations. Yet only **3/9 paired user-cluster 95% intervals exclude zero**; these are all three seeds of fold0. The remaining six intervals include zero. Average deltas over seeds by temporal fold:

| Fold | Mean AUC delta | Individual seed CI lower-bound range | Individual seed CI upper-bound range |
|---|---:|---:|---:|
| 0 | +.03596 | +.01932 to +.03363 | +.03617 to +.04968 |
| 1 | +.00062 | -.00709 to -.00605 | +.00720 to +.00830 |
| 2 | +.00717 | -.00148 to -.00025 | +.01368 to +.01528 |

Thus the headline AUC mean is driven mostly by the earliest test period. There is no pooled or cross-time confidence interval. Seed repetition does not increase the number of independent temporal periods from three to nine. Per-cell bootstraps are conditional on fitted scores/features and observed populations, not training/selection/time-shift uncertainty. The descriptive SD of the nine paired AUC deltas is .01669; it is not a standard error or CI.

Snapshot-expanded ranking delta intervals are above zero **and above +.02 in all nine cells** (envelope of lower bounds +.03108 to upper bounds +.25274). Other variants have ranking delta lower bounds above zero in 5/9 prequential-base, 6/9 prequential-expanded, and 6/9 snapshot-base cells. This supports a local sampled-ranking pattern, not exposure ranking or a win over popularity. No simultaneous multiple-comparison correction was applied to these exploratory intervals.

## Hypothesis decisions

- **Warmup/block protocol removes empty training-prefix defect:** supported by implementation tests, diagnostics and successful informative-gain selection; not a guarantee of true feedback availability in KION.
- **Auto AUC improves by at least .01, robustly over time:** **inconclusive**. Expanded mean exceeds threshold, but support is concentrated in one fold; two later folds are below .01 and their per-cell intervals include zero. Base grammar is not supported.
- **Expanded grammar improves over base on this fixed 10k comparison:** **supported descriptively/local point estimates** in both history modes. A paired expanded-minus-base inferential interval was not computed; the study does not identify which transform causes the improvement.
- **Snapshot-expanded sampled NDCG improves over manual by .02:** **supported locally/exploratorily** by all nine conditional intervals, not as a confirmatory general ranking claim.
- **Automatic method beats the strongest popularity ranking reference:** **not supported** (.32869 versus .48542).
- **Deployment or cross-dataset generalization:** **unmeasured/inconclusive**; MIND logged-impression evidence remains absent.

## Compute

Whole job including loading, preflight and tooling smoke: **22,450.33 s wall (6:14:10)**, 28,316.05 s CPU, process peak RSS **1,120,968 KiB (~1.07 GiB)**. RSS is a lifetime high-water mark, not independent memory cost of each variant/model.

| Variant (all three seeds, three folds each) | Wall seconds | Approximate minutes |
|---|---:|---:|
| Prequential base | 3,304.46 | 55.07 |
| Prequential expanded | 3,518.72 | 58.65 |
| Snapshot base | 7,230.88 | 120.51 |
| Snapshot expanded | 7,981.47 | 133.02 |

These are end-to-end variant costs (all model families/feature selection/ranking/uncertainty), not per-method training or serving latency. No distributed-scaling conclusion is justified.

## Reproduction boundaries and next analysis

Restore source revision `388e488...` and the locked Python/packages/thread settings; provide raw CSVs with recorded hashes. The retained lock has absolute paths and binds runtime/source metadata. On another checkout, create a new explicit lock with relocated input paths rather than editing the historical lock. Use `run_ablation.py --execute-lock ... --output <new-path>`; existing outputs are never overwritten. The lock's `locked_not_executed` status describes the immutable pre-run configuration, while the result's `completed` status describes execution.

Next priority is mechanism and temporal-error analysis: explain early-versus-late fold differences using retained history/prevalence/support diagnostics, then compare rate/recency/contrast subfamilies and sensitivity to candidate protocols. Fresh temporal or second-dataset validation is needed before calling AUC superiority robust. Do not tune against these already inspected test folds or relabel this run confirmatory. KION aggregate feedback availability and snapshot test freshness-horizon mismatch remain unresolved limits.
