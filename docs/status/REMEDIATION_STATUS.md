# Audit remediation status

This ledger separates implemented corrections from experiments and data limitations. No historical result has been relabeled as a new run.

## Implemented and regression-tested

- KION loader: valid missing feedback is not rejected merely because it occurs in the last row; required fields and malformed feedback are validated separately.
- Source provenance: source/config-only hashing excludes result costs, build output, editable-install metadata, caches and virtual environments.
- End-to-end KION rolling fixture and wheel installation smoke in CI.
- Feature search: auditable rates, short/long-window contrasts and point-in-time recency; deterministic entity-family-balanced forward-search budget instead of first-enumerated candidates.
- Protocol-v6 snapshot training: separate warmup and strict-prior training blocks, predictor-only per-role diagnostics, exact split/candidate fingerprints. Legacy role-frozen is retained for diagnostic replay, not promoted as valid learned feature selection.
- Candidate sensitivity API/CLI: requested negative count, query cap and uniform/popularity/recent sampling.
- Static catalog safety: explicit item/category/first_seen contract; future items excluded.
- Paired user-cluster event AUC uncertainty and explicit conditionality metadata.
- Empty candidate evaluation avoids model prediction on an empty matrix.
- Selection guards reject all-constant/no-positive-gain training and zero/nonfinite forward improvements.
- Locked 2×2 real-data factorial runner with preflight class/candidate support, finite-metric success gates, atomic checkpoints and cost metadata.
- Validator rejects mixed semantic versions/splits despite identical outer schema version.

## Completed experimental evidence

The 10k-user, three-seed, three-temporal-fold, four-variant locked factorial follow-up completed all 12 cells. Exact split/candidate fingerprints, finite endpoints and diagnostic gates passed; conditional user-cluster AUC/ranking intervals and end-to-end costs are retained. See `../audits/v6_10k_results.md`. Expanded-snapshot mean AUC gain +.01458 is mostly early-period-specific; sampled NDCG gain is locally supported but popularity remains stronger. This closes the planned factorial follow-up, not robustness across all candidate/target policies or a deployment claim.

## Evidence still required — not claimed complete

- New 50k/all-user canonical experiment on the corrected implementation, with AUC uncertainty, cost, stderr and immutable source snapshot.
- Real MIND official_train_dev experiment. MIND data are not present in this checkout; loader/tests are not empirical benchmark evidence.
- Candidate/target sensitivity matrix across negative counts/sampling policies/watch thresholds. Three-seed/three-fold coverage is now completed for the fixed uniform99, threshold50 factorial experiment only.
- Cold-start, popularity/history-length slices, ranking coverage/novelty/diversity and per-stage scaling measurements.
- A broader algorithmic novelty claim or SQL-agent search. Current method is expanded but still bounded template search.

## Inherent data/claim limits

KION cannot recover impressions, exposure propensities, first recommendation decisions, or within-day event ordering from pair-aggregated last-watch records. Timestamp-safe code does not establish real-time feedback availability. The supported task remains retrospective observed-pair classification plus explicitly conditional sampled discrimination. A negative result or lack of superiority is a research conclusion, not a software defect to tune away.

## Replay boundaries

The protocol-v4 JSON remains immutable historical evidence at revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c`. The rewritten signed-history equivalent of revision `6f8d386...` is `b3c962f`; trees are unchanged. Protocol-v5 completed but its training was degenerate; see `../audits/v5_validity.md`. Current runs use warmup/block snapshots and rolling protocol metadata version 6. Do not pool their metrics with protocol v4. The outer result schema is still version 4; rolling semantics are separately versioned in `rolling_protocol_version` and `rolling_protocol`.
