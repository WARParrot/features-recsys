# Audit remediation status

This ledger separates implemented corrections from experiments and data limitations. No historical result has been relabeled as a new run.

## Implemented and regression-tested

- KION loader: valid missing feedback is not rejected merely because it occurs in the last row; required fields and malformed feedback are validated separately.
- Source provenance: source/config-only hashing excludes result costs, build output, editable-install metadata, caches and virtual environments.
- End-to-end KION rolling fixture and wheel installation smoke in CI.
- Feature search: auditable rates, short/long-window contrasts and point-in-time recency; deterministic entity-family-balanced forward-search budget instead of first-enumerated candidates.
- Rolling role-frozen history: strict pre-role prefixes, no intra-role updates; explicit legacy prequential option.
- Candidate sensitivity API/CLI: requested negative count, query cap and uniform/popularity/recent sampling.
- Static catalog safety: explicit item/category/first_seen contract; future items excluded.
- Paired user-cluster event AUC uncertainty and explicit conditionality metadata.
- Empty candidate evaluation avoids model prediction on an empty matrix.

## Evidence still required — not claimed complete

- New 50k/all-user canonical experiment on the corrected implementation, with AUC uncertainty, cost, stderr and immutable source snapshot.
- Real MIND official_train_dev experiment. MIND data are not present in this checkout; loader/tests are not empirical benchmark evidence.
- Multi-seed, >=3 temporal-fold sensitivity matrix across negative counts/sampling policies/watch thresholds.
- Cold-start, popularity/history-length slices, ranking coverage/novelty/diversity and per-stage scaling measurements.
- A broader algorithmic novelty claim or SQL-agent search. Current method is expanded but still bounded template search.

## Inherent data/claim limits

KION cannot recover impressions, exposure propensities, first recommendation decisions, or within-day event ordering from pair-aggregated last-watch records. Timestamp-safe code does not establish real-time feedback availability. The supported task remains retrospective observed-pair classification plus explicitly conditional sampled discrimination. A negative result or lack of superiority is a research conclusion, not a software defect to tune away.

## Replay boundaries

The protocol-v4 JSON remains immutable historical evidence at revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c`. Current runs use expanded search and rolling protocol metadata version 5. Do not pool their metrics with protocol v4. The outer result schema is still version 4; rolling semantics are separately versioned in `rolling_protocol_version` and `rolling_protocol`.
