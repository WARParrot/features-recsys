# Protocol-v6 validity and factorial experiment lock

## Decision and scope

Determine whether expanded template features improve observed-pair watch classification under informative, leakage-safe training history. KION sampled discrimination is secondary diagnostic evidence, never logged-exposure or causal evidence. This is an exploratory follow-up on already inspected KION periods, not a preregistered fresh holdout.

## Known failure to resolve

Protocol v5 freezes training features at the first training timestamp. For the existing expanding splits this leaves zero source rows for all training examples. Its AUC delta cannot by itself establish a strong-baseline win. Preserve the v5 JSON and runtime log as diagnostic evidence, with no retrospective modification.

## Design fixed before follow-up metrics

- Sample users, retaining their complete recorded histories; fixed cohort seed 42.
- Reserve initial 20% timestamp groups as history-only warmup.
- Same chronological train/select/valid/test boundaries for all four cells.
- Factor 1: prequential historical features versus historical training-block snapshots.
- Factor 2: base aggregate template grammar versus expanded rate/contrast/recency grammar.
- Snapshot training: cadence derived from `floor(train_timestamp_groups / training_blocks)` with target parameter four, strict pre-block sources. A remainder can produce an additional block: the completed pilot has four blocks in fold0 and five in fold1. This cadence-based behavior is explicitly accepted for the next lock; `training_blocks` is a cadence target, not an exact block-count promise. No current-block feedback enters a feature.
- Selection/validation/test snapshot timing must be explicit in output. Different snapshot horizon distributions are a limitation, not assumed serving equivalence.
- Manual LightGBM and popularity references are retained in every cell; tuning budget and candidate sets fixed.
- Uniform 99 unseen historical negatives, cap 400 attempted positive queries per fold. Identical query IDs and split metadata across cells must be asserted.
- Primary endpoint: paired auto_gain minus manual event AUC, with user-cluster conditional bootstrap intervals. Minimum relevant AUC delta 0.01 is a declared exploratory threshold, not a guarantee.
- Secondary: sampled NDCG@10, popularity comparison, per-role history/constant-feature diagnostics and runtime/RSS.
- Training seeds cannot be counted as independent temporal replications.

## Execution and failure policy

1. Unit/integration tests and synthetic tooling smoke (not research evidence).
2. Deterministic 1,000-user, seed 42, two-fold four-cell pilot. Do not claim generalization from this pilot.
3. If validity gates pass, retain a separate locked 10,000-user, seeds 42/43/44, three-fold follow-up. Do not change settings based on test metric sign.
4. Outputs written atomically; status/exit errors retained. Source/input hashes and elapsed costs saved. Missing/failed cells do not become zeros or omitted successes.
5. Invalid training-history, split mismatch, differing candidate identities or nonfinite metrics block scientific conclusions. Correctness repairs require a new manifest and rerun, never editing metrics.

## Conclusions and unclosed external evidence

Label each effect supported locally / not supported / inconclusive with explicit cell/time scope. An observed failure is a valid research conclusion. MIND real-data evaluation, retrieval impact, broader cold-start/scalability and novel algorithmic claims are not established by this factorial study. Missing impression times cannot be recovered from KION aggregates.
