# Modular comparison artifact

This directory now contains a modular, reproducible comparison pipeline.

## Modules

| Module | Responsibility |
|---|---|
| `data.py` | Synthetic interaction log with preference drift and timestamp-safe splits |
| `causal_features.py` | Reusable point-in-time indexes, window aggregates and exponential decay |
| `decay_features.py` | Decay features as an alternative to hard windows |
| `auto_features.py` | Template-based feature specs and candidate computation |
| `feature_selectors.py` | Training-gain top-K and selection-interval forward search with degeneracy guards |
| `models.py` | LightGBM fit/predict interface |
| `linear_model.py` | One-hot, regularized Logistic Regression alternative |
| `pipeline.py` | Shared single-split experiment orchestration and comparison |
| `rolling_eval.py` | Warmup/block-snapshot and prequential rolling temporal evaluation |
| `diagnostics.py` | Predictor-only role diagnostics and history-support summaries |
| `run_ablation.py` | Locked factorial comparison with exact split/candidate identity checks |
| `metrics.py` | Query-group ranking metrics |
| `compare_methods.py` | CLI and multi-seed summaries |
| `mind_data.py` | MIND/MIND-small TSV loader and logged-impression flattening |
| `history.py` | Atomic timestamp-batch history replay |
| `ranking_references.py` | Shared causal ranking baselines for all evaluation modes |
| `experiment_contracts.py` | Result-shape and serialization validation |
| `tests/` | Causality, snapshot, split and history checks |

## Compared methods

- `manual`: hand-designed baseline features + tuned LightGBM.
- `auto_gain`: template-generated features ranked by training-only LightGBM gain.
- `auto_forward`: greedy forward selection by selection AUC, using a regularized linear probe.
- `decay`: baseline features augmented with exponential-decay watch/click counts (KION watch semantics; MIND click semantics).
- `linear`: regularized Logistic Regression on baseline features, compared with LightGBM.

The latest real-data evidence is the completed [corrected v7 10k factorial run](results/kion_10k_v7_final.json), with its [lock](results/kion_10k_v7_final.lock.json). It compares prequential and snapshot history with base and expanded feature grammars on matched splits and candidate sets, using three seeds and three temporal folds. The v7 result is the current post-correction artifact; v6 is retained as pre-correction historical evidence, v4 is historical, and v5 is diagnostic due to degenerate training (see the [v5 audit](audits/v5_validity.md)). See [REPORT.md](REPORT.md) for the full interpretation. KION has no impressions: its event target is
`watched_pct >= 50.0` on observed pair-aggregated watch rows, and sampled ranking
is only historical-catalog unseen-item discrimination. MIND/MIND-small is a
separate logged-impression benchmark: its impression candidates are kept intact
and evaluated with `score_logged_impressions`. Synthetic sampled candidates are
also diagnostic only. MIND `history` and news body/text are retained as metadata
but are not model features in the first feature-search comparison.

MIND supports two explicit protocols. `combined_temporal` (the default) combines requested source splits before chronological splitting. `official_train_dev` keeps train and dev separate, uses chronological train/select/valid partitions inside train, and reports dev as final held-out evaluation. The latter is benchmark-shaped but still not an official leaderboard submission.

Selection data chooses feature/model configurations. The later validation interval is used for LightGBM early stopping, and test is held out for final reporting. Test rows are excluded from feature sources and ranking baselines, so the final metrics use a frozen pre-test snapshot rather than feedback from within the test interval.

## Evaluation protocol

The artifact reports two different tasks and they must not be conflated:

- KION `*_test_auc` is row-level ROC AUC for `watched_pct >= 50.0` versus lower-watch observed user-item records in a chronological held-out interval. Missing `watched_pct` is mapped to zero; this is not click prediction or an impression CTR. MIND's event AUC uses its click labels.
- `*_candidate_ndcg10`, `*_candidate_mrr10` and `*_candidate_recall10` are query-weighted sampled unseen-item discrimination diagnostics in synthetic/KION paths. KION protocol v4 uses one held-out high-watch positive and 99 negatives sampled without replacement from items first observed strictly before the query, excluding items previously seen by that user; positive queries whose item was not historically available are omitted. This is not logged-impression ranking or the production candidate set.
- MIND runs report the same metric names over logged impression candidates. They use
  the actual `query_id` groups from `behaviors.tsv` and do not add sampled negatives.

The ranking output includes point-in-time reference baselines:

- `popularity_candidate_*`: cumulative item popularity before each query;
- `decay_popularity_candidate_*`: seven-day exponentially decayed item popularity before each query;
- `category_popularity_candidate_*`: cumulative category popularity before each query;
- `user_item_ctr_candidate_*`: prior user-item positive-watch-rate baseline for KION; click-rate only where the source actually contains clicks.

For MIND, these popularity baselines use string news IDs directly and remain causal.

Synthetic candidate JSON records the sampling protocol, requested negatives, query cap, and actual query/row counts. MIND ranking uses `logged_impression_candidates` and does not introduce synthetic negatives.



The default `rolling` protocol uses expanding temporal folds and `history_mode=snapshot`, reserving a 20% history-only warmup. Training/select/valid features use historical blocks; test remains frozen for its whole role, a disclosed horizon mismatch. `training_blocks=4` is a cadence target; remainder groups may add blocks. `prequential` with the same positive warmup is the factorial comparator; zero-warmup legacy semantics must be requested explicitly. `single_split` is retained as an explicitly exploratory compatibility mode because feature search and method choice reuse one selection interval.

```bash
python compare_methods.py --evaluation rolling --folds 3 --seeds 42,43,44 --n-events 30000 --json
```

Each fold selects features/configuration on its own selection interval, uses validation for LightGBM early stopping, and reports later test AUCs. Fold selection histories are retained in JSON.


## Reproducibility status

Install dependencies with `python -m pip install -r requirements.lock`, then run `python -m pytest -q`. The lock was verified with Python 3.11.15. The test suite covers timestamp atomicity, snapshot masks, history replay for numeric and string IDs, split boundaries, candidate-label isolation, and causal popularity baselines.

The latest [10k factorial artifact](results/kion_10k_v7_final.json) contains 12 cells and 36 fold evaluations under the `kion_factorial_ablation_v1` schema. Validate it with `run_ablation.validate_result` and `assert_comparable`, not the separate compare-method schema validator. Follow the [REPORT reproduction instructions](REPORT.md#6-reproduction). [`results/kion_50k_protocol_v4.json`](results/kion_50k_protocol_v4.json) remains canonical historical v4 evidence, with six fold/seed cells from 50k users. `results/kion_50k_corrected.json` is superseded pre-v4 provenance; `results/kion_10k_corrected.json` is an older protocol-validation artifact, not the 10k factorial study. `historical-results.md` contains pre-correction synthetic smoke, not KION evidence. Do not pool these different protocols or cohorts. Preserve the command, seed list, dataset size, dependency versions, and JSON output with any new report.

Candidate-based evaluation is implemented in two explicitly separate modes: sampled
historical-catalog unseen-item discrimination for synthetic/KION, and logged impression candidates
for MIND. Only the latter supports an impression-level ranking claim. The MIND path
is available through `python compare_methods.py --mind-root <extracted-mind-root>`.

The causal index reuses sorting and time-window boundaries. A local smoke benchmark on 15k rows and 90 specs took about 4.1 seconds; this is not a distributed/production-scale benchmark.
