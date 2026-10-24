# Independent audit: KION protocol-v5 train-history degeneracy

## Scope and verdict

Read-only audit of revision `9a62d2789ad99a3591d8dc7a97c4d1eff9d920aa` and `results/kion_50k_protocol_v5.json`. Only this new report was written; no production changes, commits, subagents, model fitting, or model experiment. Existing result files were untracked at initial inspection. Other threads were active; the source-manifest match below describes the audit's initial snapshot, not any subsequent changes by others.

**Decisive finding: the frozen-role implementation is temporally conservative but degenerates training. It does not provide a valid test of learned gain-based feature selection.** Both folds' train role starts at the first dataset timestamp. Its strict preceding prefix is empty, even when the second fold's train block expands. Observed first-fold training has **19/19 manual-history, 144/144 generated, and 24/24 decay features constant zero**. Only the two categorical manual predictors vary. All six saved `auto_gain_select_auc` values are exactly 0.5. The apparently favorable final `auto_gain` results do not rescue the selection claim: final refitting includes select rows with populated history, whereas feature/configuration selection had none.

### Hypothesis verdicts

- **Train-history degeneracy exists: supported**, by direct distributions and source logic; this is deterministic, not a statistical inference.
- **`top_gain` identifies informative features: not supported.** All training gains must tie at zero. Exact pandas zero-gain ordering reproduces all six saved top-five lists.
- **`role_frozen` excludes within-role outcomes from aggregate sources: supported by source inspection**, conditional on the supplied row timestamps being actual availability times. This is not a full raw-data availability guarantee.
- **Protocol-v5 establishes superiority of automatic feature selection or production recommendation quality: inconclusive / not established.** Selection is degenerate, refit distribution differs, and KION candidate scoring is sampled discrimination rather than logged-exposure evaluation.

## Identity and provenance

Observed cohort: **50,000 users, 8,385 items, 285,343 records, 163 distinct daily timestamps; zero duplicate user-item pairs**. Cohort sampling seed is 42 for all three model seeds (42,43,44), not three independently sampled cohorts. Two temporal folds, six model-seed/fold results. Metadata `n_events=30000` is the synthetic CLI default, not the actual KION row count.

All files in the JSON source-tree manifest matched their recorded SHA256 at audit start. All three local raw CSV hashes matched JSON metadata:

| Evidence | SHA256 |
|---|---|
| JSON audited | `01805133950479278142e67995e309b1224f3de9ae7bf683ece599b97c08439e` |
| interactions.csv | `201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf` |
| users.csv | `37b2945240628a3f215e1fb5f79b4cff386d8988b21fa61e5b135aef4541182c` |
| items.csv | `027c3a0027b4fe4a8db9d5fa626ab40bab507c96aac81d462db0771eb2cb4a8e` |

This checks consistency with recorded provenance, not an independent cryptographic verification of a commit signature.

## Mechanism and evidence ledger

| Source | Observed contract / consequence |
|---|---|
| `data.py::rolling_time_splits` | Each fold's cuts start at zero. Train expands, but never has a preceding warm-up block. Timestamp groups are not split. |
| `rolling_eval.py::_fold_features` | Each role uses `timestamp < role_start` and `source_until=role_start`. For train, that mask is empty. |
| `causal_features.py::CausalIndex` | Inactive rows contribute neither counts nor feedback; no prior recency returns zero. Windows still age with query time. |
| `auto_features.py` | Six key families, count/sum/rate/contrast/recency produce 144 generated features. Empty-source transforms are all zero. |
| `decay_features.py` | Four key families × three half-lives × count/feedback = 24 features; empty-source decay is zero. |
| `kion_data.py::load_kion` | Label is `watched_pct >= 50`; missing watched percentage is filled with zero; purchases are fabricated constant zero. `is_click` is an alias, not click feedback. |
| `feature_selectors.py::gain_top_k` | Fits only train then sorts importance, without filtering zero gains or constant features. An all-zero gain tie returns arbitrary implementation-order features. |
| `feature_selectors.py::forward_select` | Zero improvement is accepted (`improvement < 0`, not `<= 0`, at default threshold). |
| `rolling_eval.py::_fold_result` | Configuration/feature selection fits train; final trees refit train+select with valid early stopping. Populated select history becomes learnable only at refit. |
| Saved JSON | All six auto selection AUCs = .5; identical top-five ties; fold 1 forward search adds three zero-improvement features for every seed. |

These are local primary code/data sources. No external literature was needed for the deterministic zero-feature conclusion. Raw KION documentation/source attribution is recorded by the adapter as https://github.com/irsafilo/KION_DATASET; upstream documentation and label-availability timing were not independently fetched/validated here.

## First-fold actual populations and history availability

Availability below is **fraction of query rows** whose key exists in the strict role prefix, not fraction of unique users. Complete selected-user histories means retained records for sampled users; it does not supply pre-dataset history.

| Role | Rows | Prefix rows | Positive labels | Prevalence | User history | Item history | User-item history | User-category history | Category history | Segment-category history |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 23,413 | 0 | 11,317 | 48.3364% | 0% | 0% | 0% | 0% | 0% | 0% |
| select | 38,475 | 23,413 | 16,934 | 44.0130% | 39.9272% | 65.3073% | 0% | 4.2937% | 97.7804% | 67.5867% |
| valid | 52,264 | 61,888 | 23,856 | 45.6452% | 44.9908% | 79.8867% | 0% | 5.7822% | 98.6645% | 81.3811% |
| test | 77,000 | 114,152 | 38,825 | 50.4221% | 51.0662% | 86.7649% | 0% | 7.2078% | 97.8247% | 89.1091% |

Train timestamps 1615593600–1618272000; select 1618358400–1621123200; valid 1621209600–1623888000; test 1623974400–1626739200. Train has 1,213 distinct item-category codes and 134 distinct user-segment codes: metadata, not history, explains its nonconstant manual input.

Second-fold train has 61,888 rows but the same initial timestamp and empty prefix. Its later roles have 52,264 / 77,000 / 94,191 rows. Full second-fold distributions were not recomputed; train zero-feature conclusion follows exactly from the code and prefix definition.

## Constant / zero feature counts

Each reported constant is also identically zero; no nonzero constant was observed in these computed history families.

| Role | Manual history (19) | Generated (144) | Decay (24) |
|---|---:|---:|---:|
| train | **19** | **144** | **24** |
| select | 5 | 69 | 6 |
| valid | 5 | 69 | 6 |
| test | 5 | not fully recomputed | not fully recomputed |

The five persistent manual zeros are `u_purch_all`, `u_cvr_all`, `ui_views_all`, `ui_clicks_all`, `ui_ctr_all`. Pair uniqueness means query pairs never occur in prior observed records. All 54 purchase-derived generated features are structurally zero. The user-item family has 24 features, nine overlapping those purchase features: **54 + 24 − 9 = 69 structural generated zeros**. All six user-item decay features are structurally zero. Thus test has **at least** 69 generated and six decay zeros, but the exact total is not claimed as directly measured. These are dataset/target-specific failures, not evidence that user-item histories are useless in repeat-event datasets.

## Observed baseline distributions, first fold

For compactness, each cell is **zero fraction / number of distinct values / mean**. All baseline-history minima are zero. Train is uniformly **1 / 1 / 0** for every row and omitted from this table. The table reports actual values, not synthetic/small-sample approximations.

| Feature | select | valid | test | test maximum |
|---|---|---|---|---:|
| u_views_all | .600728 / 53 / 3.723847 | .550092 / 79 / 6.010236 | .489338 / 106 / 7.515351 | 209 |
| u_views_7d | .951631 / 25 / .206056 | .940839 / 25 / .224878 | .924221 / 28 / .268623 | 44 |
| u_clicks_all | .683223 / 37 / 2.066901 | .633744 / 51 / 3.126913 | .573221 / 71 / 3.860156 | 138 |
| u_clicks_7d | .963145 / 16 / .106563 | .956356 / 18 / .117729 | .941688 / 22 / .141584 | 26 |
| u_purch_all | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 | 0 |
| u_ctr_all | .683223 / 153 / .205587 | .633744 / 282 / .215897 | .573221 / 392 / .245296 | 1 |
| u_ctr_7d | .963145 / 50 / .026391 | .956356 / 60 / .030733 | .941688 / 79 / .042338 | 1 |
| u_cvr_all | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 | 0 |
| i_views_all | .346927 / 97 / 27.446186 | .201133 / 164 / 136.194474 | .132351 / 222 / 510.540208 | 3488 |
| i_views_7d | .909604 / 56 / 1.056140 | .877449 / 98 / 3.655327 | .838000 / 129 / 42.827273 | 2076 |
| i_clicks_all | .427602 / 70 / 13.843509 | .279638 / 101 / 47.161277 | .163688 / 135 / 200.340117 | 1451 |
| i_ctr_all | .427602 / 221 / .316911 | .279638 / 443 / .329845 | .163688 / 630 / .414717 | 1 |
| ui_views_all | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 | 0 |
| ui_clicks_all | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 | 0 |
| ui_ctr_all | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 | 0 |
| uc_views_all | .957063 / 12 / .091696 | .942178 / 16 / .155805 | .927922 / 20 / .172130 | 93 |
| uc_clicks_30d | .982872 / 10 / .026875 | .981727 / 11 / .029236 | .980584 / 10 / .027584 | 10 |
| uc_views_30d | .974477 / 12 / .042599 | .970458 / 13 / .051068 | .968649 / 17 / .061416 | 54 |
| uc_ctr_30d | .982872 / 15 / .015762 | .981727 / 15 / .016512 | .980584 / 21 / .017680 | 1 |

Short-window zeros are additionally induced by freezing a source prefix for an approximately month-long role: history ages out as query time advances. Frozen **sources** do not mean frozen numerical feature values; windows, decay, and recency continue changing with query time. This is coherent for stale batch features, not automatically representative of refreshed online serving.

## Gain/forward selection and saved performance

All six recorded top-five names, in order:

1. `user__count_rows__all`
2. `user__sum_is_positive_watch__all`
3. `user__rate_is_positive_watch__2592000s`
4. `user__rate_is_purchase__2592000s`
5. `user__rate_is_positive_watch__7776000s`

`pd.Series(np.zeros(144), index=generated_columns).sort_values(ascending=False).head(5)` exactly reproduces them under installed pandas 2.2.3. Reconstructed top-40 tie order includes **16 purchase-derived features**, necessarily zero in every role, and multiple user-item features. The JSON stores top five, not the full selected 40 or actual gain magnitudes; the top-40 description is a deterministic reconstruction from the audited code/environment, not a saved fitted-model inspection. With every training column constant, positive split gain is impossible; no retraining was necessary to establish this.

Fold 0 forward history is empty for every seed. Fold 1 selects, for every seed, `item__contrast_is_positive_watch__604800s_vs_2592000s`, `item__contrast_is_purchase__604800s_vs_2592000s`, and `item__count_rows__2592000s`, each at **zero improvement**, score .6343574787101469. No positive selection evidence exists for these additions.

Saved fold/seed summary means (descriptive, not independent replications):

| Method | select AUC | test AUC | sampled candidate NDCG@10 |
|---|---:|---:|---:|
| manual | .629422 | .624278 | .149171 |
| auto_gain | **.500000** | .655269 | .196582 |
| linear | .635182 | .634061 | .306863 |
| popularity reference | — | — | .497852 |
| decay popularity reference | — | — | .521378 |

The test advantage of `auto_gain` over manual is a real saved metric of these fitted procedures, but does **not** demonstrate informative importance ranking. Train+select refitting gives the arbitrary retained features a chance to learn; initial manual tuning predominantly tests categorical predictors, and decay tuning cannot evaluate useful decay signals in train. The comparison is confounded by the degenerate selection stage. Fixed-model user-bootstrap CIs cannot repair a deterministic protocol defect or capture selection/retraining uncertainty. Three seeds reuse the same cohort and folds; pooled counts are not distinct users or independent temporal replications.

## `role_frozen` validity boundaries

1. **Causality of source rows:** strict timestamp prefixes and atomic dates prevent same-role feedback entering source aggregates. Train/select/valid/test chronology is explicit. This audit did not independently perturb all feature targets or run the full tests.
2. **Availability semantics unresolved:** KION rows are pair-aggregated watch records dated by `last_watch_dt`. A record's final watched percentage and last-watch date do not reconstruct when partial outcomes or previous watches became available. Timestamp-causal code is not proof of event-time observational fidelity.
3. **Refit history policy:** train+select concatenation retains two different feature-source policies, with all train rows zero and select rows computed from train. It does not rebuild a homogeneous warm-history supervised dataset. Valid/test have still larger prefixes, substantial support shift, and changing label prevalence.
4. **Cold start:** approximately 49% of first-fold test records have no selected-cohort user history, versus 100% of train records. Most user-category records lack history. Empty history must not be mistaken for negative preference.
5. **Target/exposure mismatch:** event AUC predicts high watch among observed watches. Sampled candidate relevance instead treats heldout high-watch items as positives and unseen sampled items as negatives, not observed non-interest. Neither is an impression CTR or causal business metric.
6. **Candidate restrictions:** saved runs use uniform 99-negative cohort catalog sampling; 345–378 queries per fold/seed survive historical-availability filtering, with 22–55 positive queries skipped. This conditions ranking evaluation on retained historically available positives. The JSON flag `positive_items_historically_available=false` must not alone be read as proof the filter was absent; the query skipping and implementation govern interpretation.
7. **Baseline strength:** popularity and decay-popularity dominate learned methods on this sampled diagnostic. An auto-vs-manual win is not a win over the strongest recorded reference.

## Reproduction commands (no models)

Run from `/root/t-lab/recsys_corrected_artifact`, preserving the audited source revision and original input hashes. The diagnostic avoids bytecode writes. Role-prefix truncation below is equivalent for the target rows: removed rows are strictly later and inactive, and feature indices remain their original positional indices.

```bash
git rev-parse HEAD
git status --short
sha256sum results/kion_50k_protocol_v5.json datasets/kion/{interactions,users,items}.csv
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u - <<'PY'
import json
import numpy as np
import pandas as pd
from kion_data import load_kion
from data import rolling_time_splits
from causal_features import build_baseline_features, BASELINE_FEATURES
from auto_features import enumerate_specs, compute_specs, spec_name
from decay_features import build_decay_features
x = json.load(open('results/kion_50k_protocol_v5.json'))
log = load_kion('datasets/kion/interactions.csv', 'datasets/kion/users.csv',
                'datasets/kion/items.csv', max_users=50000, seed=42)
log, folds = rolling_time_splits(log, n_folds=2)
specs = enumerate_specs(feedback_target='is_positive_watch')
for role, pos in folds[0].items():
    sub = log.iloc[:int(pos[-1])+1]
    cutoff = int(log.iloc[pos[0]].timestamp)
    mask = sub.timestamp.to_numpy() < cutoff
    q, prefix = sub.iloc[pos], sub[mask]
    print(role, len(q), mask.sum(), q.binary_label.sum(), q.binary_label.mean())
    for keys in [('user',), ('item',), ('user','item'), ('user','item_cat'),
                 ('item_cat',), ('user_segment','item_cat')]:
        print(keys, pd.MultiIndex.from_frame(q[list(keys)]).isin(
            pd.MultiIndex.from_frame(prefix[list(keys)])).mean())
    b = build_baseline_features(sub, source_mask=mask, source_until=cutoff).iloc[pos]
    for c in BASELINE_FEATURES:
        a = b[c]
        print(c, a.nunique(), (a == 0).mean(), a.min(), a.max(), a.mean())
    # Optional, expensive: independently inspect each family; do not fit models.
    # for builder in [lambda d, **kw: compute_specs(d, specs, **kw), build_decay_features]:
    #     a = builder(sub, source_mask=mask, source_until=cutoff).iloc[pos]
    #     print('constant', a.nunique().eq(1).sum(), 'allzero', a.eq(0).all().sum())
g = pd.Series(np.zeros(len(specs)), index=[spec_name(s) for s in specs]).sort_values(ascending=False)
print(g.head(40).index.tolist())
print(all(f['top_gain'] == list(g.head(5).index) for r in x['runs'] for f in r['folds']))
PY
```

Additional manifest verification actually performed: for each `metadata.source_tree.files` entry, SHA256 of local path was asserted equal to stored `sha256`; raw CSV hashes were compared with `metadata.source_hashes`.

## Execution limits and next decision

The full-family run completed train/select/valid in approximately 46/119/329 seconds respectively, then hit a 600-second command timeout during test after emitting test labels/history availability. A separate baseline pass hit a 180-second timeout during test after finishing earlier roles. A bounded **test-only baseline rerun succeeded in 27.7 seconds** and supplies the test feature table above. Host was memory/swap constrained and other threads ran tests concurrently; elapsed times are not clean performance benchmarks. No complete per-column generated/decay distribution dump survived the timeout, so this report deliberately limits their observed claims to aggregate zero/constant counts for completed roles. No model fitting or full experiment was started. No pytest process was started by this audit.

### Recommendations, in priority order

1. **Do not use v5 as confirmatory evidence of automatic feature selection quality.** Preserve it as a diagnosed degenerate/stale-history diagnostic; do not silently overwrite it.
2. **Introduce an explicit pre-training history/warm-up block**, excluded from supervised loss, so every supervised role can have a nonempty strict prefix. Alternatively use time-indexed training snapshots with the same historical-availability and refresh contract as evaluation. Prequential training is a materially different estimand and must be declared, not silently substituted.
3. Before a new model experiment, require nonempty train prefix, per-role label counts, nonconstant-feature counts, zero-gain diagnostics, nonfinite checks, and key-history coverage assertions. Test expanding folds specifically: expanding train must not imply an empty history source. Filter structural purchase and unavailable repeat-pair families for KION or label them inapplicable.
4. Gain selection must reject/report all-zero importance instead of calling arbitrary tied columns top features. Forward selection should require a predeclared strictly positive practical threshold, not accept zero-improvement additions. Save the full retained feature list and actual importances.
5. Run only a small protocol-validation pilot after agreement on the warm-up/refresh design: inspect distributions and split invariants first, then compare manual/auto/linear and strong popularity references under identical histories. Save a new artifact with code/input/configuration hashes; predeclare selection budgets and paired uncertainty. No such pilot was run here.
6. For production interpretation, validate raw last-watch/feedback availability semantics, report cold/warm and recent-history slices, and distinguish observed-watch AUC from sampled-unseen-item ranking. Any online-impact claim requires separate exposure-aware or randomized evidence.

**Decision:** repair/validate the history design before spending on a full rerun. The immediate blocker is not statistical power; it is an empty-source supervised training role and consequent non-identifiable feature selection.
