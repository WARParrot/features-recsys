# Research result index

## Latest result: corrected v7 factorial run

- [`kion_10k_v7_final.json`](kion_10k_v7_final.json) is the current completed artifact: 10,000 users, 55,922 observed watch records, three seeds, three temporal folds, and four matched variants. All 12 cells and 36 fold evaluations passed the result, cohort, split, and candidate checks.
- [`kion_10k_v7_final.lock.json`](kion_10k_v7_final.lock.json) contains the initial configuration and data/runtime provenance. The final JSON also records how the continuation was assembled.
- Nine cells came from the locked Linux run at revision `647f393`; the three missing `snapshot/expanded` cells were computed on Windows at revision `30065af`. Both environments used Python 3.11.15, the same package versions, data hashes, seeds, splits, candidate protocol, and single-thread settings. Recorded cell time was 4:04:20 for the reused Linux cells and 17:05 for the Windows cells; the Windows continuation took 17:42 including startup and preflight. These are mixed-host timings, not a single-machine benchmark.

| History | Features | Manual AUC | Auto gain AUC | Auto - manual | Manual NDCG@10 | Auto NDCG@10 | Decayed popularity |
|---|---|---:|---:|---:|---:|---:|---:|
| Prequential | Base | .66892 | .66163 | -.00728 | .20500 | .26546 | .48542 |
| Prequential | Expanded | .66892 | .68088 | +.01196 | .20500 | .26776 | .48542 |
| Snapshot | Base | .66629 | .66029 | -.00600 | .19988 | .27413 | .48542 |
| Snapshot | Expanded | .66629 | .68241 | +.01612 | .19988 | .31507 | .48542 |

These are unweighted means over nine fold/seed results per variant, not pooled estimates. Snapshot-expanded improves on manual in the sampled ranking metric, but decayed popularity remains stronger. The result is offline evidence from observed watch records, not evidence about logged impressions, causal impact, or production lift.

## Historical v6 factorial run

- [`kion_10k_v6_factorial.json`](kion_10k_v6_factorial.json) is the completed pre-correction run on the same 10k cohort. It remains useful for comparison, but its numbers belong to the earlier source revision.
- [`kion_10k_v6_factorial.lock.json`](kion_10k_v6_factorial.lock.json) records the configuration and provenance used for that run.
- `kion_10k_v6_factorial.stderr.log` records 22,450.33 seconds of wall time and a 1,120,968 KiB peak RSS.
- `kion_1k_v6_factorial.json` is an exploratory pilot and should not be pooled with the 10k results.
- `kion_1k_v6_failed_smoke.json` records an intentionally retained smoke failure, not a successful experiment.
- `kion_50k_protocol_v5.json` is diagnostic only because its first training-history prefix was degenerate; see the [v5 audit](../audits/v5_validity.md).

The historical v6 snapshot-expanded result was .68194 AUC versus .66736 for manual, and .32869 versus .20979 on sampled NDCG@10. Decayed popularity remained higher at .48542. See the [full v6 report](../audits/v6_10k_results.md) for the original uncertainty analysis.

The factorial artifact has its own `kion_factorial_ablation_v1` schema, not the `compare_methods` schema. Validate its completed cells with `run_ablation.validate_result` and `assert_comparable`; `validate_protocol.py` applies to separate compare-method artifacts. Do not use format-schema compatibility as evidence of experiment comparability.

## Historical expanded KION run (2026-10-22)

Status: complete and JSON-validated. The earlier empty file was an interrupted attempt and must not be used. The non-empty legacy file is also not final evidence because it predates historical-catalog filtering and semantic/uncertainty corrections.

Input: official `data_original.zip`, not the truncated standalone interactions archive.

Locally verified full source: 5,476,251 rows; 962,179 users; 15,706 items; no duplicate user-item pairs. 828 missing watched_pct values are currently mapped to zero (an explicit target-label limitation). `last_watch_dt` is a last-watch date and feedback is pair-aggregated, not an event stream; point-in-time availability of total completion remains an unresolved assumption.

SHA-256 (extracted source files):

```
201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf  interactions.csv
37b2945240628a3f215e1fb5f79b4cff386d8988b21fa61e5b135aef4541182c  users.csv
027c3a0027b4fe4a8db9d5fa626ab40bab507c96aac81d462db0771eb2cb4a8e  items.csv
```

Selected cohort: 50,000 users, sample seed 42, all 285,343 rows from selected users. Training seeds: 42,43,44; two temporal folds. Forward search capped to first 15 enumerated candidates and 3 steps (a compute-budget comparator, not full search). The target is inclusive `watched_pct >= 50.0`; the pair-aggregated observed-watch rows are not impression/click labels, and missing percentages are mapped to zero.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u compare_methods.py \
  --kion-interactions datasets/kion/interactions.csv \
  --kion-users datasets/kion/users.csv --kion-items datasets/kion/items.csv \
  --kion-max-users 50000 --evaluation rolling --folds 2 --seeds 42,43,44 \
  --forward-max-candidates 15 --forward-steps 3 --json \
  > results/kion_50k_reproduced.json 2> results/kion_50k_reproduced.stderr.log
```

**Command boundary:** the command above, when run on current sources, is a **new current-code 50k extension, not historical v4 reproduction**. Historical replay requires its recorded source/environment snapshot. For the latest 10k factorial result use [REPORT reproduction instructions](../REPORT.md#6-reproduction); do not run this command and label its output v4.

Cost diagnostic: load ~42 seconds; feature construction for one fold ~432 seconds; three-configuration manual tuning ~25 seconds. These are measured local timings, not end-to-end totals. Earlier foreground runs hit 20-minute timeouts and produced no usable result. The replacement runs detached to avoid per-tool timeouts.

The canonical protocol-v4 run is `kion_50k_protocol_v4.json`: 50,000 selected users, 285,343 retained interactions, three seeds, two folds, 2,187 fold×seed ranking queries (345–378 per cell). Each cell samples 99 negatives per positive query from the 8,385-item selected-cohort historical catalog, excluding previously seen items and items first seen at/after the query; unavailable positives are skipped. The six cells record 213 skipped-positive occurrences in total (22–55 per cell; these are repeated evaluation cells, not unique source queries). The run completed in 1:58:19 with peak RSS 2,509,416 KiB. Its metadata includes `protocol_version: 4`, source-tree provenance, and per-cell paired user-cluster bootstrap intervals for query-weighted NDCG@10. It does **not** include `event_auc_uncertainty`: the event-level AUC bootstrap was added afterward and was not recomputed for this checked-in canonical run. New rolling outputs from the current code may emit `event_auc_uncertainty` for pooled event-level AUC; do not interpret that field as present in the canonical JSON. The canonical immutable provenance snapshot is code revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c` with source-tree hash `c77188b69f3d65af97879d48dcecc1b5554fb9b780cbabdb6f374a865903fe13`. `kion_50k_corrected.json` predates this candidate correction and is provenance only; the legacy 50k JSON is historical; the 10k corrected JSON remains protocol-validation evidence only. Do not use these artifacts to replace or pool with the canonical result.
