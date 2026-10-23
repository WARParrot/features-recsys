# Expanded KION run (2026-10-22)

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

Cost diagnostic: load ~42 seconds; feature construction for one fold ~432 seconds; three-configuration manual tuning ~25 seconds. These are measured local timings, not end-to-end totals. Earlier foreground runs hit 20-minute timeouts and produced no usable result. The replacement runs detached to avoid per-tool timeouts.

The canonical protocol-v4 run is `kion_50k_protocol_v4.json`: 50,000 selected users, 285,343 retained interactions, three seeds, two folds, 2,187 fold×seed ranking queries (345–378 per cell). Each cell samples 99 negatives per positive query from the 8,385-item selected-cohort historical catalog, excluding previously seen items and items first seen at/after the query; unavailable positives are skipped. The six cells record 213 skipped-positive occurrences in total (22–55 per cell; these are repeated evaluation cells, not unique source queries). The run completed in 1:58:19 with peak RSS 2,509,416 KiB. Its metadata includes `protocol_version: 4`, source-tree provenance, and per-cell paired user-cluster bootstrap intervals for query-weighted NDCG@10. It does **not** include `event_auc_uncertainty`: the event-level AUC bootstrap was added afterward and was not recomputed for this checked-in canonical run. New rolling outputs from the current code may emit `event_auc_uncertainty` for pooled event-level AUC; do not interpret that field as present in the canonical JSON. The canonical immutable provenance snapshot is code revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c` with source-tree hash `c77188b69f3d65af97879d48dcecc1b5554fb9b780cbabdb6f374a865903fe13`. `kion_50k_corrected.json` predates this candidate correction and is provenance only; the legacy 50k JSON is historical; the 10k corrected JSON remains protocol-validation evidence only. Do not use these artifacts to replace or pool with the canonical result.
