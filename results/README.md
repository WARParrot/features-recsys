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

Selected cohort: 50,000 users, sample seed 42, all 285,343 rows from selected users. Training seeds: 42,43,44; two temporal folds. Forward search capped to first 15 enumerated candidates and 3 steps (a compute-budget comparator, not full search).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u compare_methods.py \
  --kion-interactions datasets/kion/interactions.csv \
  --kion-users datasets/kion/users.csv --kion-items datasets/kion/items.csv \
  --kion-max-users 50000 --evaluation rolling --folds 2 --seeds 42,43,44 \
  --forward-max-candidates 15 --forward-steps 3 --json \
  > results/kion_50k_corrected.json 2> results/kion_50k_corrected.stderr.log
```

Cost diagnostic: load ~42 seconds; feature construction for one fold ~432 seconds; three-configuration manual tuning ~25 seconds. These are measured local timings, not end-to-end totals. Earlier foreground runs hit 20-minute timeouts and produced no usable result. The replacement runs detached to avoid per-tool timeouts.

The corrected pre-candidate-fix run is `kion_50k_corrected.json`; it is retained for provenance only because it includes positive queries outside the historical catalog. The current protocol-v4 run is `kion_50k_protocol_v4.json`: 50,000 selected users, 285,343 retained interactions, three seeds, two folds, 2,187 valid ranking queries, 186 queries excluded because the positive item was not historically available, completed in 1:58:19 with peak RSS 2.39 GiB. Its metadata includes `protocol_version: 4` and a source-tree hash. The legacy 50k JSON is historical only. The 10k corrected JSON remains a protocol-validation artifact. New rolling runs emit `ranking_ndcg10_uncertainty` with query-weighted user-cluster bootstrap intervals that preserve each user's query count; older JSON must not be interpreted as having those intervals.
