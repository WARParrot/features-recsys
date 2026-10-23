# Historical Synthetic Smoke Results

These values are retained for provenance only. They predate the snapshot-mask correction and must not be used as current evidence.

Command: `python compare_methods.py --seeds 42,43 --n-events 10000`

| Metric | Mean | Std |
|---|---:|---:|
| Manual selection AUC | 0.61110 | 0.01466 |
| Auto gain selection AUC | 0.59690 | 0.00892 |
| Auto forward selection AUC | 0.62663 | 0.00420 |
| Decay selection AUC | 0.63305 | 0.01363 |
| Logistic selection AUC | 0.65159 | 0.00625 |
| Manual test AUC | 0.73153 | 0.00518 |
| Logistic test AUC | 0.75688 | 0.00311 |
| Selection-winner test AUC | 0.75409 | 0.01799 |
| Manual sampled NDCG@10 | 0.41894 | 0.16044 |
| Linear sampled NDCG@10 | 0.36632 | 0.01677 |
| Selected sampled NDCG@10 | 0.34442 | 0.04770 |
| Manual sampled Recall@10 | 0.68024 | 0.15935 |
| Linear sampled Recall@10 | 0.62444 | 0.00451 |
| Selected sampled Recall@10 | 0.63545 | 0.04928 |
