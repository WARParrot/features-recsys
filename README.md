# Исправленный модульный RecSys feature-search artifact

Актуальный модульный pipeline описан в [MODULAR_COMPARISON.md](MODULAR_COMPARISON.md).

## Решение по датасету

**Основной датасет выбран: KION.**

KION соответствует исходной постановке про user-item interactions, timestamp,
watch feedback, user demographics и item metadata. MIND остаётся отдельным
logged-impression benchmark для проверки ranking на реальных candidate sets; KION sampled metrics не являются таким ranking benchmark.

KION используется для основного исследования автоматического feature search:
целью является high-watch event (`watched_pct >= 50%`), а ranking выполняется
против historical-catalog sampled unseen items. В KION нет показов/clicks: это
offline positive-watch discrimination, а не CTR или causal product ranking. Это sampled unseen-item
**discrimination** protocol, а не logged exposure evaluation и не доказательство
production ranking lift. Для ранжирования внутри реально показанного набора
MIND `behaviors.tsv`
содержит impression ID, anonymous user, время, историю кликов и список показанных
news с labels `0/1`; `news.tsv` содержит category/subcategory. Поэтому MIND
позволяет оценивать `NDCG/MRR/Recall` на logged impression candidates без
искусственного добавления unseen-item negatives. Официальное описание: 
<https://raw.githubusercontent.com/msnews/msnews.github.io/master/assets/doc/introduction.md>.

Основной KION experiment описан в [REPORT.md](REPORT.md). MIND используется только
для non-commercial research по условиям набора и остаётся дополнительным
logged-impression benchmark. Текстовые поля KION/MIND не подаются в первую модель:
задача artifact — сравнить ручные и автоматически найденные point-in-time aggregate
features, а не content/history encoder.

**KuaiRand-Pure остаётся дополнительным causal/temporal benchmark**, потому что
содержит timestamp, random exposure и много feedback-сигналов, но не даёт полного
logged candidate set для каждого запроса. Поэтому он хорошо проверяет feature
causality и temporal drift, но хуже подходит для основного ranking claim.

## Формат MIND-эксперимента

```bash
# После распаковки MIND-small в mind_data/ с train/ и dev/:
python compare_methods.py --mind-root mind_data --seeds 42 --json > mind_comparison.json

# Official-style train/dev evaluation, with train-only internal selection:
python compare_methods.py --mind-root mind_data --evaluation single_split --mind-protocol official_train_dev --json
```


`mind_data.py` разворачивает `behaviors.tsv` в одну строку на candidate:
`query_id`, `user`, `item`, `timestamp`, `label`, `item_cat`. Путь MIND использует
`score_logged_impressions`; synthetic-путь и его sampled-ranking диагностика
остаются совместимыми и не смешиваются с impression-level результатами.

Исправленные causal-гарантии:

- события одного `timestamp` атомарны и не видят друг друга;
- split проводится только между timestamp-группами;
- `source_mask`/`source_until` поддерживают snapshot-протокол;
- history replay сохраняет контекст предыдущих временных групп;
- feature indexes и границы окон переиспользуются между specs.

## Setup

The artifact supports Python 3.11. From this directory:

KION run:
```bash
python compare_methods.py --kion-interactions /path/interactions.csv.zip \
  --kion-users /path/users_en.csv --kion-items /path/items_en.csv \
  --kion-max-users 10000 --evaluation rolling --folds 2 --seeds 42,43 --json
```


```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

The installed `recsys-compare` command and the compatibility command
`python compare_methods.py` execute the same CLI. The latter remains available
for source-tree use and existing scripts.

Continuous integration runs compilation, the complete test suite, and a small
single-split smoke experiment. A clean environment is required before treating
experiment output as valid evidence.

## Result interpretation

Every report should retain the JSON metadata, dependency versions, seed list,
data size or source split names, candidate protocol, and chronological split
metadata. A result is not comparable unless those fields match. Synthetic
sampled-candidate metrics are diagnostics; only MIND logged-impression metrics
support an impression-level ranking claim.


For stronger claims, use expanding rolling temporal folds:

```bash
python compare_methods.py --evaluation rolling --folds 3 --seeds 42,43,44 --n-events 30000 --json
```

The default `rolling` mode uses expanding temporal folds. The `single_split` mode remains available as an exploratory compatibility path because forward selection and method choice reuse one selection interval. The primary comparison is always `auto_gain` versus `manual`; `selected` is descriptive only and must not be treated as an unbiased primary endpoint. Results are query-weighted ranking means; cluster bootstrap resampling preserves each user's query count. In the KION historical-catalog protocol, positive queries whose item was not observed before the query are excluded from ranking evaluation; the JSON records `skipped_positive_queries`. Such ranking results must be recomputed after this protocol correction and are exploratory until the candidate protocol is locked.

The default combined protocol prefixes query IDs by source split, then performs the same timestamp-group chronological split and is reported as `combined_temporal`. The `official_train_dev` protocol keeps train and dev separate: train is divided chronologically into internal train/select/valid roles, and dev is held out for final evaluation. Neither mode should be confused with an official leaderboard submission.

The lock was verified with Python 3.11.15; use a compatible Python 3.11 environment for reproducible runs.


- ручные temporal features + tuned LightGBM;
- template-based auto features + gain top-K;
- template-based auto features + validation forward selection;
- exponential-decay features вместо только hard windows;
- regularized Logistic Regression как независимая модельная альтернатива LightGBM;
- sampled-candidate NDCG/MRR/Recall в synthetic diagnostics;
- sampled unseen-item discrimination NDCG/MRR/Recall для KION и logged-impression NDCG/MRR/Recall для MIND в дополнение к event-level AUC;
- cumulative, decayed, category-popularity, and prior user-item positive-watch-rate baselines for KION (MIND retains click terminology).

## Current Results

The current protocol-v4 result is `results/kion_50k_protocol_v4.json` (three seeds, 50k selected users, two folds, with invalid positive queries excluded). `results/kion_50k_corrected.json` is pre-candidate-fix provenance only. The 10k corrected JSON is retained as protocol-validation evidence; the legacy 50k JSON is historical only. Regenerate reports in the pinned environment before making comparisons:

```bash
python compare_methods.py --evaluation rolling --folds 3 --seeds 42,43,44 --n-events 30000 --json > rolling.json
```

The former smoke table is retained separately in [historical-results.md](historical-results.md) for provenance only. JSON output includes package versions, source split names, unknown-item metadata, candidate protocol metadata, and exact chronological split ranges.
