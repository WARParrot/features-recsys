# Исправленный модульный RecSys feature-search artifact

Актуальный модульный pipeline описан в [MODULAR_COMPARISON.md](MODULAR_COMPARISON.md).

## Решение по датасету

**Основной датасет выбран: KION.**

KION соответствует исходной постановке про user-item interactions, timestamp,
watch feedback, user demographics и item metadata. MIND остаётся отдельным
logged-impression benchmark для проверки ranking на реальных candidate sets; KION sampled metrics не являются таким ranking benchmark.

KION используется для основного исследования автоматического feature search.
Цель — `watched_pct >= 50.0` (порог включительный): это бинарная метка на
наблюдаемой агрегированной user-item watch-записи; пропуск `watched_pct`
адаптер относит к нулевому классу. `last_watch_dt` имеет дневную точность и
отражает последнюю дату просмотра, а не порядок событий внутри дня. Поэтому
KION event-level AUC — классификация high-watch против lower-watch наблюдаемых
пар, не CTR, не next-event и не prediction по impression-логам.

KION sampled ranking сравнивает held-out high-watch positive с 99 случайными
unseen negatives (без возвращения) из selected-cohort historical catalog:
item должен быть впервые замечен строго до query timestamp и не встречаться
ранее у этого user. Положительные запросы с item, ещё не доступным исторически,
исключаются. Это query-weighted offline sampled discrimination, не logged
exposure evaluation и не production candidate set. Для ранжирования внутри реально показанного набора
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

For the current warmup/block-snapshot protocol and candidate sensitivity controls (synthetic tooling example, not KION research evidence):

```bash
python compare_methods.py --evaluation rolling --folds 3 --seeds 42,43,44 \
  --history-mode snapshot --warmup-fraction .2 --training-blocks 4 \
  --candidate-n-negatives 99 \
  --candidate-max-queries 400 --candidate-sampling uniform \
  --candidate-catalog cohort --n-events 30000 --json
```

The default `rolling` mode uses expanding temporal folds. The `single_split` mode remains an exploratory compatibility path because forward selection and method choice reuse one selection interval. Current rolling runs default to `history_mode=snapshot`, reserving a history-only warmup and training on timestamp-atomic historical blocks. The old `role_frozen` design is retained explicitly for diagnostic replay; see the validity warning below. `prequential` is also a valid factorial comparator with the same explicit positive warmup; use `--warmup-fraction 0` for legacy boundaries. Strict selection guards can reject a historically degenerate run rather than reproduce arbitrary selections. The primary comparison is `auto_gain` versus `manual`; `selected` is descriptive only. Ranking means are query-weighted; in the canonical JSON, `ranking_ndcg10_uncertainty` contains separate per-fold/seed user-cluster bootstrap intervals for sampled NDCG@10, not event-level AUC uncertainty. The canonical protocol-v4 JSON predates the event-level AUC bootstrap change: it has no `event_auc_uncertainty`, and that uncertainty was not recomputed for this checked-in canonical run. New rolling outputs from the current code may contain `event_auc_uncertainty` for pooled event-level AUC, but those outputs are new evaluations and are not fields of the canonical artifact. The canonical protocol-v4 JSON applies the historical-availability filter and records `skipped_positive_queries`; older ranking scores from before this filter are superseded. Even protocol-v4 KION ranking remains a sampled-discrimination diagnostic, not evidence about logged impressions or production ranking.

The default combined protocol prefixes query IDs by source split, then performs the same timestamp-group chronological split and is reported as `combined_temporal`. The `official_train_dev` protocol keeps train and dev separate: train is divided chronologically into internal train/select/valid roles, and dev is held out for final evaluation. Neither mode should be confused with an official leaderboard submission.

The lock was verified with Python 3.11.15; use a compatible Python 3.11 environment for reproducible runs. The KION adapter cannot recover impression exposure or first-decision timestamps from pair-aggregated `last_watch_dt`; all conclusions remain retrospective and protocol-conditional.


- ручные temporal features + tuned LightGBM;
- template-based auto features + gain top-K;
- template-based auto features + validation forward selection;
- exponential-decay features вместо только hard windows;
- regularized Logistic Regression как независимая модельная альтернатива LightGBM;
- sampled-candidate NDCG/MRR/Recall в synthetic diagnostics;
- sampled unseen-item discrimination NDCG/MRR/Recall для KION и logged-impression NDCG/MRR/Recall для MIND в дополнение к event-level AUC;
- cumulative, decayed, category-popularity, and prior user-item positive-watch-rate baselines for KION (MIND retains click terminology).

## Latest completed research: 10k factorial follow-up

The completed [10k-user study](audits/v6_10k_results.md) compares prequential/snapshot history × base/expanded grammar on exactly matched splits and candidates, with three seeds and three temporal folds. All 12 cells completed. [Result JSON](results/kion_10k_v6_factorial.json), [configuration lock](results/kion_10k_v6_factorial.lock.json), [completion log](results/kion_10k_v6_factorial.stderr.log).

| History | Grammar | Manual AUC | Auto gain AUC | Auto − manual |
|---|---|---:|---:|---:|
| Prequential | Base | .66936 | .66161 | -.00775 |
| Prequential | Expanded | .66936 | .68087 | +.01150 |
| Snapshot | Base | .66736 | .66029 | -.00707 |
| Snapshot | Expanded | .66736 | .68194 | +.01458 |

These are means over nine fold×seed estimates per variant, not pooled metrics or confidence intervals. Snapshot-expanded AUC support is concentrated in fold0: only 3/9 per-cell paired CIs exclude zero. Its sampled NDCG@10 is .32869 versus manual .20979 (all nine delta CI lower bounds > .02), but decayed popularity is stronger at .48542. Thus **local expanded-feature gains are observed; time-robust AUC superiority and production ranking impact are not established**. Run: 55,922 records, 6:14:10 wall, ~1.07 GiB peak RSS. The 1k pilot below remains separate evidence, not a contradiction to conceal or a result to pool.

## Current validity status

**Protocol-v5 is diagnostic only, not evidence of learned feature-search superiority.** Its role-frozen train prefix was empty: all 187 history features were zero in first-fold training and all six automatic selection AUCs were 0.5. See [independent audit](audits/v5_validity.md).

The current default is protocol-v6 `snapshot`, with a history-only warmup and historical training blocks. Explicit `role_frozen`/`prequential` with zero warmup retain legacy semantics for inspection, not endorsement. Use identical positive warmup fractions for factorial comparisons. Snapshot train/select/valid share a timestamp-group cadence; test remains one frozen role to match ranking candidates, so its longer freshness horizon is a disclosed limitation.

```bash
python run_ablation.py --dry-run --lock results/pilot.lock.json \
  --kion-interactions datasets/kion/interactions.csv \
  --kion-users datasets/kion/users.csv --kion-items datasets/kion/items.csv \
  --max-users 1000 --seeds 42 --folds 2 \
  --candidate-n-negatives 99 --candidate-max-queries 400 --smoke-users 1000 \
  --forward-max-candidates 15 --forward-steps 3
python run_ablation.py --execute-lock results/pilot.lock.json \
  --output results/pilot.json
```

Inspect the lock before execution. It binds input/source hashes, runtime settings and the factorial configuration. Do not change sources between locking and execution. Four cells compare prequential/snapshot × base/expanded on exactly matched split and candidate fingerprints. Smoke is only a tooling check; the 1k cohort is exploratory evidence, not the final generalization result. See [locked design](EXPERIMENT_PLAN_V6.md).

The completed [1k-user factorial pilot](audits/v6_pilot_results.md) does **not** support auto-gain superiority: snapshot-expanded AUC .62542 versus manual .62717; all four mean AUC deltas are negative. Its [artifact](results/kion_1k_v6_factorial.json) retains diagnostics, exact comparison fingerprints, costs and conditional intervals. This is an exploratory pilot, not a generalized negative result.

## Historical Results

The canonical historical result is [`results/kion_50k_protocol_v4.json`](results/kion_50k_protocol_v4.json): 50k selected users, 285,343 interactions, three seeds, two folds, and 2,187 fold×seed ranking queries. It is preserved as an immutable earlier-protocol artifact; current protocol-v6 factorial outputs are new evaluations and must not be mixed with it. Across six cells, test AUC is 0.65271 (manual) vs 0.65408 (`auto_gain`); sampled NDCG@10 is 0.12418 vs 0.34345, while decayed popularity is 0.52138. These are means across six fold/seed summaries, not pooled estimates; the canonical JSON has ranking uncertainty for NDCG@10 but no event-level AUC uncertainty. Its immutable provenance snapshot is code revision `6f8d386d8460be83cc061f5075a714ec32ac3f0c` and source-tree hash `c77188b69f3d65af97879d48dcecc1b5554fb9b780cbabdb6f374a865903fe13`. `kion_50k_corrected.json` is superseded pre-v4 provenance; `kion_10k_corrected.json` is protocol validation only; `kion_50k_legacy_pre_correction.json` and the synthetic smoke table in [historical-results.md](historical-results.md) are historical, non-comparable artifacts. See [REPORT.md](REPORT.md) for target semantics, estimands, uncertainty boundaries, and limitations.

```bash
python compare_methods.py --evaluation rolling --folds 3 --seeds 42,43,44 --n-events 30000 --json > rolling.json
```

The former smoke table is retained separately in [historical-results.md](historical-results.md) for provenance only. JSON output includes package versions, source split names, unknown-item metadata, candidate protocol metadata, and exact chronological split ranges.
