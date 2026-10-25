# Карта репозитория

## С чего начать

1. [`../README.md`](../README.md) — основной русскоязычный отчёт проекта.
2. [`reports/REPORT.md`](reports/REPORT.md) — подробный технический отчёт и ограничения интерпретации.
3. [`../results/README.md`](../results/README.md) — индекс числовых результатов и их статус.
4. [`audits/`](audits/) — аудиты исторических запусков.

## Разделы

### Код

Производственный код находится в [`../src/recsys_corrected_artifact/`](../src/recsys_corrected_artifact/) и устанавливается как пакет `recsys_corrected_artifact`.

- `src/recsys_corrected_artifact/compare_methods.py` — CLI для отдельных сравнений и smoke-запусков;
- `src/recsys_corrected_artifact/run_ablation.py` — locked factorial experiment;
- `src/recsys_corrected_artifact/rolling_eval.py` — временная оценка;
- `src/recsys_corrected_artifact/causal_features.py`, `src/recsys_corrected_artifact/history.py`, `src/recsys_corrected_artifact/data.py` — временные границы и point-in-time признаки;
- `src/recsys_corrected_artifact/auto_features.py`, `src/recsys_corrected_artifact/feature_selectors.py` — генерация и отбор признаков;
- `src/recsys_corrected_artifact/pipeline.py`, `src/recsys_corrected_artifact/kion_data.py`, `src/recsys_corrected_artifact/mind_data.py` — загрузка данных и сценарии экспериментов;
- `src/recsys_corrected_artifact/models.py`, `src/recsys_corrected_artifact/linear_model.py`, `src/recsys_corrected_artifact/metrics.py` — модели и метрики;
- `src/recsys_corrected_artifact/validate_protocol.py` — проверка JSON-артефактов.

Запуск модулей выполняется из корня репозитория:

```bash
python -m recsys_corrected_artifact.compare_methods --help
python -m recsys_corrected_artifact.run_ablation --help
python -m recsys_corrected_artifact.validate_protocol --help
```

### Документы

- `reports/` — подробные отчёты и описание модульного pipeline;
- `audits/` — аудиты v5, v6 и pilot;
- `plans/` — планы экспериментов;
- `history/` — исторические synthetic smoke результаты;
- `status/` — журнал исправлений и remediation status.

### Результаты

- [`../results/README.md`](../results/README.md) — индекс артефактов;
- `../results/current/` — завершённый актуальный v7 JSON и lock-файл;
- [`../results/checkpoints/`](../results/checkpoints/) — сохранённый legacy checkpoint; его исходный lock относится к прежней структуре исходников и не подходит для продолжения из текущей версии.
- `../results/archive/` — исторические v4–v6 и старые validation-артефакты.

### Проверки и утилиты

- `../tests/` — unit, integration и protocol tests;
- `../tools/` — утилиты продолжения legacy checkpoint; текущий пакетный lock-файл может исполняться через CLI без этого скрипта;
- `../.github/workflows/` — CI.

Данные KION находятся в `../datasets/` локально и исключены из Git. Их хеши и требования к версии Python указаны в корневом README и lock-файлах.
