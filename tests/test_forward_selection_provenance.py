import hashlib
import json

import numpy as np
import pandas as pd

from recsys_corrected_artifact.feature_selectors import budgeted_candidate_order, forward_select
import recsys_corrected_artifact.rolling_eval as rolling_eval


def _selection_frame():
    labels = np.array([0, 1, 0, 1, 0, 1])
    return pd.DataFrame(
        {
            "binary_label": labels,
            "family_a__train_constant": [0] * len(labels),
            "family_b__signal": labels,
            "family_c__later": np.arange(len(labels)) % 3,
        }
    )


def test_forward_select_metadata_tracks_budgeted_input_and_train_pool():
    train = _selection_frame()
    select = train.copy()
    select["family_a__train_constant"] = [0, 1, 0, 1, 0, 1]
    candidates = [
        "family_b__signal",
        "family_a__train_constant",
        "family_c__later",
    ]

    legacy_result = forward_select(train, select, candidates, k=0, max_candidates=1)
    selected, added, history, metadata = forward_select(
        train, select, candidates, k=0, max_candidates=np.int64(1), return_metadata=True
    )

    ordered_input = budgeted_candidate_order(candidates)
    expected_input_hash = hashlib.sha256(
        json.dumps(ordered_input, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert len(legacy_result) == 3
    assert (selected, added, history) == legacy_result
    assert ordered_input == [
        "family_a__train_constant",
        "family_b__signal",
        "family_c__later",
    ]
    assert metadata == {
        "effective_pool": ["family_b__signal"],
        "effective_pool_sha256": hashlib.sha256(
            b'["family_b__signal"]'
        ).hexdigest(),
        "ordered_candidate_input_sha256": expected_input_hash,
        "ordered_candidate_input_count": 3,
        "train_constant_candidates_removed": 1,
        "max_candidates": 1,
        "effective_pool_size": 1,
    }
    assert metadata["effective_pool_size"] == len(metadata["effective_pool"])
    assert metadata["effective_pool_size"] <= metadata["max_candidates"]
    json.dumps(metadata, allow_nan=False)


def test_rolling_fold_persists_serializable_forward_provenance(monkeypatch):
    row_count = 20
    labels = np.arange(row_count) % 2
    log = pd.DataFrame(
        {
            "event_id": np.arange(row_count),
            "timestamp": np.arange(row_count),
            "binary_label": labels,
        }
    )
    parts = {
        "train": np.arange(0, 8),
        "select": np.arange(8, 12),
        "valid": np.arange(12, 16),
        "test": np.arange(16, 20),
    }
    baseline = pd.DataFrame(
        {
            "base": labels,
            "binary_label": labels,
            "timestamp": log["timestamp"],
            "event_id": log["event_id"],
        },
        index=log.index,
    )
    auto = pd.DataFrame(
        {
            "family_a__train_constant": [0] * 8 + list(labels[8:]),
            "family_b__signal": labels,
            "family_c__later": np.arange(row_count) % 3,
        },
        index=log.index,
    )
    decay = pd.DataFrame(index=log.index)
    monkeypatch.setattr(
        rolling_eval, "_fold_features", lambda *args, **kwargs: (baseline, auto, decay, [])
    )
    monkeypatch.setattr(rolling_eval, "BASELINE_FEATURES", ["base"])
    monkeypatch.setattr(rolling_eval, "CATEGORICAL_FEATURES", [])
    monkeypatch.setattr(rolling_eval, "tune_model", lambda *args, **kwargs: (0.5, {}))
    monkeypatch.setattr(
        rolling_eval,
        "gain_top_k",
        lambda *args, **kwargs: (auto.columns.tolist(), pd.Series(dtype=float)),
    )
    monkeypatch.setattr(rolling_eval, "fit_lgbm", lambda *args, **kwargs: object())
    monkeypatch.setattr(rolling_eval, "fit_logistic", lambda *args, **kwargs: object())
    monkeypatch.setattr(rolling_eval, "auc", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(rolling_eval, "logistic_auc", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(rolling_eval, "_fold_diagnostic_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rolling_eval,
        "query_weighted_cluster_bootstrap_mean_ci",
        lambda *args, **kwargs: [0.0, 1.0],
    )
    monkeypatch.setattr(
        rolling_eval,
        "paired_query_weighted_cluster_bootstrap_mean_ci",
        lambda *args, **kwargs: [0.0, 1.0],
    )

    def fake_ranking_result(*args, **kwargs):
        metrics = {
            "ndcg10": 0.5,
            "mrr10": 0.5,
            "recall10": 0.5,
            "query_metrics": {"ndcg10": [0.5], "users": [1]},
        }
        rankings = {name: dict(metrics) for name in args[3]}
        ranking_log = pd.DataFrame({"query_id": [1], "item": [1], "label": [1]})
        return rankings, ranking_log

    monkeypatch.setattr(rolling_eval, "_ranking_result", fake_ranking_result)

    result = rolling_eval._fold_result(
        log,
        parts,
        seed=3,
        fold=0,
        dataset="synthetic",
        forward_max_candidates=1,
        forward_steps=0,
        history_mode="prequential",
    )

    provenance = result["forward_selection_provenance"]
    assert provenance["effective_pool"] == ["family_b__signal"]
    assert provenance["ordered_candidate_input_count"] == 3
    assert provenance["train_constant_candidates_removed"] == 1
    assert provenance["max_candidates"] == provenance["effective_pool_size"] == 1
    assert json.loads(json.dumps(result, allow_nan=False))[
        "forward_selection_provenance"
    ] == provenance
