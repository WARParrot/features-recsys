"""Strict-prior warmup/snapshot protocol regression tests (no model tuning)."""
import numpy as np
import pandas as pd
import pytest

from data import rolling_time_splits
from rolling_eval import _fold_features, _history_blocks, run_rolling_frame, run_rolling_many


def frame(n=100):
    # Duplicate timestamps, shuffled non-positional index and noncontiguous IDs.
    t = np.repeat(np.arange(n), 2)
    return pd.DataFrame({
        "event_id": np.arange(len(t)) * 3 + 7, "timestamp": t,
        "user": 1, "item": 2, "item_cat": 0, "user_segment": 0,
        "binary_label": t % 2, "is_click": (t % 2).astype(float),
        "is_purchase": (t % 3 == 0).astype(float),
    }, index=np.arange(len(t)) * 5).sample(frac=1, random_state=1)


def test_warmup_atomic_disjoint_and_full_position_coverage():
    log, folds = rolling_time_splits(frame(), n_folds=3, warmup_fraction=.2)
    assert log.index.equals(pd.RangeIndex(len(log)))
    for parts in folds:
        joined = np.concatenate(list(parts.values()))
        np.testing.assert_array_equal(joined, np.arange(parts["test"][-1] + 1))
        assert len(np.unique(joined)) == len(joined)
        for left, right in zip(list(parts.values())[:-1], list(parts.values())[1:]):
            assert log.iloc[left].timestamp.max() < log.iloc[right].timestamp.min()


def test_informative_training_history_and_snapshot_consistency():
    log, folds = rolling_time_splits(frame(), n_folds=1, warmup_fraction=.2)
    parts = folds[0]
    baseline, auto, decay, _ = _fold_features(log, parts, "snapshot")
    assert len(baseline) == len(log)  # no positional drift from excluded warmup
    np.testing.assert_array_equal(baseline.iloc[parts["train"]].event_id,
                                  log.iloc[parts["train"]].event_id)
    assert (baseline.iloc[parts["train"]].u_views_all > 0).all()
    assert baseline.iloc[parts["train"]].u_views_all.nunique() > 1
    for positions in _history_blocks(log, parts, "snapshot").values():
        for p in positions:
            expected = p[0]  # single user, count of rows in strict prefix
            assert (baseline.iloc[p].u_views_all == expected).all()
            assert (auto.iloc[p]["user__count_rows__all"] == expected).all()
    for features in (auto, decay):
        assert np.isfinite(features.iloc[np.concatenate(list(parts.values())[1:])]).all().all()


def test_future_and_intrablock_labels_never_change_block_features():
    log, folds = rolling_time_splits(frame(), n_folds=1, warmup_fraction=.2)
    parts = folds[0]
    p = _history_blocks(log, parts, "snapshot")["train"][1]
    changed = log.copy()
    future = changed.index >= p[0]
    changed.loc[future, ["is_click", "is_purchase", "binary_label"]] = 1
    original = _fold_features(log, parts, "snapshot")
    perturbed = _fold_features(changed, parts, "snapshot")
    for before, after in zip(original[:3], perturbed[:3]):
        # Labels are raw data, not features. All computed columns stay fixed.
        cols = before.columns.difference(log.columns)
        pd.testing.assert_frame_equal(before.loc[p, cols], after.loc[p, cols])
    # Test feedback is never admitted even at subsequent test timestamps.
    test_changed = log.copy()
    test_changed.loc[parts["test"], ["is_click", "is_purchase"]] = 123
    test_features = _fold_features(test_changed, parts, "snapshot")
    for before, after in zip(original[:3], test_features[:3]):
        cols = before.columns.difference(log.columns)
        pd.testing.assert_frame_equal(before.loc[parts["test"], cols], after.loc[parts["test"], cols])


def test_spaces_share_rows_and_base_features_and_cadence():
    log, folds = rolling_time_splits(frame(), n_folds=1, warmup_fraction=.2)
    parts = folds[0]
    base = _fold_features(log, parts, "snapshot", feature_space="base")
    expanded = _fold_features(log, parts, "snapshot", feature_space="expanded")
    pd.testing.assert_frame_equal(base[0], expanded[0])
    pd.testing.assert_frame_equal(base[1], expanded[1][base[1].columns])
    assert len(expanded[3]) > len(base[3])
    blocks = _history_blocks(log, parts, "snapshot")
    assert len(blocks["train"]) == 4
    assert len(blocks["select"]) == 4
    assert len(blocks["valid"]) == 4
    assert len(blocks["test"]) == 1


def test_large_identifiers_are_not_coerced_by_warmup_reindexing():
    source = frame(30)
    source["event_id"] += 2 ** 60
    log, folds = rolling_time_splits(source, n_folds=1, warmup_fraction=.2)
    baseline, _, _, _ = _fold_features(log, folds[0], "snapshot", training_blocks=2)
    pd.testing.assert_series_equal(baseline.event_id, log.event_id)
    pd.testing.assert_series_equal(baseline.user, log.user)


def test_legacy_zero_warmup_keeps_v5_degenerate_history():
    log, folds = rolling_time_splits(frame(), n_folds=1)
    assert "warmup" not in folds[0]
    baseline, _, _, _ = _fold_features(log, folds[0], "role_frozen")
    assert (baseline.iloc[folds[0]["train"]].u_views_all == 0).all()


@pytest.mark.parametrize("fraction", [-.1, 1, float("nan"), float("inf")])
def test_invalid_warmup(fraction):
    with pytest.raises(ValueError, match="warmup_fraction"):
        rolling_time_splits(frame(), warmup_fraction=fraction)


def test_sparse_blocks_and_missing_warmup_fail_explicitly():
    log, folds = rolling_time_splits(frame(8), n_folds=1, warmup_fraction=.2)
    with pytest.raises(ValueError, match="training_blocks"):
        _fold_features(log, folds[0], "snapshot", training_blocks=4)
    log, folds = rolling_time_splits(frame(), n_folds=1)
    with pytest.raises(ValueError, match="warmup"):
        _fold_features(log, folds[0], "snapshot")


def test_exact_split_fingerprints_and_predictor_only_diagnostics():
    from rolling_eval import _fold_diagnostic_payload
    log, folds = rolling_time_splits(frame(), n_folds=1, warmup_fraction=.2)
    parts = folds[0]
    full = log.assign(predictor=1.0, varying_predictor=np.arange(len(log)))
    payload = _fold_diagnostic_payload(log, parts, full,
                                       ["predictor", "varying_predictor", "predictor"])
    assert payload["diagnostics_feature_scope"].endswith("predictors_only")
    assert set(payload["split_fingerprint"]) == set(parts)
    for role in ("train", "select", "valid", "test"):
        stats = payload["diagnostics"][role]["features"]
        assert set(stats["columns"]) == {"predictor", "varying_predictor"}
        assert stats["constant_fraction"] == .5
    # Identity hash ignores pandas index labels, but includes every event ID
    # in order: even swapping same-timestamp IDs must be detected.
    changed = log.copy()
    p = parts["test"][:2]
    changed.loc[p, "event_id"] = changed.loc[p, "event_id"].to_numpy()[::-1]
    other = _fold_diagnostic_payload(changed, parts, full, ["predictor"])
    assert payload["split_fingerprint"]["test"] != other["split_fingerprint"]["test"]
    for role in ("warmup", "train", "select", "valid"):
        assert payload["split_fingerprint"][role] == other["split_fingerprint"][role]
    relabeled = log.set_axis(np.arange(len(log)) * 7)
    relabeled_full = full.set_axis(relabeled.index)
    same = _fold_diagnostic_payload(relabeled, parts, relabeled_full, ["predictor"])
    assert same["split_fingerprint"] == payload["split_fingerprint"]


def test_api_forwards_identical_splits_without_fitting(monkeypatch):
    import rolling_eval
    seen = []

    def fake(log, parts, seed, fold, dataset, **kwargs):
        seen.append((parts, kwargs))
        return {"fold": fold, "manual_test_auc": .5}

    monkeypatch.setattr(rolling_eval, "_fold_result", fake)
    for mode in ("snapshot", "prequential"):
        for space in ("base", "expanded"):
            result = run_rolling_frame(frame(), n_folds=1, history_mode=mode,
                                       warmup_fraction=.2, feature_space=space)
            assert result["rolling_protocol_version"] == 6
    for parts, _ in seen[1:]:
        for role in parts:
            np.testing.assert_array_equal(parts[role], seen[0][0][role])
    legacy = run_rolling_frame(frame(), n_folds=1, history_mode="role_frozen", warmup_fraction=0)
    assert legacy["rolling_protocol_version"] == 5
    runs = run_rolling_many(frame(), seeds=(1, 2), n_folds=1, feature_space="base")
    assert len(runs) == 2
    assert all(kwargs["feature_space"] == "base" for _, kwargs in seen[-2:])
