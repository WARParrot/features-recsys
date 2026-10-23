import numpy as np
import pandas as pd
import pytest

from rolling_eval import _fold_features, run_rolling_frame


def _frame():
    rows = []
    for t in range(12):
        rows.append({
            "event_id": t, "user": 1, "item": t % 3, "timestamp": t,
            "binary_label": t % 2, "is_click": float(t % 2),
            "is_purchase": 0.0, "item_cat": t % 2, "user_segment": 0,
        })
    return pd.DataFrame(rows)


def test_role_frozen_features_use_prefix_before_each_role():
    frame = _frame()
    from data import rolling_time_splits
    ordered, folds = rolling_time_splits(frame, n_folds=1)
    parts = folds[0]
    baseline, _, _, _ = _fold_features(ordered, parts, "role_frozen")
    for role, positions in parts.items():
        start = int(ordered.iloc[positions[0]]["timestamp"])
        role_values = baseline.iloc[positions]
        assert (role_values["u_views_all"] <= int((ordered["timestamp"] < start).sum())).all()


def test_full_catalog_requires_explicit_first_seen():
    with pytest.raises(ValueError, match="static[_ ]catalog"):
        run_rolling_frame(_frame(), n_folds=1, candidate_catalog="full", item_catalog=None)
