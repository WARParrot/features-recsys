import numpy as np
import pandas as pd
import pytest

from candidate_eval import candidate_features, score_candidate_sets
from causal_features import CausalIndex, build_baseline_features
from data import generate_synthetic


def test_causal_features_are_invariant_to_appending_future_rows():
    frame = generate_synthetic(n_events=300, seed=11)
    prefix = frame.iloc[:180].copy()
    prefix_features = build_baseline_features(prefix)
    full_features = build_baseline_features(frame)
    np.testing.assert_allclose(
        prefix_features.to_numpy(dtype=float),
        full_features.iloc[: len(prefix)].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_causal_index_rejects_duplicate_event_ids():
    frame = pd.DataFrame(
        {"event_id": [1, 1], "timestamp": [1, 2], "user": [1, 1]}
    )
    with pytest.raises(ValueError, match="unique"):
        CausalIndex(frame, ["user"])


def test_candidate_features_require_query_ids():
    full = pd.DataFrame(
        {"event_id": [0], "user": [1], "item": [2], "timestamp": [1], "item_cat": [0]}
    )
    with pytest.raises(KeyError, match="query_id"):
        candidate_features(full, full)


def test_ranking_rejects_non_finite_scores():
    candidates = pd.DataFrame({"query_id": ["q"], "label": [1]})
    features = pd.DataFrame({"score": [np.nan]})

    class Model:
        def predict_proba(self, _frame):
            return np.array([[0.5, np.nan]])

    with pytest.raises(ValueError, match="finite"):
        score_candidate_sets(Model(), candidates, features, ["score"])
