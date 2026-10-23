import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from feature_selectors import forward_select


def _frame(offset, rows=30):
    values = np.arange(rows) + offset
    return pd.DataFrame(
        {
            "base": values % 3,
            "feature_a": values % 5,
            "feature_b": (values // 2) % 7,
            "binary_label": (values % 2).astype(int),
        }
    )


def test_forward_selection_minimum_improvement_is_measured_from_base():
    train = _frame(0)
    select = _frame(1)
    selected, added, history = forward_select(
        train,
        select,
        ["feature_a", "feature_b"],
        base_columns=["base"],
        k=2,
        min_improvement=2.0,
        random_state=4,
    )

    assert selected == ["base"]
    assert added == []
    assert history == []


    train = _frame(0)
    select = _frame(1)
    selected, added, history = forward_select(
        train,
        select,
        ["feature_b", "feature_a"],
        base_columns=["base"],
        k=2,
        max_candidates=1,
        selection_interval="fold_select",
    )

    assert selected[:1] == ["base"]
    assert len(added) == len(history)
    assert history[0]["step"] == 1
    assert history[0]["feature"] in {"feature_a", "feature_b"}
    assert history[0]["candidates_evaluated"] == 1
    assert set(history[0]) == {
        "step",
        "feature",
        "score",
        "previous_score",
        "improvement",
        "candidates_evaluated",
        "selection_interval",
    }
