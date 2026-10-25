import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from recsys_corrected_artifact.feature_selectors import forward_select


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
    assert len(history) == 1
    assert history[0]["accepted"] is False
    assert history[0]["reason"] in {"nonpositive_improvement", "below_min_improvement"}


def test_forward_budget_skips_leading_constants_before_truncation():
    train = _frame(0).assign(
        user__a_zero=0,
        user__b_zero=0,
        user__z_signal=lambda frame: frame["binary_label"],
    )
    selected, added, history = forward_select(
        train,
        train,
        ["user__z_signal", "user__b_zero", "user__a_zero"],
        k=1,
        max_candidates=1,
    )

    assert selected == added == ["user__z_signal"]
    assert len(history) == 1
    assert history[0]["feature"] == "user__z_signal"
    assert "accepted" not in history[0]
    assert history[0]["improvement"] > 0
    assert history[0]["candidates_evaluated"] == 1


@pytest.mark.parametrize("max_candidates", [None, 1, 2, np.int32(1), np.int64(2)])
def test_forward_max_candidates_accepts_none_and_positive_integers(max_candidates):
    frame = _frame(0)
    selected, added, history = forward_select(
        frame,
        frame,
        ["feature_a"],
        max_candidates=max_candidates,
        k=0,
    )

    assert selected == added == history == []


@pytest.mark.parametrize(
    "max_candidates",
    [
        True,
        False,
        np.bool_(True),
        0,
        -1,
        1.0,
        1.5,
        np.nan,
        np.inf,
        -np.inf,
        "1",
        object(),
    ],
)
def test_forward_max_candidates_rejects_non_positive_or_non_integer_values(max_candidates):
    frame = _frame(0)
    with pytest.raises(
        ValueError,
        match="max_candidates must be a positive integer or None",
    ):
        forward_select(
            frame,
            frame,
            ["feature_a"],
            max_candidates=max_candidates,
            k=0,
        )


def test_forward_selection_reports_budget_and_selection_interval():
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
    assert len(added) == sum(step.get("accepted", True) for step in history)
    assert history[0]["step"] == 1
    assert history[0]["feature"] in {"feature_a", "feature_b"}
    assert history[0]["candidates_evaluated"] == 1
    expected_fields = {
        "step",
        "feature",
        "score",
        "previous_score",
        "improvement",
        "candidates_evaluated",
        "selection_interval",
    }
    if history[0].get("accepted") is False:
        expected_fields |= {"accepted", "reason"}
    assert set(history[0]) == expected_fields
