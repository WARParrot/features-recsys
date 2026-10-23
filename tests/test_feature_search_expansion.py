import numpy as np
import pandas as pd

from auto_features import compute_specs, enumerate_specs, spec_name
from causal_features import CausalIndex
from feature_selectors import budgeted_candidate_order


def _frame():
    return pd.DataFrame({
        "event_id": [0, 1, 2, 3, 4],
        "user": [1, 1, 1, 2, 1],
        "item": [10, 10, 11, 10, 10],
        "timestamp": [0, 5, 10, 10, 20],
        "item_cat": [3, 3, 4, 3, 3],
        "user_segment": [1, 1, 1, 2, 1],
        "is_click": [1.0, 0.0, 1.0, 1.0, 0.0],
        "is_purchase": [0.0, 0.0, 1.0, 0.0, 0.0],
    })


def test_expanded_specs_have_traceable_names_and_candidate_compatible_values():
    frame = _frame()
    specs = enumerate_specs(feedback_target="is_click")
    derived = [spec for spec in specs if spec.get("transform")]
    assert {spec["transform"] for spec in derived} == {"rate", "contrast", "recency"}
    values = compute_specs(frame, derived, source_mask=np.ones(len(frame), dtype=bool))
    assert len(values) == len(frame)
    assert values.columns.is_unique
    assert np.isfinite(values.to_numpy()).all()
    assert all(spec_name(spec) in values for spec in derived)


def test_derived_features_are_point_in_time_and_equal_timestamp_atomic():
    frame = _frame()
    specs = [
        {"transform": "rate", "keys": ("user",), "window": None,
         "target": "is_click", "denominator": "count"},
        {"transform": "contrast", "keys": ("user",), "short_window": 7,
         "long_window": 30, "target": "is_click", "agg": "sum"},
        {"transform": "recency", "keys": ("user",)},
    ]
    values = compute_specs(frame, specs)
    # The two user-1 rows at timestamp 10 see only timestamps 0 and 5.
    rate = values[spec_name(specs[0])].to_numpy()
    contrast = values[spec_name(specs[1])].to_numpy()
    recency = values[spec_name(specs[2])].to_numpy()
    np.testing.assert_allclose(rate, [0.0, 1.0, 0.5, 0.0, 2 / 3])
    # Contrast is short-window sum minus long-window sum.
    np.testing.assert_allclose(contrast, [0.0, 0.0, -1.0, 0.0, -2.0])
    np.testing.assert_allclose(recency, [0.0, 5.0, 5.0, 0.0, 10.0])


def test_recency_respects_source_mask_and_cutoff():
    frame = _frame()
    index = CausalIndex(frame, ("user",))
    mask = np.ones(len(frame), dtype=bool)
    mask[1] = False
    actual = index.recency(source_mask=mask, source_until=15)
    # The cutoff limits source history, but target rows after the cutoff can
    # still query that frozen prefix.
    np.testing.assert_allclose(actual, [0.0, 5.0, 10.0, 0.0, 10.0])


def test_budgeted_order_is_deterministic_and_balances_families():
    columns = ["user__z", "item__a", "user__a", "item_cat__b", "item__z"]
    expected = ["item__a", "item_cat__b", "user__a", "item__z", "user__z"]
    assert budgeted_candidate_order(columns) == expected
    assert budgeted_candidate_order(list(reversed(columns))) == expected
