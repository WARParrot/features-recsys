import pandas as pd
import pytest

from candidate_eval import sampled_candidate_log


def _logs():
    history = pd.DataFrame([
        {"event_id": 0, "user": 1, "item": 10, "timestamp": 1, "binary_label": 0, "item_cat": 0, "user_segment": 0},
        {"event_id": 1, "user": 2, "item": 11, "timestamp": 2, "binary_label": 0, "item_cat": 0, "user_segment": 0},
        {"event_id": 2, "user": 2, "item": 12, "timestamp": 3, "binary_label": 0, "item_cat": 1, "user_segment": 0},
    ])
    events = pd.DataFrame([
        {"event_id": 3, "user": 1, "item": 11, "timestamp": 4, "binary_label": 1, "item_cat": 0, "user_segment": 0},
    ])
    return history, events


def test_candidate_count_and_sampling_metadata_are_honest():
    history, events = _logs()
    result = sampled_candidate_log(
        events, history, n_items=3, n_negatives=1, max_queries=1,
        sampling="recent", seed=7,
    )
    protocol = result.attrs["candidate_protocol"]
    assert protocol["requested_negatives"] == 1
    assert protocol["max_queries"] == 1
    assert protocol["sampling"] == "recent"
    assert protocol["candidate_catalog"] == "cohort"
    assert protocol["sampling_weights_source"] == "pre_query_snapshot"
    assert protocol["rows"] == protocol["queries"] * 2


def test_static_catalog_requires_first_seen_and_excludes_future_items():
    history, events = _logs()
    catalog = pd.DataFrame({
        "item": [10, 11, 12, 99], "item_cat": [0, 0, 1, 2],
        "first_seen": [1, 2, 3, 100],
    })
    result = sampled_candidate_log(
        events, history, n_items=4, n_negatives=10, item_catalog=catalog,
        catalog_protocol="static_catalog", seed=1,
    )
    assert 99 not in set(result["item"])
    assert result.attrs["candidate_protocol"]["candidate_catalog"] == "full"

    with pytest.raises(ValueError, match="first_seen"):
        sampled_candidate_log(
            events, history, n_items=4, n_negatives=1,
            item_catalog=catalog.drop(columns="first_seen"),
            catalog_protocol="static_catalog",
        )
