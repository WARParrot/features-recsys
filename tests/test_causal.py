import numpy as np
import pandas as pd
import pytest

from recsys_corrected_artifact.data import generate_synthetic, rolling_time_splits, split_metadata, time_split
from recsys_corrected_artifact.history import build_histories


def test_causal_aggregate_excludes_current_and_all_equal_timestamp_events():
    from recsys_corrected_artifact.causal_features import causal_agg

    df = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "user": [1, 1, 1, 1],
            "timestamp": [5, 5, 5, 9],
            "label": [1, 0, 1, 1],
        }
    )
    result = causal_agg(df, ["user"], agg="sum", target="label").to_numpy()
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 2.0])


def test_window_lower_boundary_is_inclusive_and_current_is_excluded():
    from recsys_corrected_artifact.causal_features import causal_agg

    df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "user": [1, 1, 1],
            "timestamp": [0, 10, 11],
        }
    )
    result = causal_agg(df, ["user"], window=10, agg="count").to_numpy()
    np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])


def test_history_replays_prior_events_and_preserves_context_across_split():
    df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "user": [4, 7, 4, 4],
            "item": [21, 30, 22, 23],
            "timestamp": [1, 2, 3, 4],
            "split": ["train", "train", "valid", "test"],
        }
    )
    history, mask = build_histories(df, max_hist=2)
    np.testing.assert_array_equal(history[2], [None, 21])
    np.testing.assert_array_equal(mask[2], [0.0, 1.0])
    assert history[2, 1] == 21
    np.testing.assert_array_equal(history[3], [21, 22])
    np.testing.assert_array_equal(mask[3], [1.0, 1.0])


def test_snapshot_mask_excludes_unavailable_labels():
    from recsys_corrected_artifact.causal_features import causal_agg

    df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "user": [1, 1, 1],
            "timestamp": [1, 2, 3],
            "label": [1, 1, 1],
        }
    )
    known = np.array([1, 0, 0], dtype=bool)
    result = causal_agg(
        df, ["user"], agg="sum", target="label", mask=known
    ).to_numpy()
    np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    cutoff_result = causal_agg(
        df, ["user"], agg="sum", target="label", source_until=2
    ).to_numpy()
    np.testing.assert_array_equal(cutoff_result, [0.0, 1.0, 1.0])


@pytest.mark.parametrize("operation", ["sum", "mean", "decay"])
@pytest.mark.parametrize("selection", ["mask", "cutoff", "both"])
def test_nonfinite_inactive_feedback_is_irrelevant(operation, selection):
    from recsys_corrected_artifact.causal_features import CausalIndex

    # Shuffled rows, two keys, timestamp ties, and an inclusive window boundary.
    df = pd.DataFrame({
        "event_id": [5, 1, 6, 0, 4, 2, 3],
        "user": [1, 1, 2, 1, 1, 1, 1],
        "timestamp": [4, 1, 5, 0, 3, 1, 2],
        "target": [np.inf, 4.0, 8.0, 2.0, -np.inf, 6.0, np.nan],
    })
    mask = np.array([0, 1, 1, 1, 0, 1, 0], dtype=float)
    source_mask = mask if selection != "cutoff" else None
    source_until = 2 if selection != "mask" else None
    active = np.ones(len(df), dtype=bool) if source_mask is None else mask != 0
    if source_until is not None:
        active &= df["timestamp"].to_numpy() < source_until

    expected = []
    for row in df.itertuples(index=False):
        prior = active & (df["user"] == row.user) & (df["timestamp"] < row.timestamp)
        if operation == "decay":
            weights = 2.0 ** (-(row.timestamp - df.loc[prior, "timestamp"]))
            expected.append((df.loc[prior, "target"] * weights).sum())
        else:
            prior &= df["timestamp"] >= row.timestamp - 2
            values = df.loc[prior, "target"]
            expected.append(values.sum() if operation == "sum" else values.mean() if len(values) else 0.0)

    index = CausalIndex(df, ["user"])
    kwargs = dict(target="target", source_mask=source_mask, source_until=source_until)
    # Inactive nonfinite values must not even trigger floating-point warnings.
    with np.errstate(invalid="raise"):
        if operation == "decay":
            actual = index.decay(half_life=1, **kwargs)
        else:
            actual = index.aggregate(window=2, agg=operation, **kwargs)
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(mask, [0, 1, 1, 1, 0, 1, 0])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("operation", ["sum", "mean", "decay"])
def test_active_nonfinite_feedback_is_rejected(operation, bad_value):
    from recsys_corrected_artifact.causal_features import CausalIndex

    # Validate every active source, even one not yet used by a query.
    df = pd.DataFrame({
        "event_id": [0, 1], "user": [1, 1],
        "timestamp": [0, 1], "target": [2.0, bad_value],
    })
    index = CausalIndex(df, ["user"])
    with pytest.raises(ValueError, match="target.*finite.*active"):
        if operation == "decay":
            index.decay(half_life=1, target="target")
        else:
            index.aggregate(agg=operation, target="target")
    # Counts do not consume feedback and therefore do not validate it.
    np.testing.assert_array_equal(index.aggregate(), [0.0, 1.0])


@pytest.mark.parametrize("feedback_column", ["is_click", "is_positive_watch"])
def test_baseline_features_ignore_masked_nonfinite_feedback(feedback_column):
    from recsys_corrected_artifact.causal_features import BASELINE_FEATURES, build_baseline_features

    frame = pd.DataFrame({
        "event_id": range(5), "user": [1] * 5, "item": [10] * 5,
        "timestamp": [0, 1, 1, 2, 3], "item_cat": [3] * 5,
        feedback_column: [1.0, np.nan, np.inf, -np.inf, 0.0],
        "is_purchase": [1.0, -np.inf, np.nan, np.inf, 0.0],
    })
    mask = [True, False, False, False, True]
    clean = frame.copy()
    clean.loc[[1, 2, 3], [feedback_column, "is_purchase"]] = 0.0
    actual = build_baseline_features(frame, source_mask=mask)[BASELINE_FEATURES]
    expected = build_baseline_features(clean, source_mask=mask)[BASELINE_FEATURES]
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual.to_numpy()).all()


@pytest.mark.parametrize("bad_weight", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_mask_is_rejected(bad_weight):
    from recsys_corrected_artifact.causal_features import CausalIndex

    frame = pd.DataFrame({"event_id": [0], "user": [1], "timestamp": [0]})
    index = CausalIndex(frame, ["user"])
    with pytest.raises(ValueError, match="source_mask.*finite"):
        index.aggregate(source_mask=[bad_weight])
    with pytest.raises(ValueError, match="source_mask.*finite"):
        index.decay(half_life=1, source_mask=[bad_weight])


def test_source_cutoff_does_not_mutate_mask():
    from recsys_corrected_artifact.causal_features import CausalIndex

    frame = pd.DataFrame({
        "event_id": [0, 1, 2], "user": [1] * 3,
        "timestamp": [0, 1, 2], "target": [1.0, 2.0, 3.0],
    })
    index = CausalIndex(frame, ["user"])
    mask = np.ones(3, dtype=float)
    index.aggregate(agg="sum", target="target", source_mask=mask, source_until=1)
    np.testing.assert_array_equal(mask, np.ones(3))
    np.testing.assert_array_equal(
        index.aggregate(agg="sum", target="target", source_mask=mask), [0.0, 1.0, 3.0]
    )


def test_sampled_candidate_log_records_sampling_protocol():
    from recsys_corrected_artifact.candidate_eval import sampled_candidate_log

    full_log = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "user": [1, 1, 2],
            "item": [10, 11, 12],
            "timestamp": [1, 2, 1],
            "binary_label": [0, 1, 0],
            "item_cat": [0, 1, 0],
            "user_segment": [0, 0, 1],
        }
    )
    candidates = sampled_candidate_log(
        full_log.iloc[[1]], full_log, n_items=13, n_negatives=1, seed=3
    )

    assert candidates.attrs["candidate_protocol"]["name"] == "sampled_unseen_item_discrimination"
    assert candidates.attrs["candidate_protocol"]["requested_negatives"] == 1
    assert candidates.empty
    assert candidates.attrs["candidate_protocol"]["skipped_positive_queries"] == 1
    assert candidates.attrs["candidate_protocol"]["negatives_exclude_previously_seen_items"]
    assert candidates.attrs["candidate_protocol"]["future_catalog_items_excluded"]


def test_sampled_candidates_use_history_for_seen_items_but_full_catalog_for_negatives():
    from recsys_corrected_artifact.candidate_eval import sampled_candidate_log

    history = pd.DataFrame(
        {
            "event_id": [0],
            "user": [1],
            "item": [12],
            "timestamp": [1],
            "binary_label": [0],
            "item_cat": [0],
            "user_segment": [0],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [1],
            "user": [1],
            "item": [10],
            "timestamp": [2],
            "binary_label": [1],
            "item_cat": [0],
            "user_segment": [0],
        }
    )
    catalog = pd.concat(
        [
            history,
            events,
            pd.DataFrame(
                {
                    "event_id": [2],
                    "user": [1],
                    "item": [11],
                    "timestamp": [3],
                    "binary_label": [0],
                    "item_cat": [1],
                    "user_segment": [0],
                }
            ),
        ],
        ignore_index=True,
    )

    candidates = sampled_candidate_log(
        events,
        history,
        n_items=3,
        n_negatives=1,
        item_catalog=catalog,
        seed=3,
    )

    assert set(candidates.loc[candidates["label"] == 0, "item"]) == set()
    assert candidates.empty
    assert candidates.attrs["candidate_protocol"]["skipped_positive_queries"] == 1
    static_catalog = catalog[["item", "item_cat"]].drop_duplicates("item").copy()
    static_catalog["first_seen"] = [1, 1, 1]
    static_candidates = sampled_candidate_log(
        events,
        history,
        n_items=3,
        n_negatives=1,
        item_catalog=static_catalog,
        seed=3,
        catalog_protocol="static_catalog",
    )
    assert set(static_candidates.loc[static_candidates["label"] == 0, "item"]) == {11}


def test_causal_item_popularity_does_not_see_current_or_future_events():
    from recsys_corrected_artifact.candidate_eval import causal_item_popularity_scores

    full_log = pd.DataFrame({
        "event_id": [0, 1], "user": [1, 2], "item": [10, 20],
        "timestamp": [1, 3],
    })
    candidates = pd.DataFrame({
        "event_id": [10, 11], "query_id": [0, 1], "user": [1, 1],
        "item": [10, 20], "timestamp": [1, 2],
    })
    scores = causal_item_popularity_scores(full_log, candidates)
    np.testing.assert_array_equal(scores, [0.0, 0.0])



def test_causal_category_popularity_scores_prior_categories():
    from recsys_corrected_artifact.candidate_eval import causal_group_popularity_scores

    full_log = pd.DataFrame(
        {
            "event_id": [0, 1],
            "item_cat": [2, 2],
            "timestamp": [1, 2],
        }
    )
    candidates = pd.DataFrame(
        {
            "event_id": [10, 11],
            "query_id": [0, 1],
            "item_cat": [2, 3],
            "timestamp": [2, 3],
        }
    )

    np.testing.assert_array_equal(
        causal_group_popularity_scores(full_log, candidates, key_col="item_cat"),
        [1.0, 0.0],
    )


def test_causal_item_popularity_scores_prior_item():
    from recsys_corrected_artifact.candidate_eval import causal_item_popularity_scores

    full_log = pd.DataFrame({
        "event_id": [0, 1], "user": [1, 2], "item": [10, 10],
        "timestamp": [1, 2],
    })
    candidates = pd.DataFrame({
        "event_id": [10], "query_id": [0], "user": [3], "item": [10],
        "timestamp": [3],
    })
    np.testing.assert_array_equal(causal_item_popularity_scores(full_log, candidates), [2.0])


def test_decayed_popularity_scores_decay_each_key_lazily():
    from recsys_corrected_artifact.candidate_eval import causal_item_popularity_scores

    full_log = pd.DataFrame({
        "event_id": [0, 1, 2], "item": [10, 20, 10],
        "timestamp": [0, 0, 1],
    })
    candidates = pd.DataFrame({
        "event_id": [10, 11], "query_id": [0, 0], "item": [10, 20],
        "timestamp": [2, 2],
    })

    np.testing.assert_allclose(
        causal_item_popularity_scores(full_log, candidates, half_life=1),
        [0.75, 0.25],
    )


def test_snapshot_features_exclude_held_out_source_rows():
    from recsys_corrected_artifact.causal_features import build_baseline_features

    frame = pd.DataFrame({
        "event_id": [0, 1, 2],
        "user": [1, 1, 1],
        "item": [10, 10, 10],
        "timestamp": [1, 2, 3],
        "item_cat": [3, 3, 3],
        "user_segment": [0, 0, 0],
        "is_click": [1.0, 1.0, 1.0],
        "is_purchase": [0.0, 0.0, 0.0],
    })

    features = build_baseline_features(frame, source_mask=[True, False, False])

    np.testing.assert_array_equal(features["u_clicks_all"], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(features["ui_clicks_all"], [0.0, 1.0, 1.0])


def test_candidate_auto_features_fill_kion_feedback_placeholder():
    from recsys_corrected_artifact.candidate_eval import candidate_features

    full_log = pd.DataFrame({
        "event_id": [0], "user": [1], "item": [10], "timestamp": [1],
        "label": [1], "is_positive_watch": [1.0], "is_click": [1.0],
        "is_purchase": [0.0], "binary_label": [1], "item_cat": [2], "user_segment": [0],
    })
    candidates = pd.DataFrame({
        "event_id": [1], "query_id": [0], "user": [1], "item": [11], "timestamp": [2],
        "label": [0], "is_click": [0.0], "is_purchase": [0.0], "binary_label": [0],
        "item_cat": [2], "user_segment": [0],
    })
    _, features = candidate_features(full_log, candidates)
    assert features["u_clicks_all"].notna().all()


def test_candidate_auto_features_ignore_synthetic_candidate_labels():
    from recsys_corrected_artifact.candidate_eval import candidate_features

    full_log = pd.DataFrame(
        {
            "event_id": [0],
            "user": [1],
            "item": [10],
            "timestamp": [1],
            "label": [1],
            "is_click": [1.0],
            "is_purchase": [0.0],
            "binary_label": [1],
            "item_cat": [2],
            "user_segment": [0],
        }
    )
    candidates = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "query_id": [0, 0, 1],
            "user": [1, 1, 1],
            "item": [11, 12, 13],
            "timestamp": [2, 2, 3],
            "label": [1, 0, 0],
            "is_click": [1.0, 0.0, 0.0],
            "is_purchase": [0.0, 0.0, 0.0],
            "binary_label": [1, 0, 0],
            "item_cat": [2, 2, 2],
            "user_segment": [0, 0, 0],
        }
    )
    specs = [
        {
            "keys": ("user",),
            "window": None,
            "agg": "sum",
            "target": "is_click",
        }
    ]

    _, features = candidate_features(full_log, candidates, specs=specs)
    feature_name = "user__sum_is_click__all"
    np.testing.assert_array_equal(features[feature_name], [1.0, 1.0, 1.0])

    changed_labels = candidates.copy()
    changed_labels["is_click"] = [0.0, 1.0, 1.0]
    changed_labels["label"] = [0, 1, 1]
    _, changed_features = candidate_features(full_log, changed_labels, specs=specs)
    np.testing.assert_array_equal(
        changed_features[feature_name], features[feature_name]
    )


def test_causal_index_matches_reference_for_shuffled_rows_and_masks():
    from recsys_corrected_artifact.causal_features import CausalIndex

    df = pd.DataFrame(
        {
            "event_id": [5, 2, 4, 1, 3],
            "user": [1, 1, 2, 1, 2],
            "item": [8, 8, 9, 8, 9],
            "timestamp": [12, 4, 12, 4, 8],
            "target": [1.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    mask = np.array([1, 1, 0, 1, 1], dtype=bool)
    index = CausalIndex(df, ["user", "item"])

    for window in (None, 8):
        expected = []
        for row in df.itertuples(index=False):
            prior = df[
                (df["user"] == row.user)
                & (df["item"] == row.item)
                & (df["timestamp"] < row.timestamp)
            ]
            prior_positions = prior.index.to_numpy()
            if window is not None:
                prior_positions = prior_positions[
                    prior["timestamp"].to_numpy() >= row.timestamp - window
                ]
            known_positions = prior_positions[mask[prior_positions]]
            expected.append(df.loc[known_positions, "target"].sum())

        actual = index.aggregate(
            window=window,
            agg="sum",
            target="target",
            source_mask=mask,
        )
        np.testing.assert_array_equal(actual, expected)


def test_decay_is_causal_and_excludes_same_timestamp():
    from recsys_corrected_artifact.causal_features import CausalIndex

    df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "user": [1, 1, 1],
            "timestamp": [1, 1, 2],
            "is_click": [1.0, 1.0, 0.0],
        }
    )
    index = CausalIndex(df, ["user"])
    result = index.decay(half_life=1, target="is_click")
    np.testing.assert_array_equal(result[:2], [0.0, 0.0])
    assert result[2] == 1.0


def test_history_treats_equal_timestamps_as_atomic():
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "user": [7, 7, 7],
            "item": [10, 11, 12],
            "timestamp": [1, 1, 2],
        }
    )
    history, mask = build_histories(df, max_hist=2)
    np.testing.assert_array_equal(mask[0], [0.0, 0.0])
    np.testing.assert_array_equal(mask[1], [0.0, 0.0])
    np.testing.assert_array_equal(history[2], [10, 11])
    np.testing.assert_array_equal(mask[2], [1.0, 1.0])


def test_split_is_chronological_disjoint_and_has_four_roles():
    df = generate_synthetic(n_events=500, seed=8)
    ordered, parts = time_split(df)
    assert list(parts) == ["train", "select", "valid", "test"]
    joined = np.concatenate(list(parts.values()))
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(df)))
    for left, right in zip(
        [parts["train"], parts["select"], parts["valid"]],
        [parts["select"], parts["valid"], parts["test"]],
    ):
        assert ordered.iloc[left[-1]]["timestamp"] < ordered.iloc[right[0]]["timestamp"]


def test_rolling_splits_expand_training_and_preserve_timestamp_groups():
    ordered, folds = rolling_time_splits(generate_synthetic(n_events=800, seed=9), n_folds=3)
    assert len(folds) == 3
    previous_train_rows = 0
    for fold in folds:
        metadata = split_metadata(ordered, fold)
        assert metadata["train"]["rows"] > previous_train_rows
        previous_train_rows = metadata["train"]["rows"]
        for left, right in zip(("train", "select", "valid"), ("select", "valid", "test")):
            assert metadata[left]["max_timestamp"] < metadata[right]["min_timestamp"]


def test_split_metadata_reports_rows_and_timestamp_bounds():
    ordered, parts = time_split(generate_synthetic(n_events=500, seed=8))
    metadata = split_metadata(ordered, parts)

    assert set(metadata) == {"train", "select", "valid", "test"}
    assert sum(entry["rows"] for entry in metadata.values()) == len(ordered)
    for left, right in zip(("train", "select", "valid"), ("select", "valid", "test")):
        assert metadata[left]["max_timestamp"] < metadata[right]["min_timestamp"]


def test_split_never_breaks_timestamp_groups():
    df = pd.DataFrame({"event_id": range(100), "timestamp": [1] * 40 + [2] * 20 + [3] * 20 + [4] * 20})
    ordered, parts = time_split(df)
    for left, right in zip(
        [parts["train"], parts["select"], parts["valid"]],
        [parts["select"], parts["valid"], parts["test"]],
    ):
        assert ordered.iloc[left[-1]]["timestamp"] < ordered.iloc[right[0]]["timestamp"]
