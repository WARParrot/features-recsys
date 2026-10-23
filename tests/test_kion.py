import zipfile

import pandas as pd
import pytest

from kion_data import load_kion


def _write_fixture(tmp_path):
    interactions = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 11, 10, 12],
            "last_watch_dt": ["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-03"],
            "total_dur": [1, 2, 3, 4],
            "watched_pct": [10, 80, 70, 20],
        }
    ).to_csv(interactions, index=False)
    users = tmp_path / "users.csv"
    pd.DataFrame({"user_id": [1, 2], "age": ["a", "b"], "income": ["x", "y"], "sex": ["M", "F"], "kids_flg": [0, 1]}).to_csv(users, index=False)
    items = tmp_path / "items.csv"
    pd.DataFrame({"item_id": [10, 11, 12], "content_type": ["film"] * 3, "genres": ["drama", "comedy", "drama"]}).to_csv(items, index=False)
    return interactions, users, items


def test_kion_sampling_retains_complete_user_history(tmp_path):
    interactions, users, items = _write_fixture(tmp_path)
    frame = load_kion(interactions, users, items, max_users=1, seed=0)
    assert frame["user"].nunique() == 1
    assert len(frame) == 2
    assert frame["timestamp"].is_monotonic_increasing
    # Same-day events are intentionally atomic; the adapter only establishes
    # deterministic row order, not within-day causal order.
    assert set(frame["binary_label"]) == {0, 1}
    assert frame.attrs["kion_metadata"]["target"] == "watched_pct >= threshold"
    assert frame.attrs["kion_metadata"]["feedback_semantics"].startswith("observed_watch")
    assert "is_positive_watch" in frame


def test_kion_allows_missing_watch_percentage_in_last_row(tmp_path):
    interactions, users, items = _write_fixture(tmp_path)
    frame = pd.read_csv(interactions)
    frame.loc[frame.index[-1], "watched_pct"] = None
    frame.to_csv(interactions, index=False)

    loaded = load_kion(interactions, users, items)

    assert loaded.iloc[-1]["watched_pct"] == 0.0
    assert loaded.iloc[-1]["binary_label"] == 0


def test_kion_rejects_missing_required_values_and_malformed_inputs(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "user_id": [1], "item_id": [None], "last_watch_dt": ["2021-01-01"],
            "watched_pct": [50],
        }
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required values"):
        load_kion(bad)

    malformed = tmp_path / "malformed.csv"
    pd.DataFrame(
        {
            "user_id": [1], "item_id": [2], "last_watch_dt": ["not-a-date"],
            "watched_pct": [50],
        }
    ).to_csv(malformed, index=False)
    with pytest.raises(ValueError, match="invalid timestamps"):
        load_kion(malformed)

    bad_watch = tmp_path / "bad-watch.csv"
    pd.DataFrame(
        {
            "user_id": [1], "item_id": [2], "last_watch_dt": ["2021-01-01"],
            "watched_pct": ["not-a-number"],
        }
    ).to_csv(bad_watch, index=False)
    with pytest.raises(ValueError, match="watched_pct contains non-numeric"):
        load_kion(bad_watch)


def test_kion_zip_input_and_missing_columns(tmp_path):
    interactions, _, _ = _write_fixture(tmp_path)
    archive = tmp_path / "interactions.csv.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(interactions, "interactions.csv")
    assert len(load_kion(archive)) == 4
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"user_id": [1]}).to_csv(bad, index=False)
    with pytest.raises(KeyError, match="missing"):
        load_kion(bad)
