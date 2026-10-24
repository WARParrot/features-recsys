import numpy as np
import pandas as pd


def generate_synthetic(
    n_users=500,
    n_items=300,
    n_events=30_000,
    n_categories=20,
    n_segments=8,
    n_days=120,
    seed=42,
):
    """Generate interactions with drift in users' preferred category over time.

    ``event_id`` is only a stable identifier. Equal timestamps are treated as
    simultaneous by feature construction and splitting.
    """
    rng = np.random.default_rng(seed)
    user_segment = rng.integers(0, n_segments, size=n_users)
    interest_a = rng.integers(0, n_categories, size=n_users)
    interest_b = rng.integers(0, n_categories, size=n_users)
    item_cat = rng.integers(0, n_categories, size=n_items)
    item_pop = rng.gamma(2.0, 1.0, size=n_items) + 0.1
    user_activity = rng.gamma(2.0, 1.0, size=n_users) + 0.1
    users = rng.choice(n_users, size=n_events, p=user_activity / user_activity.sum())
    timestamps = rng.integers(0, n_days * 86400, size=n_events, dtype=np.int64)
    drift_time = (n_days * 86400) // 2
    post_drift = timestamps >= drift_time
    active_interest = np.where(post_drift, interest_b[users], interest_a[users])

    items = np.empty(n_events, dtype=np.int64)
    for user in np.unique(users):
        user_mask = users == user
        for phase in (False, True):
            mask = user_mask & (post_drift == phase)
            if not mask.any():
                continue
            interest = interest_b[user] if phase else interest_a[user]
            match = item_cat == interest
            weights = item_pop * (1.0 + 8.0 * match)
            weights /= weights.sum()
            items[mask] = rng.choice(n_items, size=int(mask.sum()), p=weights)

    matched = item_cat[items] == active_interest
    p_click = np.where(matched, 0.55, 0.06) + 0.05 * rng.random(n_events)
    clicked = rng.random(n_events) < p_click
    purchased = clicked & (rng.random(n_events) < np.where(matched, 0.25, 0.05))

    label = np.zeros(n_events, dtype=np.int8)
    label[clicked] = 1
    label[purchased] = 2
    frame = pd.DataFrame(
        {
            "event_id": np.arange(n_events, dtype=np.int64),
            "user": users.astype(np.int64),
            "item": items.astype(np.int64),
            "timestamp": timestamps,
            "label": label,
            "item_cat": item_cat[items].astype(np.int64),
            "user_segment": user_segment[users].astype(np.int64),
        }
    )
    frame["is_click"] = (frame["label"] >= 1).astype(np.float32)
    frame["is_purchase"] = (frame["label"] == 2).astype(np.float32)
    frame["binary_label"] = (frame["label"] >= 1).astype(np.int8)
    return frame.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)


def split_metadata(ordered, parts):
    """Summarize chronological split rows for experiment reporting."""
    result = {}
    for name, positions in parts.items():
        frame = ordered.iloc[positions]
        if frame.empty:
            raise ValueError(f"split {name!r} must be non-empty")
        result[name] = {
            "rows": int(len(frame)),
            "min_timestamp": int(frame["timestamp"].min()),
            "max_timestamp": int(frame["timestamp"].max()),
        }
    return result


def rolling_time_splits(df, n_folds=3, warmup_fraction=0.0):
    """Return expanding-train chronological folds without splitting timestamps.

    Optional ``warmup_fraction`` reserves an initial timestamp-atomic prefix
    (nearest row-count fraction) for history only. The remaining timestamps
    determine train/select/valid/test identically for every feature space and
    history mode. The train block expands across folds; the later three blocks
    move forward together. Returned positions always refer to the full ordered
    log, including warmup; no source rows are dropped or reindexed per role.
    Zero warmup preserves the original v5 split boundaries.
    """
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, (int, np.integer)) or n_folds <= 0:
        raise ValueError("n_folds must be a positive integer")
    if not np.isscalar(warmup_fraction) or not np.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be finite and in [0, 1)")
    ordered = df.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("cannot split an empty dataframe")
    group_ends = np.flatnonzero(
        ordered["timestamp"].to_numpy()[1:] != ordered["timestamp"].to_numpy()[:-1]
    ) + 1
    boundaries = np.r_[0, group_ends, len(ordered)]
    n_groups = len(boundaries) - 1
    required_blocks = n_folds + 3
    warmup_groups = 0
    if warmup_fraction > 0:
        # Reserve whole timestamp groups, never rows within a simultaneous batch.
        choices = np.arange(1, n_groups - required_blocks + 1)
        if not len(choices):
            raise ValueError("not enough distinct timestamps for warmup and rolling roles")
        warmup_groups = int(choices[np.argmin(np.abs(boundaries[choices] - warmup_fraction * len(ordered)))])
    if n_groups - warmup_groups < required_blocks:
        raise ValueError(
            f"at least {required_blocks} distinct timestamps are required for {n_folds} folds"
        )
    points = np.linspace(warmup_groups, n_groups, required_blocks + 1, dtype=int)
    if np.any(np.diff(points) < 1):
        raise ValueError("timestamp groups are too sparse for requested rolling folds")
    folds = []
    for fold in range(n_folds):
        group_points = np.asarray([warmup_groups, *points[fold + 1:fold + 5]])
        cuts = boundaries[group_points]
        folds.append({
            **({"warmup": np.arange(0, cuts[0])} if warmup_groups else {}),
            "train": np.arange(cuts[0], cuts[1]),
            "select": np.arange(cuts[1], cuts[2]),
            "valid": np.arange(cuts[2], cuts[3]),
            "test": np.arange(cuts[3], cuts[4]),
        })
    return ordered, folds


def time_split(df, fractions=(0.55, 0.15, 0.15, 0.15)):
    """Return disjoint chronological positions: train/select/valid/test.

    ``select`` is used only for feature/spec selection, ``valid`` only for
    model early stopping and ``test`` only for the final report.
    """
    if len(fractions) != 4 or not np.isfinite(fractions).all() or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain four finite values summing to 1")
    if any(value <= 0 for value in fractions):
        raise ValueError("every split fraction must be positive")
    ordered = df.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot split an empty dataframe")
    group_ends = np.flatnonzero(
        ordered["timestamp"].to_numpy()[1:] != ordered["timestamp"].to_numpy()[:-1]
    ) + 1
    boundaries = np.r_[0, group_ends, n]
    n_groups = len(boundaries) - 1
    if n_groups < 4:
        raise ValueError("at least four distinct timestamps are required for four non-empty splits")

    target_positions = np.cumsum(fractions)[:-1] * n
    cut_indices = []
    previous = 0
    for split_number, target in enumerate(target_positions, start=1):
        min_idx = previous + 1
        max_idx = n_groups - (4 - split_number)
        choices = np.arange(min_idx, max_idx + 1)
        chosen = int(choices[np.argmin(np.abs(boundaries[choices] - target))])
        cut_indices.append(chosen)
        previous = chosen
    cuts = boundaries[np.asarray([0, *cut_indices, n_groups])]
    return ordered, {
        "train": np.arange(cuts[0], cuts[1]),
        "select": np.arange(cuts[1], cuts[2]),
        "valid": np.arange(cuts[2], cuts[3]),
        "test": np.arange(cuts[3], cuts[4]),
    }
