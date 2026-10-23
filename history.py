import numpy as np


def build_histories(df, max_hist=20, user_col="user", item_col="item"):
    """Return previous-item histories aligned to rows of the full event stream.

    The input may be in any row order. Rows are replayed by timestamp in
    batches, so every event at a timestamp sees only strictly earlier groups.
    A test row can receive context from train/validation when called on the full
    event stream.
    """
    if max_hist <= 0:
        raise ValueError("max_hist must be positive")
    required = {user_col, item_col, "timestamp", "event_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    base = df.reset_index(drop=True)
    n = len(base)
    histories = np.empty((n, max_hist), dtype=object)
    histories[:] = None
    masks = np.zeros((n, max_hist), dtype=np.float32)
    order = base.sort_values(["timestamp", "event_id"], kind="mergesort").index.to_numpy()
    previous = {}
    timestamps = base["timestamp"].to_numpy()
    starts = np.r_[0, np.flatnonzero(timestamps[order][1:] != timestamps[order][:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts, ends):
        batch = order[start:end]
        for position in batch:
            user = base.at[position, user_col]
            prior = previous.get(user, [])
            if prior:
                items = prior[-max_hist:]
                histories[position, -len(items):] = items
                masks[position, -len(items):] = 1.0
        for position in batch:
            user = base.at[position, user_col]
            previous.setdefault(user, []).append(base.at[position, item_col])
    return histories, masks


def histories_for_positions(df, positions, max_hist=20):
    histories, masks = build_histories(df, max_hist=max_hist)
    positions = np.asarray(positions, dtype=np.int64)
    return histories[positions], masks[positions]
