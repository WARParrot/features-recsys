import numpy as np
import pandas as pd

WINDOWS = {"1d": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "90d": 90 * 86400}


class CausalIndex:
    """Reusable point-in-time index; equal timestamps are atomic."""

    def __init__(self, df, key_cols, time_col="timestamp", event_col="event_id"):
        missing = (set(key_cols) | {time_col, event_col}) - set(df.columns)
        if missing:
            raise KeyError(f"missing columns: {sorted(missing)}")
        self.base = df.reset_index(drop=True)
        self.key_cols = tuple(key_cols)
        if self.base[event_col].isna().any() or self.base[event_col].duplicated().any():
            raise ValueError(f"{event_col} must contain unique non-null values")
        if self.base[time_col].isna().any():
            raise ValueError(f"{time_col} must not contain null values")
        self.time_col = time_col
        self.order = self.base.sort_values(
            list(self.key_cols) + [time_col, event_col], kind="mergesort"
        ).index.to_numpy()
        sorted_frame = self.base.iloc[self.order]
        self.times = sorted_frame[time_col].to_numpy(dtype=np.int64)
        codes = pd.factorize(
            pd.MultiIndex.from_frame(sorted_frame[list(self.key_cols)]), sort=False
        )[0]
        self.boundaries = np.r_[0, np.flatnonzero(np.diff(codes) != 0) + 1, len(self.order)]
        self._right = np.empty(len(self.order), dtype=np.int64)
        self._group_start = np.empty(len(self.order), dtype=np.int64)
        self._left_cache = {}
        for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
            times = self.times[start:end]
            self._right[start:end] = start + np.searchsorted(times, times, side="left")
            self._group_start[start:end] = start

    def _known_values(self, target, source_mask, source_until):
        """Return source values and weights under the point-in-time policy.

        Mask weights must be finite. A source row is active when its mask
        weight is nonzero and, when a cutoff is supplied, its timestamp is
        strictly before ``source_until``.
        Inactive target values are replaced with zero before arithmetic, so
        missing or non-finite placeholders in held-out rows cannot propagate.
        Active target values must be finite; silently accepting them would make
        aggregate and decay features non-finite or otherwise data-dependent.
        """
        n = len(self.base)
        if source_mask is None:
            known = np.ones(n, dtype=np.float64)
        else:
            known = np.asarray(source_mask, dtype=np.float64).copy()
            if len(known) != n:
                raise ValueError("source_mask must have the same length as df")
            if not np.isfinite(known).all():
                raise ValueError("source_mask must contain only finite values")
        if source_until is not None:
            known *= self.base[self.time_col].to_numpy() < source_until
        active = known != 0
        if target is None:
            values = np.ones(n, dtype=np.float64)
        else:
            values = self.base[target].to_numpy(dtype=np.float64).copy()
            if not np.isfinite(values[active]).all():
                raise ValueError(f"{target} must be finite for active source rows")
            values[~active] = 0.0
        return values[self.order], known[self.order]

    def aggregate(self, window=None, agg="count", target=None, source_mask=None, source_until=None):
        if window is not None and (not isinstance(window, (int, np.integer)) or window <= 0):
            raise ValueError("window must be a positive integer or None")
        if agg not in {"count", "sum", "mean"}:
            raise ValueError(f"unsupported aggregation: {agg}")
        if agg == "count" and target is not None:
            raise ValueError("count does not accept target")
        if agg != "count" and target is None:
            raise ValueError(f"{agg} requires target")
        values, known = self._known_values(target, source_mask, source_until)
        n = len(self.base)
        if window not in self._left_cache:
            left = np.empty(n, dtype=np.int64)
            if window is None:
                left[:] = self._group_start
            else:
                for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
                    times = self.times[start:end]
                    left[start:end] = start + np.searchsorted(times, times - int(window), side="left")
            self._left_cache[window] = left
        left = self._left_cache[window]
        right = self._right
        prefix_count = np.r_[0.0, np.cumsum(known)]
        count = prefix_count[right] - prefix_count[left]
        if agg == "count":
            result_sorted = count
        else:
            prefix_sum = np.r_[0.0, np.cumsum(values * known)]
            total = prefix_sum[right] - prefix_sum[left]
            if agg == "sum":
                result_sorted = total
            else:
                result_sorted = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        result = np.empty(n, dtype=np.float64)
        result[self.order] = result_sorted
        return result

    def decay(self, half_life, target=None, source_mask=None, source_until=None):
        """Exponentially decayed prior sum; current timestamp batch excluded."""
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        values, known = self._known_values(target, source_mask, source_until)
        result_sorted = np.zeros(len(self.base), dtype=np.float64)
        rate = np.log(2.0) / float(half_life)
        for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
            state = 0.0
            last_time = None
            position = start
            while position < end:
                batch_end = position + 1
                while batch_end < end and self.times[batch_end] == self.times[position]:
                    batch_end += 1
                current_time = self.times[position]
                if last_time is not None:
                    state *= np.exp(-rate * (current_time - last_time))
                result_sorted[position:batch_end] = state
                state += float(np.sum(values[position:batch_end] * known[position:batch_end]))
                last_time = current_time
                position = batch_end
        result = np.empty(len(self.base), dtype=np.float64)
        result[self.order] = result_sorted
        return result


def causal_agg(df, key_cols, time_col="timestamp", event_col="event_id", window=None, agg="count", target=None, mask=None, source_until=None):
    index = CausalIndex(df, key_cols, time_col=time_col, event_col=event_col)
    values = index.aggregate(window=window, agg=agg, target=target, source_mask=mask, source_until=source_until)
    return pd.Series(values, index=df.index, name=f"{agg}_{target or 'rows'}")


def safe_div(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, np.maximum(denominator, 1.0), out=np.zeros_like(numerator), where=denominator > 0)


def build_baseline_features(df, source_mask=None, source_until=None):
    """Build causal aggregates over observed feedback.

    KION uses ``is_positive_watch``; the legacy ``is_click`` fallback is kept
    for synthetic/MIND compatibility. KION rate features are not CTR.
    """
    out = df.copy().reset_index(drop=True)
    feedback_column = "is_positive_watch" if "is_positive_watch" in out else "is_click"
    add, indexes = {}, {}

    def put(name, keys, window=None, agg="count", target=None):
        key = tuple(keys)
        if key not in indexes:
            indexes[key] = CausalIndex(out, key)
        add[name] = indexes[key].aggregate(
            window=window,
            agg=agg,
            target=target,
            source_mask=source_mask,
            source_until=source_until,
        )

    put("u_views_all", ["user"])
    put("u_views_7d", ["user"], WINDOWS["7d"])
    put("u_clicks_all", ["user"], agg="sum", target=feedback_column)
    put("u_clicks_7d", ["user"], WINDOWS["7d"], agg="sum", target=feedback_column)
    put("u_purch_all", ["user"], agg="sum", target="is_purchase")
    add["u_ctr_all"] = safe_div(add["u_clicks_all"], add["u_views_all"])
    add["u_ctr_7d"] = safe_div(add["u_clicks_7d"], add["u_views_7d"])
    add["u_cvr_all"] = safe_div(add["u_purch_all"], add["u_clicks_all"])

    put("i_views_all", ["item"])
    put("i_views_7d", ["item"], WINDOWS["7d"])
    put("i_clicks_all", ["item"], agg="sum", target=feedback_column)
    put("i_ctr_all", ["item"], agg="mean", target=feedback_column)

    put("ui_views_all", ["user", "item"])
    put("ui_clicks_all", ["user", "item"], agg="sum", target=feedback_column)
    put("ui_ctr_all", ["user", "item"], agg="mean", target=feedback_column)

    put("uc_views_all", ["user", "item_cat"])
    put(
        "uc_clicks_30d",
        ["user", "item_cat"],
        WINDOWS["30d"],
        agg="sum",
        target=feedback_column,
    )
    put("uc_views_30d", ["user", "item_cat"], WINDOWS["30d"])
    add["uc_ctr_30d"] = safe_div(add["uc_clicks_30d"], add["uc_views_30d"])

    for name, values in add.items():
        out[name] = values
    return out


BASELINE_FEATURES = ["u_views_all", "u_views_7d", "u_clicks_all", "u_clicks_7d", "u_purch_all", "u_ctr_all", "u_ctr_7d", "u_cvr_all", "i_views_all", "i_views_7d", "i_clicks_all", "i_ctr_all", "ui_views_all", "ui_clicks_all", "ui_ctr_all", "uc_views_all", "uc_clicks_30d", "uc_views_30d", "uc_ctr_30d"]
CATEGORICAL_FEATURES = ["item_cat", "user_segment"]
