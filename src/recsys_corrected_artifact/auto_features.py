from itertools import product

import pandas as pd

from .causal_features import CausalIndex, WINDOWS, safe_div


KEY_GROUPS = [
    ("user",),
    ("item",),
    ("user", "item"),
    ("user", "item_cat"),
    ("item_cat",),
    ("user_segment", "item_cat"),
]


def enumerate_specs(extra_windows=None, feedback_target="is_click", include_derived=True):
    windows = {None, *WINDOWS.values()}
    if extra_windows:
        windows.update(int(value) for value in extra_windows)
    ordered_windows = sorted(windows, key=lambda x: -1 if x is None else x)
    specs = []
    for keys, window in product(KEY_GROUPS, ordered_windows):
        specs.append({"keys": keys, "window": window, "agg": "count", "target": None})
        for target in (feedback_target, "is_purchase"):
            specs.append({"keys": keys, "window": window, "agg": "sum", "target": target})
    if include_derived:
        # Keep derived features explicit and auditable: every denominator/window
        # is encoded in the spec rather than inferred from a feature name.
        for keys, window in product(KEY_GROUPS, (WINDOWS["7d"], WINDOWS["30d"], WINDOWS["90d"])):
            for target in (feedback_target, "is_purchase"):
                specs.append({
                    "transform": "rate", "keys": keys, "window": window,
                    "target": target, "denominator": "count",
                })
        for keys in KEY_GROUPS:
            for target in (feedback_target, "is_purchase"):
                specs.append({
                    "transform": "contrast", "keys": keys, "short_window": WINDOWS["7d"],
                    "long_window": WINDOWS["30d"], "target": target, "agg": "sum",
                })
            specs.append({"transform": "recency", "keys": keys})
    return specs


def spec_name(spec):
    transform = spec.get("transform")
    keys = "_".join(spec["keys"])
    if transform == "rate":
        window = "all" if spec["window"] is None else f"{spec['window']}s"
        return f"{keys}__rate_{spec['target'] or 'rows'}__{window}"
    if transform == "contrast":
        return f"{keys}__contrast_{spec['target'] or 'rows'}__{spec['short_window']}s_vs_{spec['long_window']}s"
    if transform == "recency":
        return f"{keys}__recency_seconds"
    window = "all" if spec["window"] is None else f"{spec['window']}s"
    target = "rows" if spec["target"] is None else spec["target"]
    return f"{keys}__{spec['agg']}_{target}__{window}"


def compute_specs(df, specs, source_mask=None, source_until=None):
    features = {}
    indexes = {}
    for spec in specs:
        name = spec_name(spec)
        if name in features:
            continue
        keys = tuple(spec["keys"])
        if keys not in indexes:
            indexes[keys] = CausalIndex(df, keys)
        index = indexes[keys]
        transform = spec.get("transform")
        if transform == "rate":
            numerator = index.aggregate(
                window=spec["window"], agg="sum", target=spec["target"],
                source_mask=source_mask, source_until=source_until,
            )
            denominator = index.aggregate(
                window=spec["window"], agg="count", source_mask=source_mask,
                source_until=source_until,
            )
            values = safe_div(numerator, denominator)
        elif transform == "contrast":
            short = index.aggregate(
                window=spec["short_window"], agg=spec.get("agg", "sum"),
                target=spec["target"], source_mask=source_mask, source_until=source_until,
            )
            long = index.aggregate(
                window=spec["long_window"], agg=spec.get("agg", "sum"),
                target=spec["target"], source_mask=source_mask, source_until=source_until,
            )
            values = short - long
        elif transform == "recency":
            values = index.recency(source_mask=source_mask, source_until=source_until)
        else:
            values = index.aggregate(
                window=spec["window"], agg=spec["agg"], target=spec["target"],
                source_mask=source_mask, source_until=source_until,
            )
        features[name] = values
    return pd.DataFrame(features, index=df.index)


def nonstandard_specs(best_specs, windows=(5, 14, 45, 60)):
    """Deterministic comparator for searching additional time windows."""
    result = []
    seen = set()
    for spec in best_specs:
        for days in windows:
            candidate = {
                **spec,
                "window": int(days) * 86400,
            }
            name = spec_name(candidate)
            if name not in seen:
                seen.add(name)
                result.append(candidate)
    return result
