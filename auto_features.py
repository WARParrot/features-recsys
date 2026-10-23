from itertools import product

import pandas as pd

from causal_features import CausalIndex, WINDOWS


KEY_GROUPS = [
    ("user",),
    ("item",),
    ("user", "item"),
    ("user", "item_cat"),
    ("item_cat",),
    ("user_segment", "item_cat"),
]


def enumerate_specs(extra_windows=None, feedback_target="is_click"):
    windows = {None, *WINDOWS.values()}
    if extra_windows:
        windows.update(int(value) for value in extra_windows)
    specs = []
    for keys, window in product(KEY_GROUPS, sorted(windows, key=lambda x: -1 if x is None else x)):
        specs.append({"keys": keys, "window": window, "agg": "count", "target": None})
        for target in (feedback_target, "is_purchase"):
            specs.append({"keys": keys, "window": window, "agg": "sum", "target": target})
    return specs


def spec_name(spec):
    window = "all" if spec["window"] is None else f"{spec['window']}s"
    target = "rows" if spec["target"] is None else spec["target"]
    return f"{'_'.join(spec['keys'])}__{spec['agg']}_{target}__{window}"


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
        features[name] = indexes[keys].aggregate(
            window=spec["window"],
            agg=spec["agg"],
            target=spec["target"],
            source_mask=source_mask,
            source_until=source_until,
        )
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
