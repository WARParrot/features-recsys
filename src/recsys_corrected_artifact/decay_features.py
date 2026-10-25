import pandas as pd
from .causal_features import CausalIndex


def build_decay_features(df, half_lives=(86400, 7 * 86400, 30 * 86400), source_mask=None, source_until=None):
    """Alternative to hard windows: exponentially decayed causal counts/rates."""
    groups = {
        "u": ("user",),
        "i": ("item",),
        "ui": ("user", "item"),
        "uc": ("user", "item_cat"),
    }
    out = {}
    feedback_column = "is_positive_watch" if "is_positive_watch" in df else "is_click"
    indexes = {name: CausalIndex(df, keys) for name, keys in groups.items()}
    for name, index in indexes.items():
        for half_life in half_lives:
            tag = f"{half_life}s"
            out[f"{name}_decay_views_{tag}"] = index.decay(
                half_life, source_mask=source_mask, source_until=source_until
            )
            out[f"{name}_decay_clicks_{tag}"] = index.decay(
                half_life, target=feedback_column, source_mask=source_mask,
                source_until=source_until
            )
    return pd.DataFrame(out, index=df.index)
