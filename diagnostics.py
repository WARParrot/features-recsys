"""JSON-safe descriptive diagnostics; no labels enter history or slice definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)), "nonfinite": int(len(values) - len(finite)),
        "min": float(finite.min()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "p90": float(np.quantile(finite, .9)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
    }


def feature_diagnostics(features):
    """Column-level missingness, constants and zeros (denominator: all rows)."""
    columns = {}
    for name in features:
        values = features[name]
        columns[str(name)] = {
            "constant": bool(len(values) and values.nunique(dropna=False) <= 1),
            "zero_fraction": float(values.eq(0).mean()) if len(values) else None,
            "missing_fraction": float(values.isna().mean()) if len(values) else None,
        }
    return {"rows": len(features), "columns": columns,
            "constant_fraction": (sum(v["constant"] for v in columns.values()) / len(columns)
                                  if columns else None)}


def role_diagnostics(frame, features=None, *, history_lengths=None,
                     item_popularity=None, predictions=None):
    """Summarize an aligned role and optional event-probability predictions.

    History/popularity arrays must be computed from permitted history, not the
    evaluation labels. Prediction slices are descriptive event-level AUC/Brier,
    NOT ranking metrics or causal effects. Thresholds are fixed, not label-tuned.
    """
    n = len(frame)
    label = frame["binary_label"].to_numpy(dtype=float)
    if not np.isin(label, [0, 1]).all():
        raise ValueError("binary_label must contain only 0/1")
    result = {"rows": n, "positives": int(label.sum()),
              "prevalence": float(label.mean()) if n else None,
              "users": int(frame["user"].nunique()), "items": int(frame["item"].nunique())}
    if features is not None:
        if len(features) != n or not features.index.equals(frame.index):
            raise ValueError("features must have the same index as the role")
        result["features"] = feature_diagnostics(features)
    masks = {"all": np.ones(n, dtype=bool)}
    for name, values in (("history_length", history_lengths), ("item_popularity", item_popularity)):
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must be an aligned finite nonnegative vector")
        result[name] = distribution(values)
        masks.update({f"{name}:0": values == 0, f"{name}:1-5": (values >= 1) & (values <= 5),
                      f"{name}:6-20": (values > 5) & (values <= 20), f"{name}:21+": values > 20})
    scores = {}
    for model, values in (predictions or {}).items():
        values = np.asarray(values, dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("predictions must be aligned finite probabilities")
        scores[model] = values
    result["slices"] = {}
    for name, mask in masks.items():
        y = label[mask]
        entry = {"rows": int(mask.sum()), "positives": int(y.sum()),
                 "prevalence": float(y.mean()) if len(y) else None}
        if scores:
            from sklearn.metrics import roc_auc_score
            entry["predictions"] = {
                model: {"auc": float(roc_auc_score(y, values[mask])) if len(np.unique(y)) == 2 else None,
                        "auc_unestimable_reason": None if len(np.unique(y)) == 2 else
                        ("empty_slice" if not len(y) else "single_class"),
                        "brier": float(np.mean((values[mask] - y) ** 2)) if len(y) else None}
                for model, values in scores.items()
            }
        result["slices"][name] = entry
    result["prediction_slices_status"] = "available" if scores else "unavailable_no_predictions"
    return result


def fold_diagnostics(log, parts, features=None):
    """Prefix-frozen history coverage per role (strictly before role timestamp).

    This definition intentionally stays fixed across ablation cells: descriptive
    coverage is not a claim about the cell's within-role feature update policy.
    """
    result = {}
    for role, positions in parts.items():
        frame = log.iloc[positions]
        cutoff = frame["timestamp"].min()
        prefix = log.loc[log["timestamp"] < cutoff]
        result[role] = role_diagnostics(
            frame, features.iloc[positions] if features is not None else None,
            history_lengths=frame["user"].map(prefix["user"].value_counts()).fillna(0).to_numpy(),
            item_popularity=frame["item"].map(prefix["item"].value_counts()).fillna(0).to_numpy(),
        )
        result[role]["history_definition"] = "strict_prefix_before_role_min_timestamp"
    return result
