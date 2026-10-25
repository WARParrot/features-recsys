import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from models import fit_lgbm_fixed


def gain_top_k(train, candidate_columns, k=40, random_state=17):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    candidate_columns = list(dict.fromkeys(candidate_columns))
    candidate_columns = [c for c in candidate_columns if train[c].nunique(dropna=False) > 1]
    if not candidate_columns:
        raise ValueError("gain_top_k: no nonconstant training candidates; check history availability")
    model = fit_lgbm_fixed(train, candidate_columns, rounds=400, random_state=random_state)
    gains = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=candidate_columns,
    ).sort_values(ascending=False)
    gains = gains[np.isfinite(gains) & (gains > 0)]
    if gains.empty:
        raise ValueError("gain_top_k: no finite positive feature gains; selection is not informative")
    return gains.head(k).index.tolist(), gains


def _linear_model(frame, columns, categorical_columns, random_state=17):
    categories = [c for c in categorical_columns if c in columns]
    numeric = [c for c in columns if c not in categories]
    transforms = []
    if numeric:
        transforms.append(("numeric", StandardScaler(), numeric))
    if categories:
        transforms.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categories))
    prep = ColumnTransformer(transforms, remainder="drop")
    return make_pipeline(prep, LogisticRegression(C=0.1, max_iter=250, solver="liblinear", random_state=random_state))


def budgeted_candidate_order(candidates):
    """Return a deterministic, family-balanced candidate order.

    Generated names use ``<entity/group>__...``.  Round-robinning sorted
    families prevents a large window/aggregation family from consuming the
    entire budget while retaining stable, auditable ordering.
    """
    unique = list(dict.fromkeys(candidates))
    families = {}
    for column in unique:
        family = column.split("__", 1)[0]
        families.setdefault(family, []).append(column)
    for values in families.values():
        values.sort()
    ordered = []
    for position in range(max((len(values) for values in families.values()), default=0)):
        for family in sorted(families):
            values = families[family]
            if position < len(values):
                ordered.append(values[position])
    return ordered


def forward_select(
    train,
    select,
    candidates,
    base_columns=(),
    categorical_columns=(),
    k=8,
    random_state=17,
    min_improvement=0.0,
    max_candidates=60,
    selection_interval="select",
    return_metadata=False,
):
    """Select only finite, strictly improving candidates.

    History includes a terminal rejected step with ``accepted=False`` and a
    reason; use ``chosen_features`` rather than history length for additions.
    Empty/constant bases use intercept-only AUC 0.5 for the first comparison.
    When requested, input provenance uses the unique family-balanced order before
    filtering; ``effective_pool`` records that order after train filtering/capping.
    Set ``return_metadata=True`` to append selection-pool provenance as a fourth
    return value; the default three-value API is unchanged.
    """
    if max_candidates is not None and (
        not isinstance(max_candidates, (int, np.integer))
        or isinstance(max_candidates, (bool, np.bool_))
        or max_candidates <= 0
    ):
        raise ValueError("max_candidates must be a positive integer or None")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and nonnegative")
    base_columns = list(dict.fromkeys(base_columns))
    candidates = list(dict.fromkeys(candidates))
    for name, frame in (("train", train), ("select", select)):
        missing = set(base_columns + candidates + ["binary_label"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing selection columns: {sorted(missing)}")
        if frame["binary_label"].nunique() != 2:
            raise ValueError(f"{name} must contain both label classes for selection AUC")
    candidate_order = budgeted_candidate_order(candidates)
    original_budget = (
        candidate_order if max_candidates is None else candidate_order[:max_candidates]
    )
    constant_candidates = {
        column for column in candidate_order
        if train[column].nunique(dropna=False) <= 1
    }
    candidate_pool = [
        column for column in candidate_order if column not in constant_candidates
    ]
    if max_candidates is not None:
        candidate_pool = candidate_pool[:max_candidates]
    candidate_metadata = {
        "effective_pool": list(candidate_pool),
        "effective_pool_sha256": hashlib.sha256(
            json.dumps(
                candidate_pool, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest(),
        "ordered_candidate_input_sha256": hashlib.sha256(
            json.dumps(
                candidate_order, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest(),
        "ordered_candidate_input_count": int(len(candidate_order)),
        "train_constant_candidates_removed": int(len(constant_candidates)),
        "max_candidates": None if max_candidates is None else int(max_candidates),
        "effective_pool_size": int(len(candidate_pool)),
    }
    selected = list(dict.fromkeys(base_columns))
    terminal_constants = [
        column for column in original_budget
        if column in constant_candidates and column not in selected
    ]
    remaining = sorted(column for column in candidate_pool if column not in selected)
    chosen_features = []
    history = []
    # An empty or entirely constant base is an intercept-only AUC baseline.
    previous_score = 0.5
    if selected and any(train[c].nunique(dropna=False) > 1 for c in selected):
        baseline_model = _linear_model(
            train, selected, categorical_columns, random_state=random_state
        )
        baseline_model.fit(train[selected], train["binary_label"])
        previous_score = float(
            roc_auc_score(
                select["binary_label"],
                baseline_model.predict_proba(select[selected])[:, 1],
            )
        )
    if not np.isfinite(previous_score):
        raise ValueError("forward_select: nonfinite base AUC")
    if not remaining and terminal_constants and k > 0:
        history.append({
            "step": 1, "feature": None, "score": None,
            "previous_score": previous_score, "improvement": None,
            "candidates_evaluated": len(terminal_constants),
            "selection_interval": selection_interval,
            "accepted": False,
            "reason": "no_nonconstant_candidate_with_finite_score",
        })
        result = (selected, chosen_features, history)
        return (*result, candidate_metadata) if return_metadata else result
    for step in range(1, min(k, len(remaining)) + 1):
        best_score, best_column = -np.inf, None
        candidates_evaluated = 0
        for column in remaining:
            candidates_evaluated += 1
            if train[column].nunique(dropna=False) <= 1:
                continue
            columns = selected + [column]
            model = _linear_model(
                train, columns, categorical_columns, random_state=random_state
            )
            model.fit(train[columns], train["binary_label"])
            score = roc_auc_score(
                select["binary_label"], model.predict_proba(select[columns])[:, 1]
            )
            if not np.isfinite(score):
                continue
            if (
                best_column is None
                or score > best_score
                or (score == best_score and column < best_column)
            ):
                best_score, best_column = float(score), column
        improvement = best_score - previous_score
        reason = None
        if best_column is None:
            reason = "no_nonconstant_candidate_with_finite_score"
        elif not np.isfinite(improvement):
            reason = "nonfinite_improvement"
        elif improvement <= 0:
            reason = "nonpositive_improvement"
        elif improvement < min_improvement:
            reason = "below_min_improvement"
        if reason is not None:
            history.append({
                "step": step, "feature": best_column,
                "score": best_score if np.isfinite(best_score) else None,
                "previous_score": previous_score,
                "improvement": improvement if np.isfinite(improvement) else None,
                "candidates_evaluated": candidates_evaluated,
                "selection_interval": selection_interval,
                "accepted": False, "reason": reason,
            })
            break
        selected.append(best_column)
        chosen_features.append(best_column)
        remaining.remove(best_column)
        history.append(
            {
                "step": step,
                "feature": best_column,
                "score": best_score,
                "previous_score": previous_score,
                "improvement": improvement,
                "candidates_evaluated": candidates_evaluated,
                "selection_interval": selection_interval,
            }
        )
        previous_score = best_score
    result = (selected, chosen_features, history)
    return (*result, candidate_metadata) if return_metadata else result
