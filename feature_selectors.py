import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from models import fit_lgbm_fixed


def gain_top_k(train, candidate_columns, k=40, random_state=17):
    model = fit_lgbm_fixed(train, candidate_columns, rounds=400, random_state=random_state)
    gains = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=candidate_columns,
    ).sort_values(ascending=False)
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
):
    """Greedily select candidates with auditable per-step validation records."""
    candidate_pool = budgeted_candidate_order(candidates)
    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive or None")
        candidate_pool = candidate_pool[:max_candidates]
    selected = list(dict.fromkeys(base_columns))
    remaining = sorted(column for column in candidate_pool if column not in selected)
    chosen_features = []
    history = []
    previous_score = None
    if selected:
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
    for step in range(1, min(k, len(remaining)) + 1):
        best_score, best_column = -np.inf, None
        candidates_evaluated = 0
        for column in remaining:
            columns = selected + [column]
            model = _linear_model(
                train, columns, categorical_columns, random_state=random_state
            )
            model.fit(train[columns], train["binary_label"])
            score = roc_auc_score(
                select["binary_label"], model.predict_proba(select[columns])[:, 1]
            )
            candidates_evaluated += 1
            if (
                best_column is None
                or score > best_score
                or (score == best_score and column < best_column)
            ):
                best_score, best_column = float(score), column
        if best_column is None:
            break
        improvement = (
            None if previous_score is None else best_score - previous_score
        )
        if improvement is not None and improvement < min_improvement:
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
    return selected, chosen_features, history
