"""Shared ranking reference baselines used by single and rolling evaluation."""

from __future__ import annotations

import pandas as pd

from candidate_eval import causal_group_popularity_scores, causal_item_popularity_scores


def score_reference_rankings(log, ranking_log, ranking_features, scorer, *, include_user_item=True):
    """Score causal popularity references through either ranking protocol."""
    references = {
        "popularity": pd.DataFrame(
            {"popularity": causal_item_popularity_scores(log, ranking_log)}
        ),
        "decay_popularity": pd.DataFrame(
            {
                "popularity": causal_item_popularity_scores(
                    log, ranking_log, half_life=7 * 86400
                )
            }
        ),
        "category_popularity": pd.DataFrame(
            {
                "popularity": causal_group_popularity_scores(
                    log, ranking_log, key_col="item_cat"
                )
            }
        ),
    }
    results = {
        name: scorer(
            None,
            ranking_log,
            frame,
            ["popularity"],
            score_fn=lambda _model, values, _columns: values["popularity"].to_numpy(),
        )
        for name, frame in references.items()
    }
    if include_user_item:
        results["user_item_ctr"] = scorer(
            None,
            ranking_log,
            ranking_features[["ui_ctr_all"]].rename(columns={"ui_ctr_all": "ui_ctr"}),
            ["ui_ctr"],
            score_fn=lambda _model, values, _columns: values["ui_ctr"].to_numpy(),
        )
    return results
