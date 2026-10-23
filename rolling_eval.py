"""Rolling temporal evaluation for the comparison artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_features import compute_specs, enumerate_specs
from candidate_eval import (
    candidate_features,
    causal_group_popularity_scores,
    causal_item_popularity_scores,
    predict_linear,
    sampled_candidate_log,
    score_candidate_sets,
    score_logged_impressions,
)
from causal_features import CATEGORICAL_FEATURES, BASELINE_FEATURES, build_baseline_features
from data import rolling_time_splits, split_metadata
from decay_features import build_decay_features
from feature_selectors import forward_select, gain_top_k
from linear_model import fit_logistic, logistic_auc, tune_logistic
from mind_data import summarize_mind_metadata
from models import auc, fit_lgbm, predict_proba
from pipeline import tune_model
from ranking_references import score_reference_rankings
from metrics import (
    query_weighted_cluster_bootstrap_mean_ci,
    paired_query_weighted_cluster_bootstrap_mean_ci,
    cluster_bootstrap_auc_comparison,
)


_AUC_BOOTSTRAP_REPLICATES = 1000
ROLLING_PROTOCOL_VERSION = 5
ROLLING_AUC_UNCERTAINTY = {
    "estimand": "pooled_event_level_auc",
    "unit": "user_cluster_resampling",
    "interval": "percentile_95_user_cluster_bootstrap",
    "effect_interval": "paired_percentile_95",
    "replicates": _AUC_BOOTSTRAP_REPLICATES,
    "primary_comparison": "auto_gain_minus_manual",
    "conditional_on": ["fitted_models", "feature_selection", "temporal_fold"],
    "excludes": ["retraining_variability", "selection_variability", "temporal_shift"],
}


def _rolling_features(log, source_mask=None, source_until=None):
    baseline = build_baseline_features(log, source_mask=source_mask, source_until=source_until)
    feedback_target = "is_positive_watch" if "is_positive_watch" in log else "is_click"
    specs = [
        spec for spec in enumerate_specs(feedback_target=feedback_target)
        if set(spec["keys"]).issubset(log.columns)
    ]
    auto = compute_specs(log, specs, source_mask=source_mask, source_until=source_until)
    decay = build_decay_features(log, source_mask=source_mask, source_until=source_until)
    return baseline, auto, decay, specs


def _ranking_result(
    dataset, log, test, models, specs, fold_seed, source_log=None,
    candidate_n_negatives=99, candidate_max_queries=400,
    candidate_sampling="uniform", candidate_catalog="cohort", item_catalog=None,
):
    source_log = log if source_log is None else source_log
    if dataset == "mind":
        ranking_log = test
        ranking_features = test
        scorer = score_logged_impressions
    else:
        ranking_log = sampled_candidate_log(
            test,
            source_log,
            int((log["item"].nunique() if candidate_catalog == "cohort" else len(item_catalog))),
            n_negatives=candidate_n_negatives,
            max_queries=candidate_max_queries,
            seed=fold_seed,
            item_catalog=log if candidate_catalog == "cohort" else item_catalog,
            catalog_protocol="historical_seen_items" if candidate_catalog == "cohort" else "static_catalog",
            sampling=candidate_sampling,
        )
        _, ranking_features = candidate_features(
            source_log, ranking_log, specs=specs, decay_builder=build_decay_features
        )
        scorer = score_candidate_sets

    rankings = {}
    for name, (model, feature_columns, category_columns, score_fn) in models.items():
        if ranking_features.empty:
            scores = np.empty(0, dtype=float)
        else:
            scores = (
                score_fn(model, ranking_features, feature_columns)
                if score_fn is not None
                else predict_proba(model, ranking_features, feature_columns, category_columns)
            )
        rankings[name] = scorer(
            model,
            ranking_log,
            ranking_features,
            feature_columns,
            category_columns,
            score_fn=lambda _model, _features, _columns, values=scores: values,
        )

    rankings.update(
        score_reference_rankings(
            source_log,
            ranking_log,
            ranking_features,
            scorer,
        )
    )
    for name, result in rankings.items():
        query_metrics = result.get("query_metrics", {})
        if "ndcg10" in query_metrics:
            result["ndcg10_query_weighted_cluster_ci95"] = query_weighted_cluster_bootstrap_mean_ci(
                query_metrics["ndcg10"], query_metrics.get("users", []),
                seed=fold_seed, n_boot=2000,
            )
            result["ndcg10_delta_vs_manual_query_weighted_cluster_ci95"] = (
                paired_query_weighted_cluster_bootstrap_mean_ci(
                    query_metrics["ndcg10"],
                    rankings["manual"]["query_metrics"]["ndcg10"],
                    query_metrics.get("users", []),
                    seed=fold_seed,
                    n_boot=2000,
                )
                if name != "manual"
                else [0.0, 0.0]
            )
    return rankings, ranking_log


def _fold_features(log, parts, history_mode):
    source_mask = np.ones(len(log), dtype=bool)
    source_mask[parts["test"]] = False
    if history_mode == "prequential":
        return _rolling_features(log, source_mask=source_mask)
    blocks = [[], [], []]
    for positions in parts.values():
        cutoff = log.iloc[positions]["timestamp"].min()
        values = _rolling_features(
            log,
            source_mask=log["timestamp"].to_numpy() < cutoff,
            source_until=cutoff,
        )
        for block, features in zip(blocks, values[:3]):
            block.append(features.iloc[positions])
    return (*(pd.concat(block).sort_index() for block in blocks), values[3])


def _fold_result(log, parts, seed, fold, dataset, forward_max_candidates=60, forward_steps=8,
                 history_mode="prequential", **candidate_options):
    baseline, auto, decay, specs = _fold_features(log, parts, history_mode)
    categorical = [column for column in CATEGORICAL_FEATURES if column in log.columns]
    manual_columns = [
        column for column in BASELINE_FEATURES + categorical
        if column in baseline.columns
    ]
    auto_columns = auto.columns.tolist()
    decay_columns = decay.columns.tolist()
    full = baseline.join(auto).join(decay)
    frames = {name: full.iloc[index].copy() for name, index in parts.items()}
    train, select, valid, test = (
        frames[name] for name in ("train", "select", "valid", "test")
    )
    state = seed + fold

    manual_select, manual_config = tune_model(
        train, select, manual_columns, categorical, random_state=state
    )
    top_auto, gains = gain_top_k(
        train, auto_columns, k=40, random_state=state
    )
    auto_select, auto_config = tune_model(
        train, select, top_auto, random_state=state
    )
    forward_columns, forward_added, forward_history = forward_select(
        train,
        select,
        auto_columns,
        base_columns=manual_columns,
        categorical_columns=categorical,
        k=forward_steps,
        max_candidates=forward_max_candidates,
        random_state=state,
        selection_interval=f"fold_{fold}_select",
    )
    forward_select_auc, forward_config = tune_model(
        train, select, forward_columns, categorical, random_state=state
    )
    decay_select, decay_config = tune_model(
        train,
        select,
        manual_columns + decay_columns,
        categorical,
        random_state=state,
    )
    linear_select, linear_c = tune_logistic(
        train, select, manual_columns, categorical, random_state=state
    )

    choices = {
        "manual": (manual_select, manual_columns, categorical, manual_config),
        "auto_gain": (auto_select, top_auto, (), auto_config),
        "auto_forward": (
            forward_select_auc,
            forward_columns,
            categorical,
            forward_config,
        ),
        "decay": (
            decay_select,
            manual_columns + decay_columns,
            categorical,
            decay_config,
        ),
    }
    selected = max(choices, key=lambda name: (choices[name][0], name))
    _, selected_columns, selected_categories, selected_config = choices[selected]

    train_select = pd.concat([train, select], axis=0).sort_values(
        ["timestamp", "event_id"]
    )
    selected_model = fit_lgbm(
        train_select, valid, selected_columns, selected_categories,
        selected_config, random_state=state,
    )
    manual_model = fit_lgbm(
        train_select, valid, manual_columns, categorical,
        manual_config, random_state=state,
    )
    linear_model = fit_logistic(
        train_select, manual_columns, categorical, c=linear_c,
        random_state=state,
    )
    decay_model = fit_lgbm(
        train_select, valid, manual_columns + decay_columns, categorical,
        decay_config, random_state=state,
    )
    auto_model = fit_lgbm(
        train_select, valid, top_auto, (), auto_config, random_state=state,
    )
    forward_model = fit_lgbm(
        train_select, valid, forward_columns, categorical,
        forward_config, random_state=state,
    )

    ranking_specs = specs
    models = {
        "manual": (manual_model, manual_columns, categorical, None),
        "auto_gain": (auto_model, top_auto, (), None),
        "auto_forward": (forward_model, forward_columns, categorical, None),
        "decay": (decay_model, manual_columns + decay_columns, categorical, None),
        "selected": (selected_model, selected_columns, selected_categories, None),
        "linear": (linear_model, manual_columns, categorical, predict_linear),
    }
    rankings, ranking_log = _ranking_result(
        dataset,
        log,
        test,
        models,
        ranking_specs,
        state,
        source_log=log.iloc[: int(parts["test"][0])].copy(),
        **candidate_options,
    )
    ranking_metrics = {
        f"{name}_candidate_{metric}10": result[key]
        for name, result in rankings.items()
        for metric, key in (("ndcg", "ndcg10"), ("mrr", "mrr10"), ("recall", "recall10"))
    }
    event_auc_uncertainty = None
    if "user" in test.columns:
        event_auc_uncertainty = cluster_bootstrap_auc_comparison(
            test["binary_label"].to_numpy(),
            {
                "manual": predict_proba(manual_model, test, manual_columns, categorical),
                "auto_gain": predict_proba(auto_model, test, top_auto),
            },
            test["user"].to_numpy(),
            baseline="manual",
            seed=state,
            n_boot=_AUC_BOOTSTRAP_REPLICATES,
        )

    return {
        "fold": fold,
        "rolling_protocol_version": ROLLING_PROTOCOL_VERSION,
        "history_mode": history_mode,
        "history_metadata": {
            role: {
                "prefix_end_exclusive": int(positions[0]),
                "prefix_end_timestamp": int(log.iloc[positions[0]]["timestamp"]),
                "frozen_within_role": history_mode == "role_frozen" or role == "test",
            }
            for role, positions in parts.items()
        },
        "seed": seed,
        "selected": selected,
        "ranking_queries": int(ranking_log["query_id"].nunique()) if not ranking_log.empty else 0,
        "ranking_queries_with_positive": int(
            ranking_log.groupby("query_id")["label"].sum().gt(0).sum()
        ) if not ranking_log.empty else 0,
        "ranking_protocol": (
            "logged_impression_candidates" if dataset == "mind"
            else "sampled_unseen_item_discrimination"
        ),
        "candidate_protocol": (
            ranking_log.attrs.get("candidate_protocol", {})
            if dataset != "mind"
            else {"name": "logged_impression_candidates"}
        ),
        "split_metadata": split_metadata(log, parts),
        "manual_select_auc": manual_select,
        "auto_gain_select_auc": auto_select,
        "auto_forward_select_auc": forward_select_auc,
        "decay_select_auc": decay_select,
        "linear_select_auc": linear_select,
        "manual_test_auc": auc(manual_model, test, manual_columns, categorical),
        "auto_gain_test_auc": auc(auto_model, test, top_auto),
        "auto_forward_test_auc": auc(forward_model, test, forward_columns, categorical),
        "decay_test_auc": auc(decay_model, test, manual_columns + decay_columns, categorical),
        "linear_test_auc": logistic_auc(linear_model, test, manual_columns),
        "selected_test_auc": auc(selected_model, test, selected_columns, selected_categories),
        "event_auc_uncertainty": event_auc_uncertainty,
        "event_auc_estimand": ROLLING_AUC_UNCERTAINTY["estimand"],
        "event_auc_uncertainty_unit": ROLLING_AUC_UNCERTAINTY["unit"],
        "event_auc_uncertainty_conditional_on": "fixed_fitted_models_features_and_observed_test_population",
        "event_auc_comparison": "paired_auto_gain_minus_manual",
        **ranking_metrics,
        "ranking_query_metrics": {
            name: scores.get("query_metrics", {}) for name, scores in rankings.items()
        },
        "ranking_ndcg10_uncertainty": {
            name: {
                "estimand": "query_weighted_macro_mean_ndcg10",
                "unit": "user_cluster_resampling_preserving_query_counts",
                "ci95": scores.get("ndcg10_query_weighted_cluster_ci95"),
                "delta_vs_manual_ci95": scores.get("ndcg10_delta_vs_manual_query_weighted_cluster_ci95"),
            }
            for name, scores in rankings.items() if "query_metrics" in scores
        },
        "forward_features": len(forward_added),
        "top_gain": list(gains.head(5).index),
        "forward_history": forward_history,
        "candidate_configuration": {
            "n_negatives": int(candidate_options.get("candidate_n_negatives", 99)),
            "max_queries": int(candidate_options.get("candidate_max_queries", 400)),
            "sampling": candidate_options.get("candidate_sampling", "uniform"),
            "catalog": candidate_options.get("candidate_catalog", "cohort"),
        },
    }


def run_rolling_frame(
    frame, seed=42, n_folds=3, dataset="synthetic",
    forward_max_candidates=60, forward_steps=8,
    history_mode="role_frozen", candidate_n_negatives=99, candidate_max_queries=400,
    candidate_sampling="uniform", candidate_catalog="cohort", item_catalog=None,
):
    """Evaluate folds; role_frozen uses only the prefix before each role."""
    if history_mode not in {"prequential", "role_frozen"}:
        raise ValueError("history_mode must be prequential or role_frozen")
    if candidate_sampling not in {"uniform", "popularity", "recent"}:
        raise ValueError("invalid candidate_sampling")
    if candidate_catalog not in {"cohort", "full"}:
        raise ValueError("candidate_catalog must be cohort or full")
    for name, value, minimum in (("candidate_n_negatives", candidate_n_negatives, 0),
                                 ("candidate_max_queries", candidate_max_queries, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if candidate_catalog == "full":
        from candidate_eval import validate_static_catalog
        item_catalog = validate_static_catalog(item_catalog)
    protocol = dict(history_mode=history_mode, candidate_n_negatives=int(candidate_n_negatives),
                    candidate_max_queries=int(candidate_max_queries), candidate_sampling=candidate_sampling,
                    candidate_catalog=candidate_catalog, static_catalog_validated=candidate_catalog == "full",
                    candidate_parameters_applied=dataset != "mind", ranking_history="frozen_prefix_before_test")
    source_splits = tuple(frame.attrs.get("mind_metadata", {}).get("splits", ()))
    log, folds = rolling_time_splits(frame, n_folds=n_folds)
    results = [
        _fold_result(
            log, parts, seed, fold, dataset,
            forward_max_candidates=forward_max_candidates,
            forward_steps=forward_steps,
            history_mode=history_mode, candidate_n_negatives=candidate_n_negatives,
            candidate_max_queries=candidate_max_queries, candidate_sampling=candidate_sampling,
            candidate_catalog=candidate_catalog, item_catalog=item_catalog,
        )
        for fold, parts in enumerate(folds)
    ]
    summary = _summary(results)
    return {
        "dataset": dataset,
        "seed": seed,
        "folds": results,
        "summary": summary.reset_index().to_dict("records"),
        "evaluation": "rolling_temporal",
        "rolling_protocol_version": ROLLING_PROTOCOL_VERSION,
        "rolling_protocol": protocol,
        "metadata": {"rolling_protocol_version": ROLLING_PROTOCOL_VERSION, **protocol},
        "evaluation_uncertainty": {"event_level_auc": ROLLING_AUC_UNCERTAINTY.copy()},
        "forward_search": {"max_candidates": forward_max_candidates, "steps": forward_steps},
        "mind_metadata": (
            summarize_mind_metadata(log, splits=source_splits)
            if dataset == "mind"
            else None
        ),
        "kion_metadata": log.attrs.get("kion_metadata") if dataset == "kion" else None,
        "mind_split_protocol": (
            "combined_requested_splits_then_chronological_split"
            if dataset == "mind"
            else None
        ),
    }


def _summary(results):
    table = pd.DataFrame(results)
    metric_columns = [
        column for column in table.select_dtypes(include="number").columns
        if column.endswith("_auc") or "_candidate_ndcg" in column
        or "_candidate_mrr" in column or "_candidate_recall" in column
    ]
    numeric = table[metric_columns]
    return pd.DataFrame({
        "mean": numeric.mean(),
        "std": numeric.std(ddof=1).fillna(0.0),
    })


def summarize_rolling_runs(runs):
    """Aggregate fold-level metrics across all requested seeds and folds."""
    folds = [fold for run in runs for fold in run["folds"]]
    return _summary(folds).reset_index().to_dict("records")


def run_rolling_many(
    frame, seeds=(42,), n_folds=3, dataset="synthetic",
    forward_max_candidates=60, forward_steps=8,
    history_mode="role_frozen", candidate_n_negatives=99, candidate_max_queries=400,
    candidate_sampling="uniform", candidate_catalog="cohort", item_catalog=None,
):
    return [
        run_rolling_frame(
            frame, seed=seed, n_folds=n_folds, dataset=dataset,
            forward_max_candidates=forward_max_candidates,
            forward_steps=forward_steps,
            history_mode=history_mode, candidate_n_negatives=candidate_n_negatives,
            candidate_max_queries=candidate_max_queries, candidate_sampling=candidate_sampling,
            candidate_catalog=candidate_catalog, item_catalog=item_catalog,
        )
        for seed in seeds
    ]
