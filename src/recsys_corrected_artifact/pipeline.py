import numpy as np
import pandas as pd

from .auto_features import compute_specs, enumerate_specs
from .candidate_eval import (
    candidate_features as build_candidate_features,
    causal_group_popularity_scores,
    causal_item_popularity_scores,
    predict_linear,
    sampled_candidate_log,
    score_candidate_sets,
    score_logged_impressions,
)
from .data import generate_synthetic, split_metadata, time_split
from .causal_features import BASELINE_FEATURES, CATEGORICAL_FEATURES, build_baseline_features
from .decay_features import build_decay_features
from .models import auc, fit_lgbm, fit_lgbm_fixed
from .ranking_references import score_reference_rankings
from .feature_selectors import forward_select, gain_top_k
from .linear_model import fit_logistic, logistic_auc, tune_logistic
from .mind_data import load_mind, summarize_mind_metadata


CONFIG_GRID = [
    {"num_leaves": 15, "min_child_samples": 30, "learning_rate": 0.03, "reg_lambda": 0.5},
    {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.04, "reg_lambda": 1.0},
    {"num_leaves": 63, "min_child_samples": 80, "learning_rate": 0.025, "reg_lambda": 2.0},
]


def tune_model(train, select, columns, categories=(), random_state=17):
    if not columns:
        raise ValueError("at least one feature column is required")
    missing_train = set(columns) - set(train.columns)
    missing_select = set(columns) - set(select.columns)
    if missing_train or missing_select:
        raise KeyError(
            f"missing features: train={sorted(missing_train)}, select={sorted(missing_select)}"
        )
    trials = []
    for config in CONFIG_GRID:
        model = fit_lgbm_fixed(
            train,
            columns,
            categories,
            config=config,
            random_state=random_state,
        )
        trials.append((auc(model, select, columns, categories), config))
    return max(trials, key=lambda x: x[0])


def split_frame(features, parts):
    return {
        name: features.iloc[index].copy()
        for name, index in parts.items()
    }


def _mind_parts(log, protocol):
    if protocol == "combined_temporal":
        return time_split(log)[1]
    if protocol != "official_train_dev":
        raise ValueError("mind_protocol must be 'combined_temporal' or 'official_train_dev'")
    if "source_split" not in log.columns:
        raise ValueError("MIND frame is missing source_split metadata")
    train_rows = np.flatnonzero(log["source_split"].to_numpy() == "train")
    dev_rows = np.flatnonzero(log["source_split"].to_numpy() == "dev")
    if not len(train_rows) or not len(dev_rows):
        raise ValueError("official_train_dev requires both train and dev MIND splits")
    _, internal = time_split(log.iloc[train_rows].reset_index(drop=True))
    return {
        name: (train_rows[index] if name != "test" else dev_rows)
        for name, index in internal.items()
    }


def run(seed=42, n_events=30_000):
    log = generate_synthetic(n_events=n_events, seed=seed)
    log, parts = time_split(log)
    split_info = split_metadata(log, parts)
    test_source_mask = np.ones(len(log), dtype=bool)
    test_source_mask[parts["test"]] = False
    baseline = build_baseline_features(log, source_mask=test_source_mask)
    specs = enumerate_specs()
    auto = compute_specs(log, specs, source_mask=test_source_mask)
    decay = build_decay_features(log, source_mask=test_source_mask)

    manual_columns = BASELINE_FEATURES + CATEGORICAL_FEATURES
    auto_columns = auto.columns.tolist()
    decay_columns = decay.columns.tolist()
    full = baseline.join(auto).join(decay)
    frames = split_frame(full, parts)

    train, select, valid, test = (frames[name] for name in ("train", "select", "valid", "test"))
    manual_auc, manual_config = tune_model(train, select, manual_columns, CATEGORICAL_FEATURES, random_state=seed)
    top_auto, gains = gain_top_k(train, auto_columns, k=40, random_state=seed)
    auto_auc, auto_config = tune_model(train, select, top_auto, random_state=seed)

    forward_columns, forward_added, forward_history = forward_select(
        train, select, auto_columns, base_columns=manual_columns,
        categorical_columns=CATEGORICAL_FEATURES, k=8, random_state=seed,
    )
    forward_auc, forward_config = tune_model(train, select, forward_columns, CATEGORICAL_FEATURES, random_state=seed)
    decay_auc, decay_config = tune_model(train, select, manual_columns + decay_columns, CATEGORICAL_FEATURES, random_state=seed)
    linear_select_auc, linear_c = tune_logistic(train, select, manual_columns, CATEGORICAL_FEATURES, random_state=seed)


    choices = {
        "manual": (manual_auc, manual_columns, CATEGORICAL_FEATURES, manual_config),
        "auto_gain": (auto_auc, top_auto, (), auto_config),
        "auto_forward": (forward_auc, forward_columns, CATEGORICAL_FEATURES, forward_config),
        "decay": (decay_auc, manual_columns + decay_columns, CATEGORICAL_FEATURES, decay_config),
    }
    chosen = max(choices, key=lambda name: choices[name][0])
    _, chosen_columns, chosen_categories, chosen_config = choices[chosen]
    train_select = pd.concat([train, select], axis=0).sort_values(["timestamp", "event_id"])
    chosen_model = fit_lgbm(train_select, valid, chosen_columns, chosen_categories, chosen_config, random_state=seed)
    manual_model = fit_lgbm(train_select, valid, manual_columns, CATEGORICAL_FEATURES, manual_config, random_state=seed)
    auto_model = fit_lgbm(train_select, valid, top_auto, (), auto_config, random_state=seed)
    forward_model = fit_lgbm(
        train_select, valid, forward_columns, CATEGORICAL_FEATURES,
        forward_config, random_state=seed,
    )
    decay_model = fit_lgbm(
        train_select, valid, manual_columns + decay_columns,
        CATEGORICAL_FEATURES, decay_config, random_state=seed,
    )
    linear_model = fit_logistic(train_select, manual_columns, CATEGORICAL_FEATURES, c=linear_c, random_state=seed)
    test_auc_manual = auc(manual_model, test, manual_columns, CATEGORICAL_FEATURES)
    test_auc_linear = logistic_auc(linear_model, test, manual_columns)
    test_auc_auto_gain = auc(auto_model, test, top_auto)
    test_auc_auto_forward = auc(
        forward_model, test, forward_columns, CATEGORICAL_FEATURES
    )
    test_auc_decay = auc(
        decay_model, test, manual_columns + decay_columns, CATEGORICAL_FEATURES
    )

    history_positions = np.concatenate(
        [parts["train"], parts["select"], parts["valid"]]
    )
    history_log = log.iloc[history_positions].copy()
    candidate_log = sampled_candidate_log(
        test,
        history_log,
        int(log["item"].max()) + 1,
        seed=seed,
        item_catalog=log,
    )
    _, candidate_feature_frame = build_candidate_features(
        history_log, candidate_log, specs=specs, decay_builder=build_decay_features
    )
    manual_ranking = score_candidate_sets(
        manual_model, candidate_log, candidate_feature_frame,
        manual_columns, CATEGORICAL_FEATURES,
    )
    auto_ranking = score_candidate_sets(
        auto_model, candidate_log, candidate_feature_frame, top_auto
    )
    forward_ranking = score_candidate_sets(
        forward_model, candidate_log, candidate_feature_frame,
        forward_columns, CATEGORICAL_FEATURES,
    )
    decay_ranking = score_candidate_sets(
        decay_model, candidate_log, candidate_feature_frame,
        manual_columns + decay_columns, CATEGORICAL_FEATURES,
    )
    linear_ranking = score_candidate_sets(
        linear_model, candidate_log, candidate_feature_frame,
        manual_columns, CATEGORICAL_FEATURES, score_fn=predict_linear,
    )
    selected_ranking = score_candidate_sets(
        chosen_model, candidate_log, candidate_feature_frame,
        chosen_columns, chosen_categories,
    )
    reference_rankings = score_reference_rankings(
        history_log, candidate_log, candidate_feature_frame, score_candidate_sets
    )
    popularity_ranking = reference_rankings["popularity"]
    decay_popularity_ranking = reference_rankings["decay_popularity"]
    category_popularity_ranking = reference_rankings["category_popularity"]
    user_item_ctr_ranking = reference_rankings["user_item_ctr"]

    result = {
        "seed": seed,
        "selected": chosen,
        "split_metadata": split_info,
        "manual_select_auc": manual_auc,
        "auto_gain_select_auc": auto_auc,
        "auto_forward_select_auc": forward_auc,
        "decay_select_auc": decay_auc,
        "linear_select_auc": linear_select_auc,
        "manual_test_auc": test_auc_manual,
        "auto_gain_test_auc": test_auc_auto_gain,
        "auto_forward_test_auc": test_auc_auto_forward,
        "decay_test_auc": test_auc_decay,
        "linear_test_auc": test_auc_linear,
        "selected_test_auc": auc(
            chosen_model, test, chosen_columns, chosen_categories
        ),
        "ranking_queries": manual_ranking["queries"],
        "ranking_queries_with_positive": manual_ranking["queries_with_positive"],
        "ranking_positive_query_fraction": (
            manual_ranking["queries_with_positive"] / manual_ranking["queries"]
            if manual_ranking["queries"] else 0.0
        ),
        "ranking_protocol": manual_ranking["protocol"],
        "manual_candidate_ndcg10": manual_ranking["ndcg10"],
        "manual_candidate_mrr10": manual_ranking["mrr10"],
        "manual_candidate_recall10": manual_ranking["recall10"],
        "auto_gain_candidate_ndcg10": auto_ranking["ndcg10"],
        "auto_gain_candidate_mrr10": auto_ranking["mrr10"],
        "auto_gain_candidate_recall10": auto_ranking["recall10"],
        "auto_forward_candidate_ndcg10": forward_ranking["ndcg10"],
        "auto_forward_candidate_mrr10": forward_ranking["mrr10"],
        "auto_forward_candidate_recall10": forward_ranking["recall10"],
        "decay_candidate_ndcg10": decay_ranking["ndcg10"],
        "decay_candidate_mrr10": decay_ranking["mrr10"],
        "decay_candidate_recall10": decay_ranking["recall10"],
        "linear_candidate_ndcg10": linear_ranking["ndcg10"],
        "linear_candidate_mrr10": linear_ranking["mrr10"],
        "linear_candidate_recall10": linear_ranking["recall10"],
        "selected_candidate_ndcg10": selected_ranking["ndcg10"],
        "selected_candidate_mrr10": selected_ranking["mrr10"],
        "selected_candidate_recall10": selected_ranking["recall10"],
        "popularity_candidate_ndcg10": popularity_ranking["ndcg10"],
        "popularity_candidate_mrr10": popularity_ranking["mrr10"],
        "popularity_candidate_recall10": popularity_ranking["recall10"],
        "decay_popularity_candidate_ndcg10": decay_popularity_ranking["ndcg10"],
        "decay_popularity_candidate_mrr10": decay_popularity_ranking["mrr10"],
        "decay_popularity_candidate_recall10": decay_popularity_ranking["recall10"],
        "category_popularity_candidate_ndcg10": category_popularity_ranking["ndcg10"],
        "category_popularity_candidate_mrr10": category_popularity_ranking["mrr10"],
        "category_popularity_candidate_recall10": category_popularity_ranking["recall10"],
        "user_item_ctr_candidate_ndcg10": user_item_ctr_ranking["ndcg10"],
        "user_item_ctr_candidate_mrr10": user_item_ctr_ranking["mrr10"],
        "user_item_ctr_candidate_recall10": user_item_ctr_ranking["recall10"],
        "forward_features": len(forward_added),
        "top_gain": list(gains.head(5).index),
        "forward_history": forward_history,
        "candidate_protocol": candidate_log.attrs.get("candidate_protocol", {}),
    }
    return result


def run_many(seeds=(42,), n_events=30_000):
    results = [run(seed, n_events) for seed in seeds]
    return results, _summary(results)


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


def _limit_mind_queries(frame, max_impressions):
    if max_impressions is None:
        return frame
    if max_impressions <= 0:
        raise ValueError("max_impressions must be positive")
    queries = (
        frame[["query_id", "timestamp"]]
        .drop_duplicates("query_id")
        .sort_values(["timestamp", "query_id"], kind="mergesort")
        .head(max_impressions)["query_id"]
    )
    return frame[frame["query_id"].isin(set(queries))].reset_index(drop=True)


def run_mind(
    root,
    seed=42,
    max_impressions=None,
    strict_mind_metadata=False,
    mind_protocol="combined_temporal",
):
    """Run the same feature-search comparison on logged MIND impressions.

    Unlike the synthetic path, the test ranking metrics use the candidates
    that were actually shown in each MIND impression.  No sampled unseen-item
    negatives are introduced.  Features are still point-in-time aggregates;
    equal-timestamp candidates are atomic and cannot see one another.
    """
    log = load_mind(root, strict=strict_mind_metadata)
    source_splits = tuple(log.attrs.get("mind_metadata", {}).get("splits", ()))
    log = _limit_mind_queries(log, max_impressions)
    mind_metadata = summarize_mind_metadata(log, splits=source_splits)
    parts = _mind_parts(log, mind_protocol)
    split_info = split_metadata(log, parts)
    test_source_mask = np.ones(len(log), dtype=bool)
    test_source_mask[parts["test"]] = False
    source_mask = test_source_mask
    if mind_protocol == "official_train_dev":
        source_mask = log["source_split"].eq("train").to_numpy()
    reference_log = log.loc[source_mask].copy()
    baseline = build_baseline_features(log, source_mask=source_mask)
    specs = [
        spec for spec in enumerate_specs()
        if set(spec["keys"]).issubset(log.columns)
    ]
    auto = compute_specs(log, specs, source_mask=source_mask)
    decay = build_decay_features(log, source_mask=source_mask)

    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in log.columns]
    manual_columns = [
        column for column in BASELINE_FEATURES + categorical_columns if column in baseline.columns
    ]
    auto_columns = auto.columns.tolist()
    decay_columns = decay.columns.tolist()
    full = baseline.join(auto).join(decay)
    frames = split_frame(full, parts)
    train, select, valid, test = (frames[name] for name in ("train", "select", "valid", "test"))

    manual_auc, manual_config = tune_model(train, select, manual_columns, categorical_columns, random_state=seed)
    top_auto, gains = gain_top_k(train, auto_columns, k=40, random_state=seed)
    auto_auc, auto_config = tune_model(train, select, top_auto, random_state=seed)

    forward_columns, forward_added, forward_history = forward_select(
        train,
        select,
        auto_columns,
        base_columns=manual_columns,
        categorical_columns=categorical_columns,
        k=8,
        random_state=seed,
    )
    forward_auc, forward_config = tune_model(
        train, select, forward_columns, categorical_columns, random_state=seed
    )
    decay_auc, decay_config = tune_model(
        train, select, manual_columns + decay_columns, categorical_columns, random_state=seed
    )
    linear_select_auc, linear_c = tune_logistic(
        train, select, manual_columns, categorical_columns, random_state=seed
    )

    choices = {
        "manual": (manual_auc, manual_columns, categorical_columns, manual_config),
        "auto_gain": (auto_auc, top_auto, (), auto_config),
        "auto_forward": (forward_auc, forward_columns, categorical_columns, forward_config),
        "decay": (
            decay_auc,
            manual_columns + decay_columns,
            categorical_columns,
            decay_config,
        ),
    }
    chosen = max(choices, key=lambda name: choices[name][0])
    _, chosen_columns, chosen_categories, chosen_config = choices[chosen]
    train_select = pd.concat([train, select], axis=0).sort_values(["timestamp", "event_id"])
    chosen_model = fit_lgbm(train_select, valid, chosen_columns, chosen_categories, chosen_config, random_state=seed)
    manual_model = fit_lgbm(train_select, valid, manual_columns, categorical_columns, manual_config, random_state=seed)
    auto_model = fit_lgbm(train_select, valid, top_auto, (), auto_config, random_state=seed)
    forward_model = fit_lgbm(
        train_select, valid, forward_columns, categorical_columns,
        forward_config, random_state=seed,
    )
    decay_model = fit_lgbm(
        train_select, valid, manual_columns + decay_columns,
        categorical_columns, decay_config, random_state=seed,
    )
    linear_model = fit_logistic(train_select, manual_columns, categorical_columns, c=linear_c, random_state=seed)

    manual_ranking = score_logged_impressions(
        manual_model, test, test, manual_columns, categorical_columns
    )
    linear_ranking = score_logged_impressions(
        linear_model,
        test,
        test,
        manual_columns,
        categorical_columns,
        score_fn=predict_linear,
    )
    forward_ranking = score_logged_impressions(
        forward_model, test, test, forward_columns, categorical_columns
    )
    auto_ranking = score_logged_impressions(auto_model, test, test, top_auto)
    decay_ranking = score_logged_impressions(
        decay_model, test, test, manual_columns + decay_columns, categorical_columns
    )
    popularity_ranking = score_logged_impressions(
        None,
        test,
        pd.DataFrame({"popularity": causal_item_popularity_scores(reference_log, test)}),
        ["popularity"],
        score_fn=lambda _model, frame, _columns: frame["popularity"].to_numpy(),
    )
    decay_popularity_ranking = score_logged_impressions(
        None,
        test,
        pd.DataFrame({
            "popularity": causal_item_popularity_scores(
                reference_log, test, half_life=7 * 86400
            )
        }),
        ["popularity"],
        score_fn=lambda _model, frame, _columns: frame["popularity"].to_numpy(),
    )
    category_popularity_ranking = score_logged_impressions(
        None,
        test,
        pd.DataFrame({
            "popularity": causal_group_popularity_scores(
                reference_log, test, key_col="item_cat"
            )
        }),
        ["popularity"],
        score_fn=lambda _model, frame, _columns: frame["popularity"].to_numpy(),
    )

    selected_ranking = score_logged_impressions(
        chosen_model, test, test, chosen_columns, chosen_categories
    )
    user_item_ctr_ranking = score_logged_impressions(
        None,
        test,
        test[["ui_ctr_all"]].rename(columns={"ui_ctr_all": "ui_ctr"}),
        ["ui_ctr"],
        score_fn=lambda _model, frame, _columns: frame["ui_ctr"].to_numpy(),
    )

    return {
        "dataset": "mind",
        "seed": seed,
        "selected": chosen,
        "queries": int(log["query_id"].nunique()),
        "events": int(len(log)),
        "ranking_queries": manual_ranking["queries"],
        "ranking_queries_with_positive": manual_ranking["queries_with_positive"],
        "ranking_positive_query_fraction": (
            manual_ranking["queries_with_positive"] / manual_ranking["queries"]
            if manual_ranking["queries"] else 0.0
        ),
        "ranking_protocol": manual_ranking["protocol"],
        "split_metadata": split_info,
        "mind_metadata": mind_metadata,
        "mind_split_protocol": mind_protocol,
        "manual_select_auc": manual_auc,
        "auto_gain_select_auc": auto_auc,
        "auto_forward_select_auc": forward_auc,
        "decay_select_auc": decay_auc,
        "linear_select_auc": linear_select_auc,
        "manual_test_auc": auc(manual_model, test, manual_columns, categorical_columns),
        "auto_gain_test_auc": auc(auto_model, test, top_auto),
        "auto_forward_test_auc": auc(forward_model, test, forward_columns, categorical_columns),
        "decay_test_auc": auc(decay_model, test, manual_columns + decay_columns, categorical_columns),
        "linear_test_auc": logistic_auc(linear_model, test, manual_columns),
        "selected_test_auc": auc(chosen_model, test, chosen_columns, chosen_categories),
        "manual_candidate_ndcg10": manual_ranking["ndcg10"],
        "manual_candidate_mrr10": manual_ranking["mrr10"],
        "manual_candidate_recall10": manual_ranking["recall10"],
        "auto_gain_candidate_ndcg10": auto_ranking["ndcg10"],
        "auto_gain_candidate_mrr10": auto_ranking["mrr10"],
        "auto_gain_candidate_recall10": auto_ranking["recall10"],
        "auto_forward_candidate_ndcg10": forward_ranking["ndcg10"],
        "auto_forward_candidate_mrr10": forward_ranking["mrr10"],
        "auto_forward_candidate_recall10": forward_ranking["recall10"],
        "decay_candidate_ndcg10": decay_ranking["ndcg10"],
        "decay_candidate_mrr10": decay_ranking["mrr10"],
        "decay_candidate_recall10": decay_ranking["recall10"],
        "linear_candidate_ndcg10": linear_ranking["ndcg10"],
        "linear_candidate_mrr10": linear_ranking["mrr10"],
        "linear_candidate_recall10": linear_ranking["recall10"],
        "selected_candidate_ndcg10": selected_ranking["ndcg10"],
        "selected_candidate_mrr10": selected_ranking["mrr10"],
        "selected_candidate_recall10": selected_ranking["recall10"],
        "popularity_candidate_ndcg10": popularity_ranking["ndcg10"],
        "popularity_candidate_mrr10": popularity_ranking["mrr10"],
        "popularity_candidate_recall10": popularity_ranking["recall10"],
        "decay_popularity_candidate_ndcg10": decay_popularity_ranking["ndcg10"],
        "decay_popularity_candidate_mrr10": decay_popularity_ranking["mrr10"],
        "decay_popularity_candidate_recall10": decay_popularity_ranking["recall10"],
        "category_popularity_candidate_ndcg10": category_popularity_ranking["ndcg10"],
        "category_popularity_candidate_mrr10": category_popularity_ranking["mrr10"],
        "category_popularity_candidate_recall10": category_popularity_ranking["recall10"],
        "user_item_ctr_candidate_ndcg10": user_item_ctr_ranking["ndcg10"],
        "user_item_ctr_candidate_mrr10": user_item_ctr_ranking["mrr10"],
        "user_item_ctr_candidate_recall10": user_item_ctr_ranking["recall10"],
        "forward_features": len(forward_added),
        "top_gain": list(gains.head(5).index),
        "forward_history": forward_history,
    }


def run_mind_many(
    root,
    seeds=(42,),
    max_impressions=None,
    strict_mind_metadata=False,
    mind_protocol="combined_temporal",
):
    results = [
        run_mind(
            root,
            seed=seed,
            max_impressions=max_impressions,
            strict_mind_metadata=strict_mind_metadata,
            mind_protocol=mind_protocol,
        )
        for seed in seeds
    ]
    return results, _summary(results)
