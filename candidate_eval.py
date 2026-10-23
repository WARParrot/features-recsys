import numpy as np
import pandas as pd

from causal_features import build_baseline_features
from metrics import mrr_at_k, ndcg_at_k, recall_at_k
from models import matrix


def validate_static_catalog(item_catalog):
    """Validate and normalize a static item catalog for candidate sampling."""
    if not isinstance(item_catalog, pd.DataFrame):
        raise ValueError("static catalog requires an item_catalog DataFrame")
    required = {"item", "item_cat", "first_seen"}
    missing = required - set(item_catalog.columns)
    if missing:
        raise ValueError(f"static catalog is missing columns: {sorted(missing)}")
    if item_catalog.empty or item_catalog["item"].duplicated().any():
        raise ValueError("static catalog must contain one row per item")
    catalog = item_catalog.copy()
    item_values = pd.to_numeric(catalog["item"], errors="coerce")
    category_values = pd.to_numeric(catalog["item_cat"], errors="coerce")
    first_seen = pd.to_numeric(catalog["first_seen"], errors="coerce")
    for values in (item_values, category_values, first_seen):
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("static catalog item, item_cat and first_seen must be finite integers")
    if first_seen.isna().any() or not np.isfinite(first_seen.to_numpy(dtype=float)).all():
        raise ValueError("static catalog first_seen must be finite")
    catalog["first_seen"] = first_seen.astype("int64")
    catalog["item"] = item_values.astype("int64")
    catalog["item_cat"] = category_values.astype("int64")
    return catalog


def sampled_candidate_log(
    events,
    full_log,
    n_items,
    n_negatives=99,
    max_queries=400,
    seed=7,
    item_catalog=None,
    catalog_protocol="historical_seen_items",
    sampling="uniform",
    candidate_catalog=None,
):
    """Sample unseen negatives from a temporally available item catalog.

    KION has no impression log, so this is explicitly a sampled unseen-item
    discrimination protocol. By default an item must have been observed in
    ``full_log`` strictly before the query timestamp; this avoids using future
    catalog entries as candidates. ``item_catalog`` may provide static metadata,
    but cannot make an item historically available.
    """
    required = {"binary_label", "event_id", "user", "item", "timestamp", "item_cat", "user_segment"}
    missing = (required - set(events.columns)) | (required - set(full_log.columns))
    if missing:
        raise KeyError(f"candidate inputs are missing columns: {sorted(missing)}")
    if full_log.empty:
        raise ValueError("full_log must not be empty")
    if not isinstance(n_items, (int, np.integer)) or n_items <= 0:
        raise ValueError("n_items must be a positive integer")
    if not isinstance(n_negatives, (int, np.integer)) or n_negatives < 0:
        raise ValueError("n_negatives must be a non-negative integer")
    if candidate_catalog is not None:
        if catalog_protocol != "historical_seen_items" and catalog_protocol != candidate_catalog:
            raise ValueError("catalog_protocol and candidate_catalog disagree")
        catalog_protocol = candidate_catalog
    if catalog_protocol not in {"historical_seen_items", "static_catalog"}:
        raise ValueError("catalog_protocol must be 'historical_seen_items' or 'static_catalog'")
    if sampling not in {"uniform", "popularity", "recent"}:
        raise ValueError("sampling must be 'uniform', 'popularity', or 'recent'")
    if not isinstance(max_queries, (int, np.integer)) or max_queries <= 0:
        raise ValueError("max_queries must be a positive integer")
    if catalog_protocol == "static_catalog":
        item_catalog = validate_static_catalog(item_catalog)
    positives = events.loc[events["binary_label"] == 1]
    if len(positives) > max_queries:
        positives = positives.sample(max_queries, random_state=seed)
    rng = np.random.default_rng(seed)
    catalog = full_log if item_catalog is None else item_catalog
    catalog_missing = {"item", "item_cat"} - set(catalog.columns)
    if catalog_missing:
        raise KeyError(f"item catalog is missing columns: {sorted(catalog_missing)}")
    item_categories = catalog.drop_duplicates("item").set_index("item")["item_cat"].to_dict()
    if catalog_protocol == "historical_seen_items":
        first_seen = full_log.groupby("item", sort=False)["timestamp"].min().to_dict()
    else:
        item_catalog = validate_static_catalog(item_catalog)
        static_first_seen = item_catalog.set_index("item")["first_seen"]
        if static_first_seen.isna().any() or not np.isfinite(static_first_seen.to_numpy(dtype=float)).all():
            raise ValueError("static_catalog first_seen must be finite for every item")
        first_seen = static_first_seen.astype("int64").to_dict()
    available_items = sorted(item_categories)
    next_event_id = int(full_log["event_id"].max()) + 1
    rows = []
    negative_counts = []
    skipped_positive_count = 0
    for event in positives.itertuples(index=False):
        # Under the historical-catalog estimand, the positive must be an item
        # that was already available before the query.  Otherwise the query
        # would contain a positive outside the candidate universe used for its
        # negatives and ranking metrics would be ill-defined.
        positive_first_seen = first_seen.get(event.item, np.inf)
        if not (positive_first_seen < int(event.timestamp)):
            skipped_positive_count += 1
            continue
        query_id = len(negative_counts)
        prior = full_log.loc[
            (full_log["user"] == event.user) & (full_log["timestamp"] < event.timestamp),
            "item",
        ]
        seen = set(prior)
        available = [
            item for item in available_items
            if item != event.item
            and item not in seen
            and first_seen.get(item, np.inf) < int(event.timestamp)
        ]
        if len(available) == 0:
            continue
        sample_size = min(n_negatives, len(available))
        if sampling == "uniform":
            probabilities = None
        else:
            prior = full_log.loc[full_log["timestamp"] < int(event.timestamp)]
            counts = prior.groupby("item").size().to_dict()
            if sampling == "popularity":
                weights = np.asarray([counts.get(item, 0) for item in available], dtype=float)
            else:
                last_seen = prior.groupby("item")["timestamp"].max().to_dict()
                half_life = 7 * 86400.0
                weights = np.asarray(
                    [np.exp(-(int(event.timestamp) - last_seen[item]) / half_life)
                     if item in last_seen else 0.0 for item in available],
                    dtype=float,
                )
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                probabilities = None
            else:
                # Retain support for static-catalog items without history.
                weights = weights + 1e-12
                probabilities = weights / weights.sum()
        negatives = rng.choice(len(available), sample_size, replace=False, p=probabilities)
        negatives = [available[int(index)] for index in negatives]
        negative_counts.append(len(negatives))
        for item, label in [(event.item, 1), *[(item, 0) for item in negatives]]:
            rows.append(
                {
                    "event_id": next_event_id,
                    "query_id": query_id,
                    "user": event.user,
                    "item": item,
                    "timestamp": int(event.timestamp),
                    "label": label,
                    "is_click": float(label),
                    "is_purchase": 0.0,
                    "binary_label": label,
                    "item_cat": int(item_categories[item]),
                    "user_segment": int(event.user_segment),
                }
            )
            next_event_id += 1
    result = pd.DataFrame(rows, columns=[
        "event_id", "query_id", "user", "item", "timestamp", "label",
        "is_click", "is_purchase", "binary_label", "item_cat", "user_segment",
    ])
    # Do not let construction order (the positive is appended first) resolve
    # score ties.  This is essential for constant baselines such as prior
    # user-item CTR, especially when negatives are sampled as unseen items.
    if not result.empty:
        result["_tie_order"] = rng.random(len(result))
        result = (
            result.sort_values(["query_id", "_tie_order"], kind="mergesort")
            .drop(columns="_tie_order")
            .reset_index(drop=True)
        )
    result.attrs["candidate_protocol"] = {
        "name": "sampled_unseen_item_discrimination",
        "catalog_protocol": catalog_protocol,
        "future_catalog_items_excluded": True,
        "static_catalog_validated": catalog_protocol == "static_catalog",
        "requested_negatives": int(n_negatives),
        "mean_actual_negatives": float(np.mean(negative_counts)) if negative_counts else 0.0,
        "sampling": sampling,
        "candidate_catalog": "cohort" if catalog_protocol == "historical_seen_items" else "full",
        "sampling_weights_source": "pre_query_snapshot" if sampling != "uniform" else None,
        "max_queries": int(max_queries),
        "item_pool_size": int(len(available_items)),
        "declared_item_count": int(n_items),
        "negatives_exclude_previously_seen_items": True,
        "positive_source": "heldout_positive_events",
        "positive_items_historically_available": catalog_protocol != "historical_seen_items" or skipped_positive_count == 0,
        "skipped_positive_queries": int(skipped_positive_count),
        "ranking_claim": "offline sampled discrimination, not logged-exposure ranking",
        "queries": int(result["query_id"].nunique()) if not result.empty else 0,
        "rows": int(len(result)),
    }
    return result


def candidate_features(full_log, candidate_log, specs=None, decay_builder=None):
    if "query_id" not in candidate_log.columns:
        raise KeyError("candidate_log is missing columns: ['query_id']")
    candidates = candidate_log.drop(columns="query_id").copy()
    if "is_positive_watch" in full_log:
        # Candidate labels are never feature sources, but a missing placeholder
        # would contaminate masked prefix sums via NaN * 0.
        candidates["is_positive_watch"] = 0.0
    combined = pd.concat([full_log, candidates], ignore_index=True)
    source_mask = np.r_[np.ones(len(full_log)), np.zeros(len(candidates))]
    features = build_baseline_features(combined, source_mask=source_mask)
    if specs:
        from auto_features import compute_specs

        features = pd.concat(
            [
                features,
                compute_specs(combined, specs, source_mask=source_mask),
            ],
            axis=1,
        )
    if decay_builder:
        features = pd.concat([features, decay_builder(combined, source_mask=source_mask)], axis=1)
    return combined, features.iloc[len(full_log):].reset_index(drop=True)


def score_candidate_sets(
    model,
    candidate_log,
    features,
    columns,
    category_columns=(),
    score_fn=None,
    protocol="sampled_candidate_sets",
):
    required = {"query_id", "label"}
    missing = required - set(candidate_log.columns)
    if missing:
        raise KeyError(f"candidate_log is missing columns: {sorted(missing)}")
    if len(features) != len(candidate_log):
        raise ValueError("candidate feature rows must match candidate log rows")
    if candidate_log.empty:
        return {
            "ndcg10": float("nan"),
            "mrr10": float("nan"),
            "recall10": float("nan"),
            "queries": 0,
            "queries_with_positive": 0,
            "protocol": protocol,
        }
    if score_fn is None:
        scores = model.predict_proba(matrix(features, columns, category_columns))[:, 1]
    else:
        scores = score_fn(model, features, columns)
    if len(scores) != len(candidate_log):
        raise ValueError("score output rows must match candidate log rows")
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise ValueError("score output must contain only finite values")
    scored = candidate_log.copy()
    scored["score"] = scores
    grouped = scored.groupby("query_id", sort=False)
    query_ndcg = []
    query_mrr = []
    query_recall = []
    query_recall_eligible = []
    query_ids = []
    query_users = []
    for query_id, group in grouped:
        query_ids.append(query_id)
        has_positive = bool((group["label"] > 0).any())
        query_recall_eligible.append(has_positive)
        query_ndcg.append(ndcg_at_k(group.assign(_group="q"), "score", k=10, label_col="label", group_col="_group"))
        query_mrr.append(mrr_at_k(group.assign(_group="q"), "score", k=10, label_col="label", group_col="_group"))
        query_recall.append(recall_at_k(group.assign(_group="q"), "score", k=10, label_col="label", group_col="_group", skip_no_positive=False))
        query_users.append(group["user"].iloc[0] if "user" in group else "unknown")
    # Keep zero placeholders for aligned legacy vectors, but aggregate Recall
    # only over queries with positives, as recall_at_k historically does.
    eligible_recall = [value for value, eligible in zip(query_recall, query_recall_eligible) if eligible]
    recall_mean = float(np.mean(eligible_recall)) if eligible_recall else 0.0
    return {
        "ndcg10": ndcg_at_k(scored, "score", k=10, label_col="label", group_col="query_id"),
        "mrr10": mrr_at_k(scored, "score", k=10, label_col="label", group_col="query_id"),
        "recall10": recall_mean,
        "queries": int(scored["query_id"].nunique()),
        "queries_with_positive": int(grouped["label"].sum().gt(0).sum()),
        "protocol": protocol,
        "query_metrics": {
            "ndcg10": query_ndcg,
            "mrr10": query_mrr,
            "recall10": query_recall,
            "recall10_eligible": query_recall_eligible,
            "query_ids": query_ids,
            "recall10_eligible_queries": int(sum(query_recall_eligible)),
            "recall10_eligibility_policy": (
                "queries with at least one positive label; zero-positive queries excluded "
                "from scalar mean and stored as 0.0; no eligible queries yields scalar 0.0"
            ),
            "users": query_users,
        },
        "estimand": "query_weighted_macro_mean",
        "query_count": int(len(query_ndcg)),
    }


def score_logged_impressions(model, impression_log, features, columns, category_columns=(), score_fn=None):
    """Score candidates from real impression groups without adding negatives."""
    required = {"query_id", "label"}
    missing = required - set(impression_log.columns)
    if missing:
        raise KeyError(f"impression_log is missing columns: {sorted(missing)}")
    return score_candidate_sets(
        model,
        impression_log,
        features,
        columns,
        category_columns=category_columns,
        score_fn=score_fn,
        protocol="logged_impression_candidates",
    )


def causal_group_popularity_scores(full_log, candidate_log, key_col="item", half_life=None):
    """Return causal cumulative or decayed popularity for an arbitrary key."""
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive")
    if candidate_log.empty:
        return np.empty(0, dtype=float)
    required = {key_col, "timestamp", "event_id"}
    missing = (required - set(full_log.columns)) | (required - set(candidate_log.columns))
    if missing:
        raise KeyError(f"popularity inputs are missing columns: {sorted(missing)}")
    events = full_log.sort_values(["timestamp", "event_id"], kind="mergesort")
    candidates = candidate_log.sort_values(["timestamp", "event_id"], kind="mergesort")
    counts = {}
    last_updates = {}
    event_pos = 0
    rate = None if half_life is None else np.log(2.0) / float(half_life)
    scores = np.zeros(len(candidates), dtype=float)
    event_times = events["timestamp"].to_numpy(dtype=np.int64)
    event_keys = events[key_col].to_numpy()
    candidate_times = candidates["timestamp"].to_numpy(dtype=np.int64)
    candidate_keys = candidates[key_col].to_numpy()
    # Keep each key at its last update time so decay touches only requested keys.
    for position, candidate_time in enumerate(candidate_times):
        while event_pos < len(events) and event_times[event_pos] < candidate_time:
            current_time = event_times[event_pos]
            key = event_keys[event_pos]
            value = counts.get(key, 0.0)
            if rate is not None:
                previous_time = last_updates.get(key)
                if previous_time is not None:
                    value *= np.exp(-rate * (current_time - previous_time))
            counts[key] = value + 1.0
            if rate is not None:
                last_updates[key] = current_time
            event_pos += 1
        key = candidate_keys[position]
        value = counts.get(key, 0.0)
        if rate is not None:
            previous_time = last_updates.get(key)
            if previous_time is not None:
                value *= np.exp(-rate * (candidate_time - previous_time))
        scores[position] = value
    return pd.Series(scores, index=candidates.index).reindex(candidate_log.index).to_numpy()


def causal_item_popularity_scores(full_log, candidate_log, half_life=None):
    """Return pre-query item-popularity scores for candidate rows."""
    return causal_group_popularity_scores(
        full_log, candidate_log, key_col="item", half_life=half_life
    )


def predict_linear(model, features, columns):
    return model.predict_proba(features[columns])[:, 1]
