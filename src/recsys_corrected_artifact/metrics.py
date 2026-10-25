import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_mean_ci(values, seed=17, n_boot=1000, alpha=0.05):
    """Percentile bootstrap CI for a macro mean over independent query values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[samples].mean(axis=1)
    return [
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    ]


def cluster_bootstrap_mean_ci(values, clusters, seed=17, n_boot=1000, alpha=0.05):
    """CI for a user-weighted mean: average within cluster, then resample clusters."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if len(values) != len(clusters):
        raise ValueError("cluster bootstrap inputs must have the same length")
    frame = {}
    for value, cluster in zip(values, clusters):
        if np.isfinite(value):
            frame.setdefault(cluster, []).append(float(value))
    if not frame:
        return [float("nan"), float("nan")]
    cluster_values = np.asarray([np.mean(values) for values in frame.values()])
    return bootstrap_mean_ci(cluster_values, seed=seed, n_boot=n_boot, alpha=alpha)


def query_weighted_cluster_bootstrap_mean_ci(values, clusters, seed=17, n_boot=1000, alpha=0.05):
    """CI for the query-weighted mean while preserving each user's query count.

    The point metric in candidate evaluation is a mean over queries. Resampling
    users and concatenating all their query values therefore targets the same
    estimand, unlike a user-weighted mean.
    """
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if len(values) != len(clusters):
        raise ValueError("cluster bootstrap inputs must have the same length")
    groups = {
        cluster: values[(clusters == cluster) & np.isfinite(values)]
        for cluster in np.unique(clusters)
    }
    groups = [group for group in groups.values() if len(group)]
    if not groups:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(groups), size=(int(n_boot), len(groups)))
    means = np.array([
        np.concatenate([groups[index] for index in row]).mean() for row in choices
    ])
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def paired_query_weighted_cluster_bootstrap_mean_ci(left, right, clusters, seed=17, n_boot=1000, alpha=0.05):
    """Paired query-weighted cluster CI for ``left - right``."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right):
        raise ValueError("paired cluster bootstrap inputs must have the same length")
    return query_weighted_cluster_bootstrap_mean_ci(
        left - right, clusters, seed=seed, n_boot=n_boot, alpha=alpha
    )


def paired_cluster_bootstrap_mean_ci(left, right, clusters, seed=17, n_boot=1000, alpha=0.05):
    """Paired cluster bootstrap CI for ``left - right``."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    clusters = np.asarray(clusters)
    if len(left) != len(right) or len(left) != len(clusters):
        raise ValueError("paired cluster bootstrap inputs must have the same length")
    frame = {}
    for l_value, r_value, cluster in zip(left, right, clusters):
        if np.isfinite(l_value) and np.isfinite(r_value):
            frame.setdefault(cluster, []).append(float(l_value - r_value))
    if not frame:
        return [float("nan"), float("nan")]
    return bootstrap_mean_ci(
        np.asarray([np.mean(values) for values in frame.values()]),
        seed=seed, n_boot=n_boot, alpha=alpha,
    )


def paired_bootstrap_mean_ci(left, right, seed=17, n_boot=1000, alpha=0.05):
    """Percentile bootstrap CI for a paired mean difference ``left - right``."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have the same length")
    return bootstrap_mean_ci(left - right, seed=seed, n_boot=n_boot, alpha=alpha)


def cluster_bootstrap_auc_comparison(
    y_true, scores_by_name, clusters, baseline="manual", seed=17,
    n_boot=1000, alpha=0.05,
):
    """Estimate pooled event-level AUCs and paired differences via user bootstrap.

    Each replicate samples users with replacement and gives every event the
    multiplicity of its user's draw. AUC is then calculated over all sampled
    events (micro/event-weighted), rather than averaging per-user AUCs. All
    models use the same user draws, so each delta from the baseline is paired.
    Replicates without both label classes are omitted from percentile intervals.
    """
    labels = np.asarray(y_true)
    clusters = np.asarray(clusters)
    if labels.ndim != 1 or clusters.ndim != 1:
        raise ValueError("AUC bootstrap inputs must be one-dimensional")
    if len(labels) != len(clusters):
        raise ValueError("AUC bootstrap inputs must have the same length")
    if not scores_by_name:
        raise ValueError("at least one named score vector is required")
    if baseline not in scores_by_name:
        raise ValueError(f"baseline {baseline!r} is missing from scores_by_name")
    classes = np.unique(labels)
    if len(labels) == 0 or len(classes) != 2:
        raise ValueError("event-level AUC requires both label classes")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if int(n_boot) <= 0:
        raise ValueError("n_boot must be positive")

    scores = {}
    for name, values in scores_by_name.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(labels):
            raise ValueError("score vectors must have the same length as labels")
        if not np.isfinite(values).all():
            raise ValueError("score vectors must contain only finite values")
        scores[name] = values
    if np.asarray(labels).dtype.kind in "biufc" and not np.isfinite(labels).all():
        raise ValueError("labels must contain only finite values")

    cluster_ids, cluster_codes = np.unique(clusters, return_inverse=True)
    if len(cluster_ids) == 0:
        raise ValueError("at least one user cluster is required")
    estimates = {
        name: float(roc_auc_score(labels, values))
        for name, values in scores.items()
    }
    # Scores do not change across replicates; sort and group ties only once.
    score_groups = {}
    for name, values in scores.items():
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_group_ids = np.cumsum(
            np.r_[True, sorted_values[1:] != sorted_values[:-1]]
        ) - 1
        group_ids = np.empty(len(values), dtype=np.int64)
        group_ids[order] = sorted_group_ids
        score_groups[name] = (group_ids, int(sorted_group_ids[-1]) + 1)

    positive = labels == classes[1]
    negative = labels == classes[0]
    rng = np.random.default_rng(seed)
    bootstraps = {name: [] for name in scores}
    bootstrap_deltas = {name: [] for name in scores}
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        multiplicities = np.bincount(sampled, minlength=len(cluster_ids))
        weights = multiplicities[cluster_codes]
        positive_weights = weights * positive
        negative_weights = weights * negative
        if positive_weights.sum() <= 0 or negative_weights.sum() <= 0:
            continue
        replicate = {}
        for name, (group_ids, n_groups) in score_groups.items():
            positive_by_score = np.bincount(
                group_ids, weights=positive_weights, minlength=n_groups
            )
            negative_by_score = np.bincount(
                group_ids, weights=negative_weights, minlength=n_groups
            )
            negative_below = np.cumsum(negative_by_score) - negative_by_score
            numerator = np.sum(
                positive_by_score * (negative_below + 0.5 * negative_by_score)
            )
            denominator = positive_by_score.sum() * negative_by_score.sum()
            replicate[name] = float(numerator / denominator)
        for name, value in replicate.items():
            bootstraps[name].append(value)
            bootstrap_deltas[name].append(value - replicate[baseline])

    def percentile(values):
        if not values:
            return [float("nan"), float("nan")]
        return [
            float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)),
        ]

    valid = len(bootstraps[baseline])
    return {
        "estimand": "pooled_event_level_auc",
        "unit": "user_cluster_resampling",
        "paired_comparison": True,
        "conditional_on": ["fixed_scores", "observed_test_population", "fixed_features"],
        "baseline": baseline,
        "observations": int(len(labels)),
        "clusters": int(len(cluster_ids)),
        "requested_replicates": int(n_boot),
        "valid_replicates": valid,
        "metrics": {
            name: {
                "estimate": estimates[name],
                "ci95": percentile(bootstraps[name]),
                f"delta_vs_{baseline}": {
                    "estimate": float(estimates[name] - estimates[baseline]),
                    "ci95": percentile(bootstrap_deltas[name]),
                },
            }
            for name in scores
        },
    }


def auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _dcg(labels, order, k):
    order = np.asarray(order[:k], dtype=np.int64)
    if len(order) == 0:
        return 0.0
    gains = np.power(2.0, np.asarray(labels)[order]) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(frame, score_col, k=10, label_col="binary_label", group_col="user", skip_no_positive=False):
    values = []
    for _, group in frame.groupby(group_col, sort=False):
        labels = group[label_col].to_numpy()
        if skip_no_positive and not np.any(labels > 0):
            continue
        scores = group[score_col].to_numpy()
        predicted = np.argsort(-scores, kind="stable")
        ideal = np.argsort(-labels, kind="stable")
        ideal_dcg = _dcg(labels, ideal, k)
        values.append(_dcg(labels, predicted, k) / ideal_dcg if ideal_dcg else 0.0)
    return float(np.mean(values)) if values else 0.0


def recall_at_k(frame, score_col, k=10, label_col="binary_label", group_col="user", skip_no_positive=True):
    """Compute per-group recall@k using every positive as the denominator."""
    values = []
    for _, group in frame.groupby(group_col, sort=False):
        labels = group[label_col].to_numpy()
        positives = int(np.sum(labels > 0))
        if positives == 0:
            if skip_no_positive:
                continue
            values.append(0.0)
            continue
        order = np.argsort(-group[score_col].to_numpy(), kind="stable")[:k]
        values.append(float(np.sum(labels[order] > 0)) / positives)
    return float(np.mean(values)) if values else 0.0


def mrr_at_k(frame, score_col, k=10, label_col="binary_label", group_col="user", skip_no_positive=False):
    """Mean reciprocal rank of the first positive in each query group."""
    values = []
    for _, group in frame.groupby(group_col, sort=False):
        labels = group[label_col].to_numpy()
        if skip_no_positive and not np.any(labels > 0):
            continue
        order = np.argsort(-group[score_col].to_numpy(), kind="stable")[:k]
        ranked_labels = labels[order]
        positive_positions = np.flatnonzero(ranked_labels > 0)
        values.append(float(1.0 / (positive_positions[0] + 1)) if len(positive_positions) else 0.0)
    return float(np.mean(values)) if values else 0.0
