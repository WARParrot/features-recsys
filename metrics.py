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
