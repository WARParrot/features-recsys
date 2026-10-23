import numpy as np
import pytest

from metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_mean_ci,
    cluster_bootstrap_mean_ci,
    paired_cluster_bootstrap_mean_ci,
    query_weighted_cluster_bootstrap_mean_ci,
)


def test_bootstrap_is_deterministic_and_constant_difference_exact():
    values = np.array([0.1, 0.2, 0.5])
    assert bootstrap_mean_ci(values, seed=3) == bootstrap_mean_ci(values, seed=3)
    np.testing.assert_allclose(paired_bootstrap_mean_ci(values + 0.2, values), [0.2, 0.2])


def test_cluster_bootstrap_resamples_users_not_queries():
    values = np.array([0.0, 0.0, 1.0, 1.0])
    users = np.array(["u1", "u1", "u2", "u2"])
    np.testing.assert_allclose(cluster_bootstrap_mean_ci(values, users, seed=3), [0.0, 1.0])
    np.testing.assert_allclose(
        paired_cluster_bootstrap_mean_ci(values + 0.2, values, users, seed=3), [0.2, 0.2]
    )


def test_query_weighted_cluster_ci_preserves_query_weighting():
    values = np.array([0.0, 0.0, 1.0])
    users = np.array(["u1", "u1", "u2"])
    interval = query_weighted_cluster_bootstrap_mean_ci(values, users, seed=3, n_boot=100)
    assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_paired_bootstrap_requires_aligned_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_mean_ci([1], [1, 2])
