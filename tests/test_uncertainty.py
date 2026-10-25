import numpy as np
import pytest

from recsys_corrected_artifact.metrics import (
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


def test_user_cluster_bootstrap_auc_reports_event_estimand_and_paired_effect():
    from sklearn.metrics import roc_auc_score
    from recsys_corrected_artifact.metrics import cluster_bootstrap_auc_comparison

    labels = np.array([0, 1, 0, 1, 0, 1])
    users = np.array(["u1", "u1", "u2", "u2", "u2", "u2"])
    manual = np.array([0.1, 0.8, 0.6, 0.5, 0.3, 0.7])
    auto_gain = np.array([0.1, 0.8, 0.6, 0.6, 0.3, 0.7])

    result = cluster_bootstrap_auc_comparison(
        labels,
        {"manual": manual, "auto_gain": auto_gain},
        users,
        baseline="manual",
        seed=13,
        n_boot=200,
    )

    assert result["estimand"] == "pooled_event_level_auc"
    assert result["unit"] == "user_cluster_resampling"
    assert result["observations"] == len(labels)
    assert result["clusters"] == 2
    assert result["valid_replicates"] == result["requested_replicates"] == 200
    assert result["metrics"]["manual"]["estimate"] == pytest.approx(
        roc_auc_score(labels, manual)
    )
    assert result["metrics"]["auto_gain"]["estimate"] == pytest.approx(
        roc_auc_score(labels, auto_gain)
    )
    delta = result["metrics"]["auto_gain"]["delta_vs_manual"]
    assert delta["estimate"] == pytest.approx(
        roc_auc_score(labels, auto_gain) - roc_auc_score(labels, manual)
    )
    assert len(result["metrics"]["auto_gain"]["ci95"]) == 2
    assert result == cluster_bootstrap_auc_comparison(
        labels,
        {"manual": manual, "auto_gain": auto_gain},
        users,
        baseline="manual",
        seed=13,
        n_boot=200,
    )

    # One fixed user resample matches sklearn's weighted event-level AUC exactly.
    user_ids, user_codes = np.unique(users, return_inverse=True)
    draw = np.random.default_rng(13).integers(0, len(user_ids), size=len(user_ids))
    weights = np.bincount(draw, minlength=len(user_ids))[user_codes]
    single = cluster_bootstrap_auc_comparison(
        labels,
        {"manual": manual, "auto_gain": auto_gain},
        users,
        baseline="manual",
        seed=13,
        n_boot=1,
    )
    expected_auto = roc_auc_score(labels, auto_gain, sample_weight=weights)
    expected_delta = expected_auto - roc_auc_score(labels, manual, sample_weight=weights)
    np.testing.assert_allclose(single["metrics"]["auto_gain"]["ci95"], [expected_auto] * 2)
    np.testing.assert_allclose(
        single["metrics"]["auto_gain"]["delta_vs_manual"]["ci95"],
        [expected_delta] * 2,
    )


def test_user_cluster_auc_bootstrap_omits_single_class_resamples():
    from recsys_corrected_artifact.metrics import cluster_bootstrap_auc_comparison

    result = cluster_bootstrap_auc_comparison(
        [0, 0, 1, 1],
        {"manual": [0.1, 0.2, 0.8, 0.9], "auto_gain": [0.2, 0.1, 0.9, 0.8]},
        ["negative_user", "negative_user", "positive_user", "positive_user"],
        seed=5,
        n_boot=100,
    )

    assert 0 < result["valid_replicates"] < result["requested_replicates"]
    assert np.isfinite(result["metrics"]["auto_gain"]["ci95"]).all()
    assert np.isfinite(
        result["metrics"]["auto_gain"]["delta_vs_manual"]["ci95"]
    ).all()


def test_user_cluster_bootstrap_auc_requires_aligned_inputs_and_baseline():
    from recsys_corrected_artifact.metrics import cluster_bootstrap_auc_comparison

    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_auc_comparison(
            [0, 1], {"manual": [0.2]}, ["u1", "u2"]
        )
    with pytest.raises(ValueError, match="baseline"):
        cluster_bootstrap_auc_comparison(
            [0, 1], {"auto_gain": [0.1, 0.9]}, ["u1", "u2"]
        )
