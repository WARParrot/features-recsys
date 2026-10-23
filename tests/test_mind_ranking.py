import pandas as pd
import pytest

from candidate_eval import score_logged_impressions
from metrics import mrr_at_k


class _ScoreModel:
    def predict_proba(self, frame):
        scores = frame["score_feature"].to_numpy(dtype=float)
        return pd.DataFrame({0: 1.0 - scores, 1: scores}).to_numpy()


def test_logged_impression_scoring_uses_query_groups_without_sampling():
    impression_log = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "label": [1, 0, 0],
        }
    )
    features = pd.DataFrame({"score_feature": [0.9, 0.1, 0.8]})

    result = score_logged_impressions(
        _ScoreModel(), impression_log, features, ["score_feature"]
    )

    assert result["queries"] == 2
    assert result["ndcg10"] > 0.0
    assert result["recall10"] == 1.0
    assert result["protocol"] == "logged_impression_candidates"


def test_mrr_at_k_uses_first_positive_rank_per_impression():
    frame = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "score": [0.9, 0.1, 0.2],
            "label": [0, 1, 1],
        }
    )

    assert mrr_at_k(frame, "score", k=10, label_col="label", group_col="query_id") == 0.75


def test_ranking_metrics_document_empty_positive_group_policy():
    frame = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "score": [0.9, 0.1, 0.8, 0.2],
            "label": [0, 0, 1, 0],
        }
    )

    from metrics import ndcg_at_k, recall_at_k

    assert ndcg_at_k(frame, "score", label_col="label", group_col="query_id") == 0.5
    assert recall_at_k(frame, "score", label_col="label", group_col="query_id") == 1.0
    assert recall_at_k(
        frame,
        "score",
        label_col="label",
        group_col="query_id",
        skip_no_positive=False,
    ) == 0.5


def test_recall_at_k_uses_all_positives_as_denominator():
    from metrics import recall_at_k

    frame = pd.DataFrame(
        {
            "query_id": ["q1"] * 12,
            "score": list(range(12, 0, -1)),
            "label": [1] * 12,
        }
    )

    assert recall_at_k(
        frame, "score", k=10, label_col="label", group_col="query_id"
    ) == 10 / 12


def test_logged_impression_scoring_requires_query_id_and_label():
    with pytest.raises(KeyError, match="query_id"):
        score_logged_impressions(
            _ScoreModel(),
            pd.DataFrame({"label": [1]}),
            pd.DataFrame({"score_feature": [0.5]}),
            ["score_feature"],
        )

