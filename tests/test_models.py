import numpy as np
import pandas as pd

import recsys_corrected_artifact.models as models


def test_fit_lgbm_early_stopping_uses_only_primary_auc(monkeypatch):
    callbacks = []
    make_early_stopping = models.early_stopping

    def capture_early_stopping(*args, **kwargs):
        callback = make_early_stopping(*args, **kwargs)
        callbacks.append(callback)
        return callback

    monkeypatch.setattr(models, "early_stopping", capture_early_stopping)

    rng = np.random.default_rng(17)
    train = pd.DataFrame(
        {
            "feature": rng.normal(size=120),
            "binary_label": np.tile([0, 1], 60),
        }
    )
    valid = pd.DataFrame(
        {
            "feature": rng.normal(size=40),
            "binary_label": np.tile([0, 1], 20),
        }
    )

    model = models.fit_lgbm(
        train,
        valid,
        ["feature"],
        config={"metric": "binary_logloss"},
        rounds=5,
        random_state=17,
    )

    assert list(model.evals_result_["valid_0"]) == ["auc"]
    assert set(model.best_score_["valid_0"]) == {"auc"}
    assert len(callbacks) == 1
    assert callbacks[0].first_metric_only is True
