"""Selection must not turn empty histories or tied scores into evidence."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import feature_selectors as fs


@pytest.fixture
def frame():
    return pd.DataFrame({"base": [0, 1, 0, 1], "signal": [0, 0, 1, 1],
                         "zero": [0] * 4, "binary_label": [0, 1, 0, 1]})


def test_gain_empty_history_fails_before_fit(frame, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constant training features must not be fitted")
    monkeypatch.setattr(fs, "fit_lgbm_fixed", forbidden)
    with pytest.raises(ValueError, match="no nonconstant.*history"):
        fs.gain_top_k(frame, ["zero"])


def test_gain_filters_constants_duplicates_and_nonpositive_importance(frame, monkeypatch):
    frame = frame.assign(other=[1, 2, 3, 4], bad=[4, 3, 2, 1], negative=[2, 3, 1, 4])
    def fit(train, columns, **kwargs):
        assert columns == ["signal", "other", "bad", "negative"]
        return SimpleNamespace(booster_=SimpleNamespace(
            feature_importance=lambda **kw: np.array([2., 0., np.nan, -1.])))
    monkeypatch.setattr(fs, "fit_lgbm_fixed", fit)
    columns, gains = fs.gain_top_k(frame, ["zero", "signal", "signal", "other", "bad", "negative"])
    assert columns == ["signal"]
    assert gains.to_dict() == {"signal": 2.}


@pytest.mark.parametrize("gain", [0., -1., np.nan, np.inf])
def test_gain_no_positive_finite_importance_fails(frame, monkeypatch, gain):
    monkeypatch.setattr(fs, "fit_lgbm_fixed", lambda *a, **kw: SimpleNamespace(
        booster_=SimpleNamespace(feature_importance=lambda **kw: np.array([gain]))))
    with pytest.raises(ValueError, match="no finite positive"):
        fs.gain_top_k(frame, ["signal"])


def mock_scores(monkeypatch, scores):
    scores = iter(scores)
    class Model:
        def fit(self, *args):
            return self
        def predict_proba(self, frame):
            return np.tile([.5, .5], (len(frame), 1))
    monkeypatch.setattr(fs, "_linear_model", lambda *a, **kw: Model())
    monkeypatch.setattr(fs, "roc_auc_score", lambda *a: next(scores))


@pytest.mark.parametrize("base", [[], ["zero"], ["base", "base"]])
def test_forward_zero_improvement_is_rejected_and_explained(frame, monkeypatch, base):
    mock_scores(monkeypatch, [.5, .5])
    selected, added, history = fs.forward_select(frame, frame, ["signal"], base_columns=base)
    assert selected == list(dict.fromkeys(base))
    assert added == []
    assert history[-1]["accepted"] is False
    assert history[-1]["reason"] == "nonpositive_improvement"
    assert history[-1]["improvement"] == 0


def test_forward_skips_nonfinite_score_and_accepts_finite_improvement(frame, monkeypatch):
    mock_scores(monkeypatch, [np.nan, .7])
    selected, added, history = fs.forward_select(frame, frame, ["base", "signal"], k=1)
    assert selected == added == ["signal"]
    assert history[0]["improvement"] == pytest.approx(.2)


def test_forward_nonfinite_candidates_record_stop(frame, monkeypatch):
    mock_scores(monkeypatch, [np.nan])
    selected, added, history = fs.forward_select(frame, frame, ["signal"])
    assert selected == added == []
    assert history[0]["reason"] == "no_nonconstant_candidate_with_finite_score"
    assert history[0]["score"] is None


def test_forward_constant_candidates_record_stop(frame):
    selected, added, history = fs.forward_select(frame, frame, ["zero"])
    assert selected == added == []
    assert history[0]["reason"] == "no_nonconstant_candidate_with_finite_score"


def test_forward_threshold_rejection_is_recorded(frame, monkeypatch):
    mock_scores(monkeypatch, [.6])
    _, added, history = fs.forward_select(frame, frame, ["signal"], min_improvement=.2)
    assert added == []
    assert history[0]["reason"] == "below_min_improvement"


@pytest.mark.parametrize("threshold", [-.1, np.nan, np.inf])
def test_forward_invalid_threshold(frame, threshold):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fs.forward_select(frame, frame, ["signal"], min_improvement=threshold)


def test_forward_missing_base_column_clear_error(frame):
    with pytest.raises(ValueError, match="missing selection columns.*missing"):
        fs.forward_select(frame, frame, ["signal"], base_columns=["missing"])


def test_forward_nonfinite_base_score_fails(frame, monkeypatch):
    mock_scores(monkeypatch, [np.nan])
    with pytest.raises(ValueError, match="nonfinite base AUC"):
        fs.forward_select(frame, frame, ["signal"], base_columns=["base"])


def test_forward_actual_signal_with_constant_base():
    labels = np.tile([0, 1], 30)
    train = pd.DataFrame({"zero": np.zeros(60), "signal": labels,
                          "binary_label": labels})
    selected, added, history = fs.forward_select(
        train, train.copy(), ["zero", "signal"], base_columns=["zero", "zero"])
    assert selected == ["zero", "signal"]
    assert added == ["signal"]
    assert history[0]["score"] == 1.
    assert history[0]["improvement"] == .5


def test_forward_single_class_clear_error(frame):
    with pytest.raises(ValueError, match="both label classes"):
        fs.forward_select(frame, frame.assign(binary_label=0), ["signal"])
