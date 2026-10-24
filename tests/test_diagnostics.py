import copy
import json

import numpy as np
import pandas as pd
import pytest

from diagnostics import feature_diagnostics, fold_diagnostics, role_diagnostics
from run_ablation import (CELLS, assert_comparable, atomic_json, execute, read_lock,
                          json_value, preflight, run_factorial, validate_result, write_lock)


def frame():
    return pd.DataFrame({"event_id": range(8), "user": [0, 0, 1, 0, 2, 1, 2, 3],
                         "item": [0, 1, 1, 0, 2, 1, 0, 3], "timestamp": range(8),
                         "binary_label": [0, 1, 0, 1, 0, 1, 0, 1]})


def config():
    return {"inputs": {"interactions": "fake.csv", "users": None, "items": None},
            "max_users": 1000, "cohort_seed": 42, "seeds": [42], "folds": 2,
            "training_blocks": 4, "warmup_fraction": .2, "watch_threshold": 50.,
            "forward_max_candidates": 2, "forward_steps": 1, "candidate_n_negatives": 2,
            "candidate_max_queries": 5, "candidate_sampling": "uniform", "smoke_users": 2,
            "smoke_only": False, "max_wall_seconds": 60.}


def evaluator(data, seed, n_folds, dataset, history_mode, feature_space,
              warmup_fraction, training_blocks, **kwargs):
    metrics = {f"{model}_{role}_auc": .5 for model in
               ("manual", "auto_gain", "auto_forward", "decay", "linear")
               for role in ("select", "test")}
    return {"numpy_scalar": np.int64(4), "undefined_auc": None,
            "folds": [{"fold": i, "seed": seed, **metrics,
                       "ranking_queries": 2, "ranking_queries_with_positive": 2,
                       "diagnostics": {role: {"rows": 2, "positives": 1,
                                              "features": {"columns": {"x": {}}}}
                                       for role in ("train", "select", "valid", "test")},
                       "split_metadata": {"test": {"rows": len(data)}},
                       "split_fingerprint": {"test": "rows-hash"},
                       "candidate_fingerprint": f"{seed}:{i}:{len(data)}"}
                      for i in range(n_folds)]}


def test_features_and_role_predictions():
    data = frame()
    features = pd.DataFrame({"zero": np.zeros(8), "varying": range(8),
                             "missing": [np.nan] * 8})
    stats = feature_diagnostics(features)
    assert stats["constant_fraction"] == pytest.approx(2 / 3)
    assert stats["columns"]["zero"]["zero_fraction"] == 1
    assert stats["columns"]["missing"]["missing_fraction"] == 1
    result = role_diagnostics(data, features, history_lengths=np.arange(8),
                              item_popularity=np.arange(8),
                              predictions={"perfect": data.binary_label.to_numpy()})
    assert result["prevalence"] == .5
    assert result["slices"]["all"]["predictions"]["perfect"] == {"auc": 1., "auc_unestimable_reason": None, "brier": 0.}
    assert result["slices"]["history_length:0"]["predictions"]["perfect"]["auc"] is None
    json.dumps(result, allow_nan=False)


def test_prefix_diagnostics_ignore_role_labels_and_within_role_history():
    data = frame()
    data.loc[2, "timestamp"] = 3  # timestamp tie must not enter history
    parts = {"train": np.arange(2), "test": np.arange(2, 8)}
    result = fold_diagnostics(data, parts)
    assert result["test"]["history_length"]["max"] == 2
    assert result["test"]["slices"]["history_length:0"]["rows"] == 5
    changed = data.copy()
    changed.loc[2:, "binary_label"] = 1 - changed.loc[2:, "binary_label"]
    other = fold_diagnostics(changed, parts)
    assert result["test"]["history_length"] == other["test"]["history_length"]
    assert result["test"]["item_popularity"] == other["test"]["item_popularity"]
    assert result["test"]["prediction_slices_status"] == "unavailable_no_predictions"


@pytest.mark.parametrize("kwargs", [
    {"history_lengths": [-1] * 8}, {"item_popularity": [np.nan] * 8},
    {"predictions": {"m": [1.1] * 8}}, {"predictions": {"m": [0] * 7}},
    {"features": pd.DataFrame({"x": range(8)}, index=range(1, 9))},
])
def test_alignment_and_invalid_values(kwargs):
    with pytest.raises(ValueError):
        role_diagnostics(frame(), **kwargs)


def test_atomic_json_preserves_previous_on_serialization_error(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"status": "old"})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"status": "old"}
    assert list(tmp_path.iterdir()) == [path]


def test_comparability_checks_candidate_identity_not_counts():
    result = evaluator(frame(), 42, 2, "kion", "snapshot", "base", .2, 4)
    assert_comparable(result, copy.deepcopy(result))
    changed = copy.deepcopy(result)
    changed["folds"][0]["candidate_fingerprint"] = "different"
    with pytest.raises(ValueError, match="candidate_fingerprint"):
        assert_comparable(result, changed)
    del changed["folds"][0]["candidate_fingerprint"]
    with pytest.raises(ValueError):
        assert_comparable(changed, changed)


def test_factorial_same_parameters_and_deterministic_cohort():
    calls = []

    def wrapped(data, seed, n_folds, dataset, history_mode, feature_space,
                warmup_fraction, training_blocks, **kwargs):
        calls.append((history_mode, feature_space, seed, warmup_fraction, training_blocks, kwargs))
        return evaluator(data, seed, n_folds, dataset, history_mode, feature_space,
                         warmup_fraction, training_blocks, **kwargs)

    cells = run_factorial(frame(), config(), wrapped)
    assert [(cell["history_mode"], cell["feature_space"]) for cell in cells] == list(CELLS)
    assert len({cell["cohort_fingerprint"] for cell in cells}) == 1
    assert len({json.dumps(call[2:], sort_keys=True) for call in calls}) == 1
    assert all(cell["status"] == "completed" for cell in cells)
    assert all(cell["cost"]["wall_seconds"] >= 0 for cell in cells)


def test_nonfinite_evaluator_metrics_rejected():
    with pytest.raises(ValueError, match="unestimable reason"):
        json_value({"auc": np.float64(np.nan)})


def test_missing_corrected_api_rejected():
    with pytest.raises(RuntimeError, match="API"):
        run_factorial(frame(), config(), lambda **kwargs: None)


def test_execute_load_once_and_smoke_before_main(tmp_path, monkeypatch):
    monkeypatch.setattr("run_ablation.preflight", lambda *a: [{"status": "tested separately"}])
    loads, calls = [], []

    def loader(*args, **kwargs):
        loads.append(kwargs)
        return frame()

    def wrapped(data, seed, n_folds, dataset, history_mode, feature_space,
                warmup_fraction, training_blocks, **kwargs):
        calls.append((len(data), n_folds))
        return evaluator(data, seed, n_folds, dataset, history_mode, feature_space,
                         warmup_fraction, training_blocks, **kwargs)

    output = tmp_path / "result.json"
    result = execute({"config": config()}, output, loader=loader, evaluator=wrapped)
    assert len(loads) == 1
    assert len(calls) == 8
    assert all(folds == 1 for _, folds in calls[:4])
    assert all((rows, folds) == (8, 2) for rows, folds in calls[4:])
    assert result["smoke"]["status"] == "passed"
    saved = json.loads(output.read_text())
    assert saved["status"] == "completed"
    assert saved["cells"][0]["result"]["numpy_scalar"] == 4
    assert saved["cells"][0]["result"]["undefined_auc"] is None
    with pytest.raises(FileExistsError):
        execute({"config": config()}, output, loader=loader, evaluator=wrapped)


def test_failure_has_atomic_artifact_and_never_runs_main(tmp_path, monkeypatch):
    monkeypatch.setattr("run_ablation.preflight", lambda *a: [])
    calls = []

    def broken(data, seed, n_folds, dataset, history_mode, feature_space,
               warmup_fraction, training_blocks, **kwargs):
        calls.append(history_mode)
        raise RuntimeError("fit failed")

    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="fit failed"):
        execute({"config": config()}, output, loader=lambda *a, **k: frame(), evaluator=broken)
    result = json.loads(output.read_text())
    assert calls == ["prequential"]
    assert result["status"] == "failed"
    assert result["smoke"]["status"] == "failed"
    assert result["smoke"]["cells"][0]["status"] == "failed"
    assert result["cells"] == []


def test_lock_integrity_and_input_changes(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n")
    cfg = config()
    cfg["inputs"]["interactions"] = str(source)
    path = tmp_path / "lock.json"
    locked = write_lock(path, cfg)
    assert read_lock(path) == locked
    with pytest.raises(FileExistsError):
        write_lock(path, cfg)
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="provenance changed"):
        read_lock(path)
    locked["config"]["folds"] = 3
    atomic_json(path, locked)
    with pytest.raises(ValueError, match="modified"):
        read_lock(path)


@pytest.mark.parametrize("failure", ["empty", "missing_metric", "null_metric", "no_queries", "no_diagnostics", "one_class"])
def test_result_success_gate_rejects_incomplete_outputs(failure):
    result = evaluator(frame(), 42, 2, "kion", "snapshot", "base", .2, 4)
    validate_result(result, config())
    fold = result["folds"][0]
    if failure == "empty":
        result["folds"] = []
    elif failure == "missing_metric":
        del fold["manual_test_auc"]
    elif failure == "null_metric":
        fold["manual_select_auc"] = None
    elif failure == "no_queries":
        fold["ranking_queries_with_positive"] = 0
    elif failure == "no_diagnostics":
        del fold["diagnostics"]
    else:
        fold["diagnostics"]["valid"]["positives"] = 0
    with pytest.raises(ValueError):
        validate_result(result, config())


def test_preflight_support_and_no_metric_driven_resampling():
    from data import generate_synthetic
    data = generate_synthetic(n_events=1200, n_users=20, n_items=30, seed=42)
    report = preflight(data, config())
    assert len(report) == 2
    assert all(row["ranking_queries"] > 0 for row in report)
    single = data.copy()
    single["binary_label"] = 1
    with pytest.raises(ValueError, match="no resampling"):
        preflight(single, config())
    no_catalog = data.copy()
    no_catalog["item"] = range(len(data))
    with pytest.raises(ValueError, match="no valid positive/negative queries"):
        preflight(no_catalog, config())


def test_preflight_failure_artifact_stops_before_models(tmp_path):
    from data import generate_synthetic
    single = generate_synthetic(n_events=200, n_users=4, n_items=10)
    single["binary_label"] = 1
    output = tmp_path / "unsupported.json"
    with pytest.raises(ValueError, match="requires both label classes"):
        execute({"config": config()}, output, loader=lambda *a, **k: single,
                evaluator=evaluator)
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed"
    assert saved["smoke"]["status"] == "failed"
    assert "cells" not in saved["smoke"]
