"""Mutation coverage for format-versus-evaluation comparability safeguards."""

from copy import deepcopy
import json

import pytest

from test_protocol_validation import CANONICAL_V4, ROOT, _valid_kion_payload
from recsys_corrected_artifact.validate_protocol import (
    ProtocolValidationError,
    validate_protocol_files,
    validate_protocol_payload,
)


def _rolling_payload():
    payload = _valid_kion_payload()
    settings = dict(
        rolling_protocol_version=6, history_mode="snapshot", feature_space="expanded",
        warmup_fraction=0.2, training_blocks=4,
        snapshot_cadence="train_timestamp_groups_divided_by_training_blocks",
        candidate_n_negatives=99, candidate_max_queries=400,
        candidate_sampling="uniform", candidate_catalog="cohort",
        watch_threshold=50.0,
    )
    payload["metadata"].update(settings, evaluation="rolling")
    run = payload["runs"][0]
    run["rolling_protocol_version"] = 6
    run["metadata"] = deepcopy(settings)
    run["rolling_protocol"] = deepcopy(settings)
    fold = run["folds"][0]
    fold.update(rolling_protocol_version=6, history_mode="snapshot", feature_space="expanded")
    fold["candidate_protocol"].update(
        sampling="uniform", max_queries=400, candidate_catalog="cohort"
    )
    return payload


def _collection(tmp_path, *payloads):
    paths = []
    for i, payload in enumerate(payloads):
        path = tmp_path / f"result-{i}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    return validate_protocol_files(paths)


def _change_setting(payload, field, value):
    payload["metadata"][field] = value
    run = payload["runs"][0]
    run["metadata"][field] = value
    run["rolling_protocol"][field] = value
    if field in run:
        run[field] = value
    for fold in run["folds"]:
        if field in fold:
            fold[field] = value


def test_immutable_v4_and_v5_standalone_are_accepted_but_not_comparable():
    modern = ROOT / "results" / "archive/v5/kion_50k_protocol_v5.json"
    assert validate_protocol_files([CANONICAL_V4])[0]["protocol_version"] == 4
    assert validate_protocol_files([modern])[0]["protocol_version"] == 4
    with pytest.raises(ProtocolValidationError, match="evaluation semantics"):
        validate_protocol_files([CANONICAL_V4, modern])


@pytest.mark.parametrize("field,value", [
    ("rolling_protocol_version", 5), ("history_mode", "prequential"),
    ("history_mode", "block_snapshot"), ("feature_space", "base"),
    ("warmup_fraction", 0.3), ("training_blocks", 8),
    ("snapshot_cadence", "daily"), ("watch_threshold", 75.0),
    ("positive_threshold", 0.8), ("candidate_sampling", "popularity"),
    ("candidate_max_queries", 200), ("candidate_catalog", "full"),
    ("candidate_n_negatives", 2),
])
def test_collection_rejects_mutated_semantics(tmp_path, field, value):
    original = _rolling_payload()
    mutated = deepcopy(original)
    _change_setting(mutated, field, value)
    candidate_field = {
        "candidate_sampling": "sampling", "candidate_max_queries": "max_queries",
        "candidate_catalog": "candidate_catalog", "candidate_n_negatives": "requested_negatives",
    }.get(field)
    if candidate_field:
        mutated["runs"][0]["folds"][0]["candidate_protocol"][candidate_field] = value
    # Each artifact is internally valid; collection comparison is what fails.
    validate_protocol_payload(mutated)
    with pytest.raises(ProtocolValidationError, match="evaluation semantics"):
        _collection(tmp_path, original, mutated)


@pytest.mark.parametrize("location", ["run", "run_metadata", "fold"])
@pytest.mark.parametrize("value", [4, True, "6", None])
def test_declared_rolling_versions_must_match_all_levels(location, value):
    payload = _rolling_payload()
    run = payload["runs"][0]
    target = {"run": run, "run_metadata": run["metadata"], "fold": run["folds"][0]}[location]
    target["rolling_protocol_version"] = value
    with pytest.raises(ProtocolValidationError, match="rolling_protocol_version"):
        validate_protocol_payload(payload)


@pytest.mark.parametrize("location", ["run", "fold"])
def test_declared_version_cannot_be_missing(location):
    payload = _rolling_payload()
    run = payload["runs"][0]
    target = run if location == "run" else run["folds"][0]
    del target["rolling_protocol_version"]
    with pytest.raises(ProtocolValidationError, match="rolling_protocol_version"):
        validate_protocol_payload(payload)


def test_collection_rejects_different_supervised_boundaries(tmp_path):
    original = _rolling_payload()
    mutated = deepcopy(original)
    mutated["runs"][0]["folds"][0]["split_metadata"]["test"]["min_timestamp"] = 123
    with pytest.raises(ProtocolValidationError, match="splits"):
        _collection(tmp_path, original, mutated)


def test_candidate_settings_must_match_declared_metadata():
    payload = _rolling_payload()
    payload["runs"][0]["folds"][0]["candidate_protocol"]["sampling"] = "popularity"
    with pytest.raises(ProtocolValidationError, match="candidate_protocol.sampling"):
        validate_protocol_payload(payload)


def test_validated_static_catalog_is_accepted_but_not_mixed(tmp_path):
    original = _rolling_payload()
    mutated = deepcopy(original)
    _change_setting(mutated, "candidate_catalog", "full")
    _change_setting(mutated, "static_catalog_validated", True)
    mutated["runs"][0]["folds"][0]["candidate_protocol"].update(
        catalog_protocol="static_catalog", candidate_catalog="full", static_catalog_validated=True
    )
    validate_protocol_payload(mutated)
    with pytest.raises(ProtocolValidationError, match="evaluation semantics"):
        _collection(tmp_path, original, mutated)
    mutated["runs"][0]["folds"][0]["candidate_protocol"]["static_catalog_validated"] = False
    with pytest.raises(ProtocolValidationError, match="static catalog"):
        validate_protocol_payload(mutated)


def test_candidate_count_changes_are_incomparable_for_synthetic(tmp_path):
    original = _valid_kion_payload()
    original["metadata"]["dataset"] = "synthetic"
    original["runs"][0]["dataset"] = "synthetic"
    mutated = deepcopy(original)
    mutated["runs"][0]["folds"][0]["candidate_protocol"]["requested_negatives"] = 2
    with pytest.raises(ProtocolValidationError, match="evaluation semantics"):
        _collection(tmp_path, original, mutated)


def test_source_revisions_and_metrics_are_not_protocol_settings(tmp_path):
    original = _rolling_payload()
    mutated = deepcopy(original)
    mutated["metadata"]["source_tree"]["sha256"] = "d" * 64
    mutated["metadata"]["source_hashes"]["kion_interactions"] = "e" * 64
    mutated["runs"][0]["folds"][0]["manual_test_auc"] = 0.9
    reports = _collection(tmp_path, original, mutated)
    assert reports[0]["comparability_signature"] == reports[1]["comparability_signature"]
    assert reports[0]["comparability_signature"]["schema_version"] == 4
