import json
from pathlib import Path
import subprocess
import sys

import pytest

from recsys_corrected_artifact.validate_protocol import (
    ProtocolValidationError,
    validate_protocol_files,
    validate_protocol_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_V4 = ROOT / "results" / "archive/v4/kion_50k_protocol_v4.json"
LEGACY_V3 = ROOT / "results" / "archive/validation/kion_10k_corrected.json"


def _valid_kion_payload():
    candidate_protocol = {
        "name": "sampled_unseen_item_discrimination",
        "catalog_protocol": "historical_seen_items",
        "future_catalog_items_excluded": True,
        "negatives_exclude_previously_seen_items": True,
        "requested_negatives": 99,
        "mean_actual_negatives": 1.0,
        "skipped_positive_queries": 0,
        "queries": 2,
        "rows": 4,
        "ranking_claim": "offline sampled discrimination, not logged-exposure ranking",
    }
    fold = {
        "seed": 42,
        "selected": "manual",
        "split_metadata": {"test": {"rows": 4}},
        "ranking_protocol": "sampled_unseen_item_discrimination",
        "ranking_queries": 2,
        "ranking_queries_with_positive": 2,
        "candidate_protocol": candidate_protocol,
        "manual_test_auc": 0.6,
        "auto_gain_test_auc": 0.7,
        "manual_candidate_ndcg10": 0.3,
        "manual_candidate_mrr10": 0.2,
        "manual_candidate_recall10": 0.4,
        "auto_gain_candidate_ndcg10": 0.5,
        "auto_gain_candidate_mrr10": 0.3,
        "auto_gain_candidate_recall10": 0.6,
    }
    return {
        "metadata": {
            "protocol_version": 4,
            "dataset": "kion",
            "seeds": [42],
            "command": "python -m recsys_corrected_artifact.compare_methods --json",
            "python": "3.11.0",
            "platform": "test-platform",
            "packages": {"numpy": "2.0.0"},
            "source_tree": {
                "sha256": "a" * 64,
                "files": [{"path": "pipeline.py", "sha256": "b" * 64}],
            },
            "source_hashes": {"kion_interactions": "c" * 64},
            "data_summary": {
                "runs": 1,
                "folds": 1,
                "test_rows": 4,
                "ranking_queries": 2,
            },
        },
        "runs": [{"dataset": "kion", "seed": 42, "folds": [fold]}],
    }


def test_canonical_protocol_v4_artifact_passes_consistency_validation():
    payload = json.loads(CANONICAL_V4.read_text())

    summary = validate_protocol_payload(payload, source=str(CANONICAL_V4))

    assert summary["protocol_version"] == 4
    assert summary["dataset"] == "kion"
    assert summary["runs"] == 3
    assert summary["folds"] == 6
    assert summary["ranking_queries"] == 2187


@pytest.mark.parametrize("version", [3, "4", True, None])
def test_only_exact_integer_protocol_v4_is_accepted(version):
    payload = _valid_kion_payload()
    payload["metadata"]["protocol_version"] = version

    with pytest.raises(ProtocolValidationError, match="protocol_version"):
        validate_protocol_payload(payload)


@pytest.mark.parametrize(
    "field",
    ["future_catalog_items_excluded", "negatives_exclude_previously_seen_items"],
)
def test_kion_protocol_requires_both_candidate_exclusion_guarantees(field):
    payload = _valid_kion_payload()
    payload["runs"][0]["folds"][0]["candidate_protocol"][field] = False

    with pytest.raises(ProtocolValidationError, match=field):
        validate_protocol_payload(payload)


def test_candidate_protocol_query_count_must_match_ranking_count():
    payload = _valid_kion_payload()
    payload["runs"][0]["folds"][0]["candidate_protocol"]["queries"] = 1

    with pytest.raises(ProtocolValidationError, match="candidate_protocol.queries"):
        validate_protocol_payload(payload)


def test_kion_rejects_canonical_artifact_mutated_to_request_one_negative():
    payload = json.loads(CANONICAL_V4.read_text())
    payload["runs"][0]["folds"][0]["candidate_protocol"]["requested_negatives"] = 1

    with pytest.raises(ProtocolValidationError, match="requested_negatives must be 99"):
        validate_protocol_payload(payload, source="mutated-canonical")


def test_synthetic_sampled_protocol_accepts_one_negative_with_explicit_semantics():
    payload = _valid_kion_payload()
    payload["metadata"]["dataset"] = "synthetic"
    payload["metadata"]["source_hashes"] = {}
    payload["runs"][0]["dataset"] = "synthetic"
    fold = payload["runs"][0]["folds"][0]
    fold["ranking_protocol"] = "sampled_candidate_sets"
    fold["candidate_protocol"]["requested_negatives"] = 1

    report = validate_protocol_payload(payload)

    assert report["dataset"] == "synthetic"


def test_synthetic_sampled_protocol_requires_offline_discrimination_semantics():
    payload = _valid_kion_payload()
    payload["metadata"]["dataset"] = "synthetic"
    payload["metadata"]["source_hashes"] = {}
    payload["runs"][0]["dataset"] = "synthetic"
    fold = payload["runs"][0]["folds"][0]
    fold["ranking_protocol"] = "sampled_candidate_sets"
    fold["candidate_protocol"]["ranking_claim"] = "logged-exposure ranking"

    with pytest.raises(ProtocolValidationError, match="ranking_claim"):
        validate_protocol_payload(payload)


def test_required_auc_and_candidate_ranking_metrics_must_be_present():
    payload = _valid_kion_payload()
    del payload["runs"][0]["folds"][0]["auto_gain_candidate_mrr10"]

    with pytest.raises(ProtocolValidationError, match="auto_gain_candidate_mrr10"):
        validate_protocol_payload(payload)


def test_ndcg_above_one_is_rejected():
    payload = json.loads(CANONICAL_V4.read_text())
    payload["runs"][0]["folds"][0]["manual_candidate_ndcg10"] = 10

    with pytest.raises(ProtocolValidationError, match="manual_candidate_ndcg10.*\\[0, 1\\]"):
        validate_protocol_payload(payload, source="mutated-canonical")


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_every_present_final_auc_and_candidate_metric_is_bounded(value):
    payload = _valid_kion_payload()
    payload["runs"][0]["folds"][0]["decay_test_auc"] = value

    with pytest.raises(ProtocolValidationError, match="decay_test_auc"):
        validate_protocol_payload(payload)


def test_metadata_ranking_query_total_must_match_fold_counts():
    payload = _valid_kion_payload()
    payload["metadata"]["data_summary"]["ranking_queries"] = 3

    with pytest.raises(ProtocolValidationError, match="data_summary.ranking_queries"):
        validate_protocol_payload(payload)


def test_run_seeds_must_match_reproducibility_metadata():
    payload = _valid_kion_payload()
    payload["metadata"]["seeds"] = [43]

    with pytest.raises(ProtocolValidationError, match="metadata.seeds"):
        validate_protocol_payload(payload)


def test_v4_collection_rejects_a_legacy_artifact():
    with pytest.raises(ProtocolValidationError, match="protocol_version"):
        validate_protocol_files([CANONICAL_V4, LEGACY_V3])


def test_validator_cli_accepts_canonical_v4_result():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recsys_corrected_artifact.validate_protocol",
            str(CANONICAL_V4),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "protocol v4" in result.stdout
    assert "ranking_queries=2187" in result.stdout


def test_validator_cli_rejects_legacy_result():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recsys_corrected_artifact.validate_protocol",
            str(LEGACY_V3),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "protocol_version" in result.stderr
