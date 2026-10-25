"""Shared experiment contracts and result validation."""

from __future__ import annotations

from collections.abc import Mapping


REQUIRED_RESULT_KEYS = frozenset({"seed", "selected", "split_metadata"})
RANKING_METRICS = ("ndcg10", "mrr10", "recall10")


def validate_result(result: Mapping, *, rolling: bool = False) -> None:
    """Reject incomplete experiment results before serialization or comparison."""
    missing = REQUIRED_RESULT_KEYS - set(result)
    if missing:
        raise ValueError(f"experiment result is missing keys: {sorted(missing)}")
    if not isinstance(result["seed"], int):
        raise TypeError("experiment result seed must be an integer")
    if not isinstance(result["selected"], str) or not result["selected"]:
        raise ValueError("experiment result selected method must be a non-empty string")
    if not isinstance(result["split_metadata"], Mapping) or not result["split_metadata"]:
        raise ValueError("experiment result must include split metadata")
    if "ranking_protocol" in result and not isinstance(result["ranking_protocol"], str):
        raise TypeError("ranking protocol must be a string")
    if "candidate_protocol" in result and not isinstance(result["candidate_protocol"], Mapping):
        raise TypeError("candidate protocol must be a mapping")
    if rolling and "folds" not in result:
        raise ValueError("rolling result must include folds")


def validate_run_collection(results) -> None:
    if not results:
        raise ValueError("at least one experiment result is required")
    for result in results:
        if "folds" in result:
            folds = result["folds"]
            if not folds:
                raise ValueError("rolling result must include at least one fold")
            for fold in folds:
                validate_result(fold)
        else:
            validate_result(result)
