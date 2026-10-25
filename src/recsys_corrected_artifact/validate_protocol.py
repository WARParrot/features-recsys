"""Validate schema-v4 artifacts and their separate evaluation semantics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

PROTOCOL_VERSION = 4
_SAMPLE_CANDIDATE_PROTOCOL = "sampled_unseen_item_discrimination"
_LOGGED_CANDIDATE_PROTOCOL = "logged_impression_candidates"
_SAMPLED_RANKING_CLAIM = "offline sampled discrimination, not logged-exposure ranking"
_REQUIRED_FINAL_METRICS = (
    "manual_test_auc",
    "auto_gain_test_auc",
    "manual_candidate_ndcg10",
    "manual_candidate_mrr10",
    "manual_candidate_recall10",
    "auto_gain_candidate_ndcg10",
    "auto_gain_candidate_mrr10",
    "auto_gain_candidate_recall10",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProtocolValidationError(ValueError):
    """Raised when an experiment artifact violates the protocol-v4 contract."""


def _fail(source: str, message: str) -> NoReturn:
    raise ProtocolValidationError(f"{source}: {message}")


def _require_mapping(value, label: str, source: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(source, f"{label} must be an object")
    return value


def _require_int(value, label: str, source: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(source, f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value, label: str, source: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(source, f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _validate_final_metrics(fold: Mapping, *, index: str, source: str) -> None:
    for name in _REQUIRED_FINAL_METRICS:
        if name not in fold:
            _fail(source, f"{index}.{name} is required")

    for name, value in fold.items():
        if not isinstance(name, str) or not (
            name.endswith("_test_auc") or "_candidate_" in name
        ):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            _fail(source, f"{index}.{name} must be a finite number in [0, 1]")


def _validate_source_metadata(metadata: Mapping, dataset: str, source: str) -> None:
    for field in ("command", "python", "platform"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            _fail(source, f"metadata.{field} must be a non-empty string")

    packages = _require_mapping(metadata.get("packages"), "metadata.packages", source)
    if not packages:
        _fail(source, "metadata.packages must not be empty")

    source_tree = _require_mapping(
        metadata.get("source_tree"), "metadata.source_tree", source
    )
    _require_sha256(source_tree.get("sha256"), "metadata.source_tree.sha256", source)
    files = source_tree.get("files")
    if not isinstance(files, list) or not files:
        _fail(source, "metadata.source_tree.files must be a non-empty list")
    for index, entry in enumerate(files):
        file_entry = _require_mapping(
            entry, f"metadata.source_tree.files[{index}]", source
        )
        path = file_entry.get("path")
        if not isinstance(path, str) or not path:
            _fail(source, f"metadata.source_tree.files[{index}].path must be non-empty")
        _require_sha256(
            file_entry.get("sha256"),
            f"metadata.source_tree.files[{index}].sha256",
            source,
        )

    source_hashes = _require_mapping(
        metadata.get("source_hashes"), "metadata.source_hashes", source
    )
    if dataset == "kion" and not source_hashes:
        _fail(source, "metadata.source_hashes must identify the KION source data")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name:
            _fail(source, "metadata.source_hashes keys must be non-empty strings")
        _require_sha256(digest, f"metadata.source_hashes.{name}", source)


def _validate_fold(fold: Mapping, *, dataset: str, run_seed: int, index: str, source: str) -> tuple[int, int]:
    _validate_final_metrics(fold, index=index, source=source)
    fold_seed = _require_int(fold.get("seed"), f"{index}.seed", source)
    if fold_seed != run_seed:
        _fail(source, f"{index}.seed does not match its run seed")
    selected = fold.get("selected")
    if not isinstance(selected, str) or not selected:
        _fail(source, f"{index}.selected must be a non-empty string")

    split_metadata = _require_mapping(
        fold.get("split_metadata"), f"{index}.split_metadata", source
    )
    test_split = _require_mapping(
        split_metadata.get("test"), f"{index}.split_metadata.test", source
    )
    test_rows = _require_int(
        test_split.get("rows"), f"{index}.split_metadata.test.rows", source
    )

    ranking_queries = _require_int(
        fold.get("ranking_queries"), f"{index}.ranking_queries", source
    )
    queries_with_positive = _require_int(
        fold.get("ranking_queries_with_positive"),
        f"{index}.ranking_queries_with_positive",
        source,
    )
    if queries_with_positive > ranking_queries:
        _fail(source, f"{index}.ranking_queries_with_positive exceeds ranking_queries")

    ranking_protocol = fold.get("ranking_protocol")
    if dataset == "mind":
        allowed_ranking_protocols = {_LOGGED_CANDIDATE_PROTOCOL}
        expected_candidate_protocol = _LOGGED_CANDIDATE_PROTOCOL
    else:
        allowed_ranking_protocols = {_SAMPLE_CANDIDATE_PROTOCOL}
        if dataset == "synthetic":
            # The single-split synthetic runner retains this older field label;
            # the candidate metadata still names the v4 sampling protocol.
            allowed_ranking_protocols.add("sampled_candidate_sets")
        expected_candidate_protocol = _SAMPLE_CANDIDATE_PROTOCOL
    if (
        not isinstance(ranking_protocol, str)
        or ranking_protocol not in allowed_ranking_protocols
    ):
        _fail(source, f"{index}.ranking_protocol is not canonical for dataset {dataset!r}")

    candidate_protocol = fold.get("candidate_protocol")
    if dataset == "mind" and candidate_protocol is None:
        return ranking_queries, test_rows
    candidate_protocol = _require_mapping(
        candidate_protocol, f"{index}.candidate_protocol", source
    )
    if candidate_protocol.get("name") != expected_candidate_protocol:
        _fail(source, f"{index}.candidate_protocol.name is not canonical for dataset {dataset!r}")
    if (
        expected_candidate_protocol == _SAMPLE_CANDIDATE_PROTOCOL
        and candidate_protocol.get("ranking_claim") != _SAMPLED_RANKING_CLAIM
    ):
        _fail(
            source,
            f"{index}.candidate_protocol.ranking_claim must be "
            f"{_SAMPLED_RANKING_CLAIM!r}",
        )

    actual_negatives: float | None = None
    if expected_candidate_protocol == _SAMPLE_CANDIDATE_PROTOCOL:
        rolling_version = fold.get("rolling_protocol_version")
        legacy_candidates = type(rolling_version) is not int or rolling_version < 5
        static_catalog = candidate_protocol.get("catalog_protocol") == "static_catalog"
        if static_catalog and not legacy_candidates:
            if (candidate_protocol.get("static_catalog_validated") is not True
                    or candidate_protocol.get("candidate_catalog") != "full"):
                _fail(source, f"{index}.candidate_protocol static catalog must be validated and full")
        elif candidate_protocol.get("catalog_protocol") != "historical_seen_items":
            _fail(source, f"{index}.candidate_protocol.catalog_protocol must be historical_seen_items")
        for field in (
            "future_catalog_items_excluded",
            "negatives_exclude_previously_seen_items",
        ):
            if candidate_protocol.get(field) is not True:
                _fail(source, f"{index}.candidate_protocol.{field} must be true")
        requested_negatives = _require_int(
            candidate_protocol.get("requested_negatives"),
            f"{index}.candidate_protocol.requested_negatives",
            source,
        )
        if dataset == "kion" and legacy_candidates and requested_negatives != 99:
            _fail(
                source,
                f"{index}.candidate_protocol.requested_negatives must be 99 "
                "for canonical KION protocol v4",
            )
        actual_negatives = candidate_protocol.get("mean_actual_negatives")
        if (
            isinstance(actual_negatives, bool)
            or not isinstance(actual_negatives, (int, float))
            or not math.isfinite(actual_negatives)
            or actual_negatives < 0
        ):
            _fail(source, f"{index}.candidate_protocol.mean_actual_negatives must be finite and >= 0")
        actual_negatives = float(actual_negatives)
        _require_int(
            candidate_protocol.get("skipped_positive_queries"),
            f"{index}.candidate_protocol.skipped_positive_queries",
            source,
        )
        if queries_with_positive != ranking_queries:
            _fail(
                source,
                f"{index}.ranking_queries_with_positive must equal ranking_queries "
                "for sampled positive-event queries",
            )
        if actual_negatives > requested_negatives:
            _fail(
                source,
                f"{index}.candidate_protocol.mean_actual_negatives exceeds requested_negatives",
            )

    protocol_queries = candidate_protocol.get("queries")
    if protocol_queries is not None:
        protocol_queries = _require_int(
            protocol_queries, f"{index}.candidate_protocol.queries", source
        )
        if protocol_queries != ranking_queries:
            _fail(source, f"{index}.candidate_protocol.queries does not match ranking_queries")
    elif expected_candidate_protocol == _SAMPLE_CANDIDATE_PROTOCOL:
        _fail(source, f"{index}.candidate_protocol.queries is required")

    protocol_rows = candidate_protocol.get("rows")
    if expected_candidate_protocol == _SAMPLE_CANDIDATE_PROTOCOL:
        protocol_rows = _require_int(
            protocol_rows, f"{index}.candidate_protocol.rows", source
        )
    elif protocol_rows is not None:
        protocol_rows = _require_int(
            protocol_rows, f"{index}.candidate_protocol.rows", source
        )
    if protocol_rows is not None:
        if protocol_rows < ranking_queries:
            _fail(source, f"{index}.candidate_protocol.rows is less than ranking_queries")
        if (
            expected_candidate_protocol == _SAMPLE_CANDIDATE_PROTOCOL
            and actual_negatives is not None
        ):
            expected_rows = ranking_queries * (1 + actual_negatives)
            if not math.isclose(protocol_rows, expected_rows, rel_tol=1e-9, abs_tol=1e-6):
                _fail(
                    source,
                    f"{index}.candidate_protocol.rows is inconsistent with queries "
                    "and mean_actual_negatives",
                )

    return ranking_queries, test_rows


# The outer version describes serialization, NOT rolling evaluation semantics.
# Missing fields stay missing: historical artifacts must not acquire modern defaults.
_SEMANTIC_FIELDS = (
    "rolling_protocol_version", "history_mode", "feature_space",
    "warmup_fraction", "training_blocks", "snapshot_cadence", "test_horizon",
    "candidate_n_negatives", "candidate_max_queries", "candidate_sampling",
    "candidate_catalog", "static_catalog_validated", "candidate_parameters_applied",
    "ranking_history", "watch_threshold", "positive_threshold", "rating_threshold",
)
_CANDIDATE_FIELDS = (
    "name", "catalog_protocol", "requested_negatives", "max_queries", "sampling",
    "candidate_catalog", "sampling_weights_source", "static_catalog_validated",
    "future_catalog_items_excluded", "negatives_exclude_previously_seen_items",
    "positive_source", "positive_items_historically_available", "ranking_claim",
)


def _recorded_semantics(mapping: Mapping) -> dict:
    return {
        key: value for key, value in mapping.items()
        if key in _SEMANTIC_FIELDS or key.endswith("_threshold")
        or key.startswith("snapshot_") or key.startswith("warmup_")
    }


def _merge_semantics(base: Mapping, extra: Mapping, label: str, source: str) -> dict:
    merged = dict(base)
    for key, value in _recorded_semantics(extra).items():
        if key in merged and (type(value) is not type(merged[key]) or value != merged[key]):
            _fail(source, f"{label}.{key} does not match declared evaluation semantics")
        merged[key] = value
    return merged


def _comparability_signature(metadata: Mapping, runs: list, source: str) -> dict:
    """Return settings and supervised splits, excluding metrics and source revisions."""
    declared_versions = []
    base = _recorded_semantics(metadata)
    if base.get("rolling_protocol_version") is None:
        base.pop("rolling_protocol_version", None)

    def check_version(mapping: Mapping, label: str, *, required: bool = False) -> None:
        value = mapping.get("rolling_protocol_version")
        if value is None and not required:
            return
        _require_int(value, f"{label}.rolling_protocol_version", source, minimum=1)
        declared_versions.append(value)

    check_version(metadata, "metadata")
    run_signatures = []
    for i, run in enumerate(runs):
        label = f"runs[{i}]"
        required = metadata.get("rolling_protocol_version") is not None
        check_version(run, label, required=required)
        settings = _merge_semantics(base, run, label, source)
        for field in ("metadata", "rolling_protocol"):
            if field in run:
                nested = _require_mapping(run[field], f"{label}.{field}", source)
                check_version(nested, f"{label}.{field}")
                settings = _merge_semantics(settings, nested, f"{label}.{field}", source)
        fold_signatures = []
        for j, fold in enumerate(run.get("folds", [run])):
            fold_label = f"{label}.folds[{j}]"
            check_version(fold, fold_label, required="rolling_protocol_version" in settings)
            fold_settings = _merge_semantics(settings, fold, fold_label, source)
            candidate = fold.get("candidate_protocol") or {}
            # Candidate metadata is authoritative for old artifacts too. Do not
            # compare realized query/negative counts, which are outcomes, not knobs.
            candidate_settings = {k: candidate[k] for k in _CANDIDATE_FIELDS if k in candidate}
            for setting, field in (
                ("candidate_n_negatives", "requested_negatives"),
                ("candidate_max_queries", "max_queries"),
                ("candidate_sampling", "sampling"),
                ("candidate_catalog", "candidate_catalog"),
            ):
                if setting in fold_settings and field in candidate_settings:
                    if fold_settings[setting] != candidate_settings[field]:
                        _fail(source, f"{fold_label}.candidate_protocol.{field} does not match {setting}")
            fold_signatures.append({
                "semantics": fold_settings,
                "candidate_protocol": candidate_settings,
                "ranking_protocol": fold["ranking_protocol"],
                "split_metadata": fold["split_metadata"],
            })
        run_signatures.append(fold_signatures)
    if len(set(declared_versions)) > 1:
        _fail(source, "inconsistent run/fold rolling_protocol_version declarations")
    # Seeds and the number/order of replicate runs do not change the protocol.
    unique_runs = sorted({json.dumps(item, sort_keys=True) for item in run_signatures})
    return {
        "schema_version": PROTOCOL_VERSION,
        "dataset": metadata["dataset"],
        "evaluation": metadata.get("evaluation"),
        "mind_protocol": metadata.get("mind_protocol"),
        "runs": [json.loads(item) for item in unique_runs],
    }


def validate_protocol_payload(payload, *, source: str = "<memory>") -> dict:
    """Validate one serialized compare_methods result and return count metadata."""
    payload = _require_mapping(payload, "result", source)
    metadata = _require_mapping(payload.get("metadata"), "metadata", source)
    version = metadata.get("protocol_version")
    if type(version) is not int or version != PROTOCOL_VERSION:
        _fail(source, f"metadata.protocol_version must be integer {PROTOCOL_VERSION}; got {version!r}")

    dataset = metadata.get("dataset")
    if not isinstance(dataset, str) or dataset not in {"synthetic", "kion", "mind"}:
        _fail(source, "metadata.dataset must be one of 'synthetic', 'kion', or 'mind'")
    _validate_source_metadata(metadata, dataset, source)

    seeds = metadata.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        _fail(source, "metadata.seeds must be a non-empty list")
    for index, seed in enumerate(seeds):
        _require_int(seed, f"metadata.seeds[{index}]", source, minimum=-(2**63))

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        _fail(source, "runs must be a non-empty list")
    if len(seeds) != len(runs):
        _fail(source, "metadata.seeds count does not match runs")

    actual_seeds = []
    total_folds = 0
    total_queries = 0
    total_test_rows = 0
    for run_index, run_value in enumerate(runs):
        run_label = f"runs[{run_index}]"
        run = _require_mapping(run_value, run_label, source)
        run_seed = _require_int(
            run.get("seed"), f"{run_label}.seed", source, minimum=-(2**63)
        )
        actual_seeds.append(run_seed)
        if "dataset" in run and run["dataset"] != dataset:
            _fail(source, f"{run_label}.dataset does not match metadata.dataset")
        has_folds = "folds" in run
        if not has_folds:
            folds = [run]
        else:
            folds_value = run["folds"]
            if not isinstance(folds_value, list) or not folds_value:
                _fail(source, f"{run_label}.folds must be a non-empty list")
            folds = folds_value
        for fold_index, fold_value in enumerate(folds):
            fold_label = (
                f"{run_label}.folds[{fold_index}]" if has_folds else run_label
            )
            fold = _require_mapping(fold_value, fold_label, source)
            query_count, test_row_count = _validate_fold(
                fold,
                dataset=dataset,
                run_seed=run_seed,
                index=fold_label,
                source=source,
            )
            total_folds += 1
            total_queries += query_count
            total_test_rows += test_row_count

    if actual_seeds != seeds:
        _fail(source, "metadata.seeds does not match run seeds and order")

    data_summary = _require_mapping(
        metadata.get("data_summary"), "metadata.data_summary", source
    )
    expected_counts = {
        "runs": len(runs),
        "folds": total_folds,
        "test_rows": total_test_rows,
        "ranking_queries": total_queries,
    }
    for field, expected in expected_counts.items():
        actual = _require_int(
            data_summary.get(field), f"metadata.data_summary.{field}", source
        )
        if actual != expected:
            _fail(
                source,
                f"metadata.data_summary.{field}={actual} does not match result total {expected}",
            )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "comparability_signature": _comparability_signature(metadata, runs, source),
        "dataset": dataset,
        "runs": len(runs),
        "folds": total_folds,
        "ranking_queries": total_queries,
        "test_rows": total_test_rows,
    }


def validate_protocol_files(paths: Sequence[str | Path]) -> list[dict]:
    """Validate schema-v4 files, rejecting incomparable evaluation collections."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ProtocolValidationError("at least one result file is required")

    validated = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolValidationError(f"{path}: cannot read result JSON: {exc}") from exc
        report = validate_protocol_payload(payload, source=str(path))
        report["path"] = str(path)
        validated.append(report)

    datasets = {report["dataset"] for report in validated}
    if len(datasets) != 1:
        joined = ", ".join(sorted(datasets))
        raise ProtocolValidationError(
            f"result collection mixes datasets ({joined}); validate comparable artifacts separately"
        )
    reference = validated[0]
    for report in validated[1:]:
        if report["comparability_signature"] != reference["comparability_signature"]:
            raise ProtocolValidationError(
                f"result collection has incompatible evaluation semantics or splits: "
                f"{reference['path']} versus {report['path']}; "
                "matching format schema protocol_version=4 does not imply comparability"
            )
    return validated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate protocol-v4 result metadata and consistency."
    )
    parser.add_argument("results", nargs="+", help="one or more compare_methods JSON results")
    args = parser.parse_args(argv)

    try:
        reports = validate_protocol_files(args.results)
    except ProtocolValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for report in reports:
        print(
            f"OK: {report['path']} — protocol v{report['protocol_version']}, "
            f"dataset={report['dataset']}, runs={report['runs']}, "
            f"folds={report['folds']}, ranking_queries={report['ranking_queries']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
