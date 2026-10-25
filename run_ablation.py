"""Standalone, locked 2x2 KION ablation. No experiment runs without --execute-lock.

Example (inspect the lock before executing)::

    python run_ablation.py --dry-run --lock /tmp/ablation.lock.json \
        --kion-interactions datasets/kion/interactions.csv
    python run_ablation.py --execute-lock /tmp/ablation.lock.json \
        --output /tmp/ablation.json

The smoke gate uses a small complete-user subset of the ONE loaded KION frame.
Smoke results validate execution only, not a recommendation-quality hypothesis.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - exercised on Windows
    resource = None
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


SCHEMA = "kion_factorial_ablation_v1"
CELLS = tuple((history, space) for history in ("prequential", "snapshot")
              for space in ("base", "expanded"))
ROOT = Path(__file__).resolve().parent


def json_value(value):
    """Normalize NumPy containers; reject nonfinite metrics without an explanation."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite evaluator metric without an explicit unestimable reason")
    return value


def atomic_json(path, payload):
    """Replace only after strict JSON serialization, flush and fsync succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_metadata(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def provenance(inputs):
    files = sorted(path for path in ROOT.iterdir() if path.is_file() and
                   (path.suffix in {".py", ".toml", ".lock"} or path.name == "requirements.txt"))
    sources = {path.name: file_metadata(path)["sha256"] for path in files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"inputs": {key: file_metadata(path) for key, path in inputs.items() if path},
            "sources": sources, "source_sha256": digest_json(sources),
            "git_revision": revision, "packages": versions, "python": platform.python_version(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def frame_fingerprint(frame):
    """Hash row content/order, column names and dtypes, excluding mutable attrs."""
    digest = hashlib.sha256()
    digest.update(str(list(zip(frame.columns, map(str, frame.dtypes)))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def validate_config(config):
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("provide at least one distinct seed")
    for key in ("max_users", "folds", "training_blocks", "forward_max_candidates", "forward_steps",
                "candidate_max_queries", "smoke_users"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["candidate_n_negatives"] < 0:
        raise ValueError("candidate_n_negatives must be nonnegative")
    if not 0 < config["warmup_fraction"] < 1:
        raise ValueError("shared warmup_fraction must be strictly between 0 and 1")
    if not np.isfinite(config["watch_threshold"]):
        raise ValueError("watch_threshold must be finite")
    if not np.isfinite(config["max_wall_seconds"]) or config["max_wall_seconds"] <= 0:
        raise ValueError("max_wall_seconds must be finite and positive")


def write_lock(path, config):
    validate_config(config)
    if Path(path).exists():
        raise FileExistsError(f"lock already exists: {path}")
    lock = {"schema": SCHEMA, "status": "locked_not_executed", "config": config,
            "provenance": provenance(config["inputs"])}
    lock["lock_sha256"] = digest_json(lock)
    atomic_json(path, lock)
    return lock


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("schema") != SCHEMA or digest_json(unsigned) != lock.get("lock_sha256"):
        raise ValueError("invalid or modified configuration lock")
    validate_config(lock["config"])
    current = provenance(lock["config"]["inputs"])
    if current != lock["provenance"]:
        raise ValueError("input/source/runtime provenance changed; create and inspect a new lock")
    return lock


def assert_comparable(reference, result):
    """Fail closed: counts or deterministic seeds alone do not identify candidates."""
    if len(reference["folds"]) != len(result["folds"]):
        raise ValueError("fold count differs across factorial cells")
    for left, right in zip(reference["folds"], result["folds"]):
        for key in ("fold", "seed", "split_metadata", "split_fingerprint", "candidate_fingerprint"):
            if key not in left or key not in right or left[key] is None or left[key] != right[key]:
                raise ValueError(f"incomparable factorial cells: {key}")
        if not left["candidate_fingerprint"]:
            raise ValueError("candidate identity fingerprint missing")


def validate_result(result, config):
    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != config["folds"]:
        raise ValueError("result must contain the expected nonempty fold count")
    for index, fold in enumerate(folds):
        if fold.get("fold") != index:
            raise ValueError("fold identities must match expected order")
        for model in ("manual", "auto_gain", "auto_forward", "decay", "linear"):
            for role in ("select", "test"):
                key = f"{model}_{role}_auc"
                value = fold.get(key)
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                        or not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"required finite primary metric missing: {key}")
        for key in ("ranking_queries", "ranking_queries_with_positive"):
            if not isinstance(fold.get(key), (int, np.integer)) or fold[key] <= 0:
                raise ValueError(f"required positive query support missing: {key}")
        diagnostics = fold.get("diagnostics", {})
        for role in ("train", "select", "valid", "test"):
            stats = diagnostics.get(role, {})
            if not (0 < stats.get("positives", 0) < stats.get("rows", 0)):
                raise ValueError(f"diagnostics missing two-class support: {role}")
            if not stats.get("features", {}).get("columns"):
                raise ValueError(f"feature diagnostics missing: {role}")


def preflight(frame, config):
    """Check the fixed cohort/splits and actual candidate sampler before fitting.

    No retries, resampling or seed changes are permitted after a support failure.
    Candidate generation is repeated deterministically by the evaluator.
    """
    from data import rolling_time_splits
    from candidate_eval import sampled_candidate_log
    log, folds = rolling_time_splits(frame, n_folds=config["folds"],
                                     warmup_fraction=config["warmup_fraction"])
    report = []
    for index, parts in enumerate(folds):
        roles = {}
        for role in ("train", "select", "valid", "test"):
            values = log.iloc[parts[role]]["binary_label"]
            if len(values) == 0 or not set(values.unique()) == {0, 1}:
                raise ValueError(f"preflight fold {index} {role}: requires both label classes; no resampling")
            roles[role] = {"rows": len(values), "positives": int(values.sum())}
        for seed in config["seeds"]:
            candidates = sampled_candidate_log(
                log.iloc[parts["test"]], log.iloc[:int(parts["test"][0])].copy(),
                int(log["item"].nunique()), n_negatives=config["candidate_n_negatives"],
                max_queries=config["candidate_max_queries"], seed=seed + index,
                item_catalog=log, catalog_protocol="historical_seen_items",
                sampling=config["candidate_sampling"],
            )
            if candidates.empty or not candidates.groupby("query_id")["label"].agg(["min", "max"]).eq(
                    pd.Series({"min": 0, "max": 1})).all().all():
                raise ValueError(f"preflight fold {index} seed {seed}: no valid positive/negative queries; no resampling")
            report.append({"fold": index, "seed": seed, "roles": roles,
                           "ranking_queries": int(candidates["query_id"].nunique())})
    return report


def require_api(evaluator):
    required = {"history_mode", "feature_space", "warmup_fraction", "training_blocks"}
    missing = required - set(inspect.signature(evaluator).parameters)
    if missing:
        raise RuntimeError(f"corrected rolling API missing explicit parameters: {sorted(missing)}")


def _peak_rss_kib():
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize // 1024
        except (AttributeError, OSError):
            pass
    return None


def cost(start_wall, start_cpu):
    return {"wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
            "process_peak_rss_kib": _peak_rss_kib(),
            "rss_scope": ("process_lifetime_high_water_mark_linux_not_incremental"
                           if resource is not None else "process_peak_working_set_windows"),
}


def run_factorial(frame, config, evaluator, *, checkpoint=lambda cells: None):
    require_api(evaluator)
    cohort = frame_fingerprint(frame)
    cells = []
    references = {}
    started = time.perf_counter()
    for history, space in CELLS:
        for seed in config["seeds"]:
            if time.perf_counter() - started > config["max_wall_seconds"]:
                raise TimeoutError("between-cell wall budget exhausted (in-flight fits are not interrupted)")
            cell = {"history_mode": history, "feature_space": space, "seed": seed,
                    "cohort_fingerprint": cohort, "status": "running"}
            cells.append(cell)
            checkpoint(cells)
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                options = {key: config[key] for key in (
                    "warmup_fraction", "training_blocks", "forward_max_candidates", "forward_steps",
                    "candidate_n_negatives", "candidate_max_queries", "candidate_sampling")}
                result = evaluator(frame, seed=seed, n_folds=config["folds"], dataset="kion",
                                   history_mode=history, feature_space=space,
                                   candidate_catalog="cohort", **options)
                if frame_fingerprint(frame) != cohort:
                    raise ValueError("evaluator mutated shared cohort")
                validate_result(result, config)
                assert_comparable(result, result)  # Also checks mandatory identity fields.
                if seed in references:
                    assert_comparable(references[seed], result)
                else:
                    references[seed] = result
                cell.update(status="completed", result=json_value(result))
            except Exception as exc:
                cell.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                cell["cost"] = cost(wall, cpu)
                checkpoint(cells)
    return cells


def execute(lock, output, *, loader=None, evaluator=None):
    """Load KION once, require a real-data tooling smoke, then execute locked cells."""
    if Path(output).exists():
        raise FileExistsError(f"output already exists: {output}")
    config = lock["config"]
    artifact = {"schema": SCHEMA, "status": "running", "lock": lock,
                "started_utc": datetime.now(timezone.utc).isoformat(), "cells": [],
                "interpretation": "exploratory offline observed-watch discrimination; not causal or online impact",
                "budget_policy": "wall budget checked between cells; in-flight fits not interrupted"}
    wall, cpu = time.perf_counter(), time.process_time()
    atomic_json(output, artifact)
    try:
        if loader is None:
            from kion_data import load_kion
            loader = load_kion
        if evaluator is None:
            from rolling_eval import run_rolling_frame
            evaluator = run_rolling_frame
        require_api(evaluator)
        load_wall, load_cpu = time.perf_counter(), time.process_time()
        frame = loader(config["inputs"]["interactions"], users=config["inputs"].get("users"),
                       items=config["inputs"].get("items"), max_users=config["max_users"],
                       seed=config["cohort_seed"], watch_threshold=config["watch_threshold"])
        artifact["load_cost"] = cost(load_wall, load_cpu)
        artifact["cohort"] = {"rows": len(frame), "users": int(frame["user"].nunique()),
                              "items": int(frame["item"].nunique()),
                              "fingerprint": frame_fingerprint(frame),
                              "kion_metadata": frame.attrs.get("kion_metadata")}
        rng = np.random.default_rng(config["cohort_seed"])
        users = np.sort(frame["user"].unique())
        chosen = rng.choice(users, min(config["smoke_users"], len(users)), replace=False)
        smoke_frame = frame[frame["user"].isin(chosen)].reset_index(drop=True)
        smoke_config = {**config, "folds": 1, "seeds": config["seeds"][:1],
                        "forward_steps": 1, "forward_max_candidates": 2,
                        "candidate_max_queries": 5, "candidate_n_negatives": 5}
        artifact["smoke"] = {"status": "running", "rows": len(smoke_frame),
                             "users": len(chosen), "config": smoke_config,
                             "interpretation": "tooling smoke only; not research evidence"}

        def smoke_checkpoint(cells):
            artifact["smoke"]["cells"] = cells
            atomic_json(output, artifact)

        artifact["smoke"]["preflight"] = preflight(smoke_frame, smoke_config)
        run_factorial(smoke_frame, smoke_config, evaluator, checkpoint=smoke_checkpoint)
        artifact["smoke"]["status"] = "passed"
        if config["smoke_only"]:
            artifact["status"] = "smoke_passed"
        else:
            def checkpoint(cells):
                artifact["cells"] = cells
                atomic_json(output, artifact)
            artifact["preflight"] = preflight(frame, config)
            run_factorial(frame, config, evaluator, checkpoint=checkpoint)
            artifact["status"] = "completed"
    except BaseException as exc:
        artifact["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        artifact["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if artifact.get("smoke", {}).get("status") == "running":
            artifact["smoke"]["status"] = "failed"
        raise
    finally:
        artifact["cost"] = cost(wall, cpu)
        artifact["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, artifact)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="hash inputs/source and write lock; no KION parse or model fit")
    mode.add_argument("--execute-lock", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kion-interactions", type=Path)
    parser.add_argument("--kion-users", type=Path)
    parser.add_argument("--kion-items", type=Path)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--cohort-seed", type=int, default=42)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--warmup-fraction", type=float, default=.2)
    parser.add_argument("--training-blocks", type=int, default=4)
    parser.add_argument("--watch-threshold", type=float, default=50.)
    parser.add_argument("--forward-max-candidates", type=int, default=10)
    parser.add_argument("--forward-steps", type=int, default=2)
    parser.add_argument("--candidate-n-negatives", type=int, default=20)
    parser.add_argument("--candidate-max-queries", type=int, default=50)
    parser.add_argument("--candidate-sampling", choices=("uniform", "popularity", "recent"), default="uniform")
    parser.add_argument("--smoke-users", type=int, default=50)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=3600.)
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            if not args.lock or not args.kion_interactions:
                parser.error("--dry-run requires --lock and --kion-interactions")
            config = {key: value for key, value in vars(args).items() if key not in {
                "dry_run", "execute_lock", "lock", "output", "kion_interactions", "kion_users", "kion_items"}}
            config["seeds"] = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            config["inputs"] = {key: str(value.resolve()) if value else None for key, value in {
                "interactions": args.kion_interactions, "users": args.kion_users, "items": args.kion_items}.items()}
            lock = write_lock(args.lock, config)
            print(json.dumps({"status": lock["status"], "lock": str(args.lock),
                              "lock_sha256": lock["lock_sha256"]}))
        else:
            if not args.output:
                parser.error("--execute-lock requires --output")
            # Execution has no overrides: use a fresh lock for any changed setting.
            import sys
            supplied = argv if argv is not None else sys.argv[1:]
            allowed = {"--execute-lock", "--output"}
            if any(value.startswith("--") and value.split("=", 1)[0] not in allowed for value in supplied):
                parser.error("execution accepts only --execute-lock and --output; change settings in a new lock")
            artifact = execute(read_lock(args.execute_lock), args.output)
            print(json.dumps({"status": artifact["status"], "output": str(args.output), "cost": artifact["cost"]}))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ablation failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
