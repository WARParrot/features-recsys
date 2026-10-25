#!/usr/bin/env python3
"""One-off continuation of a run_ablation checkpoint, including across machines.

Run from the project root, with the same pinned Python/packages and thread settings:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
    tools/resume_kion_v7.py --checkpoint results/current/kion_10k_v7_final.json \
    --data-dir datasets/kion --output results/current/kion_10k_v7_resumed.json --inspect
Remove --inspect to execute. Completed cells are reused; running cells restart.
The original process/checkpoint are never changed. Output must not exist.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from recsys_corrected_artifact import run_ablation as ab


def _normalized_source_provenance(names):
    sources = {}
    for name in names:
        path = ROOT / name
        if not path.exists() and not name.startswith("src/"):
            path = SRC / "recsys_corrected_artifact" / name
        with path.open("r", encoding="utf-8", newline=None) as stream:
            sources[name] = hashlib.sha256(stream.read().encode("utf-8")).hexdigest()
    return {"sources": sources, "source_sha256": ab.digest_json(sources)}


def _source_provenance_compatible(current, original):
    """Require an exact source match before continuing a checkpoint."""
    names = set(original["sources"]) | set(current["sources"])
    normalized = _normalized_source_provenance(names)
    return normalized == {
        "sources": original["sources"],
        "source_sha256": original["source_sha256"],
    }


def prepare(checkpoint, data_dir=None):
    original = json.loads(Path(checkpoint).read_text())
    lock = original["lock"]
    unsigned = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if lock.get("schema") != ab.SCHEMA or ab.digest_json(unsigned) != lock.get("lock_sha256"):
        raise ValueError("invalid original lock")
    if original.get("schema") != ab.SCHEMA or original.get("smoke", {}).get("status") != "passed":
        raise ValueError("requires an ablation checkpoint with a passed smoke gate")
    config = copy.deepcopy(lock["config"])
    ab.validate_config(config)
    if data_dir:
        config["inputs"] = {k: str((Path(data_dir) / Path(v).name).resolve()) if v else None
                            for k, v in config["inputs"].items()}
    current = ab.provenance(config["inputs"])
    old = lock["provenance"]
    if not _source_provenance_compatible(current, old):
        raise ValueError("runtime/source mismatch: sources/source_sha256; use the original pinned environment or the supported Windows portability revision")
    for key in ("packages", "python", "thread_environment"):
        if current[key] != old[key]:
            raise ValueError(f"runtime/source mismatch: {key}; use the original pinned environment")
    if set(current["inputs"]) != set(old["inputs"]):
        raise ValueError("input set differs")
    for name, info in current["inputs"].items():
        if any(info[k] != old["inputs"][name][k] for k in ("sha256", "bytes")):
            raise ValueError(f"input content differs: {name}")
    expected = [(h, s, seed) for h, s in ab.CELLS for seed in config["seeds"]]
    completed = {}
    seen = set()
    references = {}
    cohort = original["cohort"]["fingerprint"]
    for cell in original["cells"]:
        identity = (cell["history_mode"], cell["feature_space"], cell["seed"])
        if identity not in expected or identity in seen:
            raise ValueError(f"unexpected/duplicate cell: {identity}")
        seen.add(identity)
        if cell["status"] != "completed":
            continue
        if cell["cohort_fingerprint"] != cohort:
            raise ValueError("completed cell cohort differs")
        result = cell["result"]
        ab.validate_result(result, config)
        ab.assert_comparable(result, result)
        seed = cell["seed"]
        if seed in references:
            ab.assert_comparable(references[seed], result)
        references[seed] = result
        completed[identity] = copy.deepcopy(cell)
    return original, config, current, expected, completed, references


def execute(checkpoint, output, data_dir=None, inspect_only=False):
    # Reading once gives an immutable snapshot even if the original writer is still active.
    original, config, current, expected, completed, references = prepare(checkpoint, data_dir)
    missing = [key for key in expected if key not in completed]
    print(json.dumps({"reuse_completed": len(completed), "restart_or_run": missing}, indent=2), flush=True)
    if inspect_only:
        return
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.resolve() == Path(checkpoint).resolve():
        raise ValueError("output must differ from checkpoint")
    snapshot_hash = ab.digest_json(original)
    artifact = copy.deepcopy(original)
    artifact.pop("cost", None)
    artifact.pop("finished_utc", None)
    artifact.pop("error", None)
    artifact["status"] = "running"
    artifact["cells"] = [completed[key] for key in expected if key in completed]
    artifact["continuation"] = {
        "schema": "kion_oneoff_continuation_v1", "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_payload_sha256": snapshot_hash, "host": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "provenance": current,
        "reused_cells": [list(key) for key in expected if key in completed],
        "policy": "incomplete cells restarted; same data/code/runtime; costs belong to respective hosts",
        "previous_continuation": original.get("continuation"),
    }
    wall, cpu = time.perf_counter(), time.process_time()
    ab.atomic_json(output, artifact)
    try:
        from recsys_corrected_artifact.kion_data import load_kion
        from recsys_corrected_artifact.rolling_eval import run_rolling_frame
        frame = load_kion(config["inputs"]["interactions"], users=config["inputs"].get("users"),
                          items=config["inputs"].get("items"), max_users=config["max_users"],
                          seed=config["cohort_seed"], watch_threshold=config["watch_threshold"])
        cohort = ab.frame_fingerprint(frame)
        if cohort != original["cohort"]["fingerprint"]:
            raise ValueError("loaded cohort fingerprint differs from checkpoint")
        artifact["continuation"]["preflight"] = ab.preflight(frame, config)
        options = {k: config[k] for k in (
            "warmup_fraction", "training_blocks", "forward_max_candidates", "forward_steps",
            "candidate_n_negatives", "candidate_max_queries", "candidate_sampling")}
        for history, space, seed in missing:
            if time.perf_counter() - wall > config["max_wall_seconds"]:
                raise TimeoutError("continuation wall budget exhausted between cells")
            cell = {"history_mode": history, "feature_space": space, "seed": seed,
                    "cohort_fingerprint": cohort, "status": "running"}
            artifact["cells"].append(cell)
            ab.atomic_json(output, artifact)
            print(f"Starting {history}/{space}/seed={seed}", flush=True)
            cell_wall, cell_cpu = time.perf_counter(), time.process_time()
            try:
                result = run_rolling_frame(frame, seed=seed, n_folds=config["folds"], dataset="kion",
                                           history_mode=history, feature_space=space,
                                           candidate_catalog="cohort", **options)
                if ab.frame_fingerprint(frame) != cohort:
                    raise ValueError("evaluator mutated cohort")
                ab.validate_result(result, config)
                ab.assert_comparable(result, result)
                if seed in references:
                    ab.assert_comparable(references[seed], result)
                references[seed] = result
                cell.update(status="completed", result=ab.json_value(result))
            except BaseException as exc:
                cell.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                cell["cost"] = ab.cost(cell_wall, cell_cpu)
                ab.atomic_json(output, artifact)
            print(f"Completed {history}/{space}/seed={seed} in {cell['cost']['wall_seconds'] / 60:.1f} min", flush=True)
        artifact["cells"].sort(key=lambda cell: expected.index(
            (cell["history_mode"], cell["feature_space"], cell["seed"])))
        artifact["status"] = "completed"
    except BaseException as exc:
        artifact["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        artifact["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Do not label mixed-host cell costs as one contiguous end-to-end benchmark.
        artifact["continuation"]["cost"] = ab.cost(wall, cpu)
        artifact["continuation"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
        ab.atomic_json(output, artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--inspect", action="store_true", help="validate inputs and print pending cells; no training")
    args = parser.parse_args()
    execute(args.checkpoint, args.output, args.data_dir, args.inspect)


if __name__ == "__main__":
    main()
