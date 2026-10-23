import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from data import generate_synthetic
    from experiment_contracts import validate_run_collection
    from mind_data import load_mind
    from kion_data import load_kion
    from pipeline import CONFIG_GRID, run_many, run_mind_many
    from rolling_eval import run_rolling_many, summarize_rolling_runs
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in missing-env usage
    raise SystemExit(
        "recsys artifact dependencies are missing; install requirements.lock first"
    ) from exc


def _installed(name):
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def package_versions():
    names = ("numpy", "pandas", "scikit-learn", "lightgbm", "pytest")
    return {name: importlib.metadata.version(name) for name in names if _installed(name)}


def _sha256(path):
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(args):
    return {
        key: _sha256(value)
        for key, value in {
            "kion_interactions": args.kion_interactions,
            "kion_users": args.kion_users,
            "kion_items": args.kion_items,
        }.items()
        if value
    }


def _code_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_tree_hash():
    """Hash the source/config files when a Git revision is unavailable."""
    root = Path(__file__).resolve().parent
    files = sorted(
        path for pattern in ("*.py", "*.toml", "*.lock", "*.txt")
        for path in root.rglob(pattern)
        if path.is_file() and ".venv" not in path.parts and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    manifest = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        manifest.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode())
        digest.update(b"\\0")
        digest.update(content)
        digest.update(b"\\0")
    return {"sha256": digest.hexdigest(), "files": manifest}


def _data_summary(results):
    folds = [fold for run in results for fold in run.get("folds", [run])]
    split_rows = [fold.get("split_metadata", {}) for fold in folds]
    return {
        "runs": int(len(results)),
        "folds": int(len(folds)),
        "test_rows": int(sum(row.get("test", {}).get("rows", 0) for row in split_rows)),
        "ranking_queries": int(sum(fold.get("ranking_queries", 0) for fold in folds)),
    }


def _limit_mind_queries(frame, max_impressions):
    if max_impressions is None:
        return frame
    if max_impressions <= 0:
        raise ValueError("max_impressions must be positive")
    queries = (
        frame[["query_id", "timestamp"]]
        .drop_duplicates("query_id")
        .sort_values(["timestamp", "query_id"], kind="mergesort")
        .head(max_impressions)["query_id"]
    )
    limited = frame[frame["query_id"].isin(set(queries))].reset_index(drop=True)
    limited.attrs["mind_metadata"] = frame.attrs.get("mind_metadata", {})
    return limited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--n-events", type=int, default=30_000)
    parser.add_argument(
        "--mind-root",
        help="run the real-impression MIND experiment on an extracted MIND-small/MIND root",
    )
    parser.add_argument(
        "--kion-interactions",
        help="run the temporal KION experiment from interactions.csv or interactions.csv.zip",
    )
    parser.add_argument("--kion-users", help="optional KION users_en.csv metadata")
    parser.add_argument("--kion-items", help="optional KION items_en.csv metadata")
    parser.add_argument("--kion-max-users", type=int, help="sample users while retaining each selected user's full history")
    parser.add_argument("--watch-threshold", type=float, default=50.0)
    parser.add_argument("--forward-max-candidates", type=int, default=60)
    parser.add_argument("--forward-steps", type=int, default=8)
    parser.add_argument(
        "--max-impressions",
        type=int,
        help="optional chronological cap for MIND impressions",
    )
    parser.add_argument(
        "--evaluation",
        choices=("single_split", "rolling"),
        default="rolling",
        help="single exploratory split or expanding rolling temporal folds",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--strict-mind-metadata",
        action="store_true",
        help="fail if any impression item is absent from news.tsv",
    )
    parser.add_argument(
        "--mind-protocol",
        choices=("combined_temporal", "official_train_dev"),
        default="combined_temporal",
        help="MIND split protocol; official_train_dev evaluates dev after train-only fitting",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.folds <= 0:
        parser.error("--folds must be positive")
    if args.n_events <= 0:
        parser.error("--n-events must be positive")
    if args.forward_max_candidates <= 0 or args.forward_steps <= 0:
        parser.error("forward search limits must be positive")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        parser.error("--seeds must contain at least one integer")

    if args.kion_interactions and args.mind_root:
        parser.error("choose only one of --kion-interactions and --mind-root")
    if args.evaluation == "rolling":
        if args.mind_root and args.mind_protocol == "official_train_dev":
            parser.error("official_train_dev is available with --evaluation single_split")
        if args.kion_interactions:
            frame = load_kion(
                args.kion_interactions, users=args.kion_users, items=args.kion_items,
                max_users=args.kion_max_users, seed=seeds[0], watch_threshold=args.watch_threshold,
            )
            results = run_rolling_many(
                frame, seeds=seeds, n_folds=args.folds, dataset="kion",
                forward_max_candidates=args.forward_max_candidates,
                forward_steps=args.forward_steps,
            )
            dataset = "kion"
        elif args.mind_root:
            frame = _limit_mind_queries(
                load_mind(
                    args.mind_root,
                    strict=args.strict_mind_metadata,
                ),
                args.max_impressions,
            )
            results = run_rolling_many(
                frame, seeds=seeds, n_folds=args.folds, dataset="mind",
                forward_max_candidates=args.forward_max_candidates,
                forward_steps=args.forward_steps,
            )
            dataset = "mind"
        else:
            results = [
                run_rolling_many(
                    generate_synthetic(n_events=args.n_events, seed=seed),
                    seeds=(seed,), n_folds=args.folds, dataset="synthetic",
                    forward_max_candidates=args.forward_max_candidates,
                    forward_steps=args.forward_steps,
                )[0]
                for seed in seeds
            ]
            dataset = "synthetic"
        summary = summarize_rolling_runs(results)
    elif args.kion_interactions:
        frame = load_kion(
            args.kion_interactions, users=args.kion_users, items=args.kion_items,
            max_users=args.kion_max_users, seed=seeds[0], watch_threshold=args.watch_threshold,
        )
        results = [
            run_rolling_many(
                frame, seeds=(seed,), n_folds=args.folds, dataset="kion",
                forward_max_candidates=args.forward_max_candidates,
                forward_steps=args.forward_steps,
            )[0]
            for seed in seeds
        ]
        summary = summarize_rolling_runs(results)
        dataset = "kion"
    elif args.mind_root:
        results, summary = run_mind_many(
            args.mind_root,
            seeds=seeds,
            max_impressions=args.max_impressions,
            strict_mind_metadata=args.strict_mind_metadata,
            mind_protocol=args.mind_protocol,
        )
        dataset = "mind"
    else:
        results, summary = run_many(seeds, args.n_events)
        dataset = "synthetic"

    validate_run_collection(results)
    if args.json:
        metadata = {
            "python": sys.version.split()[0],
            "packages": package_versions(),
            "platform": platform.platform(),
            "code_revision": _code_revision(),
            "source_tree": _source_tree_hash(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "source_hashes": _source_hashes(args),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                if os.environ.get(name) is not None
            },
            "seeds": seeds,
            "n_events": args.n_events,
            "dataset": dataset,
            "mind_root": args.mind_root,
            "kion_interactions": args.kion_interactions,
            "kion_max_users": args.kion_max_users,
            "watch_threshold": args.watch_threshold,
            "forward_max_candidates": args.forward_max_candidates,
            "forward_steps": args.forward_steps,
            "max_impressions": args.max_impressions,
            "strict_mind_metadata": args.strict_mind_metadata,
            "mind_protocol": args.mind_protocol,
            "evaluation": args.evaluation,
            "folds": args.folds if args.evaluation == "rolling" else None,
            "protocol_version": 4,
            "protocol_notes": {
                "kion_candidate_protocol": "selected-cohort historical catalog sampled unseen-item discrimination",
                "kion_feedback_semantics": "pair-aggregated observed watch record; not click/impression CTR",
                "primary_comparison": "auto_gain versus manual",
                "primary_ranking_estimand": "query_weighted_macro_mean; user-cluster bootstrap preserves query counts",
                "uncertainty_unit": "user_cluster",
            },
            "data_summary": _data_summary(results),
            "config_grid": CONFIG_GRID,
        }
        payload = {"metadata": metadata, "runs": results}
        if summary is not None:
            payload["summary"] = (
                summary.reset_index().to_dict("records")
                if hasattr(summary, "reset_index")
                else summary
            )
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.evaluation == "rolling":
        print(json.dumps({"summary": summary}, indent=2, default=str))
        for result in results:
            fold_summary = result["summary"]
            selected = [fold["selected"] for fold in result["folds"]]
            print(
                f"seed={result['seed']} dataset={dataset} folds={len(result['folds'])} "
                f"selected={selected} summary={fold_summary}"
            )
        return

    print(f"method comparison dataset={dataset}")
    print(summary.to_string(float_format=lambda value: f"{value:.5f}"))
    for result in results:
        print(
            f"seed={result['seed']} selected={result['selected']} "
            f"manual_test={result['manual_test_auc']:.5f} "
            f"selected_test={result['selected_test_auc']:.5f} "
            f"selected_ndcg10={result['selected_candidate_ndcg10']:.5f} "
            f"selected_mrr10={result['selected_candidate_mrr10']:.5f}"
        )


if __name__ == "__main__":
    main()
