import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("lightgbm")


def test_cli_single_split_json_passes_protocol_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recsys_corrected_artifact.compare_methods",
            "--evaluation",
            "single_split",
            "--n-events",
            "500",
            "--seeds",
            "42",
            "--json",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Exercise the actual CLI serialization boundary, then validate the saved artifact
    # with the same protocol checker used for experiment results and CI.
    artifact = tmp_path / "single-split-synthetic.json"
    artifact.write_text(result.stdout, encoding="utf-8")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset"] == "synthetic"
    assert payload["metadata"]["evaluation"] == "single_split"
    assert payload["metadata"]["packages"]["lightgbm"]
    assert len(payload["runs"]) == 1
    assert payload["metadata"]["source_tree"]["sha256"]
    assert payload["metadata"]["protocol_version"] == 4

    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "recsys_corrected_artifact.validate_protocol",
            str(artifact),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert "protocol v4, dataset=synthetic, runs=1, folds=1" in validation.stdout
