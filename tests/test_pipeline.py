import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("lightgbm")


def test_cli_json_smoke_runs_from_artifact_directory():
    result = subprocess.run(
        [sys.executable, "compare_methods.py", "--evaluation", "single_split", "--n-events", "500", "--seeds", "42", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["metadata"]["dataset"] == "synthetic"
    assert payload["metadata"]["packages"]["lightgbm"]
    assert len(payload["runs"]) == 1
    assert payload["metadata"]["source_tree"]["sha256"]
    assert payload["metadata"]["protocol_version"] == 4
