from recsys_corrected_artifact.compare_methods import _source_tree_hash


def test_manifest_excludes_outputs_and_build_metadata(tmp_path):
    (tmp_path / "pipeline.py").write_text("x = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "requirements.lock").write_text("numpy==2.2.6\n")
    before = _source_tree_hash(tmp_path)
    for relative in (
        "results/cost.txt", "results/generated.py", "build/lib/pipeline.py",
        "package.egg-info/SOURCES.txt", ".venv/lib/module.py", "datasets/data.txt",
        "unrelated.txt", "tests/generated.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("irrelevant output\n")
    assert _source_tree_hash(tmp_path) == before
    assert [entry["path"] for entry in before["files"]] == [
        "pipeline.py", "pyproject.toml", "requirements.lock",
    ]


def test_manifest_changes_with_source_and_is_location_independent(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root in (left, right):
        root.mkdir()
        (root / "module.py").write_text("x = 1\n")
    assert _source_tree_hash(left) == _source_tree_hash(right)
    (right / "module.py").write_text("x = 2\n")
    assert _source_tree_hash(left) != _source_tree_hash(right)
