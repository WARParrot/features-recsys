"""Source provenance checks for checkpoint continuation."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "resume_kion_v7", Path(__file__).resolve().parents[1] / "tools" / "resume_kion_v7.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def _source_sets():
    names = {"run_ablation.py", "rolling_eval.py"}
    current = resume._normalized_source_provenance(names)
    original = {
        "sources": dict(current["sources"]),
        "source_sha256": current["source_sha256"],
    }
    return original, current


def test_current_sources_are_accepted():
    original, current = _source_sets()
    assert resume._source_provenance_compatible(current, original)


def test_source_change_is_rejected():
    original, current = _source_sets()
    original["sources"]["rolling_eval.py"] = "changed"
    original["source_sha256"] = resume.ab.digest_json(original["sources"])
    assert not resume._source_provenance_compatible(current, original)


def test_old_flat_module_checkpoint_is_not_resumable_after_package_migration():
    original, current = _source_sets()
    original["sources"]["run_ablation.py"] = "pre-package-layout-source"
    original["source_sha256"] = resume.ab.digest_json(original["sources"])
    assert not resume._source_provenance_compatible(current, original)
