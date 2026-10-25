"""Provenance compatibility checks for continuation across the Windows fix."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "resume_kion_v7", Path(__file__).resolve().parents[1] / "tools" / "resume_kion_v7.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def _source_sets():
    names = {"run_ablation.py", "rolling_eval.py"}
    current = resume._normalized_source_provenance(names)["sources"]
    original = dict(current)
    original["run_ablation.py"] = resume.ORIGINAL_RUN_ABLATION_SHA256
    current["run_ablation.py"] = resume.WINDOWS_PORTABLE_RUN_ABLATION_SHA256
    return ({"sources": original, "source_sha256": resume.ab.digest_json(original)},
            {"sources": current, "source_sha256": resume.ab.digest_json(current)})


def test_current_sources_are_accepted():
    _, current = _source_sets()
    assert resume._source_provenance_compatible(current, current)


def test_only_known_windows_portability_revision_is_accepted(monkeypatch):
    original, current = _source_sets()
    monkeypatch.setattr(resume.os, "name", "nt", raising=False)
    assert resume._source_provenance_compatible(current, original)


def test_unrelated_source_change_is_rejected(monkeypatch):
    original, current = _source_sets()
    original["sources"]["rolling_eval.py"] = "changed"
    monkeypatch.setattr(resume.os, "name", "nt", raising=False)
    assert not resume._source_provenance_compatible(current, original)


def test_portability_revision_is_not_accepted_on_unix(monkeypatch):
    original, current = _source_sets()
    monkeypatch.setattr(resume.os, "name", "posix", raising=False)
    assert not resume._source_provenance_compatible(current, original)
