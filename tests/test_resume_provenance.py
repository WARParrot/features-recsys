"""Provenance compatibility checks for continuation across the Windows fix."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "resume_kion_v7", Path(__file__).resolve().parents[1] / "tools" / "resume_kion_v7.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def _provenance(run_hash):
    sources = {"run_ablation.py": run_hash, "rolling_eval.py": "rolling"}
    return {"sources": sources, "source_sha256": "digest"}


def test_all_original_sources_are_accepted():
    original = _provenance(resume.ORIGINAL_RUN_ABLATION_SHA256)
    assert resume._source_provenance_compatible(original, original)


def test_only_known_windows_portability_revision_is_accepted(monkeypatch):
    original = _provenance(resume.ORIGINAL_RUN_ABLATION_SHA256)
    current = _provenance(resume.WINDOWS_PORTABLE_RUN_ABLATION_SHA256)
    monkeypatch.setattr(resume.os, "name", "nt", raising=False)
    assert resume._source_provenance_compatible(current, original)


def test_unrelated_source_change_is_rejected(monkeypatch):
    original = _provenance(resume.ORIGINAL_RUN_ABLATION_SHA256)
    current = _provenance(resume.WINDOWS_PORTABLE_RUN_ABLATION_SHA256)
    current["sources"]["rolling_eval.py"] = "changed"
    monkeypatch.setattr(resume.os, "name", "nt", raising=False)
    assert not resume._source_provenance_compatible(current, original)


def test_portability_revision_is_not_accepted_on_unix(monkeypatch):
    original = _provenance(resume.ORIGINAL_RUN_ABLATION_SHA256)
    current = _provenance(resume.WINDOWS_PORTABLE_RUN_ABLATION_SHA256)
    monkeypatch.setattr(resume.os, "name", "posix", raising=False)
    assert not resume._source_provenance_compatible(current, original)
