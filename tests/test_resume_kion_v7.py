"""Lightweight contract checks for the one-off continuation tool (no training)."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "resume_kion_v7", Path(__file__).resolve().parents[1] / "tools" / "resume_kion_v7.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    config = dict(seeds=[42], folds=1, max_users=10, training_blocks=1,
                  forward_max_candidates=2, forward_steps=1, candidate_max_queries=2,
                  smoke_users=2, candidate_n_negatives=2, warmup_fraction=.2,
                  watch_threshold=50., max_wall_seconds=100., inputs={})
    provenance = dict(sources={}, source_sha256="source", packages={}, python="3.11",
                      thread_environment={}, inputs={})
    lock = dict(schema=resume.ab.SCHEMA, config=config, provenance=provenance)
    lock["lock_sha256"] = resume.ab.digest_json(lock)
    payload = dict(schema=resume.ab.SCHEMA, lock=lock, smoke={"status": "passed"},
                   cohort={"fingerprint": "cohort"}, cells=[
                       dict(history_mode="prequential", feature_space="base", seed=42,
                            status="completed", cohort_fingerprint="cohort", result={}),
                       dict(history_mode="prequential", feature_space="expanded", seed=42,
                            status="running", cohort_fingerprint="cohort")])
    path = tmp_path / "checkpoint.json"
    resume.ab.atomic_json(path, payload)
    monkeypatch.setattr(resume.ab, "provenance", lambda _: provenance.copy())
    monkeypatch.setattr(resume.ab, "validate_result", lambda *_: None)
    monkeypatch.setattr(resume.ab, "assert_comparable", lambda *_: None)
    return path, payload, provenance


def test_reuses_only_completed_and_inspection_writes_nothing(checkpoint, tmp_path):
    path, _, _ = checkpoint
    _, _, _, expected, completed, _ = resume.prepare(path)
    assert len(expected) == 4
    assert list(completed) == [("prequential", "base", 42)]
    output = tmp_path / "new.json"
    before = path.read_bytes()
    resume.execute(path, output, inspect_only=True)
    assert not output.exists()
    assert path.read_bytes() == before


def test_rejects_source_mismatch(checkpoint):
    path, _, provenance = checkpoint
    provenance["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="source_sha256"):
        resume.prepare(path)


def test_rejects_duplicate_cells(checkpoint):
    path, payload, _ = checkpoint
    payload["cells"].append(payload["cells"][0])
    resume.ab.atomic_json(path, payload)
    with pytest.raises(ValueError, match="duplicate"):
        resume.prepare(path)
