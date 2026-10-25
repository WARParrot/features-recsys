"""Windows portability checks for run_ablation resource accounting."""
import os

import run_ablation


def test_cost_does_not_require_unix_resource_module(monkeypatch):
    monkeypatch.setattr(run_ablation, "resource", None)
    monkeypatch.setattr(run_ablation.os, "name", "nt", raising=False)
    result = run_ablation.cost(0.0, 0.0)
    assert result["rss_scope"] == "process_peak_working_set_windows"
    assert result["process_peak_rss_kib"] is None or result["process_peak_rss_kib"] > 0
