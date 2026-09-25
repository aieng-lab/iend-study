"""Regression tests for analysis/paper_figures.py's causal-stage compute-cost
aggregation.

Guards two bugs found in review (2026-08-20):
  1. core and suite_full runs/{model}/ trees used to be silently pooled
     together before averaging — a task set present in one tree but not the
     other (or a suite_full-only ablation like the SAE clamp sweep) would
     skew the pooled mean without any indication. compute_summary now reads
     exactly one subdir per call.
  2. backends were averaged over whichever tasks happened to have a record,
     even when that task set differed between backends (e.g. SAE causal
     erroring out on a task GRADIEND/ACTIEND succeeded on) — silently
     comparing different task mixes under one "mean seconds" number. Fixed
     via an explicit matched-task-set intersection.

Also covers the seconds_per_unit normalization (causal_study.py's
n_strength_points count) and its fallback when older records lack it.
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis.paper_figures import compute_cross_model_summary, compute_summary


def _cost_row(backend, phase, seconds, peak_gb, *, peak_delta_gb=None, n_strength_points=None):
    row = {"backend": backend, "phase": phase, "seconds": seconds, "peak_cuda_gb": peak_gb}
    if peak_delta_gb is not None:
        row["peak_delta_gb"] = peak_delta_gb
    if n_strength_points is not None:
        row["n_strength_points"] = n_strength_points
    return row


def _write_results(path: Path, cost_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "gpt2-small", "suite": "core", "status": "ok",
        "method_rows": [], "raw": {"cost": cost_rows},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compute_summary_never_pools_core_and_suite_full(tmp_path: Path):
    model = "gpt2-small"
    # core: sae only has data on taskA (simulating an SAE-causal error on taskB)
    _write_results(tmp_path / model / "taskA" / "results.json", [
        _cost_row("gradiend", "causal", 20.0, 3.0),
        _cost_row("sae", "causal", 100.0, 8.0),
    ])
    _write_results(tmp_path / model / "taskB" / "results.json", [
        _cost_row("gradiend", "causal", 30.0, 3.0),
    ])
    # suite_full: sae has both tasks, at a much higher cost (e.g. the opt-in clamp sweep)
    _write_results(tmp_path / model / "suite_full" / "taskA" / "results.json", [
        _cost_row("gradiend", "causal", 22.0, 3.0),
        _cost_row("sae", "causal", 400.0, 9.0),
    ])
    _write_results(tmp_path / model / "suite_full" / "taskB" / "results.json", [
        _cost_row("gradiend", "causal", 32.0, 3.0),
        _cost_row("sae", "causal", 420.0, 9.0),
    ])

    core_out = tmp_path / "out_core"
    p = compute_summary(model, tmp_path, core_out, subdir="")
    assert p is not None
    text = p.read_text(encoding="utf-8")
    # core: sae's matched task set must be {taskA} only (taskB has no sae record in core) —
    # gradiend must NOT be penalized/boosted by suite_full's taskB numbers leaking in.
    assert "matched task set (every backend has a record): n=1" in text
    assert "['taskA']" in text
    # suite_full's higher sae cost (400/420) must not appear in the core-subdir table at all
    assert "400.0" not in text and "420.0" not in text

    full_out = tmp_path / "out_full"
    p2 = compute_summary(model, tmp_path, full_out, subdir="suite_full")
    assert p2 is not None
    text2 = p2.read_text(encoding="utf-8")
    assert "matched task set (every backend has a record): n=2" in text2
    # core's cheaper sae cost (100.0) must not leak into the suite_full-subdir table
    assert "100.0" not in text2


def test_compute_summary_matches_tasks_across_backends_not_just_pooled_mean(tmp_path: Path):
    model = "gpt2-small"
    # actiend has 3 tasks; sae only has 1 of those 3 (the other 2 errored for sae).
    _write_results(tmp_path / model / "taskA" / "results.json", [
        _cost_row("actiend", "causal", 10.0, 2.0),
        _cost_row("sae", "causal", 50.0, 5.0),
    ])
    _write_results(tmp_path / model / "taskB" / "results.json", [
        _cost_row("actiend", "causal", 1000.0, 2.0),  # expensive outlier, sae has no record here
    ])
    _write_results(tmp_path / model / "taskC" / "results.json", [
        _cost_row("actiend", "causal", 1000.0, 2.0),  # expensive outlier, sae has no record here
    ])

    out = tmp_path / "out"
    p = compute_summary(model, tmp_path, out, subdir="")
    assert p is not None
    text = p.read_text(encoding="utf-8")
    # matched set must be exactly {taskA} (the only task both backends have) —
    # actiend's mean must be 10.0 (taskA only), NOT (10+1000+1000)/3 pooled across
    # tasks sae never even attempted.
    assert "matched task set (every backend has a record): n=1" in text
    assert "['taskA']" in text
    lines = [l for l in text.splitlines() if l.startswith("actiend")]
    assert lines, text
    assert "10.0" in lines[0]
    assert "1000" not in lines[0]


def test_compute_summary_uses_seconds_per_unit_when_available(tmp_path: Path):
    model = "gpt2-small"
    # gradiend: 1 call, 20 strength points, 40s total -> 2.0s/unit
    # sae: 1 call, 200 strength points (bundles many layers/classes), 400s total -> 2.0s/unit
    # Same per-unit cost despite wildly different per-call seconds (40 vs 400) and
    # bundled work (20 vs 200 points) -- this is the case the raw per-call mean gets wrong.
    _write_results(tmp_path / model / "taskA" / "results.json", [
        _cost_row("gradiend", "causal", 40.0, 3.0, n_strength_points=20),
        _cost_row("sae", "causal", 400.0, 8.0, n_strength_points=200),
    ])

    out = tmp_path / "out"
    p = compute_summary(model, tmp_path, out, subdir="")
    assert p is not None
    text = p.read_text(encoding="utf-8")
    assert "has_units" in text
    for line in text.splitlines():
        if line.startswith("gradiend") or line.startswith("sae"):
            assert "2.000" in line, line


def test_compute_summary_falls_back_when_unit_data_missing(tmp_path: Path):
    model = "gpt2-small"
    _write_results(tmp_path / model / "taskA" / "results.json", [
        _cost_row("gradiend", "causal", 40.0, 3.0, n_strength_points=20),
        _cost_row("sae", "causal", 400.0, 8.0),  # no n_strength_points -- pre-instrumentation record
    ])

    out = tmp_path / "out"
    p = compute_summary(model, tmp_path, out, subdir="")
    assert p is not None
    text = p.read_text(encoding="utf-8")
    sae_line = [l for l in text.splitlines() if l.startswith("sae")][0]
    assert "n/a" in sae_line
    assert "no" in sae_line


def test_cross_model_summary_uses_each_models_own_matched_task_set(tmp_path: Path):
    _write_results(tmp_path / "model-a" / "taskA" / "results.json", [
        _cost_row("gradiend", "causal", 10.0, 2.0),
    ])
    _write_results(tmp_path / "model-a" / "taskB" / "results.json", [
        _cost_row("gradiend", "causal", 30.0, 2.0),
    ])
    _write_results(tmp_path / "model-b" / "taskA" / "results.json", [
        _cost_row("gradiend", "causal", 100.0, 2.0),
    ])

    out = tmp_path / "out"
    p = compute_cross_model_summary(["model-a", "model-b"], tmp_path, out, "ab", subdir="")
    assert p is not None
    text = p.read_text(encoding="utf-8")
    assert "[model-a] matched_tasks(n=2)=['taskA', 'taskB']" in text
    assert "[model-b] matched_tasks(n=1)=['taskA']" in text
    assert "mean_seconds=20.0" in text  # model-a: (10+30)/2
    assert "mean_seconds=100.0" in text  # model-b: taskA only
