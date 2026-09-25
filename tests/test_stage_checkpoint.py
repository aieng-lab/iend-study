from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.config import StudyConfig
from study.deep_pipeline import write_error_results_locked, write_stage_checkpoint
from study.runner import _should_skip_existing
from study.tasks import TaskBundle
from cost_timer import record_cost, reset_cost_records, set_run_compute_meta

_ROOT = Path("runs") / "_test_stage_checkpoint"


def test_write_pipeline_progress_overwrites():
    from study.progress import write_pipeline_progress

    d = _ROOT / "progress"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    write_pipeline_progress(d, stage="sae_layer", layer=0)
    write_pipeline_progress(d, stage="caa_fit", policy="prediction", layer=1)
    payload = json.loads((d / "pipeline_progress.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "caa_fit"
    assert payload["layer"] == 1
    shutil.rmtree(_ROOT, ignore_errors=True)


def test_stage_checkpoint_is_partial_so_skip_existing_reenters():
    if _ROOT.exists():
        shutil.rmtree(_ROOT)
    cfg = StudyConfig(
        model_key="_test_stage_checkpoint",
        task_id="partial",
        suite_id="full",
        raw={"model": {"hf_model": "gpt2", "n_layers": 12}},
    )
    bundle = TaskBundle(
        task_id="induction",
        classes=["MATCH"],
        data_per_class={},
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame(),
    )
    write_stage_checkpoint(
        cfg,
        bundle,
        enabled={"sae", "caa"},
        method_rows=[
            {"method": "sae:MATCH:k1", "status": "ok", "metrics": {"roc_auc": 0.9}}
        ],
        errors=[],
        raw={"sae": {"selected_layer_by_class": {"MATCH": "all"}}},
        started="2026-01-01T00:00:00+00:00",
        stage="sae",
    )
    path = cfg.output_dir / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["pipeline_stage"] == "sae"
    assert payload["methods"][0]["method"] == "sae:MATCH:k1"
    should_skip, _config_changed = _should_skip_existing(cfg.output_dir, skip_existing=True)
    assert should_skip is False
    progress = json.loads(
        (cfg.output_dir / "pipeline_progress.json").read_text(encoding="utf-8")
    )
    assert progress["stage"] == "checkpoint_sae"
    shutil.rmtree(_ROOT, ignore_errors=True)


def test_stage_checkpoint_persists_completed_cost_ledger_before_later_failure():
    """A later SAE import failure must not erase completed train timing/RAM."""
    if _ROOT.exists():
        shutil.rmtree(_ROOT)
    reset_cost_records()
    set_run_compute_meta(gpu_name="NVIDIA A100-SXM4-80GB", gpu_total_memory_gb=80.0)
    record_cost(
        "train:gradiend:test", 42.0, peak_cuda_bytes=3 * 1024**3,
        phase="train", backend="gradiend",
    )
    cfg = StudyConfig(
        model_key="_test_stage_checkpoint", task_id="cost", suite_id="full",
        raw={"model": {"hf_model": "gpt2", "n_layers": 12}},
    )
    bundle = TaskBundle("induction", ["MATCH"], {}, pd.DataFrame(), pd.DataFrame())
    write_stage_checkpoint(
        cfg, bundle, enabled={"gradiend", "sae"}, method_rows=[], errors=[], raw={},
        started="2026-01-01T00:00:00+00:00", stage="train",
    )
    payload = json.loads((cfg.output_dir / "results.json").read_text(encoding="utf-8"))
    row = payload["raw"]["cost_ledger"][0]
    assert row["backend"] == "gradiend"
    assert row["peak_cuda_gb"] == 3.0
    assert row["run"]["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    reset_cost_records()
    shutil.rmtree(_ROOT, ignore_errors=True)


def test_stage_checkpoint_merges_previous_untouched_families():
    """A mid-run checkpoint must not drop prior rows for families this run
    hasn't reached yet (regression test: results.json used to start from an
    empty method list and only reconcile with the prior file at the very end,
    so an in-progress or interrupted run silently lost already-computed rows
    for every family besides the one currently training)."""
    if _ROOT.exists():
        shutil.rmtree(_ROOT)
    cfg = StudyConfig(
        model_key="_test_stage_checkpoint",
        task_id="merge",
        suite_id="full",
        raw={"model": {"hf_model": "gpt2", "n_layers": 12}},
    )
    bundle = TaskBundle(
        task_id="induction",
        classes=["MATCH"],
        data_per_class={},
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame(),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a prior *complete* run: gradiend + sae + caa all present.
    previous = {
        "schema_version": "1.13",
        "activation_protocol_version": 4,
        "status": "ok",
        "methods": [
            {"method": "gradiend:MATCH", "status": "ok", "metrics": {"roc_auc": 0.95}},
            {"method": "sae:MATCH:k1", "status": "ok", "metrics": {"roc_auc": 0.9}},
            {"method": "caa:MATCH:act_prediction", "status": "ok", "metrics": {"roc_auc": 0.85}},
        ],
        "raw": {
            "gradiend": {"note": "prior"},
            "sae": {"selected_layer_by_class": {"MATCH": "all"}},
            "caa": {"note": "prior"},
        },
        "errors": [],
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (cfg.output_dir / "results.json").write_text(
        json.dumps(previous), encoding="utf-8"
    )

    # This run is only through the "train" stage of a fresh actiend run;
    # gradiend/sae/caa haven't been touched yet this pass.
    write_stage_checkpoint(
        cfg,
        bundle,
        enabled={"gradiend", "actiend", "sae", "caa", "causal"},
        method_rows=[
            {"method": "actiend:MATCH", "status": "ok", "metrics": {"roc_auc": 0.8}}
        ],
        errors=[],
        raw={"enabled": ["actiend"]},
        started="2026-01-01T01:00:00+00:00",
        stage="train",
        previous=previous,
    )

    path = cfg.output_dir / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    method_ids = {m["method"] for m in payload["methods"]}

    # Untouched-this-run families survive the checkpoint.
    assert "gradiend:MATCH" in method_ids
    assert "sae:MATCH:k1" in method_ids
    assert "caa:MATCH:act_prediction" in method_ids
    # This run's new row is present too.
    assert "actiend:MATCH" in method_ids
    # Checkpoints must never claim "ok" mid-run (skip-existing must re-enter).
    assert payload["status"] == "partial"
    assert payload["pipeline_stage"] == "train"

    shutil.rmtree(_ROOT, ignore_errors=True)


def test_error_writer_preserves_rows_committed_by_other_method_cells():
    output_dir = _ROOT / "error_merge"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    previous = {
        "status": "partial",
        "methods": [{"method": "actiend:MATCH", "status": "ok", "metrics": {}}],
    }
    (output_dir / "results.json").write_text(json.dumps(previous), encoding="utf-8")

    result = write_error_results_locked(
        output_dir,
        {
            "status": "error",
            "methods": [],
            "error": "CUDA out of memory",
            "traceback": "trace",
        },
    )

    assert [row["method"] for row in result["methods"]] == ["actiend:MATCH"]
    assert result["status"] == "error"
    assert result["run_failed"] is True
    shutil.rmtree(_ROOT, ignore_errors=True)
