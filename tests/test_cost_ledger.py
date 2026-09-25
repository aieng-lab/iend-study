from cost_timer import (
    get_cost_records,
    merge_cost_ledger,
    record_cost,
    reset_cost_records,
    set_run_compute_meta,
)
import pandas as pd
from analysis.aggregate import collect_method_cost_ledger, summarize_method_cost_ledger


def test_cost_rows_carry_run_hardware_provenance():
    reset_cost_records()
    set_run_compute_meta(
        gpu_name="NVIDIA L40S",
        gpu_total_memory_gb=44.4,
        run_started_at="2026-09-22T00:00:00+00:00",
    )
    record_cost("caa_fit", 12.0, phase="feature_select", backend="caa")

    row = get_cost_records()[0]
    assert row["backend"] == "caa"
    assert row["run"]["gpu_name"] == "NVIDIA L40S"
    assert row["run"]["gpu_total_memory_gb"] == 44.4


def test_cost_ledger_retains_other_families_across_refreshes():
    old = [
        {"name": "sae_select:L0", "backend": "sae", "seconds": 10.0},
        {"name": "caa_fit", "backend": "caa", "seconds": 20.0},
    ]
    refreshed_caa = [
        {"name": "caa_fit", "backend": "caa", "seconds": 30.0},
        {"name": "data_build", "seconds": 5.0},
    ]

    merged = merge_cost_ledger(old, refreshed_caa)

    assert [(row["backend"], row["seconds"]) for row in merged] == [
        ("sae", 10.0),
        ("caa", 30.0),
    ]


def test_cost_ledger_treats_backend_variants_as_their_family():
    old = [{"name": "train:actiend", "backend": "actiend", "seconds": 1.0}]
    refreshed = [
        {"name": "train:actiend_ridge", "backend": "actiend_ridge", "seconds": 2.0}
    ]

    assert merge_cost_ledger(old, refreshed) == refreshed


def test_collect_method_cost_ledger_exposes_stage_and_memory(tmp_path):
    path = tmp_path / "gpt2-small" / "benchmark" / "gender_en" / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '''{"model":"gpt2-small","task":"gender_en","suite":"runtime_benchmark",
        "raw":{"cost_ledger":[{"name":"sae_select:L0","backend":"sae",
        "ledger_unit":"sae_layer_sweep","phase":"feature_select","seconds":4.0,
        "peak_cuda_gb":5.0,"peak_cuda_reserved_gb":6.0,"peak_delta_gb":1.0,
        "run":{"gpu_name":"NVIDIA L40S","gpu_total_memory_gb":44.4}}]}}''',
        encoding="utf-8",
    )

    rows = collect_method_cost_ledger(tmp_path)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["backend"] == "sae"
    assert row["phase"] == "feature_select"
    assert row["peak_gpu_reserved_gb"] == 6.0
    assert row["gpu_name"] == "NVIDIA L40S"


def test_ledger_summary_sums_time_but_maxes_gpu_ram():
    rows = [
        {"model": "gpt2", "task": "gender_en", "suite": "bench", "backend": "sae",
         "ledger_unit": "sae_layer_sweep", "phase": "feature_select", "timer": "L0",
         "seconds": 2.0, "peak_gpu_gb": 4.0, "peak_gpu_reserved_gb": 5.0,
         "peak_delta_gb": 1.0, "gpu_name": "NVIDIA L40S", "gpu_total_memory_gb": 44.4},
        {"model": "gpt2", "task": "gender_en", "suite": "bench", "backend": "sae",
         "ledger_unit": "sae_layer_sweep", "phase": "feature_select", "timer": "L1",
         "seconds": 3.0, "peak_gpu_gb": 6.0, "peak_gpu_reserved_gb": 7.0,
         "peak_delta_gb": 2.0, "gpu_name": "NVIDIA L40S", "gpu_total_memory_gb": 44.4},
    ]
    summary = summarize_method_cost_ledger(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["wall_seconds"] == 5.0
    assert row["peak_gpu_gb"] == 6.0
    assert row["peak_gpu_reserved_gb"] == 7.0
    assert row["peak_delta_gb"] == 2.0
