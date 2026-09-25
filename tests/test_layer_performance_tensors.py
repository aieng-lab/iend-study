from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.layer_performance_plots import (
    ALL_LAYER_LINESTYLE,
    FAMILY_ORDER,
    TENSORS_FAMILIES,
    _task_title,
    all_baseline_value,
    collect_tensors_baselines,
    collect_layer_points,
    plot_det_int_figure,
    plot_metric_figure,
)


def _write_results(path: Path, methods: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model": "gpt2-small", "task": path.parent.name, "methods": methods}),
        encoding="utf-8",
    )


def test_collect_tensors_baselines_reads_by_tensor_rows(tmp_path: Path):
    _write_results(
        tmp_path / "gpt2-small" / "gender_en" / "results.json",
        [
            {"method": "actiend:F-M:tensors:F", "status": "ok", "metrics": {"roc_auc_neutral": 0.9}},
            {"method": "actiend:F-M:tensors:M", "status": "ok", "metrics": {"roc_auc_neutral": 0.8}},
            {"method": "gradiend:F-M:tensors:F", "status": "ok", "metrics": {"roc_auc_neutral": 0.7}},
            # Headline (none-split) row must NOT be picked up here.
            {"method": "actiend:F-M:F", "status": "ok", "metrics": {"roc_auc_neutral": 0.5}},
        ],
    )
    out = collect_tensors_baselines(tmp_path, "gpt2-small")
    assert set(out.keys()) == {"gender_en"}
    actiend_val = all_baseline_value(out["gender_en"], "roc_auc_neutral", "actiend")
    gradiend_val = all_baseline_value(out["gender_en"], "roc_auc_neutral", "gradiend")
    assert actiend_val == (0.9 + 0.8) / 2
    assert gradiend_val == 0.7


def test_collect_tensors_baselines_empty_when_no_tensor_rows(tmp_path: Path):
    _write_results(
        tmp_path / "gpt2-small" / "gender_en" / "results.json",
        [{"method": "actiend:F-M:F", "status": "ok", "metrics": {"roc_auc_neutral": 0.5}}],
    )
    out = collect_tensors_baselines(tmp_path, "gpt2-small")
    assert out == {}


def test_plot_metric_figure_ignores_by_tensor_rows(tmp_path: Path):
    # By-tensor is not a layer sweep and must not create a layerwise figure.
    path = plot_metric_figure(
        {},
        {},
        metric="roc_auc_neutral",
        metric_label="AUC",
        model="gpt2-small",
        tasks_order=["gender_en"],
        out_dir=tmp_path,
    )
    assert path is None


def test_tensors_families_are_gradiend_and_actiend():
    assert set(TENSORS_FAMILIES) == {"gradiend", "actiend"}


def test_layer_plot_labels_use_rendered_task_text_and_study_method_order():
    assert _task_title("gender_en") == "Gender"
    assert _task_title("gender_en", latex=True) == r"\taskGender"
    assert _task_title("ravel_country") == "RavelCntry"
    assert FAMILY_ORDER == ("gradiend", "actiend", "cga", "caga", "agiend", "sae", "caa")
    assert "sae_pre" not in FAMILY_ORDER
    assert ALL_LAYER_LINESTYLE == {"two_pole": "-", "one_pole": ":"}


def test_combined_layer_plot_renders_three_inline_legend_groups(tmp_path: Path):
    points = {
        "gender_en": [
            {"family": "cga", "pole": "two_pole", "cls": "F", "layer": 0,
             "detection_score": 0.7, "intervention_score": 0.2},
            {"family": "cga", "pole": "two_pole", "cls": "F", "layer": 1,
             "detection_score": 0.8, "intervention_score": 0.3},
            {"family": "cga", "pole": "one_pole", "cls": "F", "layer": 0,
             "detection_score": 0.6, "intervention_score": 0.1},
            {"family": "cga", "pole": "one_pole", "cls": "F", "layer": 1,
             "detection_score": 0.65, "intervention_score": 0.15},
        ]
    }
    all_points = {
        "gender_en": [
            {"family": "cga", "pole": "two_pole", "cls": "F",
             "detection_score": 0.75, "intervention_score": 0.25},
            {"family": "cga", "pole": "one_pole", "cls": "F",
             "detection_score": 0.62, "intervention_score": 0.12},
        ]
    }
    path = plot_det_int_figure(
        points, all_points, model="gpt2-small", tasks_order=["gender_en"], out_dir=tmp_path,
    )
    assert path is not None and path.is_file()


def test_collect_layer_points_reads_cga_and_caga_pair_and_one_pole_ids(tmp_path: Path):
    _write_results(
        tmp_path / "gpt2-small" / "gender_en" / "results.json",
        [
            {"method": "cga:F-M:F:L0", "metrics": {"roc_auc_neutral": 0.8}},
            {"method": "caga:F:L1", "metrics": {"roc_auc_neutral": 0.9}},
            {"method": "cga:F-M:F", "metrics": {"roc_auc_neutral": 0.7}},
            {"method": "caga:F", "metrics": {"roc_auc_neutral": 0.6}},
        ],
    )
    layers, aggregate = collect_layer_points(tmp_path, "gpt2-small")
    assert {(p["family"], p["cls"], p["layer"]) for p in layers["gender_en"]} == {
        ("cga", "F", 0),
        ("caga", "F", 1),
    }
    assert {(p["family"], p["cls"]) for p in aggregate["gender_en"]} == {
        ("cga", "F"),
        ("caga", "F"),
    }
