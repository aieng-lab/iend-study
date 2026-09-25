from __future__ import annotations

import pandas as pd
import pytest

from analysis.headline_scatter import (
    _color_keys,
    _legend_handles,
    _plot_label,
    build_task_scope_tables,
    collect_complete_headline_rows,
)


def test_color_keys_normalize_pre_sae_concrete_groups():
    assert _color_keys(["sae_pre:k1", "sae_pre:kstar", "sae:k1"]) == {
        "sae_pre:k1": "sae",
        "sae_pre:kstar": "sae",
        "sae:k1": "sae",
    }


def test_headline_sae_legend_uses_only_the_method_name():
    labels = [handle.get_label() for handle in _legend_handles()["sae"]]
    assert labels == [r"$\mathrm{SAE}$"]
from analysis.summary_latex import headline_method_groups


def test_scatter_accepts_intervention_only_pre_sae_rows(tmp_path):
    pytest.importorskip("matplotlib")
    from analysis.headline_scatter import write_scatter_plots

    rows = pd.DataFrame(
        [
            {
                "model": "m",
                "task": "emotion",
                "method_group": "gradiend:two_pole",
                "detection_score": 0.8,
                "intervention_score": 0.2,
            },
            {
                "model": "m",
                "task": "induction",
                "method_group": "sae_pre:k1",
                "detection_score": 0.5,
                "intervention_score": 0.1,
            },
        ]
    )
    write_scatter_plots(rows, tmp_path, sources=[("m", "")])


def test_primary_headline_scatter_is_pooled_all_cell_figure(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from analysis.headline_scatter import write_headline_scatter

    summary = pd.DataFrame(
        [
            {
                "method_group": "gradiend:two_pole",
                "detection_score": 0.8,
                "intervention_score": 0.2,
            }
        ]
    )
    table_csv = tmp_path / "summary_methods_m.csv"
    summary.to_csv(table_csv, index=False)
    pd.DataFrame({"gradiend:two_pole": [0.7, 0.9]}, index=["task_a", "task_b"]).to_csv(
        tmp_path / "summary_matrix_detection_m.csv"
    )
    pd.DataFrame({"gradiend:two_pole": [0.1, 0.3]}, index=["task_a", "task_b"]).to_csv(
        tmp_path / "summary_matrix_intervention_m.csv"
    )

    written = write_headline_scatter([("m", table_csv)], tmp_path / "headline_scatter.pdf")
    assert written == [
        tmp_path / "headline_scatter.pdf",
        tmp_path / "headline_scatter_by_model.pdf",
        tmp_path / "headline_scatter_by_model_shared.pdf",
    ]
    assert (tmp_path / "headline_scatter.csv").is_file()


def test_row_based_headline_scatter_writes_pooled_and_by_model_pair(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from analysis.headline_scatter import write_headline_scatter_from_rows

    rows = pd.DataFrame([
        {"model": "m1", "task": "emotion", "method_group": "gradiend:two_pole",
         "detection_score": .8, "intervention_score": .2},
        {"model": "m2", "task": "emotion", "method_group": "gradiend:two_pole",
         "detection_score": .7, "intervention_score": .3},
    ])

    path = tmp_path / "headline_scatter_all_layer.pdf"
    assert write_headline_scatter_from_rows(rows, path, sources=[("m1", ""), ("m2", "")]) == [
        path,
        tmp_path / "headline_scatter_all_layer_by_model.pdf",
        tmp_path / "headline_scatter_all_layer_by_model_shared.pdf",
    ]
    assert path.is_file()
    assert (tmp_path / "headline_scatter_all_layer_by_model.pdf").is_file()


def test_construction_specific_plot_labels_drop_redundant_suffix():
    assert _plot_label("gradiend:two_pole", include_construction=False) == "GRADIEND"
    assert _plot_label("gradiend:one_pole", include_construction=False) == "GRADIEND"
    assert r"\mathrm{tn}" in _plot_label(
        "cga_tensor_norm:one_pole", include_construction=False
    )
    assert "1s" not in _plot_label(
        "cga_tensor_norm:one_pole", include_construction=False
    )


def test_empty_selection_preserves_scatter_schema():
    frame = pd.DataFrame(
        [
            {
                "task": "emotion",
                "method_group": "gradiend:two_pole",
                "detection_score": 0.8,
                "intervention_score": 0.2,
                "detection_complete": False,
                "intervention_complete": True,
            }
        ]
    )
    result = collect_complete_headline_rows({"m": frame})
    assert result.empty
    assert list(result.columns) == [
        "model",
        "task",
        "method_group",
        "detection_score",
        "intervention_score",
    ]


def test_sae_pre_is_excluded_from_the_detection_vs_intervention_input():
    frame = pd.DataFrame(
        [
            {
                "task": "induction",
                "method_group": "sae_pre:k1",
                "detection_score": 0.5,
                "intervention_score": 0.2,
                "detection_complete": True,
                "intervention_complete": True,
            }
        ]
    )
    assert collect_complete_headline_rows({"m": frame}).empty


def test_light_detection_is_derived_from_persisted_components():
    frame = pd.DataFrame(
        [{
            "task": "emotion",
            "method_group": "gradiend:two_pole",
            "roc_auc_neutral": 0.91,
            "neutral_specificity": 0.84,
            "intervention_score": 0.2,
            "detection_complete": True,
            "intervention_complete": True,
        }]
    )
    rows = collect_complete_headline_rows(
        {"m": frame}, detection_metric="detection_light_score"
    )
    assert len(rows) == 1
    assert rows.iloc[0]["detection_score"] == 0.84


def test_task_scope_figures_compare_constructions_only_where_defined():
    rows = pd.DataFrame([
        {"model": "m", "task": "emotion", "method_group": "gradiend:two_pole", "detection_score": .8, "intervention_score": .2},
        {"model": "m", "task": "emotion", "method_group": "gradiend:one_pole", "detection_score": .7, "intervention_score": .1},
        {"model": "m", "task": "induction", "method_group": "gradiend:two_pole", "detection_score": .9, "intervention_score": .3},
        {"model": "m", "task": "induction", "method_group": "gradiend:one_pole", "detection_score": .6, "intervention_score": .05},
        {"model": "m", "task": "induction", "method_group": "sae:k1", "detection_score": .5, "intervention_score": .04},
    ])
    non_one, _ = build_task_scope_tables(rows, scope="non_one_side_tasks")
    assert set(non_one["method_group"]) == {
        "gradiend:two_pole", "gradiend:one_pole"
    }
    one_side, _ = build_task_scope_tables(rows, scope="one_side_tasks")
    assert set(one_side["method_group"]) == {"gradiend:one_pole", "sae:k1"}


def test_pairwise_task_scope_by_model_uses_the_headline_panel_grid(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from analysis.headline_scatter import write_scatter_plots

    saved = {}

    def capture_savefig(figure, path, *args, **kwargs):
        saved[str(path)] = figure

    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    rows = pd.DataFrame([
        {"model": model, "task": "emotion", "method_group": method,
         "detection_score": detection, "intervention_score": intervention}
        for model, detection, intervention in (("m_small", .8, .2), ("m_large", .7, .3))
        for method in ("gradiend:two_pole", "gradiend:one_pole")
    ])

    write_scatter_plots(rows, tmp_path, sources=[("m_small", ""), ("m_large", "")])

    figure = saved[str(tmp_path / "headline_scatter_non_one_side_tasks_by_model.pdf")]
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["m_small", "m_large"]
    assert [legend.get_title().get_text() for legend in figure.legends] == [
        "Signal", "Estimator", "Construction", "SAE"
    ]


def test_headline_method_sets_keep_selected_and_full_distinct():
    frame = pd.DataFrame(
        {
            "method_group": [
                "sae:k1",
                "sae_pre:k1",
                "cga:two_pole",
                "cga_tensor_norm:two_pole",
            ]
        }
    )
    selected = headline_method_groups("selected", frame=frame)
    full = headline_method_groups("full", frame=frame)
    assert "sae_pre:k1" not in selected
    assert "cga_tensor_norm:two_pole" not in selected
    assert "sae_pre:k1" in full
    assert "cga_tensor_norm:two_pole" in full
