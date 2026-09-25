from __future__ import annotations

import shutil
from pathlib import Path

from analysis.plot_family_overview import (
    aggregate_family_stats,
    build_score_grid,
    common_support_mask,
    format_cross_metric_text,
    format_global_stats_text,
    pairwise_wins,
    task_winners,
    winner_rows,
    write_family_overview_figures,
    write_global_tables,
)


def _cell(model, task, family, metric, mean, lo=None, hi=None):
    return {
        "model": model,
        "task": task,
        "family": family,
        "metric": metric,
        "mean": mean,
        "min": mean if lo is None else lo,
        "max": mean if hi is None else hi,
        "n_classes": 1,
    }


def _toy_cells():
    # encoding_E: caa wins emotion; sae wins composition; gender is a tie
    # suitability: gradiend wins gender; others 0
    rows = []
    rows += [
        _cell("gpt2-small", "emotion", "gradiend", "encoding_E", 0.97, 0.95, 0.99),
        _cell("gpt2-small", "emotion", "actiend", "encoding_E", 0.95),
        _cell("gpt2-small", "emotion", "sae", "encoding_E", 0.51),
        _cell("gpt2-small", "emotion", "caa", "encoding_E", 1.00),
        _cell("gpt2-small", "function_composition", "gradiend", "encoding_E", 0.46),
        _cell("gpt2-small", "function_composition", "actiend", "encoding_E", 0.98),
        _cell("gpt2-small", "function_composition", "sae", "encoding_E", 1.00),
        _cell("gpt2-small", "function_composition", "caa", "encoding_E", 0.84),
        _cell("gpt2-small", "gender_en", "gradiend", "encoding_E", 1.00),
        _cell("gpt2-small", "gender_en", "actiend", "encoding_E", 1.00),
        _cell("gpt2-small", "gender_en", "sae", "encoding_E", 0.89),
        _cell("gpt2-small", "gender_en", "caa", "encoding_E", 1.00),
        _cell("gpt2-small", "language", "gradiend", "encoding_E", 0.48),
        _cell("gpt2-small", "language", "actiend", "encoding_E", 0.82),
        # sae/caa language left missing (enabled on gender-like tasks)
    ]
    for fam, val in (("gradiend", 1.0), ("actiend", 0.0), ("sae", 0.2), ("caa", 0.3)):
        rows.append(_cell("gpt2-small", "gender_en", fam, "suitability", val))
    return rows


def test_winners_ties_and_common_support():
    grid = build_score_grid(
        _toy_cells(),
        metric="encoding_E",
        model="gpt2-small",
        families=("gradiend", "actiend", "sae", "caa"),
    )
    assert grid.tasks == ["gender_en", "emotion", "language", "function_composition"]
    winners = task_winners(grid)
    by_task = dict(zip(grid.tasks, winners))
    assert by_task["emotion"] == ["caa"]
    assert by_task["function_composition"] == ["sae"]
    assert set(by_task["gender_en"]) == {"gradiend", "actiend", "caa"}
    assert by_task["language"] == ["actiend"]

    common = common_support_mask(grid)
    # language is not 4-way; the other three tasks are
    assert [t for t, keep in zip(grid.tasks, common) if keep] == [
        "gender_en",
        "emotion",
        "function_composition",
    ]

    stats = {r["family"]: r for r in aggregate_family_stats(grid)}
    assert stats["caa"]["wins"] == 1  # emotion unique; gender is a tie
    assert stats["sae"]["wins"] == 1
    assert stats["actiend"]["wins"] == 1  # language
    assert stats["gradiend"]["wins"] == 0
    assert stats["caa"]["ties_first"] == 2
    assert stats["sae"]["n_scored"] == 3
    assert abs(stats["caa"]["mean"] - (1.0 + 0.84 + 1.0) / 3) < 1e-9
    assert abs(stats["caa"]["mean_common"] - (1.0 + 1.0 + 0.84) / 3) < 1e-9

    wins = pairwise_wins(grid)
    # caa vs sae: emotion + gender (caa better); composition sae better
    i_caa, i_sae = grid.fam_index("caa"), grid.fam_index("sae")
    assert wins[i_caa, i_sae] == 2
    assert wins[i_sae, i_caa] == 1


def test_winner_rows_and_text_tables():
    grid = build_score_grid(_toy_cells(), metric="encoding_E", model="gpt2-small")
    rows = {r["task"]: r for r in winner_rows(grid)}
    assert rows["emotion"]["winners"] == "caa"
    assert abs(rows["emotion"]["score"] - 1.0) < 1e-9
    assert rows["emotion"]["runner_up"] == "gradiend"
    assert "gradiend" in rows["gender_en"]["winners"]
    assert "caa" in rows["gender_en"]["winners"]

    stats = aggregate_family_stats(grid)
    text = format_global_stats_text(stats, metric="encoding_E", model="gpt2-small")
    assert "caa" in text and "mean" in text and "common" in text
    roll = format_cross_metric_text({"encoding_E": stats}, model="gpt2-small", metrics=["encoding_E"])
    assert "SUM_WINS" in roll
    assert "caa" in roll


def test_write_global_tables_and_figures():
    cells = _toy_cells()
    root = Path(__file__).resolve().parents[1] / "analysis" / "_tmp_plot_family"
    if root.exists():
        shutil.rmtree(root)
    table_dir = root / "tables"
    fig_dir = root / "figures"
    table_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)
    try:
        written = write_global_tables(cells, out=table_dir, model="gpt2-small")
        names = {p.name for p in written}
        assert "family_global_stats.csv" in names
        assert "family_global_winners.csv" in names
        assert "family_global_all.txt" in names
        assert "family_global_all.tex" in names
        txt = (table_dir / "family_global_all.txt").read_text(encoding="utf-8")
        assert "Who wins" in txt
        assert "Cross-metric rollup" in txt

        figs = write_family_overview_figures(
            cells, out=fig_dir, model="gpt2-small", metrics=["encoding_E", "suitability"]
        )
        assert figs
        stems = {p.stem for p in figs}
        assert "winner_mosaic" in stems
        assert "global_portrait" in stems
        assert "heatmap_encoding_E" in stems
        assert any(p.suffix == ".pdf" for p in figs)
    finally:
        shutil.rmtree(root, ignore_errors=True)
