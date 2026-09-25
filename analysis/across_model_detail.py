#!/usr/bin/env python
"""Build detailed across-model Det./Int. tables from generated summary CSVs.

The input is deliberately ``summary_merged_<model>.csv``, the canonical CSV
written by :mod:`analysis.summary_latex`.  This keeps the across-model details
on precisely the same selected sites and causal provenance as the per-model
summary tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suitability import detection_score, intervention_score
from analysis.autorank_config import bayesian_autorank_kwargs
from analysis.summary_latex import (
    HEADLINE_METHOD_LATEX,
    MODEL_LATEX,
    PAPER_TASK_ORDER,
    TASK_COMMAND_DEFINITIONS,
    headline_method_groups,
    task_label_rendered,
)
from analysis.plot_style import method_color, show_plot_if_requested


DEFAULT_OUT = ROOT / "analysis" / "tables" / "latex_across_models"

# This registry is deliberately paper-facing rather than inferred from method
# availability: an incomplete run must not silently change a task's category.
TASK_REGISTRY = {
    "gender_en": ("Gender", "male / female", "semantic", "P/1S", 2),
    "emotion": ("Emotion", "positive / negative", "semantic", "P/1S", 2),
    "race": ("Race", "Asian / Black / White", "semantic", "P/1S", 3),
    "religion": ("Religion", "Christian / Muslim / Jewish", "semantic", "P/1S", 3),
    "pronoun_number": ("Pronoun number", "singular / plural", "grammatical", "P/1S", 2),
    "pronoun_person": ("Pronoun person", "first / second / third", "grammatical", "P/1S", 3),
    "ravel_continent": ("RAVEL continent", "Asia / Europe / Africa", "factual", "P/1S", 3),
    "ravel_country": ("RAVEL country", "United States / China / Russia", "factual", "P/1S", 3),
    "ravel_language": ("RAVEL language", "English / Portuguese / Spanish", "factual", "P/1S", 3),
    "language": ("Language ID", "English / French / German", "language", "P/1S", 3),
    # A one-sided task has one learned factual feature.  Its listed
    # counterfactual/distractor is an evaluation target, not a second feature
    # class, so it belongs in the one-class stratum.
    "ioi_mib": ("MIB-IOI", "IO (subject counterfactual)", "algorithmic", "1S", 1),
    "key_value": ("Key--value", "value (other-value counterfactual)", "algorithmic", "1S", 1),
    "induction": ("Induction", "match (distractor counterfactual)", "algorithmic", "1S", 1),
    "repetition": ("Repetition", "yes (no counterfactual)", "algorithmic", "1S", 1),
    "function_composition": ("Function composition", "result (distractor counterfactual)", "algorithmic", "1S", 1),
}

# The pooled headline matrix is intentionally limited to the fifteen primary
# study tasks.  The one-pole race/religion aliases in PAPER_TASK_ORDER are
# evaluation variants, not additional study tasks.
MAIN_STUDY_TASKS = tuple(task for task in PAPER_TASK_ORDER if task in TASK_REGISTRY)
TASK_LATEX = {
    "gender_en": r"\taskGender", "emotion": r"\taskEmotion", "race": r"\taskRace",
    "religion": r"\taskReligion", "pronoun_number": r"\taskPronNum",
    "pronoun_person": r"\taskPronPers", "ravel_continent": r"\taskRavelCont",
    "ravel_country": r"\taskRavelCountry", "ravel_language": r"\taskRavelLang",
    "language": r"\taskLangID", "ioi_mib": r"\taskMIBIOI",
    "key_value": r"\taskKeyValue", "induction": r"\taskInduction",
    "repetition": r"\taskRepetition", "function_composition": r"\taskFuncComp",
}

# The four factorial cells have a unique interpretation. Other methods are
# retained in the tables but explicitly labelled as references rather than
# forced into an inaccurate learned/mean label.
METHOD_DESIGN = {
    "gradiend": ("parameter gradient", "IEND (learned)"),
    "actiend": ("activation value", "IEND (learned)"),
    "actiend_ridge": ("activation value", "frozen-score ridge"),
    "cga": ("parameter gradient", "contrastive mean"),
    "caa": ("activation value", "contrastive mean"),
    "caga": ("activation gradient", "contrastive mean"),
    "agiend": ("activation gradient", "IEND (learned)"),
    "sae": ("activation value", "SAE feature discovery"),
    "sae_pre": ("activation value", "SAE feature discovery"),
}
HEADLINE_METHODS = frozenset(headline_method_groups("selected"))

# Paper-tuning knobs. Keep all presentation changes here rather than scattering
# magic constants through individual figures.
PLOT_STYLE = {
    "task_width": 17.0,
    "task_height": 7.3,
    "task_title_size": 20,
    "task_x_tick_size": 16,
    "task_y_tick_size": 14,
    "task_cell_size": 9,
    # The colorbars are supporting keys, not third plot panels.  Keep them
    # compact enough that the task labels and cells remain the visual focus.
    "task_cbar_label_size": 15,
    "task_cbar_tick_size": 12,
    "task_cbar_fraction": 0.032,
    "task_cbar_shrink": 1,
    "task_cbar_aspect": 28,
    "task_cbar_pad": 0.018,
    "task_panel_wspace": 0.108,
    "autorank_2x2_wspace": 0.3,
    "autorank_2x2_hspace": 0.49,
    "autorank_width_per_method": 1.75,
    "autorank_min_width": 14.5,
    "autorank_height_per_method": 0.86,
    "autorank_min_height": 5.2,
    "autorank_title_size": 14,
    "autorank_tick_size": 11,
    "autorank_annotation_size": 8,
    "overview_width": 15.5,
    "overview_height": 5.8,
    "overview_tick_size": 14,
    "autorank_grid_row_label_size": 15,
    "autorank_grid_cbar_label_size": 14,
    "autorank_grid_cbar_tick_size": 10,
    "autorank_decision_2x2_cbar_label_size": 15,
    "method_separator_width": 1.1,
    "method_separator_color": "#303030",
    "task_icon_size": 12,
}

# The local Conda distribution bundles Font Awesome 6 Free Solid.  Its glyph
# codepoints are compatible with the Font Awesome names used in the paper's
# Font Awesome 7 LaTex macros below.
MIKTEX_BIN = Path(os.environ.get("MIKTEX_BIN", "MiKTeX-not-configured"))


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _score(row: Mapping[str, object], metric: str) -> float:
    """Return a usable headline score, otherwise NaN.

    The ``*_complete`` flags audit whole evaluation bundles.  A missing
    non-headline component must not suppress an otherwise valid Det./Int.
    scalar: pooled values are means over the models that supplied that scalar.
    """
    if metric == "detection_neutral":
        values = [row.get("roc_auc_neutral", row.get("roc_auc")), row.get("neutral_specificity", row.get("specificity"))]
        return min(map(float, values)) if all(_finite(value) for value in values) else float("nan")
    value = detection_score(row) if metric == "detection" else intervention_score(row)
    return float(value) if value is not None and _finite(value) else float("nan")


def read_sources(sources: Sequence[Tuple[str, Path]]) -> pd.DataFrame:
    """Load the generated canonical CSVs and derive the two headline metrics."""
    frames: List[pd.DataFrame] = []
    required = {"task", "method_group"}
    for model, path in sources:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is not a summary_merged CSV; missing {sorted(missing)}")
        frame = frame.copy()
        frame["model"] = model  # Command-line source is the workflow's authority.
        records = frame.to_dict(orient="records")
        frame["detection"] = [_score(record, "detection") for record in records]
        # Neutral-only Detection deliberately excludes AUC_o and exclusivity.
        # It is the fair cross-regime detector metric for P and 1S tasks.
        frame["detection_neutral"] = [_score(record, "detection_neutral") for record in records]
        frame["intervention"] = [_score(record, "intervention") for record in records]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def detail_matrix(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """One ``Det./Int.`` percentage cell per method × task for one model."""
    subset = frame[
        frame["model"].eq(model) & frame["method_group"].isin(HEADLINE_METHODS)
    ].copy()
    present_tasks = set(subset["task"].dropna().astype(str))
    tasks = [task for task in MAIN_STUDY_TASKS if task in present_tasks]
    methods = [method for method in headline_method_groups("selected") if method in set(subset["method_group"].astype(str))]
    values: Dict[Tuple[str, str], str] = {}
    for record in subset.to_dict(orient="records"):
        det, intervention = record["detection"], record["intervention"]
        if not _finite(det) and not _finite(intervention):
            continue
        det_text = "--" if not _finite(det) else f"{100 * float(det):.1f}"
        int_text = "--" if not _finite(intervention) else f"{100 * float(intervention):.1f}"
        values[(str(record["method_group"]), str(record["task"]))] = f"{det_text}/{int_text}"
    out = pd.DataFrame("--", index=methods, columns=tasks)
    out.index.name = "method_group"
    for key, value in values.items():
        out.loc[key[0], key[1]] = value
    return out


def headline_pooled_detail_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Headline method × task Det./Int. cells, each averaged over models."""
    methods = list(headline_method_groups("selected"))
    present_tasks = set(frame["task"].dropna().astype(str))
    tasks = [task for task in MAIN_STUDY_TASKS if task in present_tasks]
    work = frame[frame["method_group"].isin(methods) & frame["task"].isin(tasks)].copy()
    grouped = work.groupby(["method_group", "task"], as_index=False)[
        ["detection", "intervention"]
    ].mean()
    values: Dict[Tuple[str, str], str] = {}
    for record in grouped.to_dict(orient="records"):
        det, intervention = record["detection"], record["intervention"]
        if not _finite(det) and not _finite(intervention):
            continue
        det_text = "--" if not _finite(det) else f"{100 * float(det):.1f}"
        int_text = "--" if not _finite(intervention) else f"{100 * float(intervention):.1f}"
        values[(str(record["method_group"]), str(record["task"]))] = f"{det_text}/{int_text}"
    present_methods = [method for method in methods if method in set(work["method_group"].astype(str))]
    out = pd.DataFrame("--", index=present_methods, columns=tasks)
    out.index.name = "method_group"
    for key, value in values.items():
        out.loc[key[0], key[1]] = value
    return out


def automatic_rank_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank comparable pole constructions within each available model/task.

    Pairwise methods compete only with pairwise methods, and one-sided methods
    only with one-sided methods. SAE is a shared reference and enters both
    comparisons.  Missing cells are simply absent from the corresponding
    within-model/task ranking.
    """
    rows: List[Dict[str, object]] = []
    work = frame.copy()
    strata = (("Pairwise", "two_pole"), ("One-sided", "one_pole"))
    for metric in ("detection", "intervention"):
        for stratum, pole in strata:
            valid = work[
                work["pole"].eq(pole) | work["backend"].eq("sae")
            ].dropna(subset=[metric]).copy()
            valid["rank"] = valid.groupby(["model", "task"])[metric].rank(
                ascending=False, method="average"
            )
            for method, group in valid.groupby("method_group", sort=True):
                n_cells = int(len(group))
                rank_sd = float(group["rank"].std(ddof=1)) if n_cells > 1 else 0.0
                rank_se = rank_sd / math.sqrt(n_cells) if n_cells else float("nan")
                rows.append(
                    {
                        "method_group": method,
                        "metric": "Det." if metric == "detection" else "Int.",
                        "comparison": stratum,
                        "mean_rank": group["rank"].mean(),
                        "n_cells": n_cells,
                        "mean_score_percent": 100 * group[metric].mean(),
                        # Descriptive cell-level uncertainty, not a significance test.
                        "rank_ci95_low": group["rank"].mean() - 1.96 * rank_se,
                        "rank_ci95_high": group["rank"].mean() + 1.96 * rank_se,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["metric", "comparison", "mean_rank", "mean_score_percent", "method_group"])
        out.insert(0, "rank", out.groupby(["metric", "comparison"]).cumcount() + 1)
    return out


def headline_automatic_rank_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive rank table restricted to the selected headline methods."""
    methods = set(headline_method_groups("selected"))
    return automatic_rank_table(frame[frame["method_group"].isin(methods)].copy())


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("*", r"\textasteriskcentered{}")


def _plot_method_label(method: str) -> str:
    """Matplotlib equivalent of the paper's LaTeX method label."""
    # ``\MethodSAE`` is a macro defined only in the generated LaTex document.
    # Matplotlib cannot expand it, so render its intended paper-facing label.
    if method == "sae:k1":
        return "SAE"
    return HEADLINE_METHOD_LATEX.get(method, method)


def _method_group_boundaries(methods: Sequence[str]) -> List[float]:
    """Boundaries between method families, not between pw/1s siblings."""
    boundaries: List[float] = []
    for index in range(len(methods) - 1):
        backend = methods[index].split(":", 1)[0]
        next_backend = methods[index + 1].split(":", 1)[0]
        if backend != next_backend:
            boundaries.append(index + 0.5)
    return boundaries


def _configure_latex_task_labels(plt: object) -> None:
    """Configure PGF/XeLaTeX task labels with the paper's bundled serif font."""
    try:
        miktex_available = MIKTEX_BIN.is_dir()
    except OSError:
        # A WSL Python can see a Windows-style path whose ACL it cannot query.
        miktex_available = False
    if miktex_available and str(MIKTEX_BIN) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(MIKTEX_BIN) + os.pathsep + os.environ.get("PATH", "")
    # Give XeLaTeX the same bundled Times face as Matplotlib.  This works from
    # both Windows (C:/...) and WSL (/mnt/c/...) and avoids relying on TeX Gyre
    # being installed in the local TeX distribution.
    font_dir = ROOT.as_posix().rstrip("/") + "/"
    preamble = "\n".join((
        r"\usepackage{fontspec}",
        rf"\setmainfont{{times.ttf}}[Path={{{font_dir}}}]",
        r"\usepackage{fontawesome7}",
        *TASK_COMMAND_DEFINITIONS,
    ))
    plt.rcParams["text.latex.preamble"] = preamble
    plt.rcParams["pgf.texsystem"] = "xelatex"
    plt.rcParams["pgf.preamble"] = preamble
    plt.rcParams["pgf.rcfonts"] = False
    # Generic family names make backend_pgf emit \rmfamily instead of a
    # machine-specific \setmainfont{...} command from Matplotlib's font cache.
    plt.rcParams["font.family"] = "serif"


def _draw_task_labels(ax: object, tasks: Sequence[str]) -> None:
    """Draw exact LaTex ``\faIcon…\textsc…`` task labels on the figure."""
    for index, task in enumerate(tasks):
        # Do not set ``usetex=True`` on the artist.  That routes an eager draw
        # (notably TkAgg's idle draw) through Matplotlib's legacy ``latex``
        # command, which is pdfTeX here and cannot load the XeLaTeX-only
        # ``fontspec`` preamble.  The figure itself is saved with backend_pgf,
        # whose configured xelatex engine expands these macros correctly.
        ax.text(index, -0.022, TASK_LATEX.get(task, task_label_rendered(task)), transform=ax.get_xaxis_transform(), rotation=45, ha="right", va="top", clip_on=False, fontsize=PLOT_STYLE["task_x_tick_size"])


def _matrix_latex(
    matrix: pd.DataFrame, model: str, *, headline: bool = False,
    models: Sequence[str] | None = None,
) -> str:
    columns = list(matrix.columns)
    rendered_columns = [TASK_LATEX.get(column, _latex_escape(task_label_rendered(column))) for column in columns]
    model_caption = ", ".join(MODEL_LATEX.get(item, _latex_escape(item)) for item in (models or [model]))
    lines = [
        r"\begin{table}[p]", r"\centering", r"\scriptsize",
        rf"\caption{{{MODEL_LATEX.get(model, _latex_escape(model))} per-task Det./Int. (percent). Models: {model_caption}.}}",
        r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
        "Method & " + " & ".join(rendered_columns) + r" \\", r"\midrule",
    ]
    for method, row in matrix.iterrows():
        label = HEADLINE_METHOD_LATEX.get(str(method), _latex_escape(method)) if headline else _latex_escape(method)
        lines.append(label + " & " + " & ".join(map(str, row)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)






def write_task_and_factor_tables(frame: pd.DataFrame, out: Path) -> List[Path]:
    """Write the fixed task registry plus CSV-only factor strata.

    The factor CSV keeps standard Det., neutral-only Det., and Int. separate;
    it never substitutes the neutral-only metric for the stricter pairwise
    Det. metric.
    """
    task_rows = [
        {"task": task, "display_name": name, "feature_classes": classes, "family": family,
         "evaluation": evaluation, "n_feature_classes": n_classes}
        for task, (name, classes, family, evaluation, n_classes) in TASK_REGISTRY.items()
    ]
    tasks = pd.DataFrame(task_rows)
    task_csv = out / "task_summary.csv"
    tasks.to_csv(task_csv, index=False)
    halves = [task_rows[:8], task_rows[8:]]
    lines = [r"\begin{table}[p]", r"\centering", r"\scriptsize",
             r"\caption{Summary of the 15 feature-learning tasks. P denotes pairwise and 1S one-sided evaluation.}",
             r"\begin{tabular}{llll@{\hspace{1.1em}}llll}", r"\toprule",
             "Task & Classes & Family & Eval. & Task & Classes & Family & Eval. \\\\", r"\midrule"]
    for left, right in zip(halves[0], halves[1] + [None]):
        def cells(row):
            return ["", "", "", ""] if row is None else [row["display_name"], row["feature_classes"], row["family"], row["evaluation"]]
        lines.append(" & ".join(cells(left) + cells(right)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    task_tex = out / "task_summary.tex"
    task_tex.write_text("\n".join(lines), encoding="utf-8")

    work = frame.copy()
    work["task_family"] = work["task"].map(lambda task: TASK_REGISTRY.get(str(task), ("", "", "other", "", 0))[2])
    work["n_feature_classes"] = work["task"].map(lambda task: TASK_REGISTRY.get(str(task), ("", "", "", "", 0))[4])
    # Task P/1S capability is only an internal matching variable.  It is not a
    # paper-facing factor: the regime comparison below is about the METHOD
    # construction (pairwise vs one-sided), not about different task subsets.
    work["task_support"] = work["task"].map(
        lambda task: TASK_REGISTRY.get(str(task), ("", "", "", "other", 0))[3]
    )
    work["signal"], work["estimator"] = zip(*[
        METHOD_DESIGN.get(str(backend), ("other/reference", "other/reference"))
        for backend in work.get("backend", pd.Series("", index=work.index))
    ])
    rows: List[Dict[str, object]] = []
    for factor in ("signal", "estimator", "task_family", "n_feature_classes"):
        for metric, label in (("detection", "Det."), ("detection_neutral", "min(AUC_n, Spec_n)"), ("intervention", "Int.")):
            for keys, group in work.dropna(subset=[metric]).groupby([factor, "method_group"], sort=True):
                rows.append({"factor": factor, "level": keys[0], "method_group": keys[1], "metric": label,
                             "mean_percent": 100 * group[metric].mean(),
                             "n_model_task_cells": len(group)})

    # Genuine method-evaluation-regime factor.  Compare two-pole and one-pole
    # constructions only on tasks supporting both, so PW and 1S have the same
    # task population.  SAE is absent because it has no pole-specific variant.
    regime_work = work[
        work["task_support"].eq("P/1S")
        & work["pole"].isin(["two_pole", "one_pole"])
    ].copy()
    regime_work["method_family"] = regime_work["backend"].astype(str).str.strip()
    regime_work.loc[regime_work["method_family"].eq(""), "method_family"] = (
        regime_work.loc[regime_work["method_family"].eq(""), "method_group"]
        .astype(str).str.split(":").str[0]
    )
    pair_keys = ["model", "task", "method_family"]
    paired = (
        regime_work.groupby(pair_keys)["pole"]
        .nunique()
        .eq(2)
        .rename("_has_both_poles")
        .reset_index()
    )
    regime_work = regime_work.merge(paired, on=pair_keys, how="left")
    regime_work = regime_work[regime_work["_has_both_poles"].fillna(False)].copy()
    regime_work["method_evaluation_regime"] = regime_work["pole"].map(
        {"two_pole": "PW", "one_pole": "1S"}
    )
    for metric, label in (("detection", "Det."), ("detection_neutral", "min(AUC_n, Spec_n)"), ("intervention", "Int.")):
        for keys, group in regime_work.dropna(subset=[metric]).groupby(
            ["method_evaluation_regime", "method_group"], sort=True
        ):
            rows.append({
                "factor": "method_evaluation_regime",
                "level": keys[0],
                "method_group": keys[1],
                "metric": label,
                "mean_percent": 100 * group[metric].mean(),
                "n_model_task_cells": len(group),
            })
    factor_csv = out / "factor_breakdown.csv"
    pd.DataFrame(rows).to_csv(factor_csv, index=False, float_format="%.4f")
    note = out / "factor_analysis_README.md"
    note.write_text(
        "# Factor breakdown\n\n"
        "`factor_breakdown.csv` stratifies the generated canonical summary rows by signal, estimator, task family, feature-class count, and method evaluation regime. "
        "Method evaluation regime compares PW (`two_pole`) with 1S (`one_pole`) on matched method × model × task cells from P/1S-capable tasks; SAE is excluded because it has no pole-specific variant. "
        "`Det. neutral-only` is `min(AUC_n, Spec_n)` and intentionally does not use rival-class metrics (`AUC_o`, exclusivity), so it is the fair PW-vs-1S detection view. "
        "Standard `Det.` remains the stricter pairwise-aware metric where rival metrics exist.\n",
        encoding="utf-8",
    )
    return [task_csv, task_tex, factor_csv, note]


def write_compute_hardware_inventory(frame: pd.DataFrame, out: Path) -> Path:
    """Audit recorded hardware/RAM and compute normalization availability."""
    rows: List[Dict[str, object]] = []
    for record in frame.drop_duplicates("results_path").to_dict(orient="records"):
        path = Path(str(record.get("results_path") or ""))
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        compute = payload.get("compute") or {}
        run = compute.get("run") or {}
        costs = (payload.get("raw") or {}).get("cost") or []
        rows.append({
            "model": record.get("model"), "task": record.get("task"), "gpu_name": run.get("gpu_name"),
            "gpu_total_memory_gb": run.get("gpu_total_memory_gb"),
            "peak_gpu_overall_gb": compute.get("peak_gpu_overall_gb"),
            "peak_gpu_causal_gb": compute.get("peak_gpu_causal_gb"),
            "peak_delta_causal_gb": compute.get("peak_delta_causal_gb"),
            "has_strength_point_units": any(item.get("n_strength_points") is not None for item in costs if isinstance(item, dict)),
            "results_path": str(path),
        })
    path = out / "compute_hardware_inventory.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (out / "compute_comparison_README.md").write_text(
        "# Compute-comparison protocol\n\n"
        "`compute_hardware_inventory.csv` records the GPU model, installed GPU RAM, and measured peak allocation for every source result. "
        "Do not compare raw wall-clock seconds across different GPU types, code revisions, or unmatched task sets. "
        "A fair method comparison holds model, GPU type, suite, and task set fixed, uses causal-stage peak *delta* memory, and normalizes seconds by the recorded `(class, strength-point)` count. "
        "`has_strength_point_units=false` means that historical artifact predates that counter and supports only a hardware audit, not a normalized runtime claim. "
        "`analysis/paper_figures.py` already implements the matched-task causal-stage comparison when those unit counters are available.\n",
        encoding="utf-8",
    )
    return path


def write_factor_figure(
    frame: pd.DataFrame, out: Path, *, preview: bool = False
) -> List[Path]:
    """Write one untitled Det./Int. figure per substantive task grouping.

    Pairwise and one-sided rows are collapsed before aggregation: construction
    is an evaluation regime, not a factor to present as a task breakdown.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
    except ImportError:
        return []

    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(paper_font)
        ).get_name()

    work = frame[
        frame["task"].isin(TASK_REGISTRY)
        & frame["method_group"].isin(HEADLINE_METHODS)
    ].copy()
    work["task family"] = work["task"].map(
        lambda task: TASK_REGISTRY.get(str(task), ("", "", "other", "", 0))[2]
    )
    work["feature classes"] = work["task"].map(
        lambda task: str(TASK_REGISTRY.get(str(task), ("", "", "", "", 0))[4])
    )
    work["task support"] = work["task"].map(
        lambda task: TASK_REGISTRY.get(str(task), ("", "", "", "other", 0))[3]
    )

    # Collapse canonical construction IDs (e.g. cga:two_pole / cga:one_pole)
    # to one paper-facing family.  This is both the legend identity and the bar
    # slot identity, so an inapplicable sibling no longer leaves an empty slot.
    def method_family(row: Mapping[str, object]) -> str:
        backend = str(row.get("backend") or "").strip()
        if backend:
            return "sae" if backend == "sae_pre" else backend
        return str(row.get("method_group") or "").split(":", 1)[0]

    work["plot_family"] = [method_family(row) for row in work.to_dict(orient="records")]

    # Task-family and feature-class panels are intentionally construction-agnostic.
    # Collapse PW/1S siblings within each method × model × task cell first, then
    # aggregate those equally weighted cells by factor.  This prevents two-pole
    # tasks from counting twice merely because both constructions were run, while
    # also avoiding the arbitrary old rule that selected PW for P/1S tasks.
    task_cell_keys = [
        "model", "task", "plot_family", "task family", "feature classes", "task support"
    ]
    task_work = (
        work.groupby(task_cell_keys, as_index=False, dropna=False)[
            ["detection", "detection_neutral", "intervention"]
        ]
        .mean()
    )

    # Construction is compared only on the P/1S task population and only for
    # a method family that supplied both constructions for that same
    # model--task cell.  Both sides therefore have the rival-class metrics
    # required by the ordinary Detection score.
    construction_work = work[
        work["task support"].eq("P/1S")
        & work["pole"].isin(["two_pole", "one_pole"])
    ].copy()
    construction_keys = ["model", "task", "plot_family"]
    matched_constructions = (
        construction_work.groupby(construction_keys)["pole"].nunique().eq(2)
        .rename("_has_both_constructions").reset_index()
    )
    construction_work = construction_work.merge(
        matched_constructions, on=construction_keys, how="left"
    )
    construction_work = construction_work[
        construction_work["_has_both_constructions"].fillna(False)
    ].copy()
    construction_work["construction"] = construction_work["pole"].map(
        {"two_pole": "Pairwise", "one_pole": "One-sided"}
    )

    # Preserve the canonical study-family order while collapsing pole siblings.
    family_order: List[str] = []
    representative: Dict[str, str] = {}
    available_families = set(work["plot_family"].dropna().astype(str))
    for method in headline_method_groups("selected"):
        family = str(method).split(":", 1)[0]
        family = "sae" if family == "sae_pre" else family
        if family not in available_families:
            continue
        representative.setdefault(family, method)
        if family not in family_order:
            family_order.append(family)

    paper_labels = {
        "gradiend": "GradIEND",
        "actiend": "ActIEND",
        "actiend_ridge": "ActIEND-ridge",
        "cga": "CGA",
        "caa": "CAA",
        "caga": "CAGA",
        "agiend": "AGIEND",
        "sae": "SAE",
    }
    labels = {family: paper_labels.get(family, family.upper()) for family in family_order}
    colors = {
        family: method_color(representative.get(family, family)) for family in family_order
    }

    task_levels = sorted(task_work["task family"].dropna().astype(str).unique())
    class_levels = sorted(
        task_work["feature classes"].dropna().astype(str).unique(),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    # One standalone Det./Int. figure per substantive grouping.
    groupings = [
        ("task_family", "task family", task_levels, task_work),
        ("feature_classes", "feature classes", class_levels, task_work),
    ]
    unit = 1.28
    paths: List[Path] = []
    extension = ".png" if preview else ".pdf"
    suffix = "_preview" if preview else ""

    for file_stem, factor, levels, panel_work in groupings:
        fig, axes = plt.subplots(
            1, 2, figsize=(max(8.8, 2 * len(levels) * unit + 2.0), 4.6)
        )
        # The three feature-class groups make the two panels visually denser
        # than the five task-family groups, so give only that figure a wider
        # central gutter.
        panel_wspace = {"feature_classes": 0.18}.get(file_stem, 0.10)
        fig.subplots_adjust(
            left=0.08,
            right=0.99,
            bottom=0.18,
            top=0.84,
            wspace=panel_wspace,
        )
        legend_handles: Dict[str, object] = {}
        present = set(panel_work["plot_family"].dropna().astype(str))
        panel_families = [family for family in family_order if family in present]
        width = 0.8 / max(1, len(panel_families))
        for ax, metric, metric_label in zip(
            axes, ("detection", "intervention"), ("Detection (Det.)", "Intervention (Int.)")
        ):
            finite_values = pd.to_numeric(panel_work[metric], errors="coerce").dropna()
            ymax = 100.0 if metric == "detection" else max(1.0, 100 * float(finite_values.max()))
            for index, family in enumerate(panel_families):
                series = (
                    panel_work[panel_work["plot_family"].eq(family)]
                    .groupby(factor)[metric].mean().reindex(levels) * 100
                )
                values = series.to_numpy(dtype=float)
                positions = np.arange(len(levels)) - 0.4 + width * (index + 0.5)
                bars = ax.bar(positions, values, width=width, color=colors[family])
                if family not in legend_handles and len(bars):
                    legend_handles[family] = bars[0]
                for bar, value in zip(bars, values):
                    if np.isfinite(value):
                        ax.annotate(
                            f"{value:.1f}",
                            (bar.get_x() + bar.get_width() / 2, max(0.35, value - 0.8)),
                            ha="center", va="top", rotation=90, fontsize=6,
                        )
            ax.set_xticks(range(len(levels)), levels, rotation=0, ha="center", fontsize=12)
            ax.set_ylim(0, ymax)
            if metric == "detection":
                ax.set_yticks(range(0, 101, 25))
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylabel(metric_label, fontsize=13)

        legend_families = [family for family in family_order if family in legend_handles]
        fig.legend([legend_handles[family] for family in legend_families],
                   [labels[family] for family in legend_families], loc="upper center",
                   bbox_to_anchor=(0.5, 0.92),
                   ncol=min(7, len(legend_families)), frameon=True)
        path = out / f"factor_breakdown_{file_stem}{suffix}{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        show_plot_if_requested(plt)
        plt.close(fig)
        paths.append(path)

    # Construction is a trade-off between two metrics, not a per-level bar
    # height, so it is a Det./Int. scatter (one pairwise -> one-sided line per
    # method) restricted to the matched P/1S cells built above.
    from analysis.headline_scatter import write_construction_tradeoff_scatter

    tradeoff = write_construction_tradeoff_scatter(
        construction_work, out / f"factor_breakdown_construction{suffix}{extension}"
    )
    if tradeoff is not None:
        paths.append(tradeoff)
    return paths

def write_heatmap(
    frame: pd.DataFrame, model: str | None, dest: Path, *, preview: bool = False
) -> Path | None:
    """Write adjacent headline Det./Int. heatmaps with one shared row axis.

    ``preview`` uses Matplotlib's native renderer and plain task labels, so it
    remains usable when the local XeLaTeX installation is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
    except ImportError:
        return None
    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(paper_font)).get_name()
    _configure_latex_task_labels(plt)
    subset = frame[frame["method_group"].isin(HEADLINE_METHODS)].copy()
    if model is not None:
        subset = subset[subset["model"].eq(model)]
    else:
        subset = subset.groupby(["method_group", "task"], as_index=False)[["detection", "intervention"]].mean()
    present_tasks = set(subset["task"].dropna().astype(str))
    tasks = [task for task in MAIN_STUDY_TASKS if task in present_tasks]
    methods = [method for method in headline_method_groups("selected") if method in set(subset["method_group"].astype(str))]
    fig, axes = plt.subplots(
        1, 2, sharey=True,
        figsize=(PLOT_STYLE["task_width"], PLOT_STYLE["task_height"]),
    )
    fig.subplots_adjust(wspace=PLOT_STYLE["task_panel_wspace"])
    cmap = plt.get_cmap("viridis").copy()
    # Masked = submitted but indeterminate (white); values below vmin = no
    # submitted cell at all (gray).
    cmap.set_bad("white")
    cmap.set_under("#bdbdbd")
    for axis_index, (ax, metric, panel_title) in enumerate(zip(axes, ("detection", "intervention"), ("Detection (Det.)", "Intervention (Int.)"))):
        pivot = subset.pivot_table(index="method_group", columns="task", values=metric, aggfunc="first").reindex(index=methods, columns=tasks)
        values = pivot.to_numpy(dtype=float) * 100
        observed = subset.assign(_record=1).pivot_table(
            index="method_group", columns="task", values="_record", aggfunc="max"
        ).reindex(index=methods, columns=tasks).notna().to_numpy()
        finite = values[np.isfinite(values)]
        vmax = max(1.0, float(finite.max())) if finite.size else 1.0
        # Intervention is signed.  Its scale must expose negative effects
        # while retaining zero as the lower bound when every observed effect
        # is positive. Detection is bounded below by zero by definition.
        vmin = min(0.0, float(finite.min())) if metric == "intervention" and finite.size else 0.0
        display = values.copy()
        # A below-range sentinel preserves the visual distinction: gray means
        # no score can be determined, while white is an observed but
        # indeterminate metric value.
        display[~observed] = vmin - max(1.0, abs(vmin) * 0.1)
        display = np.ma.masked_where(observed & ~np.isfinite(values), display)
        image = ax.imshow(display, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_title(panel_title, fontsize=PLOT_STYLE["task_title_size"])
        if preview:
            ax.set_xticks(
                range(len(tasks)),
                [TASK_REGISTRY.get(task, (task,))[0] for task in tasks],
                rotation=45,
                ha="right",
            )
        else:
            ax.set_xticks(range(len(tasks)), [""] * len(tasks))
        ax.tick_params(axis="x", labelsize=PLOT_STYLE["task_x_tick_size"])
        if not preview:
            _draw_task_labels(ax, tasks)
        if axis_index == 0:
            ax.set_yticks(range(len(methods)), [_plot_method_label(method) for method in methods])
            ax.tick_params(axis="y", labelsize=PLOT_STYLE["task_y_tick_size"])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for y, x in zip(*np.where(np.isfinite(values))):
            ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center", fontsize=PLOT_STYLE["task_cell_size"],
                    color="white" if values[y, x] < vmax * 0.55 else "black")
        for boundary in _method_group_boundaries(methods):
            ax.hlines(boundary, -0.5, len(tasks) - 0.5, colors=PLOT_STYLE["method_separator_color"], linewidth=PLOT_STYLE["method_separator_width"])
        colorbar = fig.colorbar(
            image,
            ax=ax,
            orientation="vertical",
            location="right",
            fraction=PLOT_STYLE["task_cbar_fraction"],
            pad=PLOT_STYLE["task_cbar_pad"],
            shrink=PLOT_STYLE["task_cbar_shrink"],
            aspect=PLOT_STYLE["task_cbar_aspect"],
        )
        colorbar.set_label("Score (%)", fontsize=PLOT_STYLE["task_cbar_label_size"], labelpad=7)
        colorbar.ax.tick_params(labelsize=PLOT_STYLE["task_cbar_tick_size"], pad=3)
        # Matplotlib's automatic labels use left alignment on a right-hand
        # colorbar, which makes the values look ragged when digit counts
        # differ.  Right-aligning them gives both Det. and Int. a common edge.
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_horizontalalignment("right")
            tick_label.set_x(1.8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if preview:
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        show_plot_if_requested(plt)
        plt.close(fig)
        return dest
    # XeLaTeX/PGF expands the Font Awesome 7 macros directly; Matplotlib's
    # ordinary PDF backend uses a legacy DVI font map that lacks FA7 entries.
    try:
        fig.savefig(dest, dpi=200, bbox_inches="tight", backend="pgf")
        show_plot_if_requested(plt)
    except RuntimeError:
        # A Windows Python can have Matplotlib available while resolving
        # ``xelatex`` to an unusable WSL shim.  Do not leave the accompanying
        # LaTeX book stale in that case: retain a previously generated vector
        # figure and continue rebuilding its tables/book.  A missing figure is
        # still a real generation failure.
        if not dest.is_file():
            raise
        print(f"Keeping existing heatmap after PGF backend failure: {dest}")
    finally:
        plt.close(fig)
    return dest


def write_autorank_posterior_maps(
    frame: pd.DataFrame, out: Path, *, preview: bool = False
) -> List[Path]:
    """Peña-style Bayesian Autorank posterior/decision maps for sparse data.

    Autorank itself requires paired observations.  For every method pair we
    therefore retain only its shared model--task cells (rather than imputing a
    missing score or incorrectly ranking unpaired observations).  The maps
    show P(row > column), P(row = column), P(row < column), and the resulting
    Bayesian decision; each cell's paired sample count is written to CSV.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
        from matplotlib.colors import BoundaryNorm
        from autorank import autorank
    except ImportError as exc:
        print(f"Autorank unavailable ({exc}); skipping Autorank posterior maps.", file=sys.stderr)
        return []
    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(paper_font)).get_name()
    artifacts: List[Path] = []
    work = frame[frame["method_group"].isin(HEADLINE_METHODS)].copy()
    for metric, metric_label in (("detection", "Det."), ("intervention", "Int.")):
        autorank_kwargs = bayesian_autorank_kwargs(f"{metric}_score")
        for comparison, pole in (("pairwise", "two_pole"), ("one_sided", "one_pole")):
            subset = work[(work["pole"].eq(pole)) | work["backend"].eq("sae")]
            pivot = subset.pivot_table(
                index=["model", "task"], columns="method_group", values=metric, aggfunc="first"
            )
            methods = [method for method in headline_method_groups("selected") if method in pivot.columns]
            pivot = pivot.reindex(columns=methods)
            n = len(methods)
            gt = np.full((n, n), -1.0)
            eq = np.full((n, n), -1.0)
            lt = np.full((n, n), -1.0)
            np.fill_diagonal(gt, np.nan)
            np.fill_diagonal(eq, np.nan)
            np.fill_diagonal(lt, np.nan)
            decisions = np.full((n, n), -2, dtype=int)
            np.fill_diagonal(decisions, -1)
            counts = np.zeros((n, n), dtype=int)
            records: List[Dict[str, object]] = []
            for i, left in enumerate(methods):
                for j in range(i + 1, n):
                    right = methods[j]
                    paired = pivot[[left, right]].dropna()
                    count = len(paired)
                    counts[i, j] = counts[j, i] = count
                    if count < 5:
                        continue
                    result = autorank(paired, **autorank_kwargs)
                    posterior = result.posterior_matrix
                    posterior_value = posterior.loc[left, right]
                    if isinstance(posterior_value, tuple):
                        p_smaller, p_equal, p_larger = posterior_value
                    else:
                        # Autorank stores the posterior only in one triangle,
                        # ordered by its internal central-tendency ranking.
                        reverse_smaller, p_equal, reverse_larger = posterior.loc[right, left]
                        p_smaller, p_larger = reverse_larger, reverse_smaller
                    # Documentation: p_smaller is P(column < row), i.e. P(left > right).
                    gt[i, j], eq[i, j], lt[i, j] = p_smaller, p_equal, p_larger
                    gt[j, i], eq[j, i], lt[j, i] = p_larger, p_equal, p_smaller
                    # Derive the displayed decision directly from the
                    # displayed posteriors, avoiding Autorank's internally
                    # reordered triangular decision matrix.
                    threshold = 1.0 - autorank_kwargs["alpha"]
                    if p_smaller >= threshold:
                        decision, decision_code = "row > column", 0
                    elif p_equal >= threshold:
                        decision, decision_code = "equivalent", 1
                    elif p_larger >= threshold:
                        decision, decision_code = "row < column", 2
                    else:
                        decision, decision_code = "inconclusive", 3
                    decisions[i, j] = decision_code
                    decisions[j, i] = {0: 2, 1: 1, 2: 0, 3: 3}[decision_code]
                    records.append({
                        "metric": metric_label, "comparison": comparison, "row_method": left,
                        "column_method": right, "n_paired_cells": count,
                        "p_row_greater": p_smaller, "p_equal": p_equal,
                        "p_row_smaller": p_larger, "decision": decision,
                    })
            csv_path = out / f"autorank_{metric}_{comparison}_pairwise.csv"
            pd.DataFrame(records).to_csv(csv_path, index=False, float_format="%.6f")
            artifacts.append(csv_path)
            # Sparse pairwise data cannot support Autorank's single global
            # rankdf (there is no complete all-method row). Order each map by
            # its Bayesian posterior win score instead, so Det. and Int. get
            # their own meaningful order rather than a fixed method list.
            strength = np.nanmean(
                np.where(gt >= 0, gt + 0.5 * np.maximum(eq, 0), np.nan), axis=1
            )
            order = np.argsort(-np.nan_to_num(strength, nan=-np.inf))
            gt, eq, lt = gt[np.ix_(order, order)], eq[np.ix_(order, order)], lt[np.ix_(order, order)]
            decisions = decisions[np.ix_(order, order)]
            methods = [methods[index] for index in order]
            n = len(methods)
            fig, axes = plt.subplots(1, 4, figsize=(max(PLOT_STYLE["autorank_min_width"], n * PLOT_STYLE["autorank_width_per_method"]), max(PLOT_STYLE["autorank_min_height"], n * PLOT_STYLE["autorank_height_per_method"])), constrained_layout=True)
            labels = [_plot_method_label(method) for method in methods]
            for ax, values, title in zip(
                axes[:3], (gt, eq, lt),
                (rf"{metric_label}: P(row $>$ column)", rf"{metric_label}: P(row $=$ column)", rf"{metric_label}: P(row $<$ column)"),
            ):
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("white")
                cmap.set_under("#bdbdbd")
                image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap)
                ax.set_title(title, fontsize=PLOT_STYLE["autorank_title_size"])
                ax.set_xticks(range(n), labels, rotation=45, ha="right")
                ax.set_yticks(range(n), labels)
                ax.tick_params(axis="both", labelsize=PLOT_STYLE["autorank_tick_size"])
                for y, x in zip(*np.where((values >= 0) & np.isfinite(values))):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"], color="white" if values[y, x] < 0.55 else "black")
                colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", location="bottom", pad=0.17, shrink=0.9)
                colorbar.ax.tick_params(labelsize=9)
            decision_cmap = plt.get_cmap("Set2", 4).copy()
            decision_cmap.set_bad("white")
            decision_cmap.set_under("#bdbdbd")
            decision_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), decision_cmap.N)
            image = axes[3].imshow(np.ma.masked_where(decisions == -1, decisions), cmap=decision_cmap, norm=decision_norm)
            axes[3].set_title(f"{metric_label}: Bayesian decision", fontsize=PLOT_STYLE["autorank_title_size"])
            axes[3].set_xticks(range(n), labels, rotation=45, ha="right")
            axes[3].set_yticks(range(n), labels)
            axes[3].tick_params(axis="both", labelsize=PLOT_STYLE["autorank_tick_size"])
            decision_labels = {0: ">", 1: "=", 2: "<", 3: "?"}
            for y, x in zip(*np.where(decisions >= 0)):
                axes[3].text(x, y, decision_labels[int(decisions[y, x])], ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"] + 2, color="black")
            colorbar = fig.colorbar(image, ax=axes[3], ticks=range(4), orientation="horizontal", location="bottom", pad=0.17, shrink=0.9)
            colorbar.ax.set_xticklabels(["row > column", "equivalent", "row < column", "inconclusive"])
            colorbar.ax.tick_params(labelsize=9)
            # The prior PDF can remain open in a desktop viewer on Windows;
            # write the refined artifact to a fresh, deterministic filename.
            suffix = "_preview.png" if preview else ".pdf"
            figure = out / f"autorank_{metric}_{comparison}_posterior_maps_refined{suffix}"
            fig.savefig(figure, dpi=220, bbox_inches="tight")
            show_plot_if_requested(plt)
            plt.close(fig)
            artifacts.append(figure)

            # A compact companion keeps the same four views but makes the
            # posterior triplet share one scale and one colorbar.  The
            # categorical decision view remains self-explanatory through its
            # in-cell symbols, so it does not need a competing second bar.
            figure_2x2, axes_2x2 = plt.subplots(
                2,
                2,
                figsize=(
                    max(PLOT_STYLE["autorank_min_width"] * 0.72, n * PLOT_STYLE["autorank_width_per_method"] * 1.45),
                    max(PLOT_STYLE["autorank_min_height"] * 1.45, n * PLOT_STYLE["autorank_height_per_method"] * 1.55),
                ),
            )
            figure_2x2.subplots_adjust(
                wspace=PLOT_STYLE["autorank_2x2_wspace"],
                hspace=PLOT_STYLE["autorank_2x2_hspace"],
            )
            probability_axes = list(axes_2x2.flat[:3])
            probability_image = None
            for ax, values, title in zip(
                probability_axes,
                (gt, eq, lt),
                (rf"{metric_label}: P(row $>$ column)", rf"{metric_label}: P(row $=$ column)", rf"{metric_label}: P(row $<$ column)"),
            ):
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("white")
                cmap.set_under("#bdbdbd")
                probability_image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap)
                ax.set_title(title, fontsize=PLOT_STYLE["autorank_title_size"])
                ax.set_xticks(range(n), labels, rotation=45, ha="right")
                ax.set_yticks(range(n), labels)
                ax.tick_params(axis="both", labelsize=PLOT_STYLE["autorank_tick_size"])
                for y, x in zip(*np.where((values >= 0) & np.isfinite(values))):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"], color="white" if values[y, x] < 0.55 else "black")
            decision_ax = axes_2x2.flat[3]
            decision_cmap = plt.get_cmap("Set2", 4).copy()
            decision_cmap.set_bad("white")
            decision_cmap.set_under("#bdbdbd")
            decision_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), decision_cmap.N)
            decision_ax.imshow(np.ma.masked_where(decisions == -1, decisions), cmap=decision_cmap, norm=decision_norm)
            decision_ax.set_title(f"{metric_label}: Bayesian decision", fontsize=PLOT_STYLE["autorank_title_size"])
            decision_ax.set_xticks(range(n), labels, rotation=45, ha="right")
            decision_ax.set_yticks(range(n), labels)
            decision_ax.tick_params(axis="both", labelsize=PLOT_STYLE["autorank_tick_size"])
            for y, x in zip(*np.where(decisions >= 0)):
                decision_ax.text(x, y, decision_labels[int(decisions[y, x])], ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"] + 2, color="black")
            assert probability_image is not None
            shared_colorbar = figure_2x2.colorbar(
                probability_image,
                ax=probability_axes,
                orientation="vertical",
                location="right",
                pad=0.035,
                shrink=0.94,
            )
            shared_colorbar.set_label("Posterior probability", fontsize=10)
            shared_colorbar.ax.tick_params(labelsize=15, rotation=90)
            figure_2x2_path = out / f"autorank_{metric}_{comparison}_posterior_maps_refined_2x2{suffix}"
            figure_2x2.savefig(figure_2x2_path, dpi=220, bbox_inches="tight")
            show_plot_if_requested(plt)
            plt.close(figure_2x2)
            artifacts.append(figure_2x2_path)
    return artifacts


def write_autorank_overview(
    out: Path, kind: str, *, preview: bool = False
) -> Path | None:
    """One-row overview of one posterior/decision view across four strata."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
        from matplotlib.colors import BoundaryNorm
    except ImportError:
        return None
    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(paper_font)).get_name()
    if kind not in {"greater", "equal", "less", "decision"}:
        raise ValueError(f"Unknown Autorank overview kind: {kind}")
    panels = []
    for metric, metric_label in (("detection", "Detection"), ("intervention", "Intervention")):
        for comparison, comparison_label in (("pairwise", "Pairwise"), ("one_sided", "One-sided")):
            path = out / f"autorank_{metric}_{comparison}_pairwise.csv"
            if not path.is_file():
                return None
            records = pd.read_csv(path)
            methods = [method for method in headline_method_groups("selected") if method in set(records["row_method"]) | set(records["column_method"])]
            n = len(methods)
            greater = np.full((n, n), -1.0)
            equal = np.full((n, n), -1.0)
            less = np.full((n, n), -1.0)
            decisions = np.full((n, n), -2, dtype=int)
            for values in (greater, equal, less):
                np.fill_diagonal(values, np.nan)
            np.fill_diagonal(decisions, -1)
            index = {method: position for position, method in enumerate(methods)}
            for row in records.to_dict(orient="records"):
                left, right = index[row["row_method"]], index[row["column_method"]]
                greater[left, right], greater[right, left] = float(row["p_row_greater"]), float(row["p_row_smaller"])
                equal[left, right] = equal[right, left] = float(row["p_equal"])
                less[left, right], less[right, left] = float(row["p_row_smaller"]), float(row["p_row_greater"])
                decision = str(row["decision"])
                code = {"row > column": 0, "equivalent": 1, "row < column": 2, "inconclusive": 3}[decision]
                decisions[left, right] = code
                decisions[right, left] = {0: 2, 1: 1, 2: 0, 3: 3}[code]
            strength = np.nanmean(np.where(greater >= 0, greater, np.nan), axis=1)
            order = np.argsort(-np.nan_to_num(strength, nan=-np.inf))
            selected = {"greater": greater, "equal": equal, "less": less, "decision": decisions}[kind]
            panels.append((f"{metric_label}: {comparison_label}", selected[np.ix_(order, order)], [methods[item] for item in order]))
    fig, axes = plt.subplots(1, 4, figsize=(PLOT_STYLE["overview_width"], PLOT_STYLE["overview_height"]), constrained_layout=True)
    cmap = (plt.get_cmap("Set2", 4).copy() if kind == "decision" else plt.get_cmap("viridis").copy())
    cmap.set_bad("white")
    cmap.set_under("#bdbdbd")
    decision_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N) if kind == "decision" else None
    image = None
    for ax, (title, values, methods) in zip(axes, panels):
        if kind == "decision":
            image = ax.imshow(np.ma.masked_where(values == -1, values), cmap=cmap, norm=decision_norm)
        else:
            image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap)
        labels = [_plot_method_label(method) for method in methods]
        ax.set_title(title, fontsize=PLOT_STYLE["autorank_title_size"])
        ax.set_xticks(range(len(methods)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(methods)), labels)
        ax.tick_params(axis="both", labelsize=PLOT_STYLE["overview_tick_size"])
        if kind == "decision":
            decision_labels = {0: ">", 1: "=", 2: "<", 3: "?"}
            for y, x in zip(*np.where(values >= 0)):
                ax.text(x, y, decision_labels[int(values[y, x])], ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"] + 2)
        else:
            for y, x in zip(*np.where((values >= 0) & np.isfinite(values))):
                ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"], color="white" if values[y, x] < 0.55 else "black")
    assert image is not None
    colorbar = fig.colorbar(image, ax=list(axes), orientation="vertical", location="right", pad=0.02, shrink=0.88)
    if kind == "decision":
        colorbar.set_ticks(range(4), labels=["row > column", "equivalent", "row < column", "inconclusive"])
    else:
        probability_label = {"greater": r"$P(\mathrm{row}>\mathrm{column})$", "equal": r"$P(\mathrm{row}=\mathrm{column})$", "less": r"$P(\mathrm{row}<\mathrm{column})$"}[kind]
        colorbar.set_label(probability_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    suffix = "_preview.png" if preview else ".pdf"
    path = out / f"autorank_posterior_{kind}_overview{suffix}"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    show_plot_if_requested(plt)
    plt.close(fig)

    # Preserve the landscape overview and add a more page-friendly grid. All
    # four panels have the same value range, so one colorbar is sufficient.
    figure_2x2, axes_2x2 = plt.subplots(
        2, 2,
        figsize=(PLOT_STYLE["overview_width"] * 0.72, PLOT_STYLE["overview_height"] * 1.5),
    )
    figure_2x2.subplots_adjust(
        wspace=PLOT_STYLE["autorank_2x2_wspace"],
        hspace=PLOT_STYLE["autorank_2x2_hspace"],
    )
    shared_image = None
    for ax, (title, values, methods) in zip(axes_2x2.flat, panels):
        if kind == "decision":
            shared_image = ax.imshow(np.ma.masked_where(values == -1, values), cmap=cmap, norm=decision_norm)
        else:
            shared_image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap)
        labels = [_plot_method_label(method) for method in methods]
        ax.set_title(title, fontsize=PLOT_STYLE["autorank_title_size"])
        ax.set_xticks(range(len(methods)), labels, rotation=45, ha="right", fontsize=20)
        ax.set_yticks(range(len(methods)), labels)
        ax.tick_params(axis="both", labelsize=PLOT_STYLE["overview_tick_size"])
        if kind == "decision":
            decision_labels = {0: ">", 1: "=", 2: "<", 3: "?"}
            for y, x in zip(*np.where(values >= 0)):
                ax.text(x, y, decision_labels[int(values[y, x])], ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"] + 2)
        else:
            for y, x in zip(*np.where((values >= 0) & np.isfinite(values))):
                ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"], color="white" if values[y, x] < 0.55 else "black")
    assert shared_image is not None
    shared_colorbar = figure_2x2.colorbar(
        shared_image,
        ax=list(axes_2x2.flat),
        orientation="vertical",
        location="right",
        pad=0.025,
        shrink=0.94,
    )
    if kind == "decision":
        shared_colorbar.set_ticks(range(4))
        shared_colorbar.set_ticklabels(["row > column", "equivalent", "row < column", "inconclusive"])
        for label in shared_colorbar.ax.get_yticklabels():
            label.set_rotation(90)
            label.set_fontsize(PLOT_STYLE["autorank_decision_2x2_cbar_label_size"])
            label.set_verticalalignment("center")
            label.set_horizontalalignment("left")
    else:
        shared_colorbar.set_label(probability_label, fontsize=10)
        shared_colorbar.ax.tick_params(labelsize=9)
    path_2x2 = out / f"autorank_posterior_{kind}_overview_2x2{suffix}"
    figure_2x2.savefig(path_2x2, dpi=220, bbox_inches="tight")
    show_plot_if_requested(plt)
    plt.close(figure_2x2)
    return path


def write_autorank_overview_4x4(out: Path, *, preview: bool = False) -> Path | None:
    """Write all four Autorank overview views in one 4 x 4 grid.

    Each column uses its own posterior-win ordering, shared by its greater,
    equal, less, and decision rows. Probability rows share one colorbar; the
    categorical decision row receives its own centered categorical colorbar.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
        from matplotlib.colors import BoundaryNorm
    except ImportError:
        return None
    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(paper_font)).get_name()

    column_data = []
    for metric, metric_label in (("detection", "Detection"), ("intervention", "Intervention")):
        for comparison, comparison_label in (("pairwise", "Pairwise"), ("one_sided", "One-sided")):
            path = out / f"autorank_{metric}_{comparison}_pairwise.csv"
            if not path.is_file():
                return None
            records = pd.read_csv(path)
            methods = [method for method in headline_method_groups("selected") if method in set(records["row_method"]) | set(records["column_method"])]
            n = len(methods)
            greater = np.full((n, n), -1.0)
            equal = np.full((n, n), -1.0)
            less = np.full((n, n), -1.0)
            decisions = np.full((n, n), -2, dtype=int)
            for values in (greater, equal, less):
                np.fill_diagonal(values, np.nan)
            np.fill_diagonal(decisions, -1)
            index = {method: position for position, method in enumerate(methods)}
            for row in records.to_dict(orient="records"):
                left, right = index[row["row_method"]], index[row["column_method"]]
                greater[left, right], greater[right, left] = float(row["p_row_greater"]), float(row["p_row_smaller"])
                equal[left, right] = equal[right, left] = float(row["p_equal"])
                less[left, right], less[right, left] = float(row["p_row_smaller"]), float(row["p_row_greater"])
                code = {"row > column": 0, "equivalent": 1, "row < column": 2, "inconclusive": 3}[str(row["decision"])]
                decisions[left, right] = code
                decisions[right, left] = {0: 2, 1: 1, 2: 0, 3: 3}[code]
            order = np.argsort(-np.nan_to_num(np.nanmean(np.where(greater >= 0, greater, np.nan), axis=1), nan=-np.inf))
            column_data.append((
                f"{metric_label}: {comparison_label}",
                {kind: values[np.ix_(order, order)] for kind, values in {
                    "greater": greater, "equal": equal, "less": less, "decision": decisions,
                }.items()},
                [methods[item] for item in order],
            ))

    kinds = ("greater", "equal", "less", "decision")
    row_labels = {
        "greater": r"$P(\mathrm{row}>\mathrm{column})$",
        "equal": r"$P(\mathrm{row}=\mathrm{column})$",
        "less": r"$P(\mathrm{row}<\mathrm{column})$",
        "decision": "Bayesian decision",
    }
    probability_cmap = plt.get_cmap("viridis").copy()
    decision_cmap = plt.get_cmap("Set2", 4).copy()
    for cmap in (probability_cmap, decision_cmap):
        cmap.set_bad("white")
        cmap.set_under("#bdbdbd")
    decision_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), decision_cmap.N)
    fig, axes = plt.subplots(
        4, 4,
        figsize=(PLOT_STYLE["overview_width"] * 1.34, PLOT_STYLE["overview_height"] * 2.18),
        sharex="col",
        gridspec_kw={"hspace": 0.06, "wspace": 0.45},
    )
    row_images = [None] * len(kinds)
    decision_labels = {0: ">", 1: "=", 2: "<", 3: "?"}
    for row_index, kind in enumerate(kinds):
        for column_index, (title, values_by_kind, methods) in enumerate(column_data):
            ax = axes[row_index, column_index]
            values = values_by_kind[kind]
            if kind == "decision":
                row_images[row_index] = ax.imshow(np.ma.masked_where(values == -1, values), cmap=decision_cmap, norm=decision_norm)
            else:
                row_images[row_index] = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=probability_cmap)
            labels = [_plot_method_label(method) for method in methods]
            if row_index == 0:
                ax.set_title(title, fontsize=PLOT_STYLE["autorank_title_size"])
            if column_index == 0:
                ax.set_ylabel(row_labels[kind], fontsize=PLOT_STYLE["autorank_grid_row_label_size"], labelpad=12)
            ax.set_xticks(range(len(methods)), labels, rotation=45, ha="right")
            ax.set_yticks(range(len(methods)), labels)
            ax.tick_params(axis="both", labelsize=PLOT_STYLE["overview_tick_size"])
            if row_index < len(kinds) - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if kind == "decision":
                for y, x in zip(*np.where(values >= 0)):
                    ax.text(x, y, decision_labels[int(values[y, x])], ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"] + 2)
            else:
                for y, x in zip(*np.where((values >= 0) & np.isfinite(values))):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=PLOT_STYLE["autorank_annotation_size"], color="white" if values[y, x] < 0.55 else "black")
    assert all(image is not None for image in row_images)
    for row_index, kind in enumerate(kinds):
        colorbar = fig.colorbar(
            row_images[row_index],
            ax=list(axes[row_index, :]),
            ticks=range(4) if kind == "decision" else None,
            orientation="vertical",
            location="right",
            pad=0.02,
            shrink=0.94,
        )
        if kind == "decision":
            colorbar.set_ticklabels(["row > column", "equivalent", "row < column", "inconclusive"])
        else:
            colorbar.set_label(row_labels[kind], fontsize=PLOT_STYLE["autorank_grid_cbar_label_size"], labelpad=10)
        colorbar.ax.tick_params(labelsize=PLOT_STYLE["autorank_grid_cbar_tick_size"])
    suffix = "_preview.png" if preview else ".pdf"
    path = out / f"autorank_posterior_overview_4x4{suffix}"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_autorank_latex_tables(out: Path, models: Sequence[str]) -> List[Path]:
    """Write one Bayesian-comparison table per metric from the map CSVs."""
    model_caption = ", ".join(MODEL_LATEX.get(model, _latex_escape(model)) for model in models)
    artifacts: List[Path] = []
    for metric, metric_label in (("detection", "Detection"), ("intervention", "Intervention")):
        frames: List[pd.DataFrame] = []
        for comparison, comparison_label in (("pairwise", "Pairwise"), ("one_sided", "One-sided")):
            path = out / f"autorank_{metric}_{comparison}_pairwise.csv"
            if not path.is_file():
                continue
            table = pd.read_csv(path)
            if table.empty:
                continue
            table["comparison_label"] = comparison_label
            frames.append(table)
        if not frames:
            continue
        rows = pd.concat(frames, ignore_index=True)
        lines = [
            r"\begin{longtable}{l l l r r r r l}",
            rf"\caption{{Bayesian Autorank {metric_label} comparisons of selected headline methods. Every pair uses its shared model--task cells only. Models: {model_caption}.}}\\",
            r"\toprule",
            r"Construction & Row & Column & $n$ & $P(\mathrm{row}>\mathrm{column})$ & $P(=)$ & $P(<)$ & Decision \\",
            r"\midrule", r"\endfirsthead",
            r"\toprule",
            r"Construction & Row & Column & $n$ & $P(\mathrm{row}>\mathrm{column})$ & $P(=)$ & $P(<)$ & Decision \\",
            r"\midrule", r"\endhead",
        ]
        for row in rows.to_dict(orient="records"):
            label = lambda value: HEADLINE_METHOD_LATEX.get(str(value), _latex_escape(value))
            lines.append(
                f"{row['comparison_label']} & {label(row['row_method'])} & {label(row['column_method'])} & "
                f"{int(row['n_paired_cells'])} & {float(row['p_row_greater']):.3f} & "
                f"{float(row['p_equal']):.3f} & {float(row['p_row_smaller']):.3f} & "
                f"{_latex_escape(row['decision'])} \\\\"
            )
        lines += [r"\bottomrule", r"\end{longtable}", ""]
        path = out / f"autorank_{metric}_comparisons.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        artifacts.append(path)
    return artifacts


def write_book(out: Path, models: Iterable[str]) -> Path:
    models = list(models)
    model_caption = ", ".join(MODEL_LATEX.get(model, _latex_escape(model)) for model in models)
    chunks = [
        r"\documentclass[10pt]{article}", r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{booktabs}", r"\usepackage{longtable}", r"\usepackage{graphicx}",
        r"\providecommand{\MethodSAE}{SAE}",
        # The paper preamble supplies Font Awesome 7's \faIcon. The fallback
        # keeps this standalone diagnostic book compilable without that package.
        r"\providecommand{\faIcon}[1]{}", *TASK_COMMAND_DEFINITIONS,
        r"\pagestyle{empty}", r"\begin{document}",
    ]
    task_table = out / "task_summary.tex"
    if task_table.is_file():
        chunks.append(task_table.read_text(encoding="utf-8"))
        chunks.append(r"\clearpage")
    pooled = out / "headline_method_task_det_int.tex"
    if pooled.is_file():
        chunks.append(pooled.read_text(encoding="utf-8"))
        chunks.append(r"\clearpage")
    pooled_figure = out / "headline_method_task_det_int_heatmap_latex.pdf"
    if pooled_figure.is_file():
        chunks += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=.96\textwidth,height=.88\textheight,keepaspectratio]{{\detokenize{{{pooled_figure.name}}}}}", rf"\caption{{Per-cell means across available models: {model_caption}.}}", r"\end{figure}", r"\clearpage"]
    for kind, caption_label in (("greater", r"$P(\mathrm{row}>\mathrm{column})$"), ("equal", r"$P(\mathrm{row}=\mathrm{column})$"), ("less", r"$P(\mathrm{row}<\mathrm{column})$"), ("decision", "Bayesian decision")):
        overview = out / f"autorank_posterior_{kind}_overview.pdf"
        if overview.is_file():
            chunks += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=.98\textwidth,height=.88\textheight,keepaspectratio]{{\detokenize{{{overview.name}}}}}", rf"\caption{{Bayesian Autorank overview: {caption_label} for each metric and construction. Each panel has its own posterior-win ordering; pairwise and one-sided constructions are not compared directly. Models: {model_caption}.}}", r"\end{figure}", r"\clearpage"]
    for metric in ("detection", "intervention"):
        table = out / f"autorank_{metric}_comparisons.tex"
        if table.is_file():
            chunks += [table.read_text(encoding="utf-8"), r"\clearpage"]
    for metric in ("detection", "intervention"):
        for comparison, comparison_label in (("pairwise", "pairwise"), ("one_sided", "one-sided")):
            figure = out / f"autorank_{metric}_{comparison}_posterior_maps_refined.pdf"
            if figure.is_file():
                chunks += [
                    r"\begin{figure}[p]\centering",
                    rf"\includegraphics[width=.98\textwidth,height=.88\textheight,keepaspectratio]{{\detokenize{{{figure.name}}}}}",
                    rf"\caption{{Bayesian Autorank posterior probabilities and decisions for {metric} ({comparison_label} methods; SAE included as a shared reference). Every pair uses its shared model--task cells only. Models: {model_caption}.}}",
                    r"\end{figure}", r"\clearpage",
                ]
    for model in models:
        stem = f"across_model_det_int_{model}"
        chunks.append((out / f"{stem}.tex").read_text(encoding="utf-8"))
        figure = out / f"{stem}_heatmap.pdf"
        if figure.is_file():
            single_caption = MODEL_LATEX.get(model, _latex_escape(model))
            chunks += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=.96\textwidth,height=.88\textheight,keepaspectratio]{{\detokenize{{{figure.name}}}}}", rf"\caption{{Model: {single_caption}.}}", r"\end{figure}", r"\clearpage"]
    chunks.append(r"\end{document}")
    path = out / "summary_model_task_tables.tex"
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return path


def parse_source(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--source must be MODEL=PATH_TO_summary_merged.csv")
    model, path = raw.split("=", 1)
    if not model or not path:
        raise ValueError("--source must be MODEL=PATH_TO_summary_merged.csv")
    return model, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="MODEL=CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fig",
        "--figure-only",
        dest="figure_only",
        action="store_true",
        help="Regenerate figures only (Det./Int. and Autorank heatmaps; no tables or book).",
    )
    parser.add_argument(
        "--factor-figures",
        action="store_true",
        help="Regenerate only the standalone factor-breakdown Det./Int. figures.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="With --fig or --factor-figures, write TeX-free PNG previews for rapid figure tuning.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open generated figures for interactive inspection.",
    )
    parser.add_argument(
        "--include-hardware", action="store_true",
        help="Read raw results.json payloads to write the hardware/VRAM inventory (can be slow for large artifacts).",
    )
    args = parser.parse_args()
    if args.show:
        os.environ["GRADIEND_SHOW_PLOTS"] = "1"
    if args.preview and not (args.figure_only or args.factor_figures):
        parser.error("--preview requires --fig or --factor-figures")
    sources = [parse_source(item) for item in args.source]
    missing = [str(path) for _model, path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing generated summary CSV(s): " + ", ".join(missing))
    frame = read_sources(sources)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.factor_figures:
        for factor_figure in write_factor_figure(frame, args.out, preview=args.preview):
            print(f"Wrote {factor_figure}")
        return
    if args.figure_only:
        for factor_figure in write_factor_figure(frame, args.out, preview=args.preview):
            print(f"Wrote {factor_figure}")
        # Autorank writes its pairwise probability/decision maps first; the
        # overview panels consume those map CSVs. Preview mode mirrors these
        # figures as PNGs so no TeX installation is needed for tuning.
        for artifact in write_autorank_posterior_maps(frame, args.out, preview=args.preview):
            print(f"Wrote {artifact}")
        for kind in ("greater", "equal", "less", "decision"):
            overview = write_autorank_overview(args.out, kind, preview=args.preview)
            if overview:
                print(f"Wrote {overview}")
        overview_4x4 = write_autorank_overview_4x4(args.out, preview=args.preview)
        if overview_4x4:
            print(f"Wrote {overview_4x4}")
        extension = ".png" if args.preview else ".pdf"
        suffix = "_preview" if args.preview else ""
        pooled_heatmap = write_heatmap(
            frame,
            None,
            args.out / f"headline_method_task_det_int_heatmap_latex{suffix}{extension}",
            preview=args.preview,
        )
        if pooled_heatmap:
            print(f"Wrote {pooled_heatmap}")
        for model, _path in sources:
            stem = args.out / f"across_model_det_int_{model}"
            heatmap = write_heatmap(
                frame,
                model,
                stem.with_name(stem.name + f"_heatmap{suffix}{extension}"),
                preview=args.preview,
            )
            if heatmap:
                print(f"Wrote {heatmap}")
        return
    for path in write_task_and_factor_tables(frame, args.out):
        print(f"Wrote {path}")
    if args.include_hardware:
        hardware = write_compute_hardware_inventory(frame, args.out)
        print(f"Wrote {hardware}")
    for factor_figure in write_factor_figure(frame, args.out):
        print(f"Wrote {factor_figure}")
    pooled_matrix = headline_pooled_detail_matrix(frame)
    pooled_csv = args.out / "headline_method_task_det_int.csv"
    pooled_tex = args.out / "headline_method_task_det_int.tex"
    pooled_matrix.to_csv(pooled_csv)
    pooled_tex.write_text(
        _matrix_latex(
            pooled_matrix,
            "Headline methods: mean across available base models",
            headline=True,
            models=[model for model, _path in sources],
        ),
        encoding="utf-8",
    )
    print(f"Wrote {pooled_csv}")
    print(f"Wrote {pooled_tex}")
    pooled_heatmap = write_heatmap(
        frame, None, args.out / "headline_method_task_det_int_heatmap_latex.pdf"
    )
    if pooled_heatmap:
        print(f"Wrote {pooled_heatmap}")
    for artifact in write_autorank_posterior_maps(frame, args.out):
        print(f"Wrote {artifact}")
    for kind in ("greater", "equal", "less", "decision"):
        overview = write_autorank_overview(args.out, kind)
        if overview:
            print(f"Wrote {overview}")
    overview_4x4 = write_autorank_overview_4x4(args.out)
    if overview_4x4:
        print(f"Wrote {overview_4x4}")
    for artifact in write_autorank_latex_tables(args.out, [model for model, _path in sources]):
        print(f"Wrote {artifact}")
    for model, _path in sources:
        matrix = detail_matrix(frame, model)
        stem_name = f"across_model_det_int_{model}"
        stem = args.out / stem_name
        csv_path = args.out / f"{stem_name}.csv"
        tex_path = args.out / f"{stem_name}.tex"
        matrix.to_csv(csv_path)
        tex_path.write_text(_matrix_latex(matrix, model, headline=True, models=[model]), encoding="utf-8")
        heatmap = write_heatmap(frame, model, stem.with_name(stem.name + "_heatmap.pdf"))
        print(f"Wrote {csv_path}")
        if heatmap:
            print(f"Wrote {heatmap}")
    book = write_book(args.out, [model for model, _path in sources])
    print(f"Wrote {book}")


if __name__ == "__main__":
    main()
