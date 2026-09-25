#!/usr/bin/env python
"""Performance-vs-layer figures for layer-resolved method families.

These are the only method families with a native per-layer sweep in
``results.json`` (``sae:{cls}:L{n}``, ``sae_pre:{cls}:L{n}``,
``caa:{cls}:L{n}_act_prediction``) — GRADIEND/ACTIEND train a single fixed
activation site, so they have no layer axis to plot.

One compact figure per model: a 5-column × 6-row grid for the 15 paper tasks,
with held-out Detection above held-out Intervention for each task and one
shared legend. Per-class rows are averaged within (task, family, layer).
Causal metrics are essentially CAA-only in the layer sweep because per-layer
SAE/SAE_pre causal was never computed; an absent series is therefore shown as
absent, not imputed.

Usage:
  python analysis/layer_performance_plots.py --model gpt2-small
  python analysis/layer_performance_plots.py --model gpt2-small --subdir suite_full
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.method_groups import _load_results  # noqa: E402
from analysis.plot_style import BACKEND_COLORS, scatter_style  # noqa: E402
from analysis.summary_latex import (  # noqa: E402
    METHOD_LATEX,
    PAPER_TASK_ORDER,
    TASK_COMMAND_DEFINITIONS,
    TASK_LATEX,
    _is_hidden_task,
    collect_canonical_group_rows,
    task_label_rendered,
)
from suitability import detection_score, intervention_score  # noqa: E402

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "analysis" / "figures"

FAMILY_COLOR = dict(BACKEND_COLORS)
# These methods have no layer sweep.  They remain useful horizontal reference
# lines, but are *not* a baseline: each is a full-model method estimate.
BASELINE_ONLY_FAMILIES: Tuple[str, ...] = ("agiend", "gradiend", "actiend")
# Match the study/table order. ``sae_pre`` is an appendix-only alternate site,
# not a method in the layerwise paper comparison.
FAMILY_ORDER: Tuple[str, ...] = (
    "gradiend", "actiend", "cga", "caga", "agiend", "sae", "caa"
)

# Visual grammar: method identity is carried by color/marker, while construction
# and reference scope are carried by line style.
LAYER_LINESTYLE = {"two_pole": "-", "one_pole": ":"}
# All-layer estimates are horizontal, so they do not need a separate dash
# grammar from physical-layer sweeps. Keep the construction grammar identical:
# pairwise solid, one-sided dotted.
ALL_LAYER_LINESTYLE = {"two_pole": "-", "one_pole": ":"}
ONE_POLE_MARKER_EDGEWIDTH = 0.55

# Opt-in by-tensor ablation (full_plus suite, see CLAUDE.md's "full_plus" /
# "which command adds ACTIEND tensor" notes) — GradiendSplit.by_tensor()
# trains a *separate* checkpoint from the headline fixed-site run, so its
# ``{backend}:{pair}:tensors:{cls}`` rows are additional to, never a
# replacement for, the BASELINE_ONLY_FAMILIES flat line above. Still just one
# aggregate number per task (by_tensor splits by weight tensor, not
# residual-stream layer — no layer axis here either), so it's drawn the same
# way: a flat line, same color as its parent family (no new palette slot),
# dotted instead of dashed so it reads as "the by-tensor sibling of the line
# above" rather than a fifth independent series.
TENSORS_FAMILIES: Tuple[str, ...] = ("gradiend", "actiend")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"

# (metric field, display label) — mirrors summary_latex.py's MATRIX_METRICS/METRIC_LATEX.
PAPER_METRICS: Tuple[Tuple[str, str], ...] = (
    ("detection_score", "Det."),
    ("intervention_score", "Int."),
)

_LAYER_RE = {
    "sae": re.compile(r"^sae:([^:]+):L(\d+)$"),
    "sae_pre": re.compile(r"^sae_pre:([^:]+):L(\d+)$"),
    "caa": re.compile(r"^caa:(?:[^:]+:)?([^:]+):L(\d+)_act_prediction$"),
    "cga": re.compile(r"^cga:(?:[^:]+:)?([^:]+):L(\d+)$"),
    "caga": re.compile(r"^caga:(?:[^:]+:)?([^:]+):L(\d+)$"),
}

# "ALL layers" baselines: sae/sae_pre pool top-k features across every layer
# jointly (all_k1) instead of picking one; caa's all_act_prediction pools all
# per-layer directions. Genuinely different numbers from any single layer —
# confirmed against real data (e.g. gender_en sae:F:all_k1 roc_auc_neutral=
# 1.000 vs sae:F:k1/kstar=0.9999), not a duplicate of k1/kstar.
_ALL_RE = {
    "sae": re.compile(r"^sae:([^:]+):all_k1$"),
    "sae_pre": re.compile(r"^sae_pre:([^:]+):all_k1$"),
    "caa": re.compile(r"^caa:(?:[^:]+:)?([^:]+):all_act_prediction$"),
    "cga": re.compile(r"^cga:(?:[^:]+:)?([^:]+)$"),
    "caga": re.compile(r"^caga:(?:[^:]+:)?([^:]+)$"),
}


def _pole_for_method(family: str, method_id: str) -> str:
    """Return the table construction label for a raw method id."""
    if family in {"sae", "sae_pre"}:
        return "one_pole"
    parts = method_id.split(":")
    return "two_pole" if len(parts) > 1 and "-" in parts[1] else "one_pole"


def _family_label(family: str, pole: str) -> str:
    """Use the exact paper-table method names where MathText can render them."""
    if family == "sae":
        return "SAE"
    if family == "sae_pre":
        return r"SAE$_{\mathrm{pre}}$"
    return METHOD_LATEX.get(f"{family}:{pole}", family.upper())


def _layer_style(family: str, pole: str) -> Dict[str, object]:
    """Reuse the headline marker style, with one-sided markers hollow and fine-edged."""
    # SAE's method id carries its recipe rather than a construction suffix.
    method = f"{family}:k1" if family in {"sae", "sae_pre"} else f"{family}:{pole}"
    style = dict(scatter_style(method, pole))
    if pole == "one_pole":
        style["facecolors"] = "none"
        style["edgecolors"] = FAMILY_COLOR[family]
        # Hollow markers otherwise look visually heavier than their filled
        # pairwise siblings at this small marker size.
        style["linewidth"] = ONE_POLE_MARKER_EDGEWIDTH
    return style


def _method_legend_label(family: str) -> str:
    """Construction-neutral method name for the split legend."""
    if family == "sae":
        return "SAE"
    if family == "sae_pre":
        return r"SAE$_{\mathrm{pre}}$"
    return family.upper()


def _split_figure_legends(
    fig: Any,
    present_families: Sequence[str],
    present_constructions: Sequence[str],
    present_references: Sequence[str],
    *,
    legend_y: float,
    fontsize: float,
    framed: bool,
) -> None:
    """Draw a compact centered row of inline-heading legend frames."""
    method_handles = []
    for family in present_families:
        # Take the marker from the same style helper used by the plotted
        # layerwise series.  This avoids maintaining a second marker mapping
        # (and, in particular, avoids relying on a FAMILY_MARKER global).
        style = _layer_style(family, "two_pole")
        method_handles.append(
            Line2D(
                [0], [0], linestyle="none", color=FAMILY_COLOR[family],
                marker=style["marker"], markersize=4.4,
                markerfacecolor=FAMILY_COLOR[family],
                markeredgecolor=FAMILY_COLOR[family], markeredgewidth=0.65,
            )
        )
    construction_order = [
        construction for construction in ("two_pole", "one_pole")
        if construction in present_constructions
    ]
    sweep_handles = []
    sweep_labels = []
    for construction in construction_order:
        filled = construction == "two_pole"
        sweep_handles.append(
            Line2D(
                [0], [0], color=INK_MUTED, linewidth=1.35,
                linestyle=LAYER_LINESTYLE[construction], marker="o", markersize=4.4,
                markerfacecolor=INK_MUTED if filled else "none",
                markeredgecolor=INK_MUTED, markeredgewidth=0.75,
            )
        )
        sweep_labels.append("Pairwise" if filled else "One-sided")

    reference_order = [
        construction for construction in ("two_pole", "one_pole")
        if construction in present_references
    ]
    reference_handles = [
        Line2D([0], [0], color=INK_MUTED, linewidth=1.35, linestyle=ALL_LAYER_LINESTYLE[construction])
        for construction in reference_order
    ]
    reference_labels = ["Pairwise" if construction == "two_pole" else "One-sided" for construction in reference_order]

    # Headings are ordinary inline entries, rather than titles placed above
    # their items. Measure the actual Legend objects before positioning them:
    # character-count estimates misalign frames whenever method names differ
    # in width.
    groups: List[Tuple[str, Sequence[Any], Sequence[str]]] = []

    def add_group(heading: str, group_handles: Sequence[Any], group_labels: Sequence[str]) -> None:
        if not group_handles:
            return
        groups.append((heading, group_handles, group_labels))

    if method_handles:
        add_group("Methods", method_handles, [_method_legend_label(family) for family in present_families])
    add_group("Construction", sweep_handles, sweep_labels)
    add_group("All-layer", reference_handles, reference_labels)
    if groups:
        legends = []
        for heading, group_handles, group_labels in groups:
            handles = [Line2D([], [], color="none", linewidth=0), *group_handles]
            labels = [heading, *group_labels]
            legend = fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                ncol=len(handles), frameon=True, fancybox=True,
                # Keep the framed legend conventional: neutral white rather
                # than the paper-tinted panel background.
                edgecolor="#cfcec6", facecolor="white", framealpha=1.0,
                fontsize=fontsize, handlelength=1.35, columnspacing=0.5,
                handletextpad=0.25, borderpad=0.38,
            )
            legend.get_texts()[0].set_fontweight("bold")
            legends.append(legend)

        renderer = fig.canvas.get_renderer()
        widths = [legend.get_window_extent(renderer).width / fig.bbox.width for legend in legends]
        gap = 0.012
        total_width = sum(widths) + gap * (len(widths) - 1)
        left = (1.0 - total_width) / 2.0
        for legend, width in zip(legends, widths):
            legend.set_bbox_to_anchor((left + width / 2.0, legend_y), transform=fig.transFigure)
            left += width + gap


def _task_title(task: str, *, latex: bool = False) -> str:
    """Use the paper macro for PGF output; derive visible text only for previews."""
    macro = TASK_LATEX.get(task)
    if latex and macro:
        return macro
    if macro:
        marker = rf"\newcommand{{{macro}}}"
        for definition in TASK_COMMAND_DEFINITIONS:
            if definition.startswith(marker):
                match = re.search(r"\\textsc\{([^}]*)\}", definition)
                if match:
                    return match.group(1)
    label = str(task_label_rendered(task))
    return label.lstrip("\\")


def _configure_paper_pgf() -> None:
    """Render paper task macros with XeLaTeX and the bundled Times face."""
    from matplotlib import font_manager

    paper_font = ROOT / "times.ttf"
    if not paper_font.is_file():
        raise FileNotFoundError(f"missing bundled paper font: {paper_font}")
    font_manager.fontManager.addfont(str(paper_font))
    font_dir = ROOT.as_posix().rstrip("/") + "/"
    preamble = "\n".join((
        r"\usepackage{fontspec}",
        rf"\setmainfont{{times.ttf}}[Path={{{font_dir}}}]",
        r"\usepackage{fontawesome7}",
        *TASK_COMMAND_DEFINITIONS,
    ))
    plt.rcParams.update({
        "font.family": "serif",
        "pgf.texsystem": "xelatex",
        "pgf.preamble": preamble,
        "pgf.rcfonts": False,
    })


def _is_num(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v)


def _metric_record(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "detection_score": detection_score(metrics),
        "intervention_score": intervention_score(metrics),
        "roc_auc_neutral": metrics.get("roc_auc_neutral"),
        "roc_auc_other": metrics.get("roc_auc_other"),
        "neutral_specificity": metrics.get("neutral_specificity"),
        "class_exclusivity": metrics.get("class_exclusivity"),
        "causal_signed_effect": metrics.get("causal_signed_effect"),
        "causal_signed_effect_weaken": metrics.get("causal_signed_effect_weaken"),
        "causal_lms": metrics.get("causal_lms"),
    }


def collect_layer_points(
    runs_root: Path, model: str, subdir: str = ""
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """task -> raw per-layer rows, and task -> raw ALL-baseline rows (all_k1 / all_act_prediction)."""
    model_dir = Path(runs_root) / model
    if subdir:
        model_dir = model_dir / subdir
    layer_out: Dict[str, List[Dict[str, Any]]] = {}
    all_out: Dict[str, List[Dict[str, Any]]] = {}
    if not model_dir.is_dir():
        return layer_out, all_out
    for path in sorted(model_dir.glob("*/results.json")):
        task = path.parent.name
        if _is_hidden_task(task):
            continue
        payload = _load_results(path)
        if not payload:
            continue
        layer_pts: List[Dict[str, Any]] = []
        all_pts: List[Dict[str, Any]] = []
        for row in payload.get("methods") or []:
            mid = str(row.get("method") or "")
            matched = False
            for family, pat in _LAYER_RE.items():
                m = pat.match(mid)
                if not m:
                    continue
                metrics = dict(row.get("metrics") or {})
                layer_pts.append(
                    {
                        "family": family,
                        "pole": _pole_for_method(family, mid),
                        "cls": m.group(1),
                        "layer": int(m.group(2)),
                        **_metric_record(metrics),
                    }
                )
                matched = True
                break
            if matched:
                continue
            for family, pat in _ALL_RE.items():
                m = pat.match(mid)
                if not m:
                    continue
                metrics = dict(row.get("metrics") or {})
                all_pts.append(
                    {
                        "family": family,
                        "pole": _pole_for_method(family, mid),
                        "cls": m.group(1),
                        **_metric_record(metrics),
                    }
                )
                break
        if layer_pts:
            layer_out[task] = layer_pts
        if all_pts:
            all_out[task] = all_pts
    return layer_out, all_out


_HEADLINE_METRIC_MAP = {
    # Required by the combined Det./Int. layer figure as horizontal
    # fixed-site baselines for AGIEND/GRADIEND/ACTIEND.
    "detection_score": "detection_score",
    "intervention_score": "intervention_score",
    "encoding_E": "encoding_E",
    "roc_auc_neutral": "roc_auc_neutral",
    "roc_auc_other": "roc_auc_other",
    "neutral_specificity": "neutral_specificity",
    "class_exclusivity": "class_exclusivity",
    "causal_signed_effect": "causal_signed_effect",
    "causal_signed_effect_weaken": "causal_signed_effect_weaken",
    "causal_lms": "causal_lms",
}


def collect_fixed_site_baselines(runs_root: Path, model: str, subdir: str = "") -> Dict[str, List[Dict[str, Any]]]:
    """task -> full-model GRADIEND/ACTIEND/AGIEND reference rows by construction.

    These methods do not have a layer sweep.  Keep pairwise and one-sided
    estimates separate, rather than silently preferring one or averaging them
    into a fictitious "all layers" baseline.
    """
    df = collect_canonical_group_rows(runs_root, model, subdir=subdir)
    out: Dict[str, List[Dict[str, Any]]] = {}
    if df.empty or "backend" not in df.columns:
        return out
    for backend in BASELINE_ONLY_FAMILIES:
        bdf = df[df["backend"] == backend]
        for task in bdf["task"].unique():
            if _is_hidden_task(task):
                continue
            tdf = bdf[bdf["task"] == task]
            for pole in ("two_pole", "one_pole"):
                candidates = tdf[tdf["pole"] == pole]
                if candidates.empty:
                    continue
                row = candidates.iloc[0]
                rec = {"family": backend, "pole": pole, "cls": "_headline"}
                for field in _HEADLINE_METRIC_MAP:
                    rec[field] = row.get(field)
                out.setdefault(str(task), []).append(rec)
    return out


_TENSORS_RE = re.compile(r"^(actiend|gradiend):([^:]+)-([^:]+):tensors:([^:]+)$")


def collect_tensors_baselines(runs_root: Path, model: str, subdir: str = "") -> Dict[str, List[Dict[str, Any]]]:
    """task -> by-tensor GRADIEND/ACTIEND rows (``{backend}:{pair}:tensors:{cls}``).

    Opt-in ablation, full_plus suite only — see TENSORS_FAMILIES comment above.
    Scanned directly off raw method ids (like collect_layer_points), not via
    collect_canonical_group_rows: that helper's headline-row filter
    (_class_headline_gradiend_actiend in analysis/method_groups.py) deliberately
    excludes the 4-part ``:tensors:{cls}`` ids from the none-split baseline
    bucket, so it never surfaces them at all.
    """
    model_dir = Path(runs_root) / model
    if subdir:
        model_dir = model_dir / subdir
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not model_dir.is_dir():
        return out
    for path in sorted(model_dir.glob("*/results.json")):
        task = path.parent.name
        if _is_hidden_task(task):
            continue
        payload = _load_results(path)
        if not payload:
            continue
        for row in payload.get("methods") or []:
            mid = str(row.get("method") or "")
            m = _TENSORS_RE.match(mid)
            if not m:
                continue
            metrics = dict(row.get("metrics") or {})
            rec = {"family": m.group(1), "cls": m.group(4), **_metric_record(metrics)}
            out.setdefault(str(task), []).append(rec)
    return out


def all_baseline_value(
    points: Sequence[Dict[str, Any]], metric: str, family: str, pole: Optional[str] = None
) -> "float | None":
    """Mean over classes of one full-model/all-layer aggregate."""
    vals = [
        float(p[metric])
        for p in points
        if p["family"] == family
        and (pole is None or p.get("pole") == pole)
        and _is_num(p.get(metric))
    ]
    return sum(vals) / len(vals) if vals else None


def series_for_metric(
    points: Sequence[Dict[str, Any]], metric: str, family: str, pole: Optional[str] = None
) -> Tuple[List[int], List[float]]:
    """Mean over classes at each layer, for one (task, family, metric)."""
    by_layer: Dict[int, List[float]] = {}
    for p in points:
        if p["family"] != family or (pole is not None and p.get("pole") != pole):
            continue
        v = p.get(metric)
        if not _is_num(v):
            continue
        by_layer.setdefault(int(p["layer"]), []).append(float(v))
    layers = sorted(by_layer)
    vals = [sum(by_layer[l]) / len(by_layer[l]) for l in layers]
    return layers, vals


def plot_metric_figure(
    task_points: Dict[str, List[Dict[str, Any]]],
    all_points: Dict[str, List[Dict[str, Any]]],
    *,
    metric: str,
    metric_label: str,
    model: str,
    tasks_order: Sequence[str],
    out_dir: Path,
    pole: Optional[str] = None,
    latex_task_labels: bool = False,
) -> "Path | None":
    def _has_any_data(t: str) -> bool:
        layer_ok = any(
            len(series_for_metric(task_points.get(t, []), metric, f, pole)[0]) >= 2
            for f in FAMILY_ORDER
        )
        aggregate_ok = any(
            all_baseline_value(all_points.get(t, []), metric, f, pole) is not None
            for f in FAMILY_ORDER
        )
        return layer_ok or aggregate_ok

    usable = [
        t
        for t in tasks_order
        if (t in task_points or t in all_points) and _has_any_data(t)
    ]
    if not usable:
        return None

    n = len(usable)
    # The paper task inventory has 15 visible tasks.  Keep the combined view
    # as a compact, stable 3 × 5 grid; construction-specific views use the
    # same five columns and simply omit unused panels.
    ncols = 5
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.65 * ncols, 2.3 * nrows), squeeze=False, sharex=True
    )

    handles: Dict[Tuple[str, str], Any] = {}
    aggregate_handles: Dict[Tuple[str, str], Any] = {}
    poles = (pole,) if pole else ("two_pole", "one_pole")
    for idx, task in enumerate(usable):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        pts = task_points.get(task, [])
        baseline_pts = all_points.get(task, [])
        any_line = False
        for family in FAMILY_ORDER:
            for construction in poles:
                layers, vals = series_for_metric(pts, metric, family, construction)
                key = (family, construction)
                if len(layers) >= 2:
                    any_line = True
                    style = _layer_style(family, construction)
                    (line,) = ax.plot(
                        layers,
                        vals,
                        color=FAMILY_COLOR[family],
                        marker=style["marker"],
                        markerfacecolor=style["facecolors"],
                        markeredgecolor=style["edgecolors"],
                        markeredgewidth=style["linewidth"],
                        markersize=4.5,
                        linewidth=1.6,
                        linestyle=LAYER_LINESTYLE[construction],
                        label=_family_label(family, construction),
                        zorder=3,
                    )
                    handles.setdefault(key, line)
                aggregate = all_baseline_value(baseline_pts, metric, family, construction)
                if aggregate is not None:
                    any_line = True
                    ax.axhline(
                        aggregate, color=FAMILY_COLOR[family],
                        linestyle=ALL_LAYER_LINESTYLE[construction],
                        linewidth=1.25, alpha=0.85, zorder=2,
                    )
                    style = _layer_style(family, construction)
                    aggregate_handles.setdefault(
                        key,
                        Line2D(
                            [0], [0], color=FAMILY_COLOR[family], linewidth=1.4,
                            linestyle=ALL_LAYER_LINESTYLE[construction],
                            marker=style["marker"], markersize=4.5,
                            markerfacecolor=style["facecolors"],
                            markeredgecolor=style["edgecolors"],
                            markeredgewidth=style["linewidth"],
                        ),
                    )
        ax.set_title(_task_title(task, latex=latex_task_labels), fontsize=13.5, color=INK_PRIMARY, pad=4)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=INK_MUTED, labelsize=7.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        if not any_line:
            ax.text(
                0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color=INK_MUTED, fontsize=8,
            )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.supxlabel("Layer", fontsize=9, color=INK_SECONDARY)
    fig.supylabel(metric_label, fontsize=9, color=INK_SECONDARY)

    present_keys = set(handles) | set(aggregate_handles)
    present_families = [
        family for family in FAMILY_ORDER
        if any(key[0] == family for key in present_keys)
    ]
    present_constructions = [
        construction for construction in poles
        if any(key[1] == construction for key in present_keys)
    ]
    present_references = [
        construction for construction in poles
        if any(key[1] == construction for key in aggregate_handles)
    ]
    _split_figure_legends(
        fig, present_families, present_constructions, present_references,
        legend_y=0.91, fontsize=11.2, framed=True,
    )
    # The task names above each panel are intentional.  There is deliberately
    # no figure-level title: the generated LaTeX figure caption names the
    # metric, model, and construction without duplicating it inside the plot.
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.88))

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
    suffix = "" if pole is None else f"_{pole}"
    pdf_path = out_dir / f"layer_vs_{slug}_{model}{suffix}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", backend="pgf" if latex_task_labels else None)
    plt.close(fig)
    return pdf_path


def plot_det_int_figure(
    task_points: Dict[str, List[Dict[str, Any]]],
    all_points: Dict[str, List[Dict[str, Any]]],
    *,
    model: str,
    tasks_order: Sequence[str],
    out_dir: Path,
    latex_task_labels: bool = False,
) -> "Path | None":
    """Draw the paper view: top three rows Det., bottom three rows Int.

    Limits are deliberately left to Matplotlib's data limits.  In particular,
    this does not force either metric to span 0--100%, which would obscure the
    layer differences the diagnostic is meant to show.
    """
    def has_data(task: str, metric: str) -> bool:
        return any(
            series_for_metric(task_points.get(task, []), metric, family)[0]
            or all_baseline_value(all_points.get(task, []), metric, family) is not None
            for family in FAMILY_ORDER
        )

    usable = [task for task in tasks_order if any(has_data(task, metric) for metric, _ in PAPER_METRICS)]
    if not usable:
        return None

    def nice_shared_upper(values: Sequence[float]) -> float | None:
        """A readable common upper bound without forcing a zero lower bound."""
        finite = [float(value) for value in values if _is_num(value)]
        if not finite:
            return None
        maximum = max(finite)
        # A small headroom prevents the highest curve touching the frame.  The
        # choice is based once per (model, metric), never panel by panel.
        target = maximum + max(0.01, abs(maximum) * 0.03)
        for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            if target / step <= 100:
                return math.ceil(target / step) * step
        return target

    shared_upper: Dict[str, float | None] = {}
    for metric, _label in PAPER_METRICS:
        values: List[float] = []
        for task in usable:
            for family in FAMILY_ORDER:
                for construction in ("two_pole", "one_pole"):
                    values.extend(series_for_metric(task_points.get(task, []), metric, family, construction)[1])
                    aggregate = all_baseline_value(all_points.get(task, []), metric, family, construction)
                    if aggregate is not None:
                        values.append(float(aggregate))
        shared_upper[metric] = nice_shared_upper(values)

    ncols = 5
    task_rows = -(-len(usable) // ncols)
    # Insert a narrow spacer row between the Det. and Int. blocks.  Using an
    # actual GridSpec row gives only that boundary extra breathing room instead
    # of increasing hspace between every task row.
    gap_row = task_rows
    fig, axes = plt.subplots(
        task_rows * 2 + 1, ncols, figsize=(2.58 * ncols, 1.60 * task_rows * 2),
        squeeze=False, sharex=True,
        gridspec_kw={"height_ratios": [1.0] * task_rows + [0.05] + [1.0] * task_rows},
    )
    for gap_ax in axes[gap_row]:
        gap_ax.axis("off")
    handles: Dict[Tuple[str, str], Any] = {}
    aggregate_handles: Dict[Tuple[str, str], Any] = {}
    for index, task in enumerate(usable):
        task_row, col = divmod(index, ncols)
        for metric_offset, (metric, metric_label) in enumerate(PAPER_METRICS):
            # First half is Detection for all 15 tasks; second half repeats
            # the same task grid for Intervention.  This makes cross-task
            # comparison within each metric immediate while retaining the
            # same column for a task across the two halves.
            plot_row = task_row if metric_offset == 0 else task_row + task_rows + 1
            ax = axes[plot_row][col]
            pts, baseline_pts = task_points.get(task, []), all_points.get(task, [])
            any_line = False
            for family in FAMILY_ORDER:
                for construction in ("two_pole", "one_pole"):
                    layers, values = series_for_metric(pts, metric, family, construction)
                    key = (family, construction)
                    if len(layers) >= 2:
                        any_line = True
                        style = _layer_style(family, construction)
                        (line,) = ax.plot(
                            layers, values, color=FAMILY_COLOR[family],
                            marker=style["marker"], markerfacecolor=style["facecolors"],
                            markeredgecolor=style["edgecolors"],
                            markeredgewidth=style["linewidth"], markersize=4.0,
                            linewidth=1.35, linestyle=LAYER_LINESTYLE[construction], zorder=3,
                        )
                        handles.setdefault(key, line)
                    aggregate = all_baseline_value(baseline_pts, metric, family, construction)
                    if aggregate is not None:
                        any_line = True
                        ax.axhline(
                            aggregate, color=FAMILY_COLOR[family],
                            linestyle=ALL_LAYER_LINESTYLE[construction],
                            linewidth=1.1, alpha=0.85, zorder=2,
                        )
                        style = _layer_style(family, construction)
                        aggregate_handles.setdefault(
                            key,
                            Line2D(
                                [0], [0], color=FAMILY_COLOR[family], linewidth=1.2,
                                linestyle=ALL_LAYER_LINESTYLE[construction],
                                marker=style["marker"], markersize=4.0,
                                markerfacecolor=style["facecolors"],
                                markeredgecolor=style["edgecolors"],
                                markeredgewidth=style["linewidth"],
                            ),
                        )
            # Repeat the task label in both metric halves so the lower grid is
            # self-identifying rather than relying on the upper half.
            ax.set_title(_task_title(task, latex=latex_task_labels), fontsize=13.5, color=INK_PRIMARY, pad=3)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8.5, color=INK_SECONDARY)
            ax.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(AXIS_COLOR)
            ax.tick_params(colors=INK_MUTED, labelsize=7)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
            # Det panels share one ceiling across every task; Int panels share
            # another.  Deliberately leave the lower bound to each panel's
            # observed data instead of implying a 0--100% range.
            if shared_upper[metric] is not None:
                ax.set_ylim(top=shared_upper[metric])
            if not any_line:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=INK_MUTED, fontsize=7)
    for index in range(len(usable), task_rows * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis("off")
        axes[row + task_rows + 1][col].axis("off")
    present_keys = set(handles) | set(aggregate_handles)
    present_families = [
        family for family in FAMILY_ORDER
        if any(key[0] == family for key in present_keys)
    ]
    present_constructions = [
        construction for construction in ("two_pole", "one_pole")
        if any(key[1] == construction for key in present_keys)
    ]
    present_references = [
        construction for construction in ("two_pole", "one_pole")
        if any(key[1] == construction for key in aggregate_handles)
    ]
    _split_figure_legends(
        fig, present_families, present_constructions, present_references,
        legend_y=0.84, fontsize=10.0, framed=True,
    )
    fig.supxlabel("Layer", y=0.018, fontsize=9, color=INK_SECONDARY)
    # Give individual task rows room to breathe as well as retaining the
    # deliberately larger Det./Int. separator row.
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.07, top=0.78, hspace=0.33, wspace=0.33)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"layer_det_int_{model}.pdf"
    fig.savefig(path, bbox_inches="tight", backend="pgf" if latex_task_labels else None)
    plt.close(fig)
    return path


def _latex_escape(text: str) -> str:
    """Escape plain labels inserted into the generated figure-book TeX."""
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def write_latex_figure_book(out_dir: Path, written: Sequence[Path], model: str) -> Path:
    """Write the titled TeX wrapper consumed by the summary-PDF merger."""
    parts = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.35in]{geometry}",
        r"\usepackage{graphicx}",
        r"\pagestyle{empty}",
        r"\begin{document}",
    ]
    for path in written:
        stem = path.stem
        metric = "Detection and Intervention" if stem.startswith("layer_det_int_") else stem
        title = (
            f"Layerwise {metric} by intervention location: {_latex_escape(model)}."
        )
        parts.extend(
            [
                r"\begin{center}",
                rf"\textbf{{{title}}}",
                r"\end{center}",
                rf"\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{_latex_escape(path.name)}}}",
                r"\clearpage",
            ]
        )
    parts.append(r"\end{document}")
    tex_path = out_dir / "layer_figures.tex"
    tex_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--subdir", default="", help="OUTPUT_SUBDIR value, e.g. 'suite_full'")
    args = parser.parse_args()

    _configure_paper_pgf()
    task_points, all_points = collect_layer_points(args.runs, args.model, subdir=args.subdir)
    fixed_site = collect_fixed_site_baselines(args.runs, args.model, subdir=args.subdir)
    for task, recs in fixed_site.items():
        all_points.setdefault(task, []).extend(recs)
    if not task_points and not all_points:
        where = f"{args.runs}/{args.model}" + (f"/{args.subdir}" if args.subdir else "")
        print(f"No per-layer sae/sae_pre/caa data found under {where}", file=sys.stderr)
        # A model may legitimately have only headline results synced. Do not
        # make the table/PDF build fail merely because its optional layerwise
        # diagnostic source is absent.
        return

    available = set(task_points) | set(all_points)
    # The paper grid is deliberately restricted to its 15 visible main-study
    # tasks.  Keep the order shared with the LaTeX tables, rather than adding
    # an uncaptioned panel for a hidden/legacy task (e.g. race_one_pole,
    # religion_one_pole, ioi) -- excluded explicitly, not just by relying on
    # PAPER_TASK_ORDER's own membership, since a fixed-site baseline row can
    # exist for a hidden task even though it must never appear in a figure.
    tasks_order = [
        task for task in PAPER_TASK_ORDER if task in available and not _is_hidden_task(task)
    ]
    suffix = args.model + (f"_{args.subdir}" if args.subdir else "")
    out_dir = args.out / f"layer_{suffix}"

    path = plot_det_int_figure(
        task_points, all_points, model=args.model, tasks_order=tasks_order, out_dir=out_dir,
        latex_task_labels=True,
    )
    written = [path] if path else []
    if path:
        print(f"Wrote {path}")

    if not written:
        print("No figures written.", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {write_latex_figure_book(out_dir, written, args.model)}")


if __name__ == "__main__":
    main()

