#!/usr/bin/env python
"""Paper-suitable figures beyond the summary tables and layer plots:

  * suitability scatter  — E vs |causal effect|, one facet per PAPER_METHOD_GROUP
  * one-pole vs two-pole  — dumbbell chart of causal_signed_effect per task
  * sae-selection heatmap — k1/kstar/all_k1 vs sel_arad_out/sel_jh_f1/sel_opp_fire
                             (suite_full only — these ablations were never run
                             under --suite core; checked 2026-08-19)
  * compute summary       — wall-clock + peak GPU per backend, from raw.cost

Usage:
  python analysis/paper_figures.py --model gpt2-small
  python analysis/paper_figures.py --model gpt2-small --sae-select-subdir suite_full
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.method_groups import _load_results  # noqa: E402
from analysis.plot_style import (  # noqa: E402
    BACKEND_COLORS,
    HOLLOW_EDGE_WIDTH,
    SAE_COLOR,
    method_marker,
    scatter_style,
    strip_figure_titles,
)
from analysis.summary_latex import (  # noqa: E402
    PAPER_METHOD_GROUPS,
    PAPER_ONE_POLE_TASKS,
    METHOD_LATEX,
    TASK_LATEX,
    _is_hidden_task,
    _order_tasks,
    collect_canonical_group_rows,
)

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "analysis" / "figures"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
ACCENT = SAE_COLOR

# Fixed identity across every figure in this repo's analysis/ scripts.
FAMILY_COLOR = dict(BACKEND_COLORS)

# suitability.py's own thresholds (configs/defaults.yaml): e_ok / causal_effect_threshold.
E_OK_THRESHOLD = 0.8
CAUSAL_EFFECT_THRESHOLD = 0.05

# Task structural category (from configs/tasks/*.yaml `classes:`), independent
# of which method/pole a point belongs to (that's already the facet).
# "one_pole_task" = ablations.pair: false (only ever trains one-pole) — the
# same 8 tasks as summary_latex.py's PAPER_ONE_POLE_TASKS. The rest split by
# real class count: 2 classes = binary, 3 = multiclass.
_TASK_N_CLASSES: Dict[str, int] = {
    "emotion": 2, "function_composition": 2, "gender_en": 2, "induction": 2,
    "ioi": 2, "ioi_mib": 2, "key_value": 2, "language": 3, "pronoun_number": 2,
    "pronoun_person": 3, "race": 3, "race_one_pole": 3, "ravel_continent": 3,
    "ravel_country": 3, "ravel_language": 3, "religion": 3, "religion_one_pole": 3,
    "repetition": 2,
}
_ONE_POLE_TASKS = frozenset(PAPER_ONE_POLE_TASKS)

TASK_CATEGORY_COLOR = {"one_pole_task": "#2a78d6", "binary": "#eb6834", "multiclass": "#1baf7a"}
TASK_CATEGORY_MARKER = {"one_pole_task": "o", "binary": "s", "multiclass": "^"}
TASK_CATEGORY_LABEL = {
    "one_pole_task": "One-pole-only task", "binary": "Binary task (2 classes)",
    "multiclass": "Multi-class task (3+)",
}


def task_category(task: str) -> str:
    if task in _ONE_POLE_TASKS:
        return "one_pole_task"
    return "binary" if _TASK_N_CLASSES.get(task, 2) == 2 else "multiclass"


def _is_num(v: Any) -> bool:
    import math
    if v is None or isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f)


# ---------------------------------------------------------------- suitability scatter

def suitability_scatter(model: str, runs_root: Path, subdir: str, out_dir: Path) -> Optional[Path]:
    df = collect_canonical_group_rows(runs_root, model, subdir=subdir)
    df = df[~df["task"].map(_is_hidden_task)]
    sub = df[df["encoding_E"].notna() & df["causal_signed_effect"].notna()]
    if sub.empty:
        print("suitability scatter: no rows with both encoding_E and causal_signed_effect", file=sys.stderr)
        return None

    groups = [g for g in PAPER_METHOD_GROUPS if g in set(sub["method_group"])]
    ncols = 3
    nrows = -(-len(groups) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)

    cats_present: List[str] = []
    for idx, group in enumerate(groups):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        gdf = sub[sub["method_group"] == group]
        for cat in ("one_pole_task", "binary", "multiclass"):
            cdf = gdf[gdf["task"].map(task_category) == cat]
            if cdf.empty:
                continue
            cats_present.append(cat)
            x = cdf["encoding_E"].astype(float).tolist()
            y = cdf["causal_signed_effect"].abs().astype(float).tolist()
            ax.scatter(
                x, y, s=30, color=TASK_CATEGORY_COLOR[cat], marker=TASK_CATEGORY_MARKER[cat],
                alpha=0.85, edgecolors="none", zorder=3, label=TASK_CATEGORY_LABEL[cat],
            )
        ax.axvline(E_OK_THRESHOLD, color=AXIS_COLOR, linewidth=1.0, linestyle=":", zorder=1)
        ax.axhline(CAUSAL_EFFECT_THRESHOLD, color=AXIS_COLOR, linewidth=1.0, linestyle=":", zorder=1)
        ax.set_title(METHOD_LATEX.get(group, group), fontsize=9.5, color=INK_PRIMARY, pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    for idx in range(len(groups), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.supxlabel("Encoding $E$", fontsize=10, color=INK_SECONDARY)
    fig.supylabel(r"$|\Delta P^{+}|$ (causal effect)", fontsize=10, color=INK_SECONDARY)

    order = [c for c in ("one_pole_task", "binary", "multiclass") if c in cats_present]
    legend_handles = [
        Line2D([0], [0], marker=TASK_CATEGORY_MARKER[c], color=TASK_CATEGORY_COLOR[c], linestyle="none", markersize=7)
        for c in order
    ]
    fig.legend(
        legend_handles, [TASK_CATEGORY_LABEL[c] for c in order],
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(order), frameon=False, fontsize=9,
    )
    fig.suptitle(
        f"Suitability: encoding vs. causal effect — {model}" + (f" ({subdir})" if subdir else "") + "\n"
        f"dotted lines: $E$={E_OK_THRESHOLD} / $|\\Delta P^+|$={CAUSAL_EFFECT_THRESHOLD} (suitability.py thresholds)",
        fontsize=10.5, color=INK_PRIMARY, y=1.06,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.87))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"suitability_scatter_{model}.pdf"
    strip_figure_titles(fig)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# ---------------------------------------------------------------- one-pole vs two-pole

def one_pole_two_pole_chart(model: str, runs_root: Path, subdir: str, out_dir: Path) -> Optional[Path]:
    df = collect_canonical_group_rows(runs_root, model, subdir=subdir)
    df = df[~df["task"].map(_is_hidden_task)]
    sub = df[df["causal_signed_effect"].notna() & df["pole"].isin(["one_pole", "two_pole"])]
    if sub.empty:
        print("one-pole vs two-pole: no causal_signed_effect rows", file=sys.stderr)
        return None

    families = ("gradiend", "actiend")
    rows: List[Dict[str, Any]] = []
    for task in sub["task"].unique():
        for backend in families:
            tdf = sub[(sub["task"] == task) & (sub["backend"] == backend)]
            two = tdf[tdf["pole"] == "two_pole"]["causal_signed_effect"]
            one = tdf[tdf["pole"] == "one_pole"]["causal_signed_effect"]
            if two.empty and one.empty:
                continue
            rows.append(
                {
                    "task": task,
                    "backend": backend,
                    "two_pole": float(two.iloc[0]) if not two.empty else None,
                    "one_pole": float(one.iloc[0]) if not one.empty else None,
                }
            )
    if not rows:
        return None

    tasks = _order_tasks(sorted({r["task"] for r in rows}))
    y_pos = {t: i for i, t in enumerate(tasks)}

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(tasks) + 1.6))
    offsets = {"gradiend": -0.16, "actiend": 0.16}
    handles: Dict[str, Any] = {}
    for row in rows:
        y = y_pos[row["task"]] + offsets[row["backend"]]
        color = FAMILY_COLOR[row["backend"]]
        two_v, one_v = row["two_pole"], row["one_pole"]
        if two_v is not None and one_v is not None:
            ax.plot([two_v, one_v], [y, y], color=color, linewidth=1.4, alpha=0.6, zorder=2)
        if two_v is not None:
            h = ax.scatter(
                [two_v],
                [y],
                s=34,
                zorder=3,
                label=f"{row['backend']} two-pole",
                **scatter_style(row["backend"], "pairwise"),
            )
            handles.setdefault((row["backend"], "two_pole"), h)
        if one_v is not None:
            h = ax.scatter(
                [one_v],
                [y],
                s=34,
                zorder=3,
                **scatter_style(row["backend"], "one_pole"),
            )
            handles.setdefault((row["backend"], "one_pole"), h)

    ax.set_yticks([y_pos[t] for t in tasks])
    ax.set_yticklabels([TASK_LATEX.get(t, t.replace("_", " ")) for t in tasks], fontsize=8.5, color=INK_PRIMARY)
    ax.axvline(0, color=AXIS_COLOR, linewidth=1.0, zorder=1)
    ax.set_xlabel(r"$\Delta P^{+}$ (causal effect)", fontsize=9.5, color=INK_SECONDARY)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.invert_yaxis()

    legend_handles = [
        Line2D([0], [0], marker=method_marker("gradiend"), color=FAMILY_COLOR["gradiend"], markerfacecolor=FAMILY_COLOR["gradiend"], linestyle="none", markersize=6),
        Line2D([0], [0], marker=method_marker("gradiend"), color=FAMILY_COLOR["gradiend"], markerfacecolor="none", linestyle="none", markersize=6, markeredgewidth=HOLLOW_EDGE_WIDTH),
        Line2D([0], [0], marker=method_marker("actiend"), color=FAMILY_COLOR["actiend"], markerfacecolor=FAMILY_COLOR["actiend"], linestyle="none", markersize=6),
        Line2D([0], [0], marker=method_marker("actiend"), color=FAMILY_COLOR["actiend"], markerfacecolor="none", linestyle="none", markersize=6, markeredgewidth=HOLLOW_EDGE_WIDTH),
    ]
    legend_labels = ["GRADIEND two-pole", "GRADIEND one-pole", "ACTIEND two-pole", "ACTIEND one-pole"]
    fig.legend(
        legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, frameon=False, fontsize=8.5,
    )
    fig.suptitle(
        f"One-pole vs. two-pole causal effect — {model}" + (f" ({subdir})" if subdir else ""),
        fontsize=11, color=INK_PRIMARY, y=1.05,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.93))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"one_pole_vs_two_pole_{model}.pdf"
    strip_figure_titles(fig)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# ---------------------------------------------------------------- SAE feature-selection heatmap

_SEL_METHODS: Tuple[str, ...] = ("k1", "kstar", "all_k1", "sel_arad_out", "sel_jh_f1", "sel_opp_fire")
_SEL_LABEL = {
    "k1": "$k{=}1$", "kstar": "$k^*$", "all_k1": "All-layer $k{=}1$",
    "sel_arad_out": "Arad output-score", "sel_jh_f1": "Jørgensen–Hansen $F_1$",
    "sel_opp_fire": "Opp-fire penalty",
}
_SEL_RE = re.compile(r"^(sae|sae_pre):([^:]+):(k1|kstar|all_k1|sel_arad_out|sel_jh_f1|sel_opp_fire)$")


def _collect_sae_selection(runs_root: Path, model: str, subdir: str) -> Dict[str, Dict[Tuple[str, str], List[float]]]:
    """family -> {(task, selection): [encoding_E per class]}"""
    model_dir = Path(runs_root) / model
    if subdir:
        model_dir = model_dir / subdir
    out: Dict[str, Dict[Tuple[str, str], List[float]]] = {"sae": {}, "sae_pre": {}}
    if not model_dir.is_dir():
        return out
    from suitability import encoding_e

    for path in sorted(model_dir.glob("*/results.json")):
        task = path.parent.name
        if _is_hidden_task(task):
            continue
        payload = _load_results(path)
        if not payload:
            continue
        for row in payload.get("methods") or []:
            mid = str(row.get("method") or "")
            m = _SEL_RE.match(mid)
            if not m:
                continue
            family, _cls, sel = m.groups()
            e = encoding_e(dict(row.get("metrics") or {}))
            if _is_num(e):
                out[family].setdefault((task, sel), []).append(float(e))
    return out


def sae_selection_heatmap(model: str, runs_root: Path, subdir: str, out_dir: Path) -> List[Path]:
    data = _collect_sae_selection(runs_root, model, subdir)
    written: List[Path] = []
    for family in ("sae", "sae_pre"):
        cell = data[family]
        tasks = _order_tasks(sorted({t for (t, _s) in cell}))
        if not tasks:
            print(f"sae-selection heatmap: no {family} selection-ablation data under {subdir or '(no subdir)'}", file=sys.stderr)
            continue
        grid = [[None] * len(_SEL_METHODS) for _ in tasks]
        for ti, task in enumerate(tasks):
            for si, sel in enumerate(_SEL_METHODS):
                vals = cell.get((task, sel))
                if vals:
                    grid[ti][si] = sum(vals) / len(vals)

        fig, ax = plt.subplots(figsize=(1.35 * len(_SEL_METHODS) + 1.6, 0.36 * len(tasks) + 1.6))
        import numpy as np

        arr = np.array([[v if v is not None else np.nan for v in row] for row in grid])
        cmap = plt.get_cmap("Blues")
        cmap.set_bad(color="#f0efec")
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for ti in range(len(tasks)):
            for si in range(len(_SEL_METHODS)):
                v = grid[ti][si]
                if v is None:
                    ax.text(si, ti, "–", ha="center", va="center", fontsize=8, color=INK_MUTED)
                else:
                    txt_color = "white" if v > 0.6 else INK_PRIMARY
                    ax.text(si, ti, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)
        ax.set_xticks(range(len(_SEL_METHODS)))
        ax.set_xticklabels([_SEL_LABEL[s] for s in _SEL_METHODS], rotation=35, ha="right", fontsize=8, color=INK_SECONDARY)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_LATEX.get(t, t.replace("_", " ")) for t in tasks], fontsize=8, color=INK_PRIMARY)
        ax.set_title(
            f"SAE{'$_{pre}$' if family == 'sae_pre' else ''} feature selection: encoding $E$ — {model}"
            + (f" ({subdir})" if subdir else ""),
            fontsize=10.5, color=INK_PRIMARY, pad=10,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=7.5, colors=INK_MUTED)
        cbar.outline.set_visible(False)
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"sae_selection_heatmap_{family}_{model}.pdf"
        strip_figure_titles(fig)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        written.append(pdf_path)
    return written


# ---------------------------------------------------------------- compute summary
#
# One ``raw.cost`` record is one causal-*stage* invocation for a (task,
# backend) pair, not a normalized unit of work: a single SAE record covers
# every SAE layer x class x k-selection variant swept in one ``CostSection``
# (see ``causal_study.py``'s three ``CostSection`` call sites — SAE's spans
# ~670 lines and, under suite_full, also the opt-in clamp-target ablation),
# while a single GRADIEND/ACTIEND record covers one strength-grid sweep for
# one split-group. That's a real, deliberate difference in how much each
# backend's causal *protocol* actually does per task — it does not mean the
# numbers are meaningless, but it does mean "seconds per call" only supports
# the claim "cost of running backend X's full causal protocol on this task",
# never "cost per equivalent unit of steering work".
#
# Two channels follow from that constraint, kept as separate functions/plots
# rather than folded into one, and NEITHER pools core with suite_full — that
# was the bug in an earlier version of this function (suite_full's SAE rows
# uniquely include the clamp sweep, so pooling inflated SAE's apparent cost
# with work no other backend even has the option to do). Pass --subdir like
# every other chart in this module; nothing here "recognizes" or merges
# OUTPUT_SUBDIR trees on your behalf.
#
#   1. compute_summary(model, subdir=...)      — backend-vs-backend, ONE
#      (model, subdir) at a time, matched to the task set where every
#      backend being compared actually has a record in that same tree.
#   2. compute_cross_model_summary(models, subdir=...) — model-vs-model for
#      each backend held fixed. This is the well-posed comparison: the
#      "amount of work per call" is constant within a backend, so varying
#      only the model isolates how cost scales with model size, without the
#      cross-backend granularity mismatch above.
#
# GPU cost uses ``peak_delta_gb`` (growth vs. allocated-at-entry) rather than
# the absolute high-water mark, since the latter can include tensors still
# resident from an earlier stage in the same process.


def _collect_task_backend_cost(
    model: str, runs_root: Path, subdir: str, *, phase: str = "causal"
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[str]]:
    """(task, backend) -> aggregate cost slot for one (model, subdir, phase).

    Reads exactly the one tree ``runs/{model}/{subdir}/`` (or
    ``runs/{model}/`` when ``subdir`` is empty) — no merging across trees.

    ``n_strength_points`` sums ``causal_study.py``'s own count of how many
    (class, strength-candidate) evaluations happened inside each timer's
    scope (added to the ``CostSection`` meta at its three call sites,
    counted from the ``CausalMethodResult`` list every backend appends to —
    an exact count, not an estimate). This is the one unit of work genuinely
    common to every backend: SAE's bigger bundle (every layer x class x
    k-selection variant) and GRADIEND/ACTIEND's smaller one both reduce to
    "how many strength points did we actually evaluate", so seconds-per-unit
    is comparable across backends in a way seconds-per-call never was.
    Records written before this instrumentation landed won't have the field
    (``has_units=False`` for that slot) — those can only be normalized by
    task, not by unit; callers must check ``has_units`` before dividing.
    """
    model_dir = Path(runs_root) / model
    base = model_dir / subdir if subdir else model_dir
    by_task_backend: Dict[Tuple[str, str], Dict[str, Any]] = {}
    tasks_with_cost: List[str] = []
    for path in sorted(base.glob("*/results.json")):
        task = path.parent.name
        if _is_hidden_task(task):
            continue
        payload = _load_results(path)
        if not payload:
            continue
        cost = ((payload.get("raw") or {}).get("cost")) or []
        if cost:
            tasks_with_cost.append(task)
        for row in cost:
            if row.get("phase") != phase:
                continue
            backend = row.get("backend")
            if not backend:
                continue
            slot = by_task_backend.setdefault(
                (task, backend),
                {
                    "seconds": 0.0, "peak_gb": 0.0, "peak_delta_gb": 0.0, "has_delta": False,
                    "n": 0, "n_strength_points": 0, "has_units": False,
                },
            )
            slot["seconds"] += float(row.get("seconds") or 0)
            slot["peak_gb"] = max(slot["peak_gb"], float(row.get("peak_cuda_gb") or 0))
            pd = row.get("peak_delta_gb")
            if pd is not None:
                slot["peak_delta_gb"] = max(slot["peak_delta_gb"], float(pd))
                slot["has_delta"] = True
            units = row.get("n_strength_points")
            if units is not None:
                slot["n_strength_points"] += int(units)
                slot["has_units"] = True
            slot["n"] += 1
    return by_task_backend, tasks_with_cost


def compute_summary(model: str, runs_root: Path, out_dir: Path, subdir: str = "") -> Optional[Path]:
    """Backend-vs-backend causal-stage cost for ONE (model, subdir) tree.

    Matches backends to the intersection of tasks where every backend being
    compared actually has a causal-phase record in *this* tree — no pooling
    across subdirs. See the module comment above for what this can and
    cannot be used to claim.
    """
    by_tb, tasks_with_cost = _collect_task_backend_cost(model, runs_root, subdir, phase="causal")
    sd_label = subdir or "core"
    if not by_tb:
        print(f"compute summary: no causal-phase raw.cost data found for {model}/{sd_label}", file=sys.stderr)
        return None

    backends = sorted({b for (_t, b) in by_tb})
    tasks_by_backend = {b: {t for (t, bb) in by_tb if bb == b} for b in backends}
    matched_tasks = set.intersection(*tasks_by_backend.values()) if tasks_by_backend else set()

    lines = [
        f"Compute summary (causal phase only) — {model} / {sd_label}",
        f"(from raw.cost records; tasks with any cost data: {', '.join(sorted(tasks_with_cost))})",
        "",
        "'seconds_per_unit' = total seconds / total (class, strength-point) evaluations",
        "actually run, per causal_study.py's own count — the one unit of work common",
        "to every backend, regardless of how many layers/k-selections a backend bundles",
        "per task. Only available for records written after this counting was added;",
        "'has_units=no' below means that backend's records predate it, in which case",
        "only 'mean_seconds_per_task' (a coarser, non-normalized total) is meaningful.",
        "",
        f"backends present: {backends}",
        f"matched task set (every backend has a record): n={len(matched_tasks)} "
        f"{sorted(matched_tasks) if matched_tasks else '(none — no task has causal data for every backend)'}",
        "",
    ]

    per_backend: Dict[str, Dict[str, Any]] = {}
    if matched_tasks:
        lines.append(
            f"{'backend':<12}{'mean_s/task':>13}{'seconds_per_unit':>18}{'has_units':>11}"
            f"{'mean_peak_delta_gb':>20}{'n_tasks':>9}"
        )
        for b in backends:
            secs = [by_tb[(t, b)]["seconds"] for t in matched_tasks]
            peaks = [by_tb[(t, b)]["peak_gb"] for t in matched_tasks]
            deltas = [by_tb[(t, b)]["peak_delta_gb"] for t in matched_tasks if by_tb[(t, b)]["has_delta"]]
            mean_delta = (sum(deltas) / len(deltas)) if deltas else None
            has_units = all(by_tb[(t, b)]["has_units"] for t in matched_tasks)
            total_units = sum(by_tb[(t, b)]["n_strength_points"] for t in matched_tasks)
            seconds_per_unit = (sum(secs) / total_units) if (has_units and total_units > 0) else None
            per_backend[b] = {
                "mean_seconds": sum(secs) / len(secs),
                "mean_peak_gb": sum(peaks) / len(peaks),
                "mean_peak_delta_gb": mean_delta,
                "seconds_per_unit": seconds_per_unit,
                "total_units": total_units,
                "n_tasks": len(matched_tasks),
            }
            d_txt = f"{mean_delta:.2f}" if mean_delta is not None else "n/a"
            spu_txt = f"{seconds_per_unit:.3f}" if seconds_per_unit is not None else "n/a"
            lines.append(
                f"{b:<12}{per_backend[b]['mean_seconds']:>13.1f}{spu_txt:>18}"
                f"{('yes' if has_units else 'no'):>11}{d_txt:>20}{len(matched_tasks):>9}"
            )
    else:
        lines.append(
            "No plot written: at least one backend has zero overlap with the others' "
            "task coverage in this tree — a mean over an empty matched set would not "
            "mean anything. Check per-backend task coverage above / in results.json "
            "error rows before assuming this is a bug."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compute_summary_{model}_{sd_label}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    if per_backend:
        use_units = all(stats["seconds_per_unit"] is not None for stats in per_backend.values())
        fig, ax = plt.subplots(figsize=(5.4, 3.7))
        for backend, stats in per_backend.items():
            x = stats["seconds_per_unit"] if use_units else stats["mean_seconds"]
            y = stats["mean_peak_delta_gb"] if stats["mean_peak_delta_gb"] is not None else stats["mean_peak_gb"]
            color = FAMILY_COLOR.get(backend, ACCENT)
            ax.scatter([x], [y], s=70, color=color, zorder=3, edgecolors="white", linewidths=0.6)
            label = f"{backend} (n={stats['n_tasks']})" if use_units else f"{backend} (n={stats['n_tasks']}, no unit data)"
            ax.annotate(
                label, (x, y),
                textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK_PRIMARY,
            )
        if use_units:
            ax.set_xlabel(
                "Seconds per (class, strength-point) evaluated\n(normalized unit cost — matched task set)",
                fontsize=8.5, color=INK_SECONDARY,
            )
        else:
            ax.set_xlabel(
                "Mean total seconds per task's causal stage\n"
                "(NOT unit-normalized — some backends predate n_strength_points instrumentation)",
                fontsize=8, color=INK_SECONDARY,
            )
        ax.set_ylabel("Mean peak GPU growth per task (GB, Δ vs. stage entry)", fontsize=9, color=INK_SECONDARY)
        ax.set_title(
            f"Causal-stage compute cost by backend — {model} / {sd_label}\n"
            f"matched-task means, n={len(matched_tasks)} tasks"
            + ("" if use_units else "  [WARNING: per-task, not per-unit — see caption]"),
            fontsize=9.5, color=INK_PRIMARY,
        )
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        fig.tight_layout()
        strip_figure_titles(fig)
        fig.savefig(out_dir / f"compute_causal_scatter_{model}_{sd_label}.pdf", bbox_inches="tight")
        plt.close(fig)

    return out_path


def compute_cross_model_summary(
    models: Sequence[str], runs_root: Path, out_dir: Path, out_name: str, subdir: str = ""
) -> Optional[Path]:
    """Model-vs-model causal-stage cost, one grouped-bar panel per backend.

    Unlike ``compute_summary``, this holds the backend (and therefore the
    "how much work is one call" definition) fixed and only varies the model
    — a well-posed comparison free of the cross-backend granularity mismatch.
    Each (model, backend) bar is matched to that model's own matched task set
    (the intersection of tasks where every backend present *for that model*
    has a record) — task sets are allowed to differ across models; the n is
    annotated on every bar so this stays auditable.
    """
    sd_label = subdir or "core"
    per_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_model_matched_tasks: Dict[str, set] = {}
    for model in models:
        by_tb, _ = _collect_task_backend_cost(model, runs_root, subdir, phase="causal")
        if not by_tb:
            continue
        backends = sorted({b for (_t, b) in by_tb})
        tasks_by_backend = {b: {t for (t, bb) in by_tb if bb == b} for b in backends}
        matched = set.intersection(*tasks_by_backend.values()) if tasks_by_backend else set()
        if not matched:
            continue
        per_model_matched_tasks[model] = matched
        stats_by_backend: Dict[str, Dict[str, Any]] = {}
        for b in backends:
            secs = [by_tb[(t, b)]["seconds"] for t in matched]
            deltas = [by_tb[(t, b)]["peak_delta_gb"] for t in matched if by_tb[(t, b)]["has_delta"]]
            has_units = all(by_tb[(t, b)]["has_units"] for t in matched)
            total_units = sum(by_tb[(t, b)]["n_strength_points"] for t in matched)
            stats_by_backend[b] = {
                "mean_seconds": sum(secs) / len(secs),
                "mean_peak_delta_gb": (sum(deltas) / len(deltas)) if deltas else None,
                "seconds_per_unit": (sum(secs) / total_units) if (has_units and total_units > 0) else None,
                "n_tasks": len(matched),
            }
        per_model[model] = stats_by_backend

    if not per_model:
        print(f"compute cross-model summary: no matched causal-phase data for any of {models} / {sd_label}", file=sys.stderr)
        return None

    all_backends = sorted({b for stats in per_model.values() for b in stats})
    present_models = [m for m in models if m in per_model]
    use_units = all(
        per_model[m].get(b, {}).get("seconds_per_unit") is not None
        for m in present_models for b in all_backends if b in per_model[m]
    )

    lines = [
        f"Cross-model causal-stage compute cost — subdir={sd_label}",
        "(each model's bar uses that model's OWN matched task set — task sets can",
        " differ across models; see n_tasks per bar. This compares model-vs-model",
        " for a fixed backend, which does not have the cross-backend call-",
        " granularity mismatch that compute_summary's backend comparison has.",
        f" seconds_per_unit available for all bars: {use_units} — see compute_summary's",
        " module docstring for what that field means and why it's preferred.)",
        "",
    ]
    for model in present_models:
        lines.append(f"[{model}] matched_tasks(n={len(per_model_matched_tasks[model])})="
                      f"{sorted(per_model_matched_tasks[model])}")
        for b in all_backends:
            stats = per_model[model].get(b)
            if stats is None:
                lines.append(f"    {b:<12} (no record for this backend on this model/subdir)")
                continue
            d = stats["mean_peak_delta_gb"]
            spu = stats["seconds_per_unit"]
            d_txt = f"{d:.2f}" if d is not None else "n/a"
            spu_txt = f"{spu:.3f}" if spu is not None else "n/a"
            lines.append(
                f"    {b:<12} mean_seconds={stats['mean_seconds']:.1f} seconds_per_unit={spu_txt} "
                f"mean_peak_delta_gb={d_txt} (n_tasks={stats['n_tasks']})"
            )
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compute_summary_cross_model_{out_name}_{sd_label}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    n_models = len(present_models)
    width = 0.8 / max(n_models, 1)
    x_base = list(range(len(all_backends)))
    for i, model in enumerate(present_models):
        xs = [x + (i - (n_models - 1) / 2) * width for x in x_base]
        if use_units:
            secs = [per_model[model].get(b, {}).get("seconds_per_unit") for b in all_backends]
        else:
            secs = [per_model[model].get(b, {}).get("mean_seconds") for b in all_backends]
        deltas = [per_model[model].get(b, {}).get("mean_peak_delta_gb") for b in all_backends]
        ns = [per_model[model].get(b, {}).get("n_tasks") for b in all_backends]
        ax_s.bar(xs, [s if s is not None else 0 for s in secs], width=width, label=model, zorder=3)
        ax_g.bar(xs, [d if d is not None else 0 for d in deltas], width=width, label=model, zorder=3)
        for x, s, n in zip(xs, secs, ns):
            if s is not None:
                ax_s.annotate(f"n={n}", (x, s), textcoords="offset points", xytext=(0, 2),
                               ha="center", fontsize=6.5, color=INK_MUTED)
    s_label = "Seconds per (class, strength-point)" if use_units else "Mean seconds per task [NOT unit-normalized]"
    for ax, ylabel, title in (
        (ax_s, s_label, "Wall-clock"),
        (ax_g, "Mean peak GPU growth (GB)", "GPU memory (Δ vs. entry)"),
    ):
        ax.set_xticks(x_base)
        ax.set_xticklabels(all_backends, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK_SECONDARY)
        ax.set_title(title, fontsize=9.5, color=INK_PRIMARY)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax_s.legend(fontsize=7.5, frameon=False)
    fig.suptitle(
        f"Causal-stage compute cost across models — subdir={sd_label}\n"
        "each backend's own matched task set per model; bars annotated with n tasks",
        fontsize=9.5, color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    strip_figure_titles(fig)
    fig.savefig(out_dir / f"compute_cross_model_{out_name}_{sd_label}.pdf", bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument(
        "--subdir", default="",
        help="OUTPUT_SUBDIR tree to read (default: core runs/{model}/ tree; pass "
        "'suite_full' etc. explicitly — nothing in this script pools/merges subdirs "
        "on your behalf, including compute_summary/compute_cross_model_summary below). "
        "The suitability scatter and one-pole/two-pole chart need dense "
        "causal_signed_effect coverage that, as of 2026-08-19, mostly only exists in "
        "core — pointing them at suite_full just makes them sparser (see CLAUDE.md).",
    )
    parser.add_argument(
        "--sae-select-subdir", default="suite_full",
        help="OUTPUT_SUBDIR tree to read the sel_arad_out/sel_jh_f1/sel_opp_fire ablations from "
        "(they were never run under --suite core; empty string reads the default tree instead)",
    )
    parser.add_argument(
        "--cross-model-models", default="",
        help="Comma-separated model keys for compute_cross_model_summary (e.g. "
        "'gpt2-small,pythia-70m-deduped'). Empty (default) skips the cross-model chart "
        "— it's a separate, multi-model output, not part of the per-model --model run.",
    )
    args = parser.parse_args()

    suffix = args.model + (f"_{args.subdir}" if args.subdir else "")
    out_dir = args.out / f"paper_{suffix}"
    written: List[Path] = []

    p = suitability_scatter(args.model, args.runs, args.subdir, out_dir)
    if p:
        written.append(p)

    p = one_pole_two_pole_chart(args.model, args.runs, args.subdir, out_dir)
    if p:
        written.append(p)

    written.extend(sae_selection_heatmap(args.model, args.runs, args.sae_select_subdir, out_dir))

    p = compute_summary(args.model, args.runs, out_dir, args.subdir)
    if p:
        written.append(p)

    if args.cross_model_models:
        models = [m.strip() for m in args.cross_model_models.split(",") if m.strip()]
        cross_out_dir = args.out / "paper_cross_model"
        p = compute_cross_model_summary(models, args.runs, cross_out_dir, "_".join(models), args.subdir)
        if p:
            written.append(p)

    for p in written:
        print(f"Wrote {p}")
    if not written:
        print("Nothing written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
