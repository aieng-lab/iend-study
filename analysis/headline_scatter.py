#!/usr/bin/env python
"""Detection-versus-Intervention scatter plots for paper method groups.

Writes a two-panel figure for per-model centroids (method color, model marker)
and a two-panel figure of model-balanced method aggregates. Only complete raw
Detection and Intervention rows enter either axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.method_statistics import (  # noqa: E402
    METHOD_DISPLAY,
    load_source_frames,
    parse_sources,
)
from analysis.summary_latex import (  # noqa: E402
    DEFAULT_CROSS_MODEL_SOURCES,
    METHOD_LATEX,
    MODEL_LATEX,
    MODEL_PARAMS,
    headline_method_groups,
    PAPER_ONE_POLE_TASKS,
    PAPER_PAIRWISE_TASKS,
)
from analysis.plot_style import (  # noqa: E402
    ESTIMATOR_LABELS,
    HOLLOW_EDGE_WIDTH,
    SAE_MARKER,
    SAE_VARIANT_COLORS,
    SIGNAL_LABELS,
    SIGNAL_MARKERS,
    SIGNAL_PALETTE,
    method_color,
    method_science,
    scatter_style,
    strip_figure_titles,
)
from analysis.task_specs import group_is_applicable, spec_for_task  # noqa: E402


DEFAULT_OUT = ROOT / "analysis" / "figures"
DEFAULT_RUNS = ROOT / "runs"
SHARED_SAE_METHODS: Tuple[str, ...] = (
    "sae:k1",
)
PLOT_LABELS: Dict[str, str] = {
    **METHOD_DISPLAY,
    **METHOD_LATEX,
    # The headline compares exactly one SAE recipe, so name the method rather
    # than its fixed-k implementation detail.  Appendix plots retain k=1.
    "sae:k1": r"$\mathrm{SAE}$",
    "sae:kstar": r"$\mathrm{SAE}^{k^{*}}$",
    "sae_pre:k1": r"$\mathrm{SAE}_{\mathrm{pre}}^{k=1}$",
    "sae_pre:kstar": r"$\mathrm{SAE}_{\mathrm{pre}}^{k^{*}}$",
}

def collect_complete_headline_rows(
    frames: Mapping[str, pd.DataFrame],
    *,
    method_groups: Sequence[str] = (),
    detection_metric: str = "detection_score",
    tasks: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        if frame.empty:
            continue
        needed = {
            "task",
            "method_group",
            "intervention_score",
            "detection_complete",
            "intervention_complete",
        }
        if detection_metric == "detection_light_score":
            needed.update({"roc_auc_neutral", "neutral_specificity"})
        else:
            needed.add(detection_metric)
        if not needed.issubset(frame.columns):
            continue
        selected = frame[
            frame["detection_complete"].eq(True)  # noqa: E712
            & frame["intervention_complete"].eq(True)  # noqa: E712
        ].copy()
        if tasks is not None:
            selected = selected[selected["task"].isin(set(tasks))]
        if method_groups:
            selected = selected[selected["method_group"].isin(set(method_groups))]
        if detection_metric == "detection_light_score":
            # The neutral-only secondary estimand is derived, not persisted in
            # summary_merged CSVs.  Requiring a physical column made every
            # light/full-ablation scatter falsely empty.
            selected["detection_score"] = pd.concat(
                [
                    pd.to_numeric(selected["roc_auc_neutral"], errors="coerce"),
                    pd.to_numeric(selected["neutral_specificity"], errors="coerce"),
                ],
                axis=1,
            ).min(axis=1, skipna=False)
        else:
            selected["detection_score"] = pd.to_numeric(
                selected[detection_metric], errors="coerce"
            )
        selected["intervention_score"] = pd.to_numeric(
            selected["intervention_score"], errors="coerce"
        )
        selected = selected.dropna(subset=["detection_score", "intervention_score"])
        applicable = pd.Series(
            [
                group_is_applicable(
                    str(row["method_group"]),
                    spec_for_task(str(row["task"])),
                    metric="detection_score",
                )
                for row in selected.to_dict(orient="records")
            ],
            index=selected.index,
            dtype=bool,
        )
        selected = selected.loc[applicable]
        selected["model"] = model
        rows.append(
            selected[
                ["model", "task", "method_group", "detection_score", "intervention_score"]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "task",
                "method_group",
                "detection_score",
                "intervention_score",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_csv_frames(
    csv_sources: Sequence[Tuple[str, Path]],
) -> Dict[str, pd.DataFrame]:
    """Load the exact canonical per-model CSVs used by summary tables."""
    frames: Dict[str, pd.DataFrame] = {}
    for model, path in csv_sources:
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            frame["model"] = model
        frames[model] = frame
    return frames


TASK_SCOPES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    # Pairwise-capable tasks are the proper ablation comparison: show both the
    # pairwise construction and its one-sided counterpart on identical tasks.
    "non_one_side_tasks": (
        PAPER_PAIRWISE_TASKS,
        ("pairwise", "one_pole"),
    ),
    # Intrinsically one-sided tasks cannot support pairwise methods.
    "one_side_tasks": (
        PAPER_ONE_POLE_TASKS,
        ("one_pole",),
    ),
}


def build_task_scope_tables(
    rows: pd.DataFrame, *, scope: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate complete rows within one scientifically valid task scope."""
    if scope not in TASK_SCOPES:
        raise ValueError(f"unknown task scope {scope!r}")
    tasks, constructions = TASK_SCOPES[scope]
    subset = rows[rows["task"].isin(tasks)].copy()
    subset["construction"] = subset["method_group"].map(_construction)
    subset = subset[subset["construction"].isin(constructions)]
    per_model = (
        subset.groupby(["model", "method_group", "construction"], as_index=False)
        .agg(
            detection_score=("detection_score", "mean"),
            intervention_score=("intervention_score", "mean"),
            n_tasks=("task", "nunique"),
        )
    )
    aggregate = (
        per_model.groupby(["method_group", "construction"], as_index=False)
        .agg(
            detection_score=("detection_score", "mean"),
            intervention_score=("intervention_score", "mean"),
            n_models=("model", "nunique"),
            n_task_model_cells=("n_tasks", "sum"),
        )
    )
    return per_model, aggregate


def _color_keys(methods: Sequence[str]) -> Dict[str, str]:
    """Return the shared palette key for each concrete method group.

    The canonical CSVs contain concrete groups such as ``sae_pre:k1`` while
    the palette is keyed by scientific family (``sae``).  Keep this
    normalization in one place so intervention-only baseline rows cannot
    accidentally be looked up as literal palette keys.
    """
    out: Dict[str, str] = {}
    for method in methods:
        backend = method.partition(":")[0]
        out[method] = {
            "cga_tensor_norm": "cga",
            "actiend_ridge": "actiend",
            "actiend_pre": "actiend",
            "sae_pre": "sae",
        }.get(backend, "sae" if backend == "sae" else backend)
    return out


def _construction(method: str) -> str:
    if method in SHARED_SAE_METHODS:
        return "one_pole"
    return "one_pole" if method.endswith(":one_pole") else "pairwise"


def _plot_label(method: str, *, include_construction: bool = True) -> str:
    label = PLOT_LABELS.get(method, method)
    if include_construction:
        return label
    return (
        label.replace(r"$_{\mathrm{pw}}$", "")
        .replace(r"$_{\mathrm{1s}}$", "")
        .replace(",pw}", "}")
        .replace(",1s}", "}")
    )


def _adjust_labels(axis, texts, points) -> None:
    """Repel annotations using adjustText instead of hand-tuned offsets."""
    if not texts:
        return
    from adjustText import adjust_text

    adjust_text(
        texts,
        x=[point[0] for point in points],
        y=[point[1] for point in points],
        ax=axis,
        expand=(1.08, 1.18),
        force_text=(0.35, 0.55),
        force_static=(0.18, 0.28),
        ensure_inside_axes=True,
        min_arrow_len=12,
        iter_lim=300,
        arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.45, "alpha": 0.7},
    )


def _legend_handles(*, sae_variants: bool = False) -> Dict[str, list]:
    from matplotlib.lines import Line2D

    construction_labels = {"pairwise": "Pairwise", "one_pole": "One-sided"}
    signal_handles = [
        Line2D(
            [0], [0], marker=SIGNAL_MARKERS[signal], linestyle="none",
            markerfacecolor=SIGNAL_PALETTE[signal]["base"],
            markeredgecolor=SIGNAL_PALETTE[signal]["base"],
            label=label, markersize=7,
        )
        for signal, label in SIGNAL_LABELS.items()
    ]
    estimator_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=("0.30" if estimator == "iend" else "0.70"),
            markeredgecolor=("0.30" if estimator == "iend" else "0.70"),
            label=label, markersize=7,
        )
        for estimator, label in ESTIMATOR_LABELS.items()
    ]
    construction_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="0.45" if construction == "pairwise" else "none",
            markeredgecolor="0.45", label=construction_labels[construction], markersize=7,
            markeredgewidth=1.15 if construction == "pairwise" else HOLLOW_EDGE_WIDTH,
        )
        for construction in ("pairwise", "one_pole")
    ]
    variants = ("k1", "kstar") if sae_variants else ("k1",)
    sae_handles = [
        Line2D(
            [0], [0], marker=SAE_MARKER, linestyle="none", markerfacecolor="none",
            markeredgecolor=SAE_VARIANT_COLORS[variant],
            label=(PLOT_LABELS[f"sae:{variant}"] if sae_variants else r"$\mathrm{SAE}$"),
            markersize=7, markeredgewidth=HOLLOW_EDGE_WIDTH,
        )
        for variant in variants
    ]

    return {
        "signal": signal_handles,
        "estimator": estimator_handles,
        "construction": construction_handles,
        "sae": sae_handles,
    }


# Line shade per estimator in signal-axis mode; matches the Estimator legend.
_ESTIMATOR_LINE_COLOR = {"contrastive_mean": "0.70", "iend": "0.30"}
# One primary method per (signal, estimator); variants (ridge, tensor-norm)
# would otherwise fork the signal-axis path.
_SIGNAL_PATH_BACKENDS = {"caa", "caga", "cga", "actiend", "agiend", "gradiend"}


def _draw_panel(
    axis, records, *, include_model: bool = False,
    include_construction: bool = False,
    connect: str = "estimator",
    annotation_fontsize: float = 7.5,
    tick_labelsize: float | None = None,
    axis_labelsize: float | None = None,
    y_tick_decimals: int | None = None,
) -> None:
    """``connect="estimator"`` links each contrastive mean to its IEND within a
    signal; ``connect="signal"`` links activation value -> activation gradient
    -> parameter gradient within an estimator (arrows show the direction);
    ``connect="construction"`` links each method's pairwise point to its
    one-sided point, so the line is that method's construction trade-off."""
    texts = []
    points = []
    items = list(records)
    if connect == "construction":
        by_backend: Dict[str, Dict[str, dict]] = {}
        for r in items:
            backend = str(r["method_group"]).partition(":")[0]
            by_backend.setdefault(backend, {})[str(r["construction"])] = r
        for pair in by_backend.values():
            if "pairwise" in pair and "one_pole" in pair:
                a, b = pair["pairwise"], pair["one_pole"]
                axis.plot(
                    [a["detection_score"], b["detection_score"]],
                    [a["intervention_score"], b["intervention_score"]],
                    color=method_color(str(a["method_group"])), linewidth=1.6,
                    alpha=0.75, zorder=1,
                )
    for construction in ("pairwise", "one_pole"):
        in_construction = [r for r in items if str(r["construction"]) == construction]
        if connect == "construction":
            continue
        if connect == "estimator":
            for signal in SIGNAL_LABELS:
                by_estimator = {
                    method_science(str(r["method_group"]))[1]: r
                    for r in in_construction
                    if method_science(str(r["method_group"]))[0] == signal
                }
                if "contrastive_mean" in by_estimator and "iend" in by_estimator:
                    a, b = by_estimator["contrastive_mean"], by_estimator["iend"]
                    axis.plot(
                        [a["detection_score"], b["detection_score"]],
                        [a["intervention_score"], b["intervention_score"]],
                        color=SIGNAL_PALETTE[signal]["base"], linewidth=0.9,
                        alpha=0.55, zorder=1,
                    )
        elif connect == "signal":
            for estimator, color in _ESTIMATOR_LINE_COLOR.items():
                by_signal = {
                    method_science(str(r["method_group"]))[0]: r
                    for r in in_construction
                    if str(r["method_group"]).partition(":")[0] in _SIGNAL_PATH_BACKENDS
                    and method_science(str(r["method_group"]))[1] == estimator
                }
                path = [by_signal[s] for s in SIGNAL_LABELS if s in by_signal]
                for a, b in zip(path, path[1:]):
                    axis.annotate(
                        "", xy=(b["detection_score"], b["intervention_score"]),
                        xytext=(a["detection_score"], a["intervention_score"]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0,
                                    "shrinkA": 6, "shrinkB": 6},
                        zorder=1,
                    )
        else:
            raise ValueError(
                f"connect must be 'estimator', 'signal' or 'construction', got {connect!r}"
            )
    for record in items:
        method = str(record["method_group"])
        construction = str(record["construction"])
        x, y = float(record["detection_score"]), float(record["intervention_score"])
        axis.scatter(x, y, s=94, zorder=3, **scatter_style(method, construction))
        label = _plot_label(method, include_construction=include_construction)
        if include_model:
            label += " · " + MODEL_LATEX.get(str(record["model"]), str(record["model"]))
        texts.append(axis.text(x, y, label, fontsize=annotation_fontsize))
        points.append((x, y))
    _adjust_labels(axis, texts, points)
    axis.set_xlabel("Detection (Det.)", fontsize=axis_labelsize)
    axis.set_ylabel("Intervention (Int.)", fontsize=axis_labelsize)
    if tick_labelsize is not None:
        axis.tick_params(axis="both", labelsize=tick_labelsize)
    if y_tick_decimals is not None:
        from matplotlib.ticker import FormatStrFormatter
        axis.yaxis.set_major_formatter(FormatStrFormatter(f"%.{y_tick_decimals}f"))
    axis.grid(alpha=0.2)


def _add_dimension_legends(
    figure, *, where: str, sae_variants: bool = False, include_sae: bool = True,
    fontsize: float = 8.5, title_fontsize: float = 8.5,
) -> None:
    """One framed, titled legend per encoding dimension.

    ``where="top"``: legends side by side above the axes, each laid out in one
    row. ``where="right"``: legends stacked top-to-bottom right of the axes.
    Boxes are measured after a draw so they abut with a fixed gap regardless
    of label widths.
    """
    handles = _legend_handles(sae_variants=sae_variants)
    groups = (
        ("Signal", handles["signal"]),
        ("Estimator", handles["estimator"]),
        ("Construction", handles["construction"]),
        *((("SAE", handles["sae"]),) if include_sae else ()),
    )
    top = where == "top"
    legends = [
        figure.legend(
            handles=group, title=title, frameon=True, fontsize=fontsize,
            title_fontsize=title_fontsize,
            ncol=len(group) if top else 1, loc="lower left" if top else "upper left",
        )
        for title, group in groups
    ]
    renderer = figure.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in legends]
    width, height = figure.get_size_inches() * figure.dpi
    gap = 8.0
    if top:
        x = (width - sum(b.width for b in boxes) - gap * (len(boxes) - 1)) / 2
        for legend, box in zip(legends, boxes):
            legend.set_bbox_to_anchor((x / width, 1.0), transform=figure.transFigure)
            x += box.width + gap
    else:
        # Align the top legend to the panel's top spine rather than vertically
        # centering the legend stack in the whole figure.
        y = figure.axes[0].get_position().y1
        for legend, box in zip(legends, boxes):
            legend.set_bbox_to_anchor((1.0, y), transform=figure.transFigure)
            y -= (box.height + gap) / height


def pooled_cell_means(table_csvs: Sequence[Tuple[str, Path]]) -> pd.DataFrame:
    """Per method: mean over all (model, task) cells, pooled across models.

    Reads ``summary_matrix_{detection,intervention}_<model>.csv`` next to each
    ``summary_methods_<model>.csv``: the task x method matrices whose column
    means are that table's Det./Int. So each model contributes every cell the
    Summary table averages, and a model with few valid tasks for a method
    weighs proportionally less instead of equal to a complete one. Each metric
    is averaged over its own non-missing cells, as the table does per model.
    """
    long = []
    for model, csv_path in table_csvs:
        for metric, column in (("detection", "detection_score"), ("intervention", "intervention_score")):
            matrix = pd.read_csv(
                Path(csv_path).with_name(f"summary_matrix_{metric}_{model}.csv"), index_col=0
            )
            summary_labels = ("Mean", "Median")
            matrix = matrix.drop(index=[i for i in matrix.index if str(i) in summary_labels])
            matrix = matrix.drop(columns=[c for c in matrix.columns if str(c) in summary_labels])
            cells = matrix.apply(pd.to_numeric, errors="coerce").stack().rename("value").reset_index()
            cells.columns = ["task", "method_group", "value"]
            cells = cells.dropna(subset=["value"])
            long.append(cells.assign(model=model, metric=column))
    cells = pd.concat(long, ignore_index=True)
    stats = cells.groupby(["method_group", "metric"])["value"].agg(["mean", "count"]).unstack("metric")
    out = pd.DataFrame({
        "method_group": stats.index,
        "detection_score": stats[("mean", "detection_score")].values,
        "intervention_score": stats[("mean", "intervention_score")].values,
        "n_detection_cells": stats[("count", "detection_score")].values,
        "n_intervention_cells": stats[("count", "intervention_score")].values,
    })
    models = cells.groupby("method_group")["model"].nunique()
    out["n_models"] = out["method_group"].map(models)
    out["construction"] = out["method_group"].map(_construction)
    return out.dropna(subset=["detection_score", "intervention_score"]).reset_index(drop=True)


def _shared_by_model_path(by_model_path: Path) -> Path:
    return by_model_path.with_name(f"{by_model_path.stem}_shared{by_model_path.suffix}")


def _save_by_model_figures(
    panels: Sequence[Tuple[str, Sequence[dict]]],
    by_model_path: Path,
    *,
    connect: str = "estimator",
    include_construction: bool = True,
) -> List[Path]:
    """Write the 1xN by-model grid twice: per-panel axes, then shared x/y axes.

    The ``_shared`` variant fixes one Detection range and one Intervention
    range across all base models (limits are set before drawing so label
    placement sees the final scale), making positions comparable across
    panels.  ``panels`` is ``[(model, records)]``.
    """
    import matplotlib.pyplot as plt

    written: List[Path] = []
    for shared in (False, True):
        figure, axes = plt.subplots(
            1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False,
            sharex=shared, sharey=shared,
        )
        if shared:
            xs = [float(r["detection_score"]) for _m, recs in panels for r in recs]
            ys = [float(r["intervention_score"]) for _m, recs in panels for r in recs]
            if xs and ys:
                pad_x = 0.05 * ((max(xs) - min(xs)) or 1.0)
                pad_y = 0.05 * ((max(ys) - min(ys)) or 1.0)
                axes[0][0].set_xlim(min(xs) - pad_x, max(xs) + pad_x)
                axes[0][0].set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        for index, (axis, (model, records)) in enumerate(zip(axes[0], panels)):
            _draw_panel(
                axis, list(records), include_construction=include_construction,
                connect=connect, annotation_fontsize=11.5,
                tick_labelsize=16, axis_labelsize=18, y_tick_decimals=2,
            )
            axis.set_title(MODEL_LATEX.get(str(model), str(model)), fontsize=19)
            if shared and index:
                axis.set_ylabel("")
        figure.tight_layout()
        _add_dimension_legends(figure, where="top", fontsize=15, title_fontsize=15)
        path = _shared_by_model_path(by_model_path) if shared else by_model_path
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def write_headline_scatter(
    table_csvs: Sequence[Tuple[str, Path]],
    path: Path,
    *,
    connect: str = "estimator",
    write_by_model: bool = True,
) -> List[Path]:
    """The paper's main scatters, from the Summary table of ``summary_tables.pdf``.

    Each point is a method's Det./Int. mean over every eligible task--model
    cell in the selected Summary matrices.  This pooled all-cell figure is
    written to ``path``: it is the paper's primary headline scatter.  The
    model-faceted diagnostic follows as ``<stem>_by_model``.  Pooled values
    and constituent-cell counts are written to ``<stem>.csv``.
    """
    import matplotlib.pyplot as plt

    mean = pooled_cell_means(table_csvs)
    # The pooled figures are paper-facing; give their labels substantially more
    # room than the dense by-model diagnostic grid.
    figure, axis = plt.subplots(figsize=(14.4, 5.4))
    _draw_panel(
        axis, mean.to_dict(orient="records"), include_construction=True, connect=connect,
        annotation_fontsize=17.5, tick_labelsize=22, axis_labelsize=26,
        y_tick_decimals=2,
    )
    figure.tight_layout()
    _add_dimension_legends(figure, where="right", fontsize=17.5, title_fontsize=17.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    mean.to_csv(path.with_suffix(".csv"), index=False, float_format="%.6f")
    if not write_by_model:
        return [path]

    panels = []
    for model, csv_path in table_csvs:
        frame = pd.read_csv(csv_path).dropna(subset=["detection_score", "intervention_score"])
        frame["construction"] = frame["method_group"].map(_construction)
        panels.append((model, frame))
    panels.sort(key=lambda item: MODEL_PARAMS.get(str(item[0]), float("inf")))
    by_model_path = path.with_name(f"{path.stem}_by_model{path.suffix}")
    by_model = _save_by_model_figures(
        [(model, frame.to_dict(orient="records")) for model, frame in panels],
        by_model_path, connect=connect,
    )
    return [path, *by_model]


def write_headline_scatter_from_rows(
    rows: pd.DataFrame,
    path: Path,
    *,
    sources: Sequence[Tuple[str, str]],
    connect: str = "estimator",
) -> List[Path]:
    """Write the pooled and faceted headline pair from complete cell rows.

    Unlike :func:`write_headline_scatter`, which reads the ordinary Summary
    matrices, this is for a representation ablation whose values come from a
    separately collected view.  It intentionally preserves the primary
    headline layout: a pooled all-cell figure plus a 1xN by-model grid.
    """
    import matplotlib.pyplot as plt

    per_model = (
        rows.groupby(["model", "method_group"], as_index=False)
        .agg(
            detection_score=("detection_score", "mean"),
            intervention_score=("intervention_score", "mean"),
            n_tasks=("task", "nunique"),
        )
    )
    per_model["construction"] = per_model["method_group"].map(_construction)
    mean = (
        rows.groupby("method_group", as_index=False)
        .agg(
            detection_score=("detection_score", "mean"),
            intervention_score=("intervention_score", "mean"),
            n_models=("model", "nunique"),
            n_task_model_cells=("task", "size"),
        )
    )
    mean["construction"] = mean["method_group"].map(_construction)

    figure, axis = plt.subplots(figsize=(14.4, 5.4))
    _draw_panel(
        axis, mean.to_dict(orient="records"), include_construction=True, connect=connect,
        annotation_fontsize=17.5, tick_labelsize=22, axis_labelsize=26,
        y_tick_decimals=2,
    )
    figure.tight_layout()
    _add_dimension_legends(figure, where="right", fontsize=17.5, title_fontsize=17.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    mean.to_csv(path.with_suffix(".csv"), index=False, float_format="%.6f")

    panels = [
        (model, per_model[per_model["model"].eq(model)])
        for model, _subdir in sources
        if not per_model[per_model["model"].eq(model)].empty
    ]
    panels.sort(key=lambda item: MODEL_PARAMS.get(str(item[0]), float("inf")))
    by_model_path = path.with_name(f"{path.stem}_by_model{path.suffix}")
    by_model = _save_by_model_figures(
        [(model, frame.to_dict(orient="records")) for model, frame in panels],
        by_model_path, connect=connect,
    )
    return [path, *by_model]


def construction_tradeoff_records(matched: pd.DataFrame) -> pd.DataFrame:
    """Pool matched pairwise / one-sided rows into one point per method x construction.

    ``matched`` needs ``plot_family``, ``pole`` (``two_pole``/``one_pole``),
    ``method_group``, ``detection`` and ``intervention``.  The caller must have
    restricted it to pairwise-capable tasks and to model--task cells where the
    method supplied both constructions; otherwise the two ends of a line would
    average different task sets and the line would not be a trade-off.
    """
    frame = matched[matched["pole"].isin(["two_pole", "one_pole"])].copy()
    frame["construction"] = frame["pole"].map({"two_pole": "pairwise", "one_pole": "one_pole"})
    return (
        frame.groupby(["plot_family", "construction"], as_index=False)
        .agg(
            method_group=("method_group", lambda s: s.mode().iloc[0]),
            detection_score=("detection", "mean"),
            intervention_score=("intervention", "mean"),
            n_cells=("task", "size"),
        )
        .dropna(subset=["detection_score", "intervention_score"])
    )


def write_construction_tradeoff_scatter(matched: pd.DataFrame, path: Path) -> Optional[Path]:
    """Detection-vs-Intervention scatter, one line per method from its pairwise
    point to its one-sided point (same encodings as the headline scatter)."""
    import matplotlib.pyplot as plt

    records = construction_tradeoff_records(matched)
    if records.empty:
        return None
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    _draw_panel(
        axis, records.to_dict(orient="records"), include_construction=True,
        connect="construction", annotation_fontsize=11, tick_labelsize=13,
        axis_labelsize=15, y_tick_decimals=2,
    )
    figure.tight_layout()
    _add_dimension_legends(
        figure, where="right", include_sae=False, fontsize=11, title_fontsize=11
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    records.to_csv(path.with_suffix(".csv"), index=False, float_format="%.6f")
    return path


def write_scatter_plots(
    rows: pd.DataFrame,
    out_dir: Path,
    *,
    sources: Sequence[Tuple[str, str]],
    name_suffix: str = "",
) -> Sequence[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        scope: build_task_scope_tables(rows, scope=scope) for scope in TASK_SCOPES
    }
    written: list[Path] = []

    handles = _legend_handles()
    signal_handles = handles["signal"]
    estimator_handles = handles["estimator"]
    construction_handles = handles["construction"]
    sae_handles = handles["sae"]
    draw = _draw_panel

    def add_legends(figure, *, show_construction: bool) -> None:
        figure.legend(handles=signal_handles, title="Signal", loc="center left",
                      bbox_to_anchor=(0.81, 0.78), frameon=False, fontsize=8,
                      title_fontsize=8)
        figure.legend(handles=estimator_handles, title="Estimator shade",
                      loc="center left", bbox_to_anchor=(0.81, 0.53),
                      frameon=False, fontsize=8, title_fontsize=8)
        if show_construction:
            figure.legend(handles=construction_handles, title="Construction",
                          loc="center left", bbox_to_anchor=(0.81, 0.28),
                          frameon=False, fontsize=8, title_fontsize=8)
        figure.legend(handles=sae_handles, loc="center left",
                      bbox_to_anchor=(0.81, 0.10), frameon=False, fontsize=8)

    for scope, (per_model, aggregate) in tables.items():
        if not per_model.empty:
            # Keep the task-scope diagnostic structurally identical to the
            # headline ``*_by_model`` figure.  This scope is an input filter,
            # not a different scientific view: each base model needs its own
            # panel, and the shared dimension legends belong above that grid.
            # The former single-axis layout mixed models into point labels and
            # used a separate legend implementation, making a simple task-set
            # restriction look like a different figure family.
            panels = [
                (model, per_model[per_model["model"].eq(model)])
                for model, _subdir in sources
                if not per_model[per_model["model"].eq(model)].empty
            ]
            panels.sort(key=lambda item: MODEL_PARAMS.get(str(item[0]), float("inf")))
            path = out_dir / f"headline_scatter_{scope}_by_model{name_suffix}.pdf"
            written.extend(_save_by_model_figures(
                [(model, table.to_dict(orient="records")) for model, table in panels],
                path, include_construction=scope == "non_one_side_tasks",
            ))

        if not aggregate.empty:
            figure, axis = plt.subplots(figsize=(10.8, 6.4))
            draw(
                axis, aggregate.to_dict(orient="records"),
                include_construction=scope == "non_one_side_tasks",
            )
            add_legends(figure, show_construction=scope == "non_one_side_tasks")
            figure.tight_layout(rect=(0, 0, 0.79, 1))
            path = out_dir / f"headline_scatter_{scope}_model_balanced{name_suffix}.pdf"
            strip_figure_titles(figure)
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            written.append(path)

        for model, _subdir in sources:
            model_table = per_model[per_model["model"].eq(model)]
            if model_table.empty:
                continue
            figure, axis = plt.subplots(figsize=(10.8, 6.4))
            draw(
                axis, model_table.to_dict(orient="records"),
                include_construction=scope == "non_one_side_tasks",
            )
            add_legends(figure, show_construction=scope == "non_one_side_tasks")
            figure.tight_layout(rect=(0, 0, 0.79, 1))
            path = out_dir / f"headline_scatter_{model}_{scope}{name_suffix}.pdf"
            strip_figure_titles(figure)
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            written.append(path)

        per_model_path = out_dir / f"headline_scatter_{scope}_by_model{name_suffix}.csv"
        aggregate_path = out_dir / f"headline_scatter_{scope}_model_balanced{name_suffix}.csv"
        per_model.to_csv(per_model_path, index=False, float_format="%.8f")
        aggregate.to_csv(aggregate_path, index=False, float_format="%.8f")
        written.extend([per_model_path, aggregate_path])
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="MODEL[=OUTPUT_SUBDIR]",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        metavar="MODEL=PATH",
        help="Use the per-model summary_merged CSV written by summary_latex.py.",
    )
    parser.add_argument(
        "--table-csv",
        action="append",
        default=None,
        metavar="MODEL=PATH",
        help="Per-model summary_methods CSV (the Summary table of summary_tables.pdf); "
        "writes the main figure headline_scatter.pdf from exactly these values.",
    )
    parser.add_argument(
        "--completion-manifest",
        type=Path,
        default=None,
        help="Restrict sources to the model set recorded by a headline coverage manifest.",
    )
    parser.add_argument(
        "--report-mode",
        choices=("essential", "full"),
        default="full",
        help="Artifact scope; full preserves the standalone historical scatter matrix.",
    )
    args = parser.parse_args()

    def model_paths(values, flag):
        out = []
        for value in values:
            model, sep, raw_path = value.partition("=")
            if not sep or not model or not raw_path:
                raise SystemExit(f"invalid {flag} value (expected MODEL=PATH): {value!r}")
            out.append((model, Path(raw_path)))
        return out

    sources = parse_sources(args.source) if args.source else DEFAULT_CROSS_MODEL_SOURCES
    suffix_prefix = ""
    if args.completion_manifest:
        manifest = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
        sources = tuple(
            (str(item["model"]), str(item.get("subdir") or ""))
            for item in manifest["sources"]
        )
        suffix_prefix = "_complete90"

    if args.table_csv:
        table_csvs = model_paths(args.table_csv, "--table-csv")
        allowed = {model for model, _subdir in sources}
        table_csvs = [(model, path) for model, path in table_csvs if model in allowed]
        # ``headline_scatter_signal_mean`` is the canonical signal-path
        # figure.  The former shorter ``headline_scatter_signal`` name was an
        # identical compatibility duplicate, which made it unclear which PDF
        # belonged in the paper and left one copy apparently stale.
        for name, connect in (
            ("headline_scatter", "estimator"),
            ("headline_scatter_signal_mean", "signal"),
        ):
            for path in write_headline_scatter(
                table_csvs, args.out / f"{name}{suffix_prefix}.pdf", connect=connect
            ):
                print(f"Wrote {path}")

    if args.csv:
        frames = load_csv_frames(model_paths(args.csv, "--csv"))
        frames = {model: frame for model, frame in frames.items() if model in {m for m, _ in sources}}
    else:
        frames = load_source_frames(args.runs, sources)

    combined = pd.concat(list(frames.values()), ignore_index=True) if frames else pd.DataFrame()
    settings = (("selected", "detection_score", None, suffix_prefix),)
    if args.report_mode == "full":
        settings += (
            ("selected", "detection_light_score", None, f"{suffix_prefix}_light"),
            ("full", "detection_score", None, f"{suffix_prefix}_full"),
            ("full", "detection_light_score", None, f"{suffix_prefix}_full_light"),
        )
    written = []
    for method_set, metric, task_scope, suffix in settings:
        methods = headline_method_groups(method_set, frame=combined)
        rows = collect_complete_headline_rows(
            frames,
            method_groups=methods,
            detection_metric=metric,
            tasks=task_scope,
        )
        written.extend(
            write_scatter_plots(rows, args.out, sources=sources, name_suffix=suffix)
        )

    # Representation-scope sensitivity figure.  This is deliberately not the
    # existing ``*_full`` family (which means every canonical *method* arm).
    # It holds the selected headline method set fixed while replacing the
    # layer-capable closed-form families with their predefined aggregate:
    # CAA all-activation, CGA/CAGA aggregate, and SAE all-layer k=1.  Learned
    # IEND rows have no representation alternative in the canonical collector
    # and therefore remain their ordinary full-scope estimates.  Read raw
    # results rather than ``--csv`` inputs: those CSVs materialize only the
    # headline-selected representation.
    #
    # The all-layer representation ablation is a first-class headline
    # comparison, so its primary estimand (``headline_scatter_all_layer.pdf``)
    # is written in every report mode.  Only the light task-scope sensitivity
    # variant is ``full``-only.
    all_layer_frames = load_source_frames(
        args.runs, sources, representation_view="all_layer"
    )
    all_layer_combined = (
        pd.concat(list(all_layer_frames.values()), ignore_index=True)
        if all_layer_frames else pd.DataFrame()
    )
    all_layer_methods = headline_method_groups("selected", frame=all_layer_combined)
    all_layer_settings = [("detection_score", "_all_layer")]
    if args.report_mode == "full":
        all_layer_settings.append(("detection_light_score", "_all_layer_light"))
    for metric, suffix in all_layer_settings:
        all_layer_rows = collect_complete_headline_rows(
            all_layer_frames,
            method_groups=all_layer_methods,
            detection_metric=metric,
        )
        # The primary estimand gets the same unscoped pooled and by-model
        # pair as the ordinary headline.
        if metric == "detection_score" and not all_layer_rows.empty:
            written.extend(
                write_headline_scatter_from_rows(
                    all_layer_rows,
                    args.out / f"headline_scatter{suffix}.pdf",
                    sources=sources,
                )
            )
        written.extend(
            write_scatter_plots(
                all_layer_rows, args.out, sources=sources, name_suffix=suffix
            )
        )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
