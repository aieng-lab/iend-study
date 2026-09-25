#!/usr/bin/env python
"""Generate appendix comparisons for representation selection.

The three views use validation detection only:

* ``all_layer``: the predefined aggregate representation;
* ``best_single_layer``: the validation-best physical layer;
* ``best_overall``: the validation-best candidate among aggregate + layers.

Each emitted detection/intervention value is the held-out value attached to
that locked candidate.  Incomplete aggregate-only CGA/CAGA pools are omitted,
never silently treated as selected headline results.

Examples
--------
python analysis/layer_selection_appendix.py \
  --source gpt2-small=suite_full2 \
  --source pythia-70m-deduped=suite_full2 \
  --source llama-3.1-8b
(scripts/summary_tables_pdf.sh passes the sources of configs/report_model_sets.json.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.method_groups import (  # noqa: E402
    REPRESENTATION_VIEWS,
    _load_results,
    collect_class_rows_for_results,
    collect_group_rows_for_results,
)
from analysis.plot_style import method_color  # noqa: E402
from analysis.summary_latex import (  # noqa: E402
    METHOD_LATEX,
    MODEL_LATEX,
    MODEL_PARAMS,
    _is_hidden_task,
)

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "analysis" / "tables" / "layer_selection"
VIEWS = ("all_layer", "best_single_layer", "best_overall")
VIEW_LABELS = {
    "all_layer": "All-layer aggregate",
    "best_single_layer": "Best physical layer (validation Det.)",
    "best_overall": "Best of aggregate + layers (validation Det.)",
}
FAMILIES = ("sae", "caa", "cga", "caga")
CACHE_VERSION = 1
CLASS_CACHE_VERSION = 1
DISPLAY_GROUPS = (
    "sae:k1",
    "caa:one_pole",
    "caa:two_pole",
    "cga:one_pole",
    "cga:two_pole",
    "caga:one_pole",
    "caga:two_pole",
)


def _model_label(model: str) -> str:
    """Return the paper's model spelling, falling back safely for new models."""
    return MODEL_LATEX.get(model, model.replace("_", r"\_"))


def _method_label(group: str) -> str:
    """Return a Matplotlib-compatible version of the paper's LaTeX method name."""
    # The paper's SAE labels are macros defined only in the manuscript.  Use
    # their explicit LaTeX expansion here so this standalone PNG needs no TeX
    # installation while retaining the paper notation.
    if group == "sae:k1":
        return r"SAE$_{\mathrm{k=1}}$"
    return METHOD_LATEX.get(group, group.replace("_", r"\_"))


def _study_model_order(models: Iterable[str]) -> List[str]:
    """Return the paper's small-to-large base-model order."""
    return sorted(
        set(models), key=lambda model: (MODEL_PARAMS.get(model, float("inf")), model)
    )


def _configure_paper_font() -> None:
    """Use the bundled Times face used throughout paper-facing figures."""
    from matplotlib import font_manager

    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(paper_font)
        ).get_name()


def parse_sources(values: Iterable[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for value in values:
        model, sep, subdir = str(value).partition("=")
        model = model.strip()
        if not model:
            raise ValueError(f"invalid --source {value!r}")
        out.append((model, subdir.strip() if sep else ""))
    return out


def _cache_paths(cache_dir: Path, results_path: Path) -> Tuple[Path, Path]:
    """Stable per-results cache paths; never overwrite a source artifact."""
    digest = hashlib.sha256(str(results_path.resolve()).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{digest}.pkl", cache_dir / f"{digest}.json"


def _cached_rows_for_results(
    path: Path,
    *,
    cache_dir: Path,
    source_subdir: str,
) -> pd.DataFrame:
    """Materialize selection rows once per immutable ``results.json`` revision.

    The source dump can be hundreds of MB.  The cache holds only the compact
    candidate rows needed for the appendix, and is accepted only when the
    source's size and nanosecond mtime still match.  Thus a causal refresh
    naturally invalidates exactly its task cache without touching raw runs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _cache_paths(cache_dir, path)
    stat = path.stat()
    signature = {
        "cache_version": CACHE_VERSION,
        "source": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored == signature and data_path.is_file():
            print(f"cache hit {path}", flush=True)
            return pd.read_pickle(data_path)
    except (OSError, ValueError, EOFError):
        pass

    print(f"cache miss {path}; extracting compact appendix rows", flush=True)
    payload = _load_results(path)
    rows: List[Dict[str, object]] = []
    if payload:
        for view in VIEWS:
            for row in collect_group_rows_for_results(
                payload, results_path=str(path), representation_view=view
            ):
                if str(row.get("backend") or "") not in FAMILIES:
                    continue
                rows.append({**row, "selection_view": view, "source_subdir": source_subdir or None})
    frame = pd.DataFrame(rows)
    # Write data before its matching signature: an interrupt can leave an
    # orphaned cache file but can never make it look valid.
    frame.to_pickle(data_path)
    meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return frame


def _cached_class_rows_for_results(
    path: Path,
    *,
    cache_dir: Path,
    source_subdir: str,
) -> pd.DataFrame:
    """Materialize atomic target-class/contrast rows for layer-win reporting."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:20]
    data_path = cache_dir / f"{digest}.class.pkl"
    meta_path = cache_dir / f"{digest}.class.json"
    stat = path.stat()
    signature = {
        "cache_version": CLASS_CACHE_VERSION,
        "source": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored == signature and data_path.is_file():
            return pd.read_pickle(data_path)
    except (OSError, ValueError, EOFError):
        pass

    payload = _load_results(path)
    rows: List[Dict[str, object]] = []
    if payload:
        for view in VIEWS:
            for row in collect_class_rows_for_results(
                payload, results_path=str(path), representation_view=view
            ):
                if str(row.get("backend") or "") not in FAMILIES:
                    continue
                rows.append({**row, "selection_view": view, "source_subdir": source_subdir or None})
    frame = pd.DataFrame(rows)
    frame.to_pickle(data_path)
    meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return frame


def _paper_task_results(run_dir: Path) -> List[Path]:
    """``<task>/results.json`` for the paper's tasks only.

    Hidden/deprecated tasks (``ioi``, ``race_one_pole``, ``religion_one_pole``,
    ``gender_en_pre``; see ``summary_latex.HIDDEN_MATRIX_TASKS``) never enter this
    appendix, whatever happens to sit in the run tree.
    """
    return [
        path
        for path in sorted(Path(run_dir).glob("*/results.json"))
        if not _is_hidden_task(path.parent.name)
    ]


def collect_source(runs: Path, model: str, subdir: str, *, cache_dir: Path) -> pd.DataFrame:
    run_dir = Path(runs) / model / subdir if subdir else Path(runs) / model
    frames: List[pd.DataFrame] = []
    for path in _paper_task_results(run_dir):
        frames.append(
            _cached_rows_for_results(path, cache_dir=cache_dir, source_subdir=subdir)
        )
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()


def collect_class_source(runs: Path, model: str, subdir: str, *, cache_dir: Path) -> pd.DataFrame:
    """Collect the unaggregated class/contrast rows for one model source."""
    run_dir = Path(runs) / model / subdir if subdir else Path(runs) / model
    frames = [
        _cached_class_rows_for_results(path, cache_dir=cache_dir, source_subdir=subdir)
        for path in _paper_task_results(run_dir)
    ]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = ["detection_score", "intervention_score", "encoding_E"]
    have = [name for name in numeric if name in frame.columns]
    grouped = (
        frame.groupby(["model", "selection_view", "method_group"], dropna=False)[have]
        .mean()
        .reset_index()
    )
    coverage = (
        frame.groupby(["model", "selection_view", "method_group"], dropna=False)
        .agg(n_task_rows=("task", "nunique"), n_class_rows=("n_sources", "sum"))
        .reset_index()
    )
    return grouped.merge(coverage, on=["model", "selection_view", "method_group"], how="left")


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("% No complete layer-selection appendix rows.\n", encoding="utf-8")
        return
    display = summary.copy()
    for metric in ("detection_score", "intervention_score", "encoding_E"):
        if metric in display:
            display[metric] = display[metric].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")
    table = display[
        ["model", "selection_view", "method_group", "detection_score", "intervention_score", "n_task_rows"]
    ].rename(
        columns={
            "selection_view": "selection view",
            "method_group": "method",
            "detection_score": "Det.",
            "intervention_score": "Int.",
            "n_task_rows": "tasks",
        }
    )
    table["model"] = table["model"].map(_model_label)
    table["method"] = table["method"].map(_method_label)
    path.write_text(
        table.to_latex(index=False, escape=False, caption=(
            "Layer-selection appendix. Each view is locked using validation detection only; "
            "detection and intervention are held-out values for that same candidate."
        ), label="tab:layer-selection-appendix"),
        encoding="utf-8",
    )


def _plot(frame: pd.DataFrame, path: Path) -> None:
    usable = frame.dropna(subset=["detection_score", "intervention_score"]).copy()
    fig, axes = plt.subplots(1, len(VIEWS), figsize=(5.1 * len(VIEWS), 4.0), squeeze=False)
    colors = {family: color for family, color in zip(FAMILIES, ("#4c78a8", "#f58518", "#54a24b", "#e45756"))}
    for axis, view in zip(axes[0], VIEWS):
        sub = usable[usable["selection_view"] == view]
        for family in FAMILIES:
            points = sub[sub["backend"] == family]
            if points.empty:
                continue
            axis.scatter(points["detection_score"], points["intervention_score"],
                         label=family.upper(), color=colors[family], alpha=0.8, s=32)
        axis.set_title(VIEW_LABELS[view], fontsize=9)
        axis.set_xlabel("Held-out detection score")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("Held-out intervention score")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _paired_views(frame: pd.DataFrame) -> pd.DataFrame:
    """Join all-layer and best-single rows for the same task/method group."""
    keys = ["model", "task", "method_group", "backend", "pole"]
    metrics = ["detection_score", "intervention_score", "encoding_E", "n_sources"]
    # SAE k* is a separate k-selection analysis, not an all-layer versus one
    # physical layer comparison.  Its identical rows would add a misleading
    # line of zeroes to this paired figure.
    usable = frame[
        frame["selection_view"].isin(["all_layer", "best_single_layer"])
        & ~frame["method_group"].eq("sae:kstar")
    ].copy()
    wide = usable.pivot_table(index=keys, columns="selection_view", values=metrics, aggfunc="first")
    wide.columns = [f"{metric}_{view}" for metric, view in wide.columns]
    out = wide.reset_index()
    for metric in ("detection_score", "intervention_score", "encoding_E"):
        all_col = f"{metric}_all_layer"
        single_col = f"{metric}_best_single_layer"
        if all_col in out and single_col in out:
            out[f"delta_{metric}"] = out[single_col] - out[all_col]
    return out


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (model, group), sub in paired.groupby(["model", "method_group"], dropna=False):
        det = pd.to_numeric(sub.get("delta_detection_score"), errors="coerce").dropna()
        inter = pd.to_numeric(sub.get("delta_intervention_score"), errors="coerce").dropna()
        rows.append(
            {
                "model": model,
                "method_group": group,
                "n_det_pairs": int(det.size),
                "single_det_win_rate": float((det > 1e-15).mean()) if det.size else float("nan"),
                "mean_delta_detection": det.mean() if det.size else float("nan"),
                "median_delta_detection": det.median() if det.size else float("nan"),
                "n_int_pairs": int(inter.size),
                "mean_delta_intervention": inter.mean() if inter.size else float("nan"),
                "median_delta_intervention": inter.median() if inter.size else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "method_group"])


def _atomic_layer_key(method: object, backend: object) -> str:
    """Normalize a candidate id so its all-layer and physical-layer forms pair."""
    return _decision_key(str(method), str(backend))


def _atomic_paired_views(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair validation-selected single layers with all layers per class/contrast.

    Unlike :func:`_paired_views`, this never averages a task's members.  Each
    row is one target-class or pairwise-contrast decision, and its delta is a
    held-out Detection score difference (single layer minus all layers).
    """
    keys = [
        "model", "task", "method_group", "backend", "construction",
        "target_class", "atomic_key",
    ]
    usable = frame[
        frame["selection_view"].isin(["all_layer", "best_single_layer"])
        & ~frame["method_group"].eq("sae:kstar")
        & ~frame["target_class"].eq("__pooled__")
    ].copy()
    if usable.empty:
        return pd.DataFrame(columns=[*keys, "detection_score_all_layer", "detection_score_best_single_layer", "delta_detection_score"])
    usable["atomic_key"] = [
        _atomic_layer_key(method, backend)
        for method, backend in zip(usable["method"], usable["backend"])
    ]
    metrics = [m for m in ("detection_score", "intervention_score") if m in usable.columns]
    wide = usable.pivot_table(
        index=keys,
        columns="selection_view",
        values=metrics,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{view}" for metric, view in wide.columns]
    out = wide.reset_index()
    for metric in metrics:
        all_col = f"{metric}_all_layer"
        single_col = f"{metric}_best_single_layer"
        if all_col in out and single_col in out:
            out[f"delta_{metric}"] = out[single_col] - out[all_col]
    return out


def _atomic_heldout_win_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize atomic held-out Detection wins per model and pooled."""
    if paired.empty or "delta_detection_score" not in paired:
        return pd.DataFrame(columns=["scope", "single_layer_wins", "all_layer_wins", "ties", "n", "single_layer_win_rate"])
    values = pd.to_numeric(paired.get("delta_detection_score"), errors="coerce")
    work = paired.loc[values.notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["scope", "single_layer_wins", "all_layer_wins", "ties", "n", "single_layer_win_rate"])
    work["_delta"] = pd.to_numeric(work["delta_detection_score"], errors="coerce")

    def summarize(scope: str, rows: pd.DataFrame) -> Dict[str, object]:
        delta = rows["_delta"]
        wins = int((delta > 1e-15).sum())
        losses = int((delta < -1e-15).sum())
        ties = int(delta.size - wins - losses)
        return {
            "scope": scope,
            "single_layer_wins": wins,
            "all_layer_wins": losses,
            "ties": ties,
            "n": int(delta.size),
            "single_layer_win_rate": wins / float(delta.size),
        }

    rows = [summarize(str(model), sub) for model, sub in work.groupby("model", sort=True)]
    rows.append(summarize("all models", work))
    return pd.DataFrame(rows)


def _classify_layer_choice(method: str, backend: str) -> str | None:
    """Return the representation selected by a layer-comparison candidate.

    The result deliberately describes the *validation-selected* candidate,
    rather than claiming that the held-out score was used to make the choice.
    """
    if backend == "caa":
        if re.search(r":L\d+_act_prediction$", method):
            return "single layer"
        if method.endswith(":act_prediction"):
            return "all layers"
    elif backend in {"cga", "caga"}:
        return "single layer" if re.search(r":L\d+$", method) else "all layers"
    elif backend == "sae":
        return "all layers" if method.endswith(":all_k1") else "single layer"
    return None


def _decision_key(method: str, backend: str) -> str:
    """Remove the representation suffix, leaving a human-readable decision id."""
    if backend == "caa":
        # ``x:L3_act_prediction`` and the all-layer ``x:act_prediction`` must
        # map to the same key (dropping only ``L3`` would leave ``x_act_...``).
        return re.sub(r":L\d+_act_prediction$", ":act_prediction", method)
    if backend in {"cga", "caga"}:
        return re.sub(r":L\d+$", "", method)
    if backend == "sae":
        return re.sub(r":(?:all_)?k1$", "", method)
    return method


def _single_layer_choice_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Emit one validation choice per target-class/contrast decision.

    ``source_methods`` contains precisely the members selected by the
    canonical per-class lock.  Keeping the selected method alongside a stable
    decision key makes the headline rate fully auditable (for example, a
    contrast/class decision can be traced back to its raw method id).
    """
    source = frame[
        frame["selection_view"].eq("best_overall")
        & ~frame["method_group"].eq("sae:kstar")
    ]
    rows: List[Dict[str, object]] = []
    for _, row in source.iterrows():
        backend, group = str(row["backend"]), str(row["method_group"])
        for method in str(row["source_methods"]).split(","):
            choice = _classify_layer_choice(method, backend)
            if choice is None:
                continue
            rows.append({
                "model": row["model"],
                "task": row["task"],
                "method_group": group,
                "backend": backend,
                "pole": row["pole"],
                "decision_key": _decision_key(method, backend),
                "selected_method": method,
                "choice": choice,
            })
    return pd.DataFrame(rows)


def _single_layer_choice_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize atomic validation choices, retaining their denominator."""
    if decisions.empty:
        return pd.DataFrame(columns=["model", "method_group", "single layer", "all layers", "n", "single_layer_rate"])
    atomic = decisions
    counts = atomic.groupby(["model", "method_group", "choice"]).size().unstack(fill_value=0)
    for column in ("single layer", "all layers"):
        if column not in counts:
            counts[column] = 0
    counts = counts.reset_index()
    counts["n"] = counts["single layer"] + counts["all layers"]
    counts["single_layer_rate"] = counts["single layer"] / counts["n"]
    return counts.sort_values(["model", "method_group"])


def _overall_choice_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return the directly reportable all-decisions rate, per model and pooled."""
    if decisions.empty:
        return pd.DataFrame(columns=["scope", "single layer", "all layers", "n", "single_layer_rate"])
    counts = decisions.groupby(["model", "choice"]).size().unstack(fill_value=0)
    for column in ("single layer", "all layers"):
        if column not in counts:
            counts[column] = 0
    counts = counts.reset_index().rename(columns={"model": "scope"})
    pooled = pd.DataFrame([{
        "scope": "all models",
        "single layer": int((decisions["choice"] == "single layer").sum()),
        "all layers": int((decisions["choice"] == "all layers").sum()),
    }])
    out = pd.concat([counts, pooled], ignore_index=True)
    out["n"] = out["single layer"] + out["all layers"]
    out["single_layer_rate"] = out["single layer"] / out["n"]
    return out


def _write_paired_latex(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("% No paired all-layer/single-layer rows.\n", encoding="utf-8")
        return
    display = summary.copy()
    for column in ("single_det_win_rate", "mean_delta_detection", "median_delta_detection",
                   "mean_delta_intervention", "median_delta_intervention"):
        display[column] = display[column].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")
    table = display.rename(columns={
        "method_group": "method",
        "n_det_pairs": "n Det.",
        "single_det_win_rate": "P(single Det. > all)",
        "mean_delta_detection": "mean ΔDet.",
        "median_delta_detection": "median ΔDet.",
        "n_int_pairs": "n Int.",
        "mean_delta_intervention": "mean ΔInt.",
        "median_delta_intervention": "median ΔInt.",
    })[["model", "method", "n Det.", "P(single Det. > all)", "mean ΔDet.",
        "median ΔDet.", "n Int.", "mean ΔInt.", "median ΔInt."]]
    table["model"] = table["model"].map(_model_label)
    table["method"] = table["method"].map(_method_label)
    path.write_text(table.to_latex(
        index=False, escape=False,
        caption=("Paired comparison of the validation-best single layer against the "
                 "predefined all-layer representation. Differences are test single-layer "
                 "minus all-layer scores (positive = single best layer better); no intervention "
                 "score participates in selection."),
        label="tab:layer-selection-paired",
    ), encoding="utf-8")


def _plot_paired_deltas(
    paired: pd.DataFrame, path: Path, *, model_order: Sequence[str],
    height_scale: float = 1.0,
) -> None:
    """Show paired effect distributions rather than an unpaired Det/Int cloud."""
    _configure_paper_font()
    groups = [group for group in DISPLAY_GROUPS if group in set(paired["method_group"])]
    available = set(paired["model"].dropna())
    # The model list is a stable study-wide small-to-large ordering, rather
    # than whichever order a particular invocation happened to receive.
    models = _study_model_order(model for model in model_order if model in available)
    base_height = max(3.3, 0.3 * len(groups) * len(models) + 0.75)
    fig, axes = plt.subplots(
        len(models), 2,
        figsize=(11.5, base_height * height_scale),
        squeeze=False,
        sharex="col",
        sharey=True,
    )
    specs = (("delta_detection_score", "Detection"),
             ("delta_intervention_score", "Intervention"))
    for row, model in enumerate(models):
        data = paired[paired["model"] == model]
        for col, (metric, label) in enumerate(specs):
            axis = axes[row][col]
            axis.axvline(0, color="black", linewidth=0.8, zorder=0)
            for y, group in enumerate(groups):
                values = pd.to_numeric(data.loc[data["method_group"] == group, metric], errors="coerce").dropna()
                if values.empty:
                    continue
                offsets = [((i % 7) - 3) * 0.055 for i in range(len(values))]
                axis.scatter(values, [y + o for o in offsets], color=method_color(group),
                             alpha=0.62, s=22, linewidths=0)
                axis.scatter([values.median()], [y], color="black", marker="|", s=170, linewidths=1.7, zorder=3)
            axis.grid(axis="x", alpha=0.25)
            axis.set_yticks(range(len(groups)))
            if col == 0:
                axis.set_yticklabels([_method_label(group) for group in groups], fontsize=8)
            else:
                axis.tick_params(axis="y", labelleft=False)
            if row == 0:
                axis.set_title(label, fontsize=11, pad=8)
            if row == len(models) - 1:
                axis.set_xlabel(r"Difference: single best layer $-$ all layers", fontsize=9)
        # Keep the model names as row labels: they are visually separate from
        # the Detection/Intervention column headers and read vertically down
        # the figure's left edge.
        axes[row][0].annotate(
            _model_label(model), xy=(-0.16, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=10,
        )
    fig.tight_layout(rect=(0.06, 0.02, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _model_averaged_paired(paired: pd.DataFrame) -> pd.DataFrame:
    """Average each task/method paired effect across models with data.

    The original paired plot shows every model separately.  This companion
    frame first collapses the model dimension, so a task/method point has the
    same weight whether it was observed for one model or many.  ``n_models``
    makes incomplete model coverage explicit in the CSV.
    """
    keys = ["task", "method_group", "backend", "pole"]
    metrics = ["delta_detection_score", "delta_intervention_score"]
    rows: List[Dict[str, object]] = []
    for values, sub in paired.groupby(keys, dropna=False):
        row: Dict[str, object] = dict(zip(keys, values))
        for metric in metrics:
            numeric = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[metric] = numeric.mean() if not numeric.empty else float("nan")
            row[f"n_models_{metric.removeprefix('delta_')}"] = int(numeric.size)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method_group", "task"])


def _plot_model_averaged_paired_deltas(paired: pd.DataFrame, path: Path) -> None:
    """Plot task-level paired deltas after averaging over available models."""
    _configure_paper_font()
    groups = [group for group in DISPLAY_GROUPS if group in set(paired["method_group"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(3.3, 0.3 * len(groups) + 0.75)),
                             squeeze=False, sharey=True)
    specs = (("delta_detection_score", "Detection"),
             ("delta_intervention_score", "Intervention"))
    for col, (metric, label) in enumerate(specs):
        axis = axes[0][col]
        axis.axvline(0, color="black", linewidth=0.8, zorder=0)
        for y, group in enumerate(groups):
            values = pd.to_numeric(
                paired.loc[paired["method_group"] == group, metric], errors="coerce"
            ).dropna()
            if values.empty:
                continue
            offsets = [((i % 7) - 3) * 0.055 for i in range(len(values))]
            axis.scatter(values, [y + offset for offset in offsets], color=method_color(group),
                         alpha=0.62, s=22, linewidths=0)
            axis.scatter([values.median()], [y], color="black", marker="|", s=170,
                         linewidths=1.7, zorder=3)
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(range(len(groups)))
        if col == 0:
            axis.set_yticklabels([_method_label(group) for group in groups], fontsize=8)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_title(f"{label}: model-averaged per task", fontsize=11, pad=8)
        axis.set_xlabel(r"Difference: single best layer $-$ all layers", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.02, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_choice_rates(choices: pd.DataFrame, path: Path) -> None:
    """Plot normal stacked selection proportions, one panel per model."""
    models = list(choices["model"].unique())
    groups = [group for group in DISPLAY_GROUPS if group in set(choices["method_group"])]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5), squeeze=False, sharey=True)
    for axis, model in zip(axes[0], models):
        sub = choices[choices["model"] == model].set_index("method_group")
        rates = [float(sub.loc[group, "single_layer_rate"]) * 100 if group in sub.index else 0 for group in groups]
        all_rates = [100 - value for value in rates]
        x = list(range(len(groups)))
        axis.bar(x, all_rates, color="#b9c1ca", label="All layers")
        axis.bar(x, rates, bottom=all_rates, color="#4c78a8", label="Single layer")
        for index, (all_rate, rate) in enumerate(zip(all_rates, rates)):
            if all_rate >= 8:
                axis.text(index, all_rate / 2, f"{all_rate:.0f}%", ha="center", va="center", fontsize=8)
            if rate >= 8:
                axis.text(index, all_rate + rate / 2, f"{rate:.0f}%", ha="center", va="center", fontsize=8,
                          color="white")
        axis.set_ylim(0, 100)
        axis.set_ylabel("Validation choices (%)")
        axis.set_title(_model_label(model))
        axis.set_xticks(x, [_method_label(group) for group in groups], rotation=30, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", action="append", required=True, metavar="MODEL[=SUBDIR]")
    args = parser.parse_args()
    sources = parse_sources(args.source)
    unknown = set(VIEWS) - set(REPRESENTATION_VIEWS)
    if unknown:
        raise RuntimeError(f"unsupported view(s): {sorted(unknown)}")

    cache_dir = args.out / "cache"
    frames = [
        collect_source(args.runs, model, subdir, cache_dir=cache_dir)
        for model, subdir in sources
    ]
    class_frames = [
        collect_class_source(args.runs, model, subdir, cache_dir=cache_dir)
        for model, subdir in sources
    ]
    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()
    class_frame = pd.concat([f for f in class_frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in class_frames
    ) else pd.DataFrame()
    args.out.mkdir(parents=True, exist_ok=True)
    long_path = args.out / "layer_selection_long.csv"
    summary_path = args.out / "layer_selection_summary.csv"
    latex_path = args.out / "layer_selection_summary.tex"
    figure_path = args.out / "layer_selection_det_int.pdf"
    paired_path = args.out / "layer_selection_paired.csv"
    paired_summary_path = args.out / "layer_selection_paired_summary.csv"
    paired_latex_path = args.out / "layer_selection_paired_summary.tex"
    # The paper figure shows atomic class/contrast comparisons; the coarser
    # model x task x method-group cells are kept as a companion.
    paired_figure_path = args.out / "layer_selection_paired_deltas_cells.pdf"
    atomic_figure_path = args.out / "layer_selection_paired_deltas.pdf"
    model_averaged_paired_path = args.out / "layer_selection_paired_deltas_model_averaged.csv"
    model_averaged_paired_figure_path = args.out / "layer_selection_paired_deltas_model_averaged.pdf"
    choice_path = args.out / "layer_selection_choice_rates.csv"
    decision_path = args.out / "layer_selection_decisions.csv"
    overall_choice_path = args.out / "layer_selection_choice_overall.csv"
    choice_figure_path = args.out / "layer_selection_choice_rates.pdf"
    atomic_paired_path = args.out / "layer_selection_atomic_heldout_detection.csv"
    atomic_summary_path = args.out / "layer_selection_atomic_heldout_detection_summary.csv"
    frame.to_csv(long_path, index=False)
    summary = _summary(frame) if not frame.empty else pd.DataFrame()
    summary.to_csv(summary_path, index=False)
    _write_latex(summary, latex_path)
    if not frame.empty:
        _plot(frame, figure_path)
        paired = _paired_views(frame)
        paired.to_csv(paired_path, index=False)
        paired_summary = _paired_summary(paired)
        paired_summary.to_csv(paired_summary_path, index=False)
        _write_paired_latex(paired_summary, paired_latex_path)
        _plot_paired_deltas(paired, paired_figure_path, model_order=[model for model, _ in sources])
        model_averaged_paired = _model_averaged_paired(paired)
        model_averaged_paired.to_csv(model_averaged_paired_path, index=False)
        _plot_model_averaged_paired_deltas(model_averaged_paired, model_averaged_paired_figure_path)
        decisions = _single_layer_choice_decisions(frame)
        decisions.to_csv(decision_path, index=False)
        choices = _single_layer_choice_summary(decisions)
        choices.to_csv(choice_path, index=False)
        if not choices.empty:
            _plot_choice_rates(choices, choice_figure_path)
        overall_choices = _overall_choice_summary(decisions)
        overall_choices.to_csv(overall_choice_path, index=False)
        pooled = overall_choices[overall_choices["scope"] == "all models"]
        if not pooled.empty:
            row = pooled.iloc[0]
            print(
                "validation selection: single layer chosen in "
                f"{float(row['single_layer_rate']):.1%} of decisions "
                f"({int(row['single layer'])}/{int(row['n'])})"
            )
    atomic_paired = _atomic_paired_views(class_frame) if not class_frame.empty else pd.DataFrame()
    atomic_paired.to_csv(atomic_paired_path, index=False)
    atomic_summary = _atomic_heldout_win_summary(atomic_paired)
    atomic_summary.to_csv(atomic_summary_path, index=False)
    if not atomic_paired.empty:
        _plot_paired_deltas(
            atomic_paired, atomic_figure_path,
            model_order=[model for model, _ in sources], height_scale=2 / 3,
        )
    pooled_atomic = atomic_summary[atomic_summary["scope"] == "all models"]
    if not pooled_atomic.empty:
        row = pooled_atomic.iloc[0]
        print(
            "held-out Detection: validation-selected single layer wins "
            f"{float(row['single_layer_win_rate']):.1%} of atomic class/contrast comparisons "
            f"({int(row['single_layer_wins'])}/{int(row['n'])}; "
            f"all-layer wins {int(row['all_layer_wins'])}, ties {int(row['ties'])})"
        )
    print(f"wrote {long_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {latex_path}")
    if figure_path.is_file():
        print(f"wrote {figure_path}")
    for path in (paired_path, paired_summary_path, paired_latex_path, paired_figure_path, atomic_figure_path,
                 model_averaged_paired_path, model_averaged_paired_figure_path,
                 decision_path, choice_path, overall_choice_path, choice_figure_path,
                 atomic_paired_path, atomic_summary_path):
        if path.is_file():
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
