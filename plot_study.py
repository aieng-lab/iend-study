"""
Study plots: layer profiles, SAE k-curves, exclusivity, causal bars, none vs tensor.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.plot_style import HOLLOW_EDGE_WIDTH
from study.method_ids import normalize_sae_method_id


def strip_figure_titles(fig: Any) -> None:
    """Keep worker-side diagnostic plots title-free without analysis imports.

    ``plot_study`` is imported by the GPU SAE stage, whereas ``analysis/`` is
    report-only code and may not be present in an older staged checkout.  Keep
    this tiny helper local so a plotting-style update can never abort a study.
    """
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in getattr(fig, "axes", ()):
        ax.set_title("")


def _optional_int_layer(value: Any) -> Optional[int]:
    if value is None or str(value) == "all":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pyplot():
    """Import pyplot with usetex off (gradiend may enable it; Unicode labels break LaTeX)."""
    import matplotlib

    matplotlib.rcParams["text.usetex"] = False
    from matplotlib import pyplot as plt

    return plt


def _parse_layer_part(method: str) -> Optional[Tuple[str, str, int]]:
    """Return (backend, class_id, layer) for ids like ``actiend:M:L11`` / ``sae:F:L6``."""
    m = re.fullmatch(r"(sae|actiend|gradiend):([^:]+):L(\d+)", str(method))
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _row_all_or_tensors(
    by_method: Dict[str, Dict[str, Any]], base_id: str
) -> Dict[str, Any]:
    """``base:tensors`` preferred; accept legacy ``base:all``."""
    return by_method.get(f"{base_id}:tensors") or by_method.get(f"{base_id}:all") or {}


def _methods_index(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in results.get("methods") or []:
        mid = row.get("method")
        if mid:
            out[str(mid)] = row
    return out


def _require_plot_classes(target_classes: Optional[Sequence[str]]) -> List[str]:
    """Plots need explicit class ids — never invent gender M/F."""
    from results_schema import require_class_ids

    return require_class_ids(target_classes, where="plot_study")


def _metric_from_row(row: Dict[str, Any], metric: str) -> Optional[float]:
    m = row.get("metrics") or {}
    if metric == "roc_auc_other":
        return _safe_float(m.get("roc_auc_other") or m.get("min_pairwise_auroc"))
    if metric in {"roc_auc", "roc_auc_neutral"}:
        return _safe_float(m.get(metric) or m.get("roc_auc_neutral") or m.get("roc_auc"))
    if metric in {"neutral_specificity", "specificity"}:
        return _safe_float(m.get("neutral_specificity") or m.get("specificity"))
    return _safe_float(m.get(metric))


def plot_layer_profile(
    results: Dict[str, Any],
    *,
    output: Path,
    metric: str = "roc_auc",
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """AUC/spec vs layer for SAE + ACTIEND; GRADIEND / SAE-k1 / SAE-k* refs."""
    try:
        plt = _pyplot()
    except Exception:
        print(f"plot_layer_profile({metric}): matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by_method = _methods_index(results)
    fig, axes = plt.subplots(1, len(target_classes), figsize=(5.0 * len(target_classes), 4.0), squeeze=False)
    for ax, cls in zip(axes[0], target_classes):
        cls = str(cls)
        sae_xy: List[Tuple[int, float]] = []
        act_xy: List[Tuple[int, float]] = []
        for mid, row in by_method.items():
            parsed = _parse_layer_part(mid)
            if not parsed:
                continue
            backend, class_id, layer = parsed
            if class_id != cls:
                continue
            val = _metric_from_row(row, metric)
            if val is None:
                continue
            if backend == "sae":
                sae_xy.append((layer, val))
            elif backend == "actiend":
                act_xy.append((layer, val))
        if sae_xy:
            sae_xy.sort()
            ax.plot([x for x, _ in sae_xy], [y for _, y in sae_xy], marker="o", label="sae L* (layer $k^*$)")
        if act_xy:
            act_xy.sort()
            ax.plot([x for x, _ in act_xy], [y for _, y in act_xy], marker="s", label="actiend L*")
        for mid, style, lab in (
            (f"gradiend:{cls}", "--", "gradiend none (ref)"),
            (f"actiend:{cls}", ":", "actiend none (ref)"),
            (f"actiend:{cls}:tensors", (0, (1, 1)), "actiend :tensors (ref)"),
            (f"sae:{cls}:k1", "-.", "sae k=1 @ L* (ref)"),
            (f"sae:{cls}:kstar", (0, (3, 1, 1, 1)), "sae $k^*$ @ L* (ref)"),
        ):
            ref = by_method.get(mid)
            if not ref:
                continue
            val = _metric_from_row(ref, metric)
            if val is None:
                continue
            ax.axhline(val, linestyle=style, alpha=0.75, label=lab)
        ax.set_title(f"class {cls}")
        ax.set_xlabel("layer")
        ax.set_ylabel(metric)
        if sae_xy or act_xy:
            ax.legend(fontsize=7, loc="best")
        # Only force [0,1] when values are probabilities; leave room if refs saturate.
        if metric in {
            "roc_auc",
            "roc_auc_neutral",
            "roc_auc_other",
            "balanced_accuracy",
            "specificity",
            "neutral_specificity",
            "class_exclusivity",
        }:
            ax.set_ylim(0.0, 1.05)
            # Note saturation: flat actiend near 1 is real ceiling, not missing data.
            if act_xy and all(abs(y - 1.0) < 1e-3 for _, y in act_xy):
                ax.text(
                    0.02,
                    0.08,
                    "actiend L* saturated (~1)",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="0.35",
                )
    fig.suptitle(f"Layer profile ({metric})")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_k_curves(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
    metrics: Sequence[str] = ("roc_auc", "roc_auc_other"),
) -> Optional[Path]:
    """SAE auc_n / auc_o vs k; mark k=1 and k*."""
    try:
        plt = _pyplot()
    except Exception:
        print("plot_k_curves: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by_method = _methods_index(results)
    n_m = len(metrics)
    fig, axes = plt.subplots(
        n_m, len(target_classes), figsize=(5.0 * len(target_classes), 3.6 * n_m), squeeze=False
    )
    for col, cls in enumerate(target_classes):
        cls = str(cls)
        row = by_method.get(f"sae:{cls}:kstar") or by_method.get(f"sae:{cls}")
        curve = []
        if row:
            curve = (row.get("extras") or {}).get("sparse_combination_curve") or []
        k_star = (row.get("metrics") or {}).get("readout_k") if row else None
        for r_i, metric in enumerate(metrics):
            ax = axes[r_i][col]
            if not curve:
                ax.set_title(f"sae:{cls}:kstar (no curve)")
                continue
            pts = []
            for pt in curve:
                if "error" in pt:
                    continue
                k = pt.get("k") or pt.get("readout_k")
                if metric == "roc_auc_other":
                    val = _safe_float(pt.get("roc_auc_other") or pt.get("min_pairwise_auroc"))
                else:
                    val = _safe_float(pt.get(metric) or pt.get("roc_auc"))
                if isinstance(k, int) and val is not None:
                    pts.append((int(k), val))
            pts.sort()
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
            ax.axvline(1, color="C2", linestyle=":", label="k=1")
            if isinstance(k_star, int):
                ax.axvline(int(k_star), color="C3", linestyle="--", label=f"$k^*$={k_star}")
            ax.set_title(f"sae:{cls} — {metric}")
            ax.set_xlabel("k")
            ax.set_ylabel(metric)
            ax.set_ylim(0.0, 1.05)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("SAE k-curves (mark k=1 and $k^*$)")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_exclusivity_vs_k(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """class_exclusivity / auc_o vs k — the bag vs exclusivity tradeoff."""
    try:
        plt = _pyplot()
    except Exception:
        print("plot_exclusivity_vs_k: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by_method = _methods_index(results)
    fig, axes = plt.subplots(1, len(target_classes), figsize=(5.0 * len(target_classes), 4.0), squeeze=False)
    for ax, cls in zip(axes[0], target_classes):
        cls = str(cls)
        row = by_method.get(f"sae:{cls}:kstar") or by_method.get(f"sae:{cls}")
        curve = (row.get("extras") or {}).get("sparse_combination_curve") or [] if row else []
        k_star = (row.get("metrics") or {}).get("readout_k") if row else None
        xs, yo, ye = [], [], []
        for pt in curve:
            if "error" in pt:
                continue
            k = pt.get("k")
            if not isinstance(k, int):
                continue
            o = _safe_float(pt.get("roc_auc_other") or pt.get("min_pairwise_auroc"))
            e = _safe_float(pt.get("class_exclusivity"))
            xs.append(k)
            yo.append(o if o is not None else float("nan"))
            ye.append(e if e is not None else float("nan"))
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]
            yo = [yo[i] for i in order]
            ye = [ye[i] for i in order]
            ax.plot(xs, yo, marker="o", label="auc_o")
            ax.plot(xs, ye, marker="s", label="excl")
        ax.axvline(1, color="C2", linestyle=":", label="k=1")
        if isinstance(k_star, int):
            ax.axvline(int(k_star), color="C3", linestyle="--", label=f"$k^*$={k_star}")
        ax.set_title(f"sae:{cls}:kstar")
        ax.set_xlabel("k")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Exclusivity / auc_o vs k")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_causal_bars(
    results: Dict[str, Any],
    *,
    output: Path,
    methods: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Bar chart of causal ΔP(target) on other / target / neutral."""
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_causal_bars: matplotlib unavailable; skipping")
        return None

    from results_schema import _causal_methods

    methods = list(methods or _causal_methods(results))
    by = {r.get("method"): r for r in (results.get("methods") or [])}
    labels, doth, dtgt, dneu = [], [], [], []
    annotations = []
    for mid in methods:
        m = (by.get(mid) or {}).get("metrics") or {}
        # Skip rows with no causal payload at all (keeps plot readable).
        if (
            m.get("causal_signed_effect") is None
            and m.get("causal_delta_mean") is None
            and m.get("causal_selected_strength") is None
        ):
            continue
        labels.append(mid)
        doth.append(_safe_float(m.get("causal_delta_other") or m.get("causal_signed_effect")) or 0.0)
        dtgt.append(_safe_float(m.get("causal_delta_target")) or 0.0)
        dneu.append(_safe_float(m.get("causal_delta_neutral")) or 0.0)
        flag = []
        if m.get("causal_grid_ceiling"):
            flag.append("ceil")
        if m.get("causal_grid_floor"):
            flag.append("floor")
        if m.get("causal_null_effect"):
            flag.append("null")
        if m.get("causal_gate_empty"):
            flag.append("gate99empty")
        annotations.append("+".join(flag) if flag else "")

    if not labels:
        return None
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels) + 2.0), 4.2))
    ax.bar(x - width, doth, width, label="doth (dP tgt|other)")
    ax.bar(x, dtgt, width, label="dtgt")
    ax.bar(x + width, dneu, width, label="dneu")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("dP(target)")
    ax.set_title("Causal effects (LMS×0.99-gated)")
    ax.legend(fontsize=8)
    ax.axhline(0.0, color="k", linewidth=0.6)
    for i, ann in enumerate(annotations):
        if ann:
            ax.text(i, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 0.01, ann, ha="center", fontsize=6)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _causal_lms_curve_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer metrics.causal_lms_curve; fall back to extras.causal.strengths."""
    m = row.get("metrics") or {}
    curve = m.get("causal_lms_curve")
    if isinstance(curve, list) and curve:
        return [c for c in curve if isinstance(c, dict)]
    extras = row.get("extras") or {}
    if isinstance(extras.get("causal_lms_curve"), list) and extras["causal_lms_curve"]:
        return [c for c in extras["causal_lms_curve"] if isinstance(c, dict)]
    strengths = (extras.get("causal") or {}).get("strengths") or []
    out: List[Dict[str, Any]] = []
    for sr in strengths:
        if not isinstance(sr, dict) or sr.get("is_random_control"):
            continue
        lms = _safe_float(sr.get("lms"))
        base = _safe_float(sr.get("base_lms"))
        out.append(
            {
                "strength": _safe_float(sr.get("strength")),
                "lms": lms,
                "base_lms": base,
                "signed_effect": _safe_float(sr.get("signed_effect")),
                "lms_ratio_to_base": (
                    None if lms is None or base in (None, 0.0) else lms / base
                ),
                "lms_ok_0.99": bool(sr.get("lms_ok")) if base is None else (
                    lms is not None and base is not None and lms >= 0.99 * base
                ),
            }
        )
    return out


def plot_lms_vs_strength(
    results: Dict[str, Any],
    *,
    output: Path,
    methods: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """LMS/base and effect vs strength; one color per method.

    Strength means LR for GRADIEND/ACTIEND (decoder grid) and SAE α for SAE.
    Horizontal line marks the LMS×0.99 gate; star markers show selected strengths.
    """
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_lms_vs_strength: matplotlib unavailable; skipping")
        return None

    from results_schema import _causal_methods

    methods = list(methods or _causal_methods(results))
    by = {r.get("method"): r for r in (results.get("methods") or [])}
    series = []
    for mid in methods:
        row = by.get(mid) or {}
        m = row.get("metrics") or {}
        curve = _causal_lms_curve_rows(row)
        xs, ys_rel, ys_eff = [], [], []
        for pt in curve:
            s = _safe_float(pt.get("strength"))
            rel = _safe_float(pt.get("lms_ratio_to_base"))
            if rel is None:
                lms = _safe_float(pt.get("lms"))
                base = _safe_float(pt.get("base_lms"))
                if lms is not None and base not in (None, 0.0):
                    rel = lms / base
            if s is None or rel is None:
                continue
            xs.append(s)
            ys_rel.append(rel)
            ys_eff.append(_safe_float(pt.get("signed_effect")) or float("nan"))
        if xs:
            series.append(
                (
                    mid,
                    xs,
                    ys_rel,
                    ys_eff,
                    _safe_float(m.get("causal_selection_strength")),
                    _safe_float(m.get("causal_selection_signed_effect")),
                    _safe_float(m.get("causal_signed_effect")),
                    _safe_float(m.get("causal_selection_lms_ratio_to_base")),
                    _safe_float(m.get("causal_lms"))
                    / _safe_float(m.get("causal_base_lms"))
                    if _safe_float(m.get("causal_lms")) is not None
                    and _safe_float(m.get("causal_base_lms")) not in (None, 0.0)
                    else None,
                )
            )

    if not series:
        print("plot_lms_vs_strength: no strength curves; skipping")
        return None

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, pack in enumerate(series):
        mid, xs, ys_rel, ys_eff, s99, e_validation, e_test, r_validation, r_test = pack
        color = cmap(i % 10)
        order = np.argsort(np.asarray(xs, dtype=float))
        xs_a = np.asarray(xs, dtype=float)[order]
        ax0.plot(
            xs_a,
            np.asarray(ys_rel, dtype=float)[order],
            "o-",
            color=color,
            label=mid,
            markersize=4,
        )
        ax1.plot(
            xs_a,
            np.asarray(ys_eff, dtype=float)[order],
            "o-",
            color=color,
            label=mid,
            markersize=4,
        )
        if s99 is not None and r_validation is not None:
            ax0.scatter(
                [s99], [r_validation], marker="*", s=90, color=color,
                zorder=5, edgecolors="k", linewidths=0.4,
                label="validation-selected" if i == 0 else None,
            )
        if s99 is not None and r_test is not None:
            ax0.scatter(
                [s99], [r_test], marker="X", s=55, color=color,
                zorder=5, edgecolors="k", linewidths=0.4,
                label="frozen test report" if i == 0 else None,
            )
        if s99 is not None and r_validation is not None and r_test is not None:
            ax0.plot(
                [s99, s99], [r_validation, r_test], linestyle="--",
                linewidth=0.8, color=color, alpha=0.8,
            )
        if s99 is not None and e_validation is not None:
            ax1.scatter(
                [s99], [e_validation], marker="*", s=90, color=color,
                zorder=5, edgecolors="k", linewidths=0.4,
            )
        if s99 is not None and e_test is not None:
            ax1.scatter(
                [s99], [e_test], marker="X", s=55, color=color,
                zorder=5, edgecolors="k", linewidths=0.4,
            )
        if s99 is not None and e_validation is not None and e_test is not None:
            ax1.plot(
                [s99, s99], [e_validation, e_test], linestyle="--",
                linewidth=0.8, color=color, alpha=0.8,
            )

    ax0.axhline(0.99, color="0.35", linestyle="--", linewidth=1.0, label="LMS×0.99")
    ax0.set_ylabel("LMS / base LMS")
    ax0.set_title("Validation LMS sweep (* = selected; X = frozen test report)")
    ax0.legend(fontsize=7, ncol=2, loc="best")
    ymin = min(min(ys) for _, _, ys, *_ in series)
    ax0.set_ylim(bottom=min(0.8, ymin - 0.02))

    ax1.axhline(0.0, color="k", linewidth=0.6)
    ax1.set_xlabel("strength (GRADIEND/ACTIEND LR, or SAE α)")
    ax1.set_ylabel("signed effect (dP tgt|other)")
    ax1.set_title("Validation causal-effect sweep (X = frozen test headline)")
    ax1.legend(fontsize=7, ncol=2, loc="best")
    all_x = [x for _, xs, *_ in series for x in xs]
    if all_x and min(all_x) > 0:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_none_vs_tensor(
    results: Dict[str, Any],
    *,
    output: Path,
    metric: str = "roc_auc",
    target_classes: Optional[Sequence[str]] = None,
    backends: Sequence[str] = ("gradiend", "actiend"),
) -> Optional[Path]:
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_none_vs_tensor: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by_method = _methods_index(results)
    labels: List[str] = []
    none_vals: List[float] = []
    all_vals: List[float] = []
    for backend in backends:
        for cls in target_classes:
            labels.append(f"{backend}:{cls}")
            n = _metric_from_row(by_method.get(f"{backend}:{cls}") or {}, metric)
            a = _metric_from_row(_row_all_or_tensors(by_method, f"{backend}:{cls}"), metric)
            none_vals.append(n if n is not None else float("nan"))
            all_vals.append(a if a is not None else float("nan"))
    if not labels:
        return None
    if not any(v == v for v in none_vals + all_vals):  # NaN != NaN
        return None
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4.0))
    ax.bar(x - width / 2, none_vals, width, label="none")
    ax.bar(x + width / 2, all_vals, width, label="by_tensor :tensors")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("none vs by_tensor aggregate")
    ax.legend()
    ax.set_ylim(0.0, 1.05 if metric in {"roc_auc", "roc_auc_other", "balanced_accuracy", "specificity", "neutral_specificity", "class_exclusivity", "youden_j", "neutral_specificity_mag", "class_exclusivity_mag"} else None)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_none_vs_tensor_panel(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
    backends: Sequence[str] = ("gradiend", "actiend"),
    metrics: Optional[Sequence[Tuple[str, str]]] = None,
) -> Optional[Path]:
    """Multi-metric panel for none vs by_tensor encoding (conclusion plot)."""
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_none_vs_tensor_panel: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    metric_specs = list(
        metrics
        or (
            ("roc_auc", "auc_n"),
            ("roc_auc_other", "auc_o"),
            ("balanced_accuracy", "bal"),
            ("cohens_d", "d"),
            ("neutral_specificity", "spec"),
            ("class_exclusivity", "excl"),
            ("neutral_specificity_mag", "smag"),
            ("class_exclusivity_mag", "emag"),
        )
    )
    by_method = _methods_index(results)
    labels = [f"{b}:{c}" for b in backends for c in target_classes]
    # Require at least one pair present.
    any_pair = False
    for lab in labels:
        if by_method.get(lab) or _row_all_or_tensors(by_method, lab):
            any_pair = True
            break
    if not any_pair:
        return None

    n = len(metric_specs)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)
    x = np.arange(len(labels))
    width = 0.35
    unit01 = {
        "roc_auc",
        "roc_auc_other",
        "balanced_accuracy",
        "neutral_specificity",
        "specificity",
        "class_exclusivity",
        "youden_j",
        "neutral_specificity_mag",
        "class_exclusivity_mag",
    }
    for i, (metric, title) in enumerate(metric_specs):
        ax = axes[i // ncols][i % ncols]
        none_vals = []
        all_vals = []
        for lab in labels:
            none_vals.append(
                _metric_from_row(by_method.get(lab) or {}, metric) or float("nan")
            )
            all_vals.append(
                _metric_from_row(_row_all_or_tensors(by_method, lab), metric) or float("nan")
            )
        ax.bar(x - width / 2, none_vals, width, label="none")
        ax.bar(x + width / 2, all_vals, width, label=":tensors")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        if metric in unit01:
            ax.set_ylim(0.0, 1.05)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Encoding: none vs by_tensor (:tensors)", fontsize=12)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_none_vs_tensor_delta(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
    backends: Sequence[str] = ("gradiend", "actiend"),
) -> Optional[Path]:
    """Grouped Δ bars (:all − none) for discriminating encoding metrics."""
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_none_vs_tensor_delta: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    from results_schema import none_vs_tensor_pairs

    pairs = [
        p
        for p in none_vs_tensor_pairs(
            results, backends=backends, target_classes=target_classes
        )
        if p.get("present_none") and p.get("present_all")
    ]
    if not pairs:
        return None
    cols = ["auc_n", "auc_o", "bal", "d", "spec", "excl", "smag", "emag"]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(pairs) * 0.5 + 4), 4.2))
    x = np.arange(len(cols))
    width = min(0.8 / max(len(pairs), 1), 0.25)
    for i, p in enumerate(pairs):
        deltas = []
        for col in cols:
            d = (p["metrics"].get(col) or {}).get("delta")
            deltas.append(d if isinstance(d, (int, float)) else float("nan"))
        offset = (i - (len(pairs) - 1) / 2) * width
        ax.bar(x + offset, deltas, width, label=f"{p['backend']}:{p['class']}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylabel("delta (:tensors - none)")
    ax.set_title("Encoding split delta — positive => by_tensor better")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_layer_rank_scores(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Feature-ranking top-1 score vs layer (same metric that picks the layer).

    Data sources (first hit wins per layer): ``raw.sae.layer_profile``, then
    ``by_layer_readouts`` / ``sae:{cls}:L*`` metrics, then
    ``layer_selection_scores`` dicts.
    """
    try:
        plt = _pyplot()
    except Exception:
        print("plot_layer_rank_scores: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    raw = ((results.get("raw") or {}).get("sae") or {})
    profile = raw.get("layer_profile") or {}
    by_layer_rd = raw.get("by_layer_readouts") or {}
    selected = raw.get("selected_layer_by_class") or {}
    selected_opp = raw.get("selected_layer_by_class_opp_fire") or {}
    scores_pc = raw.get("layer_selection_scores") or {}
    by_method = _methods_index(results)

    fig, axes = plt.subplots(
        1, len(target_classes), figsize=(5.0 * len(target_classes), 4.0), squeeze=False
    )
    any_real = False
    for ax, cls in zip(axes[0], target_classes):
        cls = str(cls)
        layers = set()
        for key, entry in profile.items():
            if isinstance(entry, dict) and isinstance(entry.get("layer"), int):
                layers.add(int(entry["layer"]))
            elif str(key).isdigit():
                layers.add(int(key))
        for key in by_layer_rd:
            if str(key).lstrip("-").isdigit():
                layers.add(int(key))
        for mid in by_method:
            parsed = _parse_layer_part(mid)
            if parsed and parsed[0] == "sae" and parsed[1] == cls:
                layers.add(parsed[2])
        if not layers:
            ax.set_title(f"{cls}: no layer rank data")
            continue
        xs = sorted(layers)
        ys_pc, ys_opp = [], []
        for layer in xs:
            entry = profile.get(str(layer)) or profile.get(layer) or {}
            pc = (entry.get("per_class_rank_score") or {}).get(cls)
            opp = (entry.get("opp_fire_rank_score") or {}).get(cls)
            if not isinstance(pc, (int, float)):
                pc = ((by_layer_rd.get(str(layer)) or by_layer_rd.get(layer) or {}).get(cls) or {}).get(
                    "layer_rank_score"
                )
            if not isinstance(opp, (int, float)):
                opp = ((by_layer_rd.get(str(layer)) or by_layer_rd.get(layer) or {}).get(cls) or {}).get(
                    "opp_fire_rank_score"
                )
            if not isinstance(pc, (int, float)):
                pc = (by_method.get(f"sae:{cls}:L{layer}") or {}).get("metrics", {}).get(
                    "layer_rank_score"
                )
            if not isinstance(opp, (int, float)):
                opp = (by_method.get(f"sae:{cls}:L{layer}") or {}).get("metrics", {}).get(
                    "opp_fire_rank_score"
                )
            # layer_selection_scores is often {cls: score_at_selected_layer} only — skip for curve
            ys_pc.append(float(pc) if isinstance(pc, (int, float)) else float("nan"))
            ys_opp.append(float(opp) if isinstance(opp, (int, float)) else float("nan"))
        if any(v == v for v in ys_pc + ys_opp):
            any_real = True
        ax.plot(xs, ys_pc, marker="o", label="mean_diff_neutral")
        ax.plot(xs, ys_opp, marker="s", label="opp_fire")
        if cls in selected and selected[cls] is not None:
            sel_x = _optional_int_layer(selected[cls])
            if sel_x is not None:
                ax.axvline(sel_x, color="C0", ls="--", alpha=0.7, label="sel per_class")
        if cls in selected_opp and selected_opp[cls] is not None:
            opp_x = _optional_int_layer(selected_opp[cls])
            if opp_x is not None:
                ax.axvline(opp_x, color="C1", ls=":", alpha=0.7, label="sel opp_fire")
        # If only selected-layer scalar scores exist, annotate them.
        sel_x = _optional_int_layer(selected.get(cls))
        if isinstance(scores_pc.get(cls), (int, float)) and sel_x is not None:
            ax.scatter(
                [sel_x],
                [float(scores_pc[cls])],
                marker="*",
                s=100,
                color="C0",
                zorder=5,
                label="score@sel",
            )
            any_real = True
        ax.set_xlabel("layer")
        ax.set_ylabel("top-1 rank score")
        ax.set_title(f"SAE layer pick score ({cls})")
        ax.legend(fontsize=8)
        if not any(v == v for v in ys_pc + ys_opp):
            ax.text(
                0.5,
                0.5,
                "no per-layer rank scores in results\n(re-run SAE stage)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="0.4",
            )
    if not any_real:
        # Still write the figure so the gap is visible.
        pass
    fig.suptitle("SAE layer-selection rank scores", fontsize=11)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _causal_metric(row: Dict[str, Any], key: str = "causal_signed_effect") -> Optional[float]:
    return _safe_float((row.get("metrics") or {}).get(key))


def _parse_actiend_layer_causal(method: str) -> Optional[Tuple[str, int, str]]:
    """``actiend:M:L11_tok_all_gate_encoder_direction`` → (class, layer, tok_tail)."""
    m = re.fullmatch(
        r"actiend:([^:]+):L(\d+)_(tok_.+)",
        str(method),
    )
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def _parse_sae_k_causal(method: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """Return (class, kind, k) for ``sae:M:k8`` / ``sae_pre:M:kstar`` / tok ablations."""
    mid = normalize_sae_method_id(str(method))
    for backend in ("sae", "sae_pre"):
        m = re.fullmatch(rf"{backend}:([^:]+):k(\d+)(?:_tok_prediction)?", mid)
        if m:
            kind = "k1_tok_prediction" if mid.endswith("_tok_prediction") else f"k{m.group(2)}"
            return m.group(1), kind, int(m.group(2))
        m = re.fullmatch(rf"{backend}:([^:]+):kstar", mid)
        if m:
            return m.group(1), "kstar", None
        m = re.fullmatch(rf"{backend}:([^:]+):sel_opp_fire", mid)
        if m:
            return m.group(1), "sel_opp_fire", 1
        m = re.fullmatch(rf"{backend}:joint:([^:]+)", mid)
        if m:
            return m.group(1), "joint", 1
    return None


def _causal_rows_index(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(r.get("method")): r
        for r in (results.get("methods") or [])
        if r.get("method")
    }


def plot_causal_effect_by_layer(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
    tok_tail: str = "tok_all_gate_encoder_direction",
) -> Optional[Path]:
    """Per-layer causal: x=layer; ACTIEND default tok + SAE L*_k1 (when present)."""
    try:
        plt = _pyplot()
    except Exception:
        print("plot_causal_effect_by_layer: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by = _causal_rows_index(results)
    act_series: Dict[str, List[Tuple[int, float]]] = {c: [] for c in target_classes}
    sae_series: Dict[str, List[Tuple[int, float]]] = {c: [] for c in target_classes}
    ceiling_pts: List[Tuple[float, float, str]] = []

    for mid, row in by.items():
        parsed = _parse_actiend_layer_causal(mid)
        if parsed:
            cls, layer, tail = parsed
            if tail != tok_tail or cls not in act_series:
                continue
            eff = _causal_metric(row)
            if eff is None:
                continue
            act_series[cls].append((layer, eff))
            if (row.get("metrics") or {}).get("causal_grid_ceiling"):
                ceiling_pts.append((layer, eff, cls))
            continue
        m = re.fullmatch(r"sae(?:_pre)?:([^:]+):L(\d+)_k1", str(mid))
        if m:
            cls, layer = m.group(1), int(m.group(2))
            if cls not in sae_series:
                continue
            eff = _causal_metric(row)
            if eff is None:
                continue
            sae_series[cls].append((layer, eff))
            if (row.get("metrics") or {}).get("causal_grid_ceiling"):
                ceiling_pts.append((layer, eff, f"sae:{cls}"))

    if not any(act_series.values()) and not any(sae_series.values()):
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for cls, pts in act_series.items():
        if not pts:
            continue
        pts = sorted(pts, key=lambda t: t[0])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "o-",
            label=f"actiend:{cls}",
            markersize=5,
        )
    for cls, pts in sae_series.items():
        if not pts:
            continue
        pts = sorted(pts, key=lambda t: t[0])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "s--",
            label=f"sae:{cls}:L*_k1",
            markersize=5,
        )
    if ceiling_pts:
        ax.scatter(
            [p[0] for p in ceiling_pts],
            [p[1] for p in ceiling_pts],
            marker="^",
            s=70,
            c="0.2",
            zorder=6,
            label="grid ceiling (extend LR)",
        )

    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("signed causal effect")
    ax.set_title("Causal effect by layer (ACTIEND default tok; SAE top-1)")
    ax.legend(fontsize=7, ncol=2)
    layers = sorted(
        {
            p[0]
            for pts in list(act_series.values()) + list(sae_series.values())
            for p in pts
        }
    )
    if layers:
        ax.set_xticks(layers)
    if not any(sae_series.values()):
        ax.text(
            0.02,
            0.02,
            "SAE L*_k1 causal not in this results.json (re-run causal)",
            transform=ax.transAxes,
            fontsize=7,
            color="0.4",
        )
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_causal_sae_vs_k(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """SAE causal effect vs bag size k (log x).

    Fixed-k points form the line; k* is a star **on that ladder** at x=readout_k
    (same ablation family). ``sel_opp_fire`` is a separate top-1 ranking (always k=1),
    drawn as a diamond at x=1 — not a missing k-curve.
    """
    try:
        plt = _pyplot()
    except Exception:
        print("plot_causal_sae_vs_k: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)
    by = _causal_rows_index(results)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    any_pts = False
    for cls_i, cls in enumerate(target_classes):
        k_pts: List[Tuple[int, float]] = []
        kstar_xy: Optional[Tuple[float, float]] = None
        opp_y: Optional[float] = None
        for mid, row in by.items():
            parsed = _parse_sae_k_causal(mid)
            if not parsed or parsed[0] != cls:
                continue
            _c, kind, k = parsed
            eff = _causal_metric(row)
            if eff is None:
                continue
            m = row.get("metrics") or {}
            if kind == "kstar":
                k_star = int(m.get("readout_k") or k or 1)
                kstar_xy = (float(k_star), eff)
                continue
            if kind == "sel_opp_fire":
                opp_y = eff
                continue
            if kind == "joint" or kind.endswith("_tok_prediction"):
                continue
            if k is not None:
                k_pts.append((k, eff))
        color = f"C{cls_i}"
        if k_pts:
            any_pts = True
            k_pts = sorted(k_pts, key=lambda t: t[0])
            ax.plot(
                [p[0] for p in k_pts],
                [p[1] for p in k_pts],
                "o-",
                color=color,
                label=f"sae:{cls} fixed-k",
                markersize=5,
            )
        if kstar_xy is not None:
            any_pts = True
            ax.scatter(
                [kstar_xy[0]],
                [kstar_xy[1]],
                marker="*",
                s=160,
                color=color,
                edgecolors="k",
                linewidths=0.5,
                zorder=6,
                label=f"sae:{cls}:$k^*$ (k={int(kstar_xy[0])})",
            )
        if opp_y is not None:
            any_pts = True
            ax.scatter(
                [1.0],
                [opp_y],
                marker="D",
                s=70,
                color=color,
                facecolors="none",
                linewidths=HOLLOW_EDGE_WIDTH,
                zorder=5,
                label=f"sae:{cls}:opp_fire (k=1 only)",
            )

    if not any_pts:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xlabel("k (bag size)")
    ax.set_ylabel("signed causal effect")
    ax.set_title("SAE causal vs k (star=$k^*$; diamond=opp_fire top-1)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output




def _causal_group_specs(
    results: Dict[str, Any],
    target_classes: Sequence[str],
) -> Dict[str, List[str]]:
    """Named method-id groups for readable causal figures."""
    by = _causal_rows_index(results)

    def _has_causal(mid: str) -> bool:
        m = (by.get(mid) or {}).get("metrics") or {}
        return (
            m.get("causal_signed_effect") is not None
            or m.get("causal_selected_strength") is not None
            or m.get("causal_error") is not None
        )

    present = {mid for mid in by if _has_causal(mid)}
    groups: Dict[str, List[str]] = {}

    headline: List[str] = []
    for cls in target_classes:
        for mid in (
            f"gradiend:{cls}",
            f"actiend:{cls}:tok_all_gate_encoder_direction",
            f"sae:{cls}:k1",
            f"sae_pre:{cls}:k1",
            f"sae:{cls}:kstar",
            f"sae_pre:{cls}:kstar",
            f"sae:{cls}:sel_opp_fire",
            f"sae_pre:joint:{cls}",
            f"sae:joint:{cls}",
        ):
            if mid in present:
                headline.append(mid)
    groups["causal_bars_headline"] = headline

    tok: List[str] = []
    for cls in target_classes:
        for mid in (
            f"actiend:{cls}:tok_all_gate_encoder_direction",
            f"actiend:{cls}:tok_all",
            f"actiend:{cls}:tok_prediction",
            f"sae:{cls}:k1",
            f"sae_pre:{cls}:k1",
            f"sae:{cls}:k1_tok_prediction",
            f"sae_pre:{cls}:k1_tok_prediction",
        ):
            if mid in present:
                tok.append(mid)
    groups["causal_bars_tok"] = tok

    split_all: List[str] = []
    has_all = False
    for cls in target_classes:
        for mid in (
            f"gradiend:{cls}",
            f"gradiend:{cls}:tensors",
            f"gradiend:{cls}:all",  # legacy
            f"actiend:{cls}:tok_all_gate_encoder_direction",
            f"actiend:{cls}:tensors_tok_all_gate_encoder_direction",
            f"actiend:{cls}:tensors_tok_all",
            f"actiend:{cls}:tensors_tok_prediction",
            f"actiend:{cls}:all_tok_all_gate_encoder_direction",  # legacy
            f"actiend:{cls}:all_tok_all",
            f"actiend:{cls}:all_tok_prediction",
        ):
            if mid in present:
                split_all.append(mid)
                tail = mid.split(":")[-1]
                if (
                    mid.endswith(":tensors")
                    or mid.endswith(":all")
                    or tail.startswith("tensors_tok_")
                    or tail.startswith("all_tok_")
                ):
                    has_all = True
    if has_all:
        groups["causal_bars_none_vs_all"] = split_all

    sae_k: List[str] = []
    for mid in sorted(present):
        if not mid.startswith("sae:"):
            continue
        if ":k" in mid or mid.endswith(":sel_opp_fire") or mid.startswith("sae:joint:"):
            sae_k.append(mid)
    groups["causal_bars_sae"] = sae_k

    layers: List[str] = []
    for mid in sorted(
        present,
        key=lambda m: (_parse_actiend_layer_causal(m) or ("z", 10**9, ""))[1],
    ):
        if _parse_actiend_layer_causal(mid):
            layers.append(mid)
    groups["causal_bars_actiend_layers"] = layers
    return {k: v for k, v in groups.items() if v}


def plot_causal_none_vs_all(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Paired bars: none-split vs by_tensor :tensors causal (when :tensors rows exist)."""
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_causal_none_vs_all: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by = _causal_rows_index(results)
    labels: List[str] = []
    none_vals: List[float] = []
    all_vals: List[float] = []
    for cls in target_classes:
        # GRADIEND
        g_none = _causal_metric(by.get(f"gradiend:{cls}") or {})
        g_all = _causal_metric(
            by.get(f"gradiend:{cls}:tensors") or by.get(f"gradiend:{cls}:all") or {}
        )
        if g_none is not None or g_all is not None:
            labels.append(f"gradiend:{cls}")
            none_vals.append(g_none if g_none is not None else float("nan"))
            all_vals.append(g_all if g_all is not None else float("nan"))
        # ACTIEND default policy
        a_none = _causal_metric(
            by.get(f"actiend:{cls}:tok_all_gate_encoder_direction") or {}
        )
        a_all = _causal_metric(
            by.get(f"actiend:{cls}:tensors_tok_all_gate_encoder_direction")
            or by.get(f"actiend:{cls}:all_tok_all_gate_encoder_direction")
            or {}
        )
        if a_none is not None or a_all is not None:
            labels.append(f"actiend:{cls}")
            none_vals.append(a_none if a_none is not None else float("nan"))
            all_vals.append(a_all if a_all is not None else float("nan"))

    if not labels or not any(v == v for v in none_vals + all_vals):
        return None
    # Need at least one finite :tensors value to be meaningful.
    if not any(v == v for v in all_vals):
        return None

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(labels)), 4.0))
    ax.bar(x - width / 2, none_vals, width, label="none")
    ax.bar(x + width / 2, all_vals, width, label="by_tensor :tensors")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("signed causal effect")
    ax.set_title("Causal: none vs by_tensor :tensors")
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_causal_actiend_tok(
    results: Dict[str, Any],
    *,
    output: Path,
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Grouped bars: ACTIEND tok policy ablations (none-split)."""
    try:
        plt = _pyplot()
        import numpy as np
    except Exception:
        print("plot_causal_actiend_tok: matplotlib unavailable; skipping")
        return None

    target_classes = _require_plot_classes(target_classes)

    by = _causal_rows_index(results)
    policies = (
        ("gate", "tok_all_gate_encoder_direction"),
        ("tok_all", "tok_all"),
        ("tok_pred", "tok_prediction"),
    )
    x = np.arange(len(target_classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    any_pts = False
    for i, (lab, tail) in enumerate(policies):
        vals = []
        for cls in target_classes:
            mid = f"actiend:{cls}:{tail}"
            eff = _causal_metric(by.get(mid) or {})
            vals.append(eff if eff is not None else float("nan"))
            if eff is not None:
                any_pts = True
        ax.bar(x + (i - 1) * width, vals, width, label=lab)
    if not any_pts:
        plt.close(fig)
        return None
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"actiend:{c}" for c in target_classes])
    ax.set_ylabel("signed causal effect")
    ax.set_title("ACTIEND causal: token / gate policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strip_figure_titles(fig)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def write_all_causal_plots(
    results: Dict[str, Any],
    plots_dir: Path,
    *,
    target_classes: Sequence[str],
) -> Dict[str, str]:
    """Causal figure bundle with readable groupings."""
    from causal_eval import apply_package_decoder_headlines

    apply_package_decoder_headlines(results)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    def _put(name: str, path: Optional[Path]) -> None:
        if path:
            written[name] = str(path)

    _put(
        "causal_effect_by_layer",
        plot_causal_effect_by_layer(
            results,
            output=plots_dir / "causal_effect_by_layer.png",
            target_classes=target_classes,
        ),
    )
    _put(
        "causal_sae_vs_k",
        plot_causal_sae_vs_k(
            results,
            output=plots_dir / "causal_sae_vs_k.png",
            target_classes=target_classes,
        ),
    )
    _put(
        "causal_actiend_tok",
        plot_causal_actiend_tok(
            results,
            output=plots_dir / "causal_actiend_tok.png",
            target_classes=target_classes,
        ),
    )
    _put(
        "causal_none_vs_all",
        plot_causal_none_vs_all(
            results,
            output=plots_dir / "causal_none_vs_all.png",
            target_classes=target_classes,
        ),
    )

    groups = _causal_group_specs(results, target_classes)
    for name, mids in groups.items():
        _put(
            name,
            plot_causal_bars(
                results,
                output=plots_dir / f"{name}.png",
                methods=mids,
            ),
        )
        # LMS curves only for compact groups (full layer dump is unreadable).
        if name in {
            "causal_bars_headline",
            "causal_bars_tok",
            "causal_bars_none_vs_all",
            "causal_bars_sae",
        }:
            lms_name = name.replace("causal_bars_", "lms_vs_strength_")
            _put(
                lms_name,
                plot_lms_vs_strength(
                    results,
                    output=plots_dir / f"{lms_name}.png",
                    methods=mids,
                ),
            )
    return written


def write_all_study_plots(results: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    """Write standard study figures; return map name → path."""
    from causal_eval import apply_package_decoder_headlines

    apply_package_decoder_headlines(results)
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    from results_schema import require_target_classes

    classes = require_target_classes(results)
    classes = [str(c) for c in classes]
    written: Dict[str, str] = {}

    def _put(name: str, path: Optional[Path]) -> None:
        if path:
            written[name] = str(path)

    _put(
        "layer_auc_n",
        plot_layer_profile(
            results, output=plots_dir / "layer_auc_n.png", metric="roc_auc", target_classes=classes
        ),
    )
    _put(
        "layer_auc_o",
        plot_layer_profile(
            results,
            output=plots_dir / "layer_auc_o.png",
            metric="roc_auc_other",
            target_classes=classes,
        ),
    )
    # Keep legacy name as alias of auc_n profile.
    if "layer_auc_n" in written:
        written["layer_profile"] = written["layer_auc_n"]
    _put(
        "layer_rank_scores",
        plot_layer_rank_scores(
            results, output=plots_dir / "layer_rank_scores.png", target_classes=classes
        ),
    )
    _put(
        "sae_k_curves",
        plot_k_curves(results, output=plots_dir / "sae_k_curves.png", target_classes=classes),
    )
    _put(
        "exclusivity_vs_k",
        plot_exclusivity_vs_k(
            results, output=plots_dir / "exclusivity_vs_k.png", target_classes=classes
        ),
    )
    _put("causal_bars", plot_causal_bars(results, output=plots_dir / "causal_bars.png"))
    _put(
        "lms_vs_strength",
        plot_lms_vs_strength(results, output=plots_dir / "lms_vs_strength.png"),
    )
    written.update(write_all_causal_plots(results, plots_dir, target_classes=classes))
    _put(
        "none_vs_tensor",
        plot_none_vs_tensor(
            results,
            output=plots_dir / "none_vs_tensor.png",
            metric="balanced_accuracy",
            target_classes=classes,
        ),
    )
    _put(
        "none_vs_tensor_panel",
        plot_none_vs_tensor_panel(
            results, output=plots_dir / "none_vs_tensor_panel.png", target_classes=classes
        ),
    )
    _put(
        "none_vs_tensor_delta",
        plot_none_vs_tensor_delta(
            results, output=plots_dir / "none_vs_tensor_delta.png", target_classes=classes
        ),
    )
    return written
