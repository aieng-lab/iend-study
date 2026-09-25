"""Concise family × task overview: best-per-class, then mean [min–max].

Families are GRADIEND / ACTIEND / SAE / SAE-PRE / CAA (not per-class or k*/layer ids).
For each feature class we take the best variant in that family; the table cell
is the mean of those per-class bests, annotated with min–max.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analysis.task_specs import (
    GAP_DISPLAY,
    MISSING_DISPLAY,
    family_is_applicable,
    resolve_spec,
)
from analysis.task_order import order_tasks
from results_schema import (
    _has_encoder_metrics,
    display_method_id,
    require_claim_classes,
    require_target_classes,
)
from suitability import compute_feature_suitability, resolve_causal_metrics
from suitability import detection_score, intervention_score
from study.method_ids import normalize_sae_method_id

ROOT = Path(__file__).resolve().parents[1]

FAMILIES: Tuple[str, ...] = ("gradiend", "actiend", "actiend_ridge", "actiend_pre", "sae", "sae_pre", "caa")

# CGA (``cga_eval.py``) is deliberately NOT in ``FAMILIES``: that tuple is the
# unconditional default row set every overview/common-support/report caller
# falls back to when it doesn't pass an explicit ``families=`` -- unlike
# actiend_ridge/sae_pre (also opt-in, but always co-trained alongside their
# parent family whenever the study runs), CGA is only requested on `full`/
# `full_plus` suites. Adding it here would put an all-``gap`` CGA row into
# every overview table for every task that never requested it, and would make
# ``common_support_mask`` (an "every listed family was scored" intersection)
# empty everywhere CGA data is absent -- i.e. everywhere except a `full`
# suite run. A caller that specifically wants CGA in the grid passes
# ``families=(*FAMILIES, "cga")`` (or ``"cga_tensor_norm"``) explicitly.
CGA_FAMILIES: Tuple[str, ...] = ("cga", "cga_tensor_norm")
# CAGA (activation-gradient mean-diff) shares the GRADIEND id shape (``caga:F-M:F``
# / ``caga:F``) and, like CGA, is not in FAMILIES by default -- included only when
# the data contains it. NOTE ``"caga".startswith("cga")`` is False, so the gates
# below must test caga explicitly, not lean on the cga prefix check.
CAGA_FAMILIES: Tuple[str, ...] = ("caga", "agiend")


def _is_gradiend_shaped_family(family: str) -> bool:
    """True for families whose method ids share the GRADIEND shape (incl. cga/caga/agiend)."""
    f = str(family)
    return f in {"gradiend", "actiend", "actiend_pre"} or f.startswith("cga") or f in CAGA_FAMILIES

OVERVIEW_METRICS: Tuple[str, ...] = (
    "encoding_E",
    "roc_auc_neutral",
    "roc_auc_other",
    "neutral_specificity",
    "class_exclusivity",
    "suitability",
    "suitability_E",
    "suitability_G",
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "causal_lms",
    "causal_weaken_lms",
)

_LAYERISH = re.compile(r"^L\d+")


def _encoding_e(m: Mapping[str, Any]) -> Optional[float]:
    auc_n = m.get("roc_auc_neutral", m.get("roc_auc"))
    if auc_n is None:
        return None
    auc_o = m.get("roc_auc_other", m.get("min_pairwise_auroc"))
    excl = m.get("class_exclusivity")
    spec = m.get("neutral_specificity", m.get("specificity"))
    if auc_o is not None and excl is not None:
        return float(min(float(auc_n), float(auc_o), float(excl)))
    if spec is not None:
        return float(min(float(auc_n), float(spec)))
    return float(auc_n)


def _iter_results_json(runs_root: Path) -> Iterable[Path]:
    runs_root = Path(runs_root)
    seen: set[Path] = set()
    for pattern in ("*/*/results.json", "*/results.json"):
        for path in sorted(runs_root.glob(pattern)):
            parts = path.parts
            model_dir = parts[-3] if len(parts) >= 3 else ""
            if str(model_dir).endswith("_old"):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


def _load_results(path: Path) -> Optional[Dict[str, Any]]:
    from study.snapshot_guard import assert_resolved

    assert_resolved(path)  # unmerged pre-pull snapshot => file is incomplete
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    from causal_eval import apply_package_decoder_headlines

    return apply_package_decoder_headlines(payload)


def _parts(method_id: str) -> List[str]:
    return [p for p in str(method_id).split(":") if p]


def feature_class_for_method(
    method_id: str,
    *,
    known_classes: Sequence[str],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Map a method id to its feature class, or None if not a class readout."""
    known = {str(c) for c in known_classes}
    parts = _parts(method_id)
    # CGA families (``cga`` / ``cga_tensor_norm``) are not in FAMILIES by default
    # but are valid readouts sharing the GRADIEND id shape -- accept them here so
    # the cga-handling branch below is reachable (it was dead code otherwise).
    if not parts or (
        parts[0] not in FAMILIES and not _is_gradiend_shaped_family(parts[0])
    ):
        return None
    family = parts[0]
    metrics = metrics or {}

    # CGA/CAGA ids share the GRADIEND shape (``cga:F-M:F`` / ``caga:F``).
    if _is_gradiend_shaped_family(family):
        if len(parts) == 1:
            return None
        # Prefer explicit class token (pair-qualified or one-pole).
        for token in reversed(parts[1:]):
            if token in known:
                return token
            if token in {"tensors", "all"} or _LAYERISH.match(token) or token.startswith("tok_"):
                continue
        # Legacy pair row: metrics.pair + method gradiend:white
        pair = metrics.get("pair")
        if str(metrics.get("ablation") or "") == "pair" and isinstance(pair, (list, tuple)):
            for token in parts[1:]:
                if token in known:
                    return token
        return None

    if family in {"sae", "sae_pre", "caa"}:
        if len(parts) >= 2 and parts[1] in known:
            return parts[1]
        return None
    return None


def _metric_value(metrics: Mapping[str, Any], metric: str) -> Optional[float]:
    import math

    if metric == "encoding_E":
        v = metrics.get("encoding_E")
        if v is None:
            v = _encoding_e(metrics)
    elif metric == "detection_score":
        v = detection_score(metrics)
    elif metric == "intervention_score":
        v = intervention_score(metrics)
    elif metric == "neutral_specificity":
        v = metrics.get("neutral_specificity", metrics.get("specificity"))
    else:
        v = metrics.get(metric)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    out = float(v)
    if math.isnan(out):
        return None
    return out


def _enrich_metrics(results: Mapping[str, Any], method_id: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    out["encoding_E"] = _encoding_e(out)
    cau, cau_src = resolve_causal_metrics(results, method_id)
    raw = (results.get("raw") or {}).get("suitability") or {}
    cfg = (results.get("config") or {}).get("suitability") or {}
    tau = float(raw.get("tau_c", cfg.get("causal_effect_threshold", 0.05)))
    soft = bool(raw.get("soft_causal", cfg.get("soft_causal", False)))
    e_ok = float(raw.get("e_ok", cfg.get("e_ok", 0.8)))
    for key in (
        "causal_signed_effect",
        "causal_signed_effect_weaken",
        "causal_lms",
        "causal_weaken_lms",
        "causal_weaken_lms_ok",
        "causal_lms_ok",
        "causal_selected_strength",
        "causal_delta_mean",
        "causal_gate_empty",
        "causal_grid_floor",
        "causal_null_effect",
    ):
        if out.get(key) is None and cau.get(key) is not None:
            out[key] = cau.get(key)
    if cau_src and cau_src != method_id:
        out["causal_source_method"] = cau_src
    if out.get("suitability") is None or out.get("suitability_E") is None:
        out.update(compute_feature_suitability(out, cau, tau_c=tau, soft_causal=soft, e_ok=e_ok))
    return out


def _candidate_rows(
    results: Mapping[str, Any],
    *,
    known_classes: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in results.get("methods") or []:
        if not isinstance(row, Mapping):
            continue
        st = row.get("status")
        metrics = dict(row.get("metrics") or {})
        # Include causal-failed rows that still have encoder metrics (legacy status=error).
        if st not in {"ok", "partial", "collapsed"}:
            if st != "error" or not _has_encoder_metrics(metrics):
                continue
        mid = normalize_sae_method_id(display_method_id(row))
        parts = _parts(mid)
        if not parts or (
            parts[0] not in FAMILIES and not _is_gradiend_shaped_family(parts[0])
        ):
            continue
        cls = feature_class_for_method(mid, known_classes=known_classes, metrics=metrics)
        if cls is None:
            continue
        # Encoder-less causal-only ids still allowed if they carry causal fields after enrich.
        enriched = _enrich_metrics(results, mid, metrics)
        out.append(
            {
                "method": mid,
                "family": parts[0],
                "feature_class": cls,
                "status": st,
                "metrics": enriched,
                "has_encoder": _has_encoder_metrics(metrics),
            }
        )
    return out


def best_per_class_for_results(
    results: Mapping[str, Any],
    *,
    metric: str,
    results_path: str = "",
) -> List[Dict[str, Any]]:
    """One row per (family, feature_class) with the winning method for ``metric``."""
    parts = Path(results_path).parts if results_path else ()
    model = str(results.get("model") or (parts[-3] if len(parts) >= 3 else "unknown"))
    task = str(results.get("task") or (parts[-2] if len(parts) >= 2 else "unknown"))
    try:
        known = sorted(set(require_claim_classes(dict(results))) | set(require_target_classes(dict(results))))
    except ValueError:
        return []

    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cand in _candidate_rows(results, known_classes=known):
        val = _metric_value(cand["metrics"], metric)
        if val is None:
            continue
        # Prefer encoder-bearing rows when values tie / for encoder metrics.
        key = (str(cand["family"]), str(cand["feature_class"]))
        prev = best.get(key)
        if prev is None or val > float(prev["value"]) + 1e-15:
            best[key] = {
                "model": model,
                "task": task,
                "family": cand["family"],
                "feature_class": cand["feature_class"],
                "metric": metric,
                "value": val,
                "winner_method": cand["method"],
                "results_path": results_path,
            }
        elif abs(val - float(prev["value"])) <= 1e-15 and cand.get("has_encoder") and not prev.get("_enc"):
            best[key] = {
                "model": model,
                "task": task,
                "family": cand["family"],
                "feature_class": cand["feature_class"],
                "metric": metric,
                "value": val,
                "winner_method": cand["method"],
                "results_path": results_path,
                "_enc": True,
            }
    for row in best.values():
        row.pop("_enc", None)
    return [best[k] for k in sorted(best)]


def aggregate_family_cells(
    per_class: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse per-class bests into family cells: mean / min / max."""
    buckets: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in per_class:
        key = (
            str(row["model"]),
            str(row["task"]),
            str(row["family"]),
            str(row["metric"]),
        )
        buckets.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (model, task, family, metric), rows in sorted(buckets.items()):
        vals = [float(r["value"]) for r in rows]
        winners = sorted({str(r["winner_method"]) for r in rows})
        classes = sorted({str(r["feature_class"]) for r in rows})
        out.append(
            {
                "model": model,
                "task": task,
                "family": family,
                "metric": metric,
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n_classes": len(vals),
                "classes": ",".join(classes),
                "winner_methods": ",".join(winners),
            }
        )
    return out


def collect_family_overview(
    runs_root: Path = ROOT / "runs",
    *,
    metrics: Sequence[str] = OVERVIEW_METRICS,
    model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (family_cells, per_class_bests) across all results.json dumps."""
    per_class: List[Dict[str, Any]] = []
    for path in _iter_results_json(Path(runs_root)):
        payload = _load_results(path)
        if not payload:
            continue
        if model is not None:
            path_model = path.parts[-3] if len(path.parts) >= 3 else ""
            run_model = str(payload.get("model") or path_model)
            if run_model != model:
                continue
        for metric in metrics:
            per_class.extend(
                best_per_class_for_results(payload, metric=metric, results_path=str(path))
            )
    cells = aggregate_family_cells(per_class)
    return cells, per_class


def format_cell(
    mean: Optional[float],
    lo: Optional[float],
    hi: Optional[float],
    *,
    expected: bool = True,
    width: int = 0,
) -> str:
    import math

    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        body = GAP_DISPLAY if not expected else MISSING_DISPLAY
        return body if not width else f"{body:>{width}}"
    if lo is None or hi is None or (abs(hi - lo) < 1e-12 and abs(mean - lo) < 1e-12):
        body = f"{mean:.3f}"
    else:
        # ASCII hyphen for Windows consoles / plain-text viewers.
        body = f"{mean:.3f} [{lo:.3f}-{hi:.3f}]"
    return body if not width else f"{body:>{width}}"


def pivot_family_task(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    families: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Mapping[str, Any]]]:
    """Row-label × tasks lookup of cell dicts for one metric."""
    sub = [c for c in cells if c.get("metric") == metric]
    if model:
        sub = [c for c in sub if c.get("model") == model]
    tasks = order_tasks(list({str(c["task"]) for c in sub}))
    if specs:
        extra = order_tasks(
            list({task for (mod, task) in specs if model is None or mod == model})
        )
        tasks = list(dict.fromkeys([*tasks, *extra]))
    row_order = list(families) if families is not None else list(FAMILIES)
    lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (str(c["family"]), str(c["task"])): c for c in sub
    }
    return row_order, tasks, lookup


def format_overview_text(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    families: Optional[Sequence[str]] = None,
    row_label: str = "family",
    title: Optional[str] = None,
    applicable=None,
) -> str:
    applicable_fn = applicable or family_is_applicable
    models = sorted({str(c["model"]) for c in cells if c.get("metric") == metric})
    if model:
        models = [m for m in models if m == model]
    if not models:
        return f"(no {row_label} overview cells for metric={metric})"

    blocks: List[str] = []
    for mod in models:
        rows, tasks, lookup = pivot_family_task(
            cells, metric=metric, model=mod, specs=specs, families=families
        )
        if not rows or not tasks:
            continue
        col_w = max(18, max((len(t) for t in tasks), default=4) + 2, 22)
        fam_w = max(10, max((len(f) for f in rows), default=6), len(row_label))
        header = f"{row_label:<{fam_w}}" + "".join(f"{t:>{col_w}}" for t in tasks) + f"{'MEAN':>{col_w}}"
        heading = (
            f"{title} ({metric}) [{mod}]"
            if title
            else f"Family x task -- best-per-class then mean [min-max] ({metric}) [{mod}]"
        )
        lines = [
            heading,
            f"{GAP_DISPLAY} = expected gap (not applicable); {MISSING_DISPLAY} = missing value",
            header,
            "-" * len(header),
        ]
        for fam in rows:
            row = f"{fam:<{fam_w}}"
            means: List[float] = []
            for task in tasks:
                cell = lookup.get((fam, task))
                spec = resolve_spec(specs, model=mod, task=task)
                expected = applicable_fn(fam, spec)
                if not cell:
                    row += format_cell(None, None, None, expected=expected, width=col_w)
                    continue
                means.append(float(cell["mean"]))
                row += format_cell(cell["mean"], cell["min"], cell["max"], width=col_w)
            mean_all = sum(means) / len(means) if means else None
            row += format_cell(mean_all, None, None, expected=True, width=col_w)
            lines.append(row)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else f"(no {row_label} overview cells for metric={metric})"


def format_overview_latex(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: Optional[str] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    families: Optional[Sequence[str]] = None,
    row_label: str = "family",
    applicable=None,
    label_prefix: str = "family",
) -> str:
    """booktabs-style tabular; one table per model when ``model`` is None."""
    applicable_fn = applicable or family_is_applicable
    models = sorted({str(c["model"]) for c in cells if c.get("metric") == metric})
    if model:
        models = [m for m in models if m == model]
    if not models:
        return "% no data\n"

    chunks: List[str] = []
    for mod in models:
        rows, tasks, lookup = pivot_family_task(
            cells, metric=metric, model=mod, specs=specs, families=families
        )
        if not rows or not tasks:
            continue
        cols = "l" + "c" * len(tasks) + "c"
        metric_tex = str(metric).replace("_", "\\_")
        row_tex = row_label.replace("_", r"\_")
        cap = caption or (
            f"Family overview ({metric_tex}): "
            f"mean [min--max] of per-class bests for {mod}."
        )
        lab = label or f"tab:{label_prefix}-{metric}-{mod}"
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{cap}}}",
            f"\\label{{{lab}}}",
            f"\\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            f"{row_tex} & " + " & ".join(t.replace("_", r"\_") for t in tasks) + r" & MEAN \\",
            r"\midrule",
        ]
        for fam in rows:
            cells_tex: List[str] = [fam.replace("_", r"\_")]
            means: List[float] = []
            for task in tasks:
                cell = lookup.get((fam, task))
                spec = resolve_spec(specs, model=mod, task=task)
                expected = applicable_fn(fam, spec)
                if not cell:
                    cells_tex.append(GAP_DISPLAY if not expected else MISSING_DISPLAY)
                    continue
                means.append(float(cell["mean"]))
                # LaTeX en-dash in ranges
                if abs(float(cell["max"]) - float(cell["min"])) < 1e-12:
                    cells_tex.append(f"{float(cell['mean']):.3f}")
                else:
                    cells_tex.append(
                        f"{float(cell['mean']):.3f} [{float(cell['min']):.3f}--{float(cell['max']):.3f}]"
                    )
            mean_all = sum(means) / len(means) if means else None
            cells_tex.append(MISSING_DISPLAY if mean_all is None else f"{mean_all:.3f}")
            lines.append(" & ".join(cells_tex) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        chunks.append("\n".join(lines))
    return "\n".join(chunks)
