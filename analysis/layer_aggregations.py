"""Canonical layer/all-layer aggregation for summary tables.

Legacy method ids remain in ``results.json``.  This module normalizes them into
explicit variants so tables never confuse a layer ablation with an all-layer
recipe.  Aggregates are analysis-only: no study rerun is needed for means.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from analysis.task_order import task_sort_key


METRICS = (
    "encoding_E", "roc_auc_neutral", "roc_auc_other", "neutral_specificity",
    "class_exclusivity", "causal_signed_effect", "causal_signed_effect_weaken",
    "causal_lms",
)

_LAYER = {
    "sae": re.compile(r"^sae:(?:[^:]+:)?([^:]+):L(\d+)(?:_k1)?$"),
    "sae_pre": re.compile(r"^sae_pre:(?:[^:]+:)?([^:]+):L(\d+)(?:_k1)?$"),
    "caa": re.compile(r"^caa:(?:[^:]+:)?([^:]+):L(\d+)_act_prediction$"),
    "cga": re.compile(r"^cga:(?:[^:]+:)?([^:]+):L(\d+)$"),
    "caga": re.compile(r"^caga:(?:[^:]+:)?([^:]+):L(\d+)$"),
}
_ALL = {
    "sae": re.compile(r"^sae:(?:[^:]+:)?([^:]+):all_k1$"),
    "sae_pre": re.compile(r"^sae_pre:(?:[^:]+:)?([^:]+):all_k1$"),
    "caa": re.compile(r"^caa:(?:[^:]+:)?([^:]+):all_act_prediction$"),
    "cga": re.compile(r"^cga:(?:[^:]+:)?([^:]+)$"),
    "caga": re.compile(r"^caga:(?:[^:]+:)?([^:]+)$"),
}


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _val_detection(metrics: Mapping[str, Any]) -> float | None:
    val = metrics.get("val_readout")
    if not isinstance(val, Mapping):
        return None
    fields = ("roc_auc_neutral", "neutral_specificity", "roc_auc_other", "class_exclusivity")
    vals = [float(val[k]) for k in fields if _num(val.get(k))]
    return min(vals) if len(vals) == len(fields) else None


def _metric(metrics: Mapping[str, Any], name: str) -> Any:
    if name == "encoding_E":
        value = metrics.get("encoding_E")
        if _num(value):
            return value
        vals = [metrics.get(k) for k in ("roc_auc_neutral", "roc_auc_other", "neutral_specificity", "class_exclusivity")]
        nums = [float(v) for v in vals if _num(v)]
        return min(nums) if len(nums) == 4 else None
    return metrics.get(name)


def _rows(results_root: Path, model: str, subdir: str = "") -> Iterable[Dict[str, Any]]:
    root = Path(results_root) / model / subdir if subdir else Path(results_root) / model
    for path in sorted(root.glob("*/results.json")):
        try:
            import json
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task = path.parent.name
        for row in payload.get("methods") or []:
            mid = str(row.get("method") or "")
            metrics = row.get("metrics") or {}
            for family, pattern in _LAYER.items():
                match = pattern.match(mid)
                if match:
                    yield {"model": model, "task": task, "family": family, "class": match.group(1),
                           "layer": int(match.group(2)), "method": mid, "kind": "layer", "metrics": metrics}
                    break
            else:
                for family, pattern in _ALL.items():
                    match = pattern.match(mid)
                    if match:
                        yield {"model": model, "task": task, "family": family, "class": match.group(1),
                               "layer": None, "method": mid, "kind": "all", "metrics": metrics}
                        break


def canonical_layer_aggregations(results_root: Path, model: str, subdir: str = "") -> pd.DataFrame:
    raw = list(_rows(results_root, model, subdir))
    if not raw:
        return pd.DataFrame()
    out: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in raw:
        groups.setdefault((row["task"], row["family"], row["class"]), []).append(row)
    for (task, family, cls), rows in groups.items():
        layers = sorted({int(r["layer"]) for r in rows if r["kind"] == "layer"})
        expected = (max(layers) + 1) if layers else 0
        native = next((r for r in rows if r["kind"] == "all"), None)
        single_rows = [r for r in rows if r["kind"] == "layer"]
        best = max(single_rows, key=lambda r: (_val_detection(r["metrics"]) is not None, _val_detection(r["metrics"]) or float("-inf")), default=None)
        for variant, selected, values, n_expected in (
            ("best_single", best, single_rows, expected),
            ("all_mean", None, single_rows, expected),
            ("all_native", native, [native] if native else [], 1),
            ("all_steer", native, [native] if native else [], 1),
        ):
            for metric in METRICS:
                if variant == "all_mean":
                    nums = [float(_metric(r["metrics"], metric)) for r in values if _num(_metric(r["metrics"], metric))]
                    value = sum(nums) / len(nums) if nums else None
                    n_present = len(nums)
                else:
                    value = _metric((selected or {}).get("metrics") or {}, metric)
                    n_present = int(_num(value))
                out.append({"model": model, "task": task, "family": family, "class": cls,
                            "variant": variant, "metric": metric, "value": value,
                            "n_present": n_present, "n_expected": n_expected,
                            "incomplete": bool(n_expected and n_present < n_expected),
                            "source_method": (selected or {}).get("method")})
    return pd.DataFrame(out)




def format_layer_aggregation_latex(frame: pd.DataFrame, *, model: str) -> str:
    """Compact task × family × variant table; ``*`` marks incomplete coverage."""
    if frame.empty:
        return ""
    rows: List[str] = []
    work = frame[frame["metric"].isin(["roc_auc_neutral", "causal_signed_effect"])].copy()
    groups = work.groupby(["task", "family", "variant", "metric"], sort=True)
    for (task, family, variant, metric), group in sorted(
        groups, key=lambda item: (task_sort_key(str(item[0][0])), *item[0][1:])
    ):
        nums = [float(v) for v in group["value"] if _num(v)]
        if not nums:
            continue
        value = sum(nums) / len(nums)
        incomplete = bool(group["incomplete"].any()) or len(nums) < len(group)
        label = f"{value:.3f}" + (r"\textsuperscript{*}" if incomplete else "")
        task_label = str(task).replace("_", r"\_")
        family_label = str(family).replace("_", r"\_")
        variant_label = str(variant).replace("_", r"\_")
        metric_label = str(metric).replace("_", r"\_")
        rows.append(
            f"{task_label} & {family_label} & {variant_label} & "
            f"{metric_label} & {label} " + r"\\"
        )
    if not rows:
        return ""
    model_label = str(model).replace("_", r"\_")
    return "\n".join([
        r"\begin{longtable}{llllr}",
        rf"\caption{{Layer aggregation diagnostics for {model_label}.}}\\",
        r"\toprule",
        r"Task & Family & Variant & Metric & Score \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Task & Family & Variant & Metric & Score \\",
        r"\midrule",
        r"\endhead",
        *rows,
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize \textsuperscript{*}Incomplete layer coverage or missing source row.}",
        r"\end{longtable}",
        "",
    ])
