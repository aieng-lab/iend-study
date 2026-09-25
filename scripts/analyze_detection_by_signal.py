#!/usr/bin/env python
"""Value vs gradient DETECTION across the full factorial grid, on the study's real
Det metric (composite worst-case), estimator-free where the method is a mean.

Det = min(roc_auc_other, roc_auc_neutral, neutral_specificity, class_exclusivity)
(suitability convention). Read one-pole method rows from suite_full2:
  value:        caa:<c>:act_prediction (mean),  actiend:<c> (learned)
  act-gradient: caga:<c> (mean),                agiend:<c> (learned)
  weight-grad:  cga:<c> (mean),                 gradiend:<c> (learned)
Reports per-method mean Det/AUC_o, and the paired value-minus-gradient contrasts
for both estimator rows. Reads results.json only -> CPU anywhere.
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMP = ["roc_auc_other", "roc_auc_neutral", "neutral_specificity", "class_exclusivity"]
TASKS = "gender_en emotion language race religion ravel_continent ravel_country ravel_language key_value ioi_mib induction function_composition".split()
METHODS = {  # label -> (id template, signal, estimator)
    "caa": ("caa:{c}:act_prediction", "value", "mean"),
    "actiend": ("actiend:{c}", "value", "learned"),
    "caga": ("caga:{c}", "act-grad", "mean"),
    "agiend": ("agiend:{c}", "act-grad", "learned"),
    "cga": ("cga:{c}", "weight-grad", "mean"),
    "gradiend": ("gradiend:{c}", "weight-grad", "learned"),
}
PAIRS = [  # (value method, gradient method) same estimator
    ("caa", "caga"), ("caa", "cga"),
    ("actiend", "agiend"), ("actiend", "gradiend"),
]


def _metrics(ms, mid):
    for r in ms:
        if str(r.get("method")) == mid:
            return r.get("metrics") or {}
    return None


def _det(m):
    vals = [m.get(k) for k in COMP]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return min(vals) if len(vals) == len(COMP) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gpt2-small", "pythia-70m-deduped"])
    ap.add_argument("--subdir", default="suite_full2")
    args = ap.parse_args(argv)

    per_method = collections.defaultdict(lambda: collections.defaultdict(list))
    cells = []  # (model,task,c) -> {method: {det, auc_o}}
    for model in args.models:
        for task in TASKS:
            f = ROOT / "runs" / model / args.subdir / task / "results.json"
            if not f.exists():
                continue
            ms = json.loads(f.read_text(encoding="utf-8")).get("methods", [])
            classes = sorted({str(r["method"]).split(":")[1] for r in ms
                              if str(r.get("method", "")).startswith("caga:") and str(r["method"]).count(":") == 1})
            for c in classes:
                cell = {}
                for lab, (tmpl, _sig, _est) in METHODS.items():
                    m = _metrics(ms, tmpl.format(c=c))
                    if not m:
                        continue
                    d, a = _det(m), m.get("roc_auc_other")
                    cell[lab] = {"det": d, "auc_o": float(a) if isinstance(a, (int, float)) else None}
                    if d is not None:
                        per_method[lab]["det"].append(d)
                    if isinstance(a, (int, float)):
                        per_method[lab]["auc_o"].append(float(a))
                cells.append(cell)

    print(f"{'method':10s} {'signal':11s} {'est':8s} {'mean Det':>9s} {'mean AUC_o':>11s} {'n':>4s}")
    for lab, (_t, sig, est) in METHODS.items():
        d = per_method[lab]["det"]; a = per_method[lab]["auc_o"]
        md = f"{np.mean(d):.3f}" if d else "-"
        ma = f"{np.mean(a):.3f}" if a else "-"
        print(f"{lab:10s} {sig:11s} {est:8s} {md:>9s} {ma:>11s} {len(d):>4d}")

    print("\nPaired value - gradient contrasts (same cell, same estimator):")
    print(f"{'contrast':24s} {'metric':8s} {'mean Δ':>8s} {'>0':>8s} {'median':>8s}")
    for vm, gm in PAIRS:
        for metric in ("det", "auc_o"):
            diffs = [cell[vm][metric] - cell[gm][metric] for cell in cells
                     if vm in cell and gm in cell
                     and isinstance(cell[vm][metric], (int, float)) and isinstance(cell[gm][metric], (int, float))]
            if not diffs:
                continue
            v = np.array(diffs)
            print(f"{vm+'-'+gm:24s} {metric:8s} {v.mean():+8.3f} {str(int((v>0).sum()))+'/'+str(len(v)):>8s} {np.median(v):+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
