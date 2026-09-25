#!/usr/bin/env python
"""Export the single canonical benchmark dataframe (paper artifact #1).

One flat, long-format CSV at the rawest grain available on disk:

    (model, task, method_group, construction, target_class)

with every headline **detection** and **intervention** metric as a column.
This is the single source of truth every downstream paper table/figure
should aggregate from -- macro method comparison, factorial paired
contrasts, detection-vs-intervention scatter, and the full per-task
appendix are all ``groupby`` views of this file. Aggregate on the fly;
do not pre-average here.

Grain note: the training pipeline selects one convergent checkpoint per
cell, so a *seed* axis does not exist in ``results.json`` -- this is the
finest grain the stored results support. ``target_class == "__pooled__"``
marks the bare pair aggregate (a class-pooled readout with no single
class); drop it when averaging over classes so it is not double counted.

Rows come from ``method_groups.collect_class_rows_for_results`` -- the same
canonical E-locked members the paper's group tables average -- so the CSV
and the LaTeX tables can never disagree about which readout site is
canonical for a method/class.

Data source per model (``--subdir``): gpt2-small / pythia-70m-deduped live
under the ``suite_full2`` OUTPUT_SUBDIR tree; larger models run later have
no subdir (default ``runs/{model}/{task}/``). Pass ``--subdir ""`` for those.

Examples::

  # both small models from their suite_full2 trees (default)
  python analysis/export_benchmark.py

  # a future large model trained in the default tree
  python analysis/export_benchmark.py --models llama-3.1-8b --subdir ""

  python analysis/export_benchmark.py --out analysis/tables/benchmark_long.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.method_groups import _load_results, collect_class_rows_for_results

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_MODELS = ("gpt2-small", "pythia-70m-deduped")
DEFAULT_SUBDIR = "suite_full2"
DEFAULT_OUT = ROOT / "analysis" / "tables" / "benchmark_long.csv"

# Denormalised task descriptors carried on every row so a table generator can
# filter (binary vs multiclass, pole availability) without a second pass.
def _task_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = payload.get("config") or {}
    target = cfg.get("target_classes") or []
    claim = cfg.get("claim_classes") or []
    abl = cfg.get("ablations") or {}
    n_target = len(target) if isinstance(target, list) else None
    pair_available = bool(abl.get("pair") or abl.get("pair_single") or abl.get("present_both"))
    return {
        "n_target_classes": n_target,
        "n_claim_classes": len(claim) if isinstance(claim, list) else None,
        "is_binary": (n_target is not None and n_target <= 2),
        "is_multiclass": (n_target is not None and n_target > 2),
        "pair_available": pair_available,
        "one_pole_available": bool(abl.get("one_pole")),
        "config_suite": payload.get("suite"),
        "run_pipeline_stage": payload.get("pipeline_stage"),
    }


def _iter_results_paths(runs_root: Path, model: str, subdir: str) -> List[Path]:
    base = runs_root / model / subdir if subdir else runs_root / model
    if not base.is_dir():
        return []
    return sorted(base.glob("*/results.json"))


def collect_benchmark_rows(
    runs_root: Path,
    models: List[str],
    subdir: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in models:
        paths = _iter_results_paths(runs_root, model, subdir)
        if not paths:
            print(f"  [warn] no results.json under {runs_root / model / subdir}", file=sys.stderr)
            continue
        for path in paths:
            payload = _load_results(path)
            if not payload:
                continue
            meta = _task_metadata(payload)
            class_rows = collect_class_rows_for_results(payload, results_path=str(path))
            for row in class_rows:
                row["subdir"] = subdir
                row.update(meta)
                rows.append(row)
        print(f"  {model}/{subdir or '(default)'}: {len(paths)} tasks", file=sys.stderr)
    return rows


# Stable, human-first column order; any extra metric columns are appended.
_LEAD_COLUMNS = (
    "model",
    "task",
    "method_group",
    "backend",
    "construction",
    "target_class",
    "method",
    "n_target_classes",
    "n_claim_classes",
    "is_binary",
    "is_multiclass",
    "pair_available",
    "one_pole_available",
    "run_status",
    "row_status",
    "subdir",
    "config_suite",
    "run_pipeline_stage",
    # detection
    "encoding_E",
    "roc_auc_neutral",
    "roc_auc_other",
    "balanced_accuracy",
    "cohens_d",
    "class_exclusivity",
    "specificity",
    "neutral_specificity",
    "min_pairwise_auroc",
    "encoder_correlation",
    "class_separation",
    "suitability",
    # intervention
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "causal_base_p",
    "causal_lms",
    "causal_effectiveness",
    "results_path",
)


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    lead = [c for c in _LEAD_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="runs root (default: ./runs)")
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated model ids (default: gpt2-small,pythia-70m-deduped)",
    )
    ap.add_argument(
        "--subdir",
        default=DEFAULT_SUBDIR,
        help='OUTPUT_SUBDIR tree for every listed model (default: suite_full2). Pass "" for the default tree.',
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output CSV (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    print(f"Exporting benchmark: models={models} subdir={args.subdir or '(default)'}", file=sys.stderr)
    rows = collect_benchmark_rows(args.runs, models, args.subdir)
    if not rows:
        print("No rows collected -- check --runs/--models/--subdir.", file=sys.stderr)
        return 1

    df = _order_columns(pd.DataFrame(rows))
    df = df.sort_values(["model", "task", "method_group", "target_class"], kind="stable").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_causal = int(df["causal_signed_effect"].notna().sum()) if "causal_signed_effect" in df else 0
    print(
        f"Wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}\n"
        f"  models={df['model'].nunique()} tasks={df['task'].nunique()} "
        f"method_groups={df['method_group'].nunique()} rows_with_causal={n_causal}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
