#!/usr/bin/env python
"""Write overview tables from ``runs/*/results.json``.

1. Family x task (best-per-class -> mean [min-max]) -- canonical overview
2. Method-group ablation pivots (two_pole / one_pole / kstar / k1 / CAA)

Both blocks print every requested metric to the console.

  python analysis/summarize_runs.py
  python analysis/summarize_runs.py --model gpt2-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.family_overview import OVERVIEW_METRICS
from analysis.method_groups import (
    HEADLINE_METRICS,
    collect_group_rows,
    format_overview_text,
    format_pivot_display,
    pivot_group_task,
)
from analysis.summarize_family_overview import run_family_overview
from analysis.task_specs import collect_task_specs

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "analysis" / "tables"
DEFAULT_FAMILY_OUT = DEFAULT_OUT / "family"

# Metrics shown for both family and ablation pivots.
DEFAULT_PIVOT_METRICS = (
    "encoding_E",
    "roc_auc_neutral",
    "roc_auc_other",
    "neutral_specificity",
    "class_exclusivity",
    "suitability",
    "suitability_E",
    "suitability_G",
    "causal_signed_effect",  # steering ΔP at LMS-gated selected strength
    "causal_lms",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--family-out",
        type=Path,
        default=DEFAULT_FAMILY_OUT,
        help="Directory for family overview artifacts",
    )
    parser.add_argument("--model", default=None, help="Filter to one model key")
    parser.add_argument(
        "--metric",
        default=None,
        help="If set, only pivot this metric",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Pivot these metrics (default: encoding_E, auc_n/o, spec, excl, S/E/G)",
    )
    parser.add_argument(
        "--skip-family",
        action="store_true",
        help="Only write method-group ablation pivots",
    )
    parser.add_argument(
        "--skip-groups",
        action="store_true",
        help="Only write family best-per-class overview",
    )
    args = parser.parse_args()

    if args.metric:
        metrics = [args.metric]
    elif args.metrics:
        metrics = list(args.metrics)
    else:
        metrics = list(DEFAULT_PIVOT_METRICS)

    # Family metrics: requested set, plus any OVERVIEW_METRICS extras when using defaults.
    if args.metric or args.metrics:
        family_metrics = list(metrics)
    else:
        family_metrics = list(dict.fromkeys([*DEFAULT_PIVOT_METRICS, *OVERVIEW_METRICS]))

    if not args.skip_family:
        print("=== Family overview (best-per-class -> mean [min-max]) ===")
        run_family_overview(
            runs=args.runs,
            out=args.family_out,
            model=args.model,
            metrics=family_metrics,
            print_tables=True,
        )
        print()

    if args.skip_groups:
        return

    print("=== Method-group ablation pivots ===")
    df = collect_group_rows(args.runs)
    specs = collect_task_specs(args.runs)
    if args.model:
        df = df[df["model"] == args.model]

    args.out.mkdir(parents=True, exist_ok=True)
    long_path = args.out / "summary_groups_long.csv"
    df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(df)} rows)")

    if df.empty:
        print("No completed encoder rows found yet for method groups.")
        return

    for metric in metrics:
        if metric not in df.columns and metric not in HEADLINE_METRICS:
            continue
        pivot = pivot_group_task(df, metric=metric, model=args.model, specs=specs)
        if pivot.empty:
            print(f"(no pivot for {metric})")
            continue
        pivot_path = args.out / f"summary_pivot_{metric}.csv"
        format_pivot_display(pivot, specs=specs, model=args.model).to_csv(pivot_path)
        print(f"Wrote {pivot_path}")

    print()
    for i, metric in enumerate(metrics):
        if i:
            print()
        print(format_overview_text(df, metric=metric, model=args.model, specs=specs))


if __name__ == "__main__":
    main()
