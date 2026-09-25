"""Aggregate partial study runs into tidy tables + stub heatmaps.

For cross-task overviews without sparse method-id columns, prefer
``python analysis/summarize_runs.py`` (headline groups like ``gradiend:two_pole``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from error_tracker import track_error
from analysis.plot_style import strip_figure_titles

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "analysis" / "tables"
PLOT_DIR = ROOT / "analysis" / "plots"


def _iter_results(runs_root: Path):
    # Both normal ``runs/{model}/{task}`` and isolated study trees such as
    # ``runs/{model}/runtime_benchmark_l40s/{task}`` are first-class inputs.
    # A recursive search avoids silently losing the latter from appendix CSVs.
    yield from sorted(runs_root.rglob("results.json"))


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        track_error(exc, context="load results", path=str(path))
        return None


def _metric_flat(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float, str, bool)) or v2 is None:
                    out[f"{k}.{k2}"] = v2
    return out


def collect_rows(runs_root: Path = RUNS) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in _iter_results(runs_root):
        payload = _load(path)
        if not payload:
            continue
        parts = path.relative_to(runs_root).parts
        model = payload.get("model") or (parts[0] if parts else None)
        task = payload.get("task") or (parts[-2] if len(parts) >= 2 else None)
        suite = payload.get("suite")
        status = payload.get("status", "unknown")
        methods = payload.get("methods") or []
        if not methods:
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "suite": suite,
                    "method": None,
                    "run_status": status,
                    "method_status": None,
                    "present": False,
                    "results_path": str(path),
                }
            )
            continue
        for m in methods:
            flat = _metric_flat(m.get("metrics"))
            row = {
                "model": model,
                "task": task,
                "suite": suite,
                "method": m.get("method"),
                "proxy": m.get("proxy"),
                "eval_split": m.get("eval_split"),
                "ablation": m.get("ablation"),
                "run_status": status,
                "method_status": m.get("status", "ok"),
                "present": m.get("status", "ok") in {"ok", "partial"},
                "results_path": str(path),
                **flat,
            }
            # Causal nested
            causal = m.get("causal") or (m.get("extras") or {}).get("causal")
            if isinstance(causal, dict):
                for ck, cv in _metric_flat(causal).items():
                    row[f"causal.{ck}"] = cv
            rows.append(row)
    return pd.DataFrame(rows)


def collect_compute(runs_root: Path = RUNS) -> pd.DataFrame:
    """One row per run with headline ``compute`` / ``raw.cost_summary`` fields."""
    rows: List[Dict[str, Any]] = []
    keys = (
        "wall_total_s",
        "wall_train_s",
        "wall_feature_select_s",
        "wall_encode_s",
        "wall_causal_s",
        "wall_localization_s",
        "peak_gpu_train_gb",
        "peak_gpu_feature_select_gb",
        "peak_gpu_encode_gb",
        "peak_gpu_causal_gb",
        "peak_gpu_overall_gb",
        "peak_delta_train_gb",
        "peak_delta_encode_gb",
        "flops_proxy_train",
        "flops_proxy_encode",
        "n_timers",
        "cache_hits",
    )
    for path in _iter_results(runs_root):
        payload = _load(path)
        if not payload:
            continue
        parts = path.relative_to(runs_root).parts
        model = payload.get("model") or (parts[0] if parts else None)
        task = payload.get("task") or (parts[-2] if len(parts) >= 2 else None)
        comp = payload.get("compute") or ((payload.get("raw") or {}).get("cost_summary") or {})
        if not isinstance(comp, dict):
            comp = {}
        run_meta = comp.get("run") if isinstance(comp.get("run"), dict) else {}
        row: Dict[str, Any] = {
            "model": model,
            "task": task,
            "suite": payload.get("suite"),
            "run_status": payload.get("status", "unknown"),
            "results_path": str(path),
            "gpu_name": run_meta.get("gpu_name"),
            "n_params": run_meta.get("n_params"),
            "hf_model": run_meta.get("hf_model") or payload.get("hf_model"),
        }
        for k in keys:
            row[k] = comp.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_method_cost_ledger(runs_root: Path = RUNS) -> pd.DataFrame:
    """Flatten durable, method-attributed timing and GPU-memory evidence.

    ``raw.cost`` is intentionally just the latest process invocation.  For an
    appendix or a partial causal refresh use ``raw.cost_ledger`` instead: it
    retains the latest complete timer batch for each method family, and every
    row carries its own device/workload provenance in ``run``.
    """
    rows: List[Dict[str, Any]] = []
    for path in _iter_results(runs_root):
        payload = _load(path)
        if not payload:
            continue
        parts = path.relative_to(runs_root).parts
        raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
        ledger = raw.get("cost_ledger") or raw.get("cost") or []
        if not isinstance(ledger, list):
            continue
        model = payload.get("model") or (parts[0] if parts else None)
        task = payload.get("task") or (parts[-2] if len(parts) >= 2 else None)
        for item in ledger:
            if not isinstance(item, dict) or not item.get("backend"):
                continue
            run = item.get("run") if isinstance(item.get("run"), dict) else {}
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "suite": payload.get("suite"),
                    "backend": item.get("backend"),
                    "ledger_unit": item.get("ledger_unit"),
                    "phase": item.get("phase"),
                    "timer": item.get("name"),
                    "seconds": item.get("seconds"),
                    "peak_gpu_gb": item.get("peak_cuda_gb"),
                    "peak_gpu_reserved_gb": item.get("peak_cuda_reserved_gb"),
                    "peak_delta_gb": item.get("peak_delta_gb"),
                    "gpu_name": run.get("gpu_name"),
                    "gpu_total_memory_gb": run.get("gpu_total_memory_gb"),
                    "slurm_job_id": run.get("slurm_job_id"),
                    "run_started_at": run.get("run_started_at"),
                    "results_path": str(path),
                }
            )
    return pd.DataFrame(rows)


def summarize_method_cost_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Produce appendix-ready method × stage totals from timer-level evidence.

    Time is additive across layers, feature classes, and causal batches.  GPU
    RAM is not: the appendix therefore reports the maximum allocated, reserved,
    and incremental peak observed within each method-stage.  In particular,
    every SAE layer timer belongs to one ``sae_layer_sweep`` row rather than
    being presented as several cherry-picked SAE models.
    """
    columns = [
        "model", "task", "suite", "backend", "ledger_unit", "phase",
        "gpu_name", "gpu_total_memory_gb", "wall_seconds", "peak_gpu_gb",
        "peak_gpu_reserved_gb", "peak_delta_gb", "n_timers",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    group_columns = columns[:8]
    numeric = ledger.copy()
    for col in ("seconds", "peak_gpu_gb", "peak_gpu_reserved_gb", "peak_delta_gb"):
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    summary = (
        numeric.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            wall_seconds=("seconds", "sum"),
            peak_gpu_gb=("peak_gpu_gb", "max"),
            peak_gpu_reserved_gb=("peak_gpu_reserved_gb", "max"),
            peak_delta_gb=("peak_delta_gb", "max"),
            n_timers=("timer", "count"),
        )
    )
    return summary[columns]


def wide_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    enc_cols = [c for c in df.columns if not str(c).startswith("causal.")]
    encoder = df[enc_cols].copy()
    causal_cols = ["model", "task", "suite", "method", "proxy", "present", "method_status"] + [
        c for c in df.columns if str(c).startswith("causal.")
    ]
    causal = df[[c for c in causal_cols if c in df.columns]].copy()
    return encoder, causal


def stub_heatmaps(df: pd.DataFrame, plot_dir: Path) -> List[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if df.empty:
        return paths
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); skip heatmaps")
        return paths

    metric = "roc_auc" if "roc_auc" in df.columns else None
    if metric is None:
        for cand in ("roc_auc_neutral", "balanced_accuracy", "cohens_d"):
            if cand in df.columns:
                metric = cand
                break
    if metric is None:
        return paths

    # Per-model: task × method
    for model, sub in df.groupby("model"):
        pivot = sub.pivot_table(index="task", columns="method", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.4), max(3, pivot.shape[0] * 0.5)))
        im = ax.imshow(pivot.fillna(np.nan).to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=90, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        ax.set_title(f"{model}: {metric}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out = plot_dir / f"heatmap_task_method_{model}.pdf"
        strip_figure_titles(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    # Fixed method patterns across models × tasks
    for pattern in ("actiend:", "sae:", "caa:"):
        sub = df[df["method"].astype(str).str.startswith(pattern)].copy()
        if sub.empty:
            continue
        # Prefer class headline ids without heavy suffixes when possible
        sub["method_short"] = sub["method"].astype(str).str.split("|").str[0]
        pivot = sub.pivot_table(index="model", columns="task", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1]), max(3, pivot.shape[0] * 0.6)))
        im = ax.imshow(pivot.fillna(np.nan).to_numpy(), aspect="auto", cmap="magma")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        ax.set_title(f"models×tasks ({pattern}* / {metric})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        safe = pattern.strip(":").replace(":", "_")
        out = plot_dir / f"heatmap_model_task_{safe}.pdf"
        strip_figure_titles(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def pivot_method_task(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    model: Optional[str] = None,
) -> pd.DataFrame:
    """Pivot: method (rows) × task (columns), with a mean column.

    Works with partial results — tasks without data are NaN.
    """
    sub = df.copy()
    if model:
        sub = sub[sub["model"] == model]
    sub = sub[sub["method_status"].isin({"ok", "partial"})]
    if metric not in sub.columns or sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index="method", columns="task", values=metric, aggfunc="mean")
    pivot["MEAN"] = pivot.mean(axis=1)
    return pivot.sort_values("MEAN", ascending=False)


def write_pivot_tables(df: pd.DataFrame, out_dir: Path, metric: str = "roc_auc") -> List[Path]:
    """Write one method×task pivot CSV per model, plus a combined one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    models = sorted(df["model"].dropna().unique())
    for model in models:
        pivot = pivot_method_task(df, metric=metric, model=model)
        if pivot.empty:
            continue
        p = out_dir / f"pivot_method_task_{model}.csv"
        pivot.to_csv(p)
        paths.append(p)
        print(f"  {model}: {pivot.shape[0]} methods × {pivot.shape[1]-1} tasks")
    if len(models) > 1:
        pivot = pivot_method_task(df, metric=metric)
        if not pivot.empty:
            p = out_dir / "pivot_method_task_all.csv"
            pivot.to_csv(p)
            paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--plots", type=Path, default=PLOT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--model", default=None, help="Filter to a single model")
    parser.add_argument("--metric", default="roc_auc", help="Pivot metric (default: roc_auc)")
    args = parser.parse_args()

    df = collect_rows(args.runs)
    if args.model:
        df = df[df["model"] == args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    long_path = args.out / "methods_long.csv"
    df.to_csv(long_path, index=False)
    encoder, causal = wide_tables(df)
    enc_path = args.out / "encoder_wide.csv"
    causal_path = args.out / "causal_wide.csv"
    encoder.to_csv(enc_path, index=False)
    causal.to_csv(causal_path, index=False)
    compute_df = collect_compute(args.runs)
    compute_path = args.out / "compute.csv"
    compute_df.to_csv(compute_path, index=False)
    ledger_df = collect_method_cost_ledger(args.runs)
    ledger_path = args.out / "method_cost_ledger.csv"
    ledger_df.to_csv(ledger_path, index=False)
    ledger_summary = summarize_method_cost_ledger(ledger_df)
    ledger_summary_path = args.out / "method_cost_summary.csv"
    ledger_summary.to_csv(ledger_summary_path, index=False)
    print(f"Wrote {long_path} ({len(df)} rows)")
    print(f"Wrote {enc_path}")
    print(f"Wrote {causal_path}")
    print(f"Wrote {compute_path} ({len(compute_df)} runs)")
    print(f"Wrote {ledger_path} ({len(ledger_df)} method-attributed timers)")
    print(f"Wrote {ledger_summary_path} ({len(ledger_summary)} method-stage rows)")

    print(f"\nPivot tables (metric={args.metric}):")
    pivot_paths = write_pivot_tables(df, args.out, metric=args.metric)
    for p in pivot_paths:
        print(f"  Wrote {p}")

    # Print summary to stdout for quick inspection
    pivot = pivot_method_task(df, metric=args.metric, model=args.model)
    if not pivot.empty:
        print(f"\n{'='*60}")
        print(f"Method × Task ({args.metric}){f' [{args.model}]' if args.model else ''}")
        print(f"{'='*60}")
        print(pivot.to_string(float_format="%.3f"))
        print()

    try:
        from suitability import refresh_global_suitability

        global_suit = refresh_global_suitability(args.runs, args.out)
        print(f"Wrote {global_suit['paths'].get('model_task_csv')}")
        print(f"Wrote {global_suit['paths'].get('long_csv')}")
    except Exception as exc:
        track_error(exc, context="global suitability refresh")
    if not args.no_plots:
        paths = stub_heatmaps(df, args.plots)
        for p in paths:
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
