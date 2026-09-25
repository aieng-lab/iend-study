"""Build a freshness-audited appendix evidence package from final summaries.

This is a report generator, not an experiment runner.  It only reads the
current merged summaries whose rows point to final ``results.json`` artifacts.
It deliberately never reads or reports ``encoding_E``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report_model_policy import PAPER_MODELS, summary_csv_path
from analysis.summary_latex import METHOD_LATEX, MODEL_LATEX, PAPER_PAIRWISE_TASKS
from analysis.plot_style import show_plot_if_requested
from study.config import _load_task_yaml, list_tasks

SAE = ("sae:k1", "sae:kstar", "sae_pre:k1", "sae_pre:kstar")
TARGETED = ("caa", "actiend", "caga", "agiend", "cga", "gradiend")
METRICS = (
    "detection_score", "detection_light_score", "intervention_score", "roc_auc_neutral",
    "roc_auc_other", "neutral_specificity", "class_exclusivity",
)


def _local_path(value: object) -> Path:
    """Resolve paths recorded by either WSL or native-Windows report builds."""
    text = str(value)
    if text.startswith("/mnt/") and len(text) > 7 and text[5].isalpha() and text[6:7] == "/":
        return Path(f"{text[5].upper()}:/{text[7:]}")
    return Path(text)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def paper_model_sources() -> tuple[tuple[str, str], ...]:
    """The paper scope, as defined solely by configs/report_model_sets.json."""
    return PAPER_MODELS


def load_final_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paper-model rows plus a freshness manifest.

    Missing source artifacts exclude a model. A merely stale generated summary
    remains usable (and is marked) so the appendix has the same paper-model
    scope as the currently reported paper table; a normal summary refresh
    clears that flag before final submission.
    """
    frames, manifest = [], []
    for model, subdir in paper_model_sources():
        path = summary_csv_path(model, subdir)
        if not path.exists():
            manifest.append({"model": model, "summary_csv": str(path), "status": "missing_summary"})
            continue
        frame = pd.read_csv(path)
        # Regime-comparable detection: it intentionally ignores rival terms
        # for BOTH constructions, unlike the paper's full Det definition.
        frame["detection_light_score"] = frame[["roc_auc_neutral", "neutral_specificity"]].min(axis=1)
        sources = [_local_path(p) for p in frame.get("results_path", pd.Series(dtype=str)).dropna().unique()]
        missing = [p for p in sources if not p.exists()]
        latest = max((p.stat().st_mtime for p in sources if p.exists()), default=0.0)
        fresh = not missing and path.stat().st_mtime >= latest
        manifest.append({
            "model": model, "summary_csv": str(path), "n_rows": len(frame),
            "n_result_files": len(sources), "n_missing_result_files": len(missing),
            "summary_mtime": path.stat().st_mtime, "latest_result_mtime": latest,
            "status": "current" if fresh else ("missing_source" if missing else "stale_summary"),
        })
        if not missing:
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), pd.DataFrame(manifest))


def _mean_table(frame: pd.DataFrame, group: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group, keys))
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna() if metric in part else pd.Series(dtype=float)
            row[f"n_{metric}"] = len(values)
            row[metric] = values.mean() if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def sae_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = frame[frame["method_group"].isin(SAE)].copy()
    rows = rows[rows["task"].isin(list_tasks())]
    by_model = _mean_table(rows, ["model", "method_group"], METRICS)
    # Equal weight per model, with no claim that an absent model contributed.
    balanced = _mean_table(by_model, ["method_group"], [m for m in METRICS])
    matched = []
    comparisons = (("sae:kstar", "sae:k1", "kstar_minus_k1"),
                   ("sae_pre:kstar", "sae_pre:k1", "pre_kstar_minus_pre_k1"),
                   ("sae_pre:k1", "sae:k1", "pre_minus_filled_k1"),
                   ("sae_pre:kstar", "sae:kstar", "pre_minus_filled_kstar"))
    for model, part in rows.groupby("model"):
        for left, right, label in comparisons:
            a = part[part.method_group.eq(left)].set_index("task")
            b = part[part.method_group.eq(right)].set_index("task")
            tasks = sorted(set(a.index) & set(b.index))
            for metric in ("detection_score", "intervention_score"):
                diff = pd.to_numeric(a.reindex(tasks)[metric], errors="coerce") - pd.to_numeric(b.reindex(tasks)[metric], errors="coerce")
                diff = diff.dropna()
                matched.append({"model": model, "comparison": label, "metric": metric,
                                "n_matched": len(diff), "mean_delta": diff.mean() if len(diff) else float("nan"),
                                "tasks": ",".join(diff.index.astype(str))})
    return {"sae_by_model": by_model, "sae_model_balanced": balanced,
            "sae_matched": pd.DataFrame(matched)}


def regime_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = frame[frame.backend.isin(TARGETED) & frame.task.isin(list_tasks())].copy()
    rows["construction"] = rows["pole"].map({"two_pole": "pairwise", "one_pole": "one-sided"})
    rows = rows[rows.construction.notna()]
    overall = _mean_table(rows, ["model", "backend", "construction"], METRICS)
    light_all = _mean_table(
        rows, ["model", "backend", "construction"],
        ("detection_light_score", "intervention_score"),
    )
    full_pairwise_capable = _mean_table(
        rows[rows.task.isin(PAPER_PAIRWISE_TASKS)],
        ["model", "backend", "construction"],
        ("detection_score", "intervention_score"),
    )
    # Direct comparison uses exactly pairwise-capable task intersections.
    matched = []
    direct = rows[rows.task.isin(PAPER_PAIRWISE_TASKS)]
    for (model, backend), part in direct.groupby(["model", "backend"]):
        pair = part[part.construction.eq("pairwise")].set_index("task")
        one = part[part.construction.eq("one-sided")].set_index("task")
        tasks = sorted(set(pair.index) & set(one.index))
        for metric in METRICS:
            delta = pd.to_numeric(one.reindex(tasks)[metric], errors="coerce") - pd.to_numeric(pair.reindex(tasks)[metric], errors="coerce")
            delta = delta.dropna()
            matched.append({"model": model, "backend": backend, "metric": metric,
                            "n_matched": len(delta), "one_sided_minus_pairwise": delta.mean() if len(delta) else float("nan"),
                            "tasks": ",".join(delta.index.astype(str))})
    feature_rows = []
    for task in list_tasks():
        cfg = _load_task_yaml(task)
        classes = list(cfg.get("classes") or [])
        k = len(classes)
        if k > 2:
            feature_rows.append({"task": task, "classes": " / ".join(map(str, classes)), "K": k,
                                 "pairwise_features": k * (k - 1) // 2,
                                 "one_sided_features": k})
    return {"regime_by_model": overall,
            "regime_det_light_all_tasks": light_all,
            "regime_full_det_pairwise_capable_tasks": full_pairwise_capable,
            "regime_matched_pairwise_tasks": pd.DataFrame(matched),
            "multiclass_feature_counts": pd.DataFrame(feature_rows)}


def _kstar_artifact_paths(frame: pd.DataFrame, extra_roots: Iterable[Path]) -> list[tuple[Path, str]]:
    """Find compact SAE encode records without assuming one machine has all runs.

    ``results.json`` can be hundreds of MB per task and is deliberately not a
    portable source of truth for this audit: the cluster, a second cluster, and workstations may
    each hold a disjoint part of an SAE run.  The encode-stage artifact is
    small and stores the selected readout directly.  Canonical task artifacts
    are considered first; source-preserving sync archives and explicit roots
    supplement them without overwriting their provenance.
    """
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(path: Path, source: str) -> None:
        try:
            path = path.resolve()
        except OSError:
            pass
        if path.is_file() and path not in seen:
            seen.add(path)
            found.append((path, source))

    relative_run_dirs: list[Path] = []
    for results_path in frame.get("results_path", pd.Series(dtype=str)).dropna().unique():
        path = _local_path(results_path)
        add(path.parent / "artifacts" / "sae" / "encode_method_rows.json", "canonical")
        try:
            relative_run_dirs.append(path.parent.relative_to(ROOT / "runs"))
        except ValueError:
            # A caller can supply an external RUNS root. Its explicit source
            # root is still supported below, but we cannot infer a portable
            # relative path from this checkout.
            pass
    for root in extra_roots:
        if root.is_file():
            add(root, f"explicit:{root}")
        elif root.is_dir():
            # Source archives retain the original runs/<model>/<subdir>/<task>
            # layout. Match that exact canonical run directory rather than
            # recursively taking every historical suite for the same task.
            # Without this, a stale suite_full artifact could conflict with a
            # selected suite_full2 paper result merely because they share a
            # model/task name.
            for relative in relative_run_dirs:
                direct = root / relative / "artifacts" / "sae" / "encode_method_rows.json"
                add(direct, f"archive:{root}")
                for source_root in root.iterdir():
                    if source_root.is_dir():
                        add(source_root / relative / "artifacts" / "sae" / "encode_method_rows.json",
                            f"archive:{source_root}")
    return found


def kstar_selection_audit(frame: pd.DataFrame, extra_roots: Iterable[Path] = ()) -> dict[str, pd.DataFrame]:
    """Gather k* choices from compact per-source SAE artifacts.

    ``readout_k`` is the stored selected k.  The previous scan looked for a
    nonexistent ``metrics.k`` field in giant result blobs, yielding an empty
    frequency table even when complete k* selection records were available.
    """
    rows: list[dict[str, object]] = []
    allowed = {
        (str(row.model), str(row.task))
        for row in frame[["model", "task"]].dropna().itertuples(index=False)
    }
    for path, source in _kstar_artifact_paths(frame, extra_roots):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model, task = str(payload.get("model") or ""), str(payload.get("task") or "")
        if allowed and (model, task) not in allowed:
            continue
        for method in payload.get("methods") or []:
            method_id = str(method.get("method") or "")
            if not method_id.startswith(("sae:", "sae_pre:")) or not method_id.endswith(":kstar"):
                continue
            metrics = method.get("metrics") or {}
            value = metrics.get("readout_k")
            rows.append({
                "model": model, "task": task, "method": method_id,
                "site_family": method_id.split(":", 1)[0],
                "selected_k": value if _number(value) else None,
                "selection_rule": metrics.get("k_selection"),
                "sae_layer": metrics.get("sae_layer"),
                "artifact_path": str(path), "artifact_source": source,
            })
    cells = pd.DataFrame(rows)
    if cells.empty:
        return {"sae_kstar_selected_k_cells": cells, "sae_kstar_selected_k_frequency": pd.DataFrame(),
                "sae_kstar_selected_k_missing": pd.DataFrame(), "sae_kstar_selected_k_conflicts": pd.DataFrame()}

    key = ["model", "task", "method"]
    # Archives can legitimately repeat a canonical cell.  Collapse exact
    # duplicates before counting; differing selected k values are retained as
    # an explicit conflict and excluded from the frequency claim.
    cells["cell_key"] = cells[key].astype(str).agg("/".join, axis=1)
    conflicts = []
    resolved_rows = []
    for _, group in cells.groupby("cell_key", sort=True):
        values = sorted({int(v) for v in group["selected_k"].dropna()})
        if len(values) > 1:
            conflicts.append({
                "cell_key": group.iloc[0]["cell_key"], "selected_k_values": ",".join(map(str, values)),
                "n_sources": len(group), "artifact_paths": " | ".join(group["artifact_path"]),
            })
            continue
        canonical = group[group["artifact_source"].eq("canonical")]
        resolved_rows.append((canonical if not canonical.empty else group).iloc[0])
    resolved = pd.DataFrame(resolved_rows)
    counts = (resolved.dropna(subset=["selected_k"]).groupby(["site_family", "selected_k"], as_index=False)
              .size().rename(columns={"size": "n_class_cells"})) if not resolved.empty else pd.DataFrame()
    missing = resolved[resolved.selected_k.isna()].copy() if not resolved.empty else pd.DataFrame()
    return {"sae_kstar_selected_k_cells": cells, "sae_kstar_selected_k_frequency": counts,
            "sae_kstar_selected_k_missing": missing,
            "sae_kstar_selected_k_conflicts": pd.DataFrame(conflicts)}


def _write_tables(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.8f")




def _write_latex_book(out: Path, tables: dict[str, pd.DataFrame]) -> Path:
    """Write the one compact, paper-facing SAE comparison table.

    The CSV/Markdown package remains an audit trail. The PDF deliberately does
    not reproduce it: it is the one table that belongs in the paper appendix.
    """
    methods = ("sae:k1", "sae:kstar", "sae_pre:k1", "sae_pre:kstar")
    source = tables.get("sae_by_model", pd.DataFrame())
    pivot = source.pivot(index="model", columns="method_group", values=["detection_score", "intervention_score"]).reindex(
        index=[model for model, _ in paper_model_sources()],
        columns=pd.MultiIndex.from_product((("detection_score", "intervention_score"), methods)),
    ) if not source.empty else pd.DataFrame()
    rows: list[str] = []
    for model in pivot.index:
        values = [MODEL_LATEX.get(str(model), str(model))]
        for method in methods:
            for metric in ("detection_score", "intervention_score"):
                value = pivot.loc[model, (metric, method)]
                values.append("--" if pd.isna(value) else f"{100 * float(value):.1f}")
        rows.append(" & ".join(values) + r" \\")
    header_groups = " & ".join(rf"\multicolumn{{2}}{{c}}{{{METHOD_LATEX[method]}}}" for method in methods)
    tex_table = "\n".join((
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{SAE-site and capacity comparison across the paper models. Entries are task means (\%) of the held-out Detection and Intervention scores; \MethodSAEkOne is the headline SAE configuration.}",
        r"\label{tab:appendix-sae-variants}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Model & " + header_groups + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r" & Det. & Int. & Det. & Int. & Det. & Int. & Det. & Int. \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ))
    tex = out / "appendix_evidence_tables.tex"
    tex.write_text("\n".join([
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{booktabs}",
        r"\providecommand{\MethodSAEkOne}{$\mathrm{SAE}^{k=1}$}",
        r"\providecommand{\MethodSAEkStar}{$\mathrm{SAE}^{k^{*}}$}",
        r"\providecommand{\MethodSAEPrekOne}{$\mathrm{SAE}_{\mathrm{pre}}^{k=1}$}",
        r"\providecommand{\MethodSAEPrekStar}{$\mathrm{SAE}_{\mathrm{pre}}^{k^{*}}$}",
        r"\begin{document}",
        tex_table,
        r"\end{document}",
        "",
    ]), encoding="utf-8")
    return tex


def _write_sae_variant_figure(out: Path, tables: dict[str, pd.DataFrame]) -> Path | None:
    """Visual companion to the compact SAE table: four variants × paper models."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
    except ImportError:
        return None
    source = tables.get("sae_by_model", pd.DataFrame())
    if source.empty:
        return None
    paper_font = ROOT / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(paper_font)).get_name()
    methods = ("sae:k1", "sae:kstar", "sae_pre:k1", "sae_pre:kstar")
    labels = (r"$\mathrm{SAE}^{k=1}$", r"$\mathrm{SAE}^{k^{*}}$",
              r"$\mathrm{SAE}_{\mathrm{pre}}^{k=1}$", r"$\mathrm{SAE}_{\mathrm{pre}}^{k^{*}}$")
    # SAE capacity/site is the one deliberate exception to the study-wide
    # method palette: four appendix variants need independent hues rather than
    # implying they are separate headline method families.
    colors = ("#4c78a8", "#72b7b2", "#f58518", "#e45756")
    hatches = ("", "", "", "")
    models = [model for model, _ in paper_model_sources() if model in set(source["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.25), sharex=True)
    width = 0.18
    for ax, metric, title in zip(axes, ("detection_score", "intervention_score"), ("Detection (Det.)", "Intervention (Int.)")):
        values_seen: list[float] = []
        for method_index, (method, label, color, hatch) in enumerate(zip(methods, labels, colors, hatches)):
            series = source[source["method_group"].eq(method)].set_index("model")[metric].reindex(models) * 100
            values = series.to_numpy(dtype=float)
            positions = np.arange(len(models)) - 0.27 + method_index * width
            bars = ax.bar(positions, values, width=width, color=color, hatch=hatch, label=label)
            values_seen.extend(series.dropna().tolist())
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
        if values_seen:
            low, high = 0, max(values_seen)
            pad = max(1.0, 0.12 * (high - low))
            ax.set_ylim(low - pad, high + 2.2 * pad)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(range(len(models)), [MODEL_LATEX.get(model, model) for model in models], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=True, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = out / "appendix_sae_variants.pdf"
    fig.savefig(path, bbox_inches="tight")
    show_plot_if_requested(plt)
    plt.close(fig)
    return path


def _compile_latex(tex: Path) -> Path | None:
    """Compile the standalone appendix book when a local TeX engine exists."""
    candidates = [shutil.which("xelatex"), shutil.which("pdflatex")]
    def usable(candidate: str | None) -> bool:
        if not candidate:
            return False
        try:
            return Path(candidate).is_file()
        except OSError:
            return False
    engine = next((candidate for candidate in candidates if usable(candidate)), None)
    if engine is None:
        print(f"warning: wrote {tex}, but no xelatex/pdflatex executable was found; PDF was not compiled.")
        return None
    for _ in range(2):
        completed = subprocess.run([str(engine), "-interaction=nonstopmode", "-halt-on-error", tex.name],
                                   cwd=tex.parent, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"LaTeX compilation failed for {tex}; see {tex.with_suffix('.log')}")
    return tex.with_suffix(".pdf")


def _md_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter (avoid optional tabulate)."""
    if frame.empty:
        return "_No eligible cells._"
    def cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", r"\\|")
    columns = [str(col) for col in frame.columns]
    body = [[cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ])


def _markdown(out: Path, manifest: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Appendix evidence package", "", "This is a data/provenance report, not paper prose.", "",
             "## Source audit", "", "Only sources marked `current` below enter numeric tables.", "",
             _md_table(manifest), "",
             "## Metric contract", "", "- Pairwise Det = min(AUC_n, AUC_o, Spec_n, Excl_o).",
             "- One-sided Det = min(AUC_n, Spec_n).", "- Int = (DeltaP+ + DeltaP-) / 2; incomplete Int rows are excluded per metric.",
             "- `encoding_E` is neither read nor reported.", "",
             "## SAE variants", "", "`sae:*` selects from filled `prediction` activations; `sae_pre:*` selects from `pre_prediction`. ",
             "k=1 is the validation-selected layer with one latent; k* chooses from k={1,2,4,8,16,32,64,128} on validation. ",
             "Both use additive decoder steering h' = h + alpha d.  SAE_pre rival quantities are intentionally unavailable on one-sided CF rows because that pre-target state is identical across the relabelled rival.", ""]
    for name in ("sae_by_model", "sae_model_balanced", "sae_matched"):
        if name not in tables:
            continue
        lines += [f"### {name}", "", _md_table(tables[name]), ""]
    lines += ["## Pairwise vs one-sided", "", "The direct-comparison file is restricted to the same pairwise-capable task and model cells. It reports component-level deltas as an apples-to-apples audit; raw Det remains regime-specific.", ""]
    for name in ("multiclass_feature_counts", "regime_det_light_all_tasks",
                 "regime_full_det_pairwise_capable_tasks", "regime_matched_pairwise_tasks"):
        if name not in tables:
            continue
        lines += [f"### {name}", "", _md_table(tables[name]), ""]
    if "sae_kstar_selected_k_frequency" in tables:
        lines += ["## SAE k* selection audit", ""]
        for name in ("sae_kstar_selected_k_frequency", "sae_kstar_selected_k_missing",
                     "sae_kstar_selected_k_conflicts"):
            lines += [f"### {name}", "", _md_table(tables[name]), ""]
    lines += ["## Not yet included", "", "Decoder reachability requires a separate config-and-trajectory audit; no values are inferred from filenames.", ""]
    (out / "evidence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "tables" / "appendix_evidence")
    parser.add_argument("--scan-kstar", action="store_true",
                        help="Read compact SAE encode artifacts to audit stored selected-k values (CPU/I/O only).")
    parser.add_argument("--kstar-source", type=Path, action="append", default=[], metavar="PATH",
                        help="Additional runs/archive root containing SAE encode artifacts; repeat for each machine.")
    parser.add_argument("--sae-figure-only", action="store_true",
                        help="Regenerate only appendix_sae_variants.pdf from canonical summary CSVs.")
    args = parser.parse_args()
    print("WARNING: appendix evidence uses the paper models from configs/report_model_sets.json. Regenerate summary tables before final export so every source is current.")
    args.out.mkdir(parents=True, exist_ok=True)
    frame, manifest = load_final_summaries()
    figure = _write_sae_variant_figure(args.out, sae_tables(frame) if not frame.empty else {})
    if figure is None:
        raise RuntimeError("No SAE data available for the requested paper-model sources.")
    print(f"Wrote {figure}")
    if args.sae_figure_only:
        return
    manifest.to_csv(args.out / "source_manifest.csv", index=False)
    tables = {**sae_tables(frame), **regime_tables(frame)} if not frame.empty else {}
    if args.scan_kstar and not frame.empty:
        archived = ROOT / "runs" / "_sae_selection_sources"
        sources = [*args.kstar_source]
        if archived.is_dir():
            sources.append(archived)
        tables.update(kstar_selection_audit(frame, sources))
    _write_tables(args.out, tables)
    _markdown(args.out, manifest, tables)
    tex = _write_latex_book(args.out, tables)
    figure = _write_sae_variant_figure(args.out, tables)
    pdf = _compile_latex(tex)
    print(f"Wrote freshness-audited evidence package to {args.out}")
    print(f"Wrote {tex}")
    if pdf:
        print(f"Wrote {pdf}")
    if figure:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
