#!/usr/bin/env python
"""GRADIEND / ACTIEND training-convergence overview, for debugging.

Reads ``runs/{model}/[{subdir}/]{task}/artifacts/{backend}__{ablation}__{class}[__tensors]/done.json``
directly -- these are the only two trained backends (see
``study/stages/train.py::reload_train_raw_from_artifacts``'s
``backends=("gradiend", "actiend")``; SAE/CAA are post-hoc readouts of a
backbone, not separately trained). ``done.json`` is the one place that
carries per-run training-convergence detail at all -- ``results.json`` /
``REPORT.md`` only carry the fair-eval encoder metrics, not whether the
underlying trainer actually converged, on what metric, at what step, or
whether the encoder collapsed.

Only runs trained after 2026-08-18 carry ``extras.convergence_info`` /
``extras.best_score_checkpoint`` (see CLAUDE.md's "``done.json`` extras now
carry a convergence summary" note) -- older runs are reported as
``unknown_no_convergence_info`` (or ``unknown_low_quality`` if their
fair-eval encoder metrics look bad anyway), not silently treated as
converged.

**This is wired into the regular paper-PDF pipeline** --
``analysis/summary_latex.py``'s ``main()`` calls ``collect_convergence_rows``
and the two ``format_convergence_*_latex`` functions below automatically for
every ``--model`` it processes, writing ``summary_convergence_counts_*.tex``
/ ``summary_convergence_problems_*.tex`` snippets that ``--write-book`` /
``scripts/summary_tables_pdf.sh`` fold into ``summary_tables.pdf`` alongside
the headline method tables -- no separate command needed. The CLI below
(``--only-problems``, ``--csv``) remains for ad-hoc terminal/CSV debugging
that doesn't need a PDF rebuild.

Examples::

  python analysis/convergence_overview.py --model gpt2-small
  python analysis/convergence_overview.py --model gpt2-small --only-problems
  python analysis/convergence_overview.py --model gpt2-small --subdir suite_full
  python analysis/convergence_overview.py --model gpt2-small --csv analysis/tables/convergence_gpt2-small.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from analysis.task_order import task_sort_key

ROOT = Path(__file__).resolve().parents[1]

BACKENDS: frozenset[str] = frozenset({"gradiend", "actiend"})

# Statuses worth surfacing for debugging, in severity order.
PROBLEM_STATUSES: Sequence[str] = (
    "collapsed",
    "not_converged",
    "unknown_low_quality",
)

STATUS_ORDER: Sequence[str] = (
    "collapsed",
    "not_converged",
    "unknown_low_quality",
    "unknown_no_convergence_info",
    "converged",
)

# Below this, a run with no convergence_info is flagged as suspicious even
# though we can't say whether it "converged" under its own criterion.
LOW_QUALITY_THRESHOLD = 0.6


def _positive_label_mean(encoder_metrics: Mapping[str, Any]) -> Optional[float]:
    means = encoder_metrics.get("mean_by_class")
    if not isinstance(means, Mapping):
        return None
    for label, value in means.items():
        try:
            is_positive = float(label) == 1.0
        except (TypeError, ValueError):
            continue
        if is_positive and isinstance(value, (int, float)):
            return float(value)
    return None


def _reinterpret_legacy_auc_convergence(
    *,
    extras: Mapping[str, Any],
    convergence_info: Mapping[str, Any],
    best_checkpoint: Mapping[str, Any],
    encoder_metrics: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Apply the corrected AUC convergence contract to legacy summaries.

    Older package versions incorrectly required the two non-neutral evaluation
    means to have opposite signs. A one-pole feature only requires its semantic
    target (numeric label +1) to encode positively; the rival class may remain
    positive as long as the configured AUC criterion separates it. This can be
    recovered without retraining when only one convergent seed was required.
    """
    metric = str(
        convergence_info.get("convergence_metric")
        or extras.get("convergent_metric")
        or best_checkpoint.get("selection_metric")
        or ""
    ).strip().lower()
    if metric not in {"roc_auc", "auroc", "min_auc_n_o", "min_auc"}:
        return None
    # New summaries already encode the target-positive requirement explicitly.
    if convergence_info.get("auc_positive_target_mean_required") is True:
        return None
    min_seeds = convergence_info.get("min_convergent_seeds")
    if isinstance(min_seeds, (int, float)) and int(min_seeds) > 1:
        return None
    threshold = convergence_info.get("threshold")
    score = best_checkpoint.get(metric)
    if score is None and metric in {"auroc"}:
        score = best_checkpoint.get("roc_auc")
    if score is None and metric == "min_auc":
        score = best_checkpoint.get("min_auc_n_o")
    step = best_checkpoint.get("global_step")
    stored_positive_mean = _positive_label_mean(encoder_metrics)
    # AUC training is globally sign-symmetric. Current package callbacks
    # canonicalize the model by flipping encoder+decoder weights whenever the
    # +1 mean is negative. Apply the same canonical orientation to legacy
    # summaries so an otherwise valid old checkpoint is not counted as a
    # failure solely because it predates that normalization.
    positive_mean = (
        abs(float(stored_positive_mean))
        if isinstance(stored_positive_mean, (int, float))
        else None
    )
    eligible = bool(
        isinstance(step, (int, float))
        and float(step) > 0
        and isinstance(score, (int, float))
        and isinstance(threshold, (int, float))
        and float(score) >= float(threshold)
        and isinstance(positive_mean, (int, float))
        and float(positive_mean) > 0.0
    )
    return {
        "converged": eligible,
        "positive_target_class_mean": positive_mean,
        "positive_target_class_mean_ok": bool(
            isinstance(positive_mean, (int, float)) and positive_mean > 0.0
        ),
        "orientation_flip_required": bool(
            isinstance(stored_positive_mean, (int, float)) and stored_positive_mean < 0.0
        ),
        "reason": "legacy_auc_contract_reinterpreted",
    }


def parse_stage(stage: str) -> Dict[str, Any]:
    """Split ``{backend}__{ablation}__{class}[__tensors]`` (see
    ``study/method_ids.py::artifact_dirname``, the writer of this on-disk name).
    """
    tokens = str(stage).split("__")
    backend = tokens[0] if tokens else ""
    ablation = tokens[1] if len(tokens) > 1 else ""
    tensors = len(tokens) > 2 and tokens[-1] == "tensors"
    key_tokens = tokens[2:-1] if tensors else tokens[2:]
    return {
        "backend": backend,
        "ablation": ablation,
        "class_key": "__".join(key_tokens),
        "tensors": tensors,
    }


def classify_status(row: Mapping[str, Any]) -> str:
    """``row["converged"]`` (from ``training.json``'s ``convergence_info.converged``,
    written by the ``gradiend`` package's multi-seed trainer -- see
    ``trainer.py``'s seed-sweep loop) is already ``convergent_count >=
    min_convergent_seeds`` across every seed *actually tried* for this run
    (``training_max_seeds`` upper-bounds how many). A seed that failed before
    a later seed converged does NOT flip this row to ``not_converged`` -- that
    aggregation happens once, in the package, not here. ``not_converged`` here
    means every attempted seed missed the threshold; don't re-derive a
    per-seed view on top of this without checking the seed-sweep code first.
    """
    if row.get("collapsed_encoder"):
        return "collapsed"
    if not row.get("has_convergence_info"):
        quality = [
            v
            for v in (
                row.get("roc_auc_neutral_all"),
                row.get("roc_auc_other_all"),
                row.get("correlation_all"),
            )
            if isinstance(v, (int, float))
        ]
        if quality and min(quality) < LOW_QUALITY_THRESHOLD:
            return "unknown_low_quality"
        return "unknown_no_convergence_info"
    if row.get("converged") is True:
        return "converged"
    if row.get("converged") is False:
        return "not_converged"
    return "unknown_no_convergence_info"


def load_convergence_row(path: Path, *, model: str, task: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    stage = str(payload.get("stage") or path.parent.name)
    parsed = parse_stage(stage)

    extras = payload.get("extras")
    extras = extras if isinstance(extras, dict) else {}

    conv_info = extras.get("convergence_info")
    conv_info = conv_info if isinstance(conv_info, dict) else {}
    has_convergence_info = bool(conv_info)

    best_ckpt = extras.get("best_score_checkpoint")
    best_ckpt = best_ckpt if isinstance(best_ckpt, dict) else {}

    encoder_metrics = (extras.get("encoder_eval") or {}).get("encoder_metrics") or {}
    encoder_metrics = encoder_metrics if isinstance(encoder_metrics, dict) else {}
    encoder_all = (encoder_metrics.get("all_data")) or {}
    if not isinstance(encoder_all, dict):
        encoder_all = {}

    convergence_metric = conv_info.get("convergence_metric") or extras.get("convergent_metric")
    selection_metric = best_ckpt.get("selection_metric") or convergence_metric
    best_value = best_ckpt.get(selection_metric) if selection_metric else None

    convergent_count = conv_info.get("convergent_count")
    min_convergent_seeds = conv_info.get("min_convergent_seeds")
    component_seed = conv_info.get("component_seed_summary")
    component_seed = component_seed if isinstance(component_seed, dict) else {}
    n_components = component_seed.get("n_components")
    n_satisfied_components = component_seed.get("n_satisfied_components")
    component_summary = None
    if n_components is not None and n_satisfied_components is not None:
        component_summary = f"{n_satisfied_components}/{n_components}"
    max_seeds = extras.get("training_max_seeds")
    seeds_summary = None
    if convergent_count is not None and min_convergent_seeds is not None:
        seeds_summary = f"{convergent_count}/{min_convergent_seeds}"
        if max_seeds is not None:
            seeds_summary += f" (max {max_seeds})"

    legacy_auc = _reinterpret_legacy_auc_convergence(
        extras=extras,
        convergence_info=conv_info,
        best_checkpoint=best_ckpt,
        encoder_metrics=encoder_metrics,
    )
    stored_converged = conv_info.get("converged") if has_convergence_info else None
    effective_converged = (
        legacy_auc["converged"] if legacy_auc is not None else stored_converged
    )
    positive_target_mean = conv_info.get("convergent_positive_target_class_mean")
    positive_target_mean_ok = conv_info.get("convergent_positive_target_class_mean_ok")
    orientation_flip_required = False
    if legacy_auc is not None:
        positive_target_mean = legacy_auc["positive_target_class_mean"]
        positive_target_mean_ok = legacy_auc["positive_target_class_mean_ok"]
        orientation_flip_required = legacy_auc["orientation_flip_required"]

    row: Dict[str, Any] = {
        "model": model,
        "task": task,
        "backend": parsed["backend"],
        "ablation": parsed["ablation"],
        "class_key": parsed["class_key"],
        "tensors": parsed["tensors"],
        "stage": stage,
        "config_hash": payload.get("config_hash"),
        "learning_rate": extras.get("learning_rate"),
        "auc_orientation_protocol_version": extras.get("auc_orientation_protocol_version"),
        "declared_convergent_metric": extras.get("convergent_metric"),
        "collapsed_encoder": bool(extras.get("collapsed_encoder")),
        "has_convergence_info": has_convergence_info,
        "converged": effective_converged,
        "stored_converged": stored_converged,
        "convergence_reinterpreted": legacy_auc is not None,
        "orientation_flip_required": orientation_flip_required,
        "convergence_metric": convergence_metric,
        "convergence_threshold": conv_info.get("threshold"),
        "auc_positive_target_mean_required": conv_info.get("auc_positive_target_mean_required"),
        "positive_target_class_mean": positive_target_mean,
        "positive_target_class_mean_ok": positive_target_mean_ok,
        "convergent_count": conv_info.get("convergent_count"),
        "min_convergent_seeds": conv_info.get("min_convergent_seeds"),
        "n_components": n_components,
        "n_satisfied_components": n_satisfied_components,
        "component_summary": component_summary,
        "missing_component_ids": component_seed.get("missing_component_ids") or [],
        "training_max_steps": extras.get("training_max_steps"),
        "training_max_seeds": max_seeds,
        "seeds_summary": seeds_summary,
        "best_step": best_ckpt.get("global_step"),
        "best_metric_value": best_value,
        "correlation_all": encoder_all.get("correlation"),
        "roc_auc_neutral_all": encoder_all.get("roc_auc_neutral"),
        "roc_auc_other_all": encoder_all.get("roc_auc_other"),
        "min_auc_n_o_all": encoder_all.get("min_auc_n_o"),
        "done_json_path": str(path),
    }
    row["status"] = classify_status(row)
    row["quality_value"], row["quality_is_fallback"] = _quality_value(row)
    return row


def _quality_value(row: Mapping[str, Any]) -> "tuple[Optional[float], bool]":
    """The one number worth putting next to a debugging row.

    Prefers the trainer's own selected-checkpoint score (``best_metric_value``,
    only present when ``training.json``'s ``convergence_info`` exists). Runs
    with no convergence_info at all -- most ``collapsed`` / ``unknown_*`` rows,
    see the module docstring -- would otherwise show a bare status label with
    every numeric column blank, which is not debuggable. For those, fall back
    to whichever fair-eval encoder metric matches the run's own declared
    convergence metric (``roc_auc_neutral`` for ``min_auc_n_o``, etc.), or the
    worst of the three available metrics if the declared metric is unknown --
    consistent with ``classify_status``'s own low-quality check.
    """
    if row.get("best_metric_value") is not None:
        return row["best_metric_value"], False
    metric = row.get("declared_convergent_metric") or row.get("convergence_metric")
    by_metric = {
        "correlation": row.get("correlation_all"),
        "roc_auc_neutral": row.get("roc_auc_neutral_all"),
        "roc_auc": row.get("roc_auc_neutral_all"),
        "min_auc_n_o": row.get("min_auc_n_o_all"),
    }
    if metric in by_metric and isinstance(by_metric[metric], (int, float)):
        return by_metric[metric], True
    candidates = [
        v
        for v in (row.get("roc_auc_neutral_all"), row.get("roc_auc_other_all"), row.get("correlation_all"))
        if isinstance(v, (int, float))
    ]
    if candidates:
        return min(candidates), True
    return None, False


def collect_convergence_rows(
    runs_root: Path,
    model: str,
    *,
    subdir: str = "",
) -> List[Dict[str, Any]]:
    """One row per trained ``{backend}__...`` artifact under ``runs/{model}/[{subdir}/]``."""
    model_dir = Path(runs_root) / model
    if subdir:
        model_dir = model_dir / subdir
    rows: List[Dict[str, Any]] = []
    if not model_dir.is_dir():
        return rows
    for path in sorted(model_dir.glob("*/artifacts/*/done.json")):
        stage_backend = path.parent.name.split("__", 1)[0]
        if stage_backend not in BACKENDS:
            continue
        task = path.relative_to(model_dir).parts[0]
        row = load_convergence_row(path, model=model, task=task)
        if row is not None:
            rows.append(row)
    return rows


def summarize_by_backend(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Counter] = {}
    for row in rows:
        buckets.setdefault(str(row["backend"]), Counter())[str(row["status"])] += 1
    out: List[Dict[str, Any]] = []
    for backend in sorted(buckets):
        counts = buckets[backend]
        out.append({"backend": backend, "total": sum(counts.values()), **counts})
    return out


def _fmt(value: Any, *, decimals: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


STATUS_LATEX: Dict[str, str] = {
    "converged": "Converged",
    "not_converged": "Not conv.",
    "collapsed": "Collapsed",
    "unknown_low_quality": r"Unk.\ (low qual.)",
    "unknown_no_convergence_info": r"Unk.\ (no data)",
}


def _tex_escape(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    # ``\_`` alone is an unbreakable atom -- a long underscore-only identifier
    # like "function_composition" has no other break opportunity in a narrow
    # p{} column and silently overflows into the next column (seen directly
    # when rendering the convergence-problems longtable to a real PDF and
    # comparing against the source -- pdftotext -layout's own reflow hides
    # this, so it must be checked as a rendered image, not just extracted
    # text). ``\allowbreak`` after each escaped underscore gives LaTeX a
    # legal wrap point without changing what's visible.
    for raw, esc in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_\allowbreak{}"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("&", r"\&"),
    ):
        text = text.replace(raw, esc)
    return text


def format_convergence_counts_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    label: Optional[str] = None,
) -> str:
    """Compact backend x status count table (paper-PDF companion to the terminal report)."""
    label = label or f"tab:convergence-counts-{model.replace('-', '')}"
    summary = summarize_by_backend(rows)
    status_cols = [s for s in STATUS_ORDER if any(s in row for row in summary)]
    col_headers = [STATUS_LATEX.get(s, s.replace("_", r"\_")) for s in status_cols]
    cols = "l" + "r" * (len(status_cols) + 1)
    header = "Backend & Total & " + " & ".join(col_headers) + r" \\"

    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{GRADIEND/ACTIEND convergence status ({_tex_escape(model)})}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]
    for row in summary:
        cells = [row["backend"].upper(), str(row["total"])]
        cells += [str(row.get(status, 0)) for status in status_cols]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def format_convergence_problems_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    label: Optional[str] = None,
) -> str:
    """Longtable of every collapsed / not_converged / unknown_low_quality run.

    Deliberately a ``longtable`` (spans pages on its own), not the
    one-page-resizebox ``table`` snippets the headline method tables use --
    a debugging listing can legitimately run to dozens of rows and should
    not be squeezed or truncated to fit one page.
    """
    label = label or f"tab:convergence-problems-{model.replace('-', '')}"
    problems = [r for r in rows if r.get("status") in PROBLEM_STATUSES]
    problems = sorted(
        problems,
        key=lambda r: (
            STATUS_ORDER.index(r["status"]) if r["status"] in STATUS_ORDER else len(STATUS_ORDER),
            task_sort_key(str(r["task"])),
            str(r["backend"]),
            str(r["ablation"]),
            str(r["class_key"]),
        ),
    )
    if not problems:
        return ""

    cap = (
        f"GRADIEND/ACTIEND convergence problems ({_tex_escape(model)}). "
        r"Selection score$^*$: no trainer convergence record for this run, value is the "
        r"fair-eval encoder metric instead. Components: passing/total layer components. "
        r"Seeds: converged/required (configured maximum)."
    )
    header_cells = ["Task", "Backend", "Ablation", "Class", "Status", "Metric", "Thresh", "Selection", "Step", "Components", "Seeds"]
    header_row = " & ".join(header_cells) + r" \\"
    lines = [
        r"\begin{longtable}{p{2.4cm}p{1.3cm}p{1.3cm}p{1.9cm}p{2.4cm}p{2.0cm}rrrp{1.5cm}p{2.3cm}}",
        f"  \\caption{{{cap}}} \\label{{{label}}} \\\\",
        r"  \toprule",
        f"  {header_row}",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \toprule",
        f"  {header_row}",
        r"  \midrule",
        r"  \endhead",
        r"  \midrule",
        r"  \multicolumn{11}{r}{\emph{continued on next page}} \\",
        r"  \endfoot",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]
    for row in problems:
        quality = _fmt(row.get("quality_value"))
        if row.get("quality_is_fallback") and quality != "-":
            quality += "$^*$"
        cells = [
            _tex_escape(row.get("task")),
            _tex_escape(row.get("backend")),
            _tex_escape(row.get("ablation")),
            _tex_escape(row.get("class_key")) + (r" (tensors)" if row.get("tensors") else ""),
            STATUS_LATEX.get(str(row.get("status")), _tex_escape(row.get("status"))),
            _tex_escape(row.get("convergence_metric") or row.get("declared_convergent_metric")),
            _fmt(row.get("convergence_threshold")),
            quality,
            _fmt(row.get("best_step"), decimals=0) if row.get("best_step") is not None else "-",
            _tex_escape(row.get("component_summary")),
            _tex_escape(row.get("seeds_summary")),
        ]
        lines.append("  " + " & ".join(cells) + r" \\")
    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def format_convergence_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
    only_problems: bool = False,
) -> str:
    if not rows:
        return "(no gradiend/actiend artifacts found)"

    lines: List[str] = []
    title = f"=== Convergence overview{f': {model}' if model else ''} ==="
    lines.append(title)

    summary = summarize_by_backend(rows)
    status_cols = [s for s in STATUS_ORDER if any(s in row for row in summary)]
    col_w = max(9, max((len(s) for s in status_cols), default=0) + 2)
    header = f"{'backend':<10}{'total':>7}" + "".join(f"{s:>{col_w}}" for s in status_cols)
    lines.append(header)
    lines.append("-" * len(header))
    for row in summary:
        line = f"{row['backend']:<10}{row['total']:>7}"
        for status in status_cols:
            line += f"{row.get(status, 0):>{col_w}}"
        lines.append(line)
    lines.append("")

    problems = [r for r in rows if r["status"] in PROBLEM_STATUSES]
    shown = problems if only_problems else list(rows)
    shown = sorted(
        shown,
        key=lambda r: (
            STATUS_ORDER.index(r["status"]) if r["status"] in STATUS_ORDER else len(STATUS_ORDER),
            str(r["task"]),
            str(r["backend"]),
            str(r["ablation"]),
            str(r["class_key"]),
        ),
    )
    lines.append(
        f"Problems ({len(problems)} of {len(rows)} total)"
        if only_problems
        else f"All runs ({len(rows)} total, {len(problems)} flagged)"
    )
    col = (
        "{task:<22} {backend:<8} {ablation:<8} {class_key:<14} {tensors:<7} "
        "{status:<24} {metric:<14} {thr:>7} {quality:>10} {step:>7} {max_steps:>10} "
        "{components:>10} {seeds:<24}"
    )
    lines.append(
        col.format(
            task="task",
            backend="backend",
            ablation="ablation",
            class_key="class",
            tensors="tensors",
            status="status",
            metric="conv_metric",
            thr="thresh",
            quality="selection",
            step="step",
            max_steps="max_steps",
            components="components",
            seeds="seeds (conv/req)",
        )
    )
    for row in shown:
        quality = _fmt(row.get("quality_value"))
        if row.get("quality_is_fallback") and quality != "-":
            quality += "*"
        lines.append(
            col.format(
                task=str(row["task"])[:21],
                backend=str(row["backend"]),
                ablation=str(row["ablation"]),
                class_key=str(row["class_key"])[:13],
                tensors="yes" if row["tensors"] else "",
                status=str(row["status"]),
                metric=str(row.get("convergence_metric") or row.get("declared_convergent_metric") or "-")[:13],
                thr=_fmt(row.get("convergence_threshold")),
                quality=quality,
                step=_fmt(row.get("best_step"), decimals=0) if row.get("best_step") is not None else "-",
                max_steps=_fmt(row.get("training_max_steps"), decimals=0)
                if row.get("training_max_steps") is not None
                else "-",
                components=str(row.get("component_summary") or "-"),
                seeds=str(row.get("seeds_summary") or "-"),
            )
        )
    lines.append("")
    lines.append("* selection score has no trainer convergence record; shown as the fair-eval encoder metric instead.")
    return "\n".join(lines)


def write_convergence_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument(
        "--subdir",
        default="",
        help="OUTPUT_SUBDIR value (see slurm/study_array.sh), e.g. 'suite_full'.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Also write the full row set as CSV.")
    parser.add_argument(
        "--only-problems",
        action="store_true",
        help="Only list collapsed / not_converged / unknown_low_quality rows.",
    )
    args = parser.parse_args()

    rows = collect_convergence_rows(args.runs, args.model, subdir=args.subdir)
    print(format_convergence_report(rows, model=args.model, only_problems=args.only_problems))
    if args.csv:
        write_convergence_csv(rows, args.csv)
        print(f"\nWrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
