"""Recompute appendix ablations on the paper's Det./Int. metrics.

This deliberately bypasses legacy ``encoding_E`` retention exports.  It reads
the completed ``results.json`` artifacts for visible ``TASKS=all`` tasks,
constructs recipe-level Detection and Intervention scores with the same
functions used by ``summary_latex.py``, and reports both within-model and
same-task cross-model paired deltas.

No model training, inference, or causal sweep is run; this is CPU/I/O report
generation over existing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ablation_retention import AXES, _index_cells, paired_deltas, variants_for_axis
from analysis.family_overview import aggregate_family_cells
from analysis.recipe_overview import recipe_per_class_for_results
from study.config import list_tasks

MODELS = ("gpt2-small", "pythia-70m-deduped")
METRICS = ("detection_score", "intervention_score")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _read_recipe_cells(runs: Path, model: str, tasks: Sequence[str]) -> list[dict[str, Any]]:
    """Stream result files one at a time, retaining only small recipe summaries."""
    per_class: list[dict[str, Any]] = []
    for task in tasks:
        path = runs / model / "suite_full2" / task / "results.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        per_class.extend(recipe_per_class_for_results(payload, metrics=METRICS, results_path=str(path)))
    return aggregate_family_cells(per_class)


def _cells_for_retention(cells: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": str(row["model"]),
            "task": str(row["task"]),
            "recipe": str(row["family"]),
            "metric": str(row["metric"]),
            "mean": float(row["mean"]),
        }
        for row in cells
        if _finite(row.get("mean"))
    ]


def _default(variant: str, axis: str, default: str) -> str:
    return f"{variant.split(':', 1)[0]}:two_pole" if axis == "iend_split" else default


def _mean(rows: Iterable[Mapping[str, Any]]) -> float | None:
    values = [float(row["delta"]) for row in rows]
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:+.1f}"


def build(runs: Path, out: Path) -> None:
    tasks = list_tasks()
    out.mkdir(parents=True, exist_ok=True)
    cells_by_model = {model: _cells_for_retention(_read_recipe_cells(runs, model, tasks)) for model in MODELS}
    indices = {model: _index_cells(cells) for model, cells in cells_by_model.items()}
    rows_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}

    for model, index in indices.items():
        recipes = {key[2] for key in index}
        for spec in AXES:
            for variant in variants_for_axis(recipes, spec):
                default = _default(variant, spec.axis, spec.default)
                det = paired_deltas(index, variant=variant, default=default, metric="detection_score", model=model)
                intervention = paired_deltas(index, variant=variant, default=default, metric="intervention_score", model=model)
                rows_by_model[model].append(
                    {
                        "model": model,
                        "axis": spec.axis,
                        "variant": variant,
                        "default": default,
                        "n_det": len(det),
                        "mean_delta_det": _mean(det),
                        "det_tasks": ",".join(row["task"] for row in det),
                        "n_int": len(intervention),
                        "mean_delta_int": _mean(intervention),
                        "int_tasks": ",".join(row["task"] for row in intervention),
                    }
                )

    all_rows = [row for rows in rows_by_model.values() for row in rows]
    fields = list(all_rows[0]) if all_rows else []
    with (out / "ablation_det_int_by_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    lookups = {model: {row["variant"]: row for row in rows} for model, rows in rows_by_model.items()}
    shared = []
    for variant in sorted(set(lookups[MODELS[0]]) & set(lookups[MODELS[1]])):
        gpt = lookups[MODELS[0]][variant]
        default = gpt["default"]
        record: dict[str, Any] = {"variant": variant, "default": default}
        for metric, short in (("detection_score", "det"), ("intervention_score", "int")):
            a = {row["task"]: row for row in paired_deltas(indices[MODELS[0]], variant=variant, default=default, metric=metric, model=MODELS[0])}
            b = {row["task"]: row for row in paired_deltas(indices[MODELS[1]], variant=variant, default=default, metric=metric, model=MODELS[1])}
            common = sorted(set(a) & set(b))
            record[f"n_shared_{short}"] = len(common)
            record[f"gpt2_delta_{short}"] = _mean(a[task] for task in common)
            record[f"pythia_delta_{short}"] = _mean(b[task] for task in common)
            record[f"shared_{short}_tasks"] = ",".join(common)
        shared.append(record)
    with (out / "ablation_det_int_cross_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shared[0]))
        writer.writeheader()
        writer.writerows(shared)

    with (out / "ablation_det_int_cross_model.tex").open("w", encoding="utf-8") as handle:
        for row in shared:
            label = row["variant"].replace("_", "\\_")
            handle.write(
                f"{label} & {row['n_shared_det']} & {_fmt(row['gpt2_delta_det'])} & {_fmt(row['pythia_delta_det'])} & "
                f"{row['n_shared_int']} & {_fmt(row['gpt2_delta_int'])} & {_fmt(row['pythia_delta_int'])} \\\\\n"
            )
    print(f"Wrote Det./Int. ablation outputs to {out} for {len(tasks)} study tasks.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "tables" / "appendix_ablations" / "det_int")
    args = parser.parse_args()
    build(args.runs, args.out)


if __name__ == "__main__":
    main()
