#!/usr/bin/env python
"""Family × task overview tables and global comparison figures.

Writes family-oracle tables (best variant per metric), a locked-recipe
extended summary (``sae:kstar``, ``caa:all_act_prediction``, …), CSV / LaTeX,
global aggregation, and comparison figures.

  python analysis/summarize_family_overview.py
  python analysis/summarize_family_overview.py --model gpt2-small
  python analysis/summarize_family_overview.py --skip-plots
  python analysis/summarize_family_overview.py --skip-recipe
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.family_overview import (
    CAGA_FAMILIES,
    CGA_FAMILIES,
    FAMILIES,
    OVERVIEW_METRICS,
    collect_family_overview,
    format_overview_latex,
    format_overview_text,
    pivot_family_task,
)
from analysis.plot_family_overview import (
    DEFAULT_FIG,
    write_family_overview_figures,
    write_global_tables,
)
from analysis.recipe_overview import collect_recipe_overview, write_recipe_overview
from analysis.task_specs import collect_task_specs, family_is_applicable, resolve_spec

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "analysis" / "tables" / "family"
DEFAULT_RECIPE_OUT = ROOT / "analysis" / "tables" / "recipe"
DEFAULT_RECIPE_FIG = ROOT / "analysis" / "figures" / "recipe"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_mean_pivot_csv(
    path: Path,
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: str,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> None:
    families, tasks, lookup = pivot_family_task(cells, metric=metric, model=model, specs=specs)
    if not families or not tasks:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", *tasks, "MEAN", *[f"{t}_min" for t in tasks], *[f"{t}_max" for t in tasks]])
        for fam in families:
            means: List[float] = []
            mins: List[str] = []
            maxs: List[str] = []
            vals: List[str] = []
            for task in tasks:
                cell = lookup.get((fam, task))
                spec = resolve_spec(specs, model=model, task=task)
                expected = family_is_applicable(fam, spec)
                if not cell:
                    mark = "-" if not expected else "NaN"
                    vals.append(mark)
                    mins.append(mark)
                    maxs.append(mark)
                    continue
                means.append(float(cell["mean"]))
                vals.append(f"{float(cell['mean']):.6f}")
                mins.append(f"{float(cell['min']):.6f}")
                maxs.append(f"{float(cell['max']):.6f}")
            mean_all = f"{sum(means) / len(means):.6f}" if means else "NaN"
            writer.writerow([fam, *vals, mean_all, *mins, *maxs])


def run_family_overview(
    *,
    runs: Path = DEFAULT_RUNS,
    out: Path = DEFAULT_OUT,
    figures: Path = DEFAULT_FIG,
    model: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    print_tables: bool = True,
    write_plots: bool = True,
    write_recipes: bool = True,
    recipe_out: Path = DEFAULT_RECIPE_OUT,
    recipe_figures: Path = DEFAULT_RECIPE_FIG,
    families: Optional[Sequence[str]] = None,
) -> bool:
    """Write family overview artifacts; optionally print every metric table.

    Returns False when no cells were found.
    """
    metric_list = list(metrics) if metrics is not None else list(OVERVIEW_METRICS)
    cells, per_class = collect_family_overview(runs, metrics=metric_list, model=model)
    specs = collect_task_specs(runs)
    out.mkdir(parents=True, exist_ok=True)

    # CGA is a normal method, not a special case: include it in the comparison
    # whenever the data actually contains it (full/full_plus suites), and leave it
    # out automatically on core suites where it is empty -- no flag, no empty rows.
    if families is None:
        present = {str(c.get("family")) for c in cells}
        extra_present = tuple(
            f for f in (*CGA_FAMILIES, *CAGA_FAMILIES) if f in present
        )
        if extra_present:
            families = tuple(FAMILIES) + extra_present

    _write_csv(
        out / "family_cells_long.csv",
        cells,
        [
            "model",
            "task",
            "family",
            "metric",
            "mean",
            "min",
            "max",
            "n_classes",
            "classes",
            "winner_methods",
        ],
    )
    _write_csv(
        out / "family_best_per_class.csv",
        per_class,
        [
            "model",
            "task",
            "family",
            "feature_class",
            "metric",
            "value",
            "winner_method",
            "results_path",
        ],
    )
    print(f"Wrote {out / 'family_cells_long.csv'} ({len(cells)} cells)")
    print(f"Wrote {out / 'family_best_per_class.csv'} ({len(per_class)} per-class bests)")

    if not cells:
        print("No family overview cells found.")
        return False

    all_txt: List[str] = []
    all_tex: List[str] = [
        "% Auto-generated by analysis/summarize_family_overview.py",
        "% Requires booktabs. Cell format: mean [min–max] of per-class bests.",
        "",
    ]

    models = sorted({str(c["model"]) for c in cells})
    for metric in metric_list:
        text = format_overview_text(cells, metric=metric, model=model, specs=specs)
        txt_path = out / f"family_overview_{metric}.txt"
        txt_path.write_text(text + "\n", encoding="utf-8")
        all_txt.append(text)
        print(f"Wrote {txt_path}")

        tex = format_overview_latex(cells, metric=metric, model=model, specs=specs)
        tex_path = out / f"family_overview_{metric}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        all_tex.append(tex)
        print(f"Wrote {tex_path}")

        for mod in models if model is None else [model]:
            _write_mean_pivot_csv(
                out / f"family_pivot_{metric}_{mod}.csv",
                cells,
                metric=metric,
                model=mod,
                specs=specs,
            )

    overview_txt = out / "family_overview_all.txt"
    overview_txt.write_text("\n\n".join(all_txt) + "\n", encoding="utf-8")
    overview_tex = out / "family_overview_all.tex"
    overview_tex.write_text("\n".join(all_tex), encoding="utf-8")
    print(f"Wrote {overview_txt}")
    print(f"Wrote {overview_tex}")

    for path in write_global_tables(
        cells, out=out, model=model, metrics=metric_list, specs=specs
    ):
        print(f"Wrote {path}")

    if write_plots:
        fig_paths = write_family_overview_figures(
            cells, out=figures, model=model, metrics=metric_list, specs=specs,
            families=families,
        )
        if fig_paths:
            print(f"Wrote {len(fig_paths)} figure files under {figures}")
            for path in fig_paths:
                print(f"  {path}")
        else:
            print("No family overview figures written.")

    if write_recipes:
        print()
        print("=== Locked-recipe overview (same instantiation across classes) ===")
        recipe_cells, recipe_per_class = collect_recipe_overview(
            runs, metrics=metric_list, model=model
        )
        write_recipe_overview(
            recipe_cells,
            recipe_per_class,
            out=recipe_out,
            figures=recipe_figures,
            model=model,
            metrics=metric_list,
            specs=specs,
            print_tables=print_tables,
            write_plots=write_plots,
        )

    if print_tables:
        print()
        for i, metric in enumerate(metric_list):
            if i:
                print()
            print(format_overview_text(cells, metric=metric, model=model, specs=specs))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=None, help="Filter to one model key")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(OVERVIEW_METRICS),
        help="Metrics to tabulate (default: full overview set)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIG,
        help="Directory for global comparison figures",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write tables (skip PNG/PDF figures)",
    )
    parser.add_argument(
        "--skip-recipe",
        action="store_true",
        help="Skip the locked-recipe extended summary",
    )
    parser.add_argument(
        "--recipe-out",
        type=Path,
        default=DEFAULT_RECIPE_OUT,
        help="Directory for locked-recipe tables",
    )
    parser.add_argument(
        "--recipe-figures",
        type=Path,
        default=DEFAULT_RECIPE_FIG,
        help="Directory for locked-recipe figures",
    )
    args = parser.parse_args()
    run_family_overview(
        runs=args.runs,
        out=args.out,
        figures=args.figures,
        model=args.model,
        metrics=args.metrics,
        print_tables=True,
        write_plots=not args.skip_plots,
        write_recipes=not args.skip_recipe,
        recipe_out=args.recipe_out,
        recipe_figures=args.recipe_figures,
    )


if __name__ == "__main__":
    main()
