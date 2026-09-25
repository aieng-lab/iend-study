"""Class-stripped config overview: same recipe across classes, then mean [min–max].

``sae:M:kstar`` / ``sae:asian:kstar`` / ``sae:F:kstar`` → ``sae:kstar``.
``caa:M:L5_act_prediction`` / ``caa:F:L5_act_prediction`` → ``caa:L5_act_prediction``.
Unlike the family oracle, a layer or k-choice is never swapped for a better one.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from analysis.family_overview import (
    FAMILIES,
    OVERVIEW_METRICS,
    _candidate_rows,
    _iter_results_json,
    _load_results,
    _metric_value,
    _parts,
    aggregate_family_cells,
    feature_class_for_method,
    format_overview_latex,
    format_overview_text,
)
from analysis.task_specs import group_is_applicable

ROOT = Path(__file__).resolve().parents[1]

# Preferred row order for headline configs; L* / other tails are appended.
RECIPES: Tuple[str, ...] = (
    "gradiend:two_pole",
    "gradiend:one_pole",
    "actiend:two_pole",
    "actiend:one_pole",
    "actiend_pre:two_pole",
    "actiend_pre:one_pole",
    "sae:kstar",
    "sae:k1",
    "sae_pre:kstar",
    "sae_pre:k1",
    "caa:act_prediction",
    "caa:all_act_prediction",
)

RECIPE_TITLE = (
    "Config x task -- class token stripped, then mean [min-max] "
    "(sae:M:L5 + sae:F:L5 -> sae:L5)"
)

_LAYER_TAIL = re.compile(r"^L(\d+)(.*)$")


def _is_pair_token(token: str, known: Sequence[str]) -> bool:
    if "-" not in token:
        return False
    bits = token.split("-")
    known_set = {str(c) for c in known}
    return all(b in known_set for b in bits)


def recipe_for_method(
    method_id: str,
    *,
    known_classes: Sequence[str],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Strip the feature-class token; the rest is the comparable config."""
    # Proxy-eval suffixes are not a different training config.
    method_id = str(method_id).split("|", 1)[0]
    parts = _parts(method_id)
    if not parts or parts[0] not in FAMILIES:
        return None
    family = parts[0]
    metrics = metrics or {}
    cls = feature_class_for_method(method_id, known_classes=known_classes, metrics=metrics)
    if cls is None:
        return None

    tail = [
        p
        for p in parts[1:]
        if p != cls and not _is_pair_token(p, known_classes)
    ]
    if tail:
        return f"{family}:{':'.join(tail)}"

    if family in {"gradiend", "actiend", "actiend_pre"}:
        abl = str(metrics.get("ablation") or "")
        if abl == "pair":
            return f"{family}:two_pole"
        if abl == "one_pole":
            return f"{family}:one_pole"
    return None


def recipe_order(names: Sequence[str]) -> List[str]:
    """Headlines first, then per-family layer tails (L0, L1, …), then other configs."""
    seen = list(dict.fromkeys(str(n) for n in names if n))
    fam_rank = {f: i for i, f in enumerate(FAMILIES)}

    def key(name: str) -> Tuple[Any, ...]:
        if name in RECIPES:
            return (0, RECIPES.index(name), name)
        fam, _, rest = name.partition(":")
        fi = fam_rank.get(fam, 9)
        m = _LAYER_TAIL.match(rest)
        if m:
            return (1, fi, int(m.group(1)), m.group(2), name)
        return (2, fi, rest, name)

    return sorted(seen, key=key)


def recipes_in_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
) -> List[str]:
    names = {
        str(c["family"])
        for c in cells
        if c.get("family") and (model is None or c.get("model") == model)
    }
    return recipe_order([*RECIPES, *sorted(names)])


def recipe_per_class_for_results(
    results: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    results_path: str = "",
) -> List[Dict[str, Any]]:
    """One locked method per (recipe, class); emit that row for every metric."""
    parts = Path(results_path).parts if results_path else ()
    model = str(results.get("model") or (parts[-3] if len(parts) >= 3 else "unknown"))
    task = str(results.get("task") or (parts[-2] if len(parts) >= 2 else "unknown"))
    try:
        from results_schema import require_claim_classes, require_target_classes

        known = sorted(set(require_claim_classes(dict(results))) | set(require_target_classes(dict(results))))
    except ValueError:
        return []

    chosen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cand in _candidate_rows(results, known_classes=known):
        recipe = recipe_for_method(
            cand["method"], known_classes=known, metrics=cand.get("metrics") or {}
        )
        if recipe is None:
            continue
        key = (recipe, str(cand["feature_class"]))
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = cand
        elif cand.get("has_encoder") and not prev.get("has_encoder"):
            chosen[key] = cand

    out: List[Dict[str, Any]] = []
    for (recipe, cls), cand in sorted(chosen.items()):
        for metric in metrics:
            val = _metric_value(cand["metrics"], metric)
            if val is None:
                continue
            out.append(
                {
                    "model": model,
                    "task": task,
                    "family": recipe,
                    "recipe": recipe,
                    "feature_class": cls,
                    "metric": metric,
                    "value": val,
                    "winner_method": cand["method"],
                    "results_path": results_path,
                }
            )
    return out


def collect_recipe_overview(
    runs_root: Path = ROOT / "runs",
    *,
    metrics: Sequence[str] = OVERVIEW_METRICS,
    model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        per_class.extend(
            recipe_per_class_for_results(payload, metrics=metrics, results_path=str(path))
        )
    return aggregate_family_cells(per_class), per_class


def format_recipe_text(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> str:
    return format_overview_text(
        cells,
        metric=metric,
        model=model,
        specs=specs,
        families=recipes_in_cells(cells, model=model),
        row_label="recipe",
        title=RECIPE_TITLE,
        applicable=group_is_applicable,
    )


def write_recipe_overview(
    cells: Sequence[Mapping[str, Any]],
    per_class: Sequence[Mapping[str, Any]],
    *,
    out: Path,
    figures: Path,
    model: Optional[str] = None,
    metrics: Sequence[str],
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    print_tables: bool = True,
    write_plots: bool = True,
) -> None:
    """Write locked-recipe tables, global aggregation, and summary figures."""
    from analysis.plot_family_overview import write_family_overview_figures, write_global_tables

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})

    def _write_mean_pivot_csv(
        path: Path,
        rows: Sequence[Mapping[str, Any]],
        *,
        metric: str,
        model: str,
    ) -> None:
        from analysis.family_overview import pivot_family_task
        from analysis.task_specs import resolve_spec

        families, tasks, lookup = pivot_family_task(
            rows,
            metric=metric,
            model=model,
            specs=specs,
            families=recipes_in_cells(rows, model=model),
        )
        if not families or not tasks:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["recipe", *tasks, "MEAN"])
            for fam in families:
                means: List[float] = []
                vals: List[str] = []
                for task in tasks:
                    cell = lookup.get((fam, task))
                    spec = resolve_spec(specs, model=model, task=task)
                    expected = group_is_applicable(fam, spec)
                    if not cell:
                        vals.append("-" if not expected else "NaN")
                        continue
                    means.append(float(cell["mean"]))
                    vals.append(f"{float(cell['mean']):.6f}")
                mean_all = f"{sum(means) / len(means):.6f}" if means else "NaN"
                writer.writerow([fam, *vals, mean_all])

    _write_csv(
        out / "recipe_cells_long.csv",
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
        out / "recipe_per_class.csv",
        per_class,
        [
            "model",
            "task",
            "family",
            "recipe",
            "feature_class",
            "metric",
            "value",
            "winner_method",
            "results_path",
        ],
    )
    print(f"Wrote {out / 'recipe_cells_long.csv'} ({len(cells)} cells)")
    print(f"Wrote {out / 'recipe_per_class.csv'} ({len(per_class)} per-class rows)")
    if not cells:
        print("No locked-recipe cells found.")
        return

    all_txt: List[str] = []
    all_tex: List[str] = [
        "% Auto-generated locked-recipe overview (same instantiation across classes).",
        "% Requires booktabs. Cell format: mean [min–max].",
        "",
    ]
    models = sorted({str(c["model"]) for c in cells})
    for metric in metrics:
        text = format_recipe_text(cells, metric=metric, model=model, specs=specs)
        txt_path = out / f"recipe_overview_{metric}.txt"
        txt_path.write_text(text + "\n", encoding="utf-8")
        all_txt.append(text)
        print(f"Wrote {txt_path}")
        tex = format_recipe_latex(cells, metric=metric, model=model, specs=specs)
        tex_path = out / f"recipe_overview_{metric}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        all_tex.append(tex)
        print(f"Wrote {tex_path}")
        for mod in models if model is None else [model]:
            _write_mean_pivot_csv(
                out / f"recipe_pivot_{metric}_{mod}.csv",
                cells,
                metric=metric,
                model=mod,
            )

    overview_txt = out / "recipe_overview_all.txt"
    overview_tex = out / "recipe_overview_all.tex"
    overview_txt.write_text("\n\n".join(all_txt) + "\n", encoding="utf-8")
    overview_tex.write_text("\n".join(all_tex), encoding="utf-8")
    print(f"Wrote {overview_txt}")
    print(f"Wrote {overview_tex}")

    order = recipes_in_cells(cells, model=model)
    for path in write_global_tables(
        cells,
        out=out,
        model=model,
        metrics=metrics,
        specs=specs,
        families=order,
        applicable=group_is_applicable,
        prefix="recipe",
    ):
        print(f"Wrote {path}")

    if write_plots:
        fig_paths = write_family_overview_figures(
            cells,
            out=figures,
            model=model,
            metrics=metrics,
            specs=specs,
            families=order,
            applicable=group_is_applicable,
            plots="summary",
        )
        if fig_paths:
            print(f"Wrote {len(fig_paths)} recipe figure files under {figures}")
        else:
            print("No locked-recipe figures written.")

    if print_tables:
        print()
        for i, metric in enumerate(metrics):
            if i:
                print()
            print(format_recipe_text(cells, metric=metric, model=model, specs=specs))


def format_recipe_latex(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> str:
    metric_tex = str(metric).replace("_", r"\_")
    return format_overview_latex(
        cells,
        metric=metric,
        model=model,
        specs=specs,
        families=recipes_in_cells(cells, model=model),
        row_label="recipe",
        applicable=group_is_applicable,
        label_prefix="recipe",
        caption=(
            f"Config overview ({metric_tex}): class token stripped, "
            f"then mean [min--max] of per-class values."
        ),
    )
