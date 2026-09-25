"""Build a reproducible appendix-ablation inventory from generated study artifacts.

This script deliberately reports *coverage and provenance*, rather than choosing
another winner from the full-suite grid.  It separates settings-selection
ablations (replicated on the two small models) from robustness checks and
single-task optimization diagnostics.

Run:
    python analysis/appendix_ablation_inventory.py
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "tables" / "appendix_ablations"
MODELS = ("gpt2-small", "pythia-70m-deduped")
STUDY_RETENTION = OUT / "study_task_retention"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tex(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def artifact(path: Path) -> str:
    return "available" if path.exists() else "missing"


def suite_coverage(model: str) -> tuple[int, int, int, int]:
    path = ROOT / "analysis" / "tables" / f"latex_{model}_suite_full2" / f"summary_merged_{model}.csv"
    rows = read_csv(path)
    cga_tn = sum(row.get("backend") == "cga_tensor_norm" for row in rows)
    return (len({row["task"] for row in rows}), len({row["method_group"] for row in rows}), len(rows), cga_tn)


def retention_summary(model: str) -> tuple[int, int, int]:
    rows = read_csv(STUDY_RETENTION / model / "retention.csv")
    return (len(rows), sum(row.get("tier") == "main" for row in rows), sum(row.get("tier") == "review_promote" for row in rows))


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpt_tasks, gpt_methods, gpt_cells, gpt_cga_tn = suite_coverage(MODELS[0])
    pyt_tasks, pyt_methods, pyt_cells, pyt_cga_tn = suite_coverage(MODELS[1])
    gpt_axes, gpt_main, gpt_review = retention_summary(MODELS[0])
    pyt_axes, pyt_main, pyt_review = retention_summary(MODELS[1])

    rows = [
        {
            "section": "Full-suite setting selection",
            "role": "protocol selection",
            "scope": f"{gpt_axes} shared variants (GPT-2 and Pythia)",
            "evidence": "paired retention: E, causal effect, rank flips",
            "status": "shared recipe IDs; unequal task coverage means no promotion claim",
            "artifact": "analysis/tables/appendix_ablations/study_task_retention/{model}/retention.csv",
        },
        {
            "section": "SAE site and capacity",
            "role": "robustness / method definition",
            "scope": "filled prediction SAE vs sae_pre; k=1, k*, all-layer k=1",
            "evidence": "paired retention plus per-task matrices",
            "status": "replicated small-model analysis",
            "artifact": "analysis/tables/appendix_ablations/study_task_retention/{model}/retention.csv",
        },
        {
            "section": "CGA tensor normalization",
            "role": "exploratory robustness",
            "scope": f"cga_tensor_norm cells: GPT-2 {gpt_cga_tn}, Pythia {pyt_cga_tn}",
            "evidence": "cga_tensor_norm rows in generated matrices",
            "status": "reported separately; not a headline method",
            "artifact": "analysis/tables/latex_{model}_suite_full2/summary_merged_{model}.csv",
        },
        {
            "section": "Layerwise detection and intervention",
            "role": "localization robustness",
            "scope": "GPT-2-small and Pythia-70M, one-pole and pairwise panels",
            "evidence": "per-layer E, AUC, specificity, signed causal effect, LMS",
            "status": "generated figures available",
            "artifact": "analysis/figures/layer_{model}_suite_full2/layer_figures.pdf",
        },
        {
            "section": "All-layer versus best layer",
            "role": "localization diagnostic",
            "scope": "where layer-vs-all diagnostic is present",
            "evidence": "best-layer minus all-layer encoding E",
            "status": artifact(ROOT / "analysis" / "tables" / "ablation_retention" / MODELS[1] / "layer_vs_all_diagnostic.csv"),
            "artifact": "analysis/tables/ablation_retention/{model}/layer_vs_all_diagnostic.csv",
        },
        {
            "section": "ACTIEND ridge and decoder LR",
            "role": "mechanistic optimization diagnostic",
            "scope": "GPT-2-small / gender_en; 3 paired seeds",
            "evidence": "decoder norm, frozen-score ridge EV, reachability",
            "status": "already typeset in Appendix",
            "artifact": "iclr2027/single_paper/tables/actiend_ablation_*.tex",
        },
    ]

    with (OUT / "ablation_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (OUT / "ablation_inventory.tex").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{tex(row['section'])} & {tex(row['role'])} & {tex(row['scope'])} & {tex(row['status'])} \\\\\n"
            )

    # Keep an auditable, paper-ready record that retention was calculated on
    # visible study tasks only.  Do not use raw recipe artifacts that include
    # hidden diagnostics such as ioi.
    by_model = {
        model: {row["variant"]: row for row in read_csv(STUDY_RETENTION / model / "retention.csv")}
        for model in MODELS
    }
    variants = sorted(set(by_model[MODELS[0]]) | set(by_model[MODELS[1]]))
    coverage_rows = [
        {
            "variant": variant,
            "gpt2_n_paired_E": by_model[MODELS[0]].get(variant, {}).get("n_paired_E", "--"),
            "pythia_n_paired_E": by_model[MODELS[1]].get(variant, {}).get("n_paired_E", "--"),
            "gpt2_tier": by_model[MODELS[0]].get(variant, {}).get("tier", "--"),
            "pythia_tier": by_model[MODELS[1]].get(variant, {}).get("tier", "--"),
        }
        for variant in variants
    ]
    with (OUT / "study_task_retention_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    with (OUT / "study_task_retention_coverage.tex").open("w", encoding="utf-8") as handle:
        for row in coverage_rows:
            handle.write(
                f"{tex(row['variant'])} & {row['gpt2_n_paired_E']} & {row['pythia_n_paired_E']} & "
                f"{tex(row['gpt2_tier'])} / {tex(row['pythia_tier'])} \\\\\n"
            )

    plan = f"""# Appendix ablation plan

Generated from current CSV/PDF artifacts by `analysis/appendix_ablation_inventory.py`.

## 1. Full-suite setting selection (main appendix result)

Use canonical `suite_full2_decision` recipe cells re-analysed with
`--study-tasks` first.  It has the
same {gpt_axes} candidate variants for GPT-2-small and Pythia-70M ({gpt_main}/
{pyt_main} core rows and {gpt_review}/{pyt_review} review rows, respectively).
The filter retains exactly the 15 visible TASKS=all study tasks and excludes
hidden diagnostics such as `ioi`.  The number of paired tasks can still differ
per contrast; report that coverage in each row.  Do not call a result a
cross-model replication unless both models are evaluated on the same task
intersection.  Do not turn full-grid rows into extra headline methods.

## 2. SAE construction and pre-prediction site

Give SAE-pre its own short subsection: it is a classical pre-prediction-site
baseline, not a post-hoc winner search.  Report k=1, k-star, and all-layer k=1
using the same retention table, then point to the full per-task matrices.  Keep
it outside the headline figures unless explicitly promoted by the selection
rule.

## 3. CGA tensor normalization

Report the `cga_tensor_norm` rows as an exploratory normalization check.  The
current generated matrices contain {gpt_cga_tn} GPT-2 and {pyt_cga_tn} Pythia
method-task rows, within full grids of {gpt_tasks}/{pyt_tasks} tasks and
{gpt_methods}/{pyt_methods} method groups.  It must remain outside headline aggregates because its current
provenance is an auxiliary artifact grid rather than the locked core protocol.

## 4. Layerwise localization and intervention

Include two multi-panel figures per small model: detection/localization and
intervention.  The existing `layer_figures.pdf` reports E, AUC, specificity,
signed effect, weakening effect, and LMS, separately for one-pole and pairwise
regimes.  The all-layer-versus-best-layer CSV is a diagnostic, not a selection
test: all-layer aggregation is not expected to dominate an oracle best layer.

## 5. ACTIEND decoder reachability and ridge ceiling

Retain the existing decoder-budget subsection.  It should be explicitly marked
as a single-task mechanism experiment (GPT-2-small/gender_en, three paired
seeds), not as a cross-model learning-rate optimization.  The current tables
already report endpoint EV, the frozen-score ridge ceiling, decoder norms, and
the matched-schedule test.

## Suggested order in the paper

1. Scaling selection: ablation retention (already present).
2. SAE construction and pre-prediction site.
3. CGA normalization scope (exploratory).
4. Layerwise detection/intervention and all-layer diagnostic.
5. ACTIEND ridge/decoder-learning-rate mechanism (already present).

This order makes the distinction visible: sections 1--2 justify the core
recipe; sections 3--5 test robustness or mechanisms and do not select a new
headline method.
"""
    (OUT / "appendix_plan.md").write_text(plan, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
