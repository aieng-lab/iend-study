#!/usr/bin/env python
"""Report which (model, task) cells are MISSING causal per method family, so a
causal refresh can be scheduled on ONLY the tasks that actually need it (not
TASKS=all). A family with zero causal rows for a task -> NaN in the causal LaTeX
tables; this finds exactly those.

"Causal present" = a method row of that family carries a numeric
``metrics.causal_signed_effect``. Row presence alone is NOT enough (encoding rows
have no causal) -- this is why a naive "does the family have rows" check
overstates completeness, e.g. after interrupted ravel/pronoun runs.

Reads the synced ``runs/<model>/<subdir>/<task>/results.json`` (re-sync slurm
logs/results first for a current picture). Prints a per-task family grid, the
gap-task list per model, and ready-to-run ``--refresh-causal`` commands limited
to the gap tasks.

Usage:
  python scripts/report_causal_gaps.py
  python scripts/report_causal_gaps.py --models gpt2-small pythia-70m-deduped --subdir suite_full2
  python scripts/report_causal_gaps.py --families gradiend actiend sae caa cga caga
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILIES = ["gradiend", "actiend", "sae", "caa", "cga", "caga"]
# K-variants are NOT part of TASKS=all (gradiend/actiend-only ablations) -> excluded.
EXCLUDE = {"ravel_country_k5", "ravel_country_k10"}


def _visible_tasks() -> set[str]:
    """Tasks that are part of the paper (TASKS=all): excludes study_hidden ones
    (race_one_pole, religion_one_pole, ...), whose leftover run dirs on disk are
    NOT real gaps — they never appear in the LaTeX tables. Falls back to an empty
    set (no filtering) only if the registry can't be imported."""
    try:
        from study.config import list_tasks
        return set(list_tasks())  # visible-only; hidden tasks omitted
    except Exception:
        return set()


def _family(mid: str, families: Sequence[str]) -> str | None:
    base = str(mid).split(":")[0].split("|")[0]
    if base.startswith("sae_pre"):
        base = "sae"
    elif base in ("actiend_pre", "actiend_ridge"):
        base = "actiend"
    elif base == "cga_tensor_norm":
        base = "cga"
    return base if base in families else None


def causal_present(results: Dict[str, Any], families: Sequence[str]) -> Dict[str, int]:
    out = {f: 0 for f in families}
    for r in results.get("methods", []) or []:
        f = _family(r.get("method", ""), families)
        if not f:
            continue
        v = (r.get("metrics", {}) or {}).get("causal_signed_effect")
        if isinstance(v, (int, float)):
            out[f] += 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["gpt2-small", "pythia-70m-deduped"])
    p.add_argument("--subdir", default="suite_full2")
    p.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    p.add_argument("--runs", default=str(ROOT / "runs"))
    args = p.parse_args(argv)
    families = list(args.families)

    visible = _visible_tasks()
    for model in args.models:
        base = os.path.join(args.runs, model, args.subdir)
        tasks = sorted(
            d for d in (os.listdir(base) if os.path.isdir(base) else [])
            if os.path.isdir(os.path.join(base, d)) and d not in EXCLUDE
            and (not visible or d in visible)  # drop study_hidden / non-registry dirs
        )
        print(f"\n===== {model} / {args.subdir}: causal rows per family (0 => NaN) =====")
        print(f"{'task':22s} " + " ".join(f"{f[:5]:>5s}" for f in families))
        gap_tasks: List[str] = []
        gap_by_family: Dict[str, List[str]] = {f: [] for f in families}
        for t in tasks:
            path = os.path.join(base, t, "results.json")
            try:
                d = json.load(open(path, encoding="utf-8"))
            except Exception:
                print(f"{t:22s} NO-RESULTS")
                gap_tasks.append(t)
                continue
            cau = causal_present(d, families)
            flag = "".join("!" if cau[f] == 0 else " " for f in families)
            print(f"{t:22s} " + " ".join(f"{cau[f]:5d}" for f in families) + "   " + flag)
            missing = [f for f in families if cau[f] == 0]
            if missing:
                gap_tasks.append(t)
                for f in missing:
                    gap_by_family[f].append(t)
        print(f"\n  gap tasks ({len(gap_tasks)}): {','.join(gap_tasks) or '(none)'}")
        for f in families:
            if gap_by_family[f]:
                print(f"    {f:9s} missing causal on: {','.join(gap_by_family[f])}")
        if gap_tasks:
            print("\n  minimal refresh (only gap tasks; --refresh-causal recomputes just the missing ids):")
            print(
                f"  REFRESH_CAUSAL=1 SKIP_EXISTING=1 MODELS={model} "
                f"TASKS={','.join(gap_tasks)} SUITE=full_plus OUTPUT_SUBDIR={args.subdir} "
                f"METHODS={','.join(families)} SKIP_LOCALIZATION=1 bash slurm/study_array.sh"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
