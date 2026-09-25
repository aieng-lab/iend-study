#!/usr/bin/env python
"""Flag weight-space causal cells whose *base* model differs from every other method's.

The base model is one fixed object per task, so ``selected.base_lms`` must be
identical (to ~0.05%) for every method of a task. Activation-space methods
(ACTIEND/AGIEND/CAGA/CAA/SAE) never modify weights and give the reference value;
weight-space methods (GRADIEND, CGA) apply ``+delta`` / ``-delta`` in place. In
bf16/fp16 that round trip is not exact, so a sweep leaves the backbone drifted
and every later probability is measured against a corrupted "base" (Llama-3.1-8B
CGA: base P(target) ~1e-6 instead of ~0.5, so every P- collapses to 0.0000).

Pure stdlib, CPU-only; reads only the synced ``results.json`` files.

    python scripts/check_causal_base_drift.py --model llama-3.1-8b
    python scripts/check_causal_base_drift.py --model llama-3.1-8b --subdir suite_x --tol 0.005

Exit status 1 if any drifted cell is found.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

REFERENCE_FAMILIES = ("actiend", "actiend_ridge", "agiend", "caga", "caa", "sae", "sae_pre")
WEIGHT_SPACE_FAMILIES = ("cga", "cga_tensor_norm", "gradiend")


def _family(method_id: str) -> str:
    return str(method_id).split(":", 1)[0]


def check_task(results_path: Path, tol: float) -> Dict[str, List[str]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    by_method = ((payload.get("raw") or {}).get("causal") or {}).get("by_method") or {}
    ref: List[float] = []
    for mid, entry in by_method.items():
        selected = (entry or {}).get("selected") or {}
        if _family(mid) in REFERENCE_FAMILIES and selected.get("base_lms") is not None:
            ref.append(float(selected["base_lms"]))
    out: Dict[str, List[str]] = {}
    if not ref:
        return out
    reference = statistics.median(ref)
    # Reference methods must agree with each other; if they do not (mixed
    # protocol eras in one tree), widen the tolerance instead of crying wolf.
    spread = (max(ref) - min(ref)) / reference
    tol = max(tol, 3.0 * spread)
    for mid, entry in by_method.items():
        fam = _family(mid)
        if fam not in WEIGHT_SPACE_FAMILIES:
            continue
        for part in ("selected", "weaken_selected"):
            cell = (entry or {}).get(part) or {}
            base = cell.get("base_lms")
            if base is None:
                continue
            dev = abs(float(base) - reference) / reference
            if dev > tol:
                out.setdefault(fam, []).append(f"{mid}[{part}] {dev * 100:.1f}%")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subdir", default="", help="OUTPUT_SUBDIR tree, if any")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--tol",
        type=float,
        default=0.005,
        help="relative base_lms deviation tolerated (reference methods agree to ~5e-4)",
    )
    args = parser.parse_args()

    root = args.runs / args.model / args.subdir if args.subdir else args.runs / args.model
    drifted = 0
    for path in sorted(root.glob("*/results.json")):
        found = check_task(path, args.tol)
        if not found:
            continue
        drifted += sum(len(v) for v in found.values())
        for fam, cells in sorted(found.items()):
            print(f"{path.parent.name:22s} {fam:9s} {len(cells)} drifted cell(s): {', '.join(cells[:4])}")
    if drifted:
        print(
            f"\n{drifted} drifted weight-space cell(s): their causal numbers were measured on a "
            "corrupted base model. Recompute with FORCE_CAUSAL=1 after the exact-restore fix."
        )
        return 1
    print("no drifted weight-space causal cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
