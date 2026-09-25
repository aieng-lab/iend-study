"""Check whether a decoder-LR validation-selection fix can be reconciled
against an already-computed test grid, with no GPU/model work.

Context: ``causal_study.py::_evaluate_decoder_split_clean`` selects the LR
(and feature_factor direction) for a GRADIEND/ACTIEND/CGA causal cell on the
*validation* split, then re-scores only that selected LR on *test*
(``decoder_study.py`` line ~1052, ``causal_eval.decoder_selected_learning_rates``).
Before that fix landed, test replayed the *entire* validation LR grid, which
means an old run's ``decoder_grid_test.json`` may already contain a score at
whatever LR the corrected selection logic would now pick -- in which case the
headline can be repaired by re-reading disk, with no rerun at all.

This script reads the raw ``decoder_grid_validation.json`` /
``decoder_grid_test.json`` (and ``_weaken_`` counterparts) pairs that
``_evaluate_decoder_split_clean`` writes directly into each artifact dir
(e.g. ``artifacts/cga__onepole__F/``), re-derives the LR(s) the CURRENT
(fixed) selection code would choose from the validation grid, and checks
whether that LR is already present in the stored test grid.

These raw grid files are NOT part of the normal ``rsync_analysis.sh`` sync
(they are large/internal, unlike ``results.json``). Pull them first with:

    MODEL=<model> OUTPUT_SUBDIR=<subdir> TASK=<task> DECODER_GRIDS=1 \\
        bash scripts/rsync_analysis.sh --go

Then run this script against the synced tree, e.g.:

    python scripts/check_decoder_grid_recovery.py runs/pythia-70m-deduped/suite_full2
    python scripts/check_decoder_grid_recovery.py runs/pythia-70m-deduped/suite_full2 --backend cga

Not yet verified against a real synced grid file (none is available in this
checkout at the time this script was written) -- sanity-check the printed
per-artifact rows against the actual JSON before trusting a "RECOVERABLE"
verdict for anything that goes into a reported number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_eval import (  # noqa: E402
    _extract_decoder_summary_and_grid,
    decoder_selected_learning_rates,
)


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _available_test_lrs(test_payload: Dict[str, Any]) -> List[float]:
    from gradiend.evaluator.decoder_eval_utils import parse_grid_candidate_id

    _, grid = _extract_decoder_summary_and_grid(test_payload)
    values = set()
    for key, entry in grid.items():
        if not isinstance(entry, dict):
            continue
        parsed = parse_grid_candidate_id(key, entry)
        if parsed is not None:
            values.add(round(float(parsed[1]), 12))
    return sorted(values)


def check_artifact_dir(art_dir: Path) -> List[Dict[str, Any]]:
    """One row per direction (strengthen/weaken) with a validation grid present."""
    name = art_dir.name  # e.g. "cga__onepole__F" or "cga__pair__F-M"
    parts = name.split("__")
    backend = parts[0] if parts else name
    cls_part = parts[-1] if len(parts) >= 3 else name
    classes = cls_part.split("-") if "pair" in name else [cls_part]

    rows: List[Dict[str, Any]] = []
    for direction, val_name, test_name in (
        ("strengthen", "decoder_grid_validation.json", "decoder_grid_test.json"),
        (
            "weaken",
            "decoder_grid_weaken_validation.json",
            "decoder_grid_weaken_test.json",
        ),
    ):
        val_path = art_dir / val_name
        if not val_path.exists():
            continue
        validation = _load(val_path)
        corrected_lrs = decoder_selected_learning_rates(validation, classes=classes)

        row: Dict[str, Any] = {
            "artifact": str(art_dir),
            "backend": backend,
            "classes": classes,
            "direction": direction,
            "corrected_selected_lrs": corrected_lrs,
        }
        test_path = art_dir / test_name
        if test_path.exists():
            available = _available_test_lrs(_load(test_path))
            missing = [
                lr for lr in corrected_lrs if round(float(lr), 12) not in available
            ]
            row["test_grid_available_lrs"] = available
            row["missing_lrs"] = missing
            row["recoverable_without_rerun"] = not missing
        else:
            row["test_grid_available_lrs"] = None
            row["missing_lrs"] = corrected_lrs
            row["recoverable_without_rerun"] = False
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "runs_dir",
        help="e.g. runs/pythia-70m-deduped/suite_full2 (a <model>/<subdir> tree)",
    )
    ap.add_argument(
        "--backend",
        action="append",
        default=None,
        help="restrict to this backend prefix (e.g. cga); repeatable",
    )
    args = ap.parse_args()
    backend_filter: Optional[Sequence[str]] = args.backend

    root = Path(args.runs_dir)
    any_grid_found = False
    for art_dir in sorted(root.glob("*/artifacts/*")):
        if not art_dir.is_dir():
            continue
        if not (art_dir / "decoder_grid_validation.json").exists():
            continue
        any_grid_found = True
        name = art_dir.name
        backend = name.split("__", 1)[0]
        if backend_filter and backend not in backend_filter:
            continue
        for row in check_artifact_dir(art_dir):
            status = "RECOVERABLE  " if row["recoverable_without_rerun"] else "NEEDS RERUN  "
            print(
                f"[{status}] {row['artifact']} ({row['direction']}) "
                f"corrected_lrs={row['corrected_selected_lrs']} "
                f"missing={row['missing_lrs']}"
            )

    if not any_grid_found:
        print(
            f"No decoder_grid_validation.json found under {root}. "
            "Sync them first with DECODER_GRIDS=1 bash scripts/rsync_analysis.sh --go"
        )


if __name__ == "__main__":
    main()
