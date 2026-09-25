"""Rebuild standalone causal summaries from results.json inventories.

Use after an older skip-existing causal refresh overwrote
``causal/summary.json`` with only the methods retried in that invocation.
The durable source is ``results.json.raw.causal.by_method``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal_eval import persist_causal_summary
from study.stages.causal import _merged_causal_summary_entries


def repair_task(output_dir: Path) -> tuple[int, int] | None:
    results_path = output_dir / "results.json"
    if not results_path.is_file():
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    causal = ((payload.get("raw") or {}).get("causal") or {})
    by_method = causal.get("by_method") or {}
    if not isinstance(by_method, dict) or not by_method:
        return None
    entries = _merged_causal_summary_entries(
        by_method,
        current=causal.get("summaries") or [],
    )
    summary_path = output_dir / "causal" / "summary.json"
    previous_count = 0
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_count = len(previous) if isinstance(previous, list) else 0
        except Exception:
            previous_count = 0
    persist_causal_summary(output_dir, entries)
    return previous_count, len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Task directory or a directory containing task subdirectories.",
    )
    args = parser.parse_args()
    root = args.root
    candidates = [root] if (root / "results.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir()
    )
    repaired = 0
    for task_dir in candidates:
        counts = repair_task(task_dir)
        if counts is None:
            continue
        before, after = counts
        repaired += 1
        print(f"{task_dir.name}: {before} -> {after} summary rows")
    print(f"repaired {repaired} task summaries")


if __name__ == "__main__":
    main()
