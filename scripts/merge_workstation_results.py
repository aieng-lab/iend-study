#!/usr/bin/env python
"""Promote workstation LR-screen shards into canonical ``runs/<model>/<task>``.

The workstation experiment stores one method/LR arm per directory, e.g.
``runs/gemma-2-2b/study_ws/caa/lr_none/gender_en/results.json``.  Analysis
expects one merged task result per model at ``runs/gemma-2-2b/gender_en``.
This utility merges the selected study_ws arms using the pipeline's own
partial-method merger, while leaving the raw screen tree intact for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.results_merge import merge_study_payload


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def enabled_methods(payload: dict[str, Any]) -> list[str]:
    config = payload.get("config")
    methods = config.get("enabled_methods") if isinstance(config, dict) else None
    return [str(method) for method in methods or []]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--workspace", default="study_ws")
    args = parser.parse_args()

    model_root = args.runs / args.model
    source_root = model_root / args.workspace
    if not source_root.is_dir():
        print(f"No workstation workspace found at {source_root}; nothing to merge.")
        return

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(source_root.glob("*/*/*/results.json")):
        payload = read_json(path)
        task = str((payload or {}).get("task") or path.parent.name)
        if payload and task:
            grouped.setdefault(task, []).append((path, payload))

    if not grouped:
        raise SystemExit(f"No readable results.json shards under {source_root}")

    for task, shards in sorted(grouped.items()):
        destination = model_root / task / "results.json"
        merged = read_json(destination)
        for source, payload in shards:
            merged = merge_study_payload(merged, payload, enabled=enabled_methods(payload))
            print(f"  {task}: merged {source.relative_to(model_root)}")
        assert merged is not None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote canonical task results for {len(grouped)} task(s) under {model_root}")


if __name__ == "__main__":
    main()
