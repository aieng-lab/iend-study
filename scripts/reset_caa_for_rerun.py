"""Force re-entry of two-pole tasks that are ``status=ok`` but have no two-pole CAA.

Why this exists: ``study/runner.py::_should_skip_existing`` decides task reuse from
``status`` + ``config_hash`` ONLY -- it does not look at ``--methods``.  So a task
that finished before two-pole CAA existed is ``ok`` and is skipped whole by
``--skip-existing`` regardless of ``METHODS=caa`` vs ``METHODS=all``; its pipeline
never runs, so the CAA-encode reuse fix (``should_reuse_caa_encode``) never fires
and it never gains ``caa:A-B:C`` pair rows.  This flips exactly those tasks
``ok -> partial`` so a normal ``--skip-existing`` relaunch re-enters them; every
other stage is still reused per-stage (train reload, SAE/CAA-causal ok-id skip),
and CAA encode recomputes because it now wants pairs it does not have.

Only touches a task when ALL hold: two-pole-capable (``config.ablations.pair`` is
true), ``status == "ok"``, and it has no ``caa:A-B:...`` pair row.  One-pole-only
tasks (no pairs to add) and already-non-ok tasks (already re-enter) are left alone.
Local/CPU, idempotent, ``--dry-run`` to preview.  Mirrors
``reset_one_pole_for_rerun.py``'s surgical-status-flip pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def _is_pair_method(method_id: str) -> bool:
    parts = str(method_id).split(":")
    return len(parts) >= 2 and "-" in parts[1]


def _task_has_pair_caa(payload: dict) -> bool:
    for row in payload.get("methods") or []:
        mid = str(row.get("method") or "")
        if mid.startswith("caa:") and _is_pair_method(mid):
            return True
    raw_caa = ((payload.get("raw") or {}).get("caa") or {}).get("method_metrics") or {}
    return any(k.startswith("caa:") and _is_pair_method(k) for k in raw_caa)


def _wants_pairs(payload: dict) -> bool:
    abl = ((payload.get("config") or {}).get("ablations")) or {}
    return bool(abl.get("pair", False))


def _iter_results(roots: List[Path]):
    for root in roots:
        for path in sorted(root.glob("*/results.json")):
            yield path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="runs/<model>/<subdir> dirs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    flipped: List[str] = []
    for path in _iter_results([Path(r) for r in args.roots]):
        task = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {task}: unreadable ({exc})")
            continue
        if payload.get("status") != "ok":
            continue  # already re-enters
        if not _wants_pairs(payload):
            continue  # one-pole-only: no pairs to add
        if _task_has_pair_caa(payload):
            continue  # already has two-pole CAA
        flipped.append(task)
        if args.dry_run:
            print(f"[dry-run] would flip {task}: status ok -> partial (no pair CAA)")
            continue
        payload["status"] = "partial"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"flipped {task}: status ok -> partial (will re-enter for two-pole CAA)")

    verb = "would flip" if args.dry_run else "flipped"
    print(f"\n{verb} {len(flipped)} task(s): {', '.join(flipped) if flipped else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
