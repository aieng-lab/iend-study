#!/usr/bin/env python
"""Promote one method family from a screen ``OUTPUT_SUBDIR`` tree into the main task results.

Example: the gemma-2-2b GRADIEND learning-rate screen
(``runs/gemma-2-2b/gradiend_lr1e7/<task>``) replaces only the ``gradiend`` rows,
causal sweeps and train summary of ``runs/gemma-2-2b/<task>/results.json``.
Every other family (caa/cga/caga/sae/actiend/agiend) is left byte-for-byte as it was.

Unlike ``merge_workstation_results.py`` this keeps the destination's *identity*
(``config_hash``, ``suite``, ``experiment_id``, ``compute``, cost block): the plain
``merge_study_payload`` lets the screen tree overwrite them, which would make the main
tree look like a different run to ``--skip-existing``.

Dry-run by default; ``--apply`` writes.  Idempotent: re-run it after every
``rsync_analysis`` pull, because a pull brings the cluster's copy of the main
``results.json`` back (with the old rows) and the smart-merge treats the cluster as
authoritative for the families it computed.

A task is promoted only when it is COMPLETE: every ``<family>__*`` artifact that exists in
the main tree also exists (with ``done.json``) in the screen tree, and the screen tree
has rows for the family with no error rows.  ``results.json`` ``status`` is NOT usable for
this: the pair and one-pole slices share one file and the first slice to finish stamps it
``ok``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.results_merge import merge_study_payload, method_family
from study.snapshot_guard import assert_resolved

# Keep the main tree's own identity; only the promoted family's content moves.
KEEP_TOP_LEVEL = (
    "config_hash",
    "suite",
    "experiment_id",
    "compute",
    "created_at",
    "started_at",
    "finished_at",
    "pipeline_stage",
)
KEEP_RAW = ("cost", "cost_summary")


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _cell_summary(done: Path) -> Tuple[Optional[bool], Optional[float], Optional[int]]:
    data = _read(done) or {}
    extras = data.get("extras", data)
    conv = (extras.get("convergence_info") or {}).get("converged")
    best = extras.get("best_score_checkpoint") or {}
    metric = best.get("min_auc_n_o") if "onepole" in done.parent.name else best.get("correlation")
    return conv, metric, best.get("global_step")


def artifact_cells(task_dir: Path, family: str) -> List[str]:
    root = task_dir / "artifacts"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.glob(f"{family}__*") if p.is_dir())


def _family_ids(payload: Dict[str, Any], family: str) -> set:
    return {
        str(r.get("method"))
        for r in payload.get("methods") or []
        if method_family(str(r.get("method") or "")) == family
    }


def completeness(src: Path, dst: Path, family: str, *, allow_missing_rows: bool = False) -> Tuple[bool, str]:
    expected = artifact_cells(dst, family)
    if not expected:
        return False, f"main tree has no {family}__* artifacts locally (sync first)"
    missing = [c for c in expected if not (src / "artifacts" / c / "done.json").is_file()]
    if missing:
        return False, f"{len(missing)}/{len(expected)} cells not finished ({', '.join(missing[:3])}...)"
    payload = _read(src / "results.json") or {}
    rows = [r for r in payload.get("methods") or [] if method_family(str(r.get("method") or "")) == family]
    if not rows:
        return False, f"no {family} rows in screen results.json"
    bad = [r.get("method") for r in rows if r.get("status") == "error"]
    if bad:
        return False, f"{len(bad)} {family} rows are errors (e.g. {bad[0]})"
    # Finished artifacts are not enough: the family's rows are written after the last cell's
    # encoder eval + causal, and a slice-merge bug (fixed 2026-09-24) erased the earlier
    # slice's encoder rows. Never let a promotion shrink the main tree's row inventory.
    dst_payload = _read(dst / "results.json") or {}
    lost = sorted(_family_ids(dst_payload, family) - _family_ids(payload, family))
    if lost and not allow_missing_rows:
        return False, f"{len(lost)} main-tree {family} rows absent in screen results.json (e.g. {lost[0]})"
    # A row can be present yet carry only causal keys (the old-code slice merge did this): the
    # headline then silently DROPS that cell. Every main-tree row that has encoder metrics must
    # still have them after promotion.
    def has_encoder(row: Dict[str, Any]) -> bool:
        metrics = row.get("metrics") or {}
        return metrics.get("roc_auc_neutral") is not None or metrics.get("roc_auc") is not None

    new_by_id = {str(r.get("method")): r for r in payload.get("methods") or []}
    stripped = sorted(
        str(r.get("method"))
        for r in dst_payload.get("methods") or []
        if method_family(str(r.get("method") or "")) == family
        and has_encoder(r)
        and str(r.get("method")) in new_by_id
        and not has_encoder(new_by_id[str(r.get("method"))])
    )
    if stripped and not allow_missing_rows:
        return False, (
            f"{len(stripped)} {family} rows lost their encoder metrics in the screen results "
            f"(e.g. {stripped[0]}); rebuild them with NO_RERUN_ORPHANS=1 first"
        )
    return True, f"{len(expected)} cells, {len(rows)} rows"


def comparison(src: Path, dst: Path, family: str) -> str:
    n_old = n_new = n = 0
    moved: List[str] = []
    for cell in artifact_cells(dst, family):
        old = _cell_summary(dst / "artifacts" / cell / "done.json")
        new = _cell_summary(src / "artifacts" / cell / "done.json")
        n += 1
        n_old += bool(old[0])
        n_new += bool(new[0])
        if bool(old[0]) != bool(new[0]):
            moved.append(f"{cell.replace(family + '__', '')}:{'conv' if new[0] else 'LOST'}")
    tail = f"  changed: {', '.join(moved)}" if moved else ""
    return f"converged {n_old}/{n} -> {n_new}/{n}{tail}"


def promote_payload(dst: Dict[str, Any], src: Dict[str, Any], families: List[str], subdir: str) -> Dict[str, Any]:
    merged = merge_study_payload(dst, src, enabled=families)
    for key in KEEP_TOP_LEVEL:
        if key in dst:
            merged[key] = dst[key]
        else:
            merged.pop(key, None)
    raw = merged.setdefault("raw", {})
    for key in KEEP_RAW:
        if key in (dst.get("raw") or {}):
            raw[key] = dst["raw"][key]
    raw["enabled"] = (dst.get("raw") or {}).get("enabled", raw.get("enabled"))
    promoted = dict(raw.get("promoted") or {})
    for fam in families:
        promoted[fam] = {
            "from_subdir": subdir,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_config_hash": src.get("config_hash"),
        }
    raw["promoted"] = promoted
    return merged


def family_counts(payload: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in payload.get("methods") or []:
        fam = method_family(str(row.get("method") or ""))
        out[fam] = out.get(fam, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subdir", required=True, help="screen tree under runs/<model>/, e.g. gradiend_lr1e7")
    ap.add_argument("--families", nargs="+", default=["gradiend"])
    ap.add_argument("--tasks", nargs="+", default=None, help="default: every task in the screen tree")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--allow-missing-rows", action="store_true",
                    help="promote even if the screen results lack some of the main tree's rows")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    model_root = args.runs / args.model
    src_root = model_root / args.subdir
    if not src_root.is_dir():
        raise SystemExit(f"no screen tree at {src_root}")
    if args.tasks:
        tasks = list(args.tasks)
    else:
        # Every task the main tree has, not just those the screen already wrote a
        # results.json for: a task missing from the screen must show up as SKIP,
        # never silently vanish from the count.
        from study.config import list_tasks

        tasks = [t for t in list_tasks() if (model_root / t).is_dir()]
        tasks += sorted(p.name for p in src_root.iterdir() if p.is_dir() and p.name not in tasks)

    promoted = skipped = 0
    for task in tasks:
        src_dir, dst_dir = src_root / task, model_root / task
        print(f"\n== {task}")
        src = _read(src_dir / "results.json")
        dst = _read(dst_dir / "results.json")
        if src is None or dst is None:
            where = "screen tree" if src is None else "main tree"
            have = f"; {len(artifact_cells(src_dir, args.families[0]))} screen artifact cell(s) synced" if src is None else ""
            print(f"   SKIP: {where} results.json missing/unreadable (not run yet, or not synced){have}")
            skipped += 1
            continue
        reasons = []
        for fam in args.families:
            ok, why = completeness(src_dir, dst_dir, fam, allow_missing_rows=args.allow_missing_rows)
            print(f"   {fam}: {'complete' if ok else 'INCOMPLETE'} ({why})")
            if ok:
                print(f"   {fam}: {comparison(src_dir, dst_dir, fam)}")
            else:
                reasons.append(why)
        if reasons:
            skipped += 1
            continue
        merged = promote_payload(dst, src, args.families, args.subdir)
        before, after = family_counts(dst), family_counts(merged)
        others = {f: (before.get(f, 0), after.get(f, 0)) for f in before if f not in args.families}
        changed_others = {f: v for f, v in others.items() if v[0] != v[1]}
        print(f"   rows {sum(before.values())} -> {sum(after.values())}; "
              f"{','.join(args.families)} {[before.get(f, 0) for f in args.families]} -> {[after.get(f, 0) for f in args.families]}")
        if changed_others:
            print(f"   ABORT: other families changed row counts: {changed_others}")
            return 1
        if not args.apply:
            promoted += 1
            continue
        target = dst_dir / "results.json"
        assert_resolved(target)  # never write over an unmerged pull
        backup = target.with_name(f"results.json.pre_promote_{args.subdir}")
        if not backup.exists():
            shutil.copy2(target, backup)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
        print(f"   PROMOTED (backup: {backup.name})")
        promoted += 1

    verb = "promoted" if args.apply else "would promote"
    print(f"\n{verb} {promoted} of {len(tasks)} task(s), skipped {skipped}" + ("" if args.apply else "  [dry run: pass --apply]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
