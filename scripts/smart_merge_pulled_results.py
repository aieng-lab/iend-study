#!/usr/bin/env python
"""Smart-merge a freshly-pulled ``results.json`` against its pre-pull snapshot.

Called by ``scripts/rsync_analysis.sh`` (the cluster) and
``scripts/rsync_analysis_munich.sh`` (a second cluster) immediately after their main
rsync call. The two clusters can independently compute *different,
non-overlapping* method families for the same (model, task) -- this is the
normal case for ``qwen3.5-2b-base``, split between a second cluster (bulk) and the cluster
(overflow cells the a second cluster job-slot cap couldn't fit). A plain rsync file copy
overwrites ``results.json`` wholesale, so whichever cluster you happen to
pull from *last* silently wins and the other cluster's method rows are
gone -- there is nothing left on disk to recover them from, since both
sides' `runs/` trees live on separate filesystems with no merge of their
own. See CLAUDE.md's "results.json checkpoint/merge semantics" section: the
exact same class of lost-update problem the pipeline's own
``merge_and_write_results_locked`` fixes for two Slurm jobs writing the
*same* tree concurrently -- this script is the cross-cluster analogue, run
locally where both trees actually meet.

Each rsync wrapper snapshots every local ``results.json`` to a sibling
``results.json.presync`` right before the rsync call overwrites it (that
snapshot is the "what did we have before this pull" state). This script
finds every such snapshot under the given root, JSON-merges it against
whatever the rsync just wrote using the pipeline's own
``study.results_merge.merge_study_payload`` -- the identical function
``deep_pipeline.py`` uses for concurrent same-cluster writers, and the same
pattern ``scripts/merge_workstation_results.py`` already established for
workstation-vs-cluster shards -- and writes the union back. The remote
payload's own ``config.enabled_methods`` says which families it is
authoritative for (exactly what ``--methods <family>`` on that run
computed); everything else falls back to whatever the snapshot already had.
The snapshot is always removed afterward: either its content is now folded
into the merged file, or (if rsync brought nothing down, e.g. the remote
lacks this task) it is restored verbatim first.

Usage (called by the bash wrappers, but works standalone too):
    python scripts/smart_merge_pulled_results.py runs/qwen3.5-2b-base/
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.results_merge import _cli_family_for, _mid_in_backends, merge_study_payload, method_family
from study.snapshot_guard import (
    SNAPSHOT_SUFFIX,
    is_unresolved,
    snapshot_path,
    unresolved_snapshots,
)


def read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def enabled_methods(payload: dict[str, Any]) -> list[str]:
    config = payload.get("config")
    methods = config.get("enabled_methods") if isinstance(config, dict) else None
    return [str(m) for m in methods or []]


def _method_ids(payload: dict[str, Any]) -> set[str]:
    return {str(m.get("method")) for m in payload.get("methods") or [] if isinstance(m, dict)}


def lost_local_methods(
    prev: dict[str, Any], merged: dict[str, Any], enabled: set[str]
) -> list[str]:
    """Method ids the snapshot had, in families the pull is NOT authoritative for,
    that the merged file no longer has.  Must be empty before a snapshot is deleted."""
    kept = _method_ids(merged)
    # The merge decides authority per *CLI* family (``sae_pre`` piggybacks on
    # ``sae``, ``cga_tensor_norm`` on ``cga``); judge it the same way, or a
    # pull authoritative for ``sae`` is wrongly refused for dropping sae_pre.
    return sorted(
        mid
        for mid in _method_ids(prev)
        if _cli_family_for(method_family(mid)) not in enabled and mid not in kept
    )


LOCAL_MERGE_MTIME_SKEW_NS = 1_000_000_000

# Keys the merge itself stamps onto ``raw``; they say nothing about the results.
_VOLATILE_RAW = ("merged_from_previous", "enabled")


def nothing_to_add(merged: dict[str, Any], current: dict[str, Any]) -> bool:
    """True if merging the snapshot into the pulled file changed no result.

    Compared per method id (order-insensitive) plus the ``raw`` blocks, ignoring the
    bookkeeping the merge adds.  When true the freshly pulled file must be LEFT ALONE:
    rewriting it gives it a new mtime/size, so the next ``rsync -a`` sees a file that
    differs from the remote and downloads it again, and the whole merge repeats on
    every sync (which is what made this step run over hundreds of unchanged files).
    """
    def by_id(payload: dict[str, Any]) -> dict[str, Any]:
        return {str(m.get("method")): m for m in payload.get("methods") or [] if isinstance(m, dict)}

    if by_id(merged) != by_id(current):
        return False

    def raw(payload: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (payload.get("raw") or {}).items() if k not in _VOLATILE_RAW}

    return raw(merged) == raw(current)


def pinned_families(prev: dict[str, Any], current: dict[str, Any]) -> set[str]:
    """Families the LOCAL file carries a ``raw.promoted`` marker for and the pull does not.

    ``scripts/promote_subdir_results.py`` replaces a family in the main results with a screen
    tree's rows.  The cluster's own copy of that file still holds the OLD rows and, once it
    is rewritten for any reason, the pull treats it as authoritative for every family in its
    ``enabled_methods`` and would silently revert the promotion.  A family promoted locally
    (marker in the snapshot, absent from the pull) is therefore pinned: the pulled rows and
    causal entries for it are ignored.
    """
    mine = set(((prev.get("raw") or {}).get("promoted") or {}).keys())
    theirs = set(((current.get("raw") or {}).get("promoted") or {}).keys())
    return {str(f) for f in mine - theirs}


def without_families(payload: dict[str, Any], families: set[str]) -> dict[str, Any]:
    """Copy of ``payload`` with the given families' method rows and causal entries removed."""
    out = dict(payload)
    out["methods"] = [
        m for m in payload.get("methods") or []
        if not (isinstance(m, dict) and _cli_family_for(method_family(str(m.get("method") or ""))) in families)
    ]
    raw = dict(payload.get("raw") or {})
    causal = dict(raw.get("causal") or {})
    if isinstance(causal.get("by_method"), dict):
        causal["by_method"] = {
            k: v for k, v in causal["by_method"].items() if not _mid_in_backends(str(k), sorted(families))
        }
        raw["causal"] = causal
    out["raw"] = raw
    return out


def superseded_target(snap: Path) -> Path:
    target = snap.with_name("results.json.superseded")
    if target.exists():
        target = snap.with_name(f"results.json.superseded.{int(snap.stat().st_mtime)}")
    return target


def snapshot_all(root: Path) -> int:
    """Hardlink every ``results.json`` under ``root`` to ``results.json.presync``.

    A task whose snapshot is still unmerged (an earlier pull was never merged) is
    SKIPPED, never re-linked: re-linking would replace the only surviving copy of
    the pre-pull data with the file that pull already overwrote.  The post-pull
    merge then folds that older snapshot into the new pull.
    """
    made = 0
    skipped: List[Path] = []
    for results in sorted(root.rglob("results.json")):
        snap = snapshot_path(results)
        if snap.exists():
            if is_unresolved(snap):
                skipped.append(snap)
                continue
            snap.unlink()  # same inode as results.json: nothing to lose
        os.link(results, snap)
        made += 1
    print(f"smart-merge: snapshotted {made} results.json file(s)")
    if skipped:
        print(f"smart-merge: kept {len(skipped)} older UNMERGED snapshot(s) untouched "
              "(they hold data results.json lacks):", file=sys.stderr)
        for snap in skipped[:10]:
            print(f"    {snap}", file=sys.stderr)
    return 0


def supersede_all(root: Path) -> int:
    """Retire unmerged snapshots WITHOUT deleting them.

    For the case where the current ``results.json`` is deliberately the source of
    truth (e.g. a later cluster rerun) and the old snapshot is a stale leftover.
    The snapshot is renamed to ``results.json.superseded`` -- kept on disk, ignored
    by the guard and by the merge.  Nothing is deleted or rewritten.
    """
    n = 0
    for snap in unresolved_snapshots(root):
        snap.rename(superseded_target(snap))
        n += 1
    print(f"smart-merge: retired {n} stale snapshot(s) to results.json.superseded (kept, not deleted)")
    return 0


def merge_all(root: Path) -> int:
    snapshots: List[Path] = sorted(root.rglob(f"results.json{SNAPSHOT_SUFFIX}"))
    if not snapshots:
        print("smart-merge: no pre-pull snapshots found, nothing to merge")
        return 0

    merged_n = restored_n = untouched_n = 0
    failures: List[str] = []
    for snap in snapshots:
        dest = snap.with_name("results.json")
        rel = snap.relative_to(root) if snap.is_relative_to(root) else snap
        try:
            if not is_unresolved(snap):
                # rsync left results.json alone: the snapshot is the same inode.
                snap.unlink(missing_ok=True)
                untouched_n += 1
                continue

            # Quick check 1: byte-identical (rsync re-sent the same content because only
            # the mtime differed).  No JSON parsing, nothing to fold in.
            if dest.exists() and filecmp.cmp(snap, dest, shallow=False):
                snap.unlink(missing_ok=True)
                untouched_n += 1
                continue

            prev = read_json(snap)
            current = read_json(dest)

            if current is None:
                # rsync didn't bring a (readable) file down for this task -- put
                # the pre-pull state back rather than leaving it deleted/broken.
                if prev is None:
                    raise ValueError("neither results.json nor its snapshot is readable JSON")
                dest.write_text(json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
                snap.unlink(missing_ok=True)
                restored_n += 1
                continue

            if prev is None:
                # Snapshot unreadable: keep it for manual inspection, never delete.
                raise ValueError("snapshot is not readable JSON; kept for manual repair")

            enabled = set(enabled_methods(current))
            # study.results_merge._merge_causal_raw only folds a payload's
            # raw.causal.by_method into the result when "causal" is in ``enabled``,
            # but a run's config.enabled_methods lists method families, not stages
            # (the a second cluster qwen files have caga/caa/... and no "causal"). Without this,
            # every pulled causal entry was silently dropped and only the encoder
            # rows survived the merge.
            cur_causal = ((current.get("raw") or {}).get("causal") or {})
            if cur_causal.get("by_method"):
                enabled.add("causal")
            pinned = pinned_families(prev, current)
            pull = without_families(current, pinned) if pinned else current
            if pinned:
                enabled -= pinned
                print(f"smart-merge: {rel}: keeping locally promoted {sorted(pinned)} "
                      "(the pull's rows for them are ignored)")
            merged = merge_study_payload(prev, pull, enabled=enabled)

            lost = lost_local_methods(prev, merged, enabled)
            if lost:
                raise ValueError(
                    f"merge would drop {len(lost)} local method row(s) "
                    f"(e.g. {lost[:3]}); snapshot kept"
                )
            # Quick check 2: the snapshot adds nothing to what was pulled.  Keep the pulled
            # file (and its mtime) so the next rsync skips it.
            unchanged = nothing_to_add(merged, current)
            if not unchanged:
                pulled_mtime_ns = dest.stat().st_mtime_ns  # the remote's mtime, kept by rsync -a
                tmp = dest.with_name(dest.name + ".merge.tmp")
                tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(dest)
                # The merged file legitimately differs from the remote, so ``rsync -a`` would
                # re-download it on every pull. Stamp it one second NEWER than the pulled
                # version: the wrappers pass ``rsync --update`` (skip files newer on the
                # receiver), so it is left alone until the cluster writes the file again
                # (a later write is newer, an equal-mtime one is also transferred).
                stamp = pulled_mtime_ns + LOCAL_MERGE_MTIME_SKEW_NS
                os.utime(dest, ns=(stamp, stamp))
            # Rows the pull is authoritative for (the remote recomputed that
            # family) may legitimately differ; but never destroy the only copy
            # of them.  Keep the old file as results.json.superseded instead of
            # deleting it (ignored by the guard, disk cost only where rows differ).
            dropped = sorted(_method_ids(prev) - _method_ids(merged))
            if unchanged and not dropped:
                snap.unlink(missing_ok=True)
                untouched_n += 1
                continue
            if dropped:
                snap.rename(superseded_target(snap))
                print(f"smart-merge: merged {rel}; {len(dropped)} row(s) superseded by the "
                      "pull kept in results.json.superseded")
            else:
                snap.unlink(missing_ok=True)
                print(f"smart-merge: merged {rel}")
            merged_n += 1
        except Exception as exc:  # noqa: BLE001 - report every task, then fail loudly
            failures.append(f"{rel}: {exc}")

    print(f"smart-merge: merged {merged_n} task(s), restored {restored_n} unchanged, "
          f"{untouched_n} untouched")
    if failures:
        print(f"smart-merge: FAILED for {len(failures)} task(s); their .presync snapshots "
              "were KEPT (they hold local data results.json lacks):", file=sys.stderr)
        for line in failures:
            print(f"    {line}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="directory to search for results.json(.presync) files"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--snapshot", action="store_true",
                      help="hardlink every results.json to results.json.presync (run BEFORE a pull)")
    mode.add_argument("--supersede", action="store_true",
                      help="keep the CURRENT results.json as truth: rename unmerged snapshots to "
                           "results.json.superseded (nothing is deleted)")
    mode.add_argument("--check", action="store_true",
                      help="list unmerged snapshots and exit 1 if any exist (changes nothing)")
    args = parser.parse_args()
    # Paths may hold characters the Windows console codepage cannot encode; a
    # crash while *reporting* must never hide the result.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    if not args.root.exists():
        print(f"smart-merge: {args.root} does not exist, nothing to do")
        return 0
    if args.snapshot:
        return snapshot_all(args.root)
    if args.supersede:
        return supersede_all(args.root)
    if args.check:
        bad = unresolved_snapshots(args.root)
        for snap in bad:
            print(f"UNMERGED: {snap}")
        return 1 if bad else 0
    return merge_all(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
