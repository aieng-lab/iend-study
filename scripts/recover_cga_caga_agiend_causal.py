#!/usr/bin/env python
"""Recover cga/caga/agiend causal rows dropped by the METHOD_FAMILIES merge bug.

``study/results_merge.py::METHOD_FAMILIES`` was missing ``"cga"``/``"caga"``/
``"agiend"``, so ``_merge_causal_raw`` computed an empty backend set for any
``--methods cga``/``caga``/``agiend`` run and silently discarded the freshly
computed causal sweep instead of merging it into ``results.json``. The sweep
itself completed correctly on GPU and was checkpointed to
``<task>/causal/progress-<jobid>.json`` *before* that merge step ran, so the
corrected results already exist on disk -- no GPU rerun needed, only a
CPU-only re-merge using the now-fixed merge machinery.

This reads every ``causal/progress-*.json`` under each task directory, keeps
only the affected-family entries that are complete (LMS-gated sweep present,
not an error stub), and attaches them onto ``results.json`` via the exact
same converter (``causal_eval.causal_method_result_from_dict`` +
``study.stages.causal._attach_causal_to_rows``) the live pipeline uses --
byte-identical to what a successful merge would have produced.

Idempotent: rerunning re-attaches the same entries and changes nothing.

    python scripts/recover_cga_caga_agiend_causal.py --root runs/pythia-70m-deduped/suite_full2
    python scripts/recover_cga_caga_agiend_causal.py --root runs/pythia-70m-deduped/suite_full2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AFFECTED_PREFIXES = ("cga:", "caga:", "agiend:", "cga_tensor_norm:")
_FAMILY_PREFIXES = {
    "cga": ("cga:", "cga_tensor_norm:"),
    "caga": ("caga:",),
    "agiend": ("agiend:",),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory holding <task>/results.json (e.g. runs/<model>/<subdir>).",
    )
    parser.add_argument(
        "--families",
        default="cga,caga,agiend",
        help=(
            "Comma list of families to recover. Use 'caga,agiend' on Llama: its CGA "
            "progress entries were measured on a bf16-drifted base model (see CLAUDE.md, "
            "2026-09-18) and must NOT be re-attached."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be recovered without writing results.json.",
    )
    return parser


def _load_progress_by_method(
    task_dir: Path, prefixes: Tuple[str, ...] = AFFECTED_PREFIXES
) -> Dict[str, Dict[str, Any]]:
    from study.results_merge import causal_raw_entry_ok

    merged: Dict[str, Dict[str, Any]] = {}
    causal_dir = task_dir / "causal"
    if not causal_dir.is_dir():
        return merged
    # A task's causal/ directory accumulates one progress-<jobid>.json per
    # job that ever ran there, going back to before DIRECTION_POLARITY_
    # PROTOCOL_VERSION=2 existed -- an old, structurally-complete-but-
    # pre-sign-fix checkpoint is exactly as "causal_raw_entry_ok" as a fresh
    # one. Iterate oldest-to-newest by mtime and let later files win, so the
    # most recently computed entry for a given method id is always kept.
    # A task's causal/ directory accumulates one progress-<jobid>.json per
    # job that ever ran there, going back to before DIRECTION_POLARITY_
    # PROTOCOL_VERSION=2 existed -- an old, structurally-complete-but-
    # pre-sign-fix checkpoint is exactly as "causal_raw_entry_ok" as a fresh
    # one. Iterate oldest-to-newest by mtime and let later files win, so the
    # most recently computed entry for a given method id is always kept.
    paths = sorted(causal_dir.glob("progress-*.json"), key=lambda p: p.stat().st_mtime)
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # A file that records a protocol version older than the current one was
        # written before the corresponding fix; never bless it (files that record
        # no version are accepted, as before).
        from study.stages.causal import (
            AGIEND_CAUSAL_GRID_PROTOCOL_VERSION,
            CGA_CAUSAL_GRID_PROTOCOL_VERSION,
            DIRECTION_POLARITY_PROTOCOL_VERSION,
        )

        def _stale(key: str, current: int) -> bool:
            value = payload.get(key)
            return value is not None and int(value) < int(current)

        stale_polarity = _stale(
            "direction_polarity_protocol_version", DIRECTION_POLARITY_PROTOCOL_VERSION
        )
        stale_cga = _stale("cga_causal_grid_protocol_version", CGA_CAUSAL_GRID_PROTOCOL_VERSION)
        stale_agiend = _stale(
            "agiend_causal_grid_protocol_version", AGIEND_CAUSAL_GRID_PROTOCOL_VERSION
        )
        for mid, entry in (payload.get("by_method") or {}).items():
            mid = str(mid)
            if not mid.startswith(prefixes):
                continue
            if stale_polarity and mid.startswith(("cga:", "caga:", "agiend:", "cga_tensor_norm:")):
                continue
            if stale_cga and mid.startswith(("cga:", "cga_tensor_norm:")):
                continue
            if stale_agiend and mid.startswith("agiend:"):
                continue
            if not isinstance(entry, dict) or not causal_raw_entry_ok(entry):
                continue
            merged[mid] = entry
    return merged


def recover_task(
    task_dir: Path, *, dry_run: bool, prefixes: Tuple[str, ...] = AFFECTED_PREFIXES
) -> Tuple[int, int]:
    """Attach recovered entries in place. Returns (n_by_method_added, n_rows_touched)."""
    from causal_eval import causal_method_result_from_dict
    from study.stages.causal import (
        AGIEND_CAUSAL_GRID_METHOD_PREFIXES,
        AGIEND_CAUSAL_GRID_PROTOCOL_VERSION,
        CGA_CAUSAL_GRID_METHOD_PREFIXES,
        CGA_CAUSAL_GRID_PROTOCOL_VERSION,
        DIRECTION_POLARITY_PROTOCOL_VERSION,
        _attach_causal_to_rows,
        _uses_direction_polarity_protocol,
    )

    results_path = task_dir / "results.json"
    if not results_path.is_file():
        return (0, 0)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    recovered = _load_progress_by_method(task_dir, prefixes)
    if not recovered:
        return (0, 0)

    # Mirror study/stages/causal.py's own post-run provenance stamping: the
    # progress-*.json checkpoint is written *before* that stamping happens,
    # so a recovered entry otherwise looks pre-fix (meta lacks the version)
    # to the next --skip-existing run's staleness check, even though the
    # value itself was already computed under the fixed protocol.
    for mid, entry in recovered.items():
        entry_meta = dict(entry.get("meta") or {})
        if _uses_direction_polarity_protocol(mid):
            entry_meta["direction_polarity_protocol_version"] = DIRECTION_POLARITY_PROTOCOL_VERSION
        if mid.startswith(CGA_CAUSAL_GRID_METHOD_PREFIXES):
            entry_meta["cga_causal_grid_protocol_version"] = CGA_CAUSAL_GRID_PROTOCOL_VERSION
        if mid.startswith(AGIEND_CAUSAL_GRID_METHOD_PREFIXES):
            entry_meta["agiend_causal_grid_protocol_version"] = AGIEND_CAUSAL_GRID_PROTOCOL_VERSION
        entry["meta"] = entry_meta

    raw = payload.setdefault("raw", {})
    causal_raw = raw.setdefault("causal", {})
    by_method = causal_raw.setdefault("by_method", {})
    added = sum(1 for mid in recovered if mid not in by_method)
    by_method.update(recovered)

    objects = {
        mid: causal_method_result_from_dict(entry) for mid, entry in recovered.items()
    }
    methods = payload.get("methods") or []
    before_ids = {str(r.get("method")) for r in methods}
    _attach_causal_to_rows(
        methods,
        {"by_method": objects},
        model_key=str(payload.get("model") or ""),
        task_id=str(payload.get("task") or ""),
    )
    payload["methods"] = methods
    touched = len({mid for mid in objects} - before_ids) + len(
        {mid for mid in objects} & before_ids
    )

    if not dry_run:
        results_path.write_text(
            json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
        )
    return added, touched


def main() -> int:
    args = _parser().parse_args()
    unknown = [f for f in args.families.split(",") if f.strip() not in _FAMILY_PREFIXES]
    if unknown:
        print(f"Unknown --families {unknown}; choose from {sorted(_FAMILY_PREFIXES)}", file=sys.stderr)
        return 2
    prefixes = tuple(
        p for f in args.families.split(",") for p in _FAMILY_PREFIXES[f.strip()]
    )
    task_dirs = sorted(
        p.parent for p in Path(args.root).glob("*/results.json") if p.is_file()
    )
    if not task_dirs:
        print(f"No results.json under {args.root}", file=sys.stderr)
        return 1

    grand_added = 0
    grand_touched = 0
    for task_dir in task_dirs:
        added, touched = recover_task(task_dir, dry_run=args.dry_run, prefixes=prefixes)
        if added or touched:
            print(
                f"{task_dir}: by_method +{added}, rows touched {touched}"
                f"{'  (dry run)' if args.dry_run else ''}"
            )
        grand_added += added
        grand_touched += touched

    print(
        f"\nTotal: {grand_added} by_method entries recovered, {grand_touched} rows touched"
        f"{'  (dry run, nothing written)' if args.dry_run else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
