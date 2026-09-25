"""Delete trained-model weight files once a (task, method) cell has fully finished.

Motivation: on a quota-limited cluster, GRADIEND/CGA checkpoints (100+ MB each,
several per task) are only needed while their own cell is running (train ->
encoder eval -> causal). Metadata (done.json, training.json, encoded values,
plots) is tiny and is kept, so results stay auditable.

Consequence to know about: an artifact without its weights is no longer
"train complete" (see study.stages.train._artifact_train_complete), so a later
forced re-run of the SAME cell (--refresh-causal / --force-causal) retrains it
instead of reloading. A plain --skip-existing re-run is unaffected (Gate 1 skips
the whole ok task on its stored results.json).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
MARKER_NAME = "checkpoints_deleted.json"


def should_cleanup_checkpoints(
    training: Optional[Mapping[str, Any]],
    *,
    enabled: Iterable[str],
    errors: Sequence[str],
    method_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """True only for a fully successful cell of THIS invocation.

    Requires: the opt-in flag, the causal stage to have run (a cell whose causal
    was skipped still needs its checkpoints), no stage errors, at least one
    row, and no row with status "error".
    """
    if not bool((training or {}).get("cleanup_checkpoints", False)):
        return False
    if "causal" not in set(enabled):
        return False
    if errors or not method_rows:
        return False
    return not any(str(r.get("status")) == "error" for r in method_rows)


def cleanup_finished_cell_checkpoints(
    output_dir: Any, backends: Iterable[str]
) -> Dict[str, Any]:
    """Unlink weight files under artifacts/<backend>__* for the given backends.

    Only the backends this invocation trained are touched, so concurrent
    per-method jobs on the same task directory never delete each other's
    checkpoints. ``actiend__*`` deliberately does not match ``actiend_pre__*``.
    """
    root = Path(output_dir) / "artifacts"
    removed = 0
    freed = 0
    failed = 0
    touched: Set[str] = set()
    if root.is_dir():
        for backend in sorted(set(backends)):
            for adir in sorted(root.glob(f"{backend}__*")):
                if not adir.is_dir():
                    continue
                for f in sorted(adir.rglob("*")):
                    if not f.is_file() or f.suffix not in WEIGHT_SUFFIXES:
                        continue
                    try:
                        size = f.stat().st_size
                        f.unlink()
                    except OSError as exc:
                        failed += 1
                        print(f"  cleanup: could not delete {f}: {exc}", flush=True)
                        continue
                    removed += 1
                    freed += size
                    touched.add(backend)
    summary = {
        "files": removed,
        "bytes": freed,
        "failed": failed,
        "backends": sorted(touched),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if removed:
        try:
            marker = Path(output_dir) / MARKER_NAME
            prior = []
            if marker.is_file():
                prior = json.loads(marker.read_text(encoding="utf-8"))
                if not isinstance(prior, list):
                    prior = [prior]
            marker.write_text(
                json.dumps(prior + [summary], indent=2), encoding="utf-8"
            )
        except (OSError, ValueError) as exc:
            print(f"  cleanup: could not write {MARKER_NAME}: {exc}", flush=True)
    print(
        f"checkpoint cleanup: removed {removed} weight files "
        f"({freed / 1e6:.1f} MB) for {sorted(touched) or 'no backends'}",
        flush=True,
    )
    return summary
