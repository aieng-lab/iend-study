"""Guards around the ``results.json.presync`` pre-pull snapshots.

Every sync wrapper hardlinks each local ``results.json`` to
``results.json.presync`` before a pull and merges it back afterwards
(``scripts/smart_merge_pulled_results.py``).  A pull replaces ``results.json``
wholesale with the remote's copy, so a snapshot that is still around *and* is a
different file than ``results.json`` means the local file was overwritten and
never merged: the analysis would silently read an incomplete file.

stdlib only, so the sync scripts and the analysis loaders can share it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

SNAPSHOT_SUFFIX = ".presync"
_PathLike = Union[str, "os.PathLike[str]"]


def snapshot_path(results_path: _PathLike) -> Path:
    path = Path(results_path)
    return path.with_name(path.name + SNAPSHOT_SUFFIX)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def is_unresolved(snapshot: Path) -> bool:
    """True if ``snapshot`` holds data that ``results.json`` does not.

    A snapshot that is still the *same inode* as ``results.json`` (rsync left the
    file untouched) carries nothing extra and is harmless.
    """
    dest = snapshot.with_name(snapshot.name[: -len(SNAPSHOT_SUFFIX)])
    return not _same_file(snapshot, dest)


def unresolved_snapshots(root: _PathLike) -> List[Path]:
    root = Path(root)
    if root.is_file():
        snap = snapshot_path(root)
        return [snap] if snap.exists() and is_unresolved(snap) else []
    return [
        snap
        for snap in sorted(root.rglob(f"results.json{SNAPSHOT_SUFFIX}"))
        if is_unresolved(snap)
    ]


class UnmergedSnapshotError(RuntimeError):
    """A pull overwrote ``results.json`` and its pre-pull snapshot was not merged."""


def assert_resolved(results_path: _PathLike) -> None:
    """Refuse to analyze a ``results.json`` whose pre-pull snapshot is unmerged."""
    snap = snapshot_path(results_path)
    if snap.exists() and is_unresolved(snap):
        raise UnmergedSnapshotError(
            f"{results_path} was overwritten by a sync and never merged with its "
            f"pre-pull snapshot ({snap.name}); the file is missing whatever only the "
            f"local copy had (e.g. workstation-computed methods). Repair with:\n"
            f"    python scripts/smart_merge_pulled_results.py {Path(results_path).parent}"
        )
