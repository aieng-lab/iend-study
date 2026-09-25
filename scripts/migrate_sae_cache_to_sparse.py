"""Migrate legacy dense sae_cache/*/*.npy shards to the sparse .npz format.

Before 2026-09-16, save_sae_cache() stored SAE latent matrices as dense
float32 .npy -- multi-GB per cache dir since SAE activations are ReLU (mostly
zero). sae_eval.load_sae_cache() reads both formats, so nothing breaks by
leaving old caches alone; they just stay large.

This converts existing dense shards in place with zero recompute (numpy only,
no torch/GPU): mmap dense -> chunked nonzero-encode -> atomic .npz -> delete
.npy. Idempotent.

Speed: dense files are memory-mapped and encoded in ~256MB chunks (bounded RAM),
cache dirs are processed in parallel (--workers), directory discovery prunes
artifacts/checkpoint trees, and --dry-run never reads a full file: it stat()s
sizes and estimates the sparse size from an evenly spaced row sample.

Cache dirs touched within --min-age-hours (default 24) are skipped untouched --
a live job may still be writing them. Truncated/unreadable shards (interrupted
earlier writes) are reported and deleted; they were already cache misses.

Usage:
    python scripts/migrate_sae_cache_to_sparse.py runs/ [--dry-run] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

_LATENT_NAMES = (
    "val_latents",
    "test_latents",
    "neutral_latents",
    "test_neutral_latents",
)
# Trees that never contain a sae_cache (model checkpoints etc.) -- not descended into.
_PRUNE_DIRS = {"artifacts", "checkpoints", "model", "seeds", "__pycache__"}
_CHUNK_BYTES = 256 * 1024 * 1024
_SAMPLE_ROWS = 256


def _iter_scan_units(root: Path) -> Iterator[Path]:
    children = sorted(p for p in root.iterdir() if p.is_dir())
    if children:
        yield from children
    else:
        yield root


def _find_cache_dirs(unit: Path) -> list[Path]:
    """All sae_cache/<L*>/ dirs under unit, without descending into checkpoint trees."""
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(unit):
        if os.path.basename(dirpath) == "sae_cache":
            found.extend(Path(dirpath) / d for d in dirnames)
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames if d not in _PRUNE_DIRS and not d.startswith("checkpoint")
        ]
    return sorted(found)


def _dir_recently_touched(root: Path, *, min_age_seconds: float, now: float) -> bool:
    candidates = list(root.glob("*.npy")) + list(root.glob("*.npz"))
    meta = root / "meta.json"
    if meta.is_file():
        candidates.append(meta)
    for p in candidates:
        try:
            if now - p.stat().st_mtime < min_age_seconds:
                return True
        except OSError:
            continue
    return False


def _write_sparse_from_dense(dense_path: Path, sparse_path: Path) -> None:
    """Chunked, memory-bounded dense .npy -> sparse .npz (format read by sae_eval)."""
    mm = np.load(dense_path, mmap_mode="r")
    tmp = sparse_path.with_name(sparse_path.stem + ".tmp.npz")
    if mm.ndim != 2:
        np.savez_compressed(tmp, dense=np.asarray(mm, dtype=np.float32))
        os.replace(tmp, sparse_path)
        return
    n_rows, n_cols = mm.shape
    step = max(1, _CHUNK_BYTES // max(1, n_cols * mm.dtype.itemsize))
    data_parts, idx_parts, count_parts = [], [], []
    for i in range(0, n_rows, step):
        chunk = np.asarray(mm[i : i + step], dtype=np.float32)
        r, c = np.nonzero(chunk)
        data_parts.append(chunk[r, c])
        idx_parts.append(c.astype(np.int32))
        count_parts.append(np.bincount(r, minlength=chunk.shape[0]))
    counts = np.concatenate(count_parts) if count_parts else np.zeros(0, dtype=np.int64)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    np.savez_compressed(
        tmp,
        data=np.concatenate(data_parts) if data_parts else np.zeros(0, dtype=np.float32),
        indices=np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int32),
        indptr=indptr,
        shape=np.asarray([n_rows, n_cols], dtype=np.int64),
    )
    os.replace(tmp, sparse_path)  # never leave a half-written .npz that looks valid


def _estimate_sparse_bytes(mm: np.ndarray) -> int:
    """Sparse size estimate from an evenly spaced row sample (reads ~256 rows only)."""
    if mm.ndim != 2 or mm.shape[0] == 0:
        return int(mm.size * 4)
    n_rows = mm.shape[0]
    stride = max(1, n_rows // _SAMPLE_ROWS)
    sample = np.asarray(mm[::stride][:_SAMPLE_ROWS])
    nnz = int(np.count_nonzero(sample)) / sample.shape[0] * n_rows
    return int(nnz * 8 + n_rows * 8 + 64)


def _migrate_one_dir(root: Path, dry_run: bool) -> tuple[Path, int, int, list[Path]]:
    """(dir, bytes_before, bytes_after, corrupt_paths) for one sae_cache/L*/ dir."""
    before = after = 0
    corrupt: list[Path] = []
    for name in _LATENT_NAMES:
        dense_path = root / f"{name}.npy"
        sparse_path = root / f"{name}.npz"
        if not dense_path.is_file():
            continue
        size_before = dense_path.stat().st_size
        if sparse_path.is_file():
            # Interrupted earlier migration: keep the .npz, drop the stale dense copy.
            before += size_before
            after += sparse_path.stat().st_size
            if not dry_run:
                dense_path.unlink()
            continue
        try:
            mm = np.load(dense_path, mmap_mode="r")  # truncated file -> raises here, O(1)
        except Exception as exc:
            print(f"  WARNING: unreadable {dense_path}: {exc}")
            corrupt.append(dense_path)
            if not dry_run:
                dense_path.unlink(missing_ok=True)
            continue
        # Only a failed READ counts as corrupt. A failed WRITE (disk full, quota,
        # permissions) must propagate and leave the dense file untouched.
        if dry_run:
            est = _estimate_sparse_bytes(mm)
            del mm
        else:
            del mm
            _write_sparse_from_dense(dense_path, sparse_path)
            est = sparse_path.stat().st_size
            dense_path.unlink()  # only reached after the .npz was written and renamed
        before += size_before
        after += est
    return root, before, after, corrupt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="Runs subtree(s) to scan, e.g. runs/")
    ap.add_argument("--dry-run", action="store_true", help="Estimate sizes only, write nothing.")
    ap.add_argument("--min-age-hours", type=float, default=24.0,
                    help="Skip cache dirs touched more recently than this (default 24; 0 disables).")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="Parallel cache dirs (default min(4, cpus)).")
    args = ap.parse_args()
    min_age_seconds = max(0.0, args.min_age_hours) * 3600.0
    now = time.time()

    total_before = total_after = 0
    n_dirs = n_skipped_recent = 0
    all_corrupt: list[Path] = []

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for root in args.roots:
            if not root.is_dir():
                print(f"skip (not a dir): {root}")
                continue
            for unit in _iter_scan_units(root):
                todo: list[Path] = []
                for cache_dir in _find_cache_dirs(unit):
                    if not any((cache_dir / f"{n}.npy").is_file() for n in _LATENT_NAMES):
                        continue
                    if min_age_seconds > 0 and _dir_recently_touched(
                        cache_dir, min_age_seconds=min_age_seconds, now=now
                    ):
                        n_skipped_recent += 1
                        print(f"skip (touched within {args.min_age_hours:g}h): {cache_dir}")
                        continue
                    todo.append(cache_dir)
                print(f"--- {unit}: {len(todo)} cache dir(s) to process ---")
                futs = [pool.submit(_migrate_one_dir, d, args.dry_run) for d in todo]
                for fut in as_completed(futs):
                    d, b, a, corrupt = fut.result()  # fail fast on a real error
                    all_corrupt.extend(corrupt)
                    if b == 0:
                        continue
                    n_dirs += 1
                    total_before += b
                    total_after += a
                    tag = "(est.)" if args.dry_run else ""
                    print(f"{d}: {b/1e6:8.1f} MB -> {a/1e6:8.1f} MB ({b/max(a,1):.1f}x) {tag}")

    print()
    print(f"{n_dirs} cache dir(s) processed")
    if n_skipped_recent:
        print(f"{n_skipped_recent} skipped as recently modified (< {args.min_age_hours:g}h)")
    print(f"total: {total_before/1e9:.2f} GB -> {total_after/1e9:.2f} GB")
    if all_corrupt:
        verb = "would be deleted" if args.dry_run else "deleted"
        print(f"{len(all_corrupt)} unreadable/truncated shard(s) ({verb}; already cache misses):")
        for p in all_corrupt:
            print(f"  {p}")
    if args.dry_run:
        print("(dry run, nothing written -- rerun without --dry-run to apply)")


if __name__ == "__main__":
    main()
