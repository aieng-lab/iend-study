#!/usr/bin/env python
"""Regenerate MIB-derived study datasets under data/mib/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.data.mib import generate_all_mib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ravel-top-k", type=int, default=3)
    p.add_argument(
        "--ravel-max-per-class-per-split",
        type=int,
        default=None,
        help="Optional cap (e.g. 3000) to keep CSVs smaller",
    )
    args = p.parse_args()
    paths = generate_all_mib(
        force=bool(args.force),
        seed=int(args.seed),
        ravel_top_k=int(args.ravel_top_k),
        ravel_max_per_class_per_split=args.ravel_max_per_class_per_split,
    )
    for k, path in paths.items():
        print(f"{k}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
