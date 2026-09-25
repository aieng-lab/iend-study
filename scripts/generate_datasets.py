#!/usr/bin/env python
"""(Re)generate the locally built task datasets under ``data/``.

Nine of the 15 study tasks need no local generation: ``gender_en``, ``emotion``,
``race``, ``religion``, ``pronoun_number`` and ``pronoun_person`` are loaded from
the Hugging Face datasets named in ``configs/tasks/*.yaml`` on first use. The other six task
families are built by deterministic generators and cached as CSV files that are
also committed to this repository, so a plain checkout already contains them.

  synthetic  key_value, induction, function_composition (easy = study setting),
             repetition                       -> data/synthetic/  (offline)
  language   language-ID cloze from OPUS-100 + MUSE lexica
                                              -> data/synthetic/  (downloads)
  mib        ioi_mib (mib-bench/ioi) and the three RAVEL tasks (hij/ravel)
                                              -> data/mib/        (downloads)

Without ``--force`` an up-to-date cache is left untouched, so the command is safe
to run on a fresh checkout. ``--force`` rebuilds from scratch.

    python scripts/generate_datasets.py                    # everything, keep valid caches
    python scripts/generate_datasets.py --only synthetic --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GROUPS = ("synthetic", "language", "mib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--force", action="store_true", help="rebuild even if a valid cache exists")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    written = {}
    if "synthetic" in args.only:
        from study.data.synthetic import DIFFICULTY_TASKS, ensure_synthetic

        for task in DIFFICULTY_TASKS:
            written[task] = ensure_synthetic(task, difficulty="easy", force=args.force, seed=args.seed)
        written["repetition"] = ensure_synthetic("repetition", force=args.force, seed=args.seed)
    if "language" in args.only:
        from study.data.synthetic import ensure_synthetic

        written["language"] = ensure_synthetic("language", force=args.force, seed=args.seed)
    if "mib" in args.only:
        from study.data.mib import generate_all_mib

        written.update(generate_all_mib(force=args.force, seed=args.seed))
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
