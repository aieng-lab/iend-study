#!/usr/bin/env python
"""Report frozen Hub emotion rows satisfying a left-context threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _left_words(masked: str) -> int:
    return len(str(masked).split("[MASK]", 1)[0].split())


def main() -> None:
    from study.tasks import load_hf_splits
    from study.tasks.emotion import DEFAULT_CONFIG, DEFAULT_DATASET

    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=int, required=True)
    args = parser.parse_args()

    df = load_hf_splits(DEFAULT_DATASET, config_name=DEFAULT_CONFIG)
    selected = df[df["masked"].map(_left_words) >= args.left]
    counts = selected["label_class"].value_counts().to_dict()
    print(f"RESULT left={args.left} counts={counts} total={len(selected)}")


if __name__ == "__main__":
    main()
