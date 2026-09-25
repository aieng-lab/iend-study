#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from study.data.synthetic import (
    build_function_composition,
    build_induction,
    build_key_value,
    build_language,
    generate_all,
)


def main() -> None:
    tiny = {"train": 3, "validation": 0, "test": 0}
    for diff in ("easy", "medium", "hard"):
        kv = build_key_value(rows_per_split=tiny, seed=0, difficulty=diff)
        print(f"KV[{diff}]", kv.iloc[0]["masked"])
        print("   label", kv.iloc[0]["label"], "alt", kv.iloc[0]["alternative"],
              "n_bindings", kv.iloc[0]["n_bindings"])
        ind = build_induction(rows_per_split=tiny, seed=0, difficulty=diff)
        print(f"IND[{diff}]", ind.iloc[0]["masked"])
        print("   label", ind.iloc[0]["label"], "unit_len", ind.iloc[0]["unit_len"])
        fc = build_function_composition(rows_per_split=tiny, seed=0, difficulty=diff)
        print(f"FC[{diff}]", fc.iloc[0]["masked"])
        print("   label", fc.iloc[0]["label"], "chain_depth", fc.iloc[0]["chain_depth"])
    lang = build_language(
        rows_per_split={"train": 4, "validation": 0, "test": 0}, seed=0, use_fixture=True
    )
    print("LANG0", lang.iloc[0]["masked"])
    print("generating full...")
    paths = generate_all(force=True)
    for k, p in paths.items():
        df = pd.read_csv(p)
        print(k, len(df), dict(df["label_class"].value_counts()))

if __name__ == "__main__":
    main()
