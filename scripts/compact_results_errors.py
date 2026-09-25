"""Streaming repair for results.json files with explosively duplicated errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(path: Path) -> tuple[int, int]:
    tmp = path.with_suffix(path.suffix + ".compact.tmp")
    before = after = 0
    found = False
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.rstrip("\r\n") != '  "errors": [':
                dst.write(line)
                continue
            found = True
            unique: dict[str, None] = {}
            for error_line in src:
                if error_line.rstrip("\r\n") == "  ],":
                    break
                text = error_line.strip().removesuffix(",")
                if not text:
                    continue
                value = json.loads(text)
                before += 1
                unique.setdefault(str(value), None)
            dst.write('  "errors": [\n')
            values = list(unique)
            for index, value in enumerate(values):
                comma = "," if index + 1 < len(values) else ""
                dst.write(f"    {json.dumps(value)}{comma}\n")
            dst.write("  ],\n")
            after = len(values)
        dst.flush()
    if not found:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"top-level errors array not found in {path}")
    tmp.replace(path)
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=Path)
    args = parser.parse_args()
    before_size = args.results_json.stat().st_size
    before, after = compact(args.results_json)
    after_size = args.results_json.stat().st_size
    print(
        f"Compacted {args.results_json}: errors {before} -> {after}; "
        f"bytes {before_size} -> {after_size}"
    )


if __name__ == "__main__":
    main()
