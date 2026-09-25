#!/usr/bin/env python
"""Print task-weighted strict headline completion for every paper model.

This reads each model's compact ``summary_merged`` CSV, never heavyweight
``results.json`` files.  It deliberately reports absent/stale model inputs
instead of silently dropping them from the status output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report_model_policy import PAPER_MODELS, summary_csv_path
from analysis.summary_latex import (
    cross_model_completion_rates,
    headline_method_groups,
    load_cross_model_summary_csvs,
)

DEFAULT_SUMMARIES = tuple(
    (model, summary_csv_path(model, subdir)) for model, subdir in PAPER_MODELS
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", action="append", default=None, metavar="MODEL=CSV",
        help="Override one or more model summary_merged CSV paths.",
    )
    args = parser.parse_args()
    summaries = list(DEFAULT_SUMMARIES)
    if args.summary:
        summaries = []
        for value in args.summary:
            model, separator, path = value.partition("=")
            if not separator or not model or not path:
                raise SystemExit(f"Invalid --summary {value!r}; expected MODEL=CSV")
            summaries.append((model, Path(path)))

    available = [(model, path) for model, path in summaries if path.is_file()]
    loaded = load_cross_model_summary_csvs(available)
    methods = headline_method_groups("selected")
    subdirs = dict(PAPER_MODELS)
    sources = tuple((model, subdirs.get(model, "")) for model, _path in available)
    rates = cross_model_completion_rates(
        ROOT / "runs", sources=sources, methods=methods, loaded_sources=loaded
    ) if sources else {}

    print("model\tstrict_completion\tstatus")
    for model, path in summaries:
        if not path.is_file():
            print(f"{model}\t—\tmissing summary CSV")
        elif loaded[0][model].empty:
            print(f"{model}\t—\tempty/stale summary CSV; regenerate its model table")
        else:
            print(f"{model}\t{rates[model]:.1%}\tready")


if __name__ == "__main__":
    main()
