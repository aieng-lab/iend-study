"""Single source of truth for report model populations and their run trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
POLICY_PATH: Final = ROOT / "configs" / "report_model_sets.json"


def load_model_set(name: str) -> tuple[tuple[str, str], ...]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    try:
        records = payload[name]
    except KeyError as exc:
        raise ValueError(f"unknown report model set {name!r}") from exc
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in records:
        model = str(record["model"]).strip()
        subdir = str(record.get("subdir") or "").strip()
        if not model or model in seen:
            raise ValueError(f"invalid or duplicate model in {name!r}: {record!r}")
        if subdir and (Path(subdir).is_absolute() or ".." in Path(subdir).parts):
            raise ValueError(f"invalid subdir for {model!r}: {subdir!r}")
        seen.add(model)
        out.append((model, subdir))
    return tuple(out)


PAPER_MODELS: Final = load_model_set("paper_models")
ABLATION_CANDIDATE_MODELS: Final = load_model_set("ablation_candidate_models")


def subdir_for(model: str) -> str:
    """Run-tree subdir the policy assigns to ``model`` ("" = directly under runs/<model>).

    Models outside both sets (one-off ``MODELS=...`` overrides) have no policy
    entry and read directly from ``runs/<model>``.
    """
    for entries in (PAPER_MODELS, ABLATION_CANDIDATE_MODELS):
        for name, subdir in entries:
            if name == model:
                return subdir
    return ""


def summary_csv_path(model: str, subdir: str) -> Path:
    """Path of the generated ``summary_merged`` CSV for one (model, subdir) source."""
    suffix = f"_{subdir}" if subdir else ""
    return ROOT / "analysis" / "tables" / f"latex_{model}{suffix}" / f"summary_merged_{model}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("paper_models", "ablation_candidate_models"))
    parser.add_argument("--format", choices=("csv", "sources"), default="csv")
    parser.add_argument("--subdir-of", metavar="MODEL", help="print the policy subdir for MODEL and exit")
    args = parser.parse_args()
    if args.subdir_of:
        print(subdir_for(args.subdir_of))
        return
    if not args.set:
        parser.error("--set is required unless --subdir-of is given")
    entries = load_model_set(args.set)
    if args.format == "csv":
        print(",".join(model for model, _subdir in entries))
    else:
        print("\n".join(f"{model}={subdir}" if subdir else model for model, subdir in entries))


if __name__ == "__main__":
    main()
