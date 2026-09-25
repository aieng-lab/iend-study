"""Compare decoder-norm trajectories between shared- and decoupled-LR ACTIEND arms.

Direct test of the reachability bound in ``IEND_THEORY_PLAN.md`` 2.7. Under an
Adam-family optimizer the per-coordinate step is bounded by roughly ``lr``
regardless of gradient magnitude, so total decoder displacement obeys

    ||d_T - d_0||  <~  lr * T * sqrt(q),

*independently of how far the optimum is*. The bound therefore predicts that
raising only the decoder's learning rate lifts the ceiling proportionally, while
leaving the encoder untouched.

Reads IEND checkpoint weights directly, so it needs no GPU and no model forward
pass -- only the small ``model.safetensors`` files (~190 KB each), which
``IEND_WEIGHTS=1 bash scripts/rsync_analysis.sh`` brings down.

Example::

  python scripts/analyze_decoder_lr_arms.py \\
      --arm baseline=runs/theory/e1/gpt2-small/gender_en/actiend/source_alternative:1e-5 \\
      --arm decoupled=runs/theory/e1_decoder_lr/gpt2-small/gender_en/actiend/source_alternative:1e-3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DECODER_WEIGHT = "decoder.0.linear.weight"
ENCODER_WEIGHT = "encoder.0.linear.weight"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH:LR",
        help="Arm to compare, e.g. baseline=runs/theory/e1/...:1e-5. Repeatable.",
    )
    parser.add_argument(
        "--ridge-optimum",
        type=float,
        default=112.0,
        help=(
            "||d*|| of the exact frozen-score ridge decoder. Default is the "
            "value E0e measured for gender_en; it is a property of the data and "
            "the encoder score, not of the decoder's learning rate."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def parse_arm(spec: str) -> tuple[str, Path, float]:
    if "=" not in spec:
        raise ValueError(f"--arm must be NAME=PATH:LR, got {spec!r}")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"--arm must be NAME=PATH:LR, got {spec!r}")
    path, lr = rest.rsplit(":", 1)
    return name.strip(), Path(path.strip()), float(lr)


def collect_arm(name: str, root: Path, lr: float) -> list[dict[str, Any]]:
    """Read decoder/encoder norms from every checkpoint under ``root/seeds``."""
    from safetensors.torch import load_file

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "seeds").glob("*/model.safetensors")):
        match = re.match(r"seed_(\d+)(?:_step_(\d+))?$", path.parent.name)
        if not match:
            continue
        tensors = load_file(str(path))
        if DECODER_WEIGHT not in tensors:
            continue
        rows.append(
            {
                "arm": name,
                "learning_rate_decoder": lr,
                "seed": int(match.group(1)),
                "step": int(match.group(2)) if match.group(2) else None,
                "is_promoted": match.group(2) is None,
                "decoder_norm": float(tensors[DECODER_WEIGHT].norm()),
                "encoder_norm": float(tensors[ENCODER_WEIGHT].norm()),
                "decoder_dim": int(tensors[DECODER_WEIGHT].numel()),
            }
        )
    return rows


def summarize_arm(frame: pd.DataFrame, *, ridge_optimum: float) -> dict[str, Any]:
    """Reachability accounting for one arm at its final retained step."""
    trajectory = frame[frame["step"].notna()].sort_values("step")
    if trajectory.empty:
        return {}
    first = trajectory.iloc[0]
    last = trajectory.iloc[-1]
    root_q = float(np.sqrt(int(last["decoder_dim"])))
    lr = float(last["learning_rate_decoder"])
    steps = float(last["step"])
    budget = lr * steps * root_q
    displacement = float(last["decoder_norm"] - first["decoder_norm"])
    distance = abs(ridge_optimum - float(first["decoder_norm"]))
    return {
        "arm": str(last["arm"]),
        "learning_rate_decoder": lr,
        "final_step": int(steps),
        "decoder_norm_first": float(first["decoder_norm"]),
        "decoder_norm_final": float(last["decoder_norm"]),
        "encoder_norm_first": float(first["encoder_norm"]),
        "encoder_norm_final": float(last["encoder_norm"]),
        "observed_displacement": displacement,
        "adam_displacement_budget": budget,
        "budget_utilization": displacement / budget if budget > 0 else float("nan"),
        "reachability_R": budget / distance if distance > 0 else float("nan"),
        "fraction_of_ridge_norm_reached": float(last["decoder_norm"]) / ridge_optimum,
        "shortfall_factor": ridge_optimum / float(last["decoder_norm"]),
    }


def main() -> None:
    args = _parser().parse_args()
    rows: list[dict[str, Any]] = []
    for spec in args.arm:
        name, path, lr = parse_arm(spec)
        if not path.is_dir():
            raise FileNotFoundError(f"arm {name!r} path does not exist: {path}")
        collected = collect_arm(name, path, lr)
        if not collected:
            raise FileNotFoundError(
                f"arm {name!r} has no readable checkpoints under {path / 'seeds'}; "
                "the analysis sync excludes weights unless IEND_WEIGHTS=1"
            )
        rows.extend(collected)

    frame = pd.DataFrame(rows)
    summaries = [
        summarize_arm(frame[frame["arm"] == arm], ridge_optimum=args.ridge_optimum)
        for arm in frame["arm"].unique()
    ]
    summaries = [s for s in summaries if s]

    output_root = args.output_root or (ROOT / "runs" / "theory" / "decoder_lr_arms")
    output_root.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["arm", "seed", "step"]).to_csv(
        output_root / "decoder_norm_trajectory.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(output_root / "arm_summary.csv", index=False)
    (output_root / "results.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "decoupled decoder learning-rate confirmation (2.7)",
                "ridge_optimum_norm": args.ridge_optimum,
                "bound": "||d_T - d_0|| <~ lr * T * sqrt(q)",
                "arms": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.set_option("display.width", 220)
    print("=== promoted checkpoints ===")
    print(
        frame[frame["is_promoted"]][["arm", "seed", "decoder_norm", "encoder_norm"]]
        .sort_values(["arm", "seed"])
        .to_string(index=False)
    )
    print()
    print("=== reachability ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    print()
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
