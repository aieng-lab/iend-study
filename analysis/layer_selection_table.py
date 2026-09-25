#!/usr/bin/env python
"""Summary table for the layer-selection appendix (all layers vs single layer).

Reads the CSVs written by ``analysis/layer_selection_appendix.py`` (no results.json
access, no torch) and emits one table with, per row:

* ``Selected``    share of validation decisions that chose a single layer
                  (``layer_selection_decisions.csv``, validation only);
* ``Det. win``    share of class/contrast comparisons where the validation-best
                  single layer has higher *test* Detection than the all-layer
                  representation;
* ``Det. mean Δ`` mean test Detection difference (single - all);
* ``Int. win``    share with higher test Intervention (only comparisons where
                  both representations have a causal score);
* ``Int. mean Δ`` mean test Intervention difference.

Ties (|Δ| <= 1e-15) count as neither win nor loss.  The CSV also carries ``n``,
``n_int`` and the loss rates.  ``Overall`` pools every method; ``--targeted-row``
adds a subtotal excluding SAE.

Examples
--------
python analysis/layer_selection_table.py                  # rows = method family
python analysis/layer_selection_table.py --by model       # rows = model
python analysis/layer_selection_table.py --by group       # rows = family x pole
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report_model_policy import PAPER_MODELS  # noqa: E402

DEFAULT_DIR = ROOT / "analysis" / "tables" / "layer_selection"
EPS = 1e-15

FAMILY_ORDER = ("sae", "caa", "cga", "caga")
FAMILY_LABEL = {"sae": r"\sae", "caa": r"\caa", "cga": r"\cga", "caga": r"\caga"}
FAMILY_PLAIN = {"sae": "SAE", "caa": "CAA", "cga": "CGA", "caga": "CAGA"}
MODEL_ORDER = tuple(model for model, _subdir in PAPER_MODELS)
MODEL_LABEL = {
    "pythia-70m-deduped": "Pythia-70M", "gpt2-small": "GPT-2", "llama-3.1-8b": "Llama-3.1-8B",
}
POLE_LABEL = {"two_pole": "pw", "one_pole": "1s", "k1": ""}


def _frac(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def summarize(decisions: pd.DataFrame, atomic: pd.DataFrame) -> Dict[str, float]:
    """One table row from the decisions and atomic-comparison rows of a slice."""
    det = pd.to_numeric(atomic["delta_detection_score"], errors="coerce").dropna()
    inter = pd.to_numeric(atomic["delta_intervention_score"], errors="coerce").dropna()
    return {
        "n": int(det.size),
        "selected_single": _frac(decisions["choice"].eq("single layer")) if len(decisions) else float("nan"),
        "det_win": _frac(det > EPS),
        "det_loss": _frac(det < -EPS),
        "det_mean": float(det.mean()) if det.size else float("nan"),
        "n_int": int(inter.size),
        "int_win": _frac(inter > EPS),
        "int_loss": _frac(inter < -EPS),
        "int_mean": float(inter.mean()) if inter.size else float("nan"),
    }


def _slices(
    by: str, decisions: pd.DataFrame, atomic: pd.DataFrame
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Ordered ``(key, decisions, atomic)`` slices for the requested row grouping."""
    out: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []
    if by == "family":
        for fam in FAMILY_ORDER:
            out.append((fam, decisions[decisions.backend == fam], atomic[atomic.backend == fam]))
    elif by == "model":
        present = set(atomic.model)
        for model in (*MODEL_ORDER, *sorted(present - set(MODEL_ORDER))):
            if model in present:
                out.append((model, decisions[decisions.model == model], atomic[atomic.model == model]))
    elif by == "group":
        for fam in FAMILY_ORDER:
            for pole in ("two_pole", "one_pole", "k1"):
                d = decisions[(decisions.backend == fam) & (decisions.pole == pole)]
                a = atomic[atomic.method_group == f"{fam}:{pole}"]
                if len(a):
                    out.append((f"{fam}:{pole}", d, a))
    else:
        raise ValueError(f"unknown --by {by!r}")
    out.append(("targeted", decisions[decisions.backend != "sae"], atomic[atomic.backend != "sae"]))
    out.append(("all", decisions, atomic))
    return out


def _label(key: str, latex: bool) -> str:
    if key == "targeted":
        return "Targeted (excl.\\ SAE)" if latex else "Targeted (excl. SAE)"
    if key == "all":
        return "Overall"
    if key in MODEL_LABEL:
        return MODEL_LABEL[key]
    fam, _, pole = key.partition(":")
    base = (FAMILY_LABEL if latex else FAMILY_PLAIN).get(fam, fam)
    suffix = POLE_LABEL.get(pole, "")
    if not suffix:
        return base
    return f"{base}$_{{\\mathrm{{{suffix}}}}}$" if latex else f"{base} ({suffix})"


def build_table(decisions: pd.DataFrame, atomic: pd.DataFrame, by: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"key": key, **summarize(d, a)} for key, d, a in _slices(by, decisions, atomic)
    ])


def _pct(v: float) -> str:
    return "--" if pd.isna(v) else f"{100 * v:.1f}\\%"


def _delta(v: float) -> str:
    if pd.isna(v):
        return "--"
    text = f"{v:+.3f}"
    if float(text) == 0.0:  # avoid a misleading "-0.000" / "+0.000"
        text = "0.000"
    return f"${text}$"


DEFAULT_CAPTION = (
    r"\textbf{Effect of representation scope.} "
    "Single-layer representations are selected by validation Detection. "
    r"$\Delta$ denotes single-layer minus all-layer performance on the held-out "
    "test split. Win rates report the fraction of class/contrast comparisons "
    r"with $\Delta>0$."
)

_HEADER = r"""\begin{table}[t]
    \centering
    \small
    \caption{%(caption)s}
    \label{%(label)s}
    \begin{tabular}{lrrrrr}
        \toprule
        & \multicolumn{1}{c}{Selected}
        & \multicolumn{2}{c}{Detection}
        & \multicolumn{2}{c}{Intervention} \\
        \cmidrule(lr){2-2}
        \cmidrule(lr){3-4}
        \cmidrule(lr){5-6}
        %(first)s
        & Single layer
        & Win
        & Mean $\Delta$
        & Win
        & Mean $\Delta$ \\
        \midrule
"""

_FOOTER = r"""        \bottomrule
    \end{tabular}
\end{table}
"""


def to_latex(table: pd.DataFrame, by: str, *, label: str, caption: str, targeted_row: bool) -> str:
    first = {"family": "Method", "model": "Model", "group": "Method"}[by]
    out = [_HEADER % {"caption": caption, "label": label, "first": first}]
    for _, row in table.iterrows():
        key = row["key"]
        if key == "targeted" and not targeted_row:
            continue
        if key in ("targeted", "all") and not (key == "all" and targeted_row):
            out.append("        \\midrule\n")
        out.append(
            f"        {_label(key, True)} & {_pct(row['selected_single'])} & "
            f"{_pct(row['det_win'])} & {_delta(row['det_mean'])} & "
            f"{_pct(row['int_win'])} & {_delta(row['int_mean'])} \\\\\n"
        )
    out.append(_FOOTER)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="directory written by layer_selection_appendix.py")
    ap.add_argument("--by", choices=("family", "model", "group"), default="family")
    ap.add_argument("--out", type=Path, default=None,
                    help="output stem (default: <dir>/layer_selection_table_<by>)")
    ap.add_argument("--label", default="tab:layer-selection-summary")
    ap.add_argument("--caption", default=DEFAULT_CAPTION)
    ap.add_argument("--targeted-row", action="store_true",
                    help="add a 'Targeted (excl. SAE)' subtotal row before Overall")
    args = ap.parse_args()

    dec_path = args.dir / "layer_selection_decisions.csv"
    atomic_path = args.dir / "layer_selection_atomic_heldout_detection.csv"
    for path in (dec_path, atomic_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}; run analysis/layer_selection_appendix.py first")
    decisions = pd.read_csv(dec_path)
    atomic = pd.read_csv(atomic_path)
    if "delta_intervention_score" not in atomic:
        raise SystemExit("atomic CSV has no delta_intervention_score; rerun layer_selection_appendix.py")
    # A comparison needs both representations' Detection; Intervention is optional.
    atomic = atomic[pd.to_numeric(atomic["delta_detection_score"], errors="coerce").notna()]

    table = build_table(decisions, atomic, args.by)
    caption = args.caption
    if caption == DEFAULT_CAPTION:
        # Intervention needs a causal score for BOTH representations; say how many
        # comparisons that leaves instead of implying full coverage.
        total = table.loc[table["key"] == "all"].iloc[0]
        caption += (
            f" Intervention is computed on the {int(total['n_int'])} of {int(total['n'])} "
            "comparisons for which both representations have a causal score."
        )
    stem = args.out or (args.dir / f"layer_selection_table_{args.by}")
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(
        to_latex(table, args.by, label=args.label, caption=caption,
                 targeted_row=args.targeted_row),
        encoding="utf-8",
    )

    show = table.copy()
    show["key"] = show["key"].map(lambda k: _label(k, False))
    pd.set_option("display.width", 200)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {stem.with_suffix('.csv')}")
    print(f"wrote {stem.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
