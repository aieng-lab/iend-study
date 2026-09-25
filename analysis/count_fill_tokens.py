#!/usr/bin/env python
"""Count how many prediction-fill targets are multi-token under a model tokenizer.

Uses the same fill tokenizer as training/encode (``_target_token_ids_for_fill``),
not a naive ``tokenizer.encode(label)``.

  python analysis/count_fill_tokens.py --model gpt2-small
  python analysis/count_fill_tokens.py --model gpt2-small --tasks emotion,gender_en
  python analysis/count_fill_tokens.py --model gpt2-small --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.config import load_study_config
from study.registry import TASK_BUILDERS, build_task
from study.tasks import labeled_df_for_eval


def _target_ids(tokenizer: Any, text: str) -> List[int]:
    try:
        from gradiend.trainer.text.prediction.dataset import _target_token_ids_for_fill

        return list(_target_token_ids_for_fill(tokenizer, str(text), prefix=""))
    except Exception:
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        return [int(x) for x in ids]


def _count_frame(df, tokenizer: Any, *, column: str = "label") -> Dict[str, Any]:
    if df is None or getattr(df, "empty", True) or column not in df.columns:
        return {
            "n_rows": 0,
            "n_multi_rows": 0,
            "pct_multi_rows": None,
            "n_unique": 0,
            "n_multi_unique": 0,
            "pct_multi_unique": None,
        }
    texts = [str(x) for x in df[column].tolist() if str(x).strip()]
    n_rows = len(texts)
    n_multi_rows = 0
    unique = {}
    for t in texts:
        n = unique.get(t)
        if n is None:
            n = len(_target_ids(tokenizer, t))
            unique[t] = n
        if n > 1:
            n_multi_rows += 1
    n_unique = len(unique)
    n_multi_unique = sum(1 for n in unique.values() if n > 1)
    return {
        "n_rows": n_rows,
        "n_multi_rows": n_multi_rows,
        "pct_multi_rows": (100.0 * n_multi_rows / n_rows) if n_rows else None,
        "n_unique": n_unique,
        "n_multi_unique": n_multi_unique,
        "pct_multi_unique": (100.0 * n_multi_unique / n_unique) if n_unique else None,
    }


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%"


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = (
        "task",
        "n_rows",
        "multi_rows",
        "%rows",
        "n_unique",
        "multi_unique",
        "%unique",
    )
    lines = [
        "Prediction-fill multi-token counts (GPT-2/BPE fill ids, same as encode)",
        f"{headers[0]:<22} {headers[1]:>8} {headers[2]:>10} {headers[3]:>7} "
        f"{headers[4]:>8} {headers[5]:>12} {headers[6]:>8}",
        "-" * 82,
    ]
    for row in rows:
        lines.append(
            f"{row['task']:<22} {row['n_rows']:>8} {row['n_multi_rows']:>10} "
            f"{_fmt_pct(row['pct_multi_rows']):>7} {row['n_unique']:>8} "
            f"{row['n_multi_unique']:>12} {_fmt_pct(row['pct_multi_unique']):>8}"
        )
    return "\n".join(lines)


def iter_task_ids(tasks: Optional[Sequence[str]]) -> List[str]:
    if tasks:
        return [str(t).strip() for t in tasks if str(t).strip()]
    return sorted(TASK_BUILDERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2-small", help="Study model key (tokenizer)")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids (default: all registered builders)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke-sized task builds (faster; not the paper split)",
    )
    parser.add_argument("--suite", default=None, help="Suite id for config merge")
    args = parser.parse_args()
    task_ids = iter_task_ids(None if args.tasks is None else args.tasks.split(","))

    from transformers import AutoTokenizer

    cfg0 = load_study_config(model=args.model, task=task_ids[0], suite=args.suite)
    tok = AutoTokenizer.from_pretrained(cfg0.hf_model)
    print(f"tokenizer={cfg0.hf_model}  tasks={len(task_ids)}  smoke={bool(args.smoke)}", flush=True)

    rows: List[Dict[str, Any]] = []
    for tid in task_ids:
        try:
            cfg = load_study_config(model=args.model, task=tid, suite=args.suite)
            bundle = build_task(tid, cfg.raw, smoke=bool(args.smoke))
            df = labeled_df_for_eval(bundle, expand_one_pole=True)
            stats = _count_frame(df, tok)
            stats["task"] = tid
            rows.append(stats)
            print(
                f"  {tid}: rows={stats['n_rows']} multi_rows={stats['n_multi_rows']} "
                f"({_fmt_pct(stats['pct_multi_rows'])})  unique={stats['n_unique']} "
                f"multi_unique={stats['n_multi_unique']} ({_fmt_pct(stats['pct_multi_unique'])})",
                flush=True,
            )
        except Exception as exc:
            print(f"  {tid}: ERROR {exc}", flush=True)
            rows.append(
                {
                    "task": tid,
                    "n_rows": 0,
                    "n_multi_rows": 0,
                    "pct_multi_rows": None,
                    "n_unique": 0,
                    "n_multi_unique": 0,
                    "pct_multi_unique": None,
                }
            )
    print()
    print(format_table(rows))


if __name__ == "__main__":
    main()
