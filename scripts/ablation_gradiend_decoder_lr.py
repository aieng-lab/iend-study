#!/usr/bin/env python
"""Ablation: does GRADIEND's decoder learning rate matter — and how much did the
silent package-default flip to ``learning_rate_decoder="auto"`` change GRADIEND
results across tasks?

THE PROBLEM this exists to quantify. ``gradiend.trainer.core.arguments.TrainingArguments``
defaults ``learning_rate_decoder="auto"`` (arguments.py:119) — an auto-adapting,
DECOUPLED decoder LR that, at the first ``eval_steps`` boundary, raises the decoder
rate to the model-local reachability floor (see IEND_THEORY_PLAN.md §2.7). The study
never sets ``learning_rate_decoder_gradiend`` (``configs/defaults.yaml`` leaves it
commented), so GRADIEND INHERITS the package default. ``configs/defaults.yaml``'s own
comment claims "unset = encoder and decoder share the rate ... what every existing run
used" — but that is only true when the package default is ``None``. Once the package
default became ``"auto"``, every GRADIEND run picked up a decoupled decoder LR, while
runs from before the flip did not — a silent, ``config_hash``-invisible inconsistency
(the same class as the GRADIEND scope drift documented in CLAUDE.md). The decoupled
decoder LR was the subject of an ACTIEND rerun and theory result (§2.7, R≈0.0043 on
gpt2-small ACTIEND), but GRADIEND's decoder reachability has never been measured, and
GRADIEND runs were never pinned to a known arm.

This ablation trains one GRADIEND (one-pole) feature class per (model, arm, case) with
``learning_rate_decoder`` fixed per arm, and reports the outcomes that decide whether
GRADIEND should adopt ``"auto"``, pin ``None``, or use an explicit ratio:

  * ``None``   — historical shared single-optimizer group (what the config comment
                 claims every run used).
  * ``auto``   — the current package default (decoupled + reachability-floor raise).
  * explicit floats (``--decoder-lrs``) — a fixed decoupled rate, e.g. a 100x ratio.

Per cell it reads back:
  * the package's own convergence verdict (``convergence_info`` / ``best_score_checkpoint``
    from ``training.json``) — the same min_auc_n_o / correlation pass/fail the pipeline uses;
  * encoding metrics (roc_auc_neutral/other, class_exclusivity, correlation) from the
    trainer's fair encoder eval;
  * the TRAINED DECODER NORM (``gradiend.decoder[0].linear.weight``) — the direct §2.7
    reachability endpoint (does the decoder move toward the ridge optimum, or stall?);
  * the decoder CAUSAL effect — the study-level HEADLINE metric, always computed (not
    optional): it is what decides whether the shared_none vs auto decoder-LR difference
    actually moved GRADIEND's causal results.

Cells are isolated (a failing arm never terminates the grid) and written under
``runs/<model>/<task>/<output-subdir>/`` — never the cached ``artifacts/`` tree.

Examples
--------
  # plumbing smoke (2 steps; not a real convergence signal)
  python scripts/ablation_gradiend_decoder_lr.py --smoke

  # one model / one case, None vs auto
  python scripts/ablation_gradiend_decoder_lr.py --models gpt2-small --cases gender_en:M

  # full arms incl. a 100x explicit ratio (causal is always computed)
  export TRAIN_CMD='python scripts/ablation_gradiend_decoder_lr.py --decoder-lrs 1e-2'
  bash slurm/submit_train.sh gpumem-24-1x
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODELS: Tuple[str, ...] = ("gpt2-small", "pythia-70m-deduped")

# GRADIEND one-pole cells spanning binary / multi-class / circuit task shapes, so
# a decoder-LR effect that is task-shape-dependent is visible rather than averaged
# away. task:feature_class, mirroring ablation_actiend_lr_scale.py's --cases.
DEFAULT_CASES: Tuple[Tuple[str, str], ...] = (
    ("gender_en", "M"),        # binary
    ("emotion", "positive"),   # binary-ish
    ("ravel_country", "china"),# multi-class
    ("ioi", "IO"),             # answer-only circuit
)

# (arm id, learning_rate_decoder value). None = historical shared; "auto" = the
# current package default this ablation exists to measure against.
BASE_ARMS: Tuple[Tuple[str, Any], ...] = (
    ("shared_none", None),
    ("auto", "auto"),
)


def _parse_cases(raw: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in raw:
        if ":" not in item:
            raise SystemExit(f"--cases entries must be 'task:feature_class', got {item!r}")
        task, cls = item.split(":", 1)
        out.append((task.strip(), cls.strip()))
    return out


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return f"{float(val):.{digits}g}"
    return str(val)


@contextmanager
def _patched_decoder_lr(decoder_lr: Any):
    """Force ``TrainingArguments.learning_rate_decoder`` for every built cell.

    Patches BOTH study aliases that build TrainingArguments (``study.stages.train``
    and ``study.training_profiles``), like ablation_actiend_lr_scale.py. Sets the
    field on the already-built args (after ``__post_init__``); the optimizer reads
    ``self.args.learning_rate_decoder`` at ``.train()`` time. ``"auto"`` needs
    ``eval_steps >= 1`` (the study default is 50, so this holds); we assert it so a
    misconfigured cell fails loudly instead of silently degrading.
    """
    from study import training_profiles as tp
    from study.stages import train as train_stage

    original_train_build = train_stage.build_training_arguments
    original_profile_build = tp.build_training_arguments

    def _build(*args: Any, **kwargs: Any):
        built = original_train_build(*args, **kwargs)
        if decoder_lr == "auto" and int(getattr(built, "eval_steps", 0) or 0) < 1:
            raise RuntimeError("learning_rate_decoder='auto' requires eval_steps >= 1")
        built.learning_rate_decoder = decoder_lr
        meta = dict(getattr(built, "metadata", None) or {})
        meta["learning_rate_decoder"] = decoder_lr
        built.metadata = meta
        return built

    train_stage.build_training_arguments = _build  # type: ignore[assignment]
    tp.build_training_arguments = _build  # type: ignore[assignment]
    try:
        yield
    finally:
        train_stage.build_training_arguments = original_train_build  # type: ignore[assignment]
        tp.build_training_arguments = original_profile_build  # type: ignore[assignment]


def _load_convergence_summary(experiment_dir: Path) -> Dict[str, Any]:
    for candidate in (experiment_dir / "training.json", experiment_dir / "model" / "training.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        out: Dict[str, Any] = {}
        if isinstance(payload.get("best_score_checkpoint"), dict):
            out["best_score_checkpoint"] = payload["best_score_checkpoint"]
        if isinstance(payload.get("convergence_info"), dict):
            out["convergence_info"] = payload["convergence_info"]
        if out:
            return out
    return {}


def _decoder_norm(trainer: Any) -> Optional[float]:
    """Trained decoder weight norm — the §2.7 reachability endpoint."""
    try:
        import torch

        gm = trainer.get_model()
        gradiend = getattr(gm, "gradiend", None)
        w = gradiend.decoder[0].linear.weight  # type: ignore[union-attr]
        return float(torch.linalg.vector_norm(w.detach().reshape(-1)))
    except Exception:
        return None


def _encoding_row(out: Mapping[str, Any], cls: str) -> Dict[str, Any]:
    per_class = out.get("per_class_readouts") or {}
    entry = per_class.get(str(cls)) if isinstance(per_class, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    return {
        "roc_auc_neutral": entry.get("roc_auc_neutral"),
        "roc_auc_other": entry.get("roc_auc_other"),
        "class_exclusivity": entry.get("class_exclusivity"),
        "correlation": entry.get("correlation"),
        "mean_target": entry.get("mean_target"),
    }


def _run_causal(raw: Mapping[str, Any], *, feature_class: str, max_size: int) -> Dict[str, Any]:
    from causal_eval import (
        evaluate_decoder_for_classes,
        invalidate_decoder_grid_cache,
        _extract_decoder_summary_and_grid,
    )

    trainer = raw.get("trainer")
    if trainer is None:
        return {"error": "trainer unavailable for causal eval"}
    try:
        decoded = evaluate_decoder_for_classes(
            trainer, [str(feature_class)], max_size=int(max_size), plot=False,
            show=False, use_cache=False, split="validation",
        )
        summary, _grid = _extract_decoder_summary_and_grid(dict(decoded or {}))
        entry = summary.get(str(feature_class)) if isinstance(summary, dict) else {}
        entry = entry if isinstance(entry, dict) else {}
        return {
            "learning_rate": entry.get("learning_rate"),
            "value": entry.get("value"),
            "lms": entry.get("lms"),
        }
    except Exception as exc:
        invalidate_decoder_grid_cache(trainer)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_cell(
    *,
    model: str,
    task: str,
    feature_class: str,
    arm_id: str,
    decoder_lr: Any,
    output_subdir: str,
    smoke: bool,
    causal_max_size: int,
) -> Dict[str, Any]:
    from study.config import load_study_config
    from study.registry import build_task
    from study import training_profiles as tp
    from study.stages.train import _train_once

    cell: Dict[str, Any] = {
        "model": model, "task": task, "feature_class": feature_class,
        "arm": arm_id, "learning_rate_decoder": decoder_lr,
    }
    try:
        cfg = load_study_config(model=model, task=task, suite="full_plus")
        bundle = build_task(task, cfg.raw, smoke=smoke)
        shared = tp.shared_training_kwargs(cfg, backend="gradiend")
        if smoke:
            shared = tp.apply_smoke_training_kwargs(shared)
        exp_dir = ROOT / "runs" / model / task / output_subdir / f"gradiend_{feature_class}_{arm_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        with _patched_decoder_lr(decoder_lr):
            out = _train_once(
                backend="gradiend",
                cfg=cfg,
                bundle=bundle,
                target_classes=[feature_class],
                experiment_dir=exp_dir,
                split_mode="none",
                shared=shared,
                smoke=smoke,
                feature_class=feature_class,
                counterfactual_classes="all",
                all_classes=list(bundle.classes),
            )
        conv = _load_convergence_summary(exp_dir)
        info = conv.get("convergence_info") or {}
        cell.update(
            converged=info.get("converged"),
            convergent_metric=info.get("metric") or (out.get("convergent_metric")),
            threshold=info.get("threshold"),
            best_score=(conv.get("best_score_checkpoint") or {}).get(
                str(info.get("metric") or "")
            ),
            decoder_norm=_decoder_norm(out.get("trainer")),
            **_encoding_row(out, feature_class),
        )
        # Causal effect is the HEADLINE metric of this study — always computed, not
        # optional. It is what decides whether the shared_none vs auto decoder-LR
        # difference actually moved GRADIEND's causal results across tasks.
        cell["causal"] = _run_causal(out, feature_class=feature_class, max_size=causal_max_size)
        trainer = out.get("trainer")
        if trainer is not None and hasattr(trainer, "unload_model"):
            try:
                trainer.unload_model()
            except Exception:
                pass
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  cell FAILED {model}/{task}:{feature_class} arm={arm_id}: {exc}", flush=True)
    return cell


def _print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    # Causal effect first — it is the headline metric this ablation is judged on.
    cols = [
        ("model", 16), ("task", 15), ("cls", 10), ("arm", 12),
        ("CAUSAL", 9), ("lms", 6), ("conv", 5), ("minAUC", 8),
        ("auc_n", 7), ("auc_o", 7), ("excl", 7), ("corr", 7),
        ("dec_norm", 9), ("error", 24),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        auc_n = r.get("roc_auc_neutral")
        auc_o = r.get("roc_auc_other")
        min_auc = None
        if isinstance(auc_n, (int, float)) and isinstance(auc_o, (int, float)):
            min_auc = min(float(auc_n), float(auc_o))
        causal = r.get("causal") or {}
        causal_val = causal.get("value") if isinstance(causal, dict) else None
        causal_lms = causal.get("lms") if isinstance(causal, dict) else None
        err = r.get("error") or (causal.get("error") if isinstance(causal, dict) else "") or ""
        vals = [
            r.get("model"), r.get("task"), r.get("feature_class"), r.get("arm"),
            causal_val, causal_lms, r.get("converged"), min_auc, auc_n, auc_o,
            r.get("class_exclusivity"), r.get("correlation"), r.get("decoder_norm"),
            str(err)[:24],
        ]
        print("  ".join(_fmt(v).ljust(w) for v, (_n, w) in zip(vals, cols)))


def build_arms(decoder_lrs: Sequence[float]) -> List[Tuple[str, Any]]:
    arms = list(BASE_ARMS)
    for lr in decoder_lrs:
        arms.append((f"lr_{lr:g}", float(lr)))
    return arms


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--cases", nargs="+", default=None, help="task:feature_class (default: a shape-spanning set)")
    p.add_argument("--decoder-lrs", nargs="+", type=float, default=(),
                   help="Explicit decoupled decoder LR arm(s) in addition to shared_none + auto.")
    p.add_argument("--output-subdir", default="ablation_gradiend_decoder_lr")
    p.add_argument("--causal-max-size", type=int, default=64)
    p.add_argument("--smoke", action="store_true", help="2-step plumbing check (not a convergence signal).")
    p.add_argument("--no-fail-fast", action="store_true",
                   help="Keep running after a cell errors (cell-isolation). OFF by default: "
                        "a systematic first-cell error aborts the run so hours of GPU aren't "
                        "burned on an all-error arm. See CLAUDE.md 'ALWAYS FAIL FAST'.")
    args = p.parse_args(argv)
    fail_fast = not args.no_fail_fast

    cases = _parse_cases(args.cases) if args.cases else list(DEFAULT_CASES)
    arms = build_arms(args.decoder_lrs)
    # One comparison.json per --output-subdir so a second invocation can never overwrite an
    # earlier one. The default subdir keeps the historical path.
    out_root = ROOT / "runs" / f"_{args.output_subdir}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for model in args.models:
        for task, cls in cases:
            for arm_id, decoder_lr in arms:
                print(f"CELL {model}/{task}:{cls} arm={arm_id} decoder_lr={decoder_lr!r}", flush=True)
                cell = _run_cell(
                    model=model, task=task, feature_class=cls,
                    arm_id=arm_id, decoder_lr=decoder_lr,
                    output_subdir=args.output_subdir, smoke=args.smoke,
                    causal_max_size=int(args.causal_max_size),
                )
                rows.append(cell)
                if fail_fast and cell.get("error"):
                    # Fail fast on the FIRST cell error: it is almost always
                    # systematic (an arm that errors on cell 1 errors on all its
                    # cells), so continuing burns GPU on an all-error arm. Persist
                    # what we have, then abort loudly. --no-fail-fast to override.
                    (out_root / "comparison.json").write_text(
                        json.dumps({"rows": rows, "arms": [a for a, _ in arms],
                                    "cases": [f"{t}:{c}" for t, c in cases],
                                    "aborted_on_error": cell}, indent=2, default=str) + "\n",
                        encoding="utf-8",
                    )
                    _print_table(rows)
                    print(f"\nFAIL-FAST: aborting after cell error "
                          f"{model}/{task}:{cls} arm={arm_id}: {cell['error']}", flush=True)
                    return 1

    _print_table(rows)
    (out_root / "comparison.json").write_text(
        json.dumps({"rows": rows, "arms": [a for a, _ in arms],
                    "cases": [f"{t}:{c}" for t, c in cases]}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_root / 'comparison.json'} ({len(rows)} cells)")
    # Non-zero exit if every cell errored (so a broken launch fails loudly).
    return 0 if any("error" not in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
