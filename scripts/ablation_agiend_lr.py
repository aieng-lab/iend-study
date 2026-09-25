#!/usr/bin/env python
"""Ablation: tune AGIEND's training learning rate, and test whether a decoupled
("auto") decoder LR is worth it for the activation-gradient (dL/dh) signal.

AGIEND is the LEARNED encoder-decoder on ``dL/dh`` (the activation-gradient signal) --
the act-grad × learned cell of the 3x2 grid, trained like ACTIEND but on a different
signal. Two things are untuned/unvalidated for it:

  1. TRAINING LR. GRADIEND (weight-grad) and ACTIEND (activation) have per-model tuned
     LRs; AGIEND's signal is ``dL/dh`` with its own magnitude/geometry, so its optimal
     encoder LR is unknown. This sweeps ``--lrs``.
  2. DECODER LR. The package default now resolves ``activation_gradient`` -> decoder LR
     None (shared), NOT the "auto" reachability lift (which was validated only for the
     ACTIVATION signal / ACTIEND, IEND_THEORY_PLAN 2.7). This ablation measures whether
     "auto" actually helps AGIEND (``--decoder-arms default auto``) so that default is a
     measured choice, not an assumption.

Grid = ``--lrs`` x ``--decoder-arms``. Per cell it trains AGIEND (``_train_once`` on the
activation_gradient signal) and reads:
  * CAUSAL effect (headline, always computed) via the decoder strengthen sweep;
  * the package's convergence verdict (min_auc_n_o) from training.json;
  * the trained decoder norm (the 2.7 reachability endpoint).

Read: down a decoder column, pick the LR where AGIEND converges / peaks causal. Compare
``default`` vs ``auto`` at that LR -- if auto materially beats shared on convergence/causal,
flip the ``activation_gradient`` default (or set ``learning_rate_decoder_agiend``); if not,
the None default stands.

Cells are isolated (a failing arm never terminates the grid); written under
``runs/<model>/<task>/<output-subdir>/`` -- never the cached ``artifacts/`` tree.

Examples
--------
  python scripts/ablation_agiend_lr.py --smoke                      # plumbing (2 steps)
  python scripts/ablation_agiend_lr.py --models gpt2-small --cases gender_en:M --lrs 1e-5 3e-5
  export TRAIN_CMD='python scripts/ablation_agiend_lr.py --lrs 1e-6 3e-6 1e-5 3e-5 1e-4 --decoder-arms default auto'
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
# gender_en converges easily (LR-insensitive floor); a harder task exposes LR
# sensitivity. task:feature_class, like ablation_actiend_lr_scale.py.
DEFAULT_CASES: Tuple[Tuple[str, str], ...] = (
    ("gender_en", "M"),
    ("emotion", "positive"),
)
DEFAULT_LRS: Tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
# (arm id, learning_rate_decoder value). "default" -> don't set it -> package resolves
# activation_gradient to None (shared). "auto" -> force the reachability lift.
DECODER_ARMS: Dict[str, Any] = {"default": "__unset__", "auto": "auto"}


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
def _patched_train_args(*, lr: float, decoder_lr: Any, max_steps: Optional[int] = None):
    """Force ``learning_rate`` (and, unless '__unset__', ``learning_rate_decoder``) on
    every built cell. Patches both study aliases (train + training_profiles), like
    ablation_actiend_lr_scale.py / ablation_gradiend_decoder_lr.py."""
    from study import training_profiles as tp
    from study.stages import train as train_stage

    orig_train = train_stage.build_training_arguments
    orig_profile = tp.build_training_arguments

    def _build(*args: Any, **kwargs: Any):
        built = orig_train(*args, **kwargs)
        built.learning_rate = float(lr)
        if max_steps is not None:
            built.max_steps = int(max_steps)
            # eval_steps must stay strictly below max_steps (also required by the 'auto' arm).
            if int(getattr(built, "eval_steps", 0) or 0) >= int(max_steps):
                built.eval_steps = max(1, int(max_steps) // 5)
        if decoder_lr != "__unset__":
            if decoder_lr == "auto" and int(getattr(built, "eval_steps", 0) or 0) < 1:
                raise RuntimeError("learning_rate_decoder='auto' requires eval_steps >= 1")
            built.learning_rate_decoder = decoder_lr
        meta = dict(getattr(built, "metadata", None) or {})
        meta["learning_rate"] = float(lr)
        if max_steps is not None:
            meta["max_steps"] = int(max_steps)
        if decoder_lr != "__unset__":
            meta["learning_rate_decoder"] = decoder_lr
        built.metadata = meta
        return built

    train_stage.build_training_arguments = _build  # type: ignore[assignment]
    tp.build_training_arguments = _build  # type: ignore[assignment]
    try:
        yield
    finally:
        train_stage.build_training_arguments = orig_train  # type: ignore[assignment]
        tp.build_training_arguments = orig_profile  # type: ignore[assignment]


def _load_convergence_summary(experiment_dir: Path) -> Dict[str, Any]:
    for c in (experiment_dir / "training.json", experiment_dir / "model" / "training.json"):
        try:
            payload = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            out: Dict[str, Any] = {}
            if isinstance(payload.get("best_score_checkpoint"), dict):
                out["best_score_checkpoint"] = payload["best_score_checkpoint"]
            if isinstance(payload.get("convergence_info"), dict):
                out["convergence_info"] = payload["convergence_info"]
            if out:
                return out
    return {}


def _decoder_norm(trainer: Any) -> Optional[float]:
    try:
        import torch

        w = trainer.get_model().gradiend.decoder[0].linear.weight  # type: ignore[union-attr]
        return float(torch.linalg.vector_norm(w.detach().reshape(-1)))
    except Exception:
        return None


def _run_causal(raw: Mapping[str, Any], *, cfg: Any, bundle: Any,
                feature_class: str, max_size: int) -> Dict[str, Any]:
    """AGIEND steers ACTIVATIONS (like CAGA/CAA), NOT weight rewrite. So its causal
    goes through the persisted learned decoder direction (``caga_specs_from_model``,
    no re-fit) + ``run_caa_causal_sweep`` — the exact path the main pipeline's
    activation-gradient causal block uses. The old ``evaluate_decoder_for_classes``
    call was the GRADIEND weight-rewrite path and errored with "Parameter
    'activation:transformer.h.0' not found for GRADIEND intervention" (see CLAUDE.md
    "NEVER rationalize a wrong shortcut" — wrong intervention, not a limitation)."""
    from caga_eval import caga_specs_from_model, caga_causal_id
    from caa_eval import run_caa_causal_sweep
    from causal_eval import DEFAULT_CAUSAL_STRENGTHS, stratified_causal_texts
    from study.tasks import labeled_df_for_eval

    trainer = raw.get("trainer")
    if trainer is None:
        return {"error": "no trainer"}
    try:
        model = trainer.get_model().base_model
        specs = caga_specs_from_model(trainer.get_model())
        labeled_df = labeled_df_for_eval(bundle, expand_one_pole=True)
        tcs = [str(c) for c in bundle.classes]
        val_rows = stratified_causal_texts(
            labeled_df, bundle.neutrals, target_classes=tcs,
            n_per_group=int(max_size), split="validation", seed=0,
        )
        meta_rows = stratified_causal_texts(
            labeled_df, bundle.neutrals, target_classes=tcs,
            n_per_group=int(max_size), split="test", seed=0,
        )
        res = run_caa_causal_sweep(
            model, trainer.tokenizer, val_rows, specs=specs,
            target_class=str(feature_class),
            strengths=list(DEFAULT_CAUSAL_STRENGTHS), report_rows=meta_rows,
            include_random_control=True, method_id=caga_causal_id(str(feature_class)),
            token_selector="all", act_policy="prediction", part=None,
        )
        sel = getattr(res, "selected", None)
        return {
            "value": getattr(sel, "signed_effect", None) if sel is not None else None,
            "lms": getattr(sel, "lms_ok", None) if sel is not None else None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_cell(
    *, model: str, task: str, feature_class: str, lr: float, arm_id: str, decoder_lr: Any,
    output_subdir: str, smoke: bool, causal_max_size: int, max_seeds: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    from study.config import load_study_config
    from study.registry import build_task
    from study import training_profiles as tp
    from study.stages.train import _train_once

    cell: Dict[str, Any] = {
        "model": model, "task": task, "feature_class": feature_class,
        "lr": lr, "decoder_arm": arm_id, "max_steps": max_steps,
    }
    try:
        cfg = load_study_config(model=model, task=task, suite="full_plus")
        bundle = build_task(task, cfg.raw, smoke=smoke)
        shared = tp.shared_training_kwargs(cfg, backend="agiend")
        if smoke:
            shared = tp.apply_smoke_training_kwargs(shared)
        if max_seeds is not None:
            # LR-SCREEN mode: 1 seed is enough to see which LR converges (we want the
            # LR trend, not seed stability -- the chosen LR gets full seeds on slurm).
            shared["max_seeds"] = int(max_seeds)
            shared["min_convergent_seeds"] = 1
        tag = f"agiend_{feature_class}_lr{lr:g}_{arm_id}"
        exp_dir = ROOT / "runs" / model / task / output_subdir / tag
        exp_dir.mkdir(parents=True, exist_ok=True)
        with _patched_train_args(lr=lr, decoder_lr=decoder_lr, max_steps=max_steps):
            out = _train_once(
                backend="agiend", cfg=cfg, bundle=bundle, target_classes=[feature_class],
                experiment_dir=exp_dir, split_mode="none", shared=shared, smoke=smoke,
                feature_class=feature_class, counterfactual_classes="all",
                all_classes=list(bundle.classes),
            )
        info = (_load_convergence_summary(exp_dir).get("convergence_info") or {})
        per_class = (out.get("per_class_readouts") or {}).get(str(feature_class)) or {}
        cell.update(
            converged=info.get("converged"),
            best_score=info.get("best_score"),
            roc_auc_neutral=per_class.get("roc_auc_neutral"),
            roc_auc_other=per_class.get("roc_auc_other"),
            decoder_norm=_decoder_norm(out.get("trainer")),
            causal=_run_causal(out, cfg=cfg, bundle=bundle, feature_class=feature_class, max_size=causal_max_size),
        )
        trainer = out.get("trainer")
        if trainer is not None and hasattr(trainer, "unload_model"):
            try:
                trainer.unload_model()
            except Exception:
                pass
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  cell FAILED {model}/{task}:{feature_class} lr={lr:g} arm={arm_id}: {exc}", flush=True)
    return cell


def _print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    cols = [
        ("model", 16), ("task", 12), ("cls", 9), ("lr", 8), ("dec_arm", 8),
        ("CAUSAL", 9), ("lms", 6), ("conv", 5), ("minAUC", 8), ("dec_norm", 9), ("error", 22),
    ]
    print("\n" + "  ".join(n.ljust(w) for n, w in cols))
    print("-" * (sum(w for _, w in cols) + 2 * len(cols)))
    for r in rows:
        an, ao = r.get("roc_auc_neutral"), r.get("roc_auc_other")
        min_auc = min(float(an), float(ao)) if isinstance(an, (int, float)) and isinstance(ao, (int, float)) else None
        causal = r.get("causal") or {}
        err = r.get("error") or (causal.get("error") if isinstance(causal, dict) else "") or ""
        vals = [
            r.get("model"), r.get("task"), r.get("feature_class"), r.get("lr"), r.get("decoder_arm"),
            causal.get("value") if isinstance(causal, dict) else None,
            causal.get("lms") if isinstance(causal, dict) else None,
            r.get("converged"), min_auc, r.get("decoder_norm"), str(err)[:22],
        ]
        print("  ".join(_fmt(v).ljust(w) for v, (_n, w) in zip(vals, cols)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--cases", nargs="+", default=None, help="task:feature_class")
    p.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    p.add_argument("--decoder-arms", nargs="+", default=["default", "auto"],
                   choices=sorted(DECODER_ARMS), help="decoder-LR arms to compare")
    p.add_argument("--output-subdir", default="ablation_agiend_lr")
    p.add_argument("--causal-max-size", type=int, default=64)
    p.add_argument("--max-seeds", type=int, default=None,
                   help="Cap training seeds per cell (e.g. 1 for a fast LR screen). "
                        "Default None = study default (3).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override the training step budget (e.g. 100 for the 100-vs-500 schedule "
                        "arm). Default None = study default (500). eval_steps is lowered to "
                        "max_steps//5 if it would not stay below max_steps.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-fail-fast", action="store_true",
                   help="Keep running after a cell raises. OFF by default: a cell 'error' is an "
                        "EXCEPTION (plumbing bug), not non-convergence (recorded as data), so the "
                        "first one aborts the run instead of burning GPU. See CLAUDE.md 'ALWAYS FAIL FAST'.")
    args = p.parse_args(argv)
    fail_fast = not args.no_fail_fast

    cases = _parse_cases(args.cases) if args.cases else list(DEFAULT_CASES)
    # One comparison.json per --output-subdir so a second invocation (other model / step budget)
    # can never overwrite an earlier one. The default subdir keeps the historical path.
    out_root = ROOT / "runs" / f"_{args.output_subdir}"
    out_root.mkdir(parents=True, exist_ok=True)
    comparison_path = out_root / "comparison.json"

    def _flush(rows: List[Dict[str, Any]]) -> None:
        # Written after EVERY cell so partial results survive if the grid is cut short.
        comparison_path.write_text(
            json.dumps({"rows": rows, "lrs": list(args.lrs), "decoder_arms": list(args.decoder_arms),
                        "cases": [f"{t}:{c}" for t, c in cases],
                        "max_steps": args.max_steps}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    rows: List[Dict[str, Any]] = []
    for model in args.models:
        for task, cls in cases:
            for lr in args.lrs:
                for arm_id in args.decoder_arms:
                    print(f"CELL {model}/{task}:{cls} lr={lr:g} arm={arm_id}", flush=True)
                    cell = _run_cell(
                        model=model, task=task, feature_class=cls, lr=lr, arm_id=arm_id,
                        decoder_lr=DECODER_ARMS[arm_id], output_subdir=args.output_subdir,
                        smoke=args.smoke, causal_max_size=int(args.causal_max_size),
                        max_seeds=args.max_seeds, max_steps=args.max_steps,
                    )
                    rows.append(cell)
                    _flush(rows)          # incremental: survive an interrupted grid
                    _print_table(rows)    # running table after each cell
                    if fail_fast and cell.get("error"):
                        print(f"\nFAIL-FAST: aborting after cell error "
                              f"{model}/{task}:{cls} lr={lr:g} arm={arm_id}: {cell['error']}", flush=True)
                        return 1
    print(f"\nWrote {comparison_path} ({len(rows)} cells)")
    return 0 if any("error" not in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
