#!/usr/bin/env python
"""Ablation: does ``Signal.activation(scale='running_rms')`` let ACTIEND use
one learning rate across model architectures, instead of the per-model
empirical LR tuning currently hardcoded in ``configs/models/*.yaml``?

Background: ``configs/models/pythia-70m-deduped.yaml`` sets
``learning_rate_actiend`` to 1/100th of the shared default (1e-7 vs 1e-5),
copying the ratio that was empirically tuned for GRADIEND (a *different*
signal — raw weight-gradients, not activations). Pythia ACTIEND then failed
to converge. Historically both GRADIEND and ACTIEND trained on a *raw,
unnormalized* signal
(confirmed against ``gradiend/model/param_mapped.py`` and
``gradiend/trainer/core/signals.py``) fed into a roughly-linear first layer
under AdamW — so architecture-dependent raw-signal magnitude is a plausible
root cause, and ``Signal.activation(scale="running_rms")``
(``gradiend/trainer/core/signals.py``) already exists to remove exactly that
confound by dividing activations by an online per-site RMS estimate before
they reach the encoder. It is implemented but was never wired into this
study's training pipeline (unlike ``scripts/poc_actiend_activation_scale.py``,
which compares raw vs running_rms *causal* effect at a single fixed training
LR, this script sweeps the *training* LR itself and reports the package's own
convergence verdict — the question here is convergence, not causal effect).

For each (model, mode, lr) cell this trains one one-pole ACTIEND feature
class on one task, then reads the package's own ``convergence_info`` /
``best_score_checkpoint`` straight out of ``training.json`` — the same
``convergent_metric``/``convergent_score_threshold`` pass/fail definition
the rest of the pipeline already uses (min_auc_n_o / 0.9 for one-pole, raised
from 0.7 on 2026-08-20; see CLAUDE.md's "Convergence/checkpoint-selection
criteria" note) — instead of inventing a bespoke metric for this ablation.
The threshold printed in this script's own table (``thr`` column) is always
read back from the trainer's actual ``training.json``/``convergent_score_threshold``,
never hardcoded here — so it reflects whichever ``gradiend`` package build
actually ran the cell, not necessarily 0.9 (see the note below about the
package fix being uncommitted).

If ``running_rms`` converges at the *same* LR for every model while ``raw``
only converges at wildly different, per-model LRs, that's the signal to drop
per-model ``learning_rate_actiend`` overrides entirely and just turn
running_rms on everywhere.

Writes into ``runs/<model>/<task>/<output-subdir>/`` — separate from the
main ``artifacts/`` tree, same convention as
``scripts/poc_actiend_activation_scale.py`` — so this never touches cached
study results.

Default ``--cases`` are the 7 genuine gpt2-small one-pole ACTIEND failures at
the study's tuned LR (see ``DEFAULT_CASES`` below) -- these are real training
difficulty, not just an untested cross-model LR guess, so they're a much
better test of whether running_rms actually helps than an easy task like
gender_en (which converged everywhere in an earlier, smaller sweep).

The runner can also force every configured training seed (rather than the
package default of stopping after the first converged seed), and can run the
package decoder causal evaluation for successful or borderline cells.  Those
checks are opt-in because they multiply the cost of the full model x task x LR
grid.  Each cell is isolated, so a deliberately high stress LR can fail without
terminating the remaining grid.

Examples
--------
  # Local / node smoke (plumbing check only — 2 steps, not a real convergence signal):
  python scripts/ablation_actiend_lr_scale.py --smoke

  # One model, one case, quick look:
  python scripts/ablation_actiend_lr_scale.py --models gpt2-small --cases ioi:IO --lrs 1e-5 1e-4

  # Full convergence grid (2 models x 2 modes x 6 LRs x 7 cases = 168 cells):
  # split across multiple Slurm jobs by --cases/--models if that's too much for one job.
  export TRAIN_CMD='python scripts/ablation_actiend_lr_scale.py'
  bash slurm/submit_train.sh gpumem-24-1x

  # Validation run: all 3 seeds, guarded 1e-3 stress cell, and causal eval for
  # checkpoints with min_auc_n_o >= 0.8 (successful + borderline):
  export TRAIN_CMD='python scripts/ablation_actiend_lr_scale.py --all-seeds --max-seeds 3 --include-stress-lr --causal --causal-all-seeds --causal-min-score 0.8'
  bash slurm/submit_train.sh gpumem-24-1x
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gradiend import Signal  # noqa: E402
from study.stages.train import ONE_POLE_DATA_PROTOCOL_VERSION  # noqa: E402

# (id, scale kw for Signal.activation)
MODES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("raw", None),
    ("running_rms", "running_rms"),
)

DEFAULT_MODELS: Tuple[str, ...] = ("gpt2-small", "pythia-70m-deduped")
# Anchored at gpt2-small's tuned ACTIEND LR (1e-5) and extended up through
# 3e-4 -- CLAUDE.md notes raw-signal ACTIEND "collapses at 1e-4 on
# several gpt2-small tasks", which is plausibly *why* 1e-5 was chosen despite
# these tasks still failing there. If running_rms tolerates 1e-4/3e-4 without
# collapsing where raw does, that's a real fix, not just LR-matching.
DEFAULT_LRS: Tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
STRESS_LRS: Tuple[float, ...] = (1e-3,)

# CAA-matched ungated intervention plus the current ACTIEND headline gate.
# Kept aligned with scripts/poc_actiend_activation_scale.py.
CAUSAL_VARIANTS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("ungated_all", "all", None),
    ("gated_all", "all", "encoder_direction"),
)

# Genuine one-pole non-convergence found in runs/gpt2-small/suite_full/*
# (learning_rate_actiend=1e-5, the study's tuned default) as of 2026-08-21,
# after excluding done.json rows matching the AUROC-orientation-bug signature
# (roc_auc_neutral near 0 while roc_auc_other is fine -- CLAUDE.md's
# "Package-side checkpoint-selection AUROC orientation bug", fixed
# 2026-08-20 but not yet reflected in these cached runs) and rows missing
# convergence_info entirely (pre-2026-08-18 done.json, no score to judge).
# Pair/two-pole ACTIEND had zero genuine failures at this LR on gpt2-small --
# only one-pole (min_auc_n_o) training struggled.
DEFAULT_CASES: Tuple[Tuple[str, str], ...] = (
    ("emotion", "negative"),
    ("emotion", "positive"),
    ("induction", "MATCH"),
    ("ioi", "IO"),
    ("key_value", "VALUE"),
    ("language", "fr"),  # siblings de/en converge fine on the same task
    ("pronoun_person", "1"),  # borderline: 0.652 vs 0.7
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


def _with_signal_scale(args: Any, scale: Optional[str]) -> Any:
    """Replace ``args.signal`` explicitly for both raw and scaled cells.

    Mirrors ``scripts/poc_actiend_activation_scale.py``'s helper of the same
    name — kept as a separate copy since these are self-contained PoC-style
    scripts by convention in this repo, not a shared library.

    Must also refresh ``args.signals`` (the plural axis), not just
    ``args.signal``: ``TrainingArguments.__post_init__`` reconciles the two
    once at construction time via ``_normalize_signal_arguments`` (see
    ``gradiend/trainer/core/arguments.py``), but overwriting ``args.signal``
    afterward leaves ``args.signals`` pointing at the *original* (pre-patch)
    signal. Trainer setup later calls ``require_single_signal(signal=...,
    signals=...)`` (``gradiend/trainer/core/signals.py``), which re-runs the
    same reconciliation and raises ``ValueError: Pass either signal=... or
    signals=..., unless signals contains exactly the same single signal.``
    the moment the two disagree — reproduced 2026-08-20 on every
    ``running_rms`` cell of this ablation on both models (``raw`` cells never
    hit it because ``scale=None`` short-circuits before reassigning
    ``args.signal`` at all). Re-running the package's own
    ``_normalize_signal_arguments`` after the patch keeps both fields
    consistent instead of duplicating its reconciliation logic here.

    Raw must be rebuilt too.  Since 2026-08-21 the study builder itself uses
    ``running_rms`` by default, so returning the study-built signal unchanged
    for ``scale=None`` silently turns the raw arm into another scaled arm.
    """
    old = getattr(args, "signal", None)
    if old is None:
        raise RuntimeError("TrainingArguments.signal is missing after build")
    opts = dict(getattr(old, "options", None) or {})
    tok = opts.get("token_selector")
    tgt = opts.get("target_token_selector")
    signal_kw: Dict[str, Any] = {
        "token_selector": tok,
        "target_token_selector": tgt,
    }
    if scale is not None:
        signal_kw.update(
            scale=scale,
            scale_reduce="per_site",
            scale_momentum=0.0,
        )
    args.signal = Signal.activation(**signal_kw)
    args.signals = None
    args._normalize_signal_arguments()
    # ``build_training_arguments`` also records its default scaling policy in
    # metadata before this ablation replaces the signal.  Keep the persisted
    # done.json metadata aligned with the signal that was actually trained;
    # otherwise a valid raw cell misleadingly reports ``running_rms``.
    metadata = dict(getattr(args, "metadata", None) or {})
    metadata["actiend_signal_scale"] = (
        {"scale": "raw"}
        if scale is None
        else {
            "scale": scale,
            "scale_reduce": "per_site",
            "scale_momentum": 0.0,
        }
    )
    args.metadata = metadata
    return args


@contextmanager
def _patched_training_arguments(
    *,
    scale: Optional[str],
    lr: float,
    captured: Optional[Dict[str, Any]] = None,
):
    """Temporarily patch both study aliases that build TrainingArguments."""
    from study import training_profiles as tp
    from study.stages import train as train_stage

    captured = captured if captured is not None else {}
    original_train_build = train_stage.build_training_arguments
    original_profile_build = tp.build_training_arguments

    def _build_scaled(*args: Any, **kwargs: Any):
        train_args = original_train_build(*args, **kwargs)
        scaled = _with_signal_scale(train_args, scale)
        sig = getattr(scaled, "signal", None)
        opts = dict(getattr(sig, "options", None) or {}) if sig is not None else {}
        got = opts.get("scale")
        if scale is None and got not in (None, ""):
            raise RuntimeError(f"raw mode expected signal.scale=None, got {got!r}")
        if scale is not None and got != scale:
            raise RuntimeError(
                f"scale patch failed: wanted signal.scale={scale!r}, got {got!r}"
            )
        captured["convergent_metric"] = getattr(scaled, "convergent_metric", None)
        captured["convergent_score_threshold"] = getattr(
            scaled, "convergent_score_threshold", None
        )
        applied_lr = float(getattr(scaled, "learning_rate", float("nan")))
        if abs(applied_lr - float(lr)) > 1e-15 * max(1.0, abs(float(lr))):
            raise RuntimeError(f"lr patch failed: wanted {lr!r}, got {applied_lr!r}")
        return scaled

    train_stage.build_training_arguments = _build_scaled  # type: ignore[assignment]
    tp.build_training_arguments = _build_scaled  # type: ignore[assignment]
    try:
        yield
    finally:
        train_stage.build_training_arguments = original_train_build  # type: ignore[assignment]
        tp.build_training_arguments = original_profile_build  # type: ignore[assignment]


def _load_convergence_summary(experiment_dir: Path) -> Dict[str, Any]:
    """Same extraction as ``study.stages.train._load_convergence_summary``,
    duplicated here so this script also works when a cell is skipped (no
    trainer object around to import the study module through)."""
    for candidate in (
        experiment_dir / "training.json",
        experiment_dir / "model" / "training.json",
    ):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        out: Dict[str, Any] = {}
        bsc = payload.get("best_score_checkpoint")
        if isinstance(bsc, dict):
            out["best_score_checkpoint"] = bsc
        conv = payload.get("convergence_info")
        if isinstance(conv, dict):
            out["convergence_info"] = conv
        if out:
            return out
    return {}


def _load_seed_summary(experiment_dir: Path) -> Dict[str, Any]:
    """Load compact per-seed stability fields from the package report."""
    path = experiment_dir / "seeds" / "seed_report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    runs = [row for row in (payload.get("runs") or []) if isinstance(row, dict)]
    scores = [
        float(row["convergence_metric_value"])
        for row in runs
        if isinstance(row.get("convergence_metric_value"), (int, float))
    ]
    threshold = payload.get("threshold")
    score_pass_count = (
        sum(float(score) >= float(threshold) for score in scores)
        if isinstance(threshold, (int, float))
        else None
    )
    return {
        "seeds_tried": list(payload.get("seeds_tried") or []),
        "seed_count": len(runs),
        "seed_convergent_count": payload.get("convergent_count"),
        "seed_score_pass_count": score_pass_count,
        "seed_score_min": min(scores) if scores else None,
        "seed_score_mean": sum(scores) / len(scores) if scores else None,
        "seed_score_max": max(scores) if scores else None,
        "best_seed": payload.get("best_seed"),
        "early_stop_reason": payload.get("early_stop_reason"),
    }


def _summary_row(decoder_results: Mapping[str, Any], cls: str) -> Dict[str, Any]:
    from causal_eval import _extract_decoder_summary_and_grid

    summary, _grid = _extract_decoder_summary_and_grid(dict(decoder_results or {}))
    entry = summary.get(str(cls)) if isinstance(summary, dict) else None
    return dict(entry) if isinstance(entry, dict) else {}


def _run_causal(
    raw: Mapping[str, Any],
    *,
    feature_class: str,
    max_size: int,
    lrs: Sequence[float],
) -> List[Dict[str, Any]]:
    """Run decoder causal panels on one trained/reloaded ablation cell."""
    from causal_eval import evaluate_decoder_for_classes, invalidate_decoder_grid_cache

    trainer = raw.get("trainer")
    if trainer is None:
        return [{"class": feature_class, "error": "trainer unavailable for causal eval"}]
    rows: List[Dict[str, Any]] = []
    for variant, token_selector, gate in CAUSAL_VARIANTS:
        eval_kw: Dict[str, Any] = {
            "max_size": int(max_size),
            "plot": False,
            "show": False,
            "use_cache": False,
            "split": "validation",
            "token_selector": token_selector,
            "lrs": list(lrs),
        }
        if gate is not None:
            eval_kw["activation_gate"] = gate
        print(
            f"  causal [{variant}] class={feature_class!r} "
            f"tok={token_selector!r} gate={gate!r}",
            flush=True,
        )
        try:
            decoded = evaluate_decoder_for_classes(
                trainer, [str(feature_class)], **eval_kw
            )
            summary = _summary_row(decoded, feature_class)
            error = None
        except Exception as exc:
            invalidate_decoder_grid_cache(trainer)
            summary = {}
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "variant": variant,
                "class": feature_class,
                "learning_rate": summary.get("learning_rate"),
                "value": summary.get("value"),
                "lms": summary.get("lms"),
                "feature_factor": summary.get("feature_factor"),
                "error": error,
            }
        )
    return rows


def _causal_requested_for_row(row: Mapping[str, Any], min_score: float) -> bool:
    score = row.get("score")
    return isinstance(score, (int, float)) and float(score) >= float(min_score)


def _write_causal_rows(experiment_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    (experiment_dir / "causal.json").write_text(
        json.dumps({"rows": list(rows)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _reload_seed_for_causal(
    *,
    cfg: Any,
    bundle: Any,
    feature_class: str,
    scale: Optional[str],
    lr: float,
    smoke: bool,
    experiment_dir: Path,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Load one package multi-seed checkpoint without recomputing encoder eval."""
    from study import training_profiles as tp
    from study.stages.train import _build_trainer_for_artifact, _find_checkpoint_dir

    seed_dir = experiment_dir / "seeds" / f"seed_{int(seed)}"
    checkpoint = _find_checkpoint_dir(seed_dir)
    if checkpoint is None:
        return None
    shared = tp.shared_training_kwargs(cfg, backend="actiend")
    if smoke:
        shared = tp.apply_smoke_training_kwargs(shared)
    shared.update(
        learning_rate=float(lr),
        seed=int(seed),
        max_seeds=1,
        min_convergent_seeds=1,
    )
    with _patched_training_arguments(scale=scale, lr=lr):
        trainer, _args, _feature, _merged, _shared = _build_trainer_for_artifact(
            backend="actiend",
            cfg=cfg,
            bundle=bundle,
            target_classes=[feature_class],
            experiment_dir=seed_dir,
            split_mode="none",
            shared=shared,
            smoke=smoke,
            feature_class=feature_class,
            counterfactual_classes="all",
            all_classes=list(bundle.classes),
        )
    model_with_gradiend = trainer.get_model(load_directory=str(checkpoint))
    gradiend = getattr(model_with_gradiend, "gradiend", None)
    if gradiend is not None and hasattr(gradiend, "_require_built"):
        gradiend._require_built()
    return {"trainer": trainer}


def _run_cell_causal(
    selected_raw: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    cfg: Any,
    bundle: Any,
    feature_class: str,
    scale: Optional[str],
    lr: float,
    smoke: bool,
    experiment_dir: Path,
    max_size: int,
    lrs: Sequence[float],
    all_seeds: bool,
) -> List[Dict[str, Any]]:
    """Evaluate the selected checkpoint or every package seed checkpoint."""
    seeds = list(row.get("seeds_tried") or []) if all_seeds else []
    if not seeds:
        return _run_causal(
            selected_raw,
            feature_class=feature_class,
            max_size=max_size,
            lrs=lrs,
        )
    best_seed = row.get("best_seed")
    rows: List[Dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        loaded = seed != best_seed
        seed_raw: Optional[Mapping[str, Any]] = None
        try:
            seed_raw = (
                selected_raw
                if not loaded
                else _reload_seed_for_causal(
                    cfg=cfg,
                    bundle=bundle,
                    feature_class=feature_class,
                    scale=scale,
                    lr=lr,
                    smoke=smoke,
                    experiment_dir=experiment_dir,
                    seed=seed,
                )
            )
            if seed_raw is None:
                seed_rows = [
                    {"class": feature_class, "error": "seed checkpoint unavailable"}
                ]
            else:
                seed_rows = _run_causal(
                    seed_raw,
                    feature_class=feature_class,
                    max_size=max_size,
                    lrs=lrs,
                )
        except Exception as exc:
            seed_rows = [
                {"class": feature_class, "error": f"{type(exc).__name__}: {exc}"}
            ]
        finally:
            if loaded and seed_raw is not None:
                trainer = seed_raw.get("trainer")
                if trainer is not None and hasattr(trainer, "unload_model"):
                    try:
                        trainer.unload_model()
                    except Exception:
                        pass
        for seed_row in seed_rows:
            rows.append({**seed_row, "seed": seed})
    return rows


def _train_cell(
    *,
    cfg: Any,
    bundle: Any,
    task: str,
    feature_class: str,
    mode: str,
    scale: Optional[str],
    lr: float,
    smoke: bool,
    exp_dir: Path,
    max_seeds: Optional[int],
    all_seeds: bool,
    run_causal: bool,
    causal_all_seeds: bool,
    causal_min_score: float,
    causal_max_size: int,
    causal_lrs: Sequence[float],
) -> Dict[str, Any]:
    from study import training_profiles as tp
    from study.stages.train import _train_once

    shared = tp.shared_training_kwargs(cfg, backend="actiend")
    if smoke:
        shared = tp.apply_smoke_training_kwargs(shared)
    shared["learning_rate"] = float(lr)
    if max_seeds is not None:
        shared["max_seeds"] = int(max_seeds)
    if all_seeds:
        # Requiring max_seeds convergences guarantees that every seed runs and
        # preserves a meaningful package convergence verdict (None would run
        # every seed but make convergence_info.converged unconditionally True).
        seed_total = int(shared.get("max_seeds") or 3)
        shared["max_seeds"] = seed_total
        shared["min_convergent_seeds"] = seed_total

    captured: Dict[str, Any] = {}

    exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(
            f"\n=== train model={cfg.model_key} task={task} mode={mode} scale={scale!r} "
            f"lr={lr:.3g} feature={feature_class} dir={exp_dir} ===",
            flush=True,
        )
        with _patched_training_arguments(scale=scale, lr=lr, captured=captured):
            raw = _train_once(
                backend="actiend",
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
        error = None
    except Exception as exc:  # keep the grid alive across a bad cell
        raw = {}
        error = f"{type(exc).__name__}: {exc}"

    if error is None:
        conv = _load_convergence_summary(exp_dir)
    else:
        # Don't let a stale training.json from a prior attempt in this same
        # exp_dir masquerade as this (failed) cell's result.
        conv = {}
    conv_info = conv.get("convergence_info") or {}
    bsc = conv.get("best_score_checkpoint") or {}
    metric = conv_info.get("convergence_metric") or captured.get("convergent_metric")
    score = bsc.get(str(metric)) if metric else None
    threshold = conv_info.get("threshold") or captured.get("convergent_score_threshold")
    score_ok = (
        score is not None and threshold is not None and float(score) >= float(threshold)
    )
    row: Dict[str, Any] = {
        "model": cfg.model_key,
        "task": task,
        "mode": mode,
        "lr": float(lr),
        "feature_class": feature_class,
        "metric": metric,
        "score": score,
        # ``best_score_checkpoint['correlation']`` is populated unconditionally
        # by the package (gradiend/trainer/core/callbacks.py) regardless of
        # which metric actually drove selection -- always available here even
        # though one-pole's selection metric is min_auc_n_o, not correlation.
        "correlation": bsc.get("correlation"),
        "threshold": threshold,
        # Package convergence is the verdict. ``score_ok`` remains a diagnostic
        # that distinguishes failure of discriminative quality from failure of
        # the required target-positive orientation or another convergence gate.
        "converged": conv_info.get("converged"),
        "score_ok": score_ok,
        "best_step": bsc.get("global_step"),
        "error": error,
        **_load_seed_summary(exp_dir),
    }
    if error is None and run_causal and _causal_requested_for_row(row, causal_min_score):
        causal_rows = _run_cell_causal(
            raw,
            row,
            cfg=cfg,
            bundle=bundle,
            feature_class=feature_class,
            scale=scale,
            lr=lr,
            smoke=smoke,
            experiment_dir=exp_dir,
            max_size=causal_max_size,
            lrs=causal_lrs,
            all_seeds=causal_all_seeds,
        )
        row["causal"] = causal_rows
        _write_causal_rows(exp_dir, causal_rows)
    return row


def _cached_row(
    *,
    model: str,
    task: str,
    mode: str,
    lr: float,
    feature_class: str,
    experiment_dir: Path,
) -> Dict[str, Any]:
    conv = _load_convergence_summary(experiment_dir)
    conv_info = conv.get("convergence_info") or {}
    bsc = conv.get("best_score_checkpoint") or {}
    metric = conv_info.get("convergence_metric")
    score = bsc.get(str(metric)) if metric else None
    threshold = conv_info.get("threshold")
    score_ok = (
        score is not None
        and threshold is not None
        and float(score) >= float(threshold)
    )
    row: Dict[str, Any] = {
        "model": model,
        "task": task,
        "mode": mode,
        "lr": float(lr),
        "feature_class": feature_class,
        "metric": metric,
        "score": score,
        "correlation": bsc.get("correlation"),
        "threshold": threshold,
        "converged": conv_info.get("converged"),
        "score_ok": score_ok,
        "best_step": bsc.get("global_step"),
        "error": None,
        "reloaded": True,
        **_load_seed_summary(experiment_dir),
    }
    causal_path = experiment_dir / "causal.json"
    try:
        causal_payload = json.loads(causal_path.read_text(encoding="utf-8"))
        causal_rows = causal_payload.get("rows") if isinstance(causal_payload, dict) else None
        if isinstance(causal_rows, list):
            row["causal"] = causal_rows
    except Exception:
        pass
    return row


def _cache_uses_current_one_pole_data_protocol(experiment_dir: Path) -> bool:
    """Reject v2 sweep cells known to predate the leakage fix/marker."""
    try:
        payload = json.loads((experiment_dir / "done.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    extras = payload.get("extras") if isinstance(payload, dict) else None
    version = extras.get("one_pole_data_protocol_version") if isinstance(extras, dict) else None
    return version == ONE_POLE_DATA_PROTOCOL_VERSION


def _reload_cell_for_causal(
    *,
    cfg: Any,
    bundle: Any,
    feature_class: str,
    scale: Optional[str],
    lr: float,
    smoke: bool,
    experiment_dir: Path,
    max_seeds: Optional[int],
    all_seeds: bool,
) -> Optional[Dict[str, Any]]:
    """Reload a cached cell under the same explicit signal configuration."""
    from study import training_profiles as tp
    from study.stages.train import _reload_train_once

    shared = tp.shared_training_kwargs(cfg, backend="actiend")
    if smoke:
        shared = tp.apply_smoke_training_kwargs(shared)
    shared["learning_rate"] = float(lr)
    if max_seeds is not None:
        shared["max_seeds"] = int(max_seeds)
    if all_seeds:
        seed_total = int(shared.get("max_seeds") or 3)
        shared["max_seeds"] = seed_total
        shared["min_convergent_seeds"] = seed_total
    with _patched_training_arguments(scale=scale, lr=lr):
        return _reload_train_once(
            backend="actiend",
            cfg=cfg,
            bundle=bundle,
            target_classes=[feature_class],
            experiment_dir=experiment_dir,
            split_mode="none",
            shared=shared,
            smoke=smoke,
            feature_class=feature_class,
            counterfactual_classes="all",
            all_classes=list(bundle.classes),
            previous_methods=[],
        )


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "",
        "ACTIEND one-pole LR x activation-scale ablation",
        "",
        f"{'task':<16} {'model':<20} {'mode':<12} {'lr':>10} {'cls':<10} "
        f"{'metric':<12} {'score':>8} {'corr':>7} {'thr':>6} {'s>=thr':<6} "
        f"{'pkg_conv':<8} {'seedok':>7} {'step':>6} err",
        "-" * 140,
    ]
    for r in rows:
        seed_n = r.get("seed_count")
        seed_ok = r.get("seed_convergent_count")
        seed_text = f"{seed_ok}/{seed_n}" if seed_n not in (None, 0) else "-"
        lines.append(
            f"{str(r.get('task') or '-'):<16} "
            f"{str(r.get('model') or '-'):<20} "
            f"{str(r.get('mode') or '-'):<12} "
            f"{_fmt(r.get('lr'), 2):>10} "
            f"{str(r.get('feature_class') or '-'):<10} "
            f"{str(r.get('metric') or '-'):<12} "
            f"{_fmt(r.get('score')):>8} "
            f"{_fmt(r.get('correlation')):>7} "
            f"{_fmt(r.get('threshold'), 2):>6} "
            f"{_fmt(r.get('score_ok')):<6} "
            f"{_fmt(r.get('converged')):<8} "
            f"{seed_text:>7} "
            f"{_fmt(r.get('best_step'), 0):>6} "
            f"{r.get('error') or '-'}"
        )
    lines.append("")
    lines.append(
        "'s>=thr' is the score-threshold diagnostic. 'pkg_conv' is the actual "
        "convergence verdict, including the required positive mean encoding for "
        "the semantic +1 target in AUC-scored one-pole training. 'seedok' counts "
        "fully converged seeds, not score-only passes."
    )
    lines.append("")

    by_case_mode: Dict[Tuple[str, str, str, str], List[float]] = {}
    for r in rows:
        if r.get("converged"):
            key = (
                str(r.get("task")),
                str(r.get("feature_class")),
                str(r.get("model")),
                str(r.get("mode")),
            )
            by_case_mode.setdefault(key, []).append(float(r.get("lr")))
    lines.append("LRs fully converging, per (task, class, model, mode):")
    for key in sorted(by_case_mode):
        task, cls, model, mode = key
        lines.append(f"  {task:<16} {cls:<10} {model:<20} {mode:<12} {sorted(by_case_mode[key])}")
    if not by_case_mode:
        lines.append("  (nothing fully converged in this grid)")
    lines.append("")
    lines.append(
        "Per case: if running_rms clears the threshold at some LR where raw "
        "never does (for either model), that's a real fix for that task, not "
        "just LR-matching. If neither mode clears it, inspect whether the score "
        "is still rising at the upper LR boundary before concluding that LR/scale "
        "cannot fix the task."
    )
    causal_rows = [
        (row, causal)
        for row in rows
        for causal in (row.get("causal") or [])
        if isinstance(causal, Mapping)
    ]
    if causal_rows:
        lines.extend(
            [
                "",
                "Decoder causal evaluation (validation split):",
                f"{'task':<16} {'model':<20} {'mode':<12} {'lr':>10} "
                f"{'seed':>5} {'variant':<14} {'cls':<10} {'alpha*':>10} {'P':>8} {'lms':>8} err",
                "-" * 132,
            ]
        )
        for parent, causal in causal_rows:
            lines.append(
                f"{str(parent.get('task') or '-'):<16} "
                f"{str(parent.get('model') or '-'):<20} "
                f"{str(parent.get('mode') or '-'):<12} "
                f"{_fmt(parent.get('lr'), 2):>10} "
                f"{_fmt(causal.get('seed'), 0):>5} "
                f"{str(causal.get('variant') or '-'):<14} "
                f"{str(causal.get('class') or '-'):<10} "
                f"{_fmt(causal.get('learning_rate'), 3):>10} "
                f"{_fmt(causal.get('value')):>8} "
                f"{_fmt(causal.get('lms')):>8} "
                f"{causal.get('error') or '-'}"
            )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--suite", default="core")
    p.add_argument(
        "--cases",
        nargs="+",
        default=[f"{task}:{cls}" for task, cls in DEFAULT_CASES],
        help="'task:feature_class' entries. Default: the genuine gpt2-small "
        "suite_full one-pole ACTIEND failures (see DEFAULT_CASES).",
    )
    p.add_argument("--modes", nargs="+", default=None, help=f"Subset of {[m[0] for m in MODES]}")
    p.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    p.add_argument(
        "--include-stress-lr",
        action="store_true",
        help=f"Append isolated high-LR stress cells {list(STRESS_LRS)}.",
    )
    p.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Override the package max_seeds value (study default: 3).",
    )
    p.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run all max_seeds for every cell instead of stopping after the first convergence.",
    )
    p.add_argument(
        "--causal",
        action="store_true",
        help="Run ungated and headline-gated decoder causal evaluation on selected cells.",
    )
    p.add_argument(
        "--causal-all-seeds",
        action="store_true",
        help="With --causal, evaluate every trained seed checkpoint instead of only the selected best seed.",
    )
    p.add_argument(
        "--causal-min-score",
        type=float,
        default=0.8,
        help="With --causal, evaluate cells whose checkpoint score is at least this value.",
    )
    p.add_argument(
        "--causal-max-size",
        type=int,
        default=None,
        help="Decoder-eval row cap (default: package/study decoder maximum; smoke: 32).",
    )
    p.add_argument(
        "--causal-lrs",
        nargs="+",
        type=float,
        default=None,
        help="Optional decoder intervention-strength grid; defaults to causal_eval.DEFAULT_CAUSAL_STRENGTHS.",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--output-subdir",
        default="ablation_actiend_lr_scale_v2",
        help="Under runs/<model>/<task>/; v2 avoids stale pre-explicit-raw cells.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If a cell's done.json already exists, read its training.json instead of retraining.",
    )
    return p


def _effective_lrs(lrs: Sequence[float], *, include_stress: bool) -> List[float]:
    values = [float(value) for value in lrs]
    if include_stress:
        values.extend(float(value) for value in STRESS_LRS)
    return sorted(set(values))


def _write_report_shards(
    report_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Path]:
    """Write per-case shards and merge matching concurrent Slurm jobs.

    The recommended launcher submits one job per (model, case).  Unique shards
    prevent those jobs from overwriting one another; whichever job finishes
    last sees every completed shard and leaves a complete comparison.json/txt.
    """
    shard_dir = report_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("model") or "unknown"),
            str(row.get("task") or "unknown"),
            str(row.get("feature_class") or "unknown"),
        )
        grouped.setdefault(key, []).append(dict(row))
    for (model, task, feature_class), group_rows in grouped.items():
        safe = "__".join((model, task, feature_class))
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe)
        target = shard_dir / f"{safe}.json"
        temporary = shard_dir / f".{safe}.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(
                {"protocol": dict(protocol), "rows": group_rows},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    merged_rows: List[Dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("protocol") != dict(protocol):
            continue
        merged_rows.extend(
            dict(row) for row in (payload.get("rows") or []) if isinstance(row, dict)
        )
    merged_rows.sort(
        key=lambda row: (
            str(row.get("model") or ""),
            str(row.get("task") or ""),
            str(row.get("feature_class") or ""),
            str(row.get("mode") or ""),
            float(row.get("lr") or 0.0),
        )
    )
    payload = {"protocol": dict(protocol), "rows": merged_rows}
    json_target = report_dir / "comparison.json"
    json_temporary = report_dir / f".comparison.{os.getpid()}.json.tmp"
    json_temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    os.replace(json_temporary, json_target)
    table_target = report_dir / "comparison.txt"
    table_temporary = report_dir / f".comparison.{os.getpid()}.txt.tmp"
    table_temporary.write_text(format_table(merged_rows) + "\n", encoding="utf-8")
    os.replace(table_temporary, table_target)
    return merged_rows, table_target


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _parser().parse_args(argv)

    from study.config import load_study_config
    from study.registry import build_task

    if args.max_seeds is not None and int(args.max_seeds) < 1:
        raise SystemExit("--max-seeds must be at least 1")
    if args.causal_all_seeds and not args.causal:
        raise SystemExit("--causal-all-seeds requires --causal")
    lrs = _effective_lrs(args.lrs, include_stress=bool(args.include_stress_lr))
    if not lrs or any(lr <= 0 for lr in lrs):
        raise SystemExit("--lrs must contain positive values")
    if args.causal:
        from causal_eval import DEFAULT_CAUSAL_STRENGTHS, DECODER_EVAL_MAX_SIZE

        causal_max_size = int(
            args.causal_max_size
            or (32 if args.smoke else DECODER_EVAL_MAX_SIZE)
        )
        causal_lrs: Sequence[float] = (
            list(args.causal_lrs)
            if args.causal_lrs
            else (
                [10.0**exponent for exponent in range(3, -5, -1)]
                if args.smoke
                else list(DEFAULT_CAUSAL_STRENGTHS)
            )
        )
    else:
        causal_max_size = 0
        causal_lrs = []

    wanted = {str(m) for m in (args.modes or [m[0] for m in MODES])}
    modes = tuple(m for m in MODES if m[0] in wanted)
    if not modes:
        raise SystemExit(f"No modes selected from {wanted}")
    cases = _parse_cases(args.cases)

    all_rows: List[Dict[str, Any]] = []
    for model in args.models:
        for task, feature_class in cases:
            cfg = load_study_config(
                model=model,
                task=task,
                suite=args.suite,
                cli_overrides={"methods": ["actiend"]},
            )
            bundle = build_task(cfg.task_id, cfg.raw, smoke=bool(args.smoke))
            if feature_class not in [str(c) for c in bundle.classes]:
                raise SystemExit(
                    f"case {task}:{feature_class!r} not in task classes {list(bundle.classes)}"
                )
            out_root = cfg.output_dir / str(args.output_subdir)
            out_root.mkdir(parents=True, exist_ok=True)

            for mode, scale in modes:
                for lr in lrs:
                    exp_dir = out_root / mode / f"lr_{lr:.0e}" / f"actiend_onepole_{feature_class}"
                    cache_current = _cache_uses_current_one_pole_data_protocol(exp_dir)
                    if args.skip_existing and (exp_dir / "done.json").exists() and not cache_current:
                        print(
                            f"Ignoring stale one-pole cell without data protocol "
                            f"v{ONE_POLE_DATA_PROTOCOL_VERSION}: {exp_dir}",
                            flush=True,
                        )
                    if args.skip_existing and cache_current:
                        row = _cached_row(
                            model=model,
                            task=task,
                            mode=mode,
                            lr=lr,
                            feature_class=feature_class,
                            experiment_dir=exp_dir,
                        )
                        if (
                            args.causal
                            and "causal" not in row
                            and _causal_requested_for_row(row, args.causal_min_score)
                        ):
                            try:
                                raw = _reload_cell_for_causal(
                                    cfg=cfg,
                                    bundle=bundle,
                                    feature_class=feature_class,
                                    scale=scale,
                                    lr=lr,
                                    smoke=bool(args.smoke),
                                    experiment_dir=exp_dir,
                                    max_seeds=args.max_seeds,
                                    all_seeds=bool(args.all_seeds),
                                )
                                if raw is None:
                                    causal_rows = [
                                        {
                                            "class": feature_class,
                                            "error": "could not reload cached trainer",
                                        }
                                    ]
                                else:
                                    causal_rows = _run_cell_causal(
                                        raw,
                                        row,
                                        cfg=cfg,
                                        bundle=bundle,
                                        feature_class=feature_class,
                                        scale=scale,
                                        lr=lr,
                                        smoke=bool(args.smoke),
                                        experiment_dir=exp_dir,
                                        max_size=causal_max_size,
                                        lrs=causal_lrs,
                                        all_seeds=bool(args.causal_all_seeds),
                                    )
                                row["causal"] = causal_rows
                                _write_causal_rows(exp_dir, causal_rows)
                            except Exception as exc:
                                row["causal"] = [
                                    {
                                        "class": feature_class,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                ]
                        all_rows.append(row)
                        continue
                    all_rows.append(
                        _train_cell(
                            cfg=cfg,
                            bundle=bundle,
                            task=task,
                            feature_class=feature_class,
                            mode=mode,
                            scale=scale,
                            lr=lr,
                            smoke=bool(args.smoke),
                            exp_dir=exp_dir,
                            max_seeds=args.max_seeds,
                            all_seeds=bool(args.all_seeds),
                            run_causal=bool(args.causal),
                            causal_all_seeds=bool(args.causal_all_seeds),
                            causal_min_score=float(args.causal_min_score),
                            causal_max_size=causal_max_size,
                            causal_lrs=causal_lrs,
                        )
                    )

    table = format_table(all_rows)
    print(table)
    report_dir = ROOT / "runs" / f"_{args.output_subdir}"
    report_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "one_pole_data_protocol_version": ONE_POLE_DATA_PROTOCOL_VERSION,
        "suite": args.suite,
        "lrs": lrs,
        "modes": [mode for mode, _scale in modes],
        "max_seeds": args.max_seeds,
        "all_seeds": bool(args.all_seeds),
        "causal": bool(args.causal),
        "causal_all_seeds": bool(args.causal_all_seeds),
        "causal_min_score": float(args.causal_min_score),
        "causal_max_size": causal_max_size if args.causal else None,
        "causal_lrs": list(causal_lrs),
        "smoke": bool(args.smoke),
        "output_subdir": args.output_subdir,
    }
    merged_rows, table_path = _write_report_shards(
        report_dir, all_rows, protocol=protocol
    )
    print(
        f"Wrote {table_path} ({len(merged_rows)} matching shard rows)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
