"""Train GRADIEND / ACTIEND (pair and one-pole ablations)."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from experiment_cache import stable_hash
from results_schema import (
    ENCODER_EVAL_RULES_VERSION,
    encoder_eval_rules_current,
    method_result,
)
from error_tracker import track_error
from study.method_ids import (
    artifact_dirname,
    onepole_feature_id,
    pair_feature_id,
    pair_key,
    pair_trainer_id,
    sorted_pair,
)
from study.tasks import TaskBundle, resolve_mask_placeholder
from study.training_profiles import (
    AGIEND_BACKENDS,
    Backend,
    CAGA_BACKENDS,
    CGA_BACKENDS,
    SplitMode,
    apply_smoke_prune,
    apply_smoke_training_kwargs,
    build_training_arguments,
    is_caga_backend,
    is_cga_backend,
    record_decoder_lr,
    shared_training_kwargs,
)
from study.config import StudyConfig
from study.resume_policy import artifact_has_successful_result


# Training-artifact protocol marker. Version 1 makes AUC-scored encodings
# semantically oriented: numeric label +1 must have mean encoding > 0.
AUC_ORIENTATION_PROTOCOL_VERSION = 1

# Version 2 keeps all factual classes available for decoder evaluation while
# restricting one-pole *training* rows to the sole configured target factual
# class. Unversioned historical artifacts may either predate full decoder
# coverage (valid target-only training) or come from the short-lived leakage
# regression; callers must not infer validity from a missing marker alone.
ONE_POLE_DATA_PROTOCOL_VERSION = 2


def _release_train_raw_trainer(raw: Dict[str, Any]) -> bool:
    """Drop one completed trainer and return its CUDA cache to the allocator.

    A trainer owns the base language model.  Keeping every completed pair and
    one-pole trainer alive while the next artifact trains makes GPU residency
    grow by one full backbone per artifact.  Large-model encode-only screens do
    not have a downstream stage that can use those live trainers, so retaining
    them only causes deterministic late-stage OOMs.

    The remaining ``raw`` payload contains all metrics and artifact metadata
    needed to build result rows and the train-stage checkpoint.
    """
    trainer = raw.pop("trainer", None)
    if trainer is None:
        return False
    del trainer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        # Memory cleanup is best-effort and must not turn a successful artifact
        # into a failed study run.
        track_error(exc, context="release completed trainer")
    return True


def _mark_cga_encoder_host_only(model_with_gradiend: Any) -> None:
    """Keep CGA's inert encoder on CPU while gradients stay on the base GPU."""
    gradiend_model = getattr(model_with_gradiend, "gradiend", None)
    if gradiend_model is None:
        return
    base_device = model_with_gradiend._get_base_forward_device()
    # ``device_encoder`` is also the package's gradient-output device. CGA's
    # encoder module is never called, so retain its parameters on CPU but keep
    # gradient extraction/reduction colocated with the base model.
    gradiend_model.device_encoder = base_device
    setattr(model_with_gradiend, "_cga_encoder_offloaded", True)


def _suite_splits(cfg: StudyConfig, backend: str) -> List[SplitMode]:
    # actiend_pre reuses the actiend split list from the suite YAML; every CGA
    # variant reads the ``cga`` key (whose values are variant names, not splits,
    # so this resolves to the ``["none"]`` fallback by design).
    lookup = "actiend" if str(backend) == "actiend_pre" else backend
    if is_cga_backend(lookup):
        lookup = "cga"
    methods = (cfg.suite.get("methods") or {}).get(lookup) or ["none"]
    out: List[SplitMode] = []
    for m in methods:
        if m in ("none", "tensors") and m not in out:
            out.append(m)  # type: ignore[arg-type]
        if m == "all" and "tensors" not in out:
            out.append("tensors")
    return out or ["none"]


# Study protocol: never train GradiendSplit.by_tensor() for these backends.
# CGA is a single hand-installed direction over the whole scoped parameter
# space; a component split has nothing to split and no training to converge.
_TENSORS_SKIP_BACKENDS = frozenset(
    {"gradiend", "actiend", "actiend_pre"} | set(CGA_BACKENDS) | set(CAGA_BACKENDS)
)

# Opt-in exception (configs/suites/full_plus.yaml): trains by-tensor GRADIEND
# and ACTIEND, kept out of the default `full` suite since it's much slower
# and, per the skip above, has historically been non-convergent on study
# tasks — full_plus is deliberately where that gets found out, not `full`.
# actiend_pre stays excluded even under full_plus (never requested — its own
# suite yaml never lists `tensors` as a candidate split for it).
_TENSORS_ALLOWED_SUITES = {
    "actiend": frozenset({"full_plus"}),
    "gradiend": frozenset({"full_plus"}),
}


def _effective_train_splits(
    cfg: StudyConfig, backend: str, *, one_pole: bool
) -> List[SplitMode]:
    """Suite splits after study protocol filters.

    - **GRADIEND / ACTIEND / actiend_pre:** never ``by_tensor`` (component-split is
      much slower and rarely converges on study tasks; scalar ``none`` is headline)
      — except GRADIEND/ACTIEND under ``_TENSORS_ALLOWED_SUITES`` (currently
      ``full_plus``).
    - **One-pole (any other backend):** never ``by_tensor``.
    """
    splits = _suite_splits(cfg, backend)
    suite_id = str(cfg.suite.get("id") or "")
    out: List[SplitMode] = []
    for mode in splits:
        if mode == "tensors":
            if backend in _TENSORS_SKIP_BACKENDS and suite_id not in _TENSORS_ALLOWED_SUITES.get(backend, frozenset()):
                continue
            if one_pole:
                continue
        out.append(mode)
    return out or ["none"]


def _requested_train_splits(
    eligible: Sequence[SplitMode], requested: Optional[Sequence[str]]
) -> List[SplitMode]:
    """Apply an operator split filter without inventing an ineligible split."""
    if not requested:
        return list(eligible)
    normalized: List[SplitMode] = []
    for raw in requested:
        value = str(raw).strip().lower()
        if value in {"tensor", "by_tensor"}:
            value = "tensors"
        if value not in {"none", "tensors"}:
            raise ValueError(
                f"Unknown train split {raw!r}; expected 'none' or 'tensors'"
            )
        if value not in normalized:
            normalized.append(value)  # type: ignore[arg-type]
    return [split for split in eligible if split in normalized]


def _actiend_pre_enabled(cfg: StudyConfig) -> bool:
    """Opt-in mixed-site ACTIEND-PRE (``training.actiend_pre: true``). Off by default."""
    t = cfg.training or {}
    return bool(t.get("actiend_pre") or t.get("train_actiend_pre"))


def train_ablations_filter(cfg: StudyConfig) -> Optional[frozenset]:
    """Requested ablation slice from ``--train-ablations``, or ``None`` (= all).

    Raises on an unknown name (fail fast: a typo must not silently train nothing
    or everything).
    """
    raw = ((getattr(cfg, "raw", None) or {}).get("cli") or {}).get("train_ablations")
    from study.config import normalize_train_ablations

    return normalize_train_ablations(raw)


def _ablations(cfg: StudyConfig, *, smoke: bool = False) -> Dict[str, bool]:
    abl = dict(cfg.raw.get("ablations") or {})
    pair = bool(abl.get("pair", True))
    one_pole = bool(abl.get("one_pole", True))
    # Smoke: one trainer is enough to exercise the pipeline — drop one_pole in
    # favor of pair, UNLESS the task has no pair trainer at all (pair: false,
    # e.g. ioi/induction/key_value/function_composition/ioi_mib/repetition/
    # race_one_pole/religion_one_pole), in which case dropping one_pole too
    # would train nothing at all (confirmed 2026-08-18: smoke ioi produced
    # methods=0 and SAE fell back to the untrained HF backbone).
    if smoke and pair:
        one_pole = False
    # ``--train-ablations``: run only a slice (e.g. pair) of this task's trainers so
    # pair and one-pole can occupy separate GPUs. Applied last, and only ever
    # narrows; it never changes ``cfg.raw["ablations"]`` (config hash, CAA/causal
    # claim-class logic and result provenance keep the full task definition).
    only = train_ablations_filter(cfg)
    if only is not None:
        pair = pair and "pair" in only
        one_pole = one_pole and "one_pole" in only
    return {"pair": pair, "one_pole": one_pole}


def _layerwise_method_enabled(cfg: StudyConfig, family: str) -> bool:
    """Whether ``methods.<family>`` requests checkpoint-derived layer rows."""
    requested = (getattr(cfg, "suite", {}).get("methods") or {}).get(str(family)) or []
    if isinstance(requested, str):
        requested = [requested]
    return "layers" in {str(item).strip().lower() for item in requested}


def one_pole_train_source(shared: Mapping[str, Any], *, is_one_pole: bool) -> Optional[str]:
    """Training ``source`` for an ablation. One-pole is always ``"both"``.

    ``source="alternative"`` only extracts the signal on the *alternative*
    (rival) prediction; for a one-pole model that collapses every rival into a
    single incoherent -1 pole. ``"both"`` alternates factual vs alternative per
    prompt (IOI-style), a coherent bipolar target. Measured on identical data:
    race one-pole ACTIEND AUC_o 0.59 (alternative) -> 0.98 (both); religion
    0.69 -> 1.00 -- the effect the race_one_pole/religion_one_pole check-tasks
    were built to expose, now folded into the default. Pairs keep the task's own
    source; a one-pole task that already sets ``"both"`` is unchanged.
    """
    if is_one_pole:
        return "both"
    return shared.get("source")


def _task_one_pole_classes(cfg: StudyConfig) -> Optional[List[str]]:
    """Optional study-level filter: which one-pole features to train.

    Not a package API. When unset, every task class gets a one-pole run.
    """
    raw = cfg.training.get("one_pole_classes")
    if not raw:
        return None
    out = [str(c) for c in raw]
    return out or None


def _merged_one_pole_frame(bundle: TaskBundle) -> Optional[pd.DataFrame]:
    """Return merged IOI-style frame with explicit alternative columns, else None.

    Both ``label_class`` and ``alternative_class`` must be task classes (e.g. IO/SUBJECT
    or en/fr). Multilingual foil tags (OTHER/de/…) are not valid one-pole partners —
    those frames fall back to ``data_per_class``.
    """
    df = bundle.merged_df
    if df is None or getattr(df, "empty", True):
        return None
    need = {"masked", "label", "label_class", "alternative", "alternative_class", "split"}
    if not need.issubset(set(df.columns)):
        return None
    classes = {str(c) for c in (bundle.classes or [])}
    if classes:
        fac = set(df["label_class"].astype(str).unique())
        alt = set(df["alternative_class"].astype(str).unique())
        if not fac.issubset(classes) or not alt.issubset(classes):
            return None
    return df


def _training_preview_frame(
    data: Any,
    *,
    target_classes: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Normalize trainer ``data`` (merged DF or per-class map) into one preview frame."""
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        parts: List[pd.DataFrame] = []
        for cls, frame in data.items():
            if frame is None or getattr(frame, "empty", True):
                continue
            part = frame.copy()
            if "label_class" not in part.columns:
                part["label_class"] = str(cls)
            if "label" not in part.columns and str(cls) in part.columns:
                part["label"] = part[str(cls)]
            parts.append(part)
        if not parts:
            return None
        df = pd.concat(parts, ignore_index=True)
    else:
        return None
    if target_classes and "label_class" in df.columns:
        wanted = {str(c) for c in target_classes}
        filtered = df[df["label_class"].astype(str).isin(wanted)]
        if not filtered.empty:
            df = filtered
    return df


def _print_data_examples(
    data: Any,
    *,
    task_id: str,
    target_classes: Optional[Sequence[str]] = None,
    n: int = 3,
) -> None:
    """Always show a few training rows (pair or one-pole; merged or per-class)."""
    df = _training_preview_frame(data, target_classes=target_classes)
    if df is None or getattr(df, "empty", True):
        print(f"{task_id} rows: (no preview data)", flush=True)
        return
    print(f"{task_id} rows: {len(df)}", flush=True)
    if "pattern" in df.columns:
        print(f"  patterns: {df['pattern'].value_counts().to_dict()}", flush=True)
    if "split" in df.columns:
        print(f"  splits: {df['split'].value_counts().to_dict()}", flush=True)
    if "label_class" in df.columns:
        print(
            f"  label_classes: {df['label_class'].astype(str).value_counts().to_dict()}",
            flush=True,
        )
    if {"label_class", "alternative_class"}.issubset(df.columns):
        transitions = (
            df["label_class"].astype(str) + "->" + df["alternative_class"].astype(str)
        ).unique().tolist()
        print(f"  transitions: {transitions}", flush=True)
    print("  examples:", flush=True)
    for _, row in df.head(n).iterrows():
        pat = row.get("pattern", "?")
        split = row.get("split", "?")
        print(f"    [{pat}/{split}] {row.get('masked')}", flush=True)
        alt = row.get("alternative")
        if alt is not None and not (isinstance(alt, float) and pd.isna(alt)):
            print(
                f"      label={row.get('label')!r}  alternative={alt!r}",
                flush=True,
            )
        else:
            print(f"      label={row.get('label')!r}", flush=True)


def _assert_still_one_pole(
    trainer,
    *,
    factual_class: str,
    partner_classes: Sequence[str],
) -> None:
    """Confirm one-pole training is scoped to exactly {factual_class} ∪ partner_classes.

    ``trainer.combined_data`` deliberately retains all task factual classes for
    decoder evaluation. ``gradiend.create_training_data`` applies the actual
    one-pole boundary non-destructively: only the configured target factual
    class enters training, while counterfactual classes provide target tokens,
    not additional factual data. This check therefore validates the retained
    evaluation population; package tests validate the narrower training frame.
    """
    from gradiend.trainer.core.unified_schema import (
        UNIFIED_FACTUAL_CLASS,
        UNIFIED_SPLIT,
        UNIFIED_TRANSITION,
    )

    trainer._ensure_data()
    combined = trainer.combined_data
    if combined is None:
        raise RuntimeError("expected combined_data after one-pole setup")
    fac = set(combined[UNIFIED_FACTUAL_CLASS].astype(str).unique())
    expected_fac = {str(factual_class)} | {str(c) for c in partner_classes}
    if str(factual_class) not in fac or not fac.issubset(expected_fac):
        raise RuntimeError(
            f"Expected factual classes within {{{factual_class}}} ∪ "
            f"{sorted(str(c) for c in partner_classes)}, got factual={fac}"
        )
    transition_counts = {
        str(key).replace("→", "->"): int(value)
        for key, value in combined[UNIFIED_TRANSITION].value_counts().items()
    }
    print(
        f"  confirmed one-pole scope ({factual_class} vs {sorted(partner_classes)}): "
        f"{len(combined)} rows, "
        f"splits={combined[UNIFIED_SPLIT].value_counts().astype(int).to_dict()}, "
        f"transitions={transition_counts}",
        flush=True,
    )




def _fair_encoder_eval(
    trainer,
    *,
    target_classes: Sequence[str],
    max_size: Optional[int],
    n_bootstrap: int,
    backend: Optional[str] = None,
    excluded_words: Optional[Sequence[str]] = None,
    activation_site: Optional[str] = None,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Val-fit / test-report encoder eval (Youden, polarity, ACTIEND shared tokens).

    ``use_cache`` reuses the package's cached ``encoded_values_*.csv``; only the
    reload of an unchanged checkpoint may set it (see ``fair_encoder_eval_from_trainer``).
    """
    from sae_eval import fair_encoder_eval_from_trainer

    classes = [str(c) for c in target_classes]
    if len(classes) < 1:
        raise ValueError("target_classes must be non-empty for encoder eval")

    fair = fair_encoder_eval_from_trainer(
        trainer,
        target_classes=classes,
        max_size=max_size,
        n_bootstrap=int(n_bootstrap),
        backend=backend,
        excluded_words=excluded_words,
        activation_site=activation_site,
        use_cache=use_cache,
    )
    package_metrics = fair.get("encoder_metrics") or {}
    per_class = fair.get("per_class_readouts") or {}
    corr = package_metrics.get("correlation")
    means = package_metrics.get("mean_by_class") or {}
    print(f"  encoder correlation: {corr}", flush=True)
    print(f"  mean encoded by label (+1/-1): {means}", flush=True)
    for cls, rd in per_class.items():
        if rd.get("error"):
            print(f"  {cls} vs neutral: error={rd['error']}", flush=True)
        else:
            print(
                f"  {cls} vs neutral: auc={rd.get('roc_auc')} "
                f"spec={rd.get('specificity')} J={rd.get('youden_j')} "
                f"tau_src={rd.get('youden_threshold_source')}",
                flush=True,
            )
    return fair


@dataclass
class _BuiltTrainer:
    """A constructed (not yet fitted) trainer plus what callers need after."""

    trainer: Any
    args: Any
    shared_kw: Dict[str, Any]
    is_one_pole: bool
    use_merged: bool
    feature_class: Optional[str]


def _build_trainer(
    *,
    backend: Backend,
    cfg: StudyConfig,
    bundle: TaskBundle,
    target_classes: Sequence[str],
    experiment_dir: Path,
    split_mode: SplitMode,
    shared: Dict[str, Any],
    smoke: bool,
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
    all_classes: Optional[Sequence[str]] = None,
) -> "_BuiltTrainer":
    """Build the trainer for one ablation, up to but not including fitting.

    Shared by ``_train_once`` (GRADIEND/ACTIEND, which then calls
    ``trainer.train()``) and ``_fit_cga_once`` (CGA, which instead streams
    gradients into a closed-form direction). Both must see the identical data
    frame, config, one-pole scoping and integrity assertion — a second copy of
    this setup is exactly how the two paths would silently drift apart.
    """
    # Import the concrete trainer module directly. The package-level lazy
    # facade deliberately wraps every underlying exception in a generic
    # "optional runtime dependency" ImportError, which made cluster failures
    # impossible to diagnose and can itself be sensitive to partial imports.
    from gradiend.trainer.text.prediction.trainer import (
        TextPredictionConfig,
        TextPredictionTrainer,
    )

    shared_kw = dict(shared)
    is_one_pole = feature_class is not None or counterfactual_classes is not None
    # One-pole training always uses source="both", regardless of the task's
    # default. source="alternative" only ever extracts the signal on the
    # *alternative* (rival) prediction, which for a one-pole model collapses
    # every rival into one incoherent -1 pole; "both" alternates factual vs
    # alternative per prompt (IOI-style), giving a coherent bipolar target.
    # Measured on identical data: race one-pole ACTIEND AUC_o 0.59 (alternative)
    # -> 0.98 (both); religion 0.69 -> 1.00. Established by the
    # race_one_pole/religion_one_pole check-tasks, now folded into the rule so
    # every one-pole task gets it (pairs keep their task source). Tasks that
    # already set source="both" (the circuit/ravel tasks) are unaffected.
    shared_kw["source"] = one_pole_train_source(shared_kw, is_one_pole=is_one_pole)
    # Prefer clean one-pole: target_classes=[A] + counterfactual_classes (no factual_classes).
    if is_one_pole and feature_class is None and len(target_classes) == 1:
        feature_class = str(target_classes[0])

    args = build_training_arguments(
        backend,
        experiment_dir=str(experiment_dir),
        shared=shared_kw,
        split_mode=split_mode,
        one_pole=is_one_pole,
        metadata={
            "model_key": cfg.model_key,
            "arch": cfg.raw.get("model", {}).get("arch"),
            "task": cfg.task_id,
            "feature_class": feature_class,
            "counterfactual_classes": counterfactual_classes,
        },
    )
    # One-pole GRADIEND used to disable pre-prune here, because the package
    # stratified on feature_class labels and raised when the train frame carried
    # no matching multi-class ids. That workaround was far worse than the bug it
    # avoided: lazy_init is gated on pre_prune_config being set, so disabling it
    # built the encoder at FULL width -- 26 GiB at 8B instead of ~280 MB pruned,
    # which is why GRADIEND could not run the `none` split at 8B at all.
    # The package now falls back to the classes actually present (see
    # gradiend/trainer/core/pruning.py::_stratified_indices), so one-pole keeps
    # its pruning.
    if smoke:
        apply_smoke_prune(args)

    merged = _merged_one_pole_frame(bundle)
    use_merged = merged is not None and is_one_pole
    # Prefer full merged_df for previews (keeps alternative_*); else training data.
    preview_data: Any = (
        bundle.merged_df
        if bundle.merged_df is not None and not getattr(bundle.merged_df, "empty", True)
        else (merged if use_merged else bundle.data_per_class)
    )
    _print_data_examples(
        preview_data,
        task_id=cfg.task_id,
        target_classes=target_classes,
    )
    config_kw: Dict[str, Any] = {
        "run_id": experiment_dir.name,
        "target_classes": list(target_classes),
        "all_classes": list(all_classes or bundle.classes),
        "masked_col": "masked",
        "split_col": "split",
        "neutral_data": bundle.neutrals,
        "decoder_eval_export_row_wise_csv": True,
        # PoC default; keep plots as PNG.
        "img_format": "png",
        "mask_placeholder": resolve_mask_placeholder(bundle, cfg.raw),
    }
    if use_merged:
        # PoC path: one dataframe with paired alternative columns.
        config_kw["data"] = merged
        config_kw["label_col"] = "label"
        config_kw["label_class_col"] = "label_class"
        config_kw["alternative_col"] = "alternative"
        config_kw["alternative_class_col"] = "alternative_class"
        config_kw["decoder_eval_targets"] = "label"
        config_kw["decoder_eval_prob_on_other_class"] = False
    else:
        config_kw["data"] = bundle.data_per_class
    if counterfactual_classes is not None:
        config_kw["counterfactual_classes"] = counterfactual_classes
    if bundle.class_merge_map:
        config_kw["class_merge_map"] = bundle.class_merge_map
    if bundle.class_merge_transition_groups:
        config_kw["class_merge_transition_groups"] = bundle.class_merge_transition_groups

    config = TextPredictionConfig(**config_kw)
    trainer = TextPredictionTrainer(
        model=cfg.hf_model,
        config=config,
        args=args,
        eval_neutral_additional_excluded_words=bundle.excluded_words or None,
    )
    if use_merged and feature_class:
        partners = [c for c in (all_classes or bundle.classes) if str(c) != str(feature_class)]
        if not partners:
            partners = [c for c in target_classes if str(c) != str(feature_class)]
        if partners:
            print(
                f"\n=== Training source={shared_kw.get('source')!r} on "
                f"{feature_class}-only factual {cfg.task_id} data ===",
                flush=True,
            )
            _assert_still_one_pole(
                trainer,
                factual_class=str(feature_class),
                partner_classes=partners,
            )
    return _BuiltTrainer(
        trainer=trainer,
        args=args,
        shared_kw=shared_kw,
        is_one_pole=is_one_pole,
        use_merged=use_merged,
        feature_class=feature_class,
    )


# Backends whose learning rate is searched by ``--tune-lr``: the learned encoder-decoders. CGA/CAGA are
# closed-form (no LR) and actiend_pre reuses ACTIEND's configured rate.
TUNABLE_LR_BACKENDS = frozenset({"gradiend", "actiend", "agiend"})
LR_PROBE_STEPS = 150


def tune_lr_enabled(cfg: StudyConfig) -> bool:
    """``--tune-lr`` (Slurm: ``TUNE_LR=1``): an execution switch, recorded under ``cli``, not hashed."""
    return bool(((getattr(cfg, "raw", None) or {}).get("cli") or {}).get("tune_lr"))


def _load_completed_lr_search(
    experiment_dir: Path, *, backend: str, initial_lr: float, probe_steps: int
) -> Optional[Dict[str, Any]]:
    """A converged ``lr_search.json`` for the same backend, first guess and probe length, else ``None``.

    A job that was preempted (or resubmitted) after the search but before the artifact was finished
    must not pay for the search again. Only a converged verdict is reused: a failed search is re-run,
    since whoever resubmits it usually changed something. Delete the file to force a new search.
    """
    path = experiment_dir / "lr_search.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        saved.get("status") == "converged"
        and saved.get("best_lr") is not None
        and saved.get("backend") == backend
        and saved.get("probe_steps") == probe_steps
        and saved.get("initial_lr") is not None
        and abs(float(saved["initial_lr"]) - initial_lr) <= 1e-9 * abs(initial_lr)
    ):
        return {
            "status": "converged",
            "initial_lr": float(saved["initial_lr"]),
            "best_lr": float(saved["best_lr"]),
            "window": saved.get("window"),
            "bracket": saved.get("bracket"),
            "probe_steps": probe_steps,
            "probes": [
                {"lr": p["lr"], "state": p["state"], "steps_run": p.get("steps_run"), "confirmed": p.get("confirmed")}
                for p in saved.get("probes", [])
            ],
        }
    return None


def _tune_lr_for_training(
    trainer: Any,
    args: Any,
    *,
    backend: str,
    experiment_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Search the learning rate with the package's LR finder; return ``(train_overrides, summary)``.

    The yaml rate is the first guess. Short probes (``LR_PROBE_STEPS`` steps, never more than the
    configured budget) are classified and bracketed; a converged probe is confirmed at the full
    ``max_steps`` (skipped when the probe already is the full run). The search trace is written to
    ``<experiment_dir>/lr_search.json`` and summarised in ``done.json``. Fails fast when no learning
    rate converges: training at an unconverged LR would only produce a row that says so.
    """
    from gradiend.trainer.core.lr_search import NoConvergentLearningRate, tune_learning_rate

    full_steps = int(getattr(args, "max_steps", 0) or 0)
    probe_steps = min(LR_PROBE_STEPS, full_steps) if full_steps > 0 else LR_PROBE_STEPS
    initial = float(args.learning_rate)
    reused = _load_completed_lr_search(experiment_dir, backend=backend, initial_lr=initial, probe_steps=probe_steps)
    if reused is not None:
        print(f"  tune-lr {backend}: reusing the finished search in {experiment_dir / 'lr_search.json'} "
              f"(chose {reused['best_lr']:g})", flush=True)
        return {"learning_rate": float(reused["best_lr"])}, {**reused, "reused": True}
    print(
        f"  tune-lr {backend} {experiment_dir.name}: first guess {initial:g}, probes of {probe_steps} steps",
        flush=True,
    )
    result = tune_learning_rate(
        trainer, initial, experiment_dir=str(experiment_dir), probe_steps=probe_steps,
        metadata={"backend": backend, "probe_steps": probe_steps},
    )
    summary = {
        "status": result.status,
        "initial_lr": initial,
        "best_lr": result.best_lr,
        "window": list(result.window) if result.window else None,
        "bracket": list(result.bracket) if result.bracket else None,
        "probe_steps": probe_steps,
        "probes": [
            {"lr": p.lr, "state": p.state.value, "steps_run": p.steps_run, "confirmed": p.confirmed}
            for p in result.probes
        ],
    }
    if result.status != "converged":
        raise NoConvergentLearningRate(
            f"LR search for {backend} ({experiment_dir.name}) ended '{result.status}' after "
            f"{len(result.probes)} probes (first guess {initial:g}); trace: {experiment_dir / 'lr_search.json'}. "
            "No learning rate converged, so training would only reproduce a non-convergent row; "
            "look at the signal/targets instead of the learning rate.",
            result,
        )
    print(f"  tune-lr {backend}: chose {result.best_lr:g} (window {summary['window']})", flush=True)
    return {"learning_rate": float(result.best_lr)}, summary


def _train_once(
    *,
    backend: Backend,
    cfg: StudyConfig,
    bundle: TaskBundle,
    target_classes: Sequence[str],
    experiment_dir: Path,
    split_mode: SplitMode,
    shared: Dict[str, Any],
    smoke: bool,
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
    all_classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    built = _build_trainer(
        backend=backend,
        cfg=cfg,
        bundle=bundle,
        target_classes=target_classes,
        experiment_dir=experiment_dir,
        split_mode=split_mode,
        shared=shared,
        smoke=smoke,
        feature_class=feature_class,
        counterfactual_classes=counterfactual_classes,
        all_classes=all_classes,
    )
    trainer = built.trainer
    args = built.args
    shared_kw = built.shared_kw
    is_one_pole = built.is_one_pole
    use_merged = built.use_merged
    feature_class = built.feature_class
    from cost_timer import cost_timer
    from gradiend.trainer.core.signal_checks import SignalDiversityCallback

    train_overrides: Dict[str, Any] = {}
    lr_search_summary: Optional[Dict[str, Any]] = None
    if tune_lr_enabled(cfg) and backend in TUNABLE_LR_BACKENDS:
        with cost_timer(f"tune_lr:{backend}:{experiment_dir.name}", phase="train", backend=backend):
            train_overrides, lr_search_summary = _tune_lr_for_training(
                trainer, args, backend=backend, experiment_dir=experiment_dir
            )
    with cost_timer(f"train:{backend}:{experiment_dir.name}", phase="train", backend=backend):
        # Free guard (reads the per-class spread every evaluation already computes): identical inputs
        # within a class -- e.g. labels on padding -- abort at step 0 instead of "converging" on nothing.
        trainer.train(callbacks=[SignalDiversityCallback()], **train_overrides)
    try:
        # PoC: plot_training_convergence(); study uses IQR band like gender.
        trainer.plot_training_convergence(class_spread="iqr")
    except Exception as exc:
        track_error(exc, context="plot_training_convergence")
    # ``shared_kw`` contains scale/smoke overrides. Reading only cfg.training
    # here silently ignored those caps and made smoke runs perform full evals.
    max_size = shared_kw.get(
        "encoder_eval_max_size", (cfg.training or {}).get("encoder_eval_max_size")
    )
    n_boot = int(
        shared_kw.get(
            "encoder_bootstrap_auc",
            (cfg.training or {}).get("encoder_bootstrap_auc") or 100,
        )
    )
    with cost_timer(f"encoder_eval:{backend}:{experiment_dir.name}", phase="encode", backend=backend):
        fair = _fair_encoder_eval(
            trainer,
            # Fair metrics need the full task class set (vs neutral + vs rivals).
            target_classes=list(all_classes or bundle.classes or target_classes),
            max_size=int(max_size) if max_size is not None else None,
            n_bootstrap=n_boot,
            backend=backend,
            excluded_words=list(bundle.excluded_words or []),
            activation_site=(cfg.training or {}).get("activation_site"),
        )
    out = {
        "trainer": trainer,
        "encoder_metrics": fair.get("encoder_metrics") or {},
        "readout_metrics": fair.get("readout_metrics") or {},
        "per_class_readouts": fair.get("per_class_readouts") or {},
        "per_component_readouts": fair.get("per_component_readouts") or {},
        "component_keys": fair.get("component_keys") or [],
        "model_path": str(experiment_dir),
        "backend": backend,
        "split_mode": split_mode,
        "target_classes": list(target_classes),
        "counterfactual_classes": counterfactual_classes,
        "data_mode": "merged_one_pole" if use_merged else "data_per_class",
        "ablation": "one_pole" if is_one_pole else "pair",
        "feature_class": feature_class,
        "learning_rate": float(
            train_overrides.get("learning_rate", getattr(args, "learning_rate", float("nan")))
        ),
        "lr_search": lr_search_summary,
        "learning_rate_decoder": record_decoder_lr(
            getattr(args, "learning_rate_decoder", None)
        ),
        "convergent_metric": getattr(args, "convergent_metric", None),
        "selection_metric": getattr(args, "selection_metric", None),
    }
    try:
        _mark_train_artifact_done(
            experiment_dir,
            config_hash=_train_artifact_hash(
                cfg,
                backend=backend,
                shared=shared_kw,
                split_mode=split_mode,
                target_classes=target_classes,
                feature_class=feature_class,
                counterfactual_classes=counterfactual_classes,
            ),
            extras={
                "ablation": out["ablation"],
                "learning_rate": out["learning_rate"],
                **({"lr_search": out["lr_search"]} if out.get("lr_search") else {}),
                **(
                    {"learning_rate_decoder": out["learning_rate_decoder"]}
                    if out.get("learning_rate_decoder") is not None
                    else {}
                ),
                "convergent_metric": out.get("convergent_metric"),
                "selection_metric": out.get("selection_metric"),
                "actiend_signal_scale": (getattr(args, "metadata", None) or {}).get(
                    "actiend_signal_scale"
                ),
                "auc_orientation_protocol_version": AUC_ORIENTATION_PROTOCOL_VERSION,
                # Which decoder-only label token the training items supervised.
                # Reloads read this back (absent = "legacy"), so an artifact is
                # never re-interpreted under a newer convention.
                "label_token_protocol": getattr(args, "label_token_protocol", "legacy"),
                "one_pole_data_protocol_version": (
                    ONE_POLE_DATA_PROTOCOL_VERSION if is_one_pole else None
                ),
                "collapsed_encoder": _any_collapsed_readout(out),
                "encoder_eval": _encoder_eval_payload(out),
                **_encoder_eval_rules_stamp(out),
                **_load_convergence_summary(experiment_dir),
            },
        )
    except Exception as exc:
        track_error(exc, context="write done.json")
    return out


def _fit_cga_once(
    *,
    backend: str,
    cfg: StudyConfig,
    bundle: TaskBundle,
    target_classes: Sequence[str],
    experiment_dir: Path,
    split_mode: SplitMode,
    shared: Dict[str, Any],
    smoke: bool,
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
    all_classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fit one CGA direction in closed form and evaluate it like a checkpoint.

    Mirrors :func:`_train_once` except that ``trainer.train()`` is replaced by
    a streaming mean of the paired factual/counterfactual gradients
    (``cga_eval.fit_cga_direction``), and encoder evaluation is CAA-style
    cosine scoring (``cga_eval.fair_cga_encoder_eval``) rather than the
    package's own ``trainer.evaluate_encoder()`` -- see ``cga_eval.py``'s
    module docstring for why that distinction matters (tanh-through-the-
    package-encoder does not discard per-row gradient magnitude the way cosine
    does, reintroducing exactly the confound the estimator is supposed to
    remove).

    The fitted direction is still installed into the trainer's own
    ``latent_dim=1`` GRADIEND *decoder* (never the encoder) and saved as an
    ordinary checkpoint, so the causal strength sweep and reload path treat
    CGA exactly like GRADIEND -- confirmed against the package that
    ``evaluate_decoder``/the causal grid never call ``.encode()``, only
    ``rewrite_base_model``/the decoder.
    """
    from cga_eval import (
        CGA_ENCODER_EVAL_VERSION,
        CGA_LAYERWISE_ENCODER_EVAL_VERSION,
        cga_layer_slices,
        cga_variant_from_backend,
        fair_cga_encoder_eval,
        fit_cga_direction,
        install_cga_direction,
    )
    from study.tasks import labeled_df_for_eval

    variant = cga_variant_from_backend(backend)
    built = _build_trainer(
        backend=backend,  # type: ignore[arg-type]
        cfg=cfg,
        bundle=bundle,
        target_classes=target_classes,
        experiment_dir=experiment_dir,
        split_mode=split_mode,
        shared=shared,
        smoke=smoke,
        feature_class=feature_class,
        counterfactual_classes=counterfactual_classes,
        all_classes=all_classes,
    )
    trainer = built.trainer
    args = built.args
    shared_kw = built.shared_kw
    is_one_pole = built.is_one_pole
    use_merged = built.use_merged
    feature_class = built.feature_class

    # Closed-form CGA bypasses ``trainer.train()``, the normal owner of lazy
    # pre-pruning.  Execute the same streaming projection explicitly before
    # constructing CGA's mean direction.
    #
    # This step was previously untimed: pre-pruning needs at least one
    # full-width (unpruned) gradient to decide which coordinates to keep,
    # so it is exactly the kind of step that could transiently need the
    # ~26 GiB full-model tensor even when the eventual pruned output is
    # tiny (~70M dims at topk=0.01) -- and it runs entirely before any
    # `[cost] cga_fit:...`/`[cost] train:gradiend:...` timer starts, so its
    # own peak was invisible in every log this investigation used. Timing
    # it here (study-side, no package change) closes that blind spot for
    # both CGA and (via the identical mechanism inside `trainer.train()`
    # for GRADIEND/ACTIEND -- not separately timed there either, a
    # follow-up worth doing) future OOM triage.
    if getattr(args, "pre_prune_config", None) is not None:
        from cost_timer import cost_timer

        with cost_timer(
            f"pre_prune:{backend}:{experiment_dir.name}", phase="feature_select", backend=backend
        ):
            trainer.pre_prune(inplace=True)

    # Fresh (randomly initialized) encoder-decoder over the scoped parameter
    # space; every weight in it is overwritten below.
    # Construct the inert encoder on CPU from the outset. Building it on CUDA
    # and moving it afterward still creates a 26 GiB peak allocation at 8B;
    # checkpoint reload can additionally hold initialization/load temporaries.
    mwg = trainer.get_model(device_encoder="cpu")
    _mark_cga_encoder_host_only(mwg)
    if getattr(mwg, "feature_class_encoding_direction", None) is None:
        # Which class the +1 pole means. ``get_model()`` only restores this from
        # a saved checkpoint's context file, and there is no checkpoint yet, so
        # take it from the trainer's own labels -- the same labels the fit
        # weights each row by, so the sign convention cannot disagree.
        # ``set_feature_class_encoding_direction`` is set-once and a no-op when
        # the package already resolved it.
        labels_fn = getattr(trainer, "get_feature_class_encoding_labels", None)
        class_labels = labels_fn() if callable(labels_fn) else None
        if class_labels:
            mwg.set_feature_class_encoding_direction(class_labels)

    # CGA never calls the encoder: fitting accumulates the closed-form mean
    # directly into the decoder, encoder metrics use cosine(g, direction), and
    # causal evaluation decodes the supplied feature factor. Keeping the inert
    # full-width encoder on CUDA costs ~26 GiB at 8B and prevents the package
    # from materializing a paired gradient. Offload it only when it is itself a
    # large allocation; small models keep the faster all-GPU path.  Do not
    # change ``device_encoder`` (gradient extraction stays on the model GPU).
    cga_model = getattr(mwg, "gradiend", None)
    cga_encoder = getattr(cga_model, "encoder", None)
    if cga_encoder is not None:
        try:
            import gc
            import torch

            encoder_bytes = sum(
                p.numel() * p.element_size() for p in cga_encoder.parameters()
            )
            # 2 GiB cleanly separates the 8B full-width encoder (~26 GiB) from
            # development-model encoders, without relying on model-name lists.
            if encoder_bytes >= 2 * 1024**3:
                cga_encoder.to("cpu")
                setattr(mwg, "_cga_encoder_offloaded", True)
                # Drop stale tensor/container references before returning the
                # CUDA caching allocator's old 26 GiB segment to the driver.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print(
                    "  cga fit: offloaded inert encoder to cpu "
                    f"({encoder_bytes / 1024**3:.2f} GiB) and released CUDA cache",
                    flush=True,
                )
        except Exception as exc:
            print(f"  cga fit: inert encoder offload failed ({exc}); continuing", flush=True)

    from cost_timer import cost_timer

    accumulate_device = (cfg.training or {}).get("cga_accumulate_device")
    # NOT train_max_size (a package TrainingArguments field GRADIEND/ACTIEND
    # never actually forward here either -- they're bounded by max_steps
    # instead, regardless of dataset size). CGA's fit is a single streaming
    # pass with no step budget, so without its own cap it silently scans the
    # entire split -- confirmed live on gender_en's 74,060-row train split
    # (2026-08-26 first real cluster run), where SCALE=small didn't help
    # because that profile only bounds causal/encoder-eval sizes, not this.
    fit_max_size = (cfg.training or {}).get("cga_max_size")
    with cost_timer(
        f"cga_fit:{backend}:{experiment_dir.name}", phase="train", backend=backend
    ):
        fit = fit_cga_direction(
            trainer,
            mwg,
            variant=variant,
            split="train",
            max_size=int(fit_max_size) if fit_max_size is not None else None,
            accumulate_device=accumulate_device,
            output_dir=cfg.output_dir,
        )
    # Decoder only -- the encoder half of this checkpoint is never used (see
    # cga_eval.py's module docstring) and is left exactly as constructed.
    install_cga_direction(mwg, fit.direction)

    model_dir = experiment_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    mwg.save_pretrained(str(model_dir))

    max_size = shared_kw.get(
        "encoder_eval_max_size", (cfg.training or {}).get("encoder_eval_max_size")
    )
    n_boot = int(
        shared_kw.get(
            "encoder_bootstrap_auc",
            (cfg.training or {}).get("encoder_bootstrap_auc") or 100,
        )
    )
    # Same task-level frame CAA scores (masked/label/label_class/split), not
    # anything derived from the trainer's own internal config -- so CGA's
    # readout is comparable row-for-row with CAA's, not just estimator-for-
    # estimator.
    eval_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    include_layers = _layerwise_method_enabled(cfg, "cga")
    layer_slices = (
        {f"L{layer}": bounds for layer, bounds in cga_layer_slices(mwg).items()}
        if include_layers
        else None
    )
    with cost_timer(
        f"encoder_eval:{backend}:{experiment_dir.name}", phase="encode", backend=backend
    ):
        fair = fair_cga_encoder_eval(
            mwg,
            fit.direction,
            eval_df,
            bundle.neutrals,
            target_classes=list(all_classes or bundle.classes or target_classes),
            class_encoding_direction=getattr(mwg, "feature_class_encoding_direction", None),
            max_size=int(max_size) if max_size is not None else None,
            max_neutral=int(max_size) if max_size is not None else None,
            n_bootstrap=n_boot,
            excluded_words=list(bundle.excluded_words or []),
            component_slices=layer_slices,
        )
    # The direction now lives in the decoder weights (just installed) and has
    # been used for every eval row; nothing else needs the 340MB (gpt2-small)
    # tensor itself. A fresh empty tensor, not a zero-length slice -- slicing
    # returns a view that keeps the original storage alive.
    import torch as _torch

    fit.direction = _torch.empty(0)
    out = {
        "trainer": trainer,
        "encoder_metrics": fair.get("encoder_metrics") or {},
        "readout_metrics": fair.get("readout_metrics") or {},
        "per_class_readouts": fair.get("per_class_readouts") or {},
        "per_component_readouts": fair.get("per_component_readouts") or {},
        "component_keys": fair.get("component_keys") or [],
        "model_path": str(experiment_dir),
        "backend": backend,
        "split_mode": split_mode,
        "target_classes": list(target_classes),
        "counterfactual_classes": counterfactual_classes,
        "data_mode": "merged_one_pole" if use_merged else "data_per_class",
        "ablation": "one_pole" if is_one_pole else "pair",
        "feature_class": feature_class,
        # Reported for parity with the trained backends' rows. CGA has no
        # optimizer, so the learning rate never enters the estimator -- it only
        # scales the causal intervention later, exactly as it does for GRADIEND.
        "learning_rate": float(getattr(args, "learning_rate", float("nan"))),
        "convergent_metric": None,
        "selection_metric": None,
        "cga_variant": variant,
        "cga_fit": fit.stats(),
        "cga_coordinate_projection": (getattr(args, "metadata", {}) or {}).get(
            "cga_coordinate_projection"
        ),
    }
    try:
        _mark_train_artifact_done(
            experiment_dir,
            config_hash=_train_artifact_hash(
                cfg,
                backend=backend,
                shared=shared_kw,
                split_mode=split_mode,
                target_classes=target_classes,
                feature_class=feature_class,
                counterfactual_classes=counterfactual_classes,
            ),
            extras={
                "ablation": out["ablation"],
                "learning_rate": out["learning_rate"],
                "auc_orientation_protocol_version": AUC_ORIENTATION_PROTOCOL_VERSION,
                # Which decoder-only label token the training items supervised.
                # Reloads read this back (absent = "legacy"), so an artifact is
                # never re-interpreted under a newer convention.
                "label_token_protocol": getattr(args, "label_token_protocol", "legacy"),
                "one_pole_data_protocol_version": (
                    ONE_POLE_DATA_PROTOCOL_VERSION if is_one_pole else None
                ),
                "collapsed_encoder": _any_collapsed_readout(out),
                "encoder_eval": _encoder_eval_payload(out),
                **_encoder_eval_rules_stamp(out),
                "encoder_eval_version": (
                    CGA_LAYERWISE_ENCODER_EVAL_VERSION
                    if include_layers
                    else CGA_ENCODER_EVAL_VERSION
                ),
                "cga_variant": variant,
                "cga_fit": fit.stats(),
                "cga_coordinate_projection": out["cga_coordinate_projection"],
            },
        )
    except Exception as exc:
        track_error(exc, context="write done.json")
    return out


def _fit_caga_once(
    *,
    backend: str,
    cfg: StudyConfig,
    bundle: TaskBundle,
    target_classes: Sequence[str],
    experiment_dir: Path,
    split_mode: SplitMode,
    shared: Dict[str, Any],
    smoke: bool,
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
    all_classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fit one CAGA direction (dL/dh mean-diff) in closed form and evaluate it.

    Structurally identical to :func:`_fit_cga_once` -- build the trainer, replace
    ``trainer.train()`` with a closed-form fit, install the direction into the
    ``latent_dim=1`` decoder, save a checkpoint, and score encoding metrics
    CAA-style. Only two things differ from CGA, and both are signal-agnostic reuse:
    the trainer is built on the ``activation_gradient`` signal (backend ``caga`` in
    ``build_training_arguments``), so ``fit_caga_direction`` and
    ``fair_caga_encoder_eval`` operate on ``dL/dh`` rather than the weight gradient.
    The installed decoder direction is read back at causal time
    (``caga_eval.caga_direction_from_model``) and used to STEER activations
    (CAA path), never to rewrite weights -- so CAGA rides the reload/skip path like
    any other checkpoint without a train-data dependency at causal time.
    """
    import torch as _torch

    from caga_eval import (
        CAGA_LAYERWISE_ENCODER_EVAL_VERSION,
        caga_layer_slices,
        fair_caga_encoder_eval,
        fit_caga_direction,
    )
    from cga_eval import install_cga_direction
    from study.tasks import labeled_df_for_eval

    built = _build_trainer(
        backend=backend,  # type: ignore[arg-type]
        cfg=cfg,
        bundle=bundle,
        target_classes=target_classes,
        experiment_dir=experiment_dir,
        split_mode=split_mode,
        shared=shared,
        smoke=smoke,
        feature_class=feature_class,
        counterfactual_classes=counterfactual_classes,
        all_classes=all_classes,
    )
    trainer = built.trainer
    args = built.args
    shared_kw = built.shared_kw
    is_one_pole = built.is_one_pole
    use_merged = built.use_merged
    feature_class = built.feature_class

    mwg = trainer.get_model()
    if getattr(mwg, "feature_class_encoding_direction", None) is None:
        labels_fn = getattr(trainer, "get_feature_class_encoding_labels", None)
        class_labels = labels_fn() if callable(labels_fn) else None
        if class_labels:
            mwg.set_feature_class_encoding_direction(class_labels)

    from cost_timer import cost_timer

    # Same bounded single-pass cap as CGA (dL/dh extraction is one streaming pass,
    # no step budget), reusing the existing cga_max_size knob.
    fit_max = (cfg.training or {}).get("cga_max_size")
    with cost_timer(
        f"caga_fit:{backend}:{experiment_dir.name}", phase="train", backend=backend
    ):
        fit = fit_caga_direction(
            trainer,
            split="train",
            max_samples=int(fit_max) if fit_max is not None else None,
        )
    direction_t = _torch.tensor(fit.direction, dtype=_torch.float32)
    # Decoder only -- the encoder half is never used; installing the direction
    # persists it in the checkpoint so causal reads it back without a re-fit.
    install_cga_direction(mwg, direction_t)

    model_dir = experiment_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    mwg.save_pretrained(str(model_dir))

    # CAGA encoder metrics: CAA-style cosine of each eval row's dL/dh to the fitted
    # direction, scored via the ActivationSignalExtractor over prediction_mask items
    # (NOT the weight-gradient model.forward path, which returns the wrong signal for
    # a caga model and crashes on frozen params). Same class_vs_neutral_metrics +
    # val-fit/test-report machinery as CGA, so caga and cga are metric-comparable.
    max_size = shared_kw.get(
        "encoder_eval_max_size", (cfg.training or {}).get("encoder_eval_max_size")
    )
    n_boot = int(
        shared_kw.get(
            "encoder_bootstrap_auc",
            (cfg.training or {}).get("encoder_bootstrap_auc") or 100,
        )
    )
    eval_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    include_layers = _layerwise_method_enabled(cfg, "caga")
    with cost_timer(
        f"encoder_eval:{backend}:{experiment_dir.name}", phase="encode", backend=backend
    ):
        fair = fair_caga_encoder_eval(
            mwg,
            direction_t,
            eval_df,
            bundle.neutrals,
            target_classes=list(all_classes or bundle.classes or target_classes),
            class_encoding_direction=getattr(mwg, "feature_class_encoding_direction", None),
            max_size=int(max_size) if max_size is not None else None,
            max_neutral=int(max_size) if max_size is not None else None,
            n_bootstrap=n_boot,
            excluded_words=list(bundle.excluded_words or []),
            layer_slices=caga_layer_slices(mwg) if include_layers else None,
        )
    caga_stats = {"n_used": int(fit.n_used), "dim": int(fit.dim), "split": str(fit.split)}
    out = {
        "trainer": trainer,
        "encoder_metrics": fair.get("encoder_metrics") or {},
        "readout_metrics": fair.get("readout_metrics") or {},
        "per_class_readouts": fair.get("per_class_readouts") or {},
        "per_component_readouts": fair.get("per_component_readouts") or {},
        "component_keys": fair.get("component_keys") or [],
        "model_path": str(experiment_dir),
        "backend": backend,
        "split_mode": split_mode,
        "target_classes": list(target_classes),
        "counterfactual_classes": counterfactual_classes,
        "data_mode": "merged_one_pole" if use_merged else "data_per_class",
        "ablation": "one_pole" if is_one_pole else "pair",
        "feature_class": feature_class,
        # CAGA has no optimizer; the learning rate only scales the causal steering
        # later (as it does for the other steering methods), never the estimator.
        "learning_rate": float(getattr(args, "learning_rate", float("nan"))),
        "convergent_metric": None,
        "selection_metric": None,
        "caga_fit": caga_stats,
    }
    try:
        _mark_train_artifact_done(
            experiment_dir,
            config_hash=_train_artifact_hash(
                cfg,
                backend=backend,
                shared=shared_kw,
                split_mode=split_mode,
                target_classes=target_classes,
                feature_class=feature_class,
                counterfactual_classes=counterfactual_classes,
            ),
            extras={
                "ablation": out["ablation"],
                "learning_rate": out["learning_rate"],
                "auc_orientation_protocol_version": AUC_ORIENTATION_PROTOCOL_VERSION,
                # Which decoder-only label token the training items supervised.
                # Reloads read this back (absent = "legacy"), so an artifact is
                # never re-interpreted under a newer convention.
                "label_token_protocol": getattr(args, "label_token_protocol", "legacy"),
                "one_pole_data_protocol_version": (
                    ONE_POLE_DATA_PROTOCOL_VERSION if is_one_pole else None
                ),
                "collapsed_encoder": _any_collapsed_readout(out),
                "encoder_eval": _encoder_eval_payload(out),
                **_encoder_eval_rules_stamp(out),
                "caga_fit": caga_stats,
                "encoder_eval_version": (
                    CAGA_LAYERWISE_ENCODER_EVAL_VERSION if include_layers else None
                ),
            },
        )
    except Exception as exc:
        track_error(exc, context="write done.json")
    return out


def _grad_norm_summary(payload: dict) -> dict:
    """Compact encoder/decoder gradient-norm summary from a training.json payload.

    Tests the one live hypothesis for the decoder shortfall: both Adam and SGD
    converge the encoder and neither converges the decoder on the same
    objective, so if the loss is far less sensitive to decoder scale the
    decoder's gradient is smaller in proportion. A ratio orders of magnitude
    below 1 supports that; near 1 refutes it.

    Returns {} when the series are absent, so an older package produces
    byte-identical done.json extras.
    """
    import statistics

    # The series live under training_stats, not at the top level -- the same
    # nesting the long-standing encoder_norms/decoder_norms use. Reading the top
    # level returned None and looked exactly like "the instrumentation did not
    # run", which cost a debugging round.
    stats = payload.get("training_stats")
    if not isinstance(stats, dict):
        stats = payload

    def _series(key):
        raw = stats.get(key)
        if not isinstance(raw, dict) or not raw:
            return []
        values = []
        for step, value in raw.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == numeric:  # drop NaN
                values.append((int(step), numeric))
        values.sort()
        return values

    enc = _series("encoder_grad_norms")
    dec = _series("decoder_grad_norms")
    if not enc and not dec:
        return {}
    out = {"n_steps": max(len(enc), len(dec))}
    for name, series in (("encoder", enc), ("decoder", dec)):
        if not series:
            continue
        vals = [v for _, v in series]
        out[f"{name}_grad_norm_median"] = float(statistics.median(vals))
        out[f"{name}_grad_norm_first"] = float(series[0][1])
        out[f"{name}_grad_norm_last"] = float(series[-1][1])
    ratios = [v for _, v in _series("decoder_over_encoder_grad_ratio")]
    if ratios:
        out["decoder_over_encoder_median"] = float(statistics.median(ratios))
    elif enc and dec:
        by_step = dict(enc)
        paired = [
            d / by_step[step]
            for step, d in dec
            if step in by_step and by_step[step] > 0.0
        ]
        if paired:
            out["decoder_over_encoder_median"] = float(statistics.median(paired))
    return out


def _load_convergence_summary(experiment_dir: Path) -> Dict[str, Any]:
    """Pull just ``best_score_checkpoint`` / ``convergence_info`` out of the
    package's ``training.json`` (written by the ``gradiend`` trainer), skipping
    the heavy per-step ``losses`` / ``training_stats`` arrays entirely.

    Answers "how many steps did this actually need before its best checkpoint"
    without ever syncing the multi-MB per-step log off the cluster — see CLAUDE.md's
    "max_steps convergence" note.
    """
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
        # Gradient-norm summary. The per-step series lives in training.json,
        # which rsync_analysis deliberately does not sync (multi-MB), so a
        # compact summary has to ride in done.json or the mechanism experiment
        # produces data nobody can read. Median over steps is used rather than
        # the mean because the first steps are transient.
        grad = _grad_norm_summary(payload)
        if grad:
            out["grad_norms"] = grad
        conv = payload.get("convergence_info")
        if isinstance(conv, dict):
            out["convergence_info"] = conv
        ta = payload.get("training_args")
        if isinstance(ta, dict):
            if ta.get("max_steps") is not None:
                out["training_max_steps"] = ta.get("max_steps")
            if ta.get("max_seeds") is not None:
                out["training_max_seeds"] = ta.get("max_seeds")
        if out:
            return out
    return {}


def _encode_method_status(err: Any, *, collapsed: bool = False) -> str:
    """Map encode failures: chance-level is ``collapsed`` (rerun via --rerun-collapsed)."""
    if collapsed or str(err or "") == "collapsed_encoder":
        return "collapsed"
    if err:
        return "error"
    return "ok"


def _any_collapsed_readout(raw: Mapping[str, Any]) -> bool:
    joint = raw.get("readout_metrics") or {}
    per_class = raw.get("per_class_readouts") or {}
    if _mark_collapsed_encoder(dict(joint or {}), joint).get("collapsed_encoder"):
        return True
    return any(
        _mark_collapsed_encoder(dict(rd or {}), joint).get("collapsed_encoder")
        for rd in per_class.values()
    )


def _artifact_marked_collapsed(experiment_dir: Path) -> bool:
    done = Path(experiment_dir) / "done.json"
    if not done.is_file():
        return False
    try:
        meta = json.loads(done.read_text(encoding="utf-8"))
    except Exception:
        return False
    extras = meta.get("extras") or {}
    return bool(extras.get("collapsed_encoder"))


def _mark_collapsed_encoder(
    rd: Dict[str, Any],
    joint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flag chance-level encoders so reports do not look like real features.

    ACTIEND none-split can collapse (joint auc≈0.5, class bal≈0.5, no Cohen's d)
    while :tensors / :L* from a separate trainer stay strong.
    """
    out = dict(rd or {})
    if out.get("error"):
        return out
    bal = out.get("balanced_accuracy")
    d = out.get("cohens_d")
    joint_auc = (joint or {}).get("roc_auc")
    collapsed = (
        isinstance(bal, (int, float))
        and abs(float(bal) - 0.5) < 0.02
        and (d is None or (isinstance(d, (int, float)) and abs(float(d)) < 1e-6))
    )
    if isinstance(joint_auc, (int, float)) and abs(float(joint_auc) - 0.5) < 0.02:
        collapsed = True
    if collapsed:
        out["error"] = "collapsed_encoder"
        out["collapsed_encoder"] = True
    return out


def _encode_error_is_soft(err: Any) -> bool:
    text = str(err or "").lower()
    return (
        "missing class labels" in text
        or "missing class column" in text
        or text.startswith("need both classes")
        or text == "empty"
        or text.startswith("empty encoder")
        or text == "no rows for class"
        or "no rows for class" in text
    )


def _estimator_provenance(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-row provenance for backends whose weights were not trained.

    CGA rows otherwise look exactly like GRADIEND rows in ``results.json``, and
    the study has been bitten before by rows whose origin could only be inferred
    from the surrounding run (see CLAUDE.md on suite-tagged results carrying
    rows from a differently-scoped invocation). Recording the estimator and its
    fit statistics on the row itself makes that answerable from the row alone.
    """
    variant = raw.get("cga_variant")
    if not variant:
        return {}
    return {"cga_variant": str(variant), "cga_fit": dict(raw.get("cga_fit") or {})}


def _rows_from_pair_train(
    *,
    backend: str,
    a: str,
    b: str,
    raw: Dict[str, Any],
    cfg: StudyConfig,
) -> List[Dict[str, Any]]:
    """Emit PoC-aligned ids: ``{backend}`` / ``:{cls}`` / ``:{cls}:tensors`` / ``:{cls}:L*``.

    Matches the SAE engine's ``_gradiend_method_rows`` so causal merge
    attaches onto the same method ids. ``:tensors`` = by_tensor aggregate
    (legacy reports used ``:all``).
    """
    from results_schema import fair_metric_fields
    from sae_eval import method_part_id
    from study.method_ids import TENSORS_SPLIT_TAG, pair_key as _pair_key

    enc = raw.get("encoder_metrics") or {}
    joint = raw.get("readout_metrics") or {}
    per_class = raw.get("per_class_readouts") or {}
    split_mode = str(raw.get("split_mode") or "none")
    is_tensor = split_mode == "tensors"
    provenance = _estimator_provenance(raw)
    joint_method = pair_trainer_id(backend, a, b, split_mode=split_mode)
    rows: List[Dict[str, Any]] = []
    # Soft encode failures leave ghost ``status=error`` rows that collide with
    # successful sibling pair / one-pole ids — skip them.
    if not _encode_error_is_soft(joint.get("error")):
        rows.append(
            method_result(
                method=joint_method,
                model=cfg.model_key,
                task=cfg.task_id,
                status=_encode_method_status(joint.get("error")),
                error=joint.get("error"),
                metrics={
                    **fair_metric_fields(joint),
                    **provenance,
                    "correlation": enc.get("correlation"),
                    "trainer_correlation": enc.get("correlation"),
                    "mean_by_class": enc.get("mean_by_class"),
                    "readout_kind": "joint_bipolar",
                    "gradiend_split": split_mode,
                    "ablation": "pair",
                    "n_neutral": joint.get("n_neutral"),
                    "pair": [a, b],
                },
                artifacts={"experiment_dir": raw.get("model_path")},
                extras={
                    "encoder_metrics": enc,
                    "readout_metrics": joint,
                    "per_class_readouts": per_class,
                    "component_keys": raw.get("component_keys") or [],
                },
            )
        )
    for cls in (a, b):
        rd = _mark_collapsed_encoder(per_class.get(str(cls)) or {}, joint)
        if _encode_error_is_soft(rd.get("error")):
            continue
        mid = pair_feature_id(backend, a, b, cls, split_mode=split_mode)
        rows.append(
            method_result(
                method=mid,
                model=cfg.model_key,
                task=cfg.task_id,
                status=_encode_method_status(
                    rd.get("error"), collapsed=bool(rd.get("collapsed_encoder"))
                ),
                error=rd.get("error"),
                metrics={
                    **fair_metric_fields(rd),
                    **provenance,
                    "target_class": cls,
                    "readout_kind": "class_vs_neutral",
                    "gradiend_split": split_mode,
                    "component_part": None if not is_tensor else TENSORS_SPLIT_TAG,
                    "ablation": "pair",
                    "mean_by_class": enc.get("mean_by_class"),
                    "correlation": enc.get("correlation"),
                    "mean_target": rd.get("mean_target"),
                    "mean_neutral": rd.get("mean_neutral"),
                    "mean_other": rd.get("mean_other"),
                    "n_neutral": rd.get("n_neutral"),
                    "encoding_direction": rd.get("encoding_direction"),
                    "pair": [a, b],
                    "collapsed_encoder": rd.get("collapsed_encoder"),
                },
                artifacts={"experiment_dir": raw.get("model_path")},
                extras={"readout_metrics": rd},
            )
        )
    for part, class_map in (raw.get("per_component_readouts") or {}).items():
        for cls in (a, b):
            rd = (class_map or {}).get(str(cls)) or {}
            if _encode_error_is_soft(rd.get("error")):
                continue
            rows.append(
                method_result(
                    method=method_part_id(
                        backend, f"{_pair_key(a, b)}:{cls}", str(part)
                    ),
                    model=cfg.model_key,
                    task=cfg.task_id,
                    status=_encode_method_status(
                        rd.get("error"), collapsed=bool(rd.get("collapsed_encoder"))
                    ),
                    error=rd.get("error"),
                    metrics={
                        **fair_metric_fields(rd),
                        "target_class": cls,
                        "readout_kind": "class_vs_neutral",
                        "gradiend_split": split_mode,
                        "component_part": str(part),
                        "ablation": "pair",
                        "mean_target": rd.get("mean_target"),
                        "mean_neutral": rd.get("mean_neutral"),
                        "mean_other": rd.get("mean_other"),
                        "n_neutral": rd.get("n_neutral"),
                        "encoding_direction": rd.get("encoding_direction"),
                        "pair": [a, b],
                    },
                    artifacts={"experiment_dir": raw.get("model_path")},
                    extras={"readout_metrics": rd},
                )
            )
    return rows


def _rows_from_onepole_train(
    *,
    backend: str,
    cls: str,
    raw: Dict[str, Any],
    cfg: StudyConfig,
) -> List[Dict[str, Any]]:
    from results_schema import fair_metric_fields
    from sae_eval import method_part_id

    enc = raw.get("encoder_metrics") or {}
    per_class = raw.get("per_class_readouts") or {}
    joint = raw.get("readout_metrics") or {}
    rd = _mark_collapsed_encoder(per_class.get(str(cls)) or {}, joint)
    split_mode = str(raw.get("split_mode") or "none")
    rows: List[Dict[str, Any]] = []
    if not _encode_error_is_soft(rd.get("error")):
        rows.append(
            method_result(
                method=onepole_feature_id(backend, cls, split_mode=split_mode),
                model=cfg.model_key,
                task=cfg.task_id,
                status=_encode_method_status(
                    rd.get("error"), collapsed=bool(rd.get("collapsed_encoder"))
                ),
                error=rd.get("error"),
                metrics={
                    **fair_metric_fields(rd),
                    **_estimator_provenance(raw),
                    "target_class": cls,
                    "readout_kind": "class_vs_neutral",
                    "gradiend_split": split_mode,
                    "ablation": "one_pole",
                    "correlation": enc.get("correlation"),
                    "mean_by_class": enc.get("mean_by_class"),
                    "mean_target": rd.get("mean_target"),
                    "mean_neutral": rd.get("mean_neutral"),
                    "mean_other": rd.get("mean_other"),
                    "n_neutral": rd.get("n_neutral"),
                    "encoding_direction": rd.get("encoding_direction"),
                    "counterfactual_classes": raw.get("counterfactual_classes"),
                    "collapsed_encoder": rd.get("collapsed_encoder"),
                },
                artifacts={"experiment_dir": raw.get("model_path")},
                extras={
                    "encoder_metrics": enc,
                    "readout_metrics": rd,
                    "per_class_readouts": per_class,
                },
            )
        )
    for part, class_map in (raw.get("per_component_readouts") or {}).items():
        rd_p = (class_map or {}).get(str(cls)) or {}
        if _encode_error_is_soft(rd_p.get("error")):
            continue
        rows.append(
            method_result(
                method=method_part_id(backend, cls, str(part)),
                model=cfg.model_key,
                task=cfg.task_id,
                status=_encode_method_status(
                    rd_p.get("error"), collapsed=bool(rd_p.get("collapsed_encoder"))
                ),
                error=rd_p.get("error"),
                metrics={
                    **fair_metric_fields(rd_p),
                    "target_class": cls,
                    "readout_kind": "class_vs_neutral",
                    "gradiend_split": split_mode,
                    "component_part": str(part),
                    "ablation": "one_pole",
                    "mean_target": rd_p.get("mean_target"),
                    "mean_neutral": rd_p.get("mean_neutral"),
                    "mean_other": rd_p.get("mean_other"),
                    "n_neutral": rd_p.get("n_neutral"),
                    "encoding_direction": rd_p.get("encoding_direction"),
                },
                artifacts={"experiment_dir": raw.get("model_path")},
                extras={"readout_metrics": rd_p},
            )
        )
    return rows


def run_train_stage(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    enabled_backends: Sequence[str],
    train_splits: Optional[Sequence[str]] = None,
    smoke: bool = False,
    skip_existing: bool = False,
    rerun_collapsed: bool = False,
    rerun_orphaned_artifacts: bool = False,
    fail_fast: bool = False,
    retain_trainers: bool = True,
    previous_methods: Optional[Sequence[Dict[str, Any]]] = None,
    refresh_encoder_eval: bool = False,
) -> Dict[str, Any]:
    """Train pair + one-pole ablations for enabled backends × suite splits.

    With ``skip_existing``, completed artifacts (``done.json`` or a promoted root
    checkpoint) are reloaded instead of retrained, and encoder analysis is restored
    from ``done.json`` / ``results.json`` (not recomputed). Mid-seed interrupts are
    not skipped. ``rerun_collapsed`` retrains artifacts whose encoder was
    chance-level even if ``done.json`` exists (distinct from hard train errors).
    ``rerun_orphaned_artifacts`` is used after a suite/config change: an
    artifact with no successful row in the prior result is new protocol work
    and is retrained, while completed artifacts still reload.
    ``fail_fast`` re-raises the first failed pair or one-pole ablation instead of
    recording an error row and continuing with the remaining ablations.
    ``retain_trainers=False`` releases each completed trainer before starting the
    next artifact.  Use it when no later SAE/CAA/causal/localization stage needs
    live trainer objects.
    """
    from itertools import combinations

    abl = _ablations(cfg, smoke=smoke)
    artifact_cache_mode = str((cfg.training or {}).get("train_cache_mode") or "always").strip().lower()

    classes = [str(c) for c in bundle.classes]
    out_dir = cfg.output_dir
    method_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    # train_raw_none[backend] = first none-split trainer (pair preferred; SAE/CAA/loc)
    train_raw_none: Dict[str, Dict[str, Any]] = {}
    train_raw_tensors: Dict[str, Dict[str, Any]] = {}
    # Per-class maps so causal uses the right pair/one-pole trainer per pole
    # (legacy single-map kept only the first trainer × all TARGET_CLASSES).
    train_raw_none_by_class: Dict[str, Dict[str, Any]] = {}
    train_raw_tensors_by_class: Dict[str, Dict[str, Any]] = {}
    all_trains: List[Dict[str, Any]] = []

    def _register_by_class(
        bucket: Dict[str, Dict[str, Dict[str, Any]]],
        backend: str,
        class_ids: Sequence[str],
        raw: Dict[str, Any],
        *,
        overwrite: bool,
    ) -> None:
        slot = bucket.setdefault(str(backend), {})
        for c in class_ids:
            key = str(c)
            prev = slot.get(key)
            if prev is None:
                slot[key] = raw
                continue
            prev_list = prev if isinstance(prev, list) else [prev]
            if id(raw) in {id(x) for x in prev_list}:
                continue
            if overwrite:
                slot[key] = raw
                continue
            slot[key] = prev_list + [raw]

    def _train_or_reload(
        *,
        backend: str,
        target_classes: Sequence[str],
        experiment_dir: Path,
        split_mode: SplitMode,
        shared: Dict[str, Any],
        feature_class: Optional[str] = None,
        counterfactual_classes: Optional[Any] = None,
    ) -> Dict[str, Any]:
        config_hash = _train_artifact_hash(
            cfg,
            backend=backend,
            shared=shared,
            split_mode=split_mode,
            target_classes=target_classes,
            feature_class=feature_class,
            counterfactual_classes=counterfactual_classes,
        )
        # ``--causal-only`` never trains: a missing/unloadable checkpoint is an error.
        causal_only = bool((cfg.raw.get("cli") or {}).get("causal_only"))
        force_orphaned_retrain = (
            rerun_orphaned_artifacts
            and not causal_only
            and not artifact_has_successful_result(previous_methods, experiment_dir.name)
        )
        # CGA's streaming accumulator placement is an estimator property, not
        # merely a resource hint: a large-model run explicitly configured to
        # keep its 26-GiB running mean on CPU must not silently reload a legacy
        # CUDA-accumulator artifact.  Keep normal ``SKIP_EXISTING=1`` behavior
        # for every other setting/artifact, and refit only this affected CGA
        # cell when its persisted fit provenance disagrees.
        cga_accumulator_changed = (
            is_cga_backend(backend)
            and not causal_only
            and not _cga_accumulator_matches(experiment_dir, cfg)
        )
        if (
            skip_existing
            and not force_orphaned_retrain
            and not cga_accumulator_changed
            and _artifact_train_complete(
            experiment_dir, config_hash=config_hash, cache_mode=artifact_cache_mode
            )
        ):
            if rerun_collapsed and _artifact_marked_collapsed(experiment_dir):
                print(
                    f"  rerun-collapsed {experiment_dir.name}: chance-level encoder; retraining",
                    flush=True,
                )
            else:
                raw = _reload_train_once(
                    backend=backend,  # type: ignore[arg-type]
                    cfg=cfg,
                    bundle=bundle,
                    target_classes=target_classes,
                    experiment_dir=experiment_dir,
                    split_mode=split_mode,
                    shared=shared,
                    smoke=smoke,
                    feature_class=feature_class,
                    counterfactual_classes=counterfactual_classes,
                    all_classes=classes,
                    previous_methods=previous_methods,
                    refresh_encoder_eval=refresh_encoder_eval,
                )
                if raw is not None:
                    return raw
                print(
                    f"  resume miss {experiment_dir.name}: marked complete but reload failed; retraining",
                    flush=True,
                )
        elif force_orphaned_retrain:
            print(
                f"  config-change: retraining new/unresolved artifact {experiment_dir.name}",
                flush=True,
            )
        elif cga_accumulator_changed:
            print(
                f"  cga accumulator config-change: refitting {experiment_dir.name}",
                flush=True,
            )
        if causal_only:
            raise RuntimeError(
                f"--causal-only: artifact {experiment_dir.name} could not be reloaded "
                "from its checkpoint; refusing to train."
            )
        if is_caga_backend(backend):
            fit_fn = _fit_caga_once
        elif is_cga_backend(backend):
            fit_fn = _fit_cga_once
        else:
            fit_fn = _train_once
        return fit_fn(
            backend=backend,  # type: ignore[arg-type]
            cfg=cfg,
            bundle=bundle,
            target_classes=target_classes,
            experiment_dir=experiment_dir,
            split_mode=split_mode,
            shared=shared,
            smoke=smoke,
            feature_class=feature_class,
            counterfactual_classes=counterfactual_classes,
            all_classes=classes,
        )

    backends: List[str] = []
    if "gradiend" in enabled_backends:
        backends.append("gradiend")
    if "actiend" in enabled_backends:
        backends.append("actiend")
        # Mixed-site ACTIEND-PRE is opt-in (weak on gender_en); not SAE-style default.
        if _actiend_pre_enabled(cfg) or "actiend_pre" in enabled_backends:
            backends.append("actiend_pre")
    if "cga" in enabled_backends:
        # One backend id per requested CGA variant (see cga_eval.cga_backend_id),
        # so the tensor-norm ablation never merges into the headline CGA rows.
        from cga_eval import cga_backend_id, resolve_cga_variants

        for variant in resolve_cga_variants(
            (cfg.suite.get("methods") or {}).get("cga")
        ):
            backends.append(cga_backend_id(variant))
    if "caga" in enabled_backends:
        # Single backend id (no variants); closed-form dL/dh mean-diff, activation
        # steering. Built by _fit_caga_once (dispatched above).
        backends.extend(CAGA_BACKENDS)
    if "agiend" in enabled_backends:
        # LEARNED dL/dh encoder-decoder; trained via _train_once (dispatched above),
        # activation steering. Single backend id, no variants.
        backends.extend(AGIEND_BACKENDS)

    for backend in backends:
        # actiend_pre uses actiend LR; sites forced below for hash + Signal.
        lr_backend = "actiend" if backend == "actiend_pre" else backend
        shared = shared_training_kwargs(cfg, backend=lr_backend)  # type: ignore[arg-type]
        if backend == "actiend_pre":
            shared["activation_site"] = "pre_prediction"
            shared["target_activation_site"] = "prediction"
            shared["actiend_source_site"] = "pre_prediction"
            shared["actiend_target_site"] = "prediction"
        if smoke:
            shared = apply_smoke_training_kwargs(shared)
        print(
            f"train backend={backend} source={shared.get('source')} "
            f"learning_rate={shared.get('learning_rate')} "
            f"activation_site={shared.get('activation_site')} "
            f"target_activation_site={shared.get('target_activation_site')} "
            f"max_steps={shared.get('max_steps')} "
            f"max_length={shared.get('max_length', 128)} "
            f"skip_existing={skip_existing} "
            f"cache_mode={artifact_cache_mode}",
            flush=True,
        )
        if str(shared.get("source") or "").lower() != str(cfg.training.get("source") or "").lower():
            raise RuntimeError(
                f"Training source mismatch: shared={shared.get('source')!r} "
                f"vs task config={cfg.training.get('source')!r}"
            )
        pair_splits = _requested_train_splits(
            _effective_train_splits(cfg, backend, one_pole=False), train_splits
        )
        one_pole_splits = _requested_train_splits(
            _effective_train_splits(cfg, backend, one_pole=True), train_splits
        )
        skipped = sorted(
            set(_suite_splits(cfg, backend))
            - (set(pair_splits) | set(one_pole_splits))
        )
        if skipped:
            print(
                f"train {backend}: skipped splits {skipped} "
                f"(suite/protocol/operator filter)",
                flush=True,
            )
        if abl["pair"]:
            for split_mode in pair_splits:
                pairs = list(combinations(classes, 2)) if len(classes) > 2 else [tuple(classes[:2])]
                if len(classes) == 2:
                    pairs = [tuple(sorted_pair(classes[0], classes[1]))]
                for a, b in pairs:
                    a, b = sorted_pair(a, b)
                    dirname = artifact_dirname(backend, kind="pair", key=pair_key(a, b), split_mode=split_mode)
                    exp_dir = out_dir / "artifacts" / dirname
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        raw = _train_or_reload(
                            backend=backend,
                            target_classes=[a, b],
                            experiment_dir=exp_dir,
                            split_mode=split_mode,
                            shared=shared,
                        )
                        raw["ablation"] = "pair"
                        raw["pair"] = [a, b]
                        all_trains.append(raw)
                        method_rows.extend(
                            _rows_from_pair_train(backend=backend, a=a, b=b, raw=raw, cfg=cfg)
                        )
                        if split_mode == "none":
                            if backend not in train_raw_none:
                                train_raw_none[backend] = raw
                            _register_by_class(
                                train_raw_none_by_class,
                                backend,
                                [a, b],
                                raw,
                                overwrite=False,
                            )
                        if split_mode == "tensors":
                            train_raw_tensors[backend] = raw
                            _register_by_class(
                                train_raw_tensors_by_class,
                                backend,
                                [a, b],
                                raw,
                                overwrite=False,
                            )
                        if not retain_trainers and _release_train_raw_trainer(raw):
                            print(
                                f"released completed trainer {dirname} "
                                "(no downstream trainer consumers)",
                                flush=True,
                            )
                    except Exception as exc:
                        track_error(exc, context="train pair", backend=backend, split_mode=split_mode)
                        if fail_fast:
                            raise
                        errors.append(f"{backend} pair {a}-{b} split={split_mode}: {exc}")
                        method_rows.append(
                            method_result(
                                method=pair_trainer_id(backend, a, b, split_mode=split_mode),
                                model=cfg.model_key,
                                task=cfg.task_id,
                                status="error",
                                error=str(exc),
                            )
                        )

        if abl["one_pole"]:
            for split_mode in one_pole_splits:
                task_fac = _task_one_pole_classes(cfg)
                # Always train one-pole when the suite enables it — even on binary
                # tasks that also train pair. Pair and one-pole are different
                # trainers (joint bipolar vs claim-vs-CF); ids must not collide
                # (pair → gradiend:{a}-{b}:{cls}, one-pole → gradiend:{cls}).
                # Optional study-level pole filter (e.g. IOI one_pole_classes: [IO]).
                # Package one-pole uses target_classes=[cls] + counterfactual_classes only.
                pole_classes = list(task_fac) if task_fac else list(classes)
                for cls in pole_classes:
                    if cls not in classes:
                        print(f"skip onepole {cls}: not in task classes {classes}", flush=True)
                        continue
                    others = [c for c in classes if c != cls]
                    if not others:
                        continue
                    target = [cls]
                    cf = "all"
                    dirname = artifact_dirname(
                        backend, kind="onepole", key=str(cls), split_mode=split_mode
                    )
                    exp_dir = out_dir / "artifacts" / dirname
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"train onepole {backend}:{cls} "
                        f"target_classes={[cls]} counterfactual_classes={cf!r} "
                        f"convergent_metric="
                        f"{(cfg.training or {}).get('convergent_metric') or 'min_auc_n_o'} data="
                        f"{'merged' if _merged_one_pole_frame(bundle) is not None else 'per_class'}",
                        flush=True,
                    )
                    try:
                        raw = _train_or_reload(
                            backend=backend,
                            target_classes=target,
                            experiment_dir=exp_dir,
                            split_mode=split_mode,
                            shared=shared,
                            feature_class=cls,
                            counterfactual_classes=cf,
                        )
                        raw["ablation"] = "one_pole"
                        raw["feature_class"] = cls
                        all_trains.append(raw)
                        method_rows.extend(
                            _rows_from_onepole_train(
                                backend=backend, cls=cls, raw=raw, cfg=cfg
                            )
                        )
                        # Keep pair + one-pole trainers (unique method ids).
                        if split_mode == "none":
                            if backend not in train_raw_none:
                                train_raw_none[backend] = raw
                            _register_by_class(
                                train_raw_none_by_class,
                                backend,
                                [cls],
                                raw,
                                overwrite=False,
                            )
                        if split_mode == "tensors":
                            train_raw_tensors[backend] = raw
                            _register_by_class(
                                train_raw_tensors_by_class,
                                backend,
                                [cls],
                                raw,
                                overwrite=False,
                            )
                        if not retain_trainers and _release_train_raw_trainer(raw):
                            print(
                                f"released completed trainer {dirname} "
                                "(no downstream trainer consumers)",
                                flush=True,
                            )
                    except Exception as exc:
                        track_error(exc, context="train onepole", backend=backend, cls=cls, split_mode=split_mode)
                        if fail_fast:
                            raise
                        errors.append(f"{backend} onepole {cls} split={split_mode}: {exc}")
                        method_rows.append(
                            method_result(
                                method=onepole_feature_id(
                                    backend, cls, split_mode=split_mode
                                ),
                                model=cfg.model_key,
                                task=cfg.task_id,
                                status="error",
                                error=str(exc),
                            )
                        )
    return {
        "methods": method_rows,
        "errors": errors,
        "train_raw_none": train_raw_none,
        "train_raw_tensors": train_raw_tensors,
        "train_raw_none_by_class": train_raw_none_by_class,
        "train_raw_tensors_by_class": train_raw_tensors_by_class,
        "all_trains": all_trains,
    }


def _find_checkpoint_dir(experiment_dir: Path) -> Optional[Path]:
    """Locate a loadable GRADIEND/ACTIEND checkpoint under an experiment dir.

    Preference order:
    1. promoted / final model at the experiment root (or ``model/``)
    2. ``*_best`` seed checkpoints
    3. live ``seeds/seed_*`` dirs (may be mid-run)
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        return None

    def _looks_like_ckpt(cand: Path) -> bool:
        if not cand.is_dir():
            return False
        try:
            names = {p.name for p in cand.iterdir()}
        except OSError:
            return False
        return any(
            n in names
            for n in (
                "config.json",
                "gradiend_config.json",
                "gradiend_context.json",
                "pytorch_model.bin",
                "model.safetensors",
                "encoder.pt",
            )
        )

    def _search(cand: Path) -> Optional[Path]:
        if _looks_like_ckpt(cand):
            return cand
        if not cand.is_dir():
            return None
        try:
            kids = [p for p in cand.iterdir() if p.is_dir()]
        except OSError:
            return None
        if len(kids) == 1:
            return _search(kids[0])
        return None

    # Promoted final model (multi-seed finished) — strongest signal of completion.
    for cand in (
        experiment_dir / "model",
        experiment_dir,
        experiment_dir / "gradiend",
        experiment_dir / "checkpoint",
    ):
        hit = _search(cand)
        if hit is not None:
            return hit

    seeds = experiment_dir / "seeds"
    if seeds.is_dir():
        best_dirs = sorted(seeds.glob("seed_*_best"), key=lambda p: p.name)
        for cand in best_dirs:
            hit = _search(cand)
            if hit is not None:
                return hit
        live_dirs = sorted(
            (p for p in seeds.glob("seed_*") if not p.name.endswith("_best")),
            key=lambda p: p.name,
        )
        for cand in live_dirs:
            hit = _search(cand)
            if hit is not None:
                return hit
    return None


def _checkpoint_is_promoted(experiment_dir: Path) -> bool:
    """True when a loadable model sits at the experiment root (not only under seeds/)."""
    experiment_dir = Path(experiment_dir)
    for cand in (
        experiment_dir / "model",
        experiment_dir,
        experiment_dir / "gradiend",
        experiment_dir / "checkpoint",
    ):
        if not cand.is_dir():
            continue
        try:
            names = {p.name for p in cand.iterdir()}
        except OSError:
            continue
        if any(
            n in names
            for n in (
                "config.json",
                "gradiend_config.json",
                "gradiend_context.json",
                "pytorch_model.bin",
                "model.safetensors",
                "encoder.pt",
            )
        ):
            return True
    return False


def _train_artifact_hash(
    cfg: StudyConfig,
    *,
    backend: str,
    shared: Dict[str, Any],
    split_mode: str,
    target_classes: Sequence[str],
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
) -> str:
    from activation_protocol import ACTIVATION_PROTOCOL_VERSION
    from study.training_profiles import actiend_signal_scale_policy, is_actiend_backend

    # Resolve source the SAME way training does (one-pole -> "both"), so the hash
    # reflects what is actually trained regardless of which `shared` a caller
    # passes. Without this the mark-done path (shared_kw, override applied) and
    # the skip-check path (raw shared) would hash different sources for one-pole.
    is_one_pole = feature_class is not None or counterfactual_classes is not None
    resolved_source = one_pole_train_source(shared, is_one_pole=is_one_pole)

    payload = {
        "backend": backend,
        "split_mode": split_mode,
        "target_classes": [str(c) for c in target_classes],
        "feature_class": feature_class,
        "counterfactual_classes": counterfactual_classes,
        "learning_rate": shared.get("learning_rate"),
        "max_steps": shared.get("max_steps"),
        "source": resolved_source,
        "target": shared.get("target"),
        "train_batch_size": shared.get("train_batch_size"),
        "max_length": shared.get("max_length"),
        "convergent_metric": shared.get("convergent_metric"),
        "selection_metric": shared.get("selection_metric"),
        "activation_site": shared.get("activation_site"),
        "target_activation_site": shared.get("target_activation_site")
        or shared.get("actiend_target_site"),
        "actiend_source_site": shared.get("actiend_source_site"),
        "actiend_signal_scale": (
            actiend_signal_scale_policy(split_mode)
            if is_actiend_backend(backend)
            else None
        ),
        "activation_protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "auc_orientation_protocol_version": AUC_ORIENTATION_PROTOCOL_VERSION,
        "one_pole_data_protocol_version": (
            ONE_POLE_DATA_PROTOCOL_VERSION
            if feature_class is not None or counterfactual_classes is not None
            else None
        ),
        "model_key": cfg.model_key,
        "task_id": cfg.task_id,
        "hf_model": cfg.hf_model,
    }
    # An explicit CGA accumulator placement changes the fitted artifact's
    # provenance.  Preserve historical hashes when it is unset.
    if is_cga_backend(backend) and (cfg.training or {}).get("cga_accumulate_device") is not None:
        payload["cga_accumulate_device"] = str(
            (cfg.training or {}).get("cga_accumulate_device")
        )
    if is_cga_backend(backend) and (cfg.training or {}).get("cga_pre_prune"):
        pre = (cfg.training or {}).get("pre_prune") or {}
        payload["cga_coordinate_projection"] = "pre_prune"
        payload["cga_pre_prune"] = {
            "n_samples": pre.get("n_samples", 16),
            "topk": pre.get("topk", 0.1),
            "source": "alternative",
        }
    # Preserve the historical hash for unconfigured runs. Only an explicit
    # decoder LR is a new training protocol and therefore needs cache
    # invalidation/provenance.
    if shared.get("learning_rate_decoder") is not None:
        payload["learning_rate_decoder"] = shared["learning_rate_decoder"]
    # Same rule for the optimizer: a non-Adam arm is a different training
    # protocol and must not reuse an Adam checkpoint, but an unconfigured run
    # (and an explicit "adamw", which is what every historical run already did)
    # keeps its original hash.
    optim = shared.get("optim")
    if optim is not None and str(optim).lower() != "adamw":
        payload["optim"] = str(optim).lower()
        if shared.get("sgd_momentum"):
            payload["sgd_momentum"] = shared["sgd_momentum"]
    # A non-default dtype is a different numerical protocol, so a bfloat16 run
    # must not reuse a float32 checkpoint. float32 is the package default and
    # what every historical run used, so it keeps the original hash.
    dtype = shared.get("torch_dtype")
    if dtype is not None and str(dtype) not in ("torch.float32", "float32"):
        payload["torch_dtype"] = str(dtype)
    return stable_hash(payload)


def _cga_accumulator_matches(experiment_dir: Path, cfg: StudyConfig) -> bool:
    """Return whether explicit CGA fit settings match ``done.json``.

    Unset keeps the historical cache policy. Explicit large-model accumulator
    or projection settings require recorded matching provenance; old/malformed
    artifacts are conservatively refit rather than silently reloaded.
    """
    requested = (cfg.training or {}).get("cga_accumulate_device")
    requested_projection = bool((cfg.training or {}).get("cga_pre_prune"))
    if requested is None and not requested_projection:
        return True
    done = Path(experiment_dir) / "done.json"
    try:
        meta = json.loads(done.read_text(encoding="utf-8"))
        extras = meta.get("extras") or {}
        observed = (extras.get("cga_fit") or {}).get(
            "accumulate_device"
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    if requested is not None and (
        str(observed).strip().lower() != str(requested).strip().lower()
    ):
        return False
    return not requested_projection or extras.get("cga_coordinate_projection") == "pre_prune"


def _artifact_train_complete(
    experiment_dir: Path,
    *,
    config_hash: Optional[str] = None,
    cache_mode: str = "always",
) -> bool:
    """Whether this artifact finished training (safe to skip / reload).

    Interrupted mid-seed runs leave weights only under ``seeds/``; completed
    multi-seed runs promote a model to the experiment root. ``done.json`` is the
    study-level marker written after the first fair encoder eval; skip-existing
    reloads that payload instead of recomputing encoder analysis.

    ``cache_mode`` (``training.train_cache_mode``, default ``"always"``):
    ``"always"`` trusts any existing ``done.json`` + checkpoint regardless of
    whether ``config_hash`` still matches (e.g. after a max_steps/LR tweak);
    ``"hash_match"`` restores the prior strict behavior and retrains whenever
    the hash drifts. Reload itself (``_reload_train_once``) always uses the
    saved weights as-is, so tolerating a hash mismatch here is safe — it just
    skips a retrain that would otherwise be triggered by an unrelated config
    change.
    """
    experiment_dir = Path(experiment_dir)
    done = experiment_dir / "done.json"
    if done.is_file():
        try:
            meta = json.loads(done.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        strict = str(cache_mode or "always").strip().lower() == "hash_match"
        if (
            strict
            and config_hash is not None
            and str(meta.get("config_hash") or "") not in ("", str(config_hash))
        ):
            return False
        return _find_checkpoint_dir(experiment_dir) is not None
    # Legacy: promoted root checkpoint implies train() finished (not mid-seed interrupt).
    return _checkpoint_is_promoted(experiment_dir)


def artifact_label_token_protocol(experiment_dir: Path) -> str:
    """Decoder-only label-token protocol an existing artifact was trained under.

    Read from the ``label_token_protocol`` stamp in ``done.json``. An artifact
    without the stamp (everything trained before the protocol existed) or without a
    readable ``done.json`` is ``"legacy"``: reloading must reproduce how it was
    trained, never silently adopt a newer default.
    """
    try:
        payload = json.loads((Path(experiment_dir) / "done.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "legacy"
    stamp = ((payload or {}).get("extras") or {}).get("label_token_protocol")
    return stamp if stamp in ("canonical", "legacy") else "legacy"


def _mark_train_artifact_done(
    experiment_dir: Path,
    *,
    config_hash: str,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "stage": experiment_dir.name,
        "config_hash": config_hash,
        "paths": {"checkpoint": str(_find_checkpoint_dir(experiment_dir) or experiment_dir)},
    }
    if extras:
        payload["extras"] = extras
    (experiment_dir / "done.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


_ENCODER_EVAL_KEYS = (
    "encoder_metrics",
    "readout_metrics",
    "per_class_readouts",
    "per_component_readouts",
    "component_keys",
)


def _encoder_eval_rules_stamp(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"encoder_eval_rules_version": N}`` only if every readout used frozen rules.

    Defensive second line behind ``require_frozen_rules``: a blob holding any
    test-fit (oracle) Spec_n/Excl must not be stamped current, or it would never
    be re-entered.
    """
    from results_schema import unfrozen_rule_readouts

    bad = list(unfrozen_rule_readouts((payload or {}).get("per_class_readouts")))
    for part, comps in ((payload or {}).get("per_component_readouts") or {}).items():
        bad.extend(f"{part}:{k}" for k in unfrozen_rule_readouts(comps))
    if bad:
        print(
            f"  WARNING: {len(bad)} readout(s) without validation-frozen Spec_n/Excl "
            f"rules ({bad[:4]}...); NOT stamping encoder_eval_rules_version",
            flush=True,
        )
        return {}
    return {"encoder_eval_rules_version": ENCODER_EVAL_RULES_VERSION}


def _encoder_eval_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-safe encoder-eval blob stored in ``done.json`` / restored on skip."""
    return {k: raw.get(k) or ({} if k != "component_keys" else []) for k in _ENCODER_EVAL_KEYS}


def _encoder_eval_usable(payload: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return bool(payload.get("encoder_metrics") or payload.get("per_class_readouts"))


def _row_matches_experiment(row: Mapping[str, Any], experiment_dir: Path) -> bool:
    art = (row.get("artifacts") or {}).get("experiment_dir")
    if not art:
        return False
    p = Path(str(art))
    try:
        return p.name == experiment_dir.name or p.resolve() == experiment_dir.resolve()
    except OSError:
        return p.name == experiment_dir.name


def encoder_eval_from_done(
    experiment_dir: Path,
    *,
    required_version: Optional[int] = None,
    require_current_rules: bool = False,
) -> Optional[Dict[str, Any]]:
    """Restore encoder eval from ``done.json`` extras (skip-existing, no recompute).

    ``require_current_rules``: reject a blob whose Spec_n/Excl were fit on the
    reported split (no ``encoder_eval_rules_version`` stamp >= current). Only the
    encode path sets this. The causal reload must not: it reads the saved
    *validation* readouts, which the rule fix left unchanged.
    """
    done = Path(experiment_dir) / "done.json"
    if not done.is_file():
        return None
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
    except Exception:
        return None
    extras = payload.get("extras") or {}
    if required_version is not None and extras.get("encoder_eval_version") != int(
        required_version
    ):
        return None
    if require_current_rules and not encoder_eval_rules_current(
        extras.get("encoder_eval_rules_version")
    ):
        return None
    blob = extras.get("encoder_eval")
    return dict(blob) if _encoder_eval_usable(blob) else None


def encoder_eval_from_method_rows(
    rows: Optional[Sequence[Mapping[str, Any]]],
    experiment_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Restore encoder eval from ``results.json`` rows for this artifact."""
    if not rows:
        return None
    matched = [
        r
        for r in rows
        if isinstance(r, Mapping) and _row_matches_experiment(r, experiment_dir)
    ]
    if not matched:
        return None
    out: Dict[str, Any] = {
        "encoder_metrics": {},
        "readout_metrics": {},
        "per_class_readouts": {},
        "per_component_readouts": {},
        "component_keys": [],
    }
    for row in matched:
        extras = row.get("extras") or {}
        if extras.get("encoder_metrics"):
            out["encoder_metrics"] = dict(extras["encoder_metrics"])
        if extras.get("per_class_readouts"):
            out["per_class_readouts"] = dict(extras["per_class_readouts"])
        if extras.get("component_keys"):
            out["component_keys"] = list(extras["component_keys"])
        metrics = row.get("metrics") or {}
        part = metrics.get("component_part")
        rd = extras.get("readout_metrics")
        if rd and not part:
            out["readout_metrics"] = dict(rd)
        elif rd and part:
            cls = str(metrics.get("target_class") or "")
            slot = out["per_component_readouts"].setdefault(str(part), {})
            if cls:
                slot[cls] = dict(rd)
    return out if _encoder_eval_usable(out) else None


def persist_encoder_eval_in_done(
    experiment_dir: Path,
    payload: Mapping[str, Any],
    *,
    version: Optional[int] = None,
) -> None:
    """Merge encoder-eval blob into existing ``done.json`` extras (legacy artifacts)."""
    done = Path(experiment_dir) / "done.json"
    if not done.is_file() or not _encoder_eval_usable(payload):
        return
    try:
        meta = json.loads(done.read_text(encoding="utf-8"))
    except Exception:
        return
    extras = dict(meta.get("extras") or {})
    extras["encoder_eval"] = _encoder_eval_payload(payload)
    # Stamp only a blob whose readouts all used validation-frozen rules; clear any
    # stale stamp otherwise so it is re-entered rather than trusted.
    extras.pop("encoder_eval_rules_version", None)
    extras.update(_encoder_eval_rules_stamp(payload))
    if version is not None:
        extras["encoder_eval_version"] = int(version)
    meta["extras"] = extras
    # A layerwise-eval upgrade can take long enough to be preempted.  Preserve
    # the prior completed checkpoint metadata until the replacement JSON is
    # fully flushed: a crash may leave a disposable ``*.tmp`` file, never a
    # torn ``done.json`` that would make the fitted artifact unrecoverable.
    tmp = done.with_name(f"{done.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    tmp.replace(done)


def load_encoder_eval_for_reload(
    experiment_dir: Path,
    *,
    previous_methods: Optional[Sequence[Mapping[str, Any]]] = None,
    required_version: Optional[int] = None,
    require_current_rules: bool = False,
) -> Optional[Dict[str, Any]]:
    """Prefer ``done.json``, then prior method rows. None → must recompute.

    With ``require_current_rules`` only a ``done.json`` blob stamped with the
    current rules version qualifies: prior ``results.json`` rows carry no stamp,
    so they can never prove their Spec_n/Excl are validation-frozen.
    """
    blob = encoder_eval_from_done(
        experiment_dir,
        required_version=required_version,
        require_current_rules=require_current_rules,
    )
    if blob is not None:
        return blob
    if required_version is not None or require_current_rules:
        return None
    return encoder_eval_from_method_rows(previous_methods, experiment_dir)


def _build_trainer_for_artifact(
    *,
    backend: Backend,
    cfg: StudyConfig,
    bundle: TaskBundle,
    target_classes: Sequence[str],
    experiment_dir: Path,
    split_mode: SplitMode,
    shared: Dict[str, Any],
    smoke: bool,
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
    all_classes: Optional[Sequence[str]] = None,
    random_state: Optional[int] = None,
):
    """Construct trainer + args without calling ``train()``."""
    from gradiend.trainer.text.prediction.trainer import (
        TextPredictionConfig,
        TextPredictionTrainer,
    )

    shared_kw = dict(shared)
    if backend == "actiend_pre":
        # Fingerprint + Signal: force mixed sites independent of task YAML.
        shared_kw["activation_site"] = "pre_prediction"
        shared_kw["target_activation_site"] = "prediction"
        shared_kw["actiend_source_site"] = "pre_prediction"
        shared_kw["actiend_target_site"] = "prediction"
    is_one_pole = feature_class is not None or counterfactual_classes is not None
    if is_one_pole and feature_class is None and len(target_classes) == 1:
        feature_class = str(target_classes[0])

    args = build_training_arguments(
        backend,
        experiment_dir=str(experiment_dir),
        shared=shared_kw,
        split_mode=split_mode,
        one_pole=is_one_pole,
        metadata={
            "model_key": cfg.model_key,
            "arch": cfg.raw.get("model", {}).get("arch"),
            "task": cfg.task_id,
            "feature_class": feature_class,
            "counterfactual_classes": counterfactual_classes,
        },
    )
    # This builder serves reloads of finished artifacts (fresh training goes
    # through ``_build_trainer``): score them under the label-token protocol they
    # were trained with, not under the current default.
    args.label_token_protocol = artifact_label_token_protocol(experiment_dir)
    # Second copy of the same one-pole pre-prune strip removed with the first;
    # see the comment above. Disabling pre-prune also disables lazy_init, so the
    # encoder was built at full width (26 GiB at 8B).
    if smoke:
        apply_smoke_prune(args)

    merged = _merged_one_pole_frame(bundle)
    use_merged = merged is not None and is_one_pole
    config_kw: Dict[str, Any] = {
        "run_id": experiment_dir.name,
        "target_classes": list(target_classes),
        "all_classes": list(all_classes or bundle.classes),
        "masked_col": "masked",
        "split_col": "split",
        "neutral_data": bundle.neutrals,
        "decoder_eval_export_row_wise_csv": True,
        "img_format": "png",
        "mask_placeholder": resolve_mask_placeholder(bundle, cfg.raw),
    }
    if random_state is not None:
        # ``TextPredictionConfig.random_state`` exists but defaults to None.
        # Passing no value makes unified transition construction choose
        # counterfactual alternatives nondeterministically, independently of
        # TrainingArguments.seed.  Keep the historical study default intact,
        # while allowing reproducibility-sensitive diagnostics to pin it.
        config_kw["random_state"] = int(random_state)
    if use_merged:
        config_kw["data"] = merged
        config_kw["label_col"] = "label"
        config_kw["label_class_col"] = "label_class"
        config_kw["alternative_col"] = "alternative"
        config_kw["alternative_class_col"] = "alternative_class"
        config_kw["decoder_eval_targets"] = "label"
        config_kw["decoder_eval_prob_on_other_class"] = False
    else:
        config_kw["data"] = bundle.data_per_class
    if counterfactual_classes is not None:
        config_kw["counterfactual_classes"] = counterfactual_classes
    if bundle.class_merge_map:
        config_kw["class_merge_map"] = bundle.class_merge_map
    if bundle.class_merge_transition_groups:
        config_kw["class_merge_transition_groups"] = bundle.class_merge_transition_groups

    trainer = TextPredictionTrainer(
        model=cfg.hf_model,
        config=TextPredictionConfig(**config_kw),
        args=args,
        eval_neutral_additional_excluded_words=bundle.excluded_words or None,
    )
    return trainer, args, feature_class, use_merged, shared_kw


def _reload_train_once(
    *,
    backend: Backend,
    cfg: StudyConfig,
    bundle: TaskBundle,
    target_classes: Sequence[str],
    experiment_dir: Path,
    split_mode: SplitMode,
    shared: Dict[str, Any],
    smoke: bool,
    feature_class: Optional[str] = None,
    counterfactual_classes: Optional[Any] = None,
    all_classes: Optional[Sequence[str]] = None,
    previous_methods: Optional[Sequence[Mapping[str, Any]]] = None,
    refresh_encoder_eval: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load a completed artifact without retraining.

    Saved encoder metrics are reused only when stamped with the current
    ``ENCODER_EVAL_RULES_VERSION`` (validation-frozen Spec_n/Excl); otherwise they
    are recomputed from the checkpoint and ``done.json`` is atomically refreshed.
    ``refresh_encoder_eval`` only changes the log wording here -- its effect is the
    task-level re-entry that lets this stale-blob check run at all.
    """
    ckpt = _find_checkpoint_dir(experiment_dir)
    if ckpt is None:
        return None
    trainer, args, feature_class, use_merged, shared_kw = _build_trainer_for_artifact(
        backend=backend,
        cfg=cfg,
        bundle=bundle,
        target_classes=target_classes,
        experiment_dir=experiment_dir,
        split_mode=split_mode,
        shared=shared,
        smoke=smoke,
        feature_class=feature_class,
        counterfactual_classes=counterfactual_classes,
        all_classes=all_classes,
    )
    try:
        load_kwargs = {"device_encoder": "cpu"} if is_cga_backend(backend) else {}
        mwg = trainer.get_model(load_directory=str(ckpt), **load_kwargs)
        if is_cga_backend(backend):
            _mark_cga_encoder_host_only(mwg)
        gradiend = getattr(mwg, "gradiend", None)
        if gradiend is not None and hasattr(gradiend, "_require_built"):
            gradiend._require_built()
    except Exception as exc:
        track_error(exc, context="reload")
        return None

    print(f"  skip existing train {experiment_dir.name} ← {ckpt}", flush=True)
    is_one_pole = feature_class is not None or counterfactual_classes is not None
    methods = list(previous_methods) if previous_methods is not None else None
    if methods is None:
        from study.results_merge import load_previous_results

        prev = load_previous_results(cfg.output_dir)
        methods = list((prev or {}).get("methods") or [])
    required_encoder_eval_version = None
    include_layers = False
    if is_cga_backend(backend):
        from cga_eval import (
            CGA_ENCODER_EVAL_VERSION,
            CGA_LAYERWISE_ENCODER_EVAL_VERSION,
        )

        include_layers = _layerwise_method_enabled(cfg, "cga")
        required_encoder_eval_version = (
            CGA_LAYERWISE_ENCODER_EVAL_VERSION
            if include_layers
            else CGA_ENCODER_EVAL_VERSION
        )
    elif is_caga_backend(backend) and _layerwise_method_enabled(cfg, "caga"):
        from caga_eval import CAGA_LAYERWISE_ENCODER_EVAL_VERSION

        include_layers = True
        required_encoder_eval_version = CAGA_LAYERWISE_ENCODER_EVAL_VERSION
    # Only a blob stamped with the current Spec_n/Excl rules version may be
    # reused; an unstamped (pre-2026-09-23) one holds test-fitted oracle
    # thresholds and is recomputed here. This applies with or without
    # ``refresh_encoder_eval`` on purpose: a refresh over a partly-migrated tree
    # (e.g. a preempted array resumed) then only redoes the artifacts still on the
    # old rules instead of re-scoring finished ones. Recomputing is deterministic,
    # so re-scoring a current blob could not change it; a future eval-logic change
    # must bump ``ENCODER_EVAL_RULES_VERSION`` to invalidate stored blobs.
    causal_only = bool((cfg.raw.get("cli") or {}).get("causal_only"))
    fair = load_encoder_eval_for_reload(
        experiment_dir,
        previous_methods=methods,
        required_version=required_encoder_eval_version,
        # ``--causal-only`` only needs the checkpoint: take whatever encoder eval is
        # stored (stamp or not) and never score, recompute or re-persist one here.
        require_current_rules=not causal_only,
    )
    if causal_only:
        fair = fair or {}
    elif fair is not None:
        src = (
            "done.json"
            if encoder_eval_from_done(
                experiment_dir,
                required_version=required_encoder_eval_version,
                require_current_rules=True,
            )
            else "results.json"
        )
        print(
            f"  skip existing encoder eval {experiment_dir.name} ← {src}",
            flush=True,
        )
        try:
            persist_encoder_eval_in_done(
                experiment_dir,
                fair,
                version=required_encoder_eval_version,
            )
        except Exception as exc:
            track_error(exc, context="persist encoder_eval")
    else:
        if refresh_encoder_eval:
            print(
                f"  refresh encoder eval {experiment_dir.name} ← checkpoint "
                f"(no encoder eval at rules v{ENCODER_EVAL_RULES_VERSION})",
                flush=True,
            )
        else:
            print(
                f"  skip-existing: no current encoder eval for {experiment_dir.name}; recomputing",
                flush=True,
            )
        max_size = (cfg.training or {}).get("encoder_eval_max_size")
        n_boot = int((cfg.training or {}).get("encoder_bootstrap_auc") or 100)
        from cost_timer import cost_timer

        # Test-only migration for the gradient-cosine backends: the stored
        # validation readouts (same direction, same scoring version) already hold
        # the frozen Spec_n/Excl rules and are what the site/layer lock ranks on,
        # so the validation split need not be scored again. Deliberately read
        # WITHOUT the rules-version guard -- this is the stale blob being migrated.
        prior_readouts = None
        if (
            is_cga_backend(backend) or is_caga_backend(backend)
        ) and required_encoder_eval_version is not None:
            prior_readouts = encoder_eval_from_done(
                experiment_dir, required_version=required_encoder_eval_version
            )
            if prior_readouts is not None:
                print(
                    f"  encoder eval {experiment_dir.name}: prior validation readouts "
                    "found (test-only migration if complete)",
                    flush=True,
                )

        with cost_timer(f"encoder_eval:{backend}:{experiment_dir.name}", phase="encode", backend=backend):
            if is_caga_backend(backend):
                # CAGA: read the direction from the checkpoint decoder and recompute
                # encoder metrics via the extractor-based dL/dh cosine (same as
                # _fit_caga_once), not the weight-gradient forward path.
                from caga_eval import caga_layer_slices, fair_caga_encoder_eval
                from cga_eval import direction_from_decoder
                from study.tasks import labeled_df_for_eval

                fair = fair_caga_encoder_eval(
                    mwg,
                    direction_from_decoder(mwg),
                    labeled_df_for_eval(bundle, expand_one_pole=True),
                    bundle.neutrals,
                    target_classes=list(all_classes or bundle.classes or target_classes),
                    class_encoding_direction=getattr(
                        mwg, "feature_class_encoding_direction", None
                    ),
                    max_size=int(max_size) if max_size is not None else None,
                    max_neutral=int(max_size) if max_size is not None else None,
                    n_bootstrap=n_boot,
                    excluded_words=list(bundle.excluded_words or []),
                    layer_slices=caga_layer_slices(mwg) if include_layers else None,
                    prior_readouts=prior_readouts,
                )
            elif is_cga_backend(backend):
                # CGA installs its direction into the decoder (never the encoder --
                # the package's tanh-through-a-module readout reintroduces exactly the
                # gradient-magnitude confound cosine is meant to remove, see cga_eval.py),
                # and the encoder eval isn't persisted separately, so recompute it from
                # the checkpoint's own decoder weight and score CAA-style.
                from cga_eval import (
                    cga_layer_slices,
                    direction_from_decoder,
                    fair_cga_encoder_eval,
                )
                from study.tasks import labeled_df_for_eval

                # The checkpoint's encoder is random/inert and the decoder is
                # already the fitted direction. Keeping the former on CUDA and
                # cloning the latter costs two unnecessary model-width tensors
                # (~52 GiB at 8B) before the first scoring gradient exists.
                cga_model = getattr(mwg, "gradiend", None)
                cga_encoder = getattr(cga_model, "encoder", None)
                if cga_encoder is not None:
                    cga_encoder.to("cpu")
                    import gc as _gc
                    import torch as _torch

                    _gc.collect()
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                direction = direction_from_decoder(mwg, copy=False)
                fair = fair_cga_encoder_eval(
                    mwg,
                    direction,
                    labeled_df_for_eval(bundle, expand_one_pole=True),
                    bundle.neutrals,
                    target_classes=list(all_classes or bundle.classes or target_classes),
                    class_encoding_direction=getattr(
                        mwg, "feature_class_encoding_direction", None
                    ),
                    max_size=int(max_size) if max_size is not None else None,
                    max_neutral=int(max_size) if max_size is not None else None,
                    n_bootstrap=n_boot,
                    excluded_words=list(bundle.excluded_words or []),
                    component_slices=(
                        {
                            f"L{layer}": bounds
                            for layer, bounds in cga_layer_slices(mwg).items()
                        }
                        if include_layers
                        else None
                    ),
                    prior_readouts=prior_readouts,
                )
            else:
                fair = _fair_encoder_eval(
                    trainer,
                    target_classes=list(all_classes or bundle.classes or target_classes),
                    max_size=int(max_size) if max_size is not None else None,
                    n_bootstrap=n_boot,
                    backend=backend,
                    excluded_words=list(bundle.excluded_words or []),
                    activation_site=(cfg.training or {}).get("activation_site"),
                    # Reload of an unchanged checkpoint: reuse the per-row encodings
                    # already on disk instead of re-encoding validation and test.
                    use_cache=True,
                )
        try:
            persist_encoder_eval_in_done(
                experiment_dir,
                fair,
                version=required_encoder_eval_version,
            )
        except Exception as exc:
            track_error(exc, context="persist encoder_eval")
    return {
        "trainer": trainer,
        "encoder_metrics": fair.get("encoder_metrics") or {},
        "readout_metrics": fair.get("readout_metrics") or {},
        "per_class_readouts": fair.get("per_class_readouts") or {},
        "per_component_readouts": fair.get("per_component_readouts") or {},
        "component_keys": fair.get("component_keys") or [],
        "model_path": str(experiment_dir),
        "backend": backend,
        "split_mode": split_mode,
        "target_classes": list(target_classes),
        "counterfactual_classes": counterfactual_classes,
        "data_mode": "merged_one_pole" if use_merged else "data_per_class",
        "ablation": "one_pole" if is_one_pole else "pair",
        "feature_class": feature_class,
        "learning_rate": _reload_learning_rate(
            experiment_dir, float(getattr(args, "learning_rate", float("nan")))
        ),
        "convergent_metric": getattr(args, "convergent_metric", None),
        "selection_metric": getattr(args, "selection_metric", None),
        "reloaded": True,
    }


def _reload_learning_rate(exp_dir: Path, fallback: float) -> float:
    """The learning rate the artifact was trained with (``done.json``); the configured one if unrecorded.

    Matters with ``--tune-lr``: the trained rate is a search result, not the yaml rate the rebuilt
    trainer arguments carry, and a reload must not silently relabel the artifact.
    """
    done = exp_dir / "done.json"
    try:
        value = ((json.loads(done.read_text(encoding="utf-8")).get("extras")) or {}).get("learning_rate")
        return float(value) if value is not None else fallback
    except Exception:
        return fallback


def _reload_convergent_metric(exp_dir: Path) -> Optional[str]:
    """Read the metric used at train time (``done.json``), not the one-pole default."""
    done = exp_dir / "done.json"
    if not done.is_file():
        return None
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
    except Exception:
        return None
    extras = payload.get("extras") or {}
    metric = extras.get("convergent_metric")
    return str(metric).strip() if metric else None


def reload_train_raw_from_artifacts(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    backends: Sequence[str] = ("gradiend", "actiend"),
    split_modes: Sequence[SplitMode] = ("none",),
) -> Dict[str, Any]:
    """Rebuild trainers from ``artifacts/`` so causal/SAE/CAA can run without retraining.

    Returns the same shape as ``run_train_stage`` trainer maps::

        {
          "train_raw_none": {backend: raw},           # prefer pair, else first onepole
          "train_raw_none_by_class": {backend: {cls: raw | [raw, ...]}},
        }

    Every class retains its one-pole trainer and all pair trainers containing
    that class. Loading only one trainer per backend (legacy), or overwriting
    earlier entries in ``by_class``, silently drops causal pole regimes.
    """
    from gradiend.trainer.text.prediction.trainer import (
        TextPredictionConfig,
        TextPredictionTrainer,
    )

    train_raw_none: Dict[str, Dict[str, Any]] = {}
    train_raw_tensors: Dict[str, Dict[str, Any]] = {}
    # A class can participate in its one-pole trainer and multiple pair
    # trainers, so each leaf is either one raw payload or a list of payloads.
    train_raw_none_by_class: Dict[str, Dict[str, Any]] = {}
    train_raw_tensors_by_class: Dict[str, Dict[str, Any]] = {}
    abl = _ablations(cfg, smoke=bool(cfg.raw.get("_smoke")))
    classes = [str(c) for c in bundle.classes]
    task_fac = _task_one_pole_classes(cfg)
    artifacts_root = cfg.output_dir / "artifacts"
    if not artifacts_root.is_dir():
        print(f"reload: no artifacts at {artifacts_root}", flush=True)
        return {
            "train_raw_none": train_raw_none,
            "train_raw_none_by_class": train_raw_none_by_class,
        }

    expanded: List[str] = []
    for b in backends:
        if b == "actiend":
            expanded.append("actiend")
            if _actiend_pre_enabled(cfg) or "actiend_pre" in backends:
                expanded.append("actiend_pre")
        elif str(b) not in expanded:
            expanded.append(str(b))

    def _register_by_class(
        slot: Dict[str, Dict[str, Any]],
        *,
        backend: str,
        key: str,
        raw: Dict[str, Any],
    ) -> None:
        backend_slot = slot.setdefault(backend, {})
        class_key = str(key)
        previous = backend_slot.get(class_key)
        if previous is None:
            backend_slot[class_key] = raw
            return
        previous_list = previous if isinstance(previous, list) else [previous]
        if id(raw) not in {id(item) for item in previous_list}:
            backend_slot[class_key] = previous_list + [raw]

    for backend in expanded:
        lr_backend = "actiend" if backend == "actiend_pre" else backend
        shared = shared_training_kwargs(cfg, backend=lr_backend)  # type: ignore[arg-type]
        if backend == "actiend_pre":
            shared["activation_site"] = "pre_prediction"
            shared["target_activation_site"] = "prediction"
            shared["actiend_source_site"] = "pre_prediction"
            shared["actiend_target_site"] = "prediction"
        candidates: List[Tuple[Path, Dict[str, Any]]] = []
        for split_mode in split_modes:
            if split_mode not in {"none", "tensors"}:
                continue
            if abl["one_pole"]:
                pole_classes = list(task_fac) if task_fac else list(classes)
                for cls in pole_classes:
                    if cls not in classes:
                        continue
                    others = [c for c in classes if c != cls]
                    if not others:
                        continue
                    dirname = artifact_dirname(backend, kind="onepole", key=str(cls), split_mode=split_mode)
                    exp_dir = artifacts_root / dirname
                    if exp_dir.is_dir():
                        candidates.append(
                            (
                                exp_dir,
                                {
                                    "target_classes": [cls],
                                    "ablation": "one_pole",
                                    "feature_class": cls,
                                    "counterfactual_classes": "all",
                                    "split_mode": split_mode,
                                },
                            )
                        )
            if abl["pair"] and len(classes) >= 2:
                # All pairwise artifacts when present (multi-class tasks).
                from itertools import combinations

                pairs = (
                    [tuple(sorted_pair(classes[0], classes[1]))]
                    if len(classes) == 2
                    else [sorted_pair(a, b) for a, b in combinations(classes, 2)]
                )
                for a, b in pairs:
                    dirname = artifact_dirname(backend, kind="pair", key=pair_key(a, b), split_mode=split_mode)
                    exp_dir = artifacts_root / dirname
                    if exp_dir.is_dir():
                        candidates.append(
                            (
                                exp_dir,
                                {
                                    "target_classes": [a, b],
                                    "ablation": "pair",
                                    "feature_class": None,
                                    "counterfactual_classes": None,
                                    "pair": [a, b],
                                    "split_mode": split_mode,
                                },
                            )
                        )

        for exp_dir, meta in candidates:
            ckpt = _find_checkpoint_dir(exp_dir)
            if ckpt is None:
                print(f"reload: no checkpoint under {exp_dir}", flush=True)
                continue
            merged = _merged_one_pole_frame(bundle)
            is_one_pole = meta.get("ablation") == "one_pole" or meta.get("feature_class") is not None
            use_merged = merged is not None and is_one_pole
            config_kw: Dict[str, Any] = {
                "run_id": exp_dir.name,
                "target_classes": list(meta["target_classes"]),
                "all_classes": list(classes),
                "masked_col": "masked",
                "split_col": "split",
                "neutral_data": bundle.neutrals,
                "decoder_eval_export_row_wise_csv": True,
                "img_format": "png",
                "mask_placeholder": resolve_mask_placeholder(bundle, cfg.raw),
            }
            if use_merged:
                config_kw["data"] = merged
                config_kw["label_col"] = "label"
                config_kw["label_class_col"] = "label_class"
                config_kw["alternative_col"] = "alternative"
                config_kw["alternative_class_col"] = "alternative_class"
                config_kw["decoder_eval_targets"] = "label"
                config_kw["decoder_eval_prob_on_other_class"] = False
            else:
                config_kw["data"] = bundle.data_per_class
            if meta.get("counterfactual_classes") is not None:
                config_kw["counterfactual_classes"] = meta["counterfactual_classes"]
            # Keep reload construction identical to the fresh-training and
            # single-artifact reload paths above.  Pronoun tasks deliberately
            # store raw HF classes (1SG/1PL/...) in ``data_per_class`` and rely
            # on these mappings to expose the study classes
            # (singular/plural or 1/2/3).  Omitting them here made
            # ``--refresh-causal`` rebuild a trainer whose ``all_classes`` and
            # data keys disagreed before decoder evaluation even started.
            if bundle.class_merge_map:
                config_kw["class_merge_map"] = bundle.class_merge_map
            if bundle.class_merge_transition_groups:
                config_kw["class_merge_transition_groups"] = (
                    bundle.class_merge_transition_groups
                )

            shared_reload = dict(shared)
            saved_metric = _reload_convergent_metric(exp_dir)
            if saved_metric:
                shared_reload["convergent_metric"] = saved_metric
            else:
                shared_reload["convergent_metric"] = "correlation"

            try:
                current_split = str(meta.get("split_mode") or "none")
                args = build_training_arguments(
                    backend,  # type: ignore[arg-type]
                    experiment_dir=str(exp_dir),
                    shared=shared_reload,
                    split_mode=current_split,
                    one_pole=bool(is_one_pole),
                    metadata={
                        "model_key": cfg.model_key,
                        "arch": cfg.raw.get("model", {}).get("arch"),
                        "task": cfg.task_id,
                        "reloaded": True,
                    },
                )
                trainer = TextPredictionTrainer(
                    model=cfg.hf_model,
                    config=TextPredictionConfig(**config_kw),
                    args=args,
                    eval_neutral_additional_excluded_words=bundle.excluded_words or None,
                )
            except Exception as exc:
                track_error(exc, context="reload build trainer", path=str(exp_dir))
                print(f"reload: skip {exp_dir.name}: {exc}", flush=True)
                continue
            # Do not materialize every artifact's base model while rebuilding
            # the causal worklist.  A Llama pair/one-pole checkpoint owns a
            # complete 8B backbone even when its learned head is tiny.  The
            # old eager reload kept several such backbones alive and caused the
            # late-stage OOM observed for GRADIEND/ACTIEND after successful
            # training.  The causal loop loads and releases each deferred
            # artifact in turn.  CGA also needs this because its head is large.
            # AGIEND/CAGA were added to the study after this gate was written
            # and never joined it -- each of their checkpoints also owns a
            # complete base backbone (a 3-class task builds 6: 3 pairs + 3
            # one-poles), and the same eager-reload accumulation reproduced
            # the identical symptom for AGIEND (~120GB resident the instant
            # causal starts, flat afterward -- see the ravel_continent/race
            # `[cost] causal:agiend:none:*` readings). Note `.startswith("cga")`
            # does NOT match "caga" ("caga" != "cga" + suffix) -- CAGA needs
            # its own explicit entry, not just AGIEND's.
            deferred_model_reload = (
                str(backend) in {"gradiend", "actiend", "actiend_pre", "agiend", "caga"}
                or str(backend).startswith("cga")
            )
            if not deferred_model_reload:
                try:
                    mwg = trainer.get_model(load_directory=str(ckpt))
                    gradiend = getattr(mwg, "gradiend", None)
                    if gradiend is not None and hasattr(gradiend, "_require_built"):
                        gradiend._require_built()
                except Exception as exc:
                    track_error(exc, context="reload get_model")
                    continue
                print(f"reload: {backend} from {ckpt}", flush=True)
            else:
                print(f"reload: deferred {backend} from {ckpt}", flush=True)
            raw = {
                "trainer": trainer,
                "model_path": str(exp_dir),
                "backend": backend,
                "split_mode": current_split,
                "target_classes": list(meta["target_classes"]),
                "counterfactual_classes": meta.get("counterfactual_classes"),
                "data_mode": "merged_one_pole" if use_merged else "data_per_class",
                "ablation": meta.get("ablation"),
                # Required by feature_or_causal_id to retain pair-shaped ids
                # (e.g. actiend_ridge:A-B:A) during --refresh-causal.
                "pair": meta.get("pair"),
                "feature_class": meta.get("feature_class"),
                "reloaded": True,
            }
            # Causal refresh happens after eager training cleanup.  Do not
            # silently throw away the versioned encoder-eval payload while
            # rebuilding the deferred trainer: CGA/CAGA core causal selection
            # must lock aggregate-vs-layer from exactly these saved validation
            # readouts.  A missing or stale layerwise payload deliberately
            # remains empty, so callers mark that representation unavailable
            # instead of falling back to a legacy aggregate result.
            required_encoder_eval_version = None
            if is_cga_backend(backend):
                from cga_eval import (
                    CGA_ENCODER_EVAL_VERSION,
                    CGA_LAYERWISE_ENCODER_EVAL_VERSION,
                )

                required_encoder_eval_version = (
                    CGA_LAYERWISE_ENCODER_EVAL_VERSION
                    if _layerwise_method_enabled(cfg, "cga")
                    else CGA_ENCODER_EVAL_VERSION
                )
            elif is_caga_backend(backend) and _layerwise_method_enabled(cfg, "caga"):
                from caga_eval import CAGA_LAYERWISE_ENCODER_EVAL_VERSION

                required_encoder_eval_version = CAGA_LAYERWISE_ENCODER_EVAL_VERSION
            fair = load_encoder_eval_for_reload(
                exp_dir, required_version=required_encoder_eval_version
            ) or {}
            raw.update(
                {
                    "encoder_metrics": fair.get("encoder_metrics") or {},
                    "readout_metrics": fair.get("readout_metrics") or {},
                    "per_class_readouts": fair.get("per_class_readouts") or {},
                    "per_component_readouts": fair.get("per_component_readouts") or {},
                    "component_keys": fair.get("component_keys") or [],
                }
            )
            if deferred_model_reload:
                raw["_reload_checkpoint"] = str(ckpt)
            target_none = current_split == "none"
            class_bucket = (
                train_raw_none_by_class
                if target_none
                else train_raw_tensors_by_class
            )
            backend_bucket = train_raw_none if target_none else train_raw_tensors
            if is_one_pole:
                _register_by_class(
                    class_bucket,
                    backend=backend,
                    key=str(meta["feature_class"] or meta["target_classes"][0]),
                    raw=raw,
                )
            else:
                for cls in meta["target_classes"]:
                    _register_by_class(
                        class_bucket,
                        backend=backend,
                        key=str(cls),
                        raw=raw,
                    )
            # Prefer pair as the backend-level handle (SAE/CAA backbone); else first.
            if meta.get("ablation") == "pair" or backend not in backend_bucket:
                backend_bucket[backend] = raw

    return {
        "train_raw_none": train_raw_none,
        "train_raw_tensors": train_raw_tensors,
        "train_raw_none_by_class": train_raw_none_by_class,
        "train_raw_tensors_by_class": train_raw_tensors_by_class,
    }


def rehydrate_train_method_rows(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    backends: Sequence[str] = ("gradiend", "actiend"),
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Rebuild GRADIEND/ACTIEND encoder rows from ``artifacts/`` when ``results.json`` lost them."""
    rows: List[Dict[str, Any]] = []
    abl = _ablations(cfg, smoke=smoke)
    if not abl.get("one_pole") and not abl.get("pair"):
        return rows
    classes = [str(c) for c in bundle.classes]
    task_fac = _task_one_pole_classes(cfg)
    artifacts_root = cfg.output_dir / "artifacts"
    if not artifacts_root.is_dir():
        return rows

    expanded: List[str] = []
    for b in backends:
        if b == "actiend":
            expanded.append("actiend")
            if _actiend_pre_enabled(cfg) or "actiend_pre" in backends:
                expanded.append("actiend_pre")
        elif str(b) not in expanded:
            expanded.append(str(b))

    for backend in expanded:
        lr_backend = "actiend" if backend == "actiend_pre" else backend
        shared = shared_training_kwargs(cfg, backend=lr_backend)  # type: ignore[arg-type]
        if backend == "actiend_pre":
            shared["activation_site"] = "pre_prediction"
            shared["target_activation_site"] = "prediction"
            shared["actiend_source_site"] = "pre_prediction"
            shared["actiend_target_site"] = "prediction"
        if abl["one_pole"]:
            pole_classes = list(task_fac) if task_fac else list(classes)
            for cls in pole_classes:
                if cls not in classes:
                    continue
                others = [c for c in classes if c != cls]
                if not others:
                    continue
                dirname = artifact_dirname(
                    backend, kind="onepole", key=str(cls), split_mode="none"
                )
                exp_dir = artifacts_root / dirname
                if not exp_dir.is_dir():
                    continue
                saved_metric = _reload_convergent_metric(exp_dir)
                if saved_metric:
                    shared = dict(shared)
                    shared["convergent_metric"] = saved_metric
                raw = _reload_train_once(
                    backend=backend,  # type: ignore[arg-type]
                    cfg=cfg,
                    bundle=bundle,
                    target_classes=[cls],
                    experiment_dir=exp_dir,
                    split_mode="none",
                    shared=shared,
                    smoke=smoke,
                    feature_class=cls,
                    counterfactual_classes="all",
                    all_classes=classes,
                )
                if raw is None:
                    continue
                raw["ablation"] = "one_pole"
                raw["feature_class"] = cls
                rows.extend(
                    _rows_from_onepole_train(backend=backend, cls=cls, raw=raw, cfg=cfg)
                )
                break  # one pole per backend
        elif abl["pair"] and len(classes) >= 2:
            a, b = sorted_pair(classes[0], classes[1])
            dirname = artifact_dirname(
                backend, kind="pair", key=pair_key(a, b), split_mode="none"
            )
            exp_dir = artifacts_root / dirname
            if not exp_dir.is_dir():
                continue
            saved_metric = _reload_convergent_metric(exp_dir)
            if saved_metric:
                shared = dict(shared)
                shared["convergent_metric"] = saved_metric
            raw = _reload_train_once(
                backend=backend,  # type: ignore[arg-type]
                cfg=cfg,
                bundle=bundle,
                target_classes=[a, b],
                experiment_dir=exp_dir,
                split_mode="none",
                shared=shared,
                smoke=smoke,
                all_classes=classes,
            )
            if raw is None:
                continue
            rows.extend(
                _rows_from_pair_train(backend=backend, a=a, b=b, raw=raw, cfg=cfg)
            )
    if rows:
        print(
            f"rehydrate: rebuilt {len(rows)} encoder method rows from {artifacts_root}",
            flush=True,
        )
    return rows
