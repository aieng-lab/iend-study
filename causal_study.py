"""Shared LMS-gated causal study (GRADIEND / ACTIEND / SAE / CAA).

Ported from the former SAE-engine ``run_study_causal`` so the deep pipeline
and the gender PoC share the same implementations. Suites explicitly select
which costly ACTIEND policy rollouts to execute.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from study.method_ids import feature_or_causal_id
from study.validation_selection import lock_by_validation_detection
from study.decoder_frames import decoder_eval_frames_for_study
from study.causal_policies import (
    ACTIEND_CAUSAL_POLICY_DEFAULTS,
    actiend_ablation_policy_specs,
    actiend_default_policy_enabled,
    normalize_actiend_causal_policies,
)
from error_tracker import track_error
from study.signals.decoder_causal import (
    apply_decoder_causal_variant,
    fit_checkpoint_ridge_decoders,
    ridge_causal_variants,
)

from caa_eval import (
    CAA_ACT_POLICIES,
    caa_encoder_id,
    load_caa_vectors_payload,
    run_caa_causal_sweep,
    specs_for_part,
)
from causal_eval import (
    CausalMethodResult,
    DECODER_EVAL_MAX_SIZE,
    DEFAULT_CAUSAL_STRENGTHS,
    _trainer_claim_classes,
    bind_decoder_selection_to_evaluation_grid,
    causal_method_result_from_dict,
    compute_lms_safe,
    decoder_selected_learning_rates,
    evaluate_decoder_for_classes_refined,
    headline_metrics,
    invalidate_decoder_grid_cache,
    meta_rows_to_frame,
    persist_causal_bundle,
    run_encoder_causal_for_class,
    sae_decoder_bag_direction,
    sae_decoder_direction,
    run_sae_causal_sweep,
    run_sae_clamp_causal_sweep,
    run_sae_multi_layer_causal_sweep,
    score_class_probs_by_dataset,
    stratified_causal_texts,
)
from cost_timer import CostSection
from study.causal_policies import ACTIEND_GATE_DEFAULT, ACTIEND_TOK_DEFAULT
from sae_eval import (
    iter_sae_site_payloads,
    load_sae,
    method_part_id,
    resid_sites,
    sae_causal_id,
    sae_encode_classes,
    sae_encoder_all_k_id,
    sae_joint_method_id,
    sae_part_for_site,
    sae_backend_for_site,
    short_component_part,
)

# Backward-compatible public inventory. Execution is filtered by
# ``actiend_causal_policies`` instead of always running both entries.
ACTIEND_RIDGE_VARIANT_ID = "ridge_tanh_centered"
SAE_FIXED_KS_DEFAULT = (1, 2, 4, 8, 16, 32, 64, 128)
CAUSAL_ABLATIONS_TODO = (
    "Encoder↔causal aligned ids; ACTIEND tok_*; SAE fixed-k + all_k + kstar "
    "+ opp_fire + arad_out + jh_f1 + joint."
)
SAE_SELECTION_OPPOSITE_CLASS_TODO = (
    "Opp-fire ranking + layer pick: sae:{cls}:sel_opp_fire (encoder + causal, same id)."
)
SAE_SELECTION_ARAD_TODO = (
    "Arad output-score ranking + layer pick: sae:{cls}:sel_arad_out "
    "(encoder + causal; adapted from technion-cs-nlp/saes-are-good-for-steering)."
)
SAE_SELECTION_JH_TODO = (
    "JH calibrated-F1 supervised ranking + layer pick: sae:{cls}:sel_jh_f1 "
    "(encoder + causal; adapted from MikkelGodsk/SAE-labelling)."
)
CAUSAL_LAYER_LR_CEILING_TODO = (
    "Per-layer ACTIEND causal uses the same LR grid as full-model; "
    "layer-local ceiling may need a denser/higher grid."
)




def selected_caa_part_by_validation_detection(
    method_metrics: Mapping[str, Any],
    *,
    cls: str,
    policy: str,
    pair: Optional[Sequence[str]],
    layers: Sequence[int],
) -> Optional[str]:
    """Return CAA's headline site using the report's validation lock.

    The ``"__concat__"`` sentinel is the all-layer concat site and ``"L<n>"``
    is a single residual site. ``None`` means the candidate pool was incomplete,
    so no headline causal row may be emitted.
    """
    candidates: List[Dict[str, Any]] = []
    # One aggregate candidate plus the individual layers. ``part=None`` is the
    # cosine in the concatenated all-layer representation, matching the CGA and
    # CAGA aggregate cosine. ``all`` (mean of layer cosines) remains an
    # appendix diagnostic, not a second all-layer headline lottery ticket.
    for part in (None, *[f"L{int(layer)}" for layer in layers]):
        method_id = caa_encoder_id(cls, policy=policy, part=part, pair=pair)
        metrics = method_metrics.get(method_id)
        if not isinstance(metrics, Mapping):
            return None
        candidates.append({"method": method_id, "metrics": dict(metrics), "part": part})
    selected = lock_by_validation_detection(candidates)
    if selected is None:
        return None
    # Sentinel makes concat (part=None) distinguishable from an incomplete pool.
    return str(selected["part"]) if selected["part"] is not None else "__concat__"


def selected_direction_part_by_validation_detection(
    raw: Mapping[str, Any],
    *,
    cls: str,
) -> Optional[str]:
    """Lock a CGA/CAGA aggregate-or-layer candidate from validation Det.

    The aggregate readout is stored in ``per_class_readouts`` and optional
    layer slices in ``per_component_readouts``. Returning ``None`` is
    deliberately ambiguous-free: no complete validation candidate pool means
    no core causal row, rather than a silent aggregate fallback.
    """
    aggregate = (raw.get("per_class_readouts") or {}).get(str(cls))
    components = raw.get("per_component_readouts") or {}
    if not isinstance(aggregate, Mapping) or not isinstance(components, Mapping):
        return None
    candidates: List[Dict[str, Any]] = [
        {"method": "aggregate", "metrics": dict(aggregate), "part": "aggregate"}
    ]
    for part, class_map in components.items():
        metrics = (class_map or {}).get(str(cls)) if isinstance(class_map, Mapping) else None
        if not isinstance(metrics, Mapping):
            return None
        candidates.append({"method": str(part), "metrics": dict(metrics), "part": str(part)})
    # A core layer-selection request with no layer rows is incomplete, not an
    # aggregate-only candidate set. Callers enforce that requested invariant.
    if len(candidates) < 2:
        return None
    selected = lock_by_validation_detection(candidates)
    return None if selected is None else str(selected["part"])


def _int_layer(value, default=None):
    """Parse a residual layer index; the virtual ``all`` site is not a layer."""
    if value is None or str(value) == "all":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def causal_group_already_complete(
    method_ids: Sequence[str], skip: Set[str]
) -> bool:
    """True when skip-existing can omit a decoder grid / sweep group."""
    if not skip:
        return False
    ids = [str(x) for x in method_ids if x]
    return bool(ids) and all(i in skip for i in ids)


def model_tok_from_trainers(
    train_raw_none: Optional[Mapping[str, Any]] = None,
    none_by_class: Optional[Mapping[str, Any]] = None,
):
    """First live ``(base_model, tokenizer)`` from reloaded trainer payloads."""
    payloads: List[Any] = []
    if train_raw_none:
        payloads.extend(train_raw_none.values())
    if none_by_class:
        for cls_map in none_by_class.values():
            if cls_map:
                payloads.extend(cls_map.values())
    for any_raw in payloads:
        trainer = (any_raw or {}).get("trainer") if isinstance(any_raw, Mapping) else None
        if trainer is None:
            continue
        mwg = trainer.get_model()
        model = getattr(mwg, "base_model", None)
        tok = getattr(trainer, "tokenizer", None)
        if model is not None and tok is not None:
            return model, tok
    return None, None


def load_hf_backbone_for_causal(study_model_key: Optional[str] = None):
    """Load the study HF model when SAE/CAA causal has no trainer in ``--methods``."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from study_models import active_model, set_active_model

    if study_model_key:
        set_active_model(str(study_model_key))
    name = str(active_model().hf_model)
    print(
        f"  causal: loading HF backbone {name!r} (no trainer in --methods)",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slurm_gpu_job = bool(
        os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE")
    )
    if slurm_gpu_job and device.type != "cuda":
        raise RuntimeError(
            "Slurm allocated a GPU but torch.cuda.is_available() is false; "
            "refusing to run causal inference silently on CPU"
        )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    actual_device = getattr(model, "device", device)
    if slurm_gpu_job and str(actual_device).split(":", 1)[0] != "cuda":
        raise RuntimeError(
            f"Causal HF backbone remained on {actual_device} despite a Slurm GPU allocation"
        )
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print(
        f"  causal: HF backbone device={actual_device}"
        + (f" gpu={gpu_name!r}" if gpu_name else ""),
        flush=True,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return model, tok


def causal_eval_n_per_group(*, n_per_group: int, decoder_max_size: int) -> int:
    """Common per-group cap for package and direct causal evaluators."""
    n = int(n_per_group)
    maximum = int(decoder_max_size)
    if n <= 0 or maximum <= 0:
        raise ValueError("causal evaluation sizes must be positive")
    return min(n, maximum)


def resolve_causal_backbone(
    *,
    live_model=None,
    live_tokenizer=None,
    train_raw_none: Optional[Mapping[str, Any]] = None,
    none_by_class: Optional[Mapping[str, Any]] = None,
    study_model_key: Optional[str] = None,
    load_hf: bool = True,
):
    """HF handles for SAE/CAA causal: live encode → trainer → ``from_pretrained``.

    ``METHODS=sae`` (or ``caa``) still needs a model. Do not skip the sweep.
    """
    if live_model is not None and live_tokenizer is not None:
        return live_model, live_tokenizer
    model, tok = model_tok_from_trainers(train_raw_none, none_by_class)
    if model is not None and tok is not None:
        return model, tok
    if not load_hf:
        return None, None
    return load_hf_backbone_for_causal(study_model_key)


def _activation_module_name(component_key: dict) -> str:
    """HF module path for ACTIEND ``activation_modules`` (strip ``activation:``)."""
    raw = str(component_key.get("component_id") or component_key.get("component_label") or "")
    if raw.startswith("activation:"):
        return raw[len("activation:") :]
    return raw


def _extract_ridge_splits_from_trainer(
    trainer,
    *,
    checkpoint_source: str,
    max_size_per_group: int,
    max_samples: int,
    extraction_seed: int = 0,
):
    """Extract paired train/val/test signal batches for post-hoc ridge fitting."""
    import random

    import numpy as np
    import torch

    from study.signals.package_adapter import (
        PairedSignalExtraction,
        extract_paired_signals_from_trainer,
    )

    batches: Dict[str, Any] = {}
    learned = None
    for split_index, split in enumerate(("train", "validation", "test")):
        seed = int(extraction_seed) + 104729 * split_index
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        extracted = extract_paired_signals_from_trainer(
            trainer,
            split=split,
            max_size_per_group=max_size_per_group,
            max_samples=max_samples,
            projection_dim=None,
            include_learned=split == "test",
            checkpoint_source=checkpoint_source,
        )
        if isinstance(extracted, PairedSignalExtraction):
            batches[split] = extracted.batch
            learned = extracted.learned
        else:
            batches[split] = extracted
    if learned is None:
        raise RuntimeError("ACTIEND ridge extraction did not return learned measurements")
    return batches, learned


def _record_decoder_artifacts(
    catalog: list,
    *,
    method_label: str,
    backend: str,
    part: str | None,
    activation_modules: str | None,
    decoder_results: dict | None,
    experiment_dir: str | Path | None = None,
) -> None:
    """Collect GRADIEND API decoder plot/CSV/JSON paths for the study catalog."""
    if not decoder_results:
        return
    plot_paths = list(decoder_results.get("plot_paths") or [])
    if decoder_results.get("plot_path") and decoder_results["plot_path"] not in plot_paths:
        plot_paths.append(decoder_results["plot_path"])
    weaken_results = decoder_results.get("weaken_results") or {}
    weaken_plot_paths = list(weaken_results.get("plot_paths") or [])
    if weaken_results.get("plot_path") and weaken_results["plot_path"] not in weaken_plot_paths:
        weaken_plot_paths.append(weaken_results["plot_path"])
    entry = {
        "method": method_label,
        "backend": backend,
        "part": part,
        "activation_modules": activation_modules,
        "plot_paths": [str(p) for p in plot_paths if p],
        "weaken_plot_paths": [str(p) for p in weaken_plot_paths if p],
        "raw_output_path": (
            str(decoder_results["raw_output_path"])
            if decoder_results.get("raw_output_path")
            else None
        ),
        "output_path": (
            str(decoder_results["output_path"])
            if decoder_results.get("output_path")
            else None
        ),
        "weaken_output_path": (
            str(weaken_results["output_path"])
            if weaken_results.get("output_path")
            else None
        ),
        "experiment_dir": str(experiment_dir) if experiment_dir else None,
    }
    # Common package cache names under experiment_dir when paths omitted.
    if experiment_dir is not None:
        exp = Path(experiment_dir)
        grid = exp / "decoder_grid_cache.json"
        if grid.is_file() and not entry["output_path"]:
            entry["output_path"] = str(grid)
        raw_dir = exp / "decoder_raw"
        if raw_dir.is_dir() and not entry["raw_output_path"]:
            csvs = sorted(raw_dir.glob("*_raw_samples.csv"))
            if csvs:
                entry["raw_output_path"] = str(csvs[-1])
            elif (exp / "decoder_row_wise_scores.csv").is_file():
                entry["raw_output_path"] = str(exp / "decoder_row_wise_scores.csv")
    catalog.append(entry)

def iter_trainer_class_groups(
    train_raw_by_backend: Optional[Mapping[str, Dict[str, Any]]],
    *,
    target_classes: Sequence[str],
    train_raw_by_class: Optional[Mapping[str, Mapping[str, Dict[str, Any]]]] = None,
) -> Iterator[Tuple[str, Dict[str, Any], List[str]]]:
    """Yield ``(backend, raw, classes)`` for causal decoder×class sweeps.

    When ``train_raw_by_class`` is provided, each class uses its own pair/one-pole
    trainer (same raw object shared across classes is evaluated once). Otherwise
    falls back to legacy: one trainer per backend × all ``target_classes``.

    ``classes`` is intersected with the trainer's claim ``target_classes`` so
    one-pole CF names (DISTRACTOR/OTHER/…) are never passed to decoder eval.
    """
    targets = [str(c) for c in target_classes]
    target_set = set(targets)

    def _claim_filter(raw: Dict[str, Any], clss: List[str]) -> List[str]:
        claim = _trainer_claim_classes(raw.get("trainer"))
        if not claim:
            return list(clss)
        claim_set = set(claim)
        filtered = [c for c in clss if c in claim_set]
        return filtered if filtered else list(claim)

    if train_raw_by_class:
        for backend, cls_map in train_raw_by_class.items():
            if not cls_map:
                continue
            grouped: Dict[int, Tuple[Dict[str, Any], List[str]]] = {}
            order: List[int] = []
            for cls, raw_or_list in cls_map.items():
                cls_s = str(cls)
                if cls_s not in target_set:
                    continue
                raws = raw_or_list if isinstance(raw_or_list, list) else [raw_or_list]
                for raw in raws:
                    if not isinstance(raw, dict) or raw.get("trainer") is None:
                        continue
                    key = id(raw)
                    if key not in grouped:
                        grouped[key] = (raw, [])
                        order.append(key)
                    if cls_s not in grouped[key][1]:
                        grouped[key][1].append(cls_s)
            for key in order:
                raw, clss = grouped[key]
                yield str(backend), raw, _claim_filter(raw, clss)
        return
    for backend, raw in (train_raw_by_backend or {}).items():
        if not isinstance(raw, dict) or raw.get("trainer") is None:
            continue
        yield str(backend), raw, _claim_filter(raw, list(targets))


def offload_inert_cga_encoder_for_causal(
    trainer: Any,
    backend: str,
    *,
    min_bytes: int = 2 * 1024**3,
) -> bool:
    """Move CGA's unused full-width encoder off CUDA before decoder sweeps.

    CGA encoding is scored directly with cosine similarity before the causal
    stage.  Its package encoder remains random and is never consulted by the
    decoder intervention, but reloading a large checkpoint places that inert
    matrix back on CUDA.  At 8B scale it occupies about 26 GiB and prevents the
    causal context from materializing/applying the equally wide decoder update.
    """
    if not str(backend).startswith("cga"):
        return False
    mwg = trainer.get_model()
    gradiend_model = getattr(mwg, "gradiend", None)
    encoder = getattr(gradiend_model, "encoder", None)
    if encoder is None:
        return False
    params = list(encoder.parameters())
    encoder_bytes = sum(p.numel() * p.element_size() for p in params)
    if encoder_bytes < int(min_bytes) or not any(p.device.type == "cuda" for p in params):
        return False

    import torch

    encoder.to("cpu")
    setattr(mwg, "_cga_encoder_offloaded", True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(
        "  cga causal: offloaded inert encoder to cpu "
        f"({encoder_bytes / 1024**3:.2f} GiB) and released CUDA cache",
        flush=True,
    )
    return True


def load_deferred_trainer_for_causal(
    trainer: Any,
    raw: Mapping[str, Any],
    backend: str,
) -> bool:
    """Load one deferred checkpoint immediately before its causal sweep.

    Large GRADIEND/ACTIEND artifacts are deferred for the same reason as CGA:
    retaining several complete base backbones during artifact rehydration is a
    deterministic late-stage VRAM leak.  CGA additionally puts its inert,
    full-width encoder on CPU during construction.
    """
    checkpoint = raw.get("_reload_checkpoint")
    if not checkpoint:
        return False
    # The encoder is random/inert for CGA. Constructing both model-width
    # matrices on CUDA during reload peaks above H200 capacity before the
    # post-load offload can run, so place the encoder on CPU at construction.
    load_kwargs = {"device_encoder": "cpu"} if str(backend).startswith("cga") else {}
    mwg = trainer.get_model(load_directory=str(checkpoint), **load_kwargs)
    gradiend_model = getattr(mwg, "gradiend", None)
    if gradiend_model is not None and hasattr(gradiend_model, "_require_built"):
        gradiend_model._require_built()
    if str(backend).startswith("cga") and gradiend_model is not None:
        # CGA always stores label * (grad factual - grad alternative), even
        # when the surrounding trainer used another source. Repair legacy
        # checkpoint metadata before deriving decoder feature factors.
        from cga_eval import set_cga_direction_semantics

        set_cga_direction_semantics(mwg)
        gradiend_model.device_encoder = mwg._get_base_forward_device()
        setattr(mwg, "_cga_encoder_offloaded", True)
    print(f"  {backend} causal: loaded deferred checkpoint {checkpoint}", flush=True)
    return True


def release_deferred_trainer_after_causal(
    trainer: Any,
    raw: Mapping[str, Any],
    backend: str,
) -> bool:
    """Release one deferred model without copying its base backbone to host."""
    if not raw.get("_reload_checkpoint"):
        return False
    model = getattr(trainer, "_model_instance", None)
    trainer._model_instance = None
    trainer._model_manually_unloaded = True
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        track_error(exc, context="release deferred trainer")
    print(f"  {backend} causal: released deferred checkpoint", flush=True)
    return True


# Kept as compatibility aliases for the CGA-specific helper names used by
# downstream callers/tests before deferred reload was extended to IEND.
load_deferred_cga_trainer_for_causal = load_deferred_trainer_for_causal
release_deferred_cga_trainer_after_causal = release_deferred_trainer_after_causal


def run_study_causal(
    train_raw_none: dict,
    sae_raw: dict,
    labeled_df,
    neutral_df,
    *,
    target_classes: Sequence[str],
    output_dir: str | Path,
    study_model_key: Optional[str] = None,
    train_raw_tensors: dict | None = None,
    train_raw_none_by_class: dict | None = None,
    train_raw_tensors_by_class: dict | None = None,
    caa_raw: dict | None = None,
    enabled: Optional[Set[str]] = None,
    sae_release: Optional[str] = None,
    sae_layers: Sequence[int] = (),
    sae_fixed_ks: Sequence[int] = SAE_FIXED_KS_DEFAULT,
    causal_strengths: Sequence[float] = DEFAULT_CAUSAL_STRENGTHS,
    sae_causal_strengths: Sequence[float] | None = None,
    caa_causal_strengths: Sequence[float] | None = None,
    caa_causal_policies: Sequence[str] = CAA_ACT_POLICIES,
    actiend_causal_policies: Sequence[str] = ACTIEND_CAUSAL_POLICY_DEFAULTS,
    actiend_decoder_lrs: Sequence[float] | None = None,
    n_per_group: int = 100,
    val_n_per_group: Optional[int] = None,
    decoder_max_size: int = DECODER_EVAL_MAX_SIZE,
    val_decoder_max_size: Optional[int] = None,
    actiend_ridge_max_size_per_group: int = 1000,
    actiend_ridge_max_samples: int = 3000,
    use_cache: bool = False,
    save_modified_models: bool = False,
    n_model_layers: Optional[int] = None,
    persist_curve_samples: bool = False,
    persist_sample_text: bool = False,
    persist_max_samples: Optional[int] = None,
    skip_causal_methods: Optional[Set[str]] = None,
    reuse_strengthen_by_method: Optional[Mapping[str, Mapping[str, Any]]] = None,
    polarity_repair_by_method: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sae_opp_fire_enabled: bool = False,
    sae_arad_enabled: bool = False,
    sae_jh_f1_enabled: bool = False,
    sae_clamp_enabled: bool = False,
    sae_kstar_enabled: bool = True,
    fail_fast: bool = False,
    causal_protocol_version: Optional[str] = None,
    direction_polarity_protocol_version: Optional[int] = None,
    cga_causal_grid_protocol_version: Optional[int] = None,
    agiend_causal_grid_protocol_version: Optional[int] = None,
    layerwise_cga: bool = False,
    layerwise_caga: bool = False,
    primary_only: bool = False,
    only_method_id_patterns: Optional[Sequence[str]] = None,
) -> dict:
    """LMS-gated causal for gradiend/actiend/sae × class, plus tensors ablations.

    GRADIEND/ACTIEND (none-split): decoder eval grid/cache (LR / P(class) plots).
    GRADIEND/ACTIEND (tensors aggregate): matched ``:{cls}:tensors`` causal on all
    trained sites (same evaluate_decoder path, no activation_modules filter).
    ACTIEND (tensors per-layer): one ``evaluate_decoder(activation_modules=…)``
    per layer so each ``actiend:{cls}:L*`` gets decoder plots/CSV/JSON and a causal row.
    SAE: package mask-slot scoring + durable all-token hooks + package LMS.
    ``primary_only`` is the core execution budget: it retains validation-selected
    SAE and CAA rows, while full suites retain layerwise and causal-only appendix
    ablations.

    All ``evaluate_decoder`` calls hardcode ``use_cache=False``. None-split ACTIEND
    Selected token_selector/gate ablations share one
    ``experiment_dir/decoder_grid_cache.json``
    (no per-ablation ``output_path``); enabling package cache would leave only the
    last write on disk. Resume at experiment level instead (PLANNING.md §8.6).
    ``use_cache`` is kept for API compat and ignored.
    """
    del use_cache  # evaluate_decoder always use_cache=False (shared-path overwrite)
    if study_model_key:
        # Avoid cross-run global-state leakage: always bind active model for this run.
        from study_models import set_active_model

        set_active_model(str(study_model_key))
        try:
            print(
                f"causal: active study model={study_model_key} hook_L0={resid_sites(0)[0]}",
                flush=True,
            )
        except Exception:
            pass

    TARGET_CLASSES = [str(c) for c in target_classes]
    OUTPUT_DIR = Path(output_dir)
    decoder_catalog: list = []
    # Array-by-method cells for one model/task run concurrently. Give each
    # Slurm job its own file so read/modify/write checkpoints cannot clobber
    # another method family's progress.
    progress_owner = os.environ.get("SLURM_JOB_ID") or f"pid-{os.getpid()}"
    progress_path = OUTPUT_DIR / "causal" / f"progress-{progress_owner}.json"

    def _persist_incremental_progress(items) -> None:
        """Atomically checkpoint every completed method for timeout-safe resume."""
        import json

        progress_path.parent.mkdir(parents=True, exist_ok=True)
        prior_by_method = {}
        if progress_path.is_file():
            try:
                prior = json.loads(progress_path.read_text(encoding="utf-8"))
                if (
                    prior.get("protocol_version") == causal_protocol_version
                    and int(prior.get("direction_polarity_protocol_version") or 1)
                    == int(direction_polarity_protocol_version or 1)
                    and int(prior.get("cga_causal_grid_protocol_version") or 1)
                    == int(cga_causal_grid_protocol_version or 1)
                    and int(prior.get("agiend_causal_grid_protocol_version") or 1)
                    == int(agiend_causal_grid_protocol_version or 1)
                ):
                    prior_by_method = dict(prior.get("by_method") or {})
            except Exception:
                # The atomic replace below repairs a stale/corrupt checkpoint.
                prior_by_method = {}
        current_by_method = {
            str(item.method): item.to_dict()
            for item in items
            if getattr(item, "method", None)
        }
        payload = {
            "protocol_version": causal_protocol_version,
            "direction_polarity_protocol_version": direction_polarity_protocol_version,
            "cga_causal_grid_protocol_version": cga_causal_grid_protocol_version,
            "agiend_causal_grid_protocol_version": agiend_causal_grid_protocol_version,
            "by_method": {**prior_by_method, **current_by_method},
        }
        tmp = progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(progress_path)

    class _CheckpointingResults(list):
        def append(self, item) -> None:
            super().append(item)
            _persist_incremental_progress(self)
            print(
                f"  causal: checkpointed {len(self)} completed method(s) "
                f"this run to {progress_path}",
                flush=True,
            )

    results = _CheckpointingResults()

    def _persist_partial_before_abort() -> None:
        """Checkpoint successful causal rows before fail-fast propagates."""
        try:
            persist_causal_bundle(OUTPUT_DIR, results)
            (OUTPUT_DIR / "decoder_artifacts.json").write_text(
                __import__("json").dumps(decoder_catalog, indent=2),
                encoding="utf-8",
            )
        except Exception as checkpoint_exc:
            print(f"  causal: partial checkpoint failed: {checkpoint_exc}", flush=True)
    CAUSAL_STRENGTHS = list(causal_strengths)
    SAE_CAUSAL_STRENGTHS = tuple(sae_causal_strengths or causal_strengths)
    CAA_CAUSAL_STRENGTHS = tuple(caa_causal_strengths or causal_strengths)
    CAA_CAUSAL_POLICIES = tuple(caa_causal_policies)
    ACTIEND_CAUSAL_POLICIES = normalize_actiend_causal_policies(
        actiend_causal_policies
    )
    # The default ACTIEND policy is now ungated all-token (``tok_all``); it is
    # computed when ``all`` is requested. ``gated_all``/``prediction`` are ablations.
    ACTIEND_DEFAULT_ENABLED = actiend_default_policy_enabled(ACTIEND_CAUSAL_POLICIES)
    ACTIEND_SELECTED_ABLATIONS = tuple(
        {"token_selector": tok, "activation_gate": gate}
        for tok, gate in actiend_ablation_policy_specs(ACTIEND_CAUSAL_POLICIES)
    )
    ACTIEND_DECODER_LRS = list(actiend_decoder_lrs or causal_strengths)
    CAUSAL_N = int(n_per_group)
    DECODER_MAX_SIZE = int(decoder_max_size)
    # Validation-only decoder-frame cap for GRADIEND/ACTIEND. The decoder GRID
    # (all LRs x feature_factors) is built on the validation frame purely to select
    # the LR; the frozen selection is reported on the full-size TEST frame. So a
    # smaller validation frame gives the same selected LR at a fraction of the grid
    # cost, with the reported effect unchanged (full test). Mirrors val_n_per_group
    # for the CAA-style path. None => validation stays at DECODER_MAX_SIZE.
    VAL_DECODER_MAX_SIZE = (
        max(1, min(DECODER_MAX_SIZE, int(val_decoder_max_size)))
        if val_decoder_max_size
        else DECODER_MAX_SIZE
    )
    # Independent of CAUSAL_N/DECODER_MAX_SIZE: those are tuned per-task for the
    # strengthen/LMS grid sweep (repeated across ~20 strengths x every backend),
    # not for how much data a one-shot closed-form ridge regression needs to
    # estimate a ~q-dim decoder direction. Reusing CAUSAL_N here used to silently
    # cap the ridge fit at whatever a task's (possibly tiny, PoC-inherited) causal
    # pool happened to be -- e.g. 64 rows/group for the circuit tasks -- even
    # though the underlying task corpora have 10k+ rows available.
    ACTIEND_RIDGE_FIT_MAX_SIZE_PER_GROUP = int(actiend_ridge_max_size_per_group)
    ACTIEND_RIDGE_FIT_MAX_SAMPLES = int(actiend_ridge_max_samples)
    SAVE_MODIFIED_MODELS = bool(save_modified_models)
    SAE_LAYERS = tuple(int(x) for x in (sae_layers or ()))
    SAE_FIXED_KS = tuple(int(k) for k in sae_fixed_ks)
    SAE_RELEASE = sae_release
    SAE_LAYER = int(SAE_LAYERS[-1]) if SAE_LAYERS else 0
    SAE_TOKEN_SELECTOR_ABLATIONS = ("prediction",)
    SAE_OPP_FIRE_ENABLED = bool(sae_opp_fire_enabled)
    SAE_ARAD_ENABLED = bool(sae_arad_enabled)
    SAE_CLAMP_ENABLED = bool(sae_clamp_enabled)
    SAE_KSTAR_ENABLED = bool(sae_kstar_enabled)
    SAE_JH_F1_ENABLED = bool(sae_jh_f1_enabled)
    _skip_causal = {str(x) for x in (skip_causal_methods or ())}
    if only_method_id_patterns:
        # Causal-only rerun: enumerate the full (non-primary) id space but treat
        # every id outside the whitelist as already done, so only the named
        # sweeps run.  See study.causal_policies.SkipUnlessWanted.
        from study.causal_policies import SkipUnlessWanted

        primary_only = False
        _skip_causal = SkipUnlessWanted(_skip_causal, only_method_id_patterns)
    _reuse_strengthen = {
        str(mid): causal_method_result_from_dict(payload)
        for mid, payload in (reuse_strengthen_by_method or {}).items()
    }
    _polarity_repairs = {
        str(mid): causal_method_result_from_dict(payload)
        for mid, payload in (polarity_repair_by_method or {}).items()
    }

    def _reuse_for(method_id: str) -> Optional[CausalMethodResult]:
        return _reuse_strengthen.get(str(method_id))

    def _polarity_repair_for(method_id: str) -> Optional[CausalMethodResult]:
        return _polarity_repairs.get(str(method_id))

    def _all_polarity_repairable(method_ids: Sequence[str]) -> bool:
        ids = [str(mid) for mid in method_ids]
        return bool(ids) and all(mid in _polarity_repairs for mid in ids)

    def _all_strengthen_reusable(method_ids: Sequence[str]) -> bool:
        ids = [str(mid) for mid in method_ids]
        return bool(ids) and all(
            mid in _reuse_strengthen or mid in _polarity_repairs for mid in ids
        )

    def _missing_strengthen_classes(
        method_ids: Sequence[str], class_ids: Sequence[str]
    ) -> List[str]:
        return [
            str(cls)
            for mid, cls in zip(method_ids, class_ids)
            if str(mid) not in _reuse_strengthen
            and str(mid) not in _polarity_repairs
        ]
    print(
        f"causal: ACTIEND policies={list(ACTIEND_CAUSAL_POLICIES)}",
        flush=True,
    )
    # CGA variants each arrive as their own backend id in ``train_raw_none``
    # (``cga`` / ``cga_tensor_norm``); the family name alone would filter them
    # all out. They take the GRADIEND weight-space branch below -- ``_is_actiend``
    # is False for them -- since a CGA checkpoint is a GRADIEND checkpoint whose
    # weights were computed rather than trained.
    from study.training_profiles import (
        expand_cga_backends,
        is_agiend_backend,
        is_caga_backend,
        is_cga_backend,
    )

    enabled_set = expand_cga_backends(
        enabled if enabled is not None
        else {"gradiend", "actiend", "sae", "caa", "causal"}
    )

    def _encoder_mids(
        backend,
        raw,
        class_ids,
        *,
        layer=None,
        token_selector=None,
        activation_gate=None,
        actiend_tokens: bool = False,
    ) -> List[str]:
        ids: List[str] = []
        for cls in class_ids:
            kw: Dict[str, Any] = {}
            if layer is not None:
                kw["layer"] = layer
            if actiend_tokens:
                kw["token_selector"] = token_selector
                kw["activation_gate"] = activation_gate
            ids.append(feature_or_causal_id(backend, cls, raw, **kw))
        return ids

    def _train_backend_enabled(key: str) -> bool:
        k = str(key)
        if k in enabled_set:
            return True
        if k == "actiend" and "actiend_ridge" in enabled_set:
            return True
        # actiend_pre is opt-in (training.actiend_pre); keep trainers if present.
        return k == "actiend_pre" and (
            "actiend_pre" in enabled_set or "actiend" in enabled_set
        )

    # Keep every reloaded trainer for SAE/CAA backbone lookup. Loop filtering
    # below still honors --methods (METHODS=sae must not skip SAE causal).
    _backbone_none = dict(train_raw_none or {})
    _backbone_none_by_class = dict(train_raw_none_by_class or {})
    train_raw_none = {
        k: v for k, v in (train_raw_none or {}).items()
        if _train_backend_enabled(k)
    }
    if train_raw_tensors:
        train_raw_tensors = {
            k: v for k, v in train_raw_tensors.items()
            if _train_backend_enabled(k)
        }
    else:
        train_raw_tensors = {}
    none_by_class = None
    if train_raw_none_by_class:
        none_by_class = {
            str(b): dict(m)
            for b, m in train_raw_none_by_class.items()
            if _train_backend_enabled(b) and m
        } or None
    tensors_by_class = None
    if train_raw_tensors_by_class:
        tensors_by_class = {
            str(b): dict(m)
            for b, m in train_raw_tensors_by_class.items()
            if _train_backend_enabled(b) and m
        } or None
    if "sae" not in enabled_set:
        sae_raw = {}
    if "caa" not in enabled_set:
        caa_raw = {}

    class _StudyCfg:
        n_layers = int(n_model_layers or (max(SAE_LAYERS) + 1 if SAE_LAYERS else 12))

    STUDY_MODEL_CFG = _StudyCfg()

    def _is_actiend(backend) -> bool:
        return str(backend) in {"actiend", "actiend_pre"}

    def _run_id(method: str, split_mode: str) -> str:
        if split_mode == "none":
            return f"{method}"
        return f"{method}_{split_mode}"

    def _trainer_model_tok():
        model, tok = model_tok_from_trainers(train_raw_none, none_by_class)
        if model is not None and tok is not None:
            return model, tok
        model, tok = model_tok_from_trainers(_backbone_none, _backbone_none_by_class)
        if model is not None and tok is not None:
            print(
                "  causal: using reloaded trainer HF model (excluded from --methods)",
                flush=True,
            )
            return model, tok
        try:
            return load_hf_backbone_for_causal(study_model_key)
        except Exception as exc:
            track_error(exc, context="causal HF backbone")
            if fail_fast:
                _persist_partial_before_abort()
                raise
            print(f"  causal: HF backbone load failed: {exc}", flush=True)
            return None, None

    print(
        "\n=== Causal (decoder strengthen: ΔP(target) on other dataset; LMS-gated) ===",
        flush=True,
    )
    # Package IEND enforces ``decoder_max_size`` while building its decoder
    # frames. SAE/CAA/CGA/CAGA consume these shared row lists directly, so cap
    # the lists here too: no backend may silently evaluate more examples than
    # IEND when n_per_group and decoder_max_size differ.
    effective_n_per_group = causal_eval_n_per_group(
        n_per_group=CAUSAL_N, decoder_max_size=DECODER_MAX_SIZE
    )
    # The strength sweep (31 strengths x bisection) runs on val_rows purely to
    # SELECT the strength; the reported signed_effect/lms_ok come from meta_rows
    # (test). Selection is a coarse grid argmax, so a smaller validation pool
    # gives the same selected strength at a fraction of the sweep cost — while the
    # test report stays full-size, so headline accuracy is unchanged. Only val is
    # capped; test is not. (No CAUSAL_PROTOCOL_VERSION bump: nil effect on the
    # reported metric; already-computed cells keep their larger-val selection.)
    val_effective_n_per_group = (
        max(1, min(effective_n_per_group, int(val_n_per_group)))
        if val_n_per_group
        else effective_n_per_group
    )
    val_rows = stratified_causal_texts(
        labeled_df,
        neutral_df,
        target_classes=TARGET_CLASSES,
        n_per_group=val_effective_n_per_group,
        split="validation",
        seed=0,
    )
    meta_rows = stratified_causal_texts(
        labeled_df,
        neutral_df,
        target_classes=TARGET_CLASSES,
        n_per_group=effective_n_per_group,
        split="test",
        seed=0,
    )
    counts = {}
    for r in meta_rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print(
        f"  Causal texts per group: {counts} "
        f"(n_per_group={CAUSAL_N}, decoder_max_size={DECODER_MAX_SIZE}, "
        f"effective_cap={effective_n_per_group}, "
        f"val_sweep_cap={val_effective_n_per_group})  strengths={CAUSAL_STRENGTHS}",
        flush=True,
    )
    print(
        "  note: method ids spell the policy. "
        "GRADIEND causal = weight rewrite (gradiend:{cls} / gradiend:{cls}:tensors). "
        "ACTIEND causal default = actiend:{cls}:tok_all_gate_encoder_direction "
        "(= package token_selector='encoder_direction': all tokens + direction gate). "
        "Tensors aggregate = actiend:{cls}:tensors_tok_*; per-layer = :L*_tok_*. "
        "Ablations: tok_all (ungated), tok_prediction (ungated). "
        "SAE causal default = sae:{cls}:k1 (tok=all, shared with encoder); "
        "ablation :k1_tok_prediction. "
        "Fair-table actiend:{cls}/gradiend:{cls} = encoder readout only.",
        flush=True,
    )
    lms_texts = [r["text"] for r in meta_rows if r["group"] == "neutral"]

    # A one-pole trainer deliberately filters its *training* combined frame to
    # the factual pole (e.g. F-only).  ``evaluate_decoder`` otherwise rebuilds
    # its input from that frame, despite the strengthen metric correctly
    # requiring P(F) on the M panel.  Give decoder evaluation a task-level,
    # split-matched frame instead: it contains every factual class, while
    # ``evaluate_decoder_for_classes`` still pins summaries to the one-pole
    # claim only.  This prevents missing probs_by_dataset[M][F] / [F][M]
    # without accidentally treating the counterfactual as another feature.
    _decoder_frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _decoder_eval_frames(split: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_key = str(split or "test").lower()
        cached = _decoder_frames.get(split_key)
        if cached is not None:
            return cached

        # Validation frame drives only LR selection -> smaller cap; test frame is
        # the frozen report -> full DECODER_MAX_SIZE.
        frame_max_size = (
            VAL_DECODER_MAX_SIZE if split_key == "validation" else DECODER_MAX_SIZE
        )
        _decoder_frames[split_key] = decoder_eval_frames_for_study(
            labeled_df,
            neutral_df,
            target_classes=TARGET_CLASSES,
            max_size=frame_max_size,
            split=split,
        )
        return _decoder_frames[split_key]

    def _evaluate_decoder(trainer, classes: Sequence[str], **eval_kw):
        # No split default: selection/report intent must be explicit. Test is
        # permitted only when the split-clean caller has frozen the exact
        # validation-generated candidate set first.
        if "split" not in eval_kw:
            raise ValueError("decoder evaluation requires an explicit split")
        split = eval_kw["split"]
        training_like_df, decoder_neutral_df = _decoder_eval_frames(split)
        eval_kw.setdefault("training_like_df", training_like_df)
        eval_kw.setdefault("neutral_df", decoder_neutral_df)
        return evaluate_decoder_for_classes_refined(trainer, classes, **eval_kw)

    def _split_output_path(value: Any, split: str) -> Any:
        if not value:
            return value
        path = Path(value)
        return str(path.with_name(f"{path.stem}_{split}{path.suffix}"))

    def _evaluate_decoder_split_clean(
        trainer,
        classes: Sequence[str],
        *,
        evaluate_strengthen: bool = True,
        evaluate_weaken: bool = True,
        strengthen_classes: Optional[Sequence[str]] = None,
        **eval_kw,
    ):
        """Select/refine strengthen and weaken on validation, then freeze on test."""
        if "split" in eval_kw:
            raise ValueError(
                "_evaluate_decoder_split_clean owns the validation/test split; "
                "do not pass split"
            )

        base_output = eval_kw.get("output_path")
        if not base_output:
            experiment_dir = getattr(getattr(trainer, "args", None), "experiment_dir", None)
            if experiment_dir:
                base_output = str(Path(experiment_dir) / "decoder_grid.json")

        def _one_direction(*, weaken: bool, direction_classes: Sequence[str]):
            direction_kw = dict(eval_kw)
            direction_kw["increase_target_probabilities"] = not weaken
            direction_output = base_output
            if weaken and direction_output:
                p = Path(direction_output)
                direction_output = str(p.with_name(f"{p.stem}_weaken{p.suffix}"))

            validation_kw = dict(direction_kw)
            if direction_output:
                validation_kw["output_path"] = _split_output_path(
                    direction_output, "validation"
                )
            validation = _evaluate_decoder(
                trainer, direction_classes, split="validation", **validation_kw
            )
            # Validation owns LR selection. Test is confirmatory, so score
            # only that selected candidate (or one selected LR per class).
            frozen_lrs = decoder_selected_learning_rates(
                validation, classes=direction_classes
            )
            if not frozen_lrs:
                label = "weaken" if weaken else "strengthen"
                raise ValueError(
                    f"validation {label} decoder grid produced no candidate learning rates"
                )

            test_kw = dict(direction_kw)
            test_kw["lrs"] = frozen_lrs
            test_kw["refine_points"] = 0
            test_kw["plot"] = False
            test_kw["show"] = False
            test_kw.pop("plot_kwargs", None)
            if direction_output:
                test_kw["output_path"] = _split_output_path(direction_output, "test")
            test = _evaluate_decoder(
                trainer, direction_classes, split="test", **test_kw
            )
            frozen = bind_decoder_selection_to_evaluation_grid(validation, test)

            # Decoder artifacts/plots describe validation selection. Test remains
            # a single frozen report and must never supply a selector or plot star.
            for key in ("plot_path", "plot_paths", "raw_output_path", "output_path"):
                if validation.get(key) is not None:
                    frozen[key] = validation[key]
            frozen["validation_output_path"] = validation.get("output_path")
            frozen["test_output_path"] = test.get("output_path")
            return frozen

        strengthen_targets = list(strengthen_classes or classes)
        frozen = (
            _one_direction(weaken=False, direction_classes=strengthen_targets)
            if evaluate_strengthen and strengthen_targets
            else {}
        )
        if evaluate_weaken:
            frozen["weaken_results"] = _one_direction(
                weaken=True, direction_classes=classes
            )
        else:
            frozen["weaken_reused"] = True
        if not evaluate_strengthen:
            frozen["strengthen_reused"] = True

        return frozen

    # SAE/CAA causal sweep the same fixed (val_rows, meta_rows) pair across
    # dozens of feature/layer/class/policy combos per section; the *unsteered*
    # base_probs/base_lms only depend on (model, rows), not on which
    # feature/layer/class is being steered, so cache per (model, rows) instead
    # of paying a fresh unsteered forward pass on every single causal call.
    _base_stats_cache: Dict[Tuple[int, int], Tuple[Dict[str, Dict[str, float]], Any]] = {}

    def _cached_base_stats(m, tok, rows) -> Tuple[Dict[str, Dict[str, float]], Any]:
        key = (id(m), id(rows))
        cached = _base_stats_cache.get(key)
        if cached is not None:
            return cached
        frame = meta_rows_to_frame(rows)
        probs = score_class_probs_by_dataset(m, tok, frame)
        texts = [r["text"] for r in rows]
        neutral_texts = [r["text"] for r in rows if r["group"] == "neutral"] or texts
        lms = compute_lms_safe(m, tok, neutral_texts)
        _base_stats_cache[key] = (probs, lms)
        return probs, lms

    save_root = (OUTPUT_DIR / "modified_models") if SAVE_MODIFIED_MODELS else None
    if save_root is not None:
        print(f"  saving selected modified models under {save_root}", flush=True)
    else:
        print("  not saving modified models (pass --save-modified-models to persist)", flush=True)


    def _append_encoder_causal(
        *,
        trainer,
        backend: str,
        cls: str,
        decoder_results,
        method_id: str,
        decoder_split: str,
        activation_modules=None,
        part: str | None = None,
        token_selector=None,
        activation_gate=None,
        lrs=None,
    ):
        if method_id in _skip_causal:
            print(f"  skip-existing: {method_id} causal already complete", flush=True)
            return
        print(
            f"  {method_id}: from decoder grid/summary"
            + (
                f" (token_selector={token_selector!r})"
                if token_selector is not None
                else ""
            )
            + " …",
            flush=True,
        )
        try:
            # The matched-random package intervention must be scored on the
            # exact same test population as the frozen real intervention.
            # _decoder_eval_frames is memoized, so this is normally a cache hit.
            training_like_df, decoder_neutral_df = _decoder_eval_frames(decoder_split)
            results.append(
                run_encoder_causal_for_class(
                    trainer,
                    meta_rows,
                    target_class=cls,
                    backend=backend,
                    decoder_results=decoder_results,
                    strengths=CAUSAL_STRENGTHS,
                    lms_texts=lms_texts,
                    save_dir=save_root,
                    include_random_control=True,
                    use_cache=False,
                    method_id=method_id,
                    activation_modules=activation_modules,
                    token_selector=token_selector,
                    activation_gate=activation_gate,
                    lrs=lrs if lrs is not None else (
                        ACTIEND_DECODER_LRS if _is_actiend(backend) else None
                    ),
                    training_like_df=training_like_df,
                    neutral_df=decoder_neutral_df,
                    decoder_split=decoder_split,
                    reuse_strengthen=_reuse_for(method_id),
                    reuse_inverted_bidirectional=_polarity_repair_for(method_id),
                )
            )
            if part is not None:
                results[-1].meta["component_part"] = part
            if activation_modules is not None:
                results[-1].meta["activation_modules"] = activation_modules
            if _is_actiend(backend) and token_selector is not None:
                results[-1].meta["token_selector"] = token_selector
                results[-1].meta["activation_gate"] = activation_gate
            elif _is_actiend(backend) and token_selector is None:
                results[-1].meta.setdefault("token_selector", ACTIEND_TOK_DEFAULT)
                results[-1].meta.setdefault("activation_gate", ACTIEND_GATE_DEFAULT)
            sel = results[-1]
            print(
                f"    selected strength={sel.selected_strength} "
                f"signed_effect={None if sel.selected is None else round(sel.selected.signed_effect, 4)} "
                f"lms_ok={None if sel.selected is None else sel.selected.lms_ok} "
                f"tok={sel.meta.get('token_selector')} gate={sel.meta.get('activation_gate')}",
                flush=True,
            )
        except Exception as exc:
            track_error(exc, context=f"{method_id} causal")
            if fail_fast:
                _persist_partial_before_abort()
                raise
            invalidate_decoder_grid_cache(trainer)
            results.append(
                CausalMethodResult(
                    method=method_id,
                    backend=backend,
                    target_class=str(cls),
                    strengths=[],
                    selected_strength=None,
                    selected=None,
                    meta={
                        "error": str(exc),
                        "component_part": part,
                        "token_selector": token_selector,
                    },
                )
            )

    # --- none-split GRADIEND / ACTIEND (full / all-site) ---
    for backend, raw, class_ids in iter_trainer_class_groups(
        train_raw_none,
        target_classes=TARGET_CLASSES,
        train_raw_by_class=none_by_class,
    ):
        if str(raw.get("split_mode") or "none") != "none":
            continue
        if is_caga_backend(backend):
            # CAGA has no learned package decoder and remains a direct CAA-style
            # baseline. AGIEND has a learned package activation-space mapping and
            # must use the same package decoder/intervention path as IEND.
            continue
        _causal_sec = CostSection(
            f"causal:{backend}:none:{'+'.join(class_ids)}",
            phase="causal",
            backend=str(backend),
            split="none",
        ).start()
        _causal_sec_start_idx = len(results)
        try:
            trainer = raw["trainer"]
            exp_dir = getattr(getattr(trainer, "args", None), "experiment_dir", None) or (
                OUTPUT_DIR / _run_id(backend, "none")
            )
            selected_direction_parts: Dict[str, str] = {}
            if primary_only and is_cga_backend(backend) and layerwise_cga:
                for cls in class_ids:
                    part = selected_direction_part_by_validation_detection(raw, cls=str(cls))
                    if part is None:
                        print(
                            f"  {backend}:{cls}: incomplete validation aggregate/layer pool; "
                            "no selected causal representation",
                            flush=True,
                        )
                        continue
                    selected_direction_parts[str(cls)] = part
                    print(
                        f"  {backend}:{cls}: validation-selected representation={part}",
                        flush=True,
                    )
                aggregate_class_ids = [
                    str(cls) for cls in class_ids
                    if selected_direction_parts.get(str(cls)) == "aggregate"
                ]
            else:
                aggregate_class_ids = [str(cls) for cls in class_ids]
            if _is_actiend(backend):
                default_ids = _encoder_mids(
                    backend,
                    raw,
                    aggregate_class_ids,
                    token_selector=ACTIEND_TOK_DEFAULT,
                    activation_gate=ACTIEND_GATE_DEFAULT,
                    actiend_tokens=True,
                )
            else:
                default_ids = _encoder_mids(backend, raw, aggregate_class_ids)
            # In primary core mode the aggregate and layer candidates are a
            # *selection pool*, not a request to reopen every old ablation.
            # If the one validation-selected causal id is already persisted,
            # avoid even materialising the deferred 8--9B model on a repeat
            # upgrade invocation.
            if primary_only and is_cga_backend(backend) and layerwise_cga:
                selected_causal_ids = list(default_ids)
                for cls, part in selected_direction_parts.items():
                    if part != "aggregate":
                        selected_causal_ids.extend(
                            _encoder_mids(backend, raw, [cls], layer=part)
                        )
                if not selected_direction_parts:
                    continue
                if selected_causal_ids and causal_group_already_complete(
                    selected_causal_ids, _skip_causal
                ):
                    print(
                        f"  skip-existing: {backend} validation-selected causal "
                        "representation(s) already complete",
                        flush=True,
                    )
                    continue
            load_deferred_trainer_for_causal(trainer, raw, backend)
            if str(backend) not in enabled_set:
                # A ridge-only run reloads ACTIEND solely as the source of the
                # frozen encoder scores; do not execute learned ACTIEND causal.
                dec = None
            elif _is_actiend(backend) and not ACTIEND_DEFAULT_ENABLED:
                print(
                    f"  skip configured-off ACTIEND default (ungated all-token) policy "
                    f"(classes={class_ids})",
                    flush=True,
                )
                dec = None
            elif not aggregate_class_ids:
                dec = None
            elif causal_group_already_complete(default_ids, _skip_causal):
                print(
                    f"  skip-existing: {backend} none-split decoder+causal already complete "
                    f"({len(default_ids)} methods)",
                    flush=True,
                )
                dec = None
            else:
                offload_inert_cga_encoder_for_causal(trainer, backend)
                print(
                    f"  {backend} classes={class_ids}: evaluate_decoder (use_cache=False) …",
                    flush=True,
                )
                try:
                    eval_kw = dict(
                        max_size=DECODER_MAX_SIZE,
                        plot=True,
                        show=False,
                        use_cache=False,
                        plot_kwargs={"show": False},
                    )
                    if _is_actiend(backend):
                        eval_kw["lrs"] = list(ACTIEND_DECODER_LRS)
                        # Explicit split form of package default token_selector="encoder_direction".
                        eval_kw["token_selector"] = ACTIEND_TOK_DEFAULT
                        eval_kw["activation_gate"] = ACTIEND_GATE_DEFAULT
                    elif is_agiend_backend(backend):
                        eval_kw["lrs"] = list(CAUSAL_STRENGTHS)
                        eval_kw["token_selector"] = "all"
                    else:
                        # Explicit split form of the package's own default lrs grid, so
                        # evaluate_decoder_for_classes' missing-probability retry (which
                        # only drops a bad lr from an explicit kw["lrs"] list) can engage
                        # for GRADIEND too instead of silently no-op'ing and re-raising.
                        eval_kw["lrs"] = list(CAUSAL_STRENGTHS)
                    dec = _evaluate_decoder_split_clean(
                        trainer,
                        aggregate_class_ids,
                        evaluate_strengthen=not _all_strengthen_reusable(default_ids),
                        evaluate_weaken=not _all_polarity_repairable(default_ids),
                        strengthen_classes=_missing_strengthen_classes(
                            default_ids, aggregate_class_ids
                        ),
                        **eval_kw,
                    )
                except Exception as exc:
                    track_error(exc, context=f"{backend}: evaluate_decoder")
                    if fail_fast:
                        _persist_partial_before_abort()
                        raise
                    invalidate_decoder_grid_cache(
                        trainer, extra_paths=[Path(exp_dir) / "decoder_grid_cache.json"]
                    )
                    dec = None
                _record_decoder_artifacts(
                    decoder_catalog,
                    method_label=str(backend),
                    backend=str(backend),
                    part=None,
                    activation_modules=None,
                    decoder_results=dec,
                    experiment_dir=exp_dir,
                )
                for cls in aggregate_class_ids:
                    # GRADIEND: weight rewrite. Pair ids: gradiend:asian-white:asian
                    # ACTIEND: tok_all + gate_encoder_direction (= package default)
                    if _is_actiend(backend):
                        mid = feature_or_causal_id(
                            backend,
                            cls,
                            raw,
                            token_selector=ACTIEND_TOK_DEFAULT,
                            activation_gate=ACTIEND_GATE_DEFAULT,
                        )
                    else:
                        mid = feature_or_causal_id(backend, cls, raw)
                    _append_encoder_causal(
                        trainer=trainer,
                        backend=backend,
                        cls=str(cls),
                        decoder_results=dec,
                        method_id=mid,
                        decoder_split="test",
                        token_selector=(
                            ACTIEND_TOK_DEFAULT
                            if _is_actiend(backend)
                            else ("all" if is_agiend_backend(backend) else None)
                        ),
                        activation_gate=ACTIEND_GATE_DEFAULT if _is_actiend(backend) else None,
                        lrs=(CAUSAL_STRENGTHS if is_agiend_backend(backend) else None),
                    )
            # CGA layer variants reuse the checkpoint's persisted decoder tensor.
            # Temporarily mask it to one residual block, evaluate the ordinary
            # weight-rewrite path, then restore the aggregate direction.  This is
            # deliberately separate from activation_modules/CAA steering.
            if layerwise_cga and is_cga_backend(backend) and str(backend) in enabled_set:
                from cga_eval import (
                    cga_layer_slices,
                    direction_from_decoder,
                    install_cga_direction,
                )

                mwg = trainer.get_model()
                layer_bounds = cga_layer_slices(mwg)
                aggregate_direction = direction_from_decoder(mwg, copy=True)
                try:
                    for layer, (lo, hi) in layer_bounds.items():
                        part = f"L{int(layer)}"
                        layer_class_ids = (
                            [
                                str(cls) for cls in class_ids
                                if selected_direction_parts.get(str(cls)) == part
                            ]
                            if primary_only
                            else [str(cls) for cls in class_ids]
                        )
                        if not layer_class_ids:
                            continue
                        layer_ids = _encoder_mids(backend, raw, layer_class_ids, layer=part)
                        if causal_group_already_complete(layer_ids, _skip_causal):
                            print(
                                f"  skip-existing: {backend}:{part} causal already complete",
                                flush=True,
                            )
                            continue
                        offload_inert_cga_encoder_for_causal(trainer, backend)
                        masked_direction = __import__("torch").zeros_like(aggregate_direction)
                        masked_direction[int(lo):int(hi)].copy_(
                            aggregate_direction[int(lo):int(hi)]
                        )
                        install_cga_direction(mwg, masked_direction)
                        layer_dir = Path(exp_dir) / "decoder_by_layer" / part
                        layer_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            dec_layer = _evaluate_decoder_split_clean(
                                trainer,
                                layer_class_ids,
                                evaluate_strengthen=not _all_strengthen_reusable(layer_ids),
                                evaluate_weaken=not _all_polarity_repairable(layer_ids),
                                strengthen_classes=_missing_strengthen_classes(layer_ids, class_ids),
                                max_size=DECODER_MAX_SIZE,
                                plot=True,
                                show=False,
                                use_cache=False,
                                lrs=list(CAUSAL_STRENGTHS),
                                output_path=str(layer_dir / "decoder_grid.json"),
                                plot_kwargs={"show": False, "title": f"{backend} {part}"},
                            )
                        except Exception as exc:
                            track_error(exc, context=f"{backend}:{part} evaluate_decoder")
                            if fail_fast:
                                _persist_partial_before_abort()
                                raise
                            dec_layer = None
                        _record_decoder_artifacts(
                            decoder_catalog,
                            method_label=f"{backend}:{part}",
                            backend=str(backend),
                            part=part,
                            activation_modules=None,
                            decoder_results=dec_layer,
                            experiment_dir=layer_dir,
                        )
                        for cls in layer_class_ids:
                            _append_encoder_causal(
                                trainer=trainer,
                                backend=str(backend),
                                cls=str(cls),
                                decoder_results=dec_layer,
                                method_id=feature_or_causal_id(
                                    str(backend), cls, raw, layer=part
                                ),
                                decoder_split="test",
                            )
                        del masked_direction
                finally:
                    install_cga_direction(mwg, aggregate_direction)
                    del aggregate_direction
            # ACTIEND only: ungated scope ablations (fresh decoder grids; costly).
            # These share the trainer experiment_dir default cache path
            # (decoder_grid_cache.json) — do not enable package use_cache here.
            if _is_actiend(backend) and str(backend) in enabled_set:
                for abl in ACTIEND_SELECTED_ABLATIONS:
                    tok = abl["token_selector"]
                    gate = abl["activation_gate"]
                    abl_ids = _encoder_mids(
                        backend,
                        raw,
                        class_ids,
                        token_selector=tok,
                        activation_gate=gate,
                        actiend_tokens=True,
                    )
                    if causal_group_already_complete(abl_ids, _skip_causal):
                        print(
                            f"  skip-existing: actiend tok={tok!r} gate={gate!r} "
                            f"decoder+causal already complete",
                            flush=True,
                        )
                        continue
                    print(
                        f"  actiend: evaluate_decoder token_selector={tok!r} "
                        f"activation_gate={gate!r} "
                        f"(ablation vs default all+gate_encoder_direction) …",
                        flush=True,
                    )
                    try:
                        dec_tok = _evaluate_decoder_split_clean(
                            trainer,
                            class_ids,
                            evaluate_strengthen=not _all_strengthen_reusable(abl_ids),
                            strengthen_classes=_missing_strengthen_classes(abl_ids, class_ids),
                            max_size=DECODER_MAX_SIZE,
                            plot=True,
                            show=False,
                            use_cache=False,
                            token_selector=tok,
                            activation_gate=gate,
                            lrs=list(ACTIEND_DECODER_LRS),
                            plot_kwargs={
                                "show": False,
                                "title": f"actiend tok={tok} gate={gate}",
                            },
                        )
                    except Exception as exc:
                        track_error(exc, context="actiend evaluate_decoder", tok=tok, gate=gate)
                        if fail_fast:
                            _persist_partial_before_abort()
                            raise
                        invalidate_decoder_grid_cache(
                            trainer, extra_paths=[Path(exp_dir) / "decoder_grid_cache.json"]
                        )
                        dec_tok = None
                    part_lbl = (
                        f"tok_{tok}_gate_{gate}" if gate else f"tok_{tok}"
                    )
                    _record_decoder_artifacts(
                        decoder_catalog,
                        method_label=f"{backend}:{part_lbl}",
                        backend=str(backend),
                        part=part_lbl,
                        activation_modules=None,
                        decoder_results=dec_tok,
                        experiment_dir=Path(exp_dir) / part_lbl,
                    )
                    for cls in class_ids:
                        _append_encoder_causal(
                            trainer=trainer,
                            backend=str(backend),
                            cls=str(cls),
                            decoder_results=dec_tok,
                            method_id=feature_or_causal_id(
                                str(backend),
                                cls,
                                raw,
                                token_selector=tok,
                                activation_gate=gate,
                            ),
                            decoder_split="test",
                            token_selector=tok,
                            activation_gate=gate,
                        )
            if _is_actiend(backend) and "actiend_ridge" in enabled_set:
                ridge_ids = _encoder_mids(
                    "actiend_ridge",
                    raw,
                    class_ids,
                    token_selector=ACTIEND_TOK_DEFAULT,
                    activation_gate=ACTIEND_GATE_DEFAULT,
                    actiend_tokens=True,
                )
                if causal_group_already_complete(ridge_ids, _skip_causal):
                    print(
                        f"  skip-existing: actiend_ridge decoder+causal already complete "
                        f"({len(ridge_ids)} methods)",
                        flush=True,
                    )
                else:
                    print(
                        "  actiend_ridge: fitting frozen-score ridge decoder "
                        "(validation select, test evaluate) …",
                        flush=True,
                    )
                    try:
                        source_kind = str(getattr(getattr(trainer, "args", None), "source", "") or "alternative")
                        batches, learned = _extract_ridge_splits_from_trainer(
                            trainer,
                            checkpoint_source=source_kind,
                            max_size_per_group=ACTIEND_RIDGE_FIT_MAX_SIZE_PER_GROUP,
                            max_samples=ACTIEND_RIDGE_FIT_MAX_SAMPLES,
                            extraction_seed=0,
                        )
                        fits = fit_checkpoint_ridge_decoders(
                            batches["train"],
                            batches["validation"],
                            batches["test"],
                            learned,
                            source_kind=source_kind,
                            ridge_grid=ACTIEND_DECODER_LRS,
                        )
                        ridge_artifact_dir = Path(exp_dir) / "actiend_ridge"
                        ridge_artifact_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            ridge_artifact_dir / "ridge_decoder_parameters.npz",
                            latent_weight=fits["latent"].decoder_weight,
                            latent_bias=fits["latent"].decoder_bias,
                            preactivation_weight=fits["preactivation"].decoder_weight,
                            preactivation_bias=fits["preactivation"].decoder_bias,
                        )
                        (ridge_artifact_dir / "ridge_decoder_meta.json").write_text(
                            __import__("json").dumps(
                                {
                                    "method_family": "actiend_ridge",
                                    "source_kind": source_kind,
                                    "ridge_grid": list(float(x) for x in ACTIEND_DECODER_LRS),
                                    "variant": ACTIEND_RIDGE_VARIANT_ID,
                                    "latent_fit": fits["latent"].summary(),
                                    "preactivation_fit": fits["preactivation"].summary(),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        gradiend = trainer.get_model().gradiend
                        trained_decoder_weight = (
                            gradiend.decoder[0].linear.weight.detach().float().cpu().numpy().reshape(-1)
                        )
                        raw_trained_bias = gradiend.decoder[0].linear.bias
                        trained_decoder_bias = (
                            np.zeros_like(trained_decoder_weight)
                            if raw_trained_bias is None
                            else raw_trained_bias.detach().float().cpu().numpy().reshape(-1)
                        )
                        variant = {
                            item.variant_id: item
                            for item in ridge_causal_variants(
                                fits,
                                trained_decoder_weight=trained_decoder_weight,
                                trained_decoder_bias=trained_decoder_bias,
                            )
                        }[ACTIEND_RIDGE_VARIANT_ID]
                        with apply_decoder_causal_variant(trainer.get_model(), variant):
                            base_eval_kw = dict(
                                max_size=DECODER_MAX_SIZE,
                                plot=False,
                                show=False,
                                use_cache=False,
                                token_selector=ACTIEND_TOK_DEFAULT,
                                activation_gate=ACTIEND_GATE_DEFAULT,
                                lrs=list(ACTIEND_DECODER_LRS),
                            )
                            frozen = _evaluate_decoder_split_clean(
                                trainer,
                                class_ids,
                                evaluate_strengthen=not _all_strengthen_reusable(ridge_ids),
                                strengthen_classes=_missing_strengthen_classes(ridge_ids, class_ids),
                                output_path=str(Path(exp_dir) / "decoder_ridge_grid.json"),
                                **base_eval_kw,
                            )
                            for cls in class_ids:
                                _append_encoder_causal(
                                    trainer=trainer,
                                    backend="actiend_ridge",
                                    cls=str(cls),
                                    decoder_results=frozen,
                                    method_id=feature_or_causal_id(
                                        "actiend_ridge",
                                        cls,
                                        raw,
                                        token_selector=ACTIEND_TOK_DEFAULT,
                                        activation_gate=ACTIEND_GATE_DEFAULT,
                                    ),
                                    # `frozen`'s grid is test_grid; both the
                                    # headline and matched-random control report
                                    # on test, never on validation selection.
                                    decoder_split="test",
                                    token_selector=ACTIEND_TOK_DEFAULT,
                                    activation_gate=ACTIEND_GATE_DEFAULT,
                                    lrs=ACTIEND_DECODER_LRS,
                                )
                    except Exception as exc:
                        track_error(exc, context="actiend_ridge causal")
                        if fail_fast:
                            _persist_partial_before_abort()
                            raise
                        print(f"  actiend_ridge failed: {exc}", flush=True)
        finally:
            # Same lag-1 retention as the CAGA loop's finally block below: this
            # iteration's layerwise-CGA branch (~line 1373) and actiend_ridge
            # branch (~line 1584) each bind a loop-scoped local straight to
            # trainer.get_model() (mwg) or trainer.get_model().gradiend
            # (gradiend), which keeps this iteration's base model reachable
            # through release_deferred_trainer_after_causal's gc.collect() and
            # into the next iteration's load until reassigned. Drop them first
            # so release() has nothing left to reclaim.
            mwg = gradiend = None
            release_deferred_trainer_after_causal(trainer, raw, backend)
            _new_results = results[_causal_sec_start_idx:]
            _causal_sec.meta["n_method_rows"] = len(_new_results)
            _causal_sec.meta["n_strength_points"] = sum(
                len(r.strengths or []) for r in _new_results
            )
            _causal_sec.stop()

    # --- tensors aggregate (:tensors): matched causal for by_tensor trainers ---
    # Same decode→LMS path as none, but on the tensors checkpoint with all trained
    # sites (activation_modules=None). Encoder rows already use :{cls}:tensors.
    tensors = train_raw_tensors or {}
    for backend, raw, class_ids in iter_trainer_class_groups(
        tensors,
        target_classes=TARGET_CLASSES,
        train_raw_by_class=tensors_by_class,
    ):
        if str(backend) not in enabled_set:
            continue
        trainer = raw.get("trainer")
        if trainer is None:
            continue
        exp_dir = Path(
            getattr(getattr(trainer, "args", None), "experiment_dir", None)
            or (OUTPUT_DIR / _run_id(str(backend), "tensors"))
        )
        agg_dir = exp_dir / "decoder_aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        grid_path = agg_dir / "decoder_grid_cache.json"
        plot_stem = agg_dir / "decoder_probability_shifts_tensors"
        if _is_actiend(backend):
            tensor_ids = _encoder_mids(
                backend,
                raw,
                class_ids,
                layer="tensors",
                token_selector=ACTIEND_TOK_DEFAULT,
                activation_gate=ACTIEND_GATE_DEFAULT,
                actiend_tokens=True,
            )
        else:
            tensor_ids = _encoder_mids(backend, raw, class_ids, layer="tensors")
        if _is_actiend(backend) and not ACTIEND_DEFAULT_ENABLED:
            print(
                f"  skip configured-off ACTIEND default (ungated all-token) tensors policy "
                f"(classes={class_ids})",
                flush=True,
            )
            dec_all = None
        elif causal_group_already_complete(tensor_ids, _skip_causal):
            print(
                f"  skip-existing: {backend} tensors decoder+causal already complete "
                f"({len(tensor_ids)} methods)",
                flush=True,
            )
            dec_all = None
        else:
            print(
                f"  {backend} tensors :tensors classes={class_ids}: evaluate_decoder "
                f"(all trained sites, use_cache=False) …",
                flush=True,
            )
            try:
                eval_kw = dict(
                    max_size=DECODER_MAX_SIZE,
                    plot=True,
                    show=False,
                    use_cache=False,
                    output_path=str(grid_path),
                    plot_kwargs={
                        "show": False,
                        "output": str(plot_stem),
                        "title": f"{backend} tensors :tensors",
                    },
                )
                if _is_actiend(backend):
                    eval_kw["lrs"] = list(ACTIEND_DECODER_LRS)
                    eval_kw["token_selector"] = ACTIEND_TOK_DEFAULT
                    eval_kw["activation_gate"] = ACTIEND_GATE_DEFAULT
                else:
                    eval_kw["lrs"] = list(CAUSAL_STRENGTHS)
                # activation_modules omitted → all trained sites / full weight map.
                dec_all = _evaluate_decoder_split_clean(
                    trainer,
                    class_ids,
                    evaluate_strengthen=not _all_strengthen_reusable(tensor_ids),
                    strengthen_classes=_missing_strengthen_classes(tensor_ids, class_ids),
                    **eval_kw,
                )
            except Exception as exc:
                track_error(exc, context="evaluate_decoder tensors", backend=backend)
                if fail_fast:
                    _persist_partial_before_abort()
                    raise
                invalidate_decoder_grid_cache(trainer, extra_paths=[grid_path])
                dec_all = None
            _record_decoder_artifacts(
                decoder_catalog,
                method_label=f"{backend}:tensors",
                backend=str(backend),
                part="tensors",
                activation_modules=None,
                decoder_results=dec_all,
                experiment_dir=agg_dir,
            )
            for cls in class_ids:
                if _is_actiend(backend):
                    mid = feature_or_causal_id(
                        backend,
                        cls,
                        raw,
                        layer="tensors",
                        token_selector=ACTIEND_TOK_DEFAULT,
                        activation_gate=ACTIEND_GATE_DEFAULT,
                    )
                else:
                    mid = feature_or_causal_id(backend, cls, raw, layer="tensors")
                _append_encoder_causal(
                    trainer=trainer,
                    backend=str(backend),
                    cls=str(cls),
                    decoder_results=dec_all,
                    method_id=mid,
                    decoder_split="test",
                    part="tensors",
                    token_selector=ACTIEND_TOK_DEFAULT if _is_actiend(backend) else None,
                    activation_gate=ACTIEND_GATE_DEFAULT if _is_actiend(backend) else None,
                )
        if _is_actiend(backend):
            for abl in ACTIEND_SELECTED_ABLATIONS:
                tok = abl["token_selector"]
                gate = abl["activation_gate"]
                part_lbl = (
                    f"tensors_tok_{tok}_gate_{gate}" if gate else f"tensors_tok_{tok}"
                )
                abl_ids = _encoder_mids(
                    backend,
                    raw,
                    class_ids,
                    layer="tensors",
                    token_selector=tok,
                    activation_gate=gate,
                    actiend_tokens=True,
                )
                if causal_group_already_complete(abl_ids, _skip_causal):
                    print(
                        f"  skip-existing: actiend tensors tok={tok!r} gate={gate!r} "
                        f"decoder+causal already complete",
                        flush=True,
                    )
                    continue
                abl_dir = agg_dir / part_lbl
                abl_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"  actiend tensors :tensors: evaluate_decoder "
                    f"token_selector={tok!r} activation_gate={gate!r} …",
                    flush=True,
                )
                try:
                    dec_tok = _evaluate_decoder_split_clean(
                        trainer,
                        class_ids,
                        evaluate_strengthen=not _all_strengthen_reusable(abl_ids),
                        strengthen_classes=_missing_strengthen_classes(abl_ids, class_ids),
                        max_size=DECODER_MAX_SIZE,
                        plot=True,
                        show=False,
                        use_cache=False,
                        token_selector=tok,
                        activation_gate=gate,
                        lrs=list(ACTIEND_DECODER_LRS),
                        output_path=str(abl_dir / "decoder_grid_cache.json"),
                        plot_kwargs={
                            "show": False,
                            "output": str(abl_dir / f"decoder_probability_shifts_{part_lbl}"),
                            "title": f"actiend :tensors tok={tok} gate={gate}",
                        },
                    )
                except Exception as exc:
                    track_error(exc, context="actiend tensors evaluate_decoder", tok=tok, gate=gate)
                    if fail_fast:
                        _persist_partial_before_abort()
                        raise
                    invalidate_decoder_grid_cache(
                        trainer, extra_paths=[abl_dir / "decoder_grid_cache.json"]
                    )
                    dec_tok = None
                _record_decoder_artifacts(
                    decoder_catalog,
                    method_label=f"{backend}:{part_lbl}",
                    backend=str(backend),
                    part=part_lbl,
                    activation_modules=None,
                    decoder_results=dec_tok,
                    experiment_dir=abl_dir,
                )
                for cls in class_ids:
                    _append_encoder_causal(
                        trainer=trainer,
                        backend=str(backend),
                        cls=str(cls),
                        decoder_results=dec_tok,
                        method_id=feature_or_causal_id(
                            str(backend),
                            cls,
                            raw,
                            layer="tensors",
                            token_selector=tok,
                            activation_gate=gate,
                        ),
                        decoder_split="test",
                        part="tensors",
                        token_selector=tok,
                        activation_gate=gate,
                    )

    # --- ACTIEND / ACTIEND-PRE per-layer (tensors split) ---
    _tensor_map = {
        k: tensors[k]
        for k in ("actiend", "actiend_pre")
        if tensors.get(k)
    }
    _tensor_by_cls = None
    if tensors_by_class:
        _tensor_by_cls = {
            k: tensors_by_class[k]
            for k in ("actiend", "actiend_pre")
            if tensors_by_class.get(k)
        } or None
    if ACTIEND_DEFAULT_ENABLED:
        actiend_tensor_groups = list(
            iter_trainer_class_groups(
                _tensor_map,
                target_classes=TARGET_CLASSES,
                train_raw_by_class=_tensor_by_cls,
            )
        )
    else:
        actiend_tensor_groups = []
        if _tensor_map:
            print(
                "  skip configured-off ACTIEND gated-all per-layer causal policies",
                flush=True,
            )
    for _backend, actiend_tensors, class_ids in actiend_tensor_groups:
        trainer = actiend_tensors["trainer"]
        exp_dir = Path(
            getattr(getattr(trainer, "args", None), "experiment_dir", None)
            or (OUTPUT_DIR / _run_id(str(_backend), "tensors"))
        )
        component_keys = list(actiend_tensors.get("component_keys") or [])
        if not component_keys:
            # Recover from per_component_readouts keys if cache omitted component_keys.
            for part in (actiend_tensors.get("per_component_readouts") or {}):
                if str(part).startswith("L") and str(part)[1:].isdigit():
                    layer = int(str(part)[1:])
                    resid = resid_sites(layer)[0]
                    component_keys.append(
                        {
                            "component_id": f"activation:{resid}",
                            "component_label": resid,
                            "part": f"L{layer}",
                        }
                    )
        print(
            f"  actiend tensors classes={class_ids}: evaluate_decoder per layer "
            f"({len(component_keys)} parts) …",
            flush=True,
        )
        print(
            f"  WARNING: {CAUSAL_LAYER_LR_CEILING_TODO} "
            "See TODO(causal-layer-lr-grid) in causal_study.py.",
            flush=True,
        )
        for ck in component_keys:
            part = str(ck.get("part") or short_component_part(ck.get("component_id"), ck.get("component_label")))
            if not (part.startswith("L") and part[1:].isdigit()):
                continue  # skip non-layer tensor parts if any
            # Smoke / short layer lists: only evaluate selected ACTIEND layers.
            if len(SAE_LAYERS) < int(STUDY_MODEL_CFG.n_layers) and part not in {
                f"L{int(L)}" for L in SAE_LAYERS
            }:
                continue
            module = _activation_module_name(ck)
            layer_dir = exp_dir / "decoder_by_layer" / part
            layer_dir.mkdir(parents=True, exist_ok=True)
            grid_path = layer_dir / "decoder_grid_cache.json"
            plot_stem = layer_dir / f"decoder_probability_shifts_{part}"
            layer_ids = _encoder_mids(
                str(_backend),
                actiend_tensors,
                class_ids,
                layer=part,
                token_selector=ACTIEND_TOK_DEFAULT,
                activation_gate=ACTIEND_GATE_DEFAULT,
                actiend_tokens=True,
            )
            if causal_group_already_complete(layer_ids, _skip_causal):
                print(
                    f"  skip-existing: actiend:{part} decoder+causal already complete",
                    flush=True,
                )
                continue
            print(
                f"  actiend:{part}: evaluate_decoder(activation_modules={module!r}) …",
                flush=True,
            )
            try:
                dec = _evaluate_decoder_split_clean(
                    trainer,
                    class_ids,
                    evaluate_strengthen=not _all_strengthen_reusable(layer_ids),
                    strengthen_classes=_missing_strengthen_classes(layer_ids, class_ids),
                    max_size=DECODER_MAX_SIZE,
                    plot=True,
                    show=False,
                    use_cache=False,
                    activation_modules=module,
                    token_selector=ACTIEND_TOK_DEFAULT,
                    activation_gate=ACTIEND_GATE_DEFAULT,
                    lrs=list(ACTIEND_DECODER_LRS),
                    output_path=str(grid_path),
                    plot_kwargs={
                        "show": False,
                        "output": str(plot_stem),
                        "title": f"actiend {part} | {module}",
                    },
                )
            except Exception as exc:
                track_error(exc, context=f"actiend:{part}: evaluate_decoder")
                if fail_fast:
                    _persist_partial_before_abort()
                    raise
                invalidate_decoder_grid_cache(trainer, extra_paths=[grid_path])
                dec = None
            _record_decoder_artifacts(
                decoder_catalog,
                method_label=f"{_backend}:{part}",
                backend=str(_backend),
                part=part,
                activation_modules=module,
                decoder_results=dec,
                experiment_dir=layer_dir,
            )
            for cls in class_ids:
                _append_encoder_causal(
                    trainer=trainer,
                    backend=str(_backend),
                    cls=str(cls),
                    decoder_results=dec,
                    method_id=feature_or_causal_id(
                        str(_backend),
                        cls,
                        actiend_tensors,
                        layer=part,
                        token_selector=ACTIEND_TOK_DEFAULT,
                        activation_gate=ACTIEND_GATE_DEFAULT,
                    ),
                    decoder_split="test",
                    activation_modules=module,
                    part=part,
                    token_selector=ACTIEND_TOK_DEFAULT,
                    activation_gate=ACTIEND_GATE_DEFAULT,
                )

    if sae_raw and not sae_raw.get("error"):
        _sae_causal_sec = CostSection(
            "causal:sae", phase="causal", backend="sae"
        ).start()
        _sae_causal_sec_start_idx = len(results)
        _site = str(sae_raw.get("act_policy") or "prediction")
        _sae_sites = [s for s, _ in iter_sae_site_payloads(sae_raw)] or [_site]
        print(f"  SAE causal select_sites={_sae_sites}", flush=True)
        active_site = [_site]
        sae_by_layer = dict(sae_raw.get("_sae_by_layer") or {})
        hf_by_layer = dict(sae_raw.get("_hf_module_by_layer") or {})
        model = sae_raw.get("_model")
        tokenizer = sae_raw.get("_tokenizer")
        if model is None or tokenizer is None:
            model, tokenizer = _trainer_model_tok()
        if model is None or tokenizer is None:
            raise RuntimeError(
                "SAE causal needs a HF model/tokenizer. METHODS=sae must load "
                "the study backbone; trainer reload and from_pretrained both failed."
            )
        else:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _sae_base_probs, _sae_base_lms = _cached_base_stats(model, tokenizer, val_rows)
            _sae_r_base_probs, _sae_r_base_lms = _cached_base_stats(model, tokenizer, meta_rows)

            def _offload_sae(sae) -> None:
                """Release an SAE's CUDA storage even if Python retains a reference."""
                if sae is None or device != "cuda":
                    return
                try:
                    sae.to("cpu")
                except Exception as exc:
                    print(f"  WARNING: failed to offload SAE to CPU: {exc}", flush=True)

            def _collect_cuda_memory() -> None:
                # SAELens modules can participate in reference cycles. empty_cache()
                # alone cannot release their live tensors; collect cycles first.
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

            def _ensure_sae(layer: int):
                # A Llama-Scope SAE is roughly 4 GiB in float32. Keep the
                # cache bounded to one layer: retaining one SAE per layer
                # exhausts an A100 before layer 16.
                stale_layers = [key for key in sae_by_layer if key != layer]
                for key in stale_layers:
                    stale_sae = sae_by_layer.pop(key)
                    _offload_sae(stale_sae)
                    del stale_sae
                if stale_layers:
                    _collect_cuda_memory()
                sae = sae_by_layer.get(layer)
                if sae is None:
                    _, sae_id = resid_sites(layer)
                    print(f"  SAE causal L{layer}: loading SAE …", flush=True)
                    sae = load_sae(SAE_RELEASE, sae_id, device=device)
                    sae_by_layer[layer] = sae
                return sae

            def _append_sae_causal(
                *,
                method_id: str,
                layer: int,
                feature_indices,
                target_class: str,
                note: str,
                bag_reduce: str = "sum",
                token_selector: str = "all",
            ) -> None:
                if method_id in _skip_causal:
                    print(
                        f"  skip-existing: {method_id} causal already complete",
                        flush=True,
                    )
                    return
                idxs = []
                for i in feature_indices or []:
                    try:
                        idxs.append(int(i))
                    except (TypeError, ValueError):
                        continue
                if not idxs:
                    print(f"  {method_id}: no feature; skip", flush=True)
                    return
                hf_module = hf_by_layer.get(layer) or resid_sites(layer)[0]
                sae = _ensure_sae(layer)
                print(
                    f"  {method_id}: L{layer} feats={idxs[:8]}"
                    f"{'…' if len(idxs) > 8 else ''} (n={len(idxs)}; tok={token_selector}; {note}) …",
                    flush=True,
                )
                try:
                    results.append(
                        run_sae_causal_sweep(
                            model,
                            tokenizer,
                            val_rows,
                            sae=sae,
                            module_path=hf_module,
                            feature_indices=idxs,
                            target_class=str(target_class),
                            strengths=SAE_CAUSAL_STRENGTHS,
                            lms_texts=None,
                            report_rows=meta_rows,
                            include_random_control=True,
                            save_dir=save_root,
                            method_id=method_id,
                            bag_reduce=bag_reduce,
                            token_selector=token_selector,
                            base_probs=_sae_base_probs,
                            base_lms=_sae_base_lms,
                            r_base_probs=_sae_r_base_probs,
                            r_base_lms=_sae_r_base_lms,
                            reuse_strengthen=_reuse_for(method_id),
                        )
                    )
                    results[-1].meta["sae_layer"] = layer
                    results[-1].meta["sae_feature_indices"] = idxs
                    results[-1].meta["sae_causal_k"] = len(idxs)
                    results[-1].meta["token_selector"] = token_selector
                    results[-1].meta["sae_select_site"] = active_site[0]
                    results[-1].meta["note"] = note
                    sel = results[-1]
                    warn = ""
                    if sel.selected is not None and "lms_gate_empty" in (sel.selected.notes or ""):
                        warn = f" [{sel.selected.notes}]"
                        print(f"    WARNING: LMS gate empty — fallback used{warn}", flush=True)
                    print(
                        f"    selected strength={sel.selected_strength} "
                        f"signed_effect="
                        f"{None if sel.selected is None else round(sel.selected.signed_effect, 4)} "
                        f"lms_ok={None if sel.selected is None else sel.selected.lms_ok}"
                        f" sign={sel.meta.get('sign')}{warn}",
                        flush=True,
                    )
                except Exception as exc:
                    track_error(exc, context=f"{method_id} causal")
                    if fail_fast:
                        _persist_partial_before_abort()
                        raise
                    results.append(
                        CausalMethodResult(
                            method=method_id,
                            backend="sae",
                            target_class=str(target_class),
                            strengths=[],
                            selected_strength=None,
                            selected=None,
                            meta={
                                "error": str(exc),
                                "token_selector": token_selector,
                                "sae_select_site": active_site[0],
                            },
                        )
                    )

            def _append_sae_clamp_causal(
                *,
                method_id: str,
                layer: int,
                feature_index,
                target_class: str,
                note: str,
                token_selector: str = "all",
            ) -> None:
                if method_id in _skip_causal:
                    print(
                        f"  skip-existing: {method_id} causal already complete",
                        flush=True,
                    )
                    return
                try:
                    feat = int(feature_index)
                except (TypeError, ValueError):
                    print(f"  {method_id}: no feature; skip", flush=True)
                    return
                hf_module = hf_by_layer.get(layer) or resid_sites(layer)[0]
                sae = _ensure_sae(layer)
                print(
                    f"  {method_id}: L{layer} feat={feat} (clamp; tok={token_selector}; {note}) …",
                    flush=True,
                )
                try:
                    results.append(
                        run_sae_clamp_causal_sweep(
                            model,
                            tokenizer,
                            val_rows,
                            sae=sae,
                            module_path=hf_module,
                            feature_index=feat,
                            target_class=str(target_class),
                            lms_texts=None,
                            report_rows=meta_rows,
                            include_random_control=True,
                            save_dir=save_root,
                            method_id=method_id,
                            token_selector=token_selector,
                            base_probs=_sae_base_probs,
                            base_lms=_sae_base_lms,
                            r_base_probs=_sae_r_base_probs,
                            r_base_lms=_sae_r_base_lms,
                            reuse_strengthen=_reuse_for(method_id),
                        )
                    )
                    results[-1].meta["sae_layer"] = layer
                    results[-1].meta["sae_feature_index"] = feat
                    results[-1].meta["token_selector"] = token_selector
                    results[-1].meta["sae_select_site"] = active_site[0]
                    results[-1].meta["note"] = note
                    sel = results[-1]
                    warn = ""
                    if sel.selected is not None and "lms_gate_empty" in (sel.selected.notes or ""):
                        warn = f" [{sel.selected.notes}]"
                        print(f"    WARNING: LMS gate empty — fallback used{warn}", flush=True)
                    print(
                        f"    selected target={sel.selected_strength} "
                        f"signed_effect="
                        f"{None if sel.selected is None else round(sel.selected.signed_effect, 4)} "
                        f"lms_ok={None if sel.selected is None else sel.selected.lms_ok}"
                        f"{warn}",
                        flush=True,
                    )
                except Exception as exc:
                    track_error(exc, context=f"{method_id} causal")
                    if fail_fast:
                        _persist_partial_before_abort()
                        raise
                    results.append(
                        CausalMethodResult(
                            method=method_id,
                            backend="sae",
                            target_class=str(target_class),
                            strengths=[],
                            selected_strength=None,
                            selected=None,
                            meta={
                                "error": str(exc),
                                "token_selector": token_selector,
                                "sae_select_site": active_site[0],
                                "mode": "clamp",
                            },
                        )
                    )

            def _iter_numeric_layer_raw(site_raw: dict):
                for layer_key, layer_raw in (site_raw.get("by_layer") or {}).items():
                    layer_i = _int_layer(layer_key)
                    if layer_i is None:
                        continue
                    yield layer_i, layer_raw

            def _all_layer_specs(site_raw: dict, cls: str, k: int) -> list:
                specs = []
                for layer_i, layer_raw in _iter_numeric_layer_raw(site_raw):
                    fbc_L = (
                        ((layer_raw.get("selections") or {}).get("per_class") or {}).get(
                            "features_by_class"
                        )
                        or {}
                    )
                    feats_L = list(fbc_L.get(cls) or [])[: int(k)]
                    if not feats_L:
                        continue
                    hf_module = hf_by_layer.get(layer_i) or resid_sites(layer_i)[0]
                    sae = _ensure_sae(layer_i)
                    if len(feats_L) == 1:
                        direction = sae_decoder_direction(sae, feats_L[0])
                    else:
                        direction = sae_decoder_bag_direction(
                            sae, feats_L, reduce="sum"
                        )
                    direction_cpu = direction.detach().float().cpu()
                    specs.append(
                        {
                            "layer": layer_i,
                            "module_path": hf_module,
                            # Only this d_model-sized CPU vector needs to
                            # survive across layers.  Keeping ``sae`` here
                            # retained all 32 full dictionaries on the GPU.
                            "direction": direction_cpu,
                            "feature_indices": feats_L,
                        }
                    )
                    # ``direction`` is a CUDA view into W_dec. Drop it before
                    # offloading the module, then force collection of any
                    # SAELens reference cycles. Merely calling empty_cache()
                    # left roughly 4 GiB allocated per Llama layer.
                    del direction
                    sae_by_layer.pop(layer_i, None)
                    _offload_sae(sae)
                    del sae
                    _collect_cuda_memory()
                    if device == "cuda":
                        allocated_gib = torch.cuda.memory_allocated() / (1024**3)
                        print(
                            f"  SAE causal L{layer_i}: after offload "
                            f"CUDA allocated={allocated_gib:.2f} GiB",
                            flush=True,
                        )
                return specs

            def _append_all_layers_sae_causal(
                *,
                method_id: str,
                site_raw: dict,
                cls: str,
                k: int,
                note: str,
                component_part: Optional[str] = None,
            ) -> None:
                if method_id in _skip_causal:
                    print(
                        f"  skip-existing: {method_id} causal already complete",
                        flush=True,
                    )
                    return
                layer_specs = _all_layer_specs(site_raw, cls, int(k))
                if not layer_specs:
                    print(f"  {method_id}: no layer features; skip", flush=True)
                    return
                part = component_part or f"all_k{int(k)}"
                print(
                    f"  {method_id}: n_layers={len(layer_specs)} k={int(k)} "
                    f"strengths={list(SAE_CAUSAL_STRENGTHS)} …",
                    flush=True,
                )
                try:
                    results.append(
                        run_sae_multi_layer_causal_sweep(
                            model,
                            tokenizer,
                            val_rows,
                            layer_specs=layer_specs,
                            target_class=cls,
                            strengths=SAE_CAUSAL_STRENGTHS,
                            lms_texts=None,
                            report_rows=meta_rows,
                            include_random_control=True,
                            method_id=method_id,
                            token_selector="all",
                            bag_reduce="sum",
                            base_probs=_sae_base_probs,
                            base_lms=_sae_base_lms,
                            r_base_probs=_sae_r_base_probs,
                            r_base_lms=_sae_r_base_lms,
                            reuse_strengthen=_reuse_for(method_id),
                        )
                    )
                    results[-1].meta["component_part"] = part
                    results[-1].meta["readout_k"] = int(k)
                    results[-1].meta["sae_layer"] = "all"
                    results[-1].meta["sae_select_site"] = active_site[0]
                    results[-1].meta["note"] = note
                    sel = results[-1]
                    print(
                        f"    selected strength={sel.selected_strength} "
                        f"signed_effect="
                        f"{None if sel.selected is None else round(sel.selected.signed_effect, 4)} "
                        f"lms_ok={None if sel.selected is None else sel.selected.lms_ok}",
                        flush=True,
                    )
                except Exception as exc:
                    track_error(exc, context=f"{method_id} causal")
                    if fail_fast:
                        _persist_partial_before_abort()
                        raise
                    print(f"  {method_id} causal failed: {exc}", flush=True)
                    results.append(
                        CausalMethodResult(
                            method=method_id,
                            backend="sae",
                            target_class=cls,
                            strengths=[],
                            selected_strength=None,
                            selected=None,
                            meta={
                                "error": str(exc),
                                "component_part": part,
                                "sae_select_site": active_site[0],
                            },
                        )
                    )

            print(
                f"  WARNING: {CAUSAL_ABLATIONS_TODO} "
                f"{SAE_SELECTION_OPPOSITE_CLASS_TODO} {SAE_SELECTION_ARAD_TODO} "
                f"{SAE_SELECTION_JH_TODO}",
                flush=True,
            )

            def _run_sae_causal_site(site: str, site_raw: dict) -> None:
                """Steer every encode method for this SAE site (ids match encoder rows)."""
                selections = site_raw.get("selections") or {}
                fbc = (selections.get("per_class") or {}).get("features_by_class") or {}
                fbc_opp = (selections.get("per_class_opp_fire") or {}).get(
                    "features_by_class"
                ) or {}
                fbc_arad = (selections.get("per_class_arad_out") or {}).get(
                    "features_by_class"
                ) or {}
                fbc_jh = (selections.get("per_class_jh_f1") or {}).get(
                    "features_by_class"
                ) or {}
                joint_feats = list(
                    (selections.get("joint") or {}).get("feature_indices") or []
                )
                selected_by_class = site_raw.get("selected_layer_by_class") or {}
                selected_opp = (
                    site_raw.get("selected_layer_by_class_opp_fire") or selected_by_class
                )
                selected_arad = (
                    site_raw.get("selected_layer_by_class_arad_out") or selected_by_class
                )
                selected_jh = (
                    site_raw.get("selected_layer_by_class_jh_f1") or selected_by_class
                )
                class_readouts = (site_raw.get("readouts") or {}).get(
                    "per_class_by_class"
                ) or {}
                global_layer = _int_layer(
                    site_raw.get("selected_layer_global"), SAE_LAYER
                )
                sae_classes = sae_encode_classes(site_raw)
                if not sae_classes:
                    print(
                        f"  SAE causal site={site}: no encoded classes; skip",
                        flush=True,
                    )
                    return
                print(
                    f"  SAE causal select_site={site} "
                    f"(full encode suite: k1/kstar/fixed-k/opp_fire/arad/jh/"
                    f"L*_k1/all_k/joint) …",
                    flush=True,
                )
                for cls in sae_classes:
                    cls = str(cls)
                    feats = list(fbc.get(cls) or [])
                    sel_raw = (
                        selected_by_class.get(cls)
                        or site_raw.get("selected_layer_global")
                        or SAE_LAYER
                    )
                    use_all = str(sel_raw) == "all"
                    layer = _int_layer(sel_raw, SAE_LAYER)
                    k_star = int((class_readouts.get(cls) or {}).get("readout_k") or 1)
                    k_star = max(1, k_star)

                    if use_all:
                        _append_all_layers_sae_causal(
                            method_id=sae_causal_id(
                                cls, part="k1", token_selector="all", site=site
                            ),
                            site_raw=site_raw,
                            cls=cls,
                            k=1,
                            note=f"all-layers top-1; tok=all; select_site={site}",
                            component_part=sae_part_for_site("k1", site),
                        )
                        k_star_eval = max(1, int(k_star))
                        if SAE_KSTAR_ENABLED:
                            _append_all_layers_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part="kstar", token_selector="all", site=site
                                ),
                                site_raw=site_raw,
                                cls=cls,
                                k=k_star_eval,
                                note=(
                                    f"all-layers k*={k_star_eval} bag; tok=all; "
                                    f"select_site={site}"
                                ),
                                component_part=sae_part_for_site("kstar", site),
                            )
                        for k_fix in SAE_FIXED_KS:
                            k_fix = int(k_fix)
                            if k_fix <= 1:
                                continue
                            _append_all_layers_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part=f"k{k_fix}", token_selector="all", site=site
                                ),
                                site_raw=site_raw,
                                cls=cls,
                                k=k_fix,
                                note=(
                                    f"all-layers fixed-k={k_fix}; tok=all; "
                                    f"select_site={site}"
                                ),
                                component_part=sae_part_for_site(f"k{k_fix}", site),
                            )
                    else:
                        k_star = max(1, min(k_star, len(feats) or 1))
                        _append_sae_causal(
                            method_id=sae_causal_id(
                                cls, part="k1", token_selector="all", site=site
                            ),
                            layer=layer,
                            feature_indices=feats[:1],
                            target_class=cls,
                            note=f"top-1; tok=all; select_site={site}",
                            token_selector="all",
                        )
                        if not primary_only:
                            for tok in SAE_TOKEN_SELECTOR_ABLATIONS:
                                _append_sae_causal(
                                    method_id=sae_causal_id(
                                        cls, part="k1", token_selector=tok, site=site
                                    ),
                                    layer=layer,
                                    feature_indices=feats[:1],
                                    target_class=cls,
                                    note=(
                                        f"top-1; token_selector={tok}; select_site={site} "
                                        f"(causal-only ablation)"
                                    ),
                                    token_selector=tok,
                                )
                        if SAE_CLAMP_ENABLED and feats:
                            _append_sae_clamp_causal(
                                method_id=sae_causal_id(
                                    cls, part="k1_clamp", token_selector="all", site=site
                                ),
                                layer=layer,
                                feature_index=feats[0],
                                target_class=cls,
                                note=(
                                    f"top-1 activation-clamp; tok=all; select_site={site} "
                                    f"(causal-only ablation)"
                                ),
                                token_selector="all",
                            )
                        if feats and SAE_KSTAR_ENABLED:
                            k_star_eval = max(1, min(k_star, len(feats)))
                            _append_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part="kstar", token_selector="all", site=site
                                ),
                                layer=layer,
                                feature_indices=feats[:k_star_eval],
                                target_class=cls,
                                note=f"k*={k_star_eval} bag; tok=all; select_site={site}",
                                bag_reduce="sum",
                                token_selector="all",
                            )
                        for k_fix in SAE_FIXED_KS:
                            k_fix = int(k_fix)
                            if k_fix <= 1:
                                continue
                            if k_fix > len(feats):
                                print(
                                    f"  skip sae:{cls}:k{k_fix} causal site={site}: "
                                    f"only {len(feats)} features",
                                    flush=True,
                                )
                                continue
                            _append_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part=f"k{k_fix}", token_selector="all", site=site
                                ),
                                layer=layer,
                                feature_indices=feats[:k_fix],
                                target_class=cls,
                                note=f"fixed-k={k_fix} bag; tok=all; select_site={site}",
                                bag_reduce="sum",
                                token_selector="all",
                            )
                    fallback_layer = layer if layer is not None else SAE_LAYER
                    if SAE_OPP_FIRE_ENABLED:
                        opp_feats = list(fbc_opp.get(cls) or [])
                        opp_layer = _int_layer(
                            selected_opp.get(cls)
                            or site_raw.get("selected_layer_global_opp_fire"),
                            fallback_layer,
                        )
                        if opp_layer is not None:
                            _append_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part="sel_opp_fire", token_selector="all", site=site
                                ),
                                layer=opp_layer,
                                feature_indices=opp_feats[:1],
                                target_class=cls,
                                note=(
                                    f"opp_fire top-1 @ L{opp_layer} "
                                    f"(opp_fire_penalty_top1); tok=all; select_site={site}"
                                ),
                                token_selector="all",
                            )
                    if SAE_ARAD_ENABLED:
                        arad_feats = list(fbc_arad.get(cls) or [])
                        arad_layer = _int_layer(
                            selected_arad.get(cls)
                            or site_raw.get("selected_layer_global_arad_out"),
                            fallback_layer,
                        )
                        if arad_layer is not None:
                            _append_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part="sel_arad_out", token_selector="all", site=site
                                ),
                                layer=arad_layer,
                                feature_indices=arad_feats[:1],
                                target_class=cls,
                                note=(
                                    f"arad_out top-1 @ L{arad_layer} "
                                    f"(arad_output_score_top1); tok=all; "
                                    f"select_site={site}"
                                ),
                                token_selector="all",
                            )
                    if SAE_JH_F1_ENABLED:
                        jh_feats = list(fbc_jh.get(cls) or [])
                        jh_layer = _int_layer(
                            selected_jh.get(cls)
                            or site_raw.get("selected_layer_global_jh_f1"),
                            fallback_layer,
                        )
                        if jh_layer is not None:
                            _append_sae_causal(
                                method_id=sae_causal_id(
                                    cls, part="sel_jh_f1", token_selector="all", site=site
                                ),
                                layer=jh_layer,
                                feature_indices=jh_feats[:1],
                                target_class=cls,
                                note=(
                                    f"jh_f1 top-1 @ L{jh_layer} "
                                    f"(jh_calibrated_f1_top1); tok=all; select_site={site}"
                                ),
                                token_selector="all",
                            )

                by_layer_raw = site_raw.get("by_layer") or {}
                if not primary_only:
                    print(
                        f"  SAE per-layer k1 causal site={site} "
                        f"({len(by_layer_raw)} layers × {len(sae_classes)} classes) …",
                        flush=True,
                    )
                for layer_key, layer_raw in (
                    by_layer_raw.items() if not primary_only else ()
                ):
                    try:
                        layer_i = int(layer_key)
                    except (TypeError, ValueError):
                        continue
                    fbc_L = (
                        ((layer_raw.get("selections") or {}).get("per_class") or {}).get(
                            "features_by_class"
                        )
                        or {}
                    )
                    for cls in sae_classes:
                        cls = str(cls)
                        feats_L = list(fbc_L.get(cls) or [])
                        if not feats_L:
                            continue
                        _append_sae_causal(
                            method_id=method_part_id(
                                sae_backend_for_site(site),
                                cls,
                                sae_part_for_site(f"L{layer_i}_k1", site),
                            ),
                            layer=layer_i,
                            feature_indices=feats_L[:1],
                            target_class=cls,
                            note=f"per-layer top-1 @ L{layer_i}; tok=all; select_site={site}",
                            token_selector="all",
                        )

                if not primary_only:
                    print(
                        f"  SAE all-layers fixed-k causal site={site} "
                        f"({len(by_layer_raw)} layers × {list(SAE_FIXED_KS)} × "
                        f"{list(sae_classes)}) …",
                        flush=True,
                    )
                for cls in (sae_classes if not primary_only else ()):
                    cls = str(cls)
                    ks_all = list(dict.fromkeys([1, *SAE_FIXED_KS]))
                    for k_fix in ks_all:
                        k_fix = int(k_fix)
                        mid = sae_encoder_all_k_id(cls, k_fix, site=site)
                        _append_all_layers_sae_causal(
                            method_id=mid,
                            site_raw=site_raw,
                            cls=cls,
                            k=k_fix,
                            note=(
                                f"top-{k_fix} per layer; tok=all; select_site={site}"
                            ),
                            component_part=sae_part_for_site(f"all_k{k_fix}", site),
                        )

                joint_layer = global_layer if global_layer is not None else SAE_LAYER
                if joint_feats:
                    for cls in sae_classes:
                        cls = str(cls)
                        _append_sae_causal(
                            method_id=sae_joint_method_id(cls, site=site),
                            layer=joint_layer,
                            feature_indices=joint_feats[:1],
                            target_class=cls,
                            note=(
                                f"joint contrastive feature; ± polarity toward class; "
                                f"select_site={site}"
                            ),
                        )
                else:
                    print(f"  sae:joint:* site={site}: no joint feature; skip", flush=True)

            site_payloads = list(iter_sae_site_payloads(sae_raw))
            if not site_payloads:
                site_payloads = [(_site, sae_raw)]
            for site, site_raw in site_payloads:
                extra_hf = site_raw.get("_hf_module_by_layer") or {}
                extra_sae_map = site_raw.get("_sae_by_layer") or {}
                hf_by_layer.update(
                    {
                        int(k): v
                        for k, v in extra_hf.items()
                        if v is not None and _int_layer(k) is not None
                    }
                )
                sae_by_layer.update(
                    {
                        int(k): v
                        for k, v in extra_sae_map.items()
                        if v is not None and _int_layer(k) is not None
                    }
                )
                active_site[0] = str(site)
                _run_sae_causal_site(str(site), site_raw)
            # `_ensure_sae` deliberately keeps the current layer resident so
            # the several causal recipes for that layer do not reload it.  No
            # later section needs that SAE, however, so release the final
            # cache entry as well.  Without this, the last 4-GiB Llama-Scope
            # dictionary remains live until the whole study process exits.
            for cached_layer, cached_sae in list(sae_by_layer.items()):
                sae_by_layer.pop(cached_layer, None)
                _offload_sae(cached_sae)
                del cached_sae
            _collect_cuda_memory()
            _sae_new_results = results[_sae_causal_sec_start_idx:]
            _sae_causal_sec.meta["n_method_rows"] = len(_sae_new_results)
            _sae_causal_sec.meta["n_strength_points"] = sum(
                len(r.strengths or []) for r in _sae_new_results
            )
            _sae_causal_sec.stop()

    # --- CAA residual steering (L* + :all; concat encode has no separate steer) ---
    caa = caa_raw or {}
    if caa.get("vectors") and not caa.get("error"):
        _caa_causal_sec = CostSection(
            "causal:caa", phase="causal", backend="caa"
        ).start()
        _caa_causal_sec_start_idx = len(results)
        print(
            "\n  === CAA causal (mean-diff residual add; tok=all; LMS-gated) ===",
            flush=True,
        )
        model = caa.get("_model")
        tokenizer = caa.get("_tokenizer")
        if model is None or tokenizer is None:
            model, tokenizer = _trainer_model_tok()
        if model is None or tokenizer is None:
            raise RuntimeError(
                "CAA causal needs a HF model/tokenizer. METHODS=caa must load "
                "the study backbone; trainer reload and from_pretrained both failed."
            )
        else:
            _caa_base_probs, _caa_base_lms = _cached_base_stats(model, tokenizer, meta_rows)
            _caa_val_base_probs, _caa_val_base_lms = _cached_base_stats(model, tokenizer, val_rows)
            fitted = load_caa_vectors_payload(caa.get("vectors") or {})
            for pol in CAA_CAUSAL_POLICIES:
                vecs = fitted.get(str(pol)) or fitted.get("mean" if pol == "all" else str(pol))
                if vecs is None:
                    print(f"  CAA causal act={pol}: no vectors; skip", flush=True)
                    continue
                for vector_key, contrast in vecs.contrasts():
                    cls = str(contrast.get("target_class"))
                    if cls not in {str(c) for c in TARGET_CLASSES}:
                        continue
                    pair = contrast.get("pair")
                    if primary_only:
                        selected_part = selected_caa_part_by_validation_detection(
                            caa.get("method_metrics") or {},
                            cls=cls,
                            policy=str(pol),
                            pair=pair,
                            layers=vecs.layers,
                        )
                        if selected_part is None:
                            print(
                                f"  CAA {cls} act={pol}: incomplete validation "
                                "candidate pool; no selected causal site",
                                flush=True,
                            )
                            continue
                        parts = (None,) if selected_part == "__concat__" else (selected_part,)
                        print(
                            f"  CAA {cls} act={pol}: validation-selected "
                            f"site={selected_part}",
                            flush=True,
                        )
                    else:
                        # Full suites retain concat, :all, and every layer.
                        parts = ("all", *[f"L{L}" for L in vecs.layers])
                    for part in parts:
                        mid = caa_encoder_id(cls, policy=pol, part=part, pair=pair)
                        if mid in _skip_causal:
                            print(
                                f"  skip-existing: {mid} causal already complete",
                                flush=True,
                            )
                            continue
                        # Skip layers not in smoke SAE_LAYERS list already filtered at fit.
                        try:
                            specs = specs_for_part(vecs, vector_key, part)
                        except Exception as exc:
                            track_error(exc, context=f"{mid}")
                            if fail_fast:
                                _persist_partial_before_abort()
                                raise
                            continue
                        print(
                            f"  {mid}: n_sites={len(specs)} strengths={list(CAA_CAUSAL_STRENGTHS)} …",
                            flush=True,
                        )
                        try:
                            results.append(
                                run_caa_causal_sweep(
                                    model,
                                    tokenizer,
                                    val_rows,
                                    specs=specs,
                                    target_class=cls,
                                    strengths=CAA_CAUSAL_STRENGTHS,
                                    lms_texts=None,
                                    report_rows=meta_rows,
                                    include_random_control=True,
                                    method_id=mid,
                                    token_selector="all",
                                    act_policy=pol,
                                    part=part,
                                    base_probs=_caa_val_base_probs,
                                    base_lms=_caa_val_base_lms,
                                    r_base_probs=_caa_base_probs,
                                    r_base_lms=_caa_base_lms,
                                    reuse_strengthen=_reuse_for(mid),
                                )
                            )
                            sel = results[-1]
                            sel.meta.update(
                                {
                                    "ablation": contrast.get("ablation"),
                                    "pair": pair,
                                    "vector_key": vector_key,
                                }
                            )
                            print(
                                f"    selected strength={sel.selected_strength} "
                                f"signed_effect="
                                f"{None if sel.selected is None else round(sel.selected.signed_effect, 4)} "
                                f"lms_ok={None if sel.selected is None else sel.selected.lms_ok}",
                                flush=True,
                            )
                        except Exception as exc:
                            track_error(exc, context=f"{mid} causal")
                            if fail_fast:
                                _persist_partial_before_abort()
                                raise
                            results.append(
                                CausalMethodResult(
                                    method=mid,
                                    backend="caa",
                                    target_class=cls,
                                    strengths=[],
                                    selected_strength=None,
                                    selected=None,
                                    meta={
                                        "error": str(exc),
                                        "act_policy": pol,
                                        "part": part,
                                        "ablation": contrast.get("ablation"),
                                        "pair": pair,
                                        "vector_key": vector_key,
                                    },
                                )
                            )
                    if primary_only:
                        continue
                    # Concat encode id gets the same multi-layer steer (fair vs actiend none).
                    mid_full = caa_encoder_id(cls, policy=pol, part=None, pair=pair)
                    if mid_full in _skip_causal:
                        print(
                            f"  skip-existing: {mid_full} causal already complete",
                            flush=True,
                        )
                        continue
                    try:
                        specs = specs_for_part(vecs, vector_key, "all")
                        print(
                            f"  {mid_full}: multi-layer steer (=:all; concat encode analogue) …",
                            flush=True,
                        )
                        results.append(
                            run_caa_causal_sweep(
                                model,
                                tokenizer,
                                val_rows,
                                specs=specs,
                                target_class=cls,
                                strengths=CAA_CAUSAL_STRENGTHS,
                                lms_texts=None,
                                report_rows=meta_rows,
                                include_random_control=True,
                                method_id=mid_full,
                                token_selector="all",
                                act_policy=pol,
                                part=None,
                                base_probs=_caa_val_base_probs,
                                base_lms=_caa_val_base_lms,
                                r_base_probs=_caa_base_probs,
                                r_base_lms=_caa_base_lms,
                                reuse_strengthen=_reuse_for(mid_full),
                            )
                        )
                        results[-1].meta.update(
                            {
                                "ablation": contrast.get("ablation"),
                                "pair": pair,
                                "vector_key": vector_key,
                            }
                        )
                    except Exception as exc:
                        track_error(exc, context=f"{mid_full} causal")
                        if fail_fast:
                            _persist_partial_before_abort()
                            raise
                        results.append(
                            CausalMethodResult(
                                method=mid_full,
                                backend="caa",
                                target_class=cls,
                                strengths=[],
                                selected_strength=None,
                                selected=None,
                                meta={
                                    "error": str(exc),
                                    "act_policy": pol,
                                    "part": None,
                                    "ablation": contrast.get("ablation"),
                                    "pair": pair,
                                    "vector_key": vector_key,
                                },
                            )
                        )
            _caa_new_results = results[_caa_causal_sec_start_idx:]
            _caa_causal_sec.meta["n_method_rows"] = len(_caa_new_results)
            _caa_causal_sec.meta["n_strength_points"] = sum(
                len(r.strengths or []) for r in _caa_new_results
            )
            _caa_causal_sec.stop()

    # --- CAGA activation-gradient steering (closed-form mean; direct CAA path) ---
    # CAGA fits a closed-form dL/dh mean direction from its own (caga) trainer and
    # STEERS activations like CAA (not weight rewrite). It rides the same
    # claim-filtered iter_trainer_class_groups the IEND backends use, so answer-only
    # circuit tasks (ioi SUBJECT, key_value OTHER) are never targeted. One-pole
    # per-class ids (caga:C); two-pole caga:A-B:C is a fast-follow.
    if "caga" in enabled_set:
        # CAGA is the closed-form activation-gradient mean baseline. AGIEND is
        # intentionally absent here: its learned decoder now runs through the
        # gradiend package evaluator/intervention API above.
        from caga_eval import caga_layer_specs, caga_specs_from_model

        caga_none = {
            k: v for k, v in (train_raw_none or {}).items() if is_caga_backend(k)
        }
        caga_by_class = None
        if none_by_class:
            caga_by_class = {
                b: m for b, m in none_by_class.items() if is_caga_backend(b)
            } or None
        if caga_none or caga_by_class:
            _caga_sec = CostSection(
                "causal:activation_gradient", phase="causal", backend="caga"
            ).start()
            _caga_start_idx = len(results)
            _caga_emitted: set[str] = set()
            print(
                "\n  === CAGA causal (closed-form dL/dh activation steering; LMS-gated) ===",
                flush=True,
            )
            for backend, raw, class_ids in iter_trainer_class_groups(
                caga_none, target_classes=TARGET_CLASSES, train_raw_by_class=caga_by_class,
            ):
                if str(raw.get("split_mode") or "none") != "none":
                    continue
                trainer = raw.get("trainer")
                if trainer is None:
                    continue
                selected_direction_parts: Dict[str, str] = {}
                if primary_only and layerwise_caga:
                    for cls in class_ids:
                        chosen = selected_direction_part_by_validation_detection(
                            raw, cls=str(cls)
                        )
                        if chosen is None:
                            print(
                                f"  {backend}:{cls}: incomplete validation aggregate/layer pool; "
                                "no selected causal representation",
                                flush=True,
                            )
                            continue
                        selected_direction_parts[str(cls)] = chosen
                        print(
                            f"  {backend}:{cls}: validation-selected representation={chosen}",
                            flush=True,
                        )
                    # Do not load the backbone merely because unselected layer
                    # ablation ids are absent.  Core owns exactly the locked
                    # aggregate-or-layer id for each class.
                    selected_caga_ids = [
                        feature_or_causal_id(
                            backend,
                            cls,
                            raw,
                            layer=None if part == "aggregate" else part,
                        )
                        for cls, part in selected_direction_parts.items()
                    ]
                    if not selected_caga_ids:
                        continue
                    if all(
                        mid in _caga_emitted or mid in _skip_causal
                        for mid in selected_caga_ids
                    ):
                        print(
                            f"  skip-existing: {backend} validation-selected causal "
                            "representation(s) already complete",
                            flush=True,
                        )
                        continue
                try:
                    c_model = trainer.get_model().base_model
                    c_tok = trainer.tokenizer
                    try:
                        # Read the direction fitted at train time and persisted in the
                        # checkpoint decoder (no re-fit; no train-data dependency here).
                        specs = caga_specs_from_model(trainer.get_model())
                        layer_specs = (
                            caga_layer_specs(trainer.get_model()) if layerwise_caga else {}
                        )
                    except Exception as exc:
                        track_error(exc, context=f"caga specs {backend}")
                        if fail_fast:
                            _persist_partial_before_abort()
                            raise
                        continue
                    spec_variants = [(None, specs)] + [
                        (f"L{int(layer)}", one_spec)
                        for layer, one_spec in layer_specs.items()
                    ]
                    caga_ids = [
                        feature_or_causal_id(backend, cls, raw, layer=part)
                        for cls in class_ids
                        for part, _variant_specs in spec_variants
                        if not primary_only
                        or not layerwise_caga
                        or selected_direction_parts.get(str(cls))
                        == ("aggregate" if part is None else str(part))
                    ]
                    pending_caga_ids = [
                        mid
                        for mid in caga_ids
                        if mid not in _caga_emitted and mid not in _skip_causal
                    ]
                    if not pending_caga_ids:
                        continue
                    # A repaired CAGA row is reselected entirely from its stored +/-
                    # validation summaries. Only the frozen selected candidates and
                    # random control run on test, so no validation baseline forward is
                    # needed when every pending row is repairable.
                    if _all_polarity_repairable(pending_caga_ids):
                        c_val_probs = c_val_lms = None
                    else:
                        c_val_probs, c_val_lms = _cached_base_stats(
                            c_model, c_tok, val_rows
                        )
                    c_meta_probs, c_meta_lms = _cached_base_stats(
                        c_model, c_tok, meta_rows
                    )
                    for cls in class_ids:
                        # Match the encoding row id (pair -> caga:A-B:C, one-pole ->
                        # caga:C) so causal ATTACHES to it and inherits its pole/ablation,
                        # exactly as the weight-rewrite backends do -- CAGA comes in both
                        # a pairwise (two_pole) and a one-pole regime like CGA.
                        for part, variant_specs in spec_variants:
                            candidate_part = "aggregate" if part is None else str(part)
                            if primary_only and layerwise_caga and (
                                selected_direction_parts.get(str(cls)) != candidate_part
                            ):
                                continue
                            mid = feature_or_causal_id(
                                backend, cls, raw, layer=part
                            )
                            if mid in _caga_emitted:
                                continue
                            if mid in _skip_causal:
                                print(f"  skip-existing: {mid} causal already complete", flush=True)
                                _caga_emitted.add(mid)
                                continue
                            _caga_emitted.add(mid)
                            try:
                                res = run_caa_causal_sweep(
                                    c_model,
                                    c_tok,
                                    val_rows,
                                    specs=variant_specs,
                                    target_class=cls,
                                    strengths=CAA_CAUSAL_STRENGTHS,
                                    report_rows=meta_rows,
                                    include_random_control=True,
                                    method_id=mid,
                                    token_selector="all",
                                    act_policy="prediction",
                                    part=part,
                                    base_probs=c_val_probs,
                                    base_lms=c_val_lms,
                                    r_base_probs=c_meta_probs,
                                    r_base_lms=c_meta_lms,
                                    reuse_strengthen=_reuse_for(mid),
                                    reuse_reselect_bidirectional=_polarity_repair_for(mid),
                                )
                                res.backend = "caga"  # sweep hardcodes 'caa'; relabel for grouping
                                results.append(res)
                                print(
                                    f"    {mid} strength={res.selected_strength} "
                                    f"signed_effect="
                                    f"{None if res.selected is None else round(res.selected.signed_effect, 4)} "
                                    f"lms_ok={None if res.selected is None else res.selected.lms_ok}",
                                    flush=True,
                                )
                            except Exception as exc:
                                track_error(exc, context=f"{mid} causal")
                                if fail_fast:
                                    _persist_partial_before_abort()
                                    raise
                                results.append(
                                    CausalMethodResult(
                                        method=mid, backend="caga", target_class=cls,
                                        strengths=[], selected_strength=None, selected=None,
                                        meta={"error": str(exc), "component_part": part},
                                    )
                                )
                finally:
                    # release_deferred_trainer_after_causal only clears
                    # trainer._model_instance -- it has no visibility into this
                    # loop's OWN c_model/specs locals, which hold direct
                    # references to the same base model (c_model = trainer.get_
                    # model().base_model, line ~3021). Until those are dropped
                    # too, the just-finished group's full backbone stays
                    # reachable (refcount > 0) through the ENTIRE next group's
                    # load *and* sweep -- release()'s del/gc.collect()/
                    # empty_cache() has nothing to reclaim, so two full
                    # backbones are resident back-to-back instead of one. Drop
                    # them here, before release(), so the cleanup actually has
                    # zero references left to free. Root cause of repeated
                    # "OOMs after a handful of groups" on large models
                    # (qwen3.5-9b-base caga, job 4222493_33/_40, 4226238_0,
                    # 4222493_47/religion). Mirrors the GRADIEND/ACTIEND loop
                    # above, which has no equivalent leak (it never binds the
                    # base model to a loop-scoped local).
                    c_model = c_tok = specs = layer_specs = spec_variants = None
                    release_deferred_trainer_after_causal(trainer, raw, backend)
            _caga_sec.meta["n_method_rows"] = len(results) - _caga_start_idx
            _caga_sec.stop()

    causal_dir = persist_causal_bundle(
        OUTPUT_DIR,
        results,
        write_curve_samples=bool(persist_curve_samples),
        write_sample_text=bool(persist_sample_text),
        max_samples_per_method=persist_max_samples,
    )
    catalog_path = OUTPUT_DIR / "decoder_artifacts.json"
    catalog_path.write_text(
        __import__("json").dumps(decoder_catalog, indent=2),
        encoding="utf-8",
    )
    print(
        f"  Wrote causal bundle to {causal_dir}  "
        f"({len(results)} methods; curve_samples={bool(persist_curve_samples)} "
        f"sample_text={bool(persist_sample_text)})  decoder catalog={catalog_path}",
        flush=True,
    )
    summaries = []
    for r in results:
        entry = r.to_dict() if hasattr(r, "to_dict") else {}
        try:
            entry["headline_metrics"] = headline_metrics(r)
        except Exception as exc:
            if fail_fast:
                raise
            entry["headline_metrics"] = {"causal_error": str(exc)}
            meta = dict(entry.get("meta") or {})
            meta.setdefault("error", str(exc))
            entry["meta"] = meta
        summaries.append(entry)
    return {
        "dir": causal_dir,
        "by_method": {r.method: r for r in results},
        "summaries": summaries,
        "decoder_artifacts": decoder_catalog,
        "decoder_artifacts_path": str(catalog_path),
    }
