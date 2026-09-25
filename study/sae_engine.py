"""SAE engine driven by ``study/stages/sae.py``.

Configured through module globals (``TARGET_CLASSES``, ``SAE_LAYERS``,
``OUTPUT_DIR``, ...) set by the stage before it calls ``run_sae`` /
``_sae_method_rows``; there is no CLI entry point.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from gender_en_data import geneutral_split_kwargs, read_geneutral
from results_schema import (
    method_result,
)
from cost_timer import (
    cost_timer,
    record_cost,
)
# Plot/report modules are deliberately not imported here: workers launched with
# --skip-reports must not require paper-analysis modules.
from causal_eval import (
    CAUSAL_N_PER_GROUP,
    DECODER_EVAL_MAX_SIZE,
    DEFAULT_CAUSAL_STRENGTHS,
)
from study_models import (
    DEFAULT_STUDY_MODEL,
    StudyModel,
    set_active_model,
)
from sae_eval import (
    bootstrap_roc_auc,
    encode_sae,
    frozen_decision_rules,
    require_frozen_rules,
    identify_sae_features,
    joint_readout_scores,
    load_sae_cache,
    method_part_id,
    sae_encoder_kstar_id,
    sae_encoder_all_k_id,
    sae_part_for_site,
    sae_backend_for_site,
    normalize_sae_select_sites,
    release_sae_layer_heavy_memory,
    release_sae_layer_device_memory,
    release_sae_raw_heavy_memory,
    compact_all_k_layer_slices,
    sae_all_layers_class_vs_neutral_metrics,
    MAX_NEUTRAL_TOKEN_ROWS,
    NEUTRAL_TOKEN_SUBSAMPLE_SEED,
    per_class_fair_curve,
    sae_cache_fingerprint,
    sae_class_vs_neutral_curve,
    sae_class_vs_neutral_metrics,
    sae_joint_readout_metrics,
    sae_per_class_readout_metrics,
    sae_sparse_readout_metrics,
    sae_specificity,
    save_sae_cache,
    select_per_class_k_on_split,
    select_readout_k_on_split,
    select_k_from_curve_rows,
    select_per_class_features,
    k_selection_score,
    K_SELECTION_BALANCED,
    selection_to_dict,
    sparse_fair_curve,
)
from localization_eval import (
    DEFAULT_RESID_M,
)
from caa_eval import (
    CAA_ACT_POLICIES,
)

# ----- active model (set by apply_study_model / CLI --model) -----
STUDY_MODEL_CFG: StudyModel = set_active_model(DEFAULT_STUDY_MODEL)
SAE_MODEL = STUDY_MODEL_CFG.sae_model
HF_MODEL = STUDY_MODEL_CFG.hf_model
SAE_RELEASE = STUDY_MODEL_CFG.sae_release
OUTPUT_DIR = STUDY_MODEL_CFG.output_dir
SAE_LAYERS = STUDY_MODEL_CFG.layers
SAE_LAYER = SAE_LAYERS[-1]

NAMES_PER_TEMPLATE = 4
MAX_STEPS = 1000
EVAL_STEPS = 100
LEARNING_RATE = 1e-5
TRAIN_BATCH_SIZE = 8
ENCODER_EVAL_MAX = 200
TARGET_CLASSES = ("M", "F")
SAE_USE_NEUTRAL_SPECIFICITY = True  # biasneutral-ajibawa as specificity control during selection
# Primary: per_class (1 SAE feat / class, readout z_M−z_F) + joint (1 contrastive feat).
# Ablations: old top-k bag rankings.
SAE_SELECTION_MODES = (
    "per_class",
    "joint",
    "pairwise",
    "pair_aware",
    "dense_probe",
)
SAE_BOOTSTRAP_SAMPLES = 0  # set >0 for selection stability
SAE_TOP_K = 128  # candidates stored per class (per_class) / bag size (ablations)
# Fixed-k ablations: strict powers of 2 (encoder + causal share ids).
SAE_FIXED_KS = (1, 2, 4, 8, 16, 32, 64, 128)
# Sparse k* search grid = same power-of-2 ladder (must cover FIXED_KS).
SAE_READOUT_KS = SAE_FIXED_KS
SAE_BOOTSTRAP_AUC = 100  # test-set AUC bootstrap resamples for ±std
# Alias used in metrics / report (must match sae_eval.K_SELECTION_BALANCED).
K_SELECTION = K_SELECTION_BALANCED
# Decoder / ACTIEND LR grid (shared with DEFAULT_CAUSAL_STRENGTHS, now ≤100000).
CAUSAL_STRENGTHS = list(DEFAULT_CAUSAL_STRENGTHS)
ACTIEND_DECODER_LRS = list(DEFAULT_CAUSAL_STRENGTHS)
CAUSAL_LAYER_LR_CEILING_TODO = (
    "ACTIEND per-layer causal uses extended LR grid (≤100000); "
    "ceiling still means extend further before concluding ineffectiveness."
)

# SAE selection opposite-class ablation: encoder + causal share sae:{cls}:sel_opp_fire.
# Layer pick uses the *same* ranking score as feature ID (top-1 @ layer; top-k stay in-layer).
SAE_SELECTION_OPPOSITE_CLASS_TODO = (
    "Opp-fire ranking + layer pick: sae:{cls}:sel_opp_fire (encoder + causal, same id)."
)
# Arad et al. 2025 output-score filter (public code): sae:{cls}:sel_arad_out.
# Re-ranks per_class candidates by output score; layer pick uses top-1 Arad score.
SAE_SELECTION_ARAD_TODO = (
    "Arad output-score ranking + layer pick: sae:{cls}:sel_arad_out "
    "(encoder + causal; adapted from technion-cs-nlp/saes-are-good-for-steering)."
)
# Jørgensen & Hansen 2026 supervised calibrated-F1 labelling (AxBench re-eval):
# sae:{cls}:sel_jh_f1 — how much supervised retrieval the SAE is granted.
SAE_SELECTION_JH_TODO = (
    "JH calibrated-F1 supervised ranking + layer pick: sae:{cls}:sel_jh_f1 "
    "(encoder + causal; adapted from MikkelGodsk/SAE-labelling)."
)
# Primary SAE site selection is locked by validation detection. The old
# encoding_E label described an obsolete selection protocol.
LAYER_SELECTION_PER_CLASS = "validation_detection"
LAYER_SELECTION_OPP_FIRE = "opp_fire_penalty_top1"
LAYER_SELECTION_ARAD_OUT = "arad_output_score_top1"
LAYER_SELECTION_JH_F1 = "jh_calibrated_f1_top1"
# How many mean-diff candidates per class to score with Arad forwards (expensive).
SAE_ARAD_CANDIDATE_K = 16
SAE_ARAD_ENABLED = True
SAE_JH_F1_ENABLED = True
SAE_OPP_FIRE_ENABLED = True  # suite may disable (core drops opp_fire)
SAE_JH_PI0 = 1e-3
SAE_JH_SUPPORT_MIN = 50

# Causal SAE ablations share encoder ids when tok=all; _tok_prediction is causal-only.
CAUSAL_ABLATIONS_TODO = (
    "Encoder↔causal aligned ids; ACTIEND tok_*; SAE fixed-k + all_k + kstar "
    "+ opp_fire + arad_out + jh_f1 + joint."
)
ACTIEND_CAUSAL_ABLATIONS = (
    # Ungated scopes (vs default all + encoder_direction gate).
    {"token_selector": "all", "activation_gate": None},
    {"token_selector": "prediction", "activation_gate": None},
)
SAE_TOKEN_SELECTOR_ABLATIONS = ("prediction",)  # default all uses shared encoder id

# SAE residual strengths (aligned to ACTIEND-style log grid, ≤100000).
SAE_CAUSAL_STRENGTHS = tuple(DEFAULT_CAUSAL_STRENGTHS)
CAUSAL_N = CAUSAL_N_PER_GROUP  # per group (M / F / neutral)
# Shared with LMS: package evaluate_decoder(max_size) caps training-like + neutral/LMS.
DECODER_MAX_SIZE = DECODER_EVAL_MAX_SIZE
# Causal disk: curve dumps are huge (sample × strength × prompt). Off by default;
# study/stages/causal.py overrides from configs/defaults.yaml ``causal.persist_*``.
PERSIST_CURVE_SAMPLES = False
PERSIST_SAMPLE_TEXT = False
PERSIST_MAX_SAMPLES = None  # None = all selected samples in samples_*.jsonl
SKIP_SAE = False
SKIP_CAUSAL = False
SKIP_CAA = False
METHODS = ("actiend", "gradiend")
# Shared ACTIEND/CAA residual site (see activation_protocol.py).
# SAE also always scores classical last-context (``pre_prediction``) unless
# ``SAE_SELECT_SITES`` is overridden. Causal decoder-only models cannot attend
# forward, so ``pre_prediction`` is the residual used to generate the fill.
ACTIVATION_SITE = "prediction"
# None: mixed-site ACTIEND when ACTIVATION_SITE is pre_prediction
# (encoder H_{t-1}, decoder target = filled prediction diff).
ACTIEND_TARGET_SITE = None
SAE_SELECT_SITES = ("prediction", "pre_prediction")
# CAA activation-pooling ablations (construction + cosine encode).
# Include shared primary sites; mean/last remain extra ablations.
CAA_POLICIES = CAA_ACT_POLICIES
CAA_CAUSAL_POLICIES = CAA_ACT_POLICIES
CAA_CAUSAL_STRENGTHS = tuple(DEFAULT_CAUSAL_STRENGTHS)
# Package evaluate_* disk cache: always False at call sites (see PLANNING.md §8.6).
# None-split ACTIEND ablations share one decoder_grid_cache.json per experiment_dir
# and would overwrite each other if use_cache=True. Resume via ExperimentCache /
# --skip-existing instead.
USE_CACHE = False
# Study-managed stage cache (file existence + config hash).
EXPERIMENT_CACHE = True
FORCE_STAGES: tuple = ()
# Split ablation: none = unpartitioned; tensors = GradiendSplit.by_tensor().
GRADIEND_SPLIT_MODES = ("none", "tensors")
# Gradient GRADIEND only: pre-prune selects parameter dims from gradient stats; ACTIEND has no pre-prune API yet.
PRE_PRUNE_N_SAMPLES = 16
PRE_PRUNE_TOPK = 0.1
POST_PRUNE_TOPK = 0.01
# Training source/target — study bridge overrides from configs/defaults.yaml.
TRAIN_SOURCE = "alternative"
TRAIN_TARGET = "diff"
# Localization: global top-k for coarse bags; resid_m = per-layer neuron bridge width.
LOCALIZATION_RESID_M = DEFAULT_RESID_M
SKIP_LOCALIZATION = False
# Persist selected causal rewrites under runs/.../modified_models/ (large; opt-in).
SAVE_MODIFIED_MODELS = False


# Optional override for non-gender tasks (deep pipeline sets STUDY_NEUTRAL_DF).
STUDY_NEUTRAL_DF = None
STUDY_EXCLUDED_WORDS = None
GENDER_EXCLUDED_WORDS = ["he", "she", "him", "her", "his", "hers"]


def apply_study_model(key: str) -> StudyModel:
    """Switch backbone + SAE suite; updates module globals used by the runner."""
    global STUDY_MODEL_CFG, SAE_MODEL, HF_MODEL, SAE_RELEASE, OUTPUT_DIR, SAE_LAYERS, SAE_LAYER
    STUDY_MODEL_CFG = set_active_model(key)
    SAE_MODEL = STUDY_MODEL_CFG.sae_model
    HF_MODEL = STUDY_MODEL_CFG.hf_model
    SAE_RELEASE = STUDY_MODEL_CFG.sae_release
    OUTPUT_DIR = STUDY_MODEL_CFG.output_dir
    SAE_LAYERS = STUDY_MODEL_CFG.layers
    SAE_LAYER = SAE_LAYERS[-1]
    return STUDY_MODEL_CFG


def apply_smoke():
    global NAMES_PER_TEMPLATE, MAX_STEPS, EVAL_STEPS, ENCODER_EVAL_MAX, SAE_TOP_K, OUTPUT_DIR
    global PRE_PRUNE_N_SAMPLES, CAUSAL_STRENGTHS, CAUSAL_N, DECODER_MAX_SIZE, SAE_LAYERS, SAE_LAYER, LOCALIZATION_RESID_M
    global SAE_READOUT_KS, SAE_FIXED_KS, SAE_CAUSAL_STRENGTHS, ACTIEND_DECODER_LRS, CAA_CAUSAL_STRENGTHS
    global CAA_CAUSAL_POLICIES, SAE_ARAD_CANDIDATE_K, SAE_JH_SUPPORT_MIN
    NAMES_PER_TEMPLATE = 1
    MAX_STEPS = 5
    EVAL_STEPS = 5
    ENCODER_EVAL_MAX = 32
    SAE_TOP_K = 16
    SAE_ARAD_CANDIDATE_K = 4
    SAE_JH_SUPPORT_MIN = 2
    PRE_PRUNE_N_SAMPLES = 4
    CAUSAL_N = 4
    DECODER_MAX_SIZE = 4
    CAUSAL_STRENGTHS = [0.1, 1.0, 10.0, 100.0]
    SAE_CAUSAL_STRENGTHS = (0.1, 1.0, 10.0, 100.0)
    CAA_CAUSAL_STRENGTHS = (0.1, 1.0, 10.0, 100.0)
    CAA_CAUSAL_POLICIES = ("prediction",)  # smoke: encode all policies; causal only prediction
    ACTIEND_DECODER_LRS = list(CAUSAL_STRENGTHS)
    SAE_LAYERS = tuple(SAE_LAYERS[:2]) if SAE_LAYERS else (0, 1)
    SAE_LAYER = SAE_LAYERS[-1]
    LOCALIZATION_RESID_M = 10
    SAE_FIXED_KS = (1, 2, 4, 8, 16)
    SAE_READOUT_KS = SAE_FIXED_KS
    OUTPUT_DIR = STUDY_MODEL_CFG.smoke_output_dir()








def _json_ready(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value








def _sae_encode_latents_by_class(
    model,
    tokenizer,
    sae,
    df,
    *,
    layer: int,
    label_tokens,
    split: str = "test",
    act_policy: str | None = None,
) -> dict:
    """Encode activations per ``label_class`` (for rival readout) at ``act_policy``."""
    if df is None or getattr(df, "empty", True) or "label_class" not in df.columns:
        return {}
    from sae_eval import collect_study_activations, encode_sae

    sub = df[df["split"] == split] if "split" in df.columns else df
    if sub.empty:
        sub = df
    out: dict = {}
    for oc in sorted(set(sub["label_class"].astype(str))):
        rows = sub[sub["label_class"].astype(str) == oc]
        if rows.empty:
            continue
        acts = collect_study_activations(
            model,
            tokenizer,
            rows,
            layer=int(layer),
            act_policy=str(act_policy or ACTIVATION_SITE),
            label_tokens=label_tokens,
        )
        if acts is None or len(acts) == 0:
            continue
        out[str(oc)] = encode_sae(sae, acts)
    return out


def _sae_rival_readout_maps(
    target_class: str,
    encode_classes,
    factual_labels,
    factual_latents,
    *,
    rival_test_by_class: dict | None,
    rival_val_by_class: dict | None,
) -> tuple[list, dict, dict, str | None]:
    """Build other-class latent maps for class-vs-neutral readout (one-pole + rivals)."""
    others = [str(x) for x in encode_classes if str(x) != str(target_class)]
    labels_arr = np.asarray(factual_labels)
    other_by: dict = {}
    val_other_by: dict = {}
    for oc in list(others):
        if rival_test_by_class and oc in rival_test_by_class:
            other_by[oc] = rival_test_by_class[oc]
        elif oc in set(labels_arr.astype(str)):
            other_by[oc] = factual_latents[labels_arr == oc]
    for oc in list(others):
        if rival_val_by_class and oc in rival_val_by_class:
            val_other_by[oc] = rival_val_by_class[oc]
    # CF rivals may not appear in encode_classes — add from rival maps.
    if rival_test_by_class:
        for oc, arr in rival_test_by_class.items():
            oc = str(oc)
            if oc == str(target_class) or arr is None or len(arr) == 0:
                continue
            if oc not in other_by:
                others.append(oc)
                other_by[oc] = arr
            if rival_val_by_class and oc in rival_val_by_class:
                val_other_by.setdefault(oc, rival_val_by_class[oc])
    other_c = others[0] if others else None
    return others, other_by, val_other_by, other_c


def run_sae_layer(
    trainer,
    gender_df,
    layer: int,
    *,
    model=None,
    tokenizer=None,
    rival_eval_df=None,
    label_tokens=None,
    act_policy: str | None = None,
) -> dict:
    """Identify SAE features + fair metrics for one resid-post layer."""
    act_policy = str(act_policy or ACTIVATION_SITE)

    def _p(msg: str) -> None:
        print(msg, flush=True)

    classes = [str(c) for c in TARGET_CLASSES]
    present = (
        {str(x) for x in gender_df["label_class"].astype(str)}
        if gender_df is not None and "label_class" in getattr(gender_df, "columns", [])
        else set(classes)
    )
    if present:
        dropped = [c for c in classes if c not in present]
        classes = [c for c in classes if c in present] or sorted(present)
        if dropped:
            _p(
                f"  SAE L{layer}: drop absent label_class {dropped} "
                f"(one-pole CF poles; present={sorted(present)})"
            )
    if not classes:
        raise ValueError(f"SAE L{layer}: no TARGET_CLASSES present on label_class")
    class_a = classes[0]
    class_b = classes[1] if len(classes) >= 2 else None
    sae_modes = list(SAE_SELECTION_MODES)
    if class_b is None:
        bipolar = {"joint", "pairwise", "pair_aware", "dense_probe"}
        sae_modes = [m for m in sae_modes if m not in bipolar]
        if "per_class" not in sae_modes:
            sae_modes = ["per_class", *sae_modes]

    _p(f"\n=== SAE layer {layer} on {SAE_MODEL} (release={SAE_RELEASE}) ===")
    # Latent/selection cache is independent of TrainingArguments.use_cache and of
    # package evaluate_* caches (those stay False). Study sets USE_CACHE=True.
    use_cache = bool(USE_CACHE)
    selection_split = "validation"
    val_df = gender_df[gender_df["split"] == selection_split]
    if val_df.empty:
        val_df = gender_df[gender_df["split"] == "train"]
    test_df = gender_df[gender_df["split"] == "test"]
    if test_df.empty:
        test_df = gender_df
    if STUDY_NEUTRAL_DF is not None:
        neutral = STUDY_NEUTRAL_DF
    else:
        neutral = read_geneutral(**geneutral_split_kwargs(smoke=True))
    ncol = "text" if "text" in getattr(neutral, "columns", []) else (
        "masked" if "masked" in getattr(neutral, "columns", []) else None
    )
    excluded_words = list(STUDY_EXCLUDED_WORDS or GENDER_EXCLUDED_WORDS)
    if ncol and SAE_USE_NEUTRAL_SPECIFICITY:
        from neutral_protocol import texts_for_split

        val_neutral_texts = texts_for_split(neutral, "validation", fallback="all")
        test_neutral_texts = texts_for_split(neutral, "test", fallback="all")
        neutral_texts = val_neutral_texts
        val_neutral_scored = texts_for_split(
            neutral, "validation", max_rows=ENCODER_EVAL_MAX, fallback="all"
        )
        test_neutral_scored = texts_for_split(
            neutral, "test", max_rows=ENCODER_EVAL_MAX, fallback="all"
        )
    else:
        val_neutral_texts = []
        test_neutral_texts = []
        neutral_texts = []
        val_neutral_scored = []
        test_neutral_scored = []
    _p(
        f"  SAE L{layer}: val={len(val_df)} test={len(test_df)} "
        f"val_neu={len(val_neutral_texts)} test_neu={len(test_neutral_texts)} "
        f"use_cache={use_cache} classes={classes}"
    )

    fingerprint = sae_cache_fingerprint(
        release=SAE_RELEASE,
        layer=layer,
        modes=sae_modes,
        top_k=SAE_TOP_K,
        readout_ks=SAE_READOUT_KS,
        selection_split=selection_split,
        class_a=class_a,
        class_b=class_b or class_a,
        neutral_specificity=SAE_USE_NEUTRAL_SPECIFICITY,
        bootstrap_samples=SAE_BOOTSTRAP_SAMPLES,
        min_firing_rate=0.01,
        val_texts=val_df["masked"].tolist(),
        test_texts=test_df["masked"].tolist(),
        neutral_texts=neutral_texts,
        causal_strengths=CAUSAL_STRENGTHS,
        causal_n=CAUSAL_N,
        act_policy=act_policy,
        val_neutral_texts=val_neutral_texts,
        test_neutral_texts=test_neutral_texts,
        excluded_words=excluded_words,
    )

    _p(f"  SAE L{layer}: checking cache …")
    cached = load_sae_cache(OUTPUT_DIR, fingerprint) if use_cache else None
    if cached and cached.get("_mismatch") is not None:
        _p(f"  SAE L{layer} cache fingerprint mismatch on {cached['_mismatch']}; recomputing.")
        cached = None

    selections = None
    val_latents = None
    latents = None
    neutral_latents = None
    test_neutral_latents = None
    sae = None
    hf_module = fingerprint["hf_module"]

    if cached is not None:
        _p(f"  Using cached SAE L{layer} at {cached['cache_dir']}")
        selections = cached["selections"]
        val_latents = cached["val_latents"]
        latents = cached["test_latents"]
        neutral_latents = cached["neutral_latents"]
        test_neutral_latents = cached.get("test_neutral_latents")
        if test_neutral_latents is None:
            test_neutral_latents = neutral_latents
        test_labels = cached["test_labels"] or test_df["label_class"].tolist()
        record_cost(
            f"sae_select:L{layer}",
            0.0,
            phase="feature_select",
            backend="sae",
            ledger_unit="sae_layer_sweep",
            cache_hit=True,
            layer=int(layer),
        )
        record_cost(
            f"sae_encode:L{layer}",
            0.0,
            phase="encode",
            backend="sae",
            ledger_unit="sae_layer_sweep",
            cache_hit=True,
            layer=int(layer),
        )
    else:
        if model is None or tokenizer is None:
            _p(f"  SAE L{layer}: loading base model from trainer …")
            model_with_gradiend = trainer.get_model()
            model = model_with_gradiend.base_model
            tokenizer = trainer.tokenizer
        _p(f"  SAE L{layer}: identify features …")
        with cost_timer(
            f"sae_select:L{layer}",
            phase="feature_select",
            backend="sae",
            ledger_unit="sae_layer_sweep",
            layer=int(layer),
        ):
            from study.method_ids import label_tokens_from_config

            # Gender PoC defaults he/she; circuit tasks leave empty → per-row ``label``.
            _lt = label_tokens if label_tokens is not None else label_tokens_from_config(
                classes, {"target_tokens": {"M": "he", "F": "she"}}
            )
            selections, sae, hf_module, val_latents, _split_df, neutral_latents = identify_sae_features(
                model,
                tokenizer,
                gender_df,
                layer=layer,
                release=SAE_RELEASE,
                selection_split=selection_split,
                top_k=SAE_TOP_K,
                modes=sae_modes,
                class_a=class_a,
                class_b=class_b,
                target_classes=classes,
                neutral_texts=val_neutral_scored or None,
                neutral_specificity=SAE_USE_NEUTRAL_SPECIFICITY,
                bootstrap_samples=SAE_BOOTSTRAP_SAMPLES,
                progress=_p,
                act_policy=act_policy,
                label_tokens=_lt,
                excluded_words=excluded_words,
            )
        _p(f"  SAE L{layer}: encoding test split ({len(test_df)} texts) act={act_policy} …")
        with cost_timer(
            f"sae_encode:L{layer}",
            phase="encode",
            backend="sae",
            ledger_unit="sae_layer_sweep",
            layer=int(layer),
        ):
            from sae_eval import collect_study_activations, collect_study_neutral_activations

            latents = encode_sae(
                sae,
                collect_study_activations(
                    model,
                    tokenizer,
                    test_df,
                    layer=int(layer),
                    act_policy=act_policy,
                    label_tokens=_lt,
                ),
            )
            test_neutral_latents = None
            if test_neutral_scored:
                test_hidden = collect_study_neutral_activations(
                    model,
                    tokenizer,
                    test_neutral_scored,
                    layer=int(layer),
                    act_policy=act_policy,
                    excluded_words=excluded_words,
                )
                n_tok = int(test_hidden.shape[0])
                if n_tok > MAX_NEUTRAL_TOKEN_ROWS:
                    from sae_eval import subsample_rows

                    test_hidden = subsample_rows(
                        test_hidden,
                        MAX_NEUTRAL_TOKEN_ROWS,
                        seed=NEUTRAL_TOKEN_SUBSAMPLE_SEED,
                    )
                test_neutral_latents = encode_sae(sae, test_hidden)
        test_labels = test_df["label_class"].tolist()
        _p(f"  SAE L{layer}: test latents shape={tuple(latents.shape)}")

    rival_test_by_class: dict = {}
    rival_val_by_class: dict = {}
    if rival_eval_df is not None and class_b is None and not getattr(rival_eval_df, "empty", True):
        if model is None or tokenizer is None:
            mwg = trainer.get_model()
            model = mwg.base_model
            tokenizer = trainer.tokenizer
        if sae is None:
            from sae_eval import load_sae, resid_sites

            _, sae_id = resid_sites(int(layer))
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sae = load_sae(SAE_RELEASE, sae_id, device=device)
        from study.method_ids import label_tokens_from_config

        _lt_rival = label_tokens if label_tokens is not None else label_tokens_from_config(
            classes, {"target_tokens": {"M": "he", "F": "she"}}
        )
        _p(
            f"  SAE L{layer}: encoding rival CF fills for readout "
            f"({sorted(set(rival_eval_df['label_class'].astype(str)))}) …",
        )
        rival_test_by_class = _sae_encode_latents_by_class(
            model,
            tokenizer,
            sae,
            rival_eval_df,
            layer=int(layer),
            label_tokens=_lt_rival,
            split="test",
            act_policy=act_policy,
        )
        rival_val_by_class = _sae_encode_latents_by_class(
            model,
            tokenizer,
            sae,
            rival_eval_df,
            layer=int(layer),
            label_tokens=_lt_rival,
            split=selection_split,
            act_policy=act_policy,
        )

    for mode, sel in selections.items():
        if mode == "per_class":
            fbc = sel.features_by_class or {}
            bits = [f"{c}={fbc.get(c, [])[:3]}" for c in classes]
            _p(f"  SAE[per_class] L{layer}: " + "  ".join(bits))
        else:
            _p(f"  SAE[{mode}] L{layer}: {sel.feature_indices[:8]}")

    val_labels = (
        cached["val_labels"]
        if cached is not None and cached.get("val_labels")
        else val_df["label_class"].tolist()
    )
    readout_ks = sorted(
        {
            int(k)
            for k in (*SAE_READOUT_KS, *SAE_FIXED_KS)
            if int(k) <= int(SAE_TOP_K)
        }
    )

    _p(f"  SAE L{layer}: computing readouts / fair metrics …")
    specificity = {}
    sparse_curves = {}
    readouts = {}
    for mode, sel in selections.items():
        if mode in {"per_class_opp_fire", "per_class_arad_out", "per_class_jh_f1"}:
            continue  # handled after main loop
        _p(f"  SAE L{layer}: metrics for mode={mode} …")
        if mode == "per_class":
            fbc = sel.features_by_class or {}
            diag_feats = []
            for c in classes:
                diag_feats.extend((fbc.get(c) or [])[:1])
            if class_b is not None:
                specificity[mode] = sae_specificity(
                    latents,
                    test_labels,
                    diag_feats,
                    class_a=class_a,
                    class_b=class_b,
                    neutral_latents=neutral_latents,
                )
                k_star_joint = select_per_class_k_on_split(
                    val_latents,
                    val_labels,
                    fbc,
                    class_a=class_a,
                    class_b=class_b,
                    ks=readout_ks,
                )
                sparse_curves[mode] = per_class_fair_curve(
                    latents,
                    test_labels,
                    fbc,
                    class_a=class_a,
                    class_b=class_b,
                    ks=readout_ks,
                    neutral_latents=neutral_latents,
                    n_bootstrap=SAE_BOOTSTRAP_AUC,
                )
                rd_joint = sae_per_class_readout_metrics(
                    latents,
                    test_labels,
                    fbc,
                    k=k_star_joint,
                    class_a=class_a,
                    class_b=class_b,
                    neutral_latents=neutral_latents,
                )
                rd_joint["readout_k"] = k_star_joint
                rd_joint["k_selection"] = K_SELECTION
                readouts[mode] = rd_joint
            else:
                specificity[mode] = []
                sparse_curves[mode] = []
                readouts[mode] = {
                    "error": "one_pole_sae_no_joint_diff",
                    "k_selection": K_SELECTION,
                    "target_classes": classes,
                }

            labels_arr = np.asarray(test_labels)
            val_arr = np.asarray(val_labels)
            class_readouts: dict = {}
            class_curves: dict = {}
            if neutral_latents is None:
                for c in classes:
                    class_readouts[str(c)] = {"error": "no neutral latents"}
            else:
                for c in classes:
                    c = str(c)
                    feats = list(fbc.get(c) or [])
                    others, other_by, val_other_by, other_c = _sae_rival_readout_maps(
                        c,
                        classes,
                        test_labels,
                        latents,
                        rival_test_by_class=rival_test_by_class,
                        rival_val_by_class=rival_val_by_class,
                    )
                    lat_c = latents[labels_arr == c]
                    lat_o = other_by.get(other_c) if other_c else None
                    k_star = 1
                    val_auc = None
                    val_score = None
                    val_curve_rows = []
                    if feats and val_latents is not None:
                        val_c = val_latents[val_arr == c]
                        val_curve_rows = []
                        for k in readout_ks:
                            if k > len(feats):
                                continue
                            tmp = sae_class_vs_neutral_metrics(
                                val_c,
                                neutral_latents,
                                feats,
                                target_class=c,
                                k=k,
                                other_latents=(
                                    val_other_by.get(other_c)
                                    if other_c and val_latents is not None
                                    else None
                                ),
                                other_class=other_c,
                                other_latents_by_class={
                                    oc: val_other_by[oc]
                                    for oc in others
                                    if oc in val_other_by
                                }
                                or {
                                    oc: val_latents[val_arr == oc]
                                    for oc in others
                                    if val_latents is not None
                                },
                                n_bootstrap=0,
                            )
                            val_curve_rows.append({"k": int(k), **tmp})
                        k_star = select_k_from_curve_rows(val_curve_rows)
                        best_row = next(
                            (r for r in val_curve_rows if r.get("k") == k_star), {}
                        )
                        va = best_row.get("roc_auc")
                        val_auc = float(va) if isinstance(va, (int, float)) else None
                        keyed = k_selection_score(best_row)
                        val_score = None if keyed is None else float(keyed[0])
                    tau_by_k = {
                        int(r["k"]): r.get("youden_threshold")
                        for r in val_curve_rows
                        if r.get("youden_threshold") is not None
                    }
                    # Spec_n / Excl rules frozen on validation (per k).
                    frozen_by_k = {
                        int(r["k"]): frozen_decision_rules(r) for r in val_curve_rows
                    }
                    neu_test = test_neutral_latents if test_neutral_latents is not None else neutral_latents
                    curve = sae_class_vs_neutral_curve(
                        lat_c,
                        neu_test,
                        feats,
                        target_class=c,
                        ks=[k for k in readout_ks if k <= len(feats)],
                        other_latents=lat_o,
                        other_class=other_c,
                        n_bootstrap=SAE_BOOTSTRAP_AUC,
                        youden_thresholds=tau_by_k or None,
                        frozen_by_k=frozen_by_k,
                    )
                    class_curves[c] = curve
                    rd = sae_class_vs_neutral_metrics(
                        lat_c,
                        neu_test,
                        feats,
                        target_class=c,
                        k=k_star,
                        other_latents=lat_o,
                        other_class=other_c,
                        other_latents_by_class=other_by,
                        n_bootstrap=SAE_BOOTSTRAP_AUC,
                        youden_threshold=tau_by_k.get(int(k_star)),
                        frozen=frozen_by_k.get(int(k_star)),
                    )
                    rd["k_selection"] = K_SELECTION
                    rd["val_roc_auc"] = val_auc
                    rd["val_k_score"] = val_score
                    k1_val = next((r for r in val_curve_rows if r.get("k") == 1), None)
                    if k1_val:
                        rd["val_readout"] = dict(k1_val)
                    class_readouts[c] = rd
                # A class without a usable validation curve would silently get a
                # test-fit (oracle) Spec_n/Excl; that is an error, not a fallback.
                require_frozen_rules(class_readouts, context="SAE per-class readout (test)")
            readouts["per_class_by_class"] = class_readouts
            sparse_curves["per_class_by_class"] = class_curves
            # Keep looping for other modes below — do not fall through into
            # the bipolar-only branch that still expects class_b.
            continue
        elif mode == "joint":
            if class_b is None:
                readouts[mode] = {"error": "one_pole_sae_skip_joint"}
                sparse_curves[mode] = []
                specificity[mode] = []
                continue
            feat = int(sel.feature_indices[0]) if sel.feature_indices else None
            specificity[mode] = sae_specificity(
                latents,
                test_labels,
                [feat] if feat is not None else [],
                class_a=class_a,
                class_b=class_b,
                neutral_latents=neutral_latents,
            )
            if feat is None:
                readouts[mode] = {"error": "no joint feature"}
                sparse_curves[mode] = []
            else:
                rd = sae_joint_readout_metrics(
                    latents,
                    test_labels,
                    feat,
                    class_a=class_a,
                    class_b=class_b,
                    neutral_latents=neutral_latents,
                )
                boot = {}
                if "error" not in rd:
                    scores = joint_readout_scores(
                        latents, feat, test_labels,
                        class_a=class_a, class_b=class_b,
                    )
                    boot = bootstrap_roc_auc(
                        scores, test_labels,
                        class_a=class_a, class_b=class_b,
                        n_bootstrap=SAE_BOOTSTRAP_AUC,
                    )
                rd["roc_auc_boot_mean"] = boot.get("roc_auc_boot_mean")
                rd["roc_auc_boot_std"] = boot.get("roc_auc_boot_std")
                rd["k_selection"] = "joint_top1"
                rd["sae_layer"] = int(layer)
                readouts[mode] = rd
                sparse_curves[mode] = [
                    {
                        "k": 1,
                        "roc_auc": rd.get("roc_auc"),
                        "roc_auc_boot_std": rd.get("roc_auc_boot_std"),
                        "balanced_accuracy": rd.get("balanced_accuracy"),
                        "cohens_d": rd.get("cohens_d"),
                        "specificity": rd.get("specificity"),
                        "class_separation": rd.get("class_separation"),
                    }
                ]
        else:
            if class_b is None:
                readouts[mode] = {"error": "one_pole_sae_skip_bipolar_mode", "mode": mode}
                sparse_curves[mode] = []
                specificity[mode] = []
                continue
            specificity[mode] = sae_specificity(
                latents,
                test_labels,
                sel.feature_indices[:10],
                class_a=class_a,
                class_b=class_b,
                neutral_latents=neutral_latents,
            )
            k_star = select_readout_k_on_split(
                val_latents,
                val_labels,
                sel.feature_indices,
                class_a=class_a,
                class_b=class_b,
                ks=readout_ks,
                neutral_latents=None,
            )
            curve = sparse_fair_curve(
                latents,
                test_labels,
                sel.feature_indices,
                class_a=class_a,
                class_b=class_b,
                ks=readout_ks,
                neutral_latents=neutral_latents,
                n_bootstrap=SAE_BOOTSTRAP_AUC,
            )
            sparse_curves[mode] = curve
            rd = sae_sparse_readout_metrics(
                latents,
                test_labels,
                sel.feature_indices,
                k=k_star,
                class_a=class_a,
                class_b=class_b,
                neutral_latents=neutral_latents,
            )
            curve_row = next((c for c in curve if c.get("k") == k_star), {})
            rd["readout_k"] = k_star
            rd["roc_auc_boot_mean"] = curve_row.get("roc_auc_boot_mean")
            rd["roc_auc_boot_std"] = curve_row.get("roc_auc_boot_std")
            rd["k_selection"] = K_SELECTION
            rd["readout_kind"] = "bag_sum"
            rd["sae_layer"] = int(layer)
            readouts[mode] = rd

    # Exclusivity ablation: re-rank with P(z>0|opposite) penalty (cheap; same latents).
    if SAE_OPP_FIRE_ENABLED and val_latents is not None:
        _p(f"  SAE L{layer}: opp_fire exclusivity ranking …")
        opp_sel = select_per_class_features(
            val_latents,
            val_labels,
            target_classes=classes,
            top_k_per_class=SAE_TOP_K,
            min_firing_rate=0.01,
            neutral_latents=neutral_latents,
            neutral_specificity=SAE_USE_NEUTRAL_SPECIFICITY,
            score_mode="opp_fire_penalty",
        )
        selections["per_class_opp_fire"] = opp_sel
        labels_arr = np.asarray(test_labels)
        opp_by_class: dict = {}
        fbc_opp = opp_sel.features_by_class or {}
        for c in classes:
            c = str(c)
            feats_opp = list(fbc_opp.get(c) or [])
            others, other_by, val_other_by, other_c = _sae_rival_readout_maps(
                c,
                classes,
                test_labels,
                latents,
                rival_test_by_class=rival_test_by_class,
                rival_val_by_class=rival_val_by_class,
            )
            lat_c = latents[labels_arr == c]
            lat_o = other_by.get(other_c) if other_c else None
            if not feats_opp or neutral_latents is None:
                continue  # no opp_fire features — do not emit error shells
            rd_opp = sae_class_vs_neutral_metrics(
                lat_c,
                neutral_latents,
                feats_opp,
                target_class=c,
                k=1,
                other_latents=lat_o,
                other_class=other_c,
                other_latents_by_class=other_by,
                n_bootstrap=SAE_BOOTSTRAP_AUC,
            )
            rd_opp["readout_k"] = 1
            rd_opp["k_selection"] = "opp_fire_top1"
            rd_opp["sae_layer"] = int(layer)
            rd_opp["readout_features"] = feats_opp[:1]
            rd_opp["sae_candidate_features"] = feats_opp
            opp_by_class[c] = rd_opp
            _p(
                f"  SAE L{layer} opp_fire:{c} top1={feats_opp[:1]} "
                f"auc_n={rd_opp.get('roc_auc')} auc_o={rd_opp.get('roc_auc_other')}"
            )
        readouts["opp_fire_by_class"] = opp_by_class

    # Jørgensen & Hansen calibrated-F1 supervised selection (cheap; same val latents).
    if SAE_JH_F1_ENABLED and val_latents is not None:
        from jh_sae_select import select_per_class_jh_f1

        _p(f"  SAE L{layer}: JH calibrated-F1 supervised ranking …")
        jh_sel = select_per_class_jh_f1(
            val_latents,
            val_labels,
            target_classes=classes,
            top_k_per_class=SAE_TOP_K,
            min_firing_rate=0.01,
            pi0=SAE_JH_PI0,
            support_min=SAE_JH_SUPPORT_MIN,
        )
        jh_sel.layer = int(layer)
        jh_sel.sae_id = str(fingerprint.get("sae_id") or "")
        selections["per_class_jh_f1"] = jh_sel
        labels_arr = np.asarray(test_labels)
        jh_by_class: dict = {}
        fbc_jh = jh_sel.features_by_class or {}
        for c in classes:
            c = str(c)
            feats_jh = list(fbc_jh.get(c) or [])
            others, other_by, val_other_by, other_c = _sae_rival_readout_maps(
                c,
                classes,
                test_labels,
                latents,
                rival_test_by_class=rival_test_by_class,
                rival_val_by_class=rival_val_by_class,
            )
            lat_c = latents[labels_arr == c]
            lat_o = other_by.get(other_c) if other_c else None
            if not feats_jh or neutral_latents is None:
                jh_by_class[c] = {"error": "no jh_f1 features"}
                continue
            rd_jh = sae_class_vs_neutral_metrics(
                lat_c,
                neutral_latents,
                feats_jh,
                target_class=c,
                k=1,
                other_latents=lat_o,
                other_class=other_c,
                other_latents_by_class=other_by,
                n_bootstrap=SAE_BOOTSTRAP_AUC,
            )
            rd_jh["readout_k"] = 1
            rd_jh["k_selection"] = "jh_f1_top1"
            rd_jh["sae_layer"] = int(layer)
            rd_jh["readout_features"] = feats_jh[:1]
            rd_jh["sae_candidate_features"] = feats_jh
            jh_by_class[c] = rd_jh
            _p(
                f"  SAE L{layer} jh_f1:{c} top1={feats_jh[:1]} "
                f"auc_n={rd_jh.get('roc_auc')} auc_o={rd_jh.get('roc_auc_other')}"
            )
        readouts["jh_f1_by_class"] = jh_by_class

    # Arad et al. output-score ablation: re-rank per_class candidates (expensive forwards).
    arad_newly_scored = False
    if (
        SAE_ARAD_ENABLED
        and "per_class" in selections
        and "per_class_arad_out" not in selections
    ):
        arad_newly_scored = True
        if model is None or tokenizer is None:
            _p(f"  SAE L{layer}: loading base model for Arad scoring …")
            model_with_gradiend = trainer.get_model()
            model = model_with_gradiend.base_model
            tokenizer = trainer.tokenizer
        if sae is None:
            import torch

            _, sae_id = resid_sites(int(layer))
            _p(f"  SAE L{layer}: loading SAE for Arad scoring …")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sae = load_sae(SAE_RELEASE, sae_id, device=device)
        from arad_sae_select import rerank_per_class_by_arad_output

        base_pc = selections["per_class"]
        _p(
            f"  SAE L{layer}: Arad output-score re-rank "
            f"(candidate_k={SAE_ARAD_CANDIDATE_K}) …"
        )
        with cost_timer(
            f"sae_arad:L{layer}",
            phase="feature_select",
            backend="sae",
            ledger_unit="sae_layer_sweep",
            layer=int(layer),
        ):
            arad_sel = rerank_per_class_by_arad_output(
                model,
                tokenizer,
                sae,
                hf_module,
                base_pc,
                candidate_k=SAE_ARAD_CANDIDATE_K,
                target_classes=classes,
                progress=_p,
            )
        arad_sel.layer = int(layer)
        arad_sel.sae_id = str(fingerprint.get("sae_id") or "")
        selections["per_class_arad_out"] = arad_sel
        labels_arr = np.asarray(test_labels)
        arad_by_class: dict = {}
        fbc_arad = arad_sel.features_by_class or {}
        for c in classes:
            c = str(c)
            feats_arad = list(fbc_arad.get(c) or [])
            others, other_by, val_other_by, other_c = _sae_rival_readout_maps(
                c,
                classes,
                test_labels,
                latents,
                rival_test_by_class=rival_test_by_class,
                rival_val_by_class=rival_val_by_class,
            )
            lat_c = latents[labels_arr == c]
            lat_o = other_by.get(other_c) if other_c else None
            if not feats_arad or neutral_latents is None:
                arad_by_class[c] = {"error": "no arad_out features"}
                continue
            rd_arad = sae_class_vs_neutral_metrics(
                lat_c,
                neutral_latents,
                feats_arad,
                target_class=c,
                k=1,
                other_latents=lat_o,
                other_class=other_c,
                other_latents_by_class=other_by,
                n_bootstrap=SAE_BOOTSTRAP_AUC,
            )
            rd_arad["readout_k"] = 1
            rd_arad["k_selection"] = "arad_out_top1"
            rd_arad["sae_layer"] = int(layer)
            rd_arad["readout_features"] = feats_arad[:1]
            rd_arad["sae_candidate_features"] = feats_arad
            arad_by_class[c] = rd_arad
            _p(
                f"  SAE L{layer} arad_out:{c} top1={feats_arad[:1]} "
                f"auc_n={rd_arad.get('roc_auc')} auc_o={rd_arad.get('roc_auc_other')}"
            )
        readouts["arad_out_by_class"] = arad_by_class
    elif (
        SAE_ARAD_ENABLED
        and "per_class_arad_out" in selections
        and "arad_out_by_class" not in readouts
    ):
        # Cache hit with Arad selection: recompute encoder readouts only (cheap).
        arad_sel = selections["per_class_arad_out"]
        labels_arr = np.asarray(test_labels)
        arad_by_class = {}
        fbc_arad = getattr(arad_sel, "features_by_class", None) or (
            arad_sel.get("features_by_class") if isinstance(arad_sel, dict) else {}
        ) or {}
        for c in classes:
            c = str(c)
            feats_arad = list(fbc_arad.get(c) or [])
            others, other_by, val_other_by, other_c = _sae_rival_readout_maps(
                c,
                classes,
                test_labels,
                latents,
                rival_test_by_class=rival_test_by_class,
                rival_val_by_class=rival_val_by_class,
            )
            lat_c = latents[labels_arr == c]
            lat_o = other_by.get(other_c) if other_c else None
            if not feats_arad or neutral_latents is None:
                arad_by_class[c] = {"error": "no arad_out features"}
                continue
            rd_arad = sae_class_vs_neutral_metrics(
                lat_c,
                neutral_latents,
                feats_arad,
                target_class=c,
                k=1,
                other_latents=lat_o,
                other_class=other_c,
                other_latents_by_class=other_by,
                n_bootstrap=SAE_BOOTSTRAP_AUC,
            )
            rd_arad["readout_k"] = 1
            rd_arad["k_selection"] = "arad_out_top1"
            rd_arad["sae_layer"] = int(layer)
            rd_arad["readout_features"] = feats_arad[:1]
            rd_arad["sae_candidate_features"] = feats_arad
            arad_by_class[c] = rd_arad
        readouts["arad_out_by_class"] = arad_by_class
        _p(f"  SAE L{layer}: reused cached Arad selection; recomputed readouts")

    for mode, rd in readouts.items():
        if mode in {"per_class_by_class", "opp_fire_by_class", "arad_out_by_class", "jh_f1_by_class"}:
            for cls, crd in (rd or {}).items():
                if crd.get("error"):
                    print(f"  SAE L{layer}[{mode}/{cls}] warning: {crd['error']}")
                    continue
                if mode in {"opp_fire_by_class", "arad_out_by_class", "jh_f1_by_class"}:
                    continue  # already printed above
                std = crd.get("roc_auc_boot_std")
                std_s = f"±{std:.3f}" if isinstance(std, (int, float)) else ""
                print(
                    f"  sae L{layer}:{cls} vs neutral: k*={crd.get('readout_k')} "
                    f"val_auc={crd.get('val_roc_auc')} "
                    f"auc={crd.get('roc_auc')}{std_s}, "
                    f"bal_acc={crd.get('balanced_accuracy')}, "
                    f"d={crd.get('cohens_d')}, spec={crd.get('specificity')}"
                )
            continue
        if rd.get("error"):
            print(f"  SAE L{layer}[{mode}] readout warning: {rd['error']}")
            continue
        std = rd.get("roc_auc_boot_std")
        std_s = f"±{std:.3f}" if isinstance(std, (int, float)) else ""
        print(
            f"  SAE L{layer}[{mode}] k*={rd.get('readout_k')} "
            f"auc={rd.get('roc_auc')}{std_s}, "
            f"bal_acc={rd.get('balanced_accuracy')}, "
            f"d={rd.get('cohens_d')}, spec={rd.get('specificity')}"
        )

    # Write-through after a fresh encode (or Arad rescore), gated on use_cache
    # like the read side -- previously this wrote regardless of use_cache, so
    # a USE_CACHE=False caller (the module default) still paid the full disk
    # cost of every layer's sae_cache/ shard with zero read-side benefit.
    if use_cache and (cached is None or arad_newly_scored):
        try:
            cache_path = save_sae_cache(
                OUTPUT_DIR,
                fingerprint=fingerprint,
                val_latents=val_latents,
                test_latents=latents,
                neutral_latents=neutral_latents,
                test_neutral_latents=test_neutral_latents,
                val_labels=val_df["label_class"].tolist(),
                test_labels=test_labels,
                selections=selections,
                causal=None,
            )
            print(f"  Wrote SAE L{layer} cache to {cache_path}", flush=True)
        except Exception as _cache_exc:
            print(f"  SAE L{layer}: cache write failed: {_cache_exc}", flush=True)

    per_class_sel = (selections or {}).get("per_class")
    fbc = (per_class_sel.features_by_class if per_class_sel is not None else None) or {}
    neu_test = test_neutral_latents if test_neutral_latents is not None else neutral_latents
    all_k_compact = compact_all_k_layer_slices(
        test_latents=latents,
        val_latents=val_latents,
        neutral_test=neu_test,
        neutral_val=neutral_latents,
        test_labels=test_labels,
        val_labels=val_labels,
        features_by_class=fbc,
        classes=classes,
        max_k=max(SAE_FIXED_KS),
        rival_test_latents_by_class=rival_test_by_class,
        rival_val_latents_by_class=rival_val_by_class,
    )
    del rival_test_by_class, rival_val_by_class
    latents = val_latents = neutral_latents = test_neutral_latents = None
    if sae is not None:
        del sae
        sae = None

    print(f"  SAE L{layer}: done.", flush=True)
    try:
        from study.progress import write_pipeline_progress

        write_pipeline_progress(
            OUTPUT_DIR,
            stage="sae_layer",
            site=act_policy,
            layer=int(layer),
            cache_hit=bool(cached is not None),
        )
    except Exception:
        pass
    return {
        "layer": int(layer),
        "act_policy": act_policy,
        "sae_model": SAE_MODEL,
        "sae_release": SAE_RELEASE,
        "selections": {mode: selection_to_dict(sel) for mode, sel in selections.items()},
        "hf_module": hf_module,
        "sae_id": fingerprint["sae_id"],
        "specificity": specificity,
        "readouts": readouts,
        "sparse_combination_curves": sparse_curves,
        "cache_hit": cached is not None,
        "_model": model,
        "_tokenizer": tokenizer,
        "_all_k_compact": all_k_compact,
    }


def _top1_rank_score(sel, cls: str) -> float | None:
    """Top-1 feature ranking score for ``cls`` (same metric used to order features)."""
    if sel is None:
        return None
    if hasattr(sel, "features_by_class"):
        fbc = sel.features_by_class or {}
        sbc = sel.scores_by_class or {}
        scores = getattr(sel, "scores", None) or {}
    else:
        fbc = (sel or {}).get("features_by_class") or {}
        sbc = (sel or {}).get("scores_by_class") or {}
        scores = (sel or {}).get("scores") or {}
    feats = list(fbc.get(str(cls)) or [])
    if not feats:
        return None
    try:
        feat = int(feats[0])
    except (TypeError, ValueError):
        return None
    class_scores = sbc.get(str(cls)) or {}
    for key in (feat, str(feat)):
        if key in class_scores and isinstance(class_scores[key], (int, float)):
            return float(class_scores[key])
    for key in (feat, str(feat)):
        if key in scores and isinstance(scores[key], (int, float)):
            return float(scores[key])
    return None


def _layer_key(layer) -> object:
    if layer == "all" or str(layer) == "all":
        return "all"
    try:
        return int(layer)
    except (TypeError, ValueError):
        return layer


def _inject_all_layer_site(by_layer: dict, classes: Sequence[str]) -> None:
    """Add a virtual ``all`` site (all_k1) so layer* can select it via E."""

    payloads: dict = {}
    for layer, raw in by_layer.items():
        try:
            int(layer)
        except (TypeError, ValueError):
            continue
        compact = raw.get("_all_k_compact") or {}
        for cls, cdata in (compact.get("by_class") or {}).items():
            cols = cdata.get("test_cols")
            neu = cdata.get("neutral_test_cols")
            labels = cdata.get("test_labels")
            val_cols = cdata.get("val_cols")
            val_neu = cdata.get("neutral_val_cols")
            val_labels = cdata.get("val_labels")
            if cols is None or neu is None or not labels:
                continue
            payloads.setdefault(str(cls), []).append(
                {
                    "layer": int(layer),
                    "latent_cols": cols,
                    "neutral_cols": neu,
                    "labels": labels,
                    "val_cols": val_cols,
                    "val_neu": val_neu,
                    "val_labels": val_labels,
                    "feature_indices": cdata.get("feature_indices") or [],
                    "rival_cols_by_class": cdata.get("rival_test_cols_by_class") or {},
                    "rival_val_cols_by_class": cdata.get("rival_val_cols_by_class") or {},
                }
            )
    readouts_all: dict = {}
    for cls in classes:
        cls = str(cls)
        entries = payloads.get(cls) or []
        usable = [p for p in entries if p.get("latent_cols") is not None]
        if not usable:
            continue
        val_usable = [
            {
                "layer": p["layer"],
                "latent_cols": p["val_cols"],
                "neutral_cols": p["val_neu"],
                "labels": p["val_labels"],
                "feature_indices": p.get("feature_indices") or [],
                "rival_cols_by_class": p.get("rival_val_cols_by_class") or {},
            }
            for p in usable
            if p.get("val_cols") is not None and p.get("val_neu") is not None
        ]
        val_rd = {}
        tau = None
        if val_usable and all(p["latent_cols"] is not None for p in val_usable):
            val_rd = sae_all_layers_class_vs_neutral_metrics(
                val_usable,
                target_class=cls,
                k=1,
                target_classes=list(classes),
                n_bootstrap=0,
            )
            tau = val_rd.get("youden_threshold")
        rd = sae_all_layers_class_vs_neutral_metrics(
            usable,
            target_class=cls,
            k=1,
            target_classes=list(classes),
            n_bootstrap=0,
            youden_threshold=tau,
            frozen=frozen_decision_rules(val_rd),
        )
        rd["sae_layer"] = "all"
        rd["readout_k"] = 1
        rd["layer_selection"] = LAYER_SELECTION_PER_CLASS
        if val_rd and not val_rd.get("error"):
            rd["val_readout"] = dict(val_rd)
        readouts_all[cls] = rd
    require_frozen_rules(readouts_all, context="SAE all-layers readout (test)")
    if not readouts_all:
        return
    by_layer["all"] = {
        "selections": {
            "per_class": {
                "features_by_class": {c: ["all"] for c in readouts_all},
                "scores_by_class": {},
            }
        },
        "readouts": {"per_class_by_class": readouts_all},
    }


def _readout_layer_score(raw: dict, cls: str, *, readout_key: str = "per_class_by_class") -> float | None:
    """Site lock score: validation Detection, never the reported test split.

    Scoring a site on test would pick the winner on the split it is then
    published on. When no validation readout is available we fall through to the
    rank-based selection rather than silently scoring on test.
    """
    from suitability import detection_score

    rd = ((raw.get("readouts") or {}).get(readout_key) or {}).get(str(cls)) or {}
    val = rd.get("val_readout")
    if isinstance(val, dict):
        score = detection_score(val)
        if score is not None:
            return float(score)
    sel = (raw.get("selections") or {}).get("per_class") or {}
    return _top1_rank_score(sel, cls)


def _select_sae_layers(
    by_layer: dict,
    *,
    selection_mode: str = "per_class",
    classes: Optional[Sequence[str]] = None,
    readout_key: str = "per_class_by_class",
) -> tuple:
    """Pick best site per class: layers plus optional ``all``, locked by validation detection.

    Candidates are the keys of ``by_layer`` (integer layers and ``\"all\"``).
    The primary per-class score is validation detection; alternate SAE
    selection modes use their own explicit ranking scores.
    """
    if not by_layer:
        raise ValueError("by_layer is empty")
    class_list = [str(c) for c in (classes if classes is not None else TARGET_CLASSES)]
    keys = list(by_layer.keys())
    fallback = _layer_key(keys[0])
    selected_by_class = {}
    score_by_class = {}
    for cls in class_list:
        cls = str(cls)
        best_layer, best_score = None, float("-inf")
        for layer, raw in by_layer.items():
            if selection_mode != "per_class" and _layer_key(layer) == "all":
                continue
            if selection_mode == "per_class":
                score = _readout_layer_score(raw, cls, readout_key=readout_key)
            else:
                sel = (raw.get("selections") or {}).get(selection_mode) or {}
                score = _top1_rank_score(sel, cls)
            if score is not None and float(score) > best_score:
                best_score = float(score)
                best_layer = _layer_key(layer)
        selected_by_class[cls] = best_layer if best_layer is not None else fallback
        score_by_class[cls] = None if best_layer is None else best_score

    best_global, best_mean = None, float("-inf")
    for layer, raw in by_layer.items():
        if selection_mode != "per_class" and _layer_key(layer) == "all":
            continue
        scores = []
        for cls in class_list:
            if selection_mode == "per_class":
                score = _readout_layer_score(raw, str(cls), readout_key=readout_key)
            else:
                sel = (raw.get("selections") or {}).get(selection_mode) or {}
                score = _top1_rank_score(sel, str(cls))
            if score is not None:
                scores.append(float(score))
        mean_score = float(sum(scores) / len(scores)) if scores else float("-inf")
        if mean_score > best_mean:
            best_mean = mean_score
            best_global = _layer_key(layer)
    if best_global is None:
        best_global = fallback
    return selected_by_class, best_global, score_by_class


def _assemble_class_headline(
    by_layer: dict,
    selected_by_class: dict,
    *,
    readout_key: str,
    selection_mode: str,
    layer_selection: str,
    score_by_class: dict | None = None,
    classes: Optional[Sequence[str]] = None,
) -> tuple:
    """Pull per-class readouts / features / curves from rank-selected layers."""
    class_list = [str(c) for c in (classes if classes is not None else TARGET_CLASSES)]
    class_readouts = {}
    class_curves = {}
    fbc_headline = {}
    merged_sel = None
    for cls in class_list:
        cls = str(cls)
        layer = _layer_key(selected_by_class[cls])
        raw = by_layer[layer] if layer in by_layer else by_layer[selected_by_class[cls]]
        rd = ((raw.get("readouts") or {}).get(readout_key) or {}).get(cls) or {}
        rd = dict(rd)
        rd["sae_layer"] = layer
        rd["layer_selection"] = layer_selection
        if score_by_class is not None and cls in score_by_class:
            rd["layer_selection_score"] = score_by_class[cls]
        class_readouts[cls] = rd
        if readout_key == "per_class_by_class":
            class_curves[cls] = (
                (raw.get("sparse_combination_curves") or {}).get("per_class_by_class") or {}
            ).get(cls) or []
        sel = (raw.get("selections") or {}).get(selection_mode) or {}
        if merged_sel is None:
            merged_sel = dict(sel)
            merged_sel["features_by_class"] = dict(sel.get("features_by_class") or {})
            merged_sel["scores_by_class"] = dict(sel.get("scores_by_class") or {})
        fbc = sel.get("features_by_class") or {}
        fbc_headline[cls] = list(fbc.get(cls) or [])
        merged_sel["features_by_class"][cls] = fbc_headline[cls]
        sc = sel.get("scores_by_class") or {}
        if cls in sc:
            merged_sel.setdefault("scores_by_class", {})[cls] = sc[cls]
    return class_readouts, class_curves, fbc_headline, merged_sel


def run_sae(
    trainer,
    gender_df,
    *,
    rival_eval_df=None,
    rival_skip_sites=frozenset(),
    label_tokens=None,
) -> dict:
    """SAE feature ID over ``SAE_LAYERS`` × ``SAE_SELECT_SITES``.

    Filled ``prediction`` stays the ACTIEND/CAA-matched encode site (unsuffixed
    ids). Classical ``pre_prediction`` is the last-context generation residual
    (``:_pre`` method ids) — fairer for next-token causal eval.

    ``rival_skip_sites``: sites to run without ``rival_eval_df`` (e.g.
    ``pre_prediction`` on a one-pole CF-expanded task, where the rival rows
    share the same left context up to that site and rival scoring there would
    be tautological — see study/stages/sae.py). Skips the rival extraction
    itself, not just the resulting metric, since it's a real per-site,
    per-layer activation pass otherwise wasted on a guaranteed-uninformative
    comparison.
    """
    sites = normalize_sae_select_sites(SAE_SELECT_SITES)
    print(
        f"\n=== SAE select sites {list(sites)} layers {list(SAE_LAYERS)} "
        f"on {SAE_MODEL} (release={SAE_RELEASE}) ===",
        flush=True,
    )
    model = None
    tokenizer = None
    by_site: dict = {}
    for site in sites:
        print(f"\n=== SAE site={site} ===", flush=True)
        site_rival_df = None if site in rival_skip_sites else rival_eval_df
        if site in rival_skip_sites and rival_eval_df is not None:
            print(
                f"  rival scoring skipped for site={site} (tautological under "
                f"one-pole CF expansion; see study/stages/sae.py)",
                flush=True,
            )
        site_raw = _run_sae_one_site(
            trainer,
            gender_df,
            act_policy=site,
            model=model,
            tokenizer=tokenizer,
            rival_eval_df=site_rival_df,
            label_tokens=label_tokens,
        )
        model = site_raw.get("_model") or model
        tokenizer = site_raw.get("_tokenizer") or tokenizer
        by_site[str(site)] = site_raw
        release_sae_raw_heavy_memory(site_raw)
        try:
            from study.progress import write_pipeline_progress

            write_pipeline_progress(
                OUTPUT_DIR,
                stage="sae_site",
                site=str(site),
                sites_done=list(by_site),
            )
        except Exception:
            pass
    primary = "prediction" if "prediction" in by_site else next(iter(by_site))
    out = dict(by_site[primary])
    out["act_policy"] = primary
    out["sae_select_sites"] = list(by_site)
    out["by_site"] = by_site
    out["_model"] = model
    out["_tokenizer"] = tokenizer
    release_sae_raw_heavy_memory(out)
    return out


def _run_sae_one_site(
    trainer,
    gender_df,
    *,
    act_policy: str,
    model=None,
    tokenizer=None,
    rival_eval_df=None,
    label_tokens=None,
) -> dict:
    """SAE feature ID over ``SAE_LAYERS`` plus virtual ``all``; site lock = encoding E."""
    print(
        f"\n=== SAE layers {list(SAE_LAYERS)} site={act_policy} on {SAE_MODEL} "
        f"(release={SAE_RELEASE}) ===",
        flush=True,
    )
    by_layer = {}
    large_backbone = None
    for layer in SAE_LAYERS:
        layer_raw = run_sae_layer(
            trainer,
            gender_df,
            int(layer),
            model=model,
            tokenizer=tokenizer,
            rival_eval_df=rival_eval_df,
            label_tokens=label_tokens,
            act_policy=act_policy,
        )
        model = layer_raw.get("_model") or model
        tokenizer = layer_raw.get("_tokenizer") or tokenizer
        by_layer[int(layer)] = layer_raw
        # The live SAE for this layer is not used again.  Keep compact CPU
        # statistics for the eventual all-layer readout, but release its GPU
        # weights immediately.  Emptying the allocator cache is reserved for
        # large backbones so small-model sweeps do not pay that synchronization
        # cost on every layer.
        if large_backbone is None and model is not None:
            try:
                large_backbone = sum(p.numel() for p in model.parameters()) >= 1_000_000_000
            except Exception:
                large_backbone = False
        release_sae_layer_device_memory(
            layer_raw, empty_cuda_cache=bool(large_backbone)
        )

    classes = [str(c) for c in TARGET_CLASSES]
    if gender_df is not None and "label_class" in getattr(gender_df, "columns", []):
        present = {str(x) for x in gender_df["label_class"].astype(str)}
        if present:
            classes = [c for c in classes if c in present] or sorted(present)

    _inject_all_layer_site(by_layer, classes)
    selected_by_class, global_layer, rank_scores = _select_sae_layers(
        by_layer, selection_mode="per_class", classes=classes
    )
    print(
        f"  SAE layer selection ({LAYER_SELECTION_PER_CLASS}): "
        f"by_class={selected_by_class} scores={rank_scores} "
        f"global_ablations={global_layer}",
        flush=True,
    )

    class_readouts, class_curves, fbc_headline, per_class_sel = _assemble_class_headline(
        by_layer,
        selected_by_class,
        readout_key="per_class_by_class",
        selection_mode="per_class",
        layer_selection=LAYER_SELECTION_PER_CLASS,
        score_by_class=rank_scores,
        classes=classes,
    )

    # Opp-fire: same ranking metric also picks the layer (independent of per_class).
    selected_opp, global_opp, opp_rank_scores = ({}, None, {})
    opp_fire_readouts, fbc_opp_headline, opp_fire_sel = {}, {}, None
    if SAE_OPP_FIRE_ENABLED:
        selected_opp, global_opp, opp_rank_scores = _select_sae_layers(
            by_layer, selection_mode="per_class_opp_fire", classes=classes
        )
        print(
            f"  SAE layer selection ({LAYER_SELECTION_OPP_FIRE}): "
            f"by_class={selected_opp} scores={opp_rank_scores} "
            f"global=L{global_opp}",
            flush=True,
        )
        opp_fire_readouts, _opp_curves, fbc_opp_headline, opp_fire_sel = _assemble_class_headline(
            by_layer,
            selected_opp,
            readout_key="opp_fire_by_class",
            selection_mode="per_class_opp_fire",
            layer_selection=LAYER_SELECTION_OPP_FIRE,
            score_by_class=opp_rank_scores,
            classes=classes,
        )

    # Arad output-score: independent layer pick by top-1 Arad score.
    selected_arad, global_arad, arad_rank_scores = ({}, None, {})
    arad_out_readouts, fbc_arad_headline, arad_out_sel = {}, {}, None
    if SAE_ARAD_ENABLED:
        selected_arad, global_arad, arad_rank_scores = _select_sae_layers(
            by_layer, selection_mode="per_class_arad_out", classes=classes
        )
        print(
            f"  SAE layer selection ({LAYER_SELECTION_ARAD_OUT}): "
            f"by_class={selected_arad} scores={arad_rank_scores} "
            f"global=L{global_arad}",
            flush=True,
        )
        arad_out_readouts, _arad_curves, fbc_arad_headline, arad_out_sel = _assemble_class_headline(
            by_layer,
            selected_arad,
            readout_key="arad_out_by_class",
            selection_mode="per_class_arad_out",
            layer_selection=LAYER_SELECTION_ARAD_OUT,
            score_by_class=arad_rank_scores,
            classes=classes,
        )

    numeric_layers = [k for k in by_layer if k != "all"]
    global_src = global_layer if global_layer != "all" else (
        numeric_layers[-1] if numeric_layers else global_layer
    )
    global_raw = by_layer[global_src]
    selections = dict(global_raw.get("selections") or {})
    if per_class_sel is not None:
        selections["per_class"] = per_class_sel
    if opp_fire_sel is not None:
        selections["per_class_opp_fire"] = opp_fire_sel
    if arad_out_sel is not None:
        selections["per_class_arad_out"] = arad_out_sel
    readouts = dict(global_raw.get("readouts") or {})
    readouts["per_class_by_class"] = class_readouts
    if opp_fire_readouts:
        readouts["opp_fire_by_class"] = opp_fire_readouts
    if arad_out_readouts:
        readouts["arad_out_by_class"] = arad_out_readouts

    # JH calibrated-F1: independent layer pick by top-1 F1^c score.
    selected_jh, global_jh, jh_rank_scores = ({}, None, {})
    jh_f1_readouts, fbc_jh_headline, jh_f1_sel = {}, {}, None
    if SAE_JH_F1_ENABLED:
        selected_jh, global_jh, jh_rank_scores = _select_sae_layers(
            by_layer, selection_mode="per_class_jh_f1", classes=classes
        )
        print(
            f"  SAE layer selection ({LAYER_SELECTION_JH_F1}): "
            f"by_class={selected_jh} scores={jh_rank_scores} "
            f"global=L{global_jh}",
            flush=True,
        )
        jh_f1_readouts, _jh_curves, fbc_jh_headline, jh_f1_sel = _assemble_class_headline(
            by_layer,
            selected_jh,
            readout_key="jh_f1_by_class",
            selection_mode="per_class_jh_f1",
            layer_selection=LAYER_SELECTION_JH_F1,
            score_by_class=jh_rank_scores,
            classes=classes,
        )
    if jh_f1_sel is not None:
        selections["per_class_jh_f1"] = jh_f1_sel
    if jh_f1_readouts:
        readouts["jh_f1_by_class"] = jh_f1_readouts
    sparse = dict(global_raw.get("sparse_combination_curves") or {})
    sparse["per_class_by_class"] = class_curves

    # Per-layer class-vs-neutral maps for sae:M:L11 rows.
    by_layer_readouts: dict = {}
    for layer, raw in by_layer.items():
        try:
            layer_i = int(layer)
        except (TypeError, ValueError):
            continue
        by_layer_readouts[layer_i] = {}
        for cls in TARGET_CLASSES:
            cls = str(cls)
            rd = ((raw.get("readouts") or {}).get("per_class_by_class") or {}).get(cls) or {}
            rd = dict(rd)
            rd["sae_layer"] = layer_i
            sel = (raw.get("selections") or {}).get("per_class") or {}
            rd["layer_rank_score"] = _top1_rank_score(sel, cls)
            opp_sel = (raw.get("selections") or {}).get("per_class_opp_fire") or {}
            rd["opp_fire_rank_score"] = _top1_rank_score(opp_sel, cls)
            arad_sel = (raw.get("selections") or {}).get("per_class_arad_out") or {}
            rd["arad_out_rank_score"] = _top1_rank_score(arad_sel, cls)
            jh_sel = (raw.get("selections") or {}).get("per_class_jh_f1") or {}
            rd["jh_f1_rank_score"] = _top1_rank_score(jh_sel, cls)
            by_layer_readouts[layer_i][cls] = rd

    # Fixed-k ablations at the rank-selected layer for each class.
    fixed_k_readouts: dict = {}
    for cls in TARGET_CLASSES:
        cls = str(cls)
        layer = _layer_key(selected_by_class[cls])
        if layer == "all":
            continue
        raw = by_layer[layer]
        curve = (
            (raw.get("sparse_combination_curves") or {}).get("per_class_by_class") or {}
        ).get(cls) or []
        by_k_curve = {
            int(pt["k"]): pt
            for pt in curve
            if isinstance(pt.get("k"), int) and "error" not in pt
        }
        feats = list(
            (((raw.get("selections") or {}).get("per_class") or {}).get("features_by_class") or {}).get(cls)
            or []
        )
        fixed_k_readouts[cls] = {}
        ks_fixed = list(dict.fromkeys([1, *SAE_FIXED_KS]))  # always include k=1
        for k in ks_fixed:
            if k > len(feats) and k not in by_k_curve:
                continue
            if k in by_k_curve:
                rd = dict(by_k_curve[k])
            else:
                rd = {"error": f"no metrics for k={k}"}
            rd["sae_layer"] = layer
            rd["readout_k"] = int(k)
            rd["layer_selection"] = LAYER_SELECTION_PER_CLASS
            if feats:
                rd["readout_features"] = feats[: int(k)]
            fixed_k_readouts[cls][int(k)] = rd
        if 1 not in fixed_k_readouts[cls]:
            fixed_k_readouts[cls][1] = {
                "error": "no features for k=1",
                "sae_layer": layer,
                "readout_k": 1,
                "layer_selection": LAYER_SELECTION_PER_CLASS,
            }
    readouts["per_class_fixed_k"] = fixed_k_readouts

    def _all_k_payloads(
        compact_key: str, label_key: str, neu_key: str, rival_key: str
    ) -> dict:
        out: dict = {}
        for layer, raw in by_layer.items():
            try:
                layer_i = int(layer)
            except (TypeError, ValueError):
                continue
            compact = raw.get("_all_k_compact") or {}
            for cls, cdata in (compact.get("by_class") or {}).items():
                cols = cdata.get(compact_key)
                neu_cols = cdata.get(neu_key)
                labels = cdata.get(label_key)
                if cols is None or neu_cols is None or not labels:
                    continue
                out.setdefault(str(cls), []).append(
                    {
                        "layer": layer_i,
                        "latent_cols": cols,
                        "neutral_cols": neu_cols,
                        "labels": labels,
                        "feature_indices": cdata.get("feature_indices") or [],
                        "rival_cols_by_class": cdata.get(rival_key) or {},
                    }
                )
        return out

    test_by_class = _all_k_payloads(
        "test_cols", "test_labels", "neutral_test_cols", "rival_test_cols_by_class"
    )
    val_by_class = _all_k_payloads(
        "val_cols", "val_labels", "neutral_val_cols", "rival_val_cols_by_class"
    )

    # Multi-layer SAE: sum top-k class features from *every* layer (encode + causal).
    all_layer_fixed_k: dict = {}
    for cls in TARGET_CLASSES:
        cls = str(cls)
        all_layer_fixed_k[cls] = {}
        layer_payloads_base = test_by_class.get(cls) or []
        val_payloads_base = val_by_class.get(cls) or []
        ks_all = list(dict.fromkeys([1, *SAE_FIXED_KS]))
        for k in ks_all:
            if not layer_payloads_base:
                all_layer_fixed_k[cls][int(k)] = {"error": "no per-layer latents for all_k"}
                continue
            usable = [p for p in layer_payloads_base if int(p["latent_cols"].shape[1]) >= 1]
            if not usable:
                all_layer_fixed_k[cls][int(k)] = {"error": f"no features for all_k={k}"}
                continue
            tau = None
            frozen_all_k = {}
            if val_payloads_base:
                val_usable = [
                    p for p in val_payloads_base if int(p["latent_cols"].shape[1]) >= 1
                ]
                if val_usable:
                    val_rd = sae_all_layers_class_vs_neutral_metrics(
                        val_usable,
                        target_class=cls,
                        k=int(k),
                        target_classes=TARGET_CLASSES,
                        n_bootstrap=0,
                    )
                    tau = val_rd.get("youden_threshold")
                    frozen_all_k = frozen_decision_rules(val_rd)
            rd = sae_all_layers_class_vs_neutral_metrics(
                usable,
                target_class=cls,
                k=int(k),
                target_classes=TARGET_CLASSES,
                n_bootstrap=SAE_BOOTSTRAP_AUC,
                youden_threshold=tau,
                frozen=frozen_all_k,
            )
            rd["sae_layer"] = "all"
            rd["readout_k"] = int(k)
            rd["layer_selection"] = "all_layers_topk"
            require_frozen_rules({f"{cls}:all_k{int(k)}": rd}, context="SAE all_k readout (test)")
            if val_usable:
                rd["val_readout"] = dict(val_rd) if val_rd.get("error") is None else None
            all_layer_fixed_k[cls][int(k)] = rd
            if not rd.get("error"):
                print(
                    f"  sae:{cls}:all_k{k}: n_layers={rd.get('n_layers')} "
                    f"auc_n={rd.get('roc_auc')} auc_o={rd.get('roc_auc_other')}",
                    flush=True,
                )
    readouts["per_class_all_layer_fixed_k"] = all_layer_fixed_k

    # When E-lock picks ALL, headline k1/k* / fixed-k are the all-layers bags.
    for cls in TARGET_CLASSES:
        cls = str(cls)
        if _layer_key(selected_by_class.get(cls)) != "all":
            continue
        by_k = all_layer_fixed_k.get(cls) or {}
        dest = fixed_k_readouts.setdefault(cls, {})
        for k, rd in by_k.items():
            dest[int(k)] = rd
        best_k = None
        best_key = None
        for k, rd in by_k.items():
            scored = k_selection_score({**rd, "k": int(k)})
            if scored is None:
                continue
            if best_key is None or scored > best_key:
                best_key = scored
                best_k = int(k)
        if best_k is None and 1 in by_k:
            best_k = 1
        src = by_k.get(best_k) if best_k is not None else None
        if isinstance(src, dict) and not src.get("error"):
            rd = dict(src)
            rd["k_selection"] = K_SELECTION
            rd["sae_layer"] = "all"
            rd["layer_selection"] = LAYER_SELECTION_PER_CLASS
            if cls in rank_scores:
                rd["layer_selection_score"] = rank_scores[cls]
            class_readouts[cls] = rd

    layer_profile = {
        str(layer): {
            "layer": layer,
            "cache_hit": raw.get("cache_hit"),
            "hf_module": raw.get("hf_module"),
            "sae_id": raw.get("sae_id"),
            "per_class_rank_score": {
                str(c): _top1_rank_score(
                    (raw.get("selections") or {}).get("per_class") or {}, str(c)
                )
                for c in TARGET_CLASSES
            },
            "opp_fire_rank_score": {
                str(c): _top1_rank_score(
                    (raw.get("selections") or {}).get("per_class_opp_fire") or {}, str(c)
                )
                for c in TARGET_CLASSES
            },
            "arad_out_rank_score": {
                str(c): _top1_rank_score(
                    (raw.get("selections") or {}).get("per_class_arad_out") or {}, str(c)
                )
                for c in TARGET_CLASSES
            },
            "jh_f1_rank_score": {
                str(c): _top1_rank_score(
                    (raw.get("selections") or {}).get("per_class_jh_f1") or {}, str(c)
                )
                for c in TARGET_CLASSES
            },
            "per_class_val_auc": {
                str(c): (
                    ((raw.get("readouts") or {}).get("per_class_by_class") or {}).get(str(c)) or {}
                ).get("val_roc_auc")
                for c in TARGET_CLASSES
            },
            "per_class_test_auc": {
                str(c): (
                    ((raw.get("readouts") or {}).get("per_class_by_class") or {}).get(str(c)) or {}
                ).get("roc_auc")
                for c in TARGET_CLASSES
            },
        }
        for layer, raw in by_layer.items()
    }

    for raw in by_layer.values():
        release_sae_layer_heavy_memory(raw)

    return {
        "sae_model": SAE_MODEL,
        "sae_release": SAE_RELEASE,
        "act_policy": str(act_policy),
        "layers": list(SAE_LAYERS),
        "selected_layer_by_class": selected_by_class,
        "selected_layer_global": global_layer,
        "selected_layer_by_class_opp_fire": selected_opp,
        "selected_layer_global_opp_fire": global_opp,
        "selected_layer_by_class_arad_out": selected_arad,
        "selected_layer_global_arad_out": global_arad,
        "selected_layer_by_class_jh_f1": selected_jh,
        "selected_layer_global_jh_f1": global_jh,
        "layer_selection": LAYER_SELECTION_PER_CLASS,
        "layer_selection_opp_fire": LAYER_SELECTION_OPP_FIRE,
        "layer_selection_arad_out": LAYER_SELECTION_ARAD_OUT if SAE_ARAD_ENABLED else None,
        "layer_selection_jh_f1": LAYER_SELECTION_JH_F1 if SAE_JH_F1_ENABLED else None,
        "layer_selection_scores": rank_scores,
        "layer_selection_scores_opp_fire": opp_rank_scores,
        "layer_selection_scores_arad_out": arad_rank_scores,
        "layer_selection_scores_jh_f1": jh_rank_scores,
        "layer_profile": layer_profile,
        "by_layer_readouts": by_layer_readouts,
        "by_layer": {
            str(layer): {
                k: v
                for k, v in raw.items()
                if k
                not in (
                    "_model",
                    "_tokenizer",
                    "_sae",
                    "_all_k_compact",
                    "_test_latents",
                    "_neutral_latents",
                    "_val_neutral_latents",
                    "_val_latents",
                    "_test_labels",
                    "_val_labels",
                )
            }
            for layer, raw in by_layer.items()
        },
        "selections": selections,
        "hf_module": global_raw.get("hf_module"),
        "specificity": global_raw.get("specificity") or {},
        "readouts": readouts,
        "sparse_combination_curves": sparse,
        "cache_hit": all(
            bool(r.get("cache_hit"))
            for layer, r in by_layer.items()
            if _layer_key(layer) != "all"
        ),
        "_model": model,
        "_tokenizer": tokenizer,
        "_sae_by_layer": {},
        "_hf_module_by_layer": {
            int(layer): raw.get("hf_module")
            for layer, raw in by_layer.items()
            if _layer_key(layer) != "all"
        },
    }



















def _fair_metric_fields(readout: dict) -> dict:
    """Core fair-comparison fields shared by GRADIEND / ACTIEND / SAE."""
    return {
        "roc_auc": readout.get("roc_auc"),
        "roc_auc_neutral": readout.get("roc_auc_neutral", readout.get("roc_auc")),
        "roc_auc_other": readout.get("roc_auc_other", readout.get("min_pairwise_auroc")),
        "roc_auc_boot_std": readout.get("roc_auc_boot_std"),
        "balanced_accuracy": readout.get("balanced_accuracy"),
        "cohens_d": readout.get("cohens_d"),
        "specificity": readout.get("specificity"),
        "neutral_specificity": readout.get("neutral_specificity", readout.get("specificity")),
        "neutral_tnr": readout.get("neutral_tnr"),
        "class_exclusivity": readout.get("class_exclusivity"),
        "min_pairwise_auroc": readout.get("min_pairwise_auroc", readout.get("roc_auc_other")),
        "youden_threshold": readout.get("youden_threshold"),
        "youden_j": readout.get("youden_j"),
        "youden_negatives": readout.get("youden_negatives"),
        "target_tpr": readout.get("target_tpr"),
        "neutral_specificity_mag": readout.get("neutral_specificity_mag"),
        "class_exclusivity_mag": readout.get("class_exclusivity_mag"),
        "specificity_mid": readout.get("specificity_mid"),
        "neutral_spread": readout.get("neutral_spread"),
        "neutral_gap": readout.get("neutral_gap"),
        "neutral_abs_mean": readout.get("neutral_abs_mean"),
        "encoder_correlation": readout.get("encoder_correlation"),
        "class_separation": readout.get("class_separation"),
        "val_readout": readout.get("val_readout"),
        "eval_split": "test",
    }





def _sae_method_rows(sae_raw: dict) -> list:
    if sae_raw.get("error"):
        return [
            method_result(
                method="sae",
                model=SAE_MODEL,
                task="gender_en",
                status="error",
                error=str(sae_raw["error"]),
            )
        ]
    by_site = sae_raw.get("by_site")
    if isinstance(by_site, dict) and by_site:
        rows = []
        for site, payload in by_site.items():
            if not isinstance(payload, dict) or payload.get("error"):
                continue
            rows.extend(_sae_method_rows_for_site(payload, site=str(site)))
        if rows:
            return rows
    return _sae_method_rows_for_site(
        sae_raw, site=str(sae_raw.get("act_policy") or "prediction")
    )


def _sae_method_rows_for_site(sae_raw: dict, *, site: str) -> list:
    if sae_raw.get("error"):
        return [
            method_result(
                method="sae",
                model=SAE_MODEL,
                task="gender_en",
                status="error",
                error=str(sae_raw["error"]),
            )
        ]
    rows = []
    backend = sae_backend_for_site(site)

    def _part(p: str) -> str:
        return sae_part_for_site(p, site)
    selections = sae_raw.get("selections") or {}
    specificity = sae_raw.get("specificity") or {}
    readouts = sae_raw.get("readouts") or {}
    sparse = sae_raw.get("sparse_combination_curves") or {}
    by_layer = sae_raw.get("by_layer_readouts") or {}

    per_class_sel = selections.get("per_class") or {}
    class_readouts = readouts.get("per_class_by_class") or {}
    class_curves = sparse.get("per_class_by_class") or {}
    fbc = per_class_sel.get("features_by_class") or {}
    selected_layers = sae_raw.get("selected_layer_by_class") or {}

    # Headline sae:M / sae:F at val-selected layer + k*
    for cls in TARGET_CLASSES:
        cls = str(cls)
        rd = class_readouts.get(cls) or {}
        layer = rd.get("sae_layer", selected_layers.get(cls))
        rows.append(
            method_result(
                method=sae_encoder_kstar_id(cls, site=site),
                model=SAE_MODEL,
                task="gender_en",
                status="error" if rd.get("error") else "ok",
                error=rd.get("error"),
                metrics={
                    **_fair_metric_fields(rd),
                    "readout_k": rd.get("readout_k"),
                    "readout_features": rd.get("readout_features"),
                    "readout_kind": "class_vs_neutral",
                    "target_class": cls,
                    "sae_layer": layer,
                    "k_selection": rd.get("k_selection") or K_SELECTION,
                    "layer_selection": rd.get("layer_selection") or LAYER_SELECTION_PER_CLASS,
                    "layer_selection_score": rd.get("layer_selection_score"),
                    "component_part": "kstar",
                    "val_roc_auc": rd.get("val_roc_auc"),
                    "sae_candidate_features": fbc.get(cls),
                    "sae_top_features": rd.get("readout_features") or (fbc.get(cls) or [])[:1],
                    "mean_target": rd.get("mean_target"),
                    "mean_neutral": rd.get("mean_neutral"),
                    "mean_other": rd.get("mean_other"),
                    "n_neutral": rd.get("n_neutral"),
                    "causal_note": (
                        "shared encoder+causal ids: sae:{cls}:k1 / :kstar / :k{n} / "
                        ":sel_opp_fire / :sel_arad_out / :sel_jh_f1; "
                        "causal-only :k1_tok_prediction; sae:joint:{cls}"
                    ),
                },
                artifacts={
                    "hf_module": (sae_raw.get("_hf_module_by_layer") or {}).get(layer)
                    or sae_raw.get("hf_module"),
                    "sae_release": sae_raw.get("sae_release"),
                    "sae_layer": layer,
                },
                extras={
                    "selection": _json_ready(per_class_sel),
                    "readout_metrics": _json_ready(rd),
                    "sparse_combination_curve": _json_ready(class_curves.get(cls) or []),
                    "layer_profile": _json_ready(sae_raw.get("layer_profile") or {}),
                },
            )
        )

    # Per-layer rows: sae:M:L11 …
    for layer, layer_cls_map in by_layer.items():
        try:
            layer_i = int(layer)
        except (TypeError, ValueError):
            continue
        for cls in TARGET_CLASSES:
            cls = str(cls)
            rd = (layer_cls_map or {}).get(cls) or {}
            if not rd:
                continue
            rows.append(
                method_result(
                    method=method_part_id(backend, cls, _part(f"L{layer_i}")),
                    model=SAE_MODEL,
                    task="gender_en",
                    status="error" if rd.get("error") else "ok",
                    error=rd.get("error"),
                    metrics={
                        **_fair_metric_fields(rd),
                        "readout_k": rd.get("readout_k"),
                        "readout_features": rd.get("readout_features"),
                        "readout_kind": "class_vs_neutral",
                        "target_class": cls,
                        "sae_layer": layer_i,
                        "component_part": f"L{layer_i}",
                        "layer_rank_score": rd.get("layer_rank_score"),
                        "opp_fire_rank_score": rd.get("opp_fire_rank_score"),
                        "val_roc_auc": rd.get("val_roc_auc"),
                        "mean_target": rd.get("mean_target"),
                        "mean_neutral": rd.get("mean_neutral"),
                        "n_neutral": rd.get("n_neutral"),
                    },
                    artifacts={
                        "sae_release": sae_raw.get("sae_release"),
                        "sae_layer": layer_i,
                    },
                    extras={"readout_metrics": _json_ready(rd)},
                )
            )

    # Opp-fire exclusivity ablation: sae:M:sel_opp_fire …
    opp_readouts = readouts.get("opp_fire_by_class") or {}
    opp_sel = selections.get("per_class_opp_fire") or {}
    for cls in TARGET_CLASSES:
        cls = str(cls)
        rd = opp_readouts.get(cls) or {}
        feats = list(
            rd.get("readout_features")
            or rd.get("sae_candidate_features")
            or (opp_sel.get("features_by_class") or {}).get(cls)
            or []
        )
        # Empty / failed opp_fire is a no-op ablation, not a pipeline error.
        if rd.get("error") or not feats:
            continue
        rows.append(
            method_result(
                method=method_part_id(backend, cls, _part("sel_opp_fire")),
                model=SAE_MODEL,
                task="gender_en",
                status="error" if rd.get("error") else "ok",
                error=rd.get("error"),
                metrics={
                    **_fair_metric_fields(rd),
                    "readout_k": rd.get("readout_k", 1),
                    "readout_features": rd.get("readout_features"),
                    "readout_kind": "class_vs_neutral",
                    "target_class": cls,
                    "sae_layer": rd.get(
                        "sae_layer",
                        (sae_raw.get("selected_layer_by_class_opp_fire") or {}).get(cls)
                        or selected_layers.get(cls),
                    ),
                    "k_selection": "opp_fire_top1",
                    "layer_selection": rd.get("layer_selection") or LAYER_SELECTION_OPP_FIRE,
                    "layer_selection_score": rd.get("layer_selection_score"),
                    "component_part": "sel_opp_fire",
                    "sae_candidate_features": rd.get("sae_candidate_features")
                    or (opp_sel.get("features_by_class") or {}).get(cls),
                    "n_neutral": rd.get("n_neutral"),
                },
                artifacts={"sae_release": sae_raw.get("sae_release")},
                extras={
                    "selection": _json_ready(opp_sel),
                    "readout_metrics": _json_ready(rd),
                },
            )
        )

    # Arad output-score ablation: sae:M:sel_arad_out …
    arad_readouts = readouts.get("arad_out_by_class") or {}
    arad_sel = selections.get("per_class_arad_out") or {}
    if SAE_ARAD_ENABLED:
        for cls in TARGET_CLASSES:
            cls = str(cls)
            rd = arad_readouts.get(cls) or {}
            feats = list(
                rd.get("readout_features")
                or rd.get("sae_candidate_features")
                or (arad_sel.get("features_by_class") or {}).get(cls)
                or []
            )
            if rd.get("error") or not feats:
                continue
            rows.append(
                method_result(
                    method=method_part_id(backend, cls, _part("sel_arad_out")),
                    model=SAE_MODEL,
                    task="gender_en",
                    status="error" if rd.get("error") else "ok",
                    error=rd.get("error"),
                    metrics={
                        **_fair_metric_fields(rd),
                        "readout_k": rd.get("readout_k", 1),
                        "readout_features": rd.get("readout_features"),
                        "readout_kind": "class_vs_neutral",
                        "target_class": cls,
                        "sae_layer": rd.get(
                            "sae_layer",
                            (sae_raw.get("selected_layer_by_class_arad_out") or {}).get(cls)
                            or selected_layers.get(cls),
                        ),
                        "k_selection": "arad_out_top1",
                        "layer_selection": rd.get("layer_selection")
                        or LAYER_SELECTION_ARAD_OUT,
                        "layer_selection_score": rd.get("layer_selection_score"),
                        "component_part": "sel_arad_out",
                        "sae_candidate_features": rd.get("sae_candidate_features")
                        or (arad_sel.get("features_by_class") or {}).get(cls),
                        "n_neutral": rd.get("n_neutral"),
                        "sae_arad_candidate_k": SAE_ARAD_CANDIDATE_K,
                        "feature_selection_cite": (
                            "Arad et al. 2025; technion-cs-nlp/saes-are-good-for-steering"
                        ),
                    },
                    artifacts={"sae_release": sae_raw.get("sae_release")},
                    extras={
                        "selection": _json_ready(arad_sel),
                        "readout_metrics": _json_ready(rd),
                    },
                )
            )

    # JH calibrated-F1 supervised ablation: sae:M:sel_jh_f1 …
    jh_readouts = readouts.get("jh_f1_by_class") or {}
    jh_sel_enc = selections.get("per_class_jh_f1") or {}
    if SAE_JH_F1_ENABLED:
        for cls in TARGET_CLASSES:
            cls = str(cls)
            rd = jh_readouts.get(cls) or {}
            feats = list(
                rd.get("readout_features")
                or rd.get("sae_candidate_features")
                or (jh_sel_enc.get("features_by_class") or {}).get(cls)
                or []
            )
            if rd.get("error") or not feats:
                continue
            rows.append(
                method_result(
                    method=method_part_id(backend, cls, _part("sel_jh_f1")),
                    model=SAE_MODEL,
                    task="gender_en",
                    status="error" if rd.get("error") else "ok",
                    error=rd.get("error"),
                    metrics={
                        **_fair_metric_fields(rd),
                        "readout_k": rd.get("readout_k", 1),
                        "readout_features": rd.get("readout_features"),
                        "readout_kind": "class_vs_neutral",
                        "target_class": cls,
                        "sae_layer": rd.get(
                            "sae_layer",
                            (sae_raw.get("selected_layer_by_class_jh_f1") or {}).get(cls)
                            or selected_layers.get(cls),
                        ),
                        "k_selection": "jh_f1_top1",
                        "layer_selection": rd.get("layer_selection")
                        or LAYER_SELECTION_JH_F1,
                        "layer_selection_score": rd.get("layer_selection_score"),
                        "component_part": "sel_jh_f1",
                        "sae_candidate_features": rd.get("sae_candidate_features")
                        or (jh_sel_enc.get("features_by_class") or {}).get(cls),
                        "n_neutral": rd.get("n_neutral"),
                        "sae_jh_pi0": SAE_JH_PI0,
                        "sae_jh_support_min": SAE_JH_SUPPORT_MIN,
                        "feature_selection_cite": (
                            "Jørgensen & Hansen 2026; MikkelGodsk/SAE-labelling "
                            "(AxBench re-eval / supervised calibrated F1)"
                        ),
                    },
                    artifacts={"sae_release": sae_raw.get("sae_release")},
                    extras={
                        "selection": _json_ready(jh_sel_enc),
                        "readout_metrics": _json_ready(rd),
                    },
                )
            )

    # Fixed-k ablations at selected layer: sae:M:k1 …
    fixed_k_readouts = readouts.get("per_class_fixed_k") or {}
    for cls in TARGET_CLASSES:
        cls = str(cls)
        by_k = fixed_k_readouts.get(cls) or {}
        for k, rd in by_k.items():
            rows.append(
                method_result(
                    method=method_part_id(backend, cls, _part(f"k{int(k)}")),
                    model=SAE_MODEL,
                    task="gender_en",
                    status="error" if rd.get("error") else "ok",
                    error=rd.get("error"),
                    metrics={
                        **_fair_metric_fields(rd),
                        "readout_k": int(k),
                        "readout_features": rd.get("readout_features"),
                        "readout_kind": "class_vs_neutral",
                        "target_class": cls,
                        "sae_layer": rd.get("sae_layer", selected_layers.get(cls)),
                        "k_selection": "fixed",
                        "layer_selection": rd.get("layer_selection") or LAYER_SELECTION_PER_CLASS,
                        "component_part": f"k{int(k)}",
                        "n_neutral": rd.get("n_neutral"),
                    },
                    artifacts={"sae_release": sae_raw.get("sae_release")},
                    extras={"readout_metrics": _json_ready(rd)},
                )
            )

    # Multi-layer fixed-k: sae:M:all_k1 … (top-k @ every layer, summed)
    all_layer_fk = readouts.get("per_class_all_layer_fixed_k") or {}
    for cls in TARGET_CLASSES:
        cls = str(cls)
        by_k = all_layer_fk.get(cls) or {}
        for k, rd in by_k.items():
            rows.append(
                method_result(
                    method=sae_encoder_all_k_id(cls, int(k), site=site),
                    model=SAE_MODEL,
                    task="gender_en",
                    status="error" if rd.get("error") else "ok",
                    error=rd.get("error"),
                    metrics={
                        **_fair_metric_fields(rd),
                        "readout_k": int(k),
                        "readout_features": rd.get("readout_features"),
                        "features_by_layer": rd.get("features_by_layer"),
                        "readout_kind": "class_vs_neutral",
                        "target_class": cls,
                        "sae_layer": "all",
                        "k_selection": "fixed",
                        "layer_selection": "all_layers_topk",
                        "component_part": f"all_k{int(k)}",
                        "n_layers": rd.get("n_layers"),
                        "n_neutral": rd.get("n_neutral"),
                        "causal_note": (
                            "shared encoder+causal id: multi-layer residual add of "
                            "top-k decoder dirs at every layer"
                        ),
                    },
                    artifacts={"sae_release": sae_raw.get("sae_release")},
                    extras={"readout_metrics": _json_ready(rd)},
                )
            )

    for mode, sel in selections.items():
        if mode in {
            "per_class",
            "per_class_opp_fire",
            "per_class_arad_out",
            "per_class_jh_f1",
        }:
            if mode == "per_class_opp_fire":
                continue  # emitted as sae:{cls}:sel_opp_fire above
            if mode == "per_class_arad_out":
                continue  # emitted as sae:{cls}:sel_arad_out above
            if mode == "per_class_jh_f1":
                continue  # emitted as sae:{cls}:sel_jh_f1 above
            readout = readouts.get(mode) or {}
            rows.append(
                method_result(
                    method=f"{backend}:{_part('per_class_diff')}",
                    model=SAE_MODEL,
                    task="gender_en",
                    status="error" if readout.get("error") else "ok",
                    error=readout.get("error"),
                    metrics={
                        **_fair_metric_fields(readout),
                        "readout_k": readout.get("readout_k"),
                        "readout_kind": "per_class_diff",
                        "features_by_class": sel.get("features_by_class"),
                        "n_neutral": readout.get("n_neutral"),
                    },
                    artifacts={
                        "hf_module": sae_raw.get("hf_module"),
                        "sae_release": sae_raw.get("sae_release"),
                    },
                    extras={
                        "selection": _json_ready(sel),
                        "readout_metrics": _json_ready(readout),
                        "sparse_combination_curve": _json_ready(sparse.get(mode) or []),
                        "specificity": _json_ready(specificity.get(mode) or []),
                    },
                )
            )
            continue
        readout = readouts.get(mode) or {}
        rows.append(
            method_result(
                method=f"{backend}:{_part(mode)}",
                model=SAE_MODEL,
                task="gender_en",
                status="error" if readout.get("error") else "ok",
                error=readout.get("error"),
                metrics={
                    **_fair_metric_fields(readout),
                    "readout_k": readout.get("readout_k"),
                    "readout_kind": readout.get("readout_kind"),
                    "sae_top_features": sel.get("feature_indices"),
                    "n_neutral": readout.get("n_neutral"),
                },
                artifacts={
                    "hf_module": sae_raw.get("hf_module"),
                    "sae_release": sae_raw.get("sae_release"),
                },
                extras={
                    "selection": _json_ready(sel),
                    "readout_metrics": _json_ready(readout),
                    "specificity": _json_ready(specificity.get(mode) or []),
                    "sparse_combination_curve": _json_ready(sparse.get(mode) or []),
                },
            )
        )
    return rows
