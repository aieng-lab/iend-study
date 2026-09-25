"""ACTIEND vs GRADIEND TrainingArguments profiles (study single source of truth).

Semantics match the gender PoC ``make_trainer`` backends; this module does not
import that script.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from study.config import StudyConfig, training_kwargs_from_config
from study.method_ids import cga_backend_ids

# Match PoC ``apply_smoke()``: ``SAE_LAYERS[:2]``. CAA/causal must use the same
# list — YAML has ``n_layers`` but no ``sae_layers``, so they used to sweep all 12.
SMOKE_LAYER_CAP = 2

Backend = Literal["gradiend", "actiend", "actiend_pre", "cga"]
SplitMode = Literal["none", "tensors"]

# Main-experiment default as of 2026-08-21 (``training.gradiend_exclude_embeddings``,
# ``configs/defaults.yaml`` — previously opt-in only via
# configs/suites/gradiend_no_embed.yaml): ``SignalScope.layers()`` — the same
# scope ACTIEND already uses (line ~185 below) — restricted to the residual
# transformer blocks only (``transformer.h.*`` for gpt2), excluding both the
# embedding matrices AND the final layer norm (``transformer.ln_f``). GRADIEND
# normally has no activation SignalScope at all (it's a gradient-over-weights
# signal — see CLAUDE.md's "GRADIEND and ACTIEND scope" note).
#
# This used to be impossible: SignalScope.layers() only set
# SignalScope.activation_selector, which was read exclusively by the
# activation-signal path (gradiend/signal_space.py::resolve_activation_modules)
# — GRADIEND's weight-parameter scope came from scope_params()/scope_mode() in
# that same module, which never looked at .activation_selector, so attaching
# SignalScope.layers() to a gradient signal silently no-op'd. Fixed in the
# package 2026-08-20/21: scope_params(scope, base_model=...) now resolves
# .activation_selector for a gradient signal too, via the same
# architecture-agnostic ModelTopology the activation path already used —
# gradient_params_from_selector() in gradiend/signal_space.py. This also
# means GRADIEND's no-embed scope is no longer limited to the hand-maintained
# {gpt2, gpt_neox} pattern dict this constant used to be (removed 2026-08-21)
# — any architecture ModelTopology supports (gpt2-like, gpt_neox, llama-like,
# bert-like, distilbert, opt) now works, and an unsupported one raises there
# (at model-construction time) with the same clear error the activation path
# already gave.
#
# Deliberately narrower than the prior "backbone minus embeddings" framing
# (which explicitly kept ln_f): the final layer norm isn't one of "the
# layers" by any reasonable reading — it's a single, non-repeated
# normalization serving the unembedding step, not part of the repeated block
# stack (GPT-2 itself defines ``self.h``/``self.ln_f`` as siblings, not
# ln_f as block N+1), and it's excluded from every one of ModelTopology's
# layers/embeddings/prediction_heads categories for exactly that reason.
# ``.layers()`` is the intuitively-correct reading of "exclude embeddings,
# keep just the layers"; if a future ablation specifically wants "whole
# backbone minus only the embedding lookup" (ln_f included) again, that's
# still expressible via an explicit
# ``SignalScope.from_values(params=[...])`` include-list.

# Filled-site ACTIEND vs mixed-site ACTIEND-PRE (source pre_prediction → target prediction).
# Mirror SAE / SAE-PRE: same CLI family ``actiend`` trains both; ids use distinct prefixes.
ACTIEND_BACKENDS: tuple[str, ...] = ("actiend", "actiend_pre")


def is_actiend_backend(backend: str) -> bool:
    return str(backend) in ACTIEND_BACKENDS


# CGA — Contrastive Gradient Addition (``cga_eval.py``): the untrained
# mean-difference estimator over GRADIEND's own gradient signal, i.e. the
# gradient-side analogue of CAA and the null for GRADIEND's learned
# encoder-decoder. It is not trained, so it shares GRADIEND's signal and
# parameter scope but none of its optimization machinery.
CGA_BACKENDS: tuple[str, ...] = cga_backend_ids()


def is_cga_backend(backend: str) -> bool:
    return str(backend) in CGA_BACKENDS


# Activation-gradient signal (``Signal.activation_gradient()``, dL/dh): the third
# signal axis alongside activation VALUE (CAA/ACTIEND) and weight GRADIENT
# (CGA/GRADIEND). It is a gradient (causally aligned by construction, like
# GRADIEND) but lives in activation space (local, steered like ACTIEND), so it
# disentangles "gradient-ness" from "weight-space" in GRADIEND's causal edge.
# ``caga`` is the untrained mean-difference estimator (Contrastive Activation-
# Gradient Addition, ``caga_eval.py``); ``agiend`` is the learned encoder-decoder
# on the same signal. Both steer ACTIVATIONS (never rewrite weights).
CAGA_BACKENDS: tuple[str, ...] = ("caga",)
AGIEND_BACKENDS: tuple[str, ...] = ("agiend",)


def is_caga_backend(backend: str) -> bool:
    return str(backend) in CAGA_BACKENDS


def is_agiend_backend(backend: str) -> bool:
    return str(backend) in AGIEND_BACKENDS


def is_activation_gradient_backend(backend: str) -> bool:
    """CAGA (mean) or AGIEND (learned) — both use ``Signal.activation_gradient``."""
    return is_caga_backend(backend) or is_agiend_backend(backend)


def record_decoder_lr(value: Any) -> Any:
    """Faithfully record a decoder LR into metadata: keep the ``"auto"`` sentinel
    (any str) as-is, ``float()`` a numeric value, pass ``None`` through. ``"auto"``
    is a legitimate value (the package's reachability lift); ``float("auto")`` would
    crash — which is exactly the systematic ablation failure documented in CLAUDE.md
    ("ALWAYS FAIL FAST"). Used at every metadata site that records the decoder LR."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return float(value)


def expand_cga_backends(enabled: Iterable[str]) -> set[str]:
    """Add each closed-form GRADIEND-family variant's backend id to the enabled set.

    ``enabled`` carries CLI/suite *family* names, but trainers are keyed by
    backend id (``cga`` / ``cga_tensor_norm``; ``caga``), so a filter that only
    knows the family name silently drops every variant's trainer. Both the study's
    causal stage and ``causal_study.run_study_causal`` filter on this set. Covers
    CGA (weight-space mean-diff, weight-rewrite causal) and CAGA (activation-gradient
    mean-diff, activation-steering causal) -- both fit a direction from a built
    checkpoint rather than training one.
    """
    out = {str(x) for x in enabled}
    if "cga" in out:
        out.update(CGA_BACKENDS)
    if "caga" in out:
        out.update(CAGA_BACKENDS)
    if "agiend" in out:
        out.update(AGIEND_BACKENDS)
    return out


def split_object(split_mode: SplitMode):
    from gradiend import GradiendSplit

    mode = str(split_mode).strip().lower()
    if mode in {"none", ""}:
        return GradiendSplit.none()
    if mode in {"tensors", "by_tensor", "tensor"}:
        return GradiendSplit.by_tensor()
    raise ValueError(f"Unknown gradiend_split mode {split_mode!r}")


def actiend_signal_scale_policy(split_mode: SplitMode) -> Dict[str, Any]:
    """Return the study's effective ACTIEND activation-scaling protocol.

    Component-split ACTIEND keeps the historical raw activation geometry.  In
    that mode every activation site already has its own encoder/decoder
    component, while first-batch ``running_rms`` changes the absolute margins
    used by per-component convergence.  The non-split path retains the current
    per-site RMS protocol pending a separate protocol decision.
    """
    mode = str(split_mode or "none").strip().lower()
    if mode in {"tensors", "by_tensor", "tensor"}:
        return {"scale": "raw"}
    return {
        "scale": "running_rms",
        "scale_reduce": "per_site",
        "scale_momentum": 0.0,
        "scale_eps": 1e-6,
    }


def shared_training_kwargs(
    cfg: StudyConfig,
    *,
    cli_lr: Optional[float] = None,
    cli_lr_decoder: Optional[float] = None,
    backend: Optional[Backend] = None,
) -> Dict[str, Any]:
    """Hyperparameters for training (profiles add signal/prune/split).

    Pass ``backend`` so ``learning_rate_gradiend`` / ``learning_rate_actiend``
    are applied; omit for the shared fallback LR only.

    ``learning_rate_decoder`` is only present in the returned dict when it is
    actually configured, so unconfigured runs keep their exact previous kwargs.
    """
    kw = training_kwargs_from_config(
        cfg, cli_lr=cli_lr, cli_lr_decoder=cli_lr_decoder, backend=backend
    )
    for drop in (
        "signal",
        "signal_scope",
    ):
        kw.pop(drop, None)
    kw.setdefault("prediction_objective", "clm_next_token")
    kw.setdefault("bias_encoder", True)
    kw.setdefault("fail_on_non_convergence", False)
    kw.setdefault("use_cache", False)
    # Study-only CGA policy. Keep it out of the generic config-to-package
    # forwarding path, but carry it into the CGA branch below.
    if is_cga_backend(str(backend or "")):
        kw["cga_pre_prune"] = bool((cfg.training or {}).get("cga_pre_prune", False))
    return kw


def build_training_arguments(
    backend: Backend,
    *,
    experiment_dir: str,
    shared: Dict[str, Any],
    split_mode: SplitMode = "none",
    metadata: Optional[Dict[str, Any]] = None,
    learning_rate: Optional[float] = None,
    one_pole: bool = False,
):
    """Build ``TrainingArguments`` for GRADIEND or ACTIEND.

    Checkpoint / convergence policy:
      - explicit ``selection_metric`` controls checkpoint/seed choice only
      - explicit ``convergent_metric`` in ``shared`` wins
      - else one-pole defaults to ``min_auc_n_o`` (``min(auc_n, auc_o)``)
      - else pair / default uses package ``correlation``
    """
    from gradiend import PostPruneConfig, PrePruneConfig, Signal, SignalScope, TrainingArguments

    args_kw = dict(shared)
    exclude_embeddings = bool(args_kw.pop("gradiend_exclude_embeddings", False))
    if learning_rate is not None:
        args_kw["learning_rate"] = float(learning_rate)
    args_kw["experiment_dir"] = str(experiment_dir)
    args_kw["gradiend_split"] = split_object(split_mode)
    # Selection metric: config override → one-pole min(auc_n,auc_o) → correlation.
    raw_metric = args_kw.get("convergent_metric")
    if raw_metric is None and (metadata or {}).get("convergent_metric") is not None:
        raw_metric = metadata.get("convergent_metric")  # type: ignore[union-attr]
    if raw_metric is None:
        args_kw["convergent_metric"] = "min_auc_n_o" if one_pole else "correlation"
    else:
        name = str(raw_metric).strip().lower()
        if name in {"auroc", "auc", "roc-auc"}:
            name = "roc_auc"
        if name in {"min_auc", "auc_min", "roc_auc_min", "min_auc_no", "min(auc_n,auc_o)"}:
            name = "min_auc_n_o"
        if name in {"corr"}:
            name = "correlation"
        args_kw["convergent_metric"] = name
    meta = dict(metadata or {})
    meta["gradiend_split"] = split_mode
    meta["backend"] = backend
    meta["learning_rate"] = float(args_kw.get("learning_rate"))
    if args_kw.get("learning_rate_decoder") is not None:
        # Only recorded when actually decoupled, so unconfigured runs keep
        # byte-identical metadata.
        meta["learning_rate_decoder"] = record_decoder_lr(args_kw["learning_rate_decoder"])
    meta["convergent_metric"] = args_kw["convergent_metric"]
    meta["selection_metric"] = (
        args_kw.get("selection_metric") or args_kw["convergent_metric"]
    )
    meta["one_pole"] = bool(one_pole)
    args_kw["metadata"] = meta

    if is_actiend_backend(backend):
        from activation_protocol import (
            actiend_token_selector,
            resolve_actiend_train_sites,
        )

        if backend == "actiend_pre":
            # Mixed-site: encoder H_{t-1}, decoder target = filled prediction diff.
            source_site, target_site = "pre_prediction", "prediction"
        else:
            source_site = (
                (metadata or {}).get("actiend_source_site")
                or (metadata or {}).get("activation_site")
                or shared.get("actiend_source_site")
                or shared.get("activation_site")
            )
            target_site = (
                (metadata or {}).get("actiend_target_site")
                or (metadata or {}).get("target_activation_site")
                or shared.get("actiend_target_site")
                or shared.get("target_activation_site")
            )
        src, tgt = resolve_actiend_train_sites(source_site, target_site)
        # Keep first-batch per-site running_rms for the concat/non-split path,
        # but preserve historical raw activations for by_tensor.  With one
        # component per layer, running_rms changed the absolute per-component
        # convergence margins and regressed the matched gpt2-small gender run
        # from 11/12 to 2/12 layers; the package PoC converged 12/12 on raw.
        scale_policy = actiend_signal_scale_policy(split_mode)
        scale_kwargs = {} if scale_policy["scale"] == "raw" else scale_policy
        args_kw["signal"] = Signal.activation(
            token_selector=actiend_token_selector(src),
            target_token_selector=actiend_token_selector(tgt),
            **scale_kwargs,
        )
        meta["activation_site"] = src
        meta["actiend_source_site"] = src
        meta["actiend_target_site"] = tgt
        meta["actiend_mixed_site"] = src != tgt
        meta["actiend_signal_scale"] = dict(scale_policy)
        args_kw.pop("pre_prune_config", None)
        args_kw.pop("post_prune_config", None)
        # Always restrict to residual layers (no wte). Concat none-split over
        # wte+all-layers collapses on high-LR pronoun tasks while per-layer
        # :tensors / :L* remain strong — keep site geometry matched.
        args_kw["signal_scope"] = SignalScope.layers()
    elif is_activation_gradient_backend(backend):
        # Activation-gradient signal (dL/dh): a gradient, but in activation space.
        # Encoder reads dL/dh at the prediction site; intervention steers
        # activations (CAA-style), never rewrites weights. ``caga`` is the
        # untrained mean-diff; ``agiend`` is the learned encoder-decoder.
        from activation_protocol import (
            actiend_token_selector,
            resolve_actiend_train_sites,
        )

        source_site = (
            (metadata or {}).get("actiend_source_site")
            or (metadata or {}).get("activation_site")
            or shared.get("actiend_source_site")
            or shared.get("activation_site")
        )
        # token_selector="prediction" for the activation_gradient signal reads
        # dL/dh at the GENERATING positions (the target span shifted one left),
        # handled inside ActivationSignalExtractor. In a causal LM dL/dh at the
        # target span itself is zero (the target at p is predicted from p-1), so
        # the extractor shifts the mask; this handles multi-token targets and,
        # by causal masking, equals the masked-prediction gradient even though the
        # activation item is filled.
        src, _tgt = resolve_actiend_train_sites(source_site, None)
        args_kw["signal"] = Signal.activation_gradient(token_selector="prediction")
        meta["activation_site"] = src
        meta["signal_kind"] = "activation_gradient"
        # Same activation-space scope discipline as ACTIEND (residual layers only).
        args_kw["signal_scope"] = SignalScope.layers()
        # Activation-space signals don't use GRADIEND's weight-space pruning (ACTIEND
        # drops it too). caga is closed-form; agiend is trained but still an
        # activation-space encoder-decoder, so neither prunes.
        args_kw.pop("pre_prune_config", None)
        args_kw.pop("post_prune_config", None)
    elif is_cga_backend(backend):
        # CGA is fitted in closed form (``cga_eval.fit_cga_direction``) and never
        # trained, so it inherits GRADIEND's gradient signal and parameter scope
        # and drops everything that exists only to serve optimization.
        #
        # By default, no pre/post pruning: pruning is part of GRADIEND's *training* pipeline
        # (a data-dependent top-k coordinate selection), not part of the
        # estimator being ablated. Keeping it would make CGA "mean-diff over the
        # coordinates GRADIEND's pruner happened to pick" instead of the plain
        # estimator, and would confound the comparison it exists to make.
        # A 27B full fp32 direction cannot coexist with the backbone.  When
        # explicitly configured, CGA uses the same *fixed pre-training*
        # coordinate projection as GRADIEND, then computes its closed-form
        # mean exactly in that projected space.  The mode is persisted, so it
        # cannot be confused with the historical full-space CGA baseline.
        # This is a study-only CGA policy, never a package TrainingArguments
        # field. ``shared_training_kwargs`` inserts it for CGA only, then this
        # branch consumes it before constructing the package arguments.
        cga_pre_prune = bool(args_kw.pop("cga_pre_prune", False))
        if cga_pre_prune:
            pre = args_kw.get("pre_prune_config")
            if pre is None:
                pre = PrePruneConfig(n_samples=16, topk=0.01, source="alternative")
                args_kw["pre_prune_config"] = pre
            else:
                pre.source = "alternative"
            args_kw["reuse_pre_prune"] = True
            meta["cga_coordinate_projection"] = "pre_prune"
            meta["cga_pre_prune_topk"] = float(pre.topk)
        else:
            args_kw["pre_prune_config"] = None
            meta["cga_coordinate_projection"] = "full"
        args_kw["post_prune_config"] = None
        args_kw.pop("signal", None)
        args_kw.pop("signal_scope", None)
        # One direction, applied as-is: ``decode(ff) = ff * delta``.
        # ``cga_eval.install_cga_direction`` writes ONLY the decoder --
        # ``rewrite_base_model``/the causal strength sweep never call
        # ``.encode()`` (confirmed against the package: ``evaluate_decoder``
        # sweeps ``feature_factor`` purely through the decoder/rewrite), so the
        # encoder half of this checkpoint is inert. ``activation_encoder``/
        # ``bias_encoder`` below are therefore irrelevant to anything CGA does.
        # CGA explicitly disables both biases so this closed-form direction has
        # no constant offset at zero feature strength. CGA's own
        # encoding-metrics readout (AUROC/exclusivity/specificity/correlation)
        # is computed directly as cosine(gradient, delta) in
        # ``cga_eval.fair_cga_encoder_eval``, mirroring how CAA scores its own
        # directions -- never through this encoder module. (This also sidesteps
        # a real package trap: ``activation_encoder="id"`` resolves to
        # ``nn.LayerNorm(1)`` for an encoder, which maps every scalar latent to
        # exactly 0 -- there is no unsquashed passthrough available through the
        # package's own activation factory, which is one more reason not to
        # route CGA's readout through it.)
        args_kw["latent_dim"] = 1
        args_kw["activation_encoder"] = "tanh"
        args_kw["activation_decoder"] = "id"
        args_kw["bias_encoder"] = False
        # A non-zero decoder bias would add a constant weight delta at
        # feature_factor=0, so the strength sweep would no longer pass through
        # the unmodified model.
        args_kw["bias_decoder"] = False
        meta["gradiend_exclude_embeddings"] = exclude_embeddings
        meta["cga_estimator"] = "paired_mean_diff"
        if exclude_embeddings:
            # Same scope as GRADIEND — the ablation only removes the learning.
            args_kw["signal_scope"] = SignalScope.layers()
            meta["gradiend_scope"] = "layers"
    else:
        # GRADIEND: gradient signal (default); always pre/post prune.
        pre = args_kw.get("pre_prune_config")
        if pre is None:
            n_samples = 16
            topk = 0.1
            # Allow callers to pass raw dict via shared under these keys
            n_samples = int(args_kw.pop("_pre_prune_n_samples", n_samples))
            topk = args_kw.pop("_pre_prune_topk", topk)
            args_kw["pre_prune_config"] = PrePruneConfig(
                n_samples=n_samples, topk=topk, source="alternative"
            )
        else:
            # Training may use source="both"; pre-pruning never does.
            pre.source = "alternative"
        post = args_kw.get("post_prune_config")
        if post is None:
            args_kw["post_prune_config"] = PostPruneConfig(
                topk=float(args_kw.pop("_post_prune_topk", 0.01)),
                part=str(args_kw.pop("_post_prune_part", "decoder-weight")),
            )
        args_kw.pop("signal", None)
        args_kw.pop("signal_scope", None)
        meta["gradiend_exclude_embeddings"] = exclude_embeddings
        if exclude_embeddings:
            # Resolved per-architecture at model-construction time via
            # ModelTopology (gradient_params_from_selector in the package);
            # an unsupported architecture raises there, not here.
            args_kw["signal_scope"] = SignalScope.layers()
            meta["gradiend_scope"] = "layers"

    # Strip helper keys (study-only; not TrainingArguments fields).
    for k in list(args_kw):
        if k.startswith("_"):
            args_kw.pop(k, None)
    # Consumed above for Signal.activation(token_selector=…); never pass through.
    args_kw.pop("activation_site", None)
    args_kw.pop("target_activation_site", None)
    args_kw.pop("actiend_source_site", None)
    args_kw.pop("actiend_target_site", None)
    # Model-level resource policy consumed only by the CGA branch above.  Keep
    # it out of every package TrainingArguments instance (ACTIEND/GRADIEND/etc.)
    # when the same model config enables multiple method families.
    args_kw.pop("cga_pre_prune", None)
    # Study-side encoder evaluation control, consumed by ``_train_once`` after
    # training. It is not a field of the upstream GRADIEND TrainingArguments.
    args_kw.pop("encoder_bootstrap_auc", None)

    return TrainingArguments(**args_kw)


_SCALE_TRAINING_CAPS: Dict[str, Dict[str, int]] = {
    "small": {
        "encoder_eval_max_size": 50,
        "encoder_eval_train_max_size": 16,
        "encoder_bootstrap_auc": 30,
    },
    "large": {
        "encoder_eval_max_size": 200,
        "encoder_eval_train_max_size": 50,
        "encoder_bootstrap_auc": 100,
    },
}




def resolve_study_layers(
    model_cfg: Mapping[str, Any],
    *,
    smoke: bool = False,
    fallback: Optional[Sequence[int]] = None,
) -> List[int]:
    """Layer indices for SAE / CAA / causal.

    Prefer ``sae_layers`` / ``layers``. If those are missing, use ``fallback``
    or ``range(n_layers)``. Smoke always returns at most ``SMOKE_LAYER_CAP``.
    """
    raw = model_cfg.get("sae_layers") or model_cfg.get("layers")
    if raw:
        layers = [int(x) for x in raw]
    elif fallback is not None:
        layers = [int(x) for x in fallback]
    elif model_cfg.get("n_layers") is not None:
        layers = list(range(int(model_cfg["n_layers"])))
    else:
        layers = []
    if smoke:
        if not layers:
            layers = list(range(SMOKE_LAYER_CAP))
        layers = layers[:SMOKE_LAYER_CAP]
    return layers


def apply_smoke_model_layers(cfg: StudyConfig) -> List[int]:
    """Write the smoke layer cap into ``cfg.raw['model']`` so every stage sees it.

    ``StudyConfig.model`` returns a copy; callers must mutate ``cfg.raw``.
    """
    raw = cfg.raw
    model = raw.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        raw["model"] = model
    capped = resolve_study_layers(model, smoke=True)
    model["sae_layers"] = capped
    model["layers"] = capped
    print(
        f"smoke: cap SAE/CAA/causal layers → {capped} "
        f"(n_layers={model.get('n_layers')})",
        flush=True,
    )
    return capped


def apply_smoke_training_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kw)
    out["max_steps"] = min(int(out.get("max_steps", 50)), 5)
    out["eval_steps"] = min(int(out.get("eval_steps", 10)), 5)
    out["encoder_eval_max_size"] = min(int(out.get("encoder_eval_max_size", 200)), 32)
    out["encoder_eval_train_max_size"] = min(int(out.get("encoder_eval_train_max_size", 100)), 16)
    # Aggressive prune for smoke so training is tractable on CPU.
    pre = out.get("pre_prune_config")
    if pre is not None and hasattr(pre, "topk"):
        try:
            pre.n_samples = min(int(getattr(pre, "n_samples", 16)), 4)
            pre.topk = min(float(pre.topk), 0.001)
        except Exception:
            pass
    return out


def apply_smoke_prune(args) -> None:
    pre = getattr(args, "pre_prune_config", None)
    if pre is not None and hasattr(pre, "n_samples"):
        try:
            # 4 was too tight for multi-class pair training: each pair (e.g.
            # race's asian-black/asian-white/black-white) only sees its own
            # slice of n_samples, and the val/test share of that slice rounded
            # to 0 rows — do_eval=True then refused a train-only split.
            # Confirmed 2026-08-18: every gradiend/actiend pair trainer
            # errored this way on race/ravel_country (3-class); binary
            # gender_en/pronoun_number (1 pair, full n_samples) were fine.
            # 20 is a heuristic bump, not a proven guarantee for every
            # class count/split ratio — verify against a real smoke re-run
            # rather than trusting this number blindly.
            pre.n_samples = min(int(pre.n_samples), 20)
        except Exception:
            pass
