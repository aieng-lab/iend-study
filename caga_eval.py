"""CAGA — Contrastive Activation-Gradient Addition (the untrained mean over dL/dh).

CAGA is the mean-difference estimator over the **activation-gradient** signal
``dL/dh`` -- the third signal axis alongside CAA (activation value) and CGA
(weight gradient).  Like both, it is a closed-form contrast (no training, no seed,
no convergence threshold); unlike CGA it lives in *activation* space, so its
intervention is activation steering (``h <- h + alpha*d``, CAA-style), never a
weight rewrite.  It is the null for AGIEND (the learned encoder-decoder on the
same signal) and the cell that disentangles "gradient-ness" from "weight-space"
in GRADIEND's causal edge: CAGA is a gradient, applied locally in activation
space.

The direction is the same supervised-decoder closed form CGA/CAA use,
``delta = sum_i label_i * target_i / sum_i label_i**2`` over the paired transition
``target_i = dL/dh(factual_i) - dL/dh(alternative_i)``, then unit-normalized like
``caa_eval.normalize_vec``.  Because ``dL/dh`` is activation-dimensional (~768),
the whole batch is materialized natively -- no CountSketch, unlike CGA's
weight-space stream.

This paired formula is the computational form of the semantic contrastive mean
``mu_positive - mu_negative``: ``label_i`` reverses rows whose factual target is
the negative pole, so every transition has the same semantic orientation.  It
equals CAA's direct class-pool calculation when the paired construction induces
the same observation weights.  In multiclass one-vs-rest settings, CAGA's
weights come from pair multiplicities, while CAA pools rival rows directly;
unbalanced inputs can therefore produce different numerical weights even
though the intended estimator is the same.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CAGA_LAYERWISE_ENCODER_EVAL_VERSION = 1


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Unit-normalize; a zero vector stays zero (mirrors ``caa_eval.normalize_vec``)."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        return np.zeros_like(v)
    return v / n


def caga_direction_from_signals(
    factual: np.ndarray, alternative: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Closed-form CAGA direction from paired activation-gradient signals.

    ``factual``/``alternative`` are ``(N, dim)`` per-example ``dL/dh`` signals;
    ``orientation`` is the per-example label in ``{+1, -1}`` (factual vs swapped
    pole).  Returns the unit-normalized ``sum(label*(F-A)) / sum(label**2)`` --
    for a one-pole cell every label is ``+1`` so this is just the normalized mean
    transition.  The label is essential: an unoriented mean of ``F-A`` would
    cancel when the factual class swaps across pairwise rows.  With balanced
    pairing, the signed expression equals the semantic-pole difference
    ``mu_positive - mu_negative`` computed directly by CAA.  A degenerate
    (zero-norm) direction returns zeros.
    """
    F = np.asarray(factual, dtype=np.float64)
    A = np.asarray(alternative, dtype=np.float64)
    lab = np.asarray(orientation, dtype=np.float64).reshape(-1)
    if F.shape != A.shape or F.ndim != 2:
        raise ValueError(f"factual/alternative must match and be 2-D: {F.shape} vs {A.shape}")
    if lab.shape[0] != F.shape[0]:
        raise ValueError(f"orientation length {lab.shape[0]} != n_rows {F.shape[0]}")
    denom = float((lab**2).sum())
    if denom <= 0.0:
        return np.zeros(F.shape[1], dtype=np.float64)
    target = F - A
    delta = (lab[:, None] * target).sum(axis=0) / denom
    return normalize_vec(delta)


def caga_causal_id(
    class_id: str,
    *,
    pair: Optional[str] = None,
    layer: Optional[int] = None,
) -> str:
    """CAGA causal/method id: ``caga:C`` (one-pole) or ``caga:A-B:C`` (two-pole).

    Mirrors the CGA id shape (``cga:C`` / ``cga:A-B:C``) so analysis grouping
    (``method_groups._pole_from_row``) buckets CAGA into two_pole/one_pole exactly
    like CGA/GRADIEND. Activation-steering, but the id carries no ``tok_``/site
    tail because a fixed mean direction has a single steering recipe.
    """
    cls = str(class_id)
    base = f"caga:{pair}:{cls}" if pair else f"caga:{cls}"
    return f"{base}:L{int(layer)}" if layer is not None else base


_LAYER_RE = re.compile(r"(?:^|\.)(?:h|layers|layer)\.(\d+)(?:\.|$)")


def _site_layer(module_name: str) -> int:
    match = _LAYER_RE.search(str(module_name))
    if not match:
        raise ValueError(f"cannot resolve residual layer from CAGA site {module_name!r}")
    return int(match.group(1))


def _resolve_signal_sites(model: Any) -> list:
    """Ordered ``[(module_path, width), ...]`` describing the direction's layout.

    The activation_gradient signal is flattened into GRADIEND input space by
    concatenating each residual site's entries in ``ParamMappedGradiendModel.
    param_map`` insertion order, each contributing ``prod(shape)`` entries — so
    this reconstructs exactly how to split the direction back per site and which
    residual module to steer with each slice. The param_map keys are
    ``activation:<module_path>`` (e.g. ``activation:transformer.h.0``); the
    ``activation:`` prefix is stripped to get the steering module. Tries the model
    directly and its inner ``.gradiend`` wrapper. Returns ``[]`` when no activation
    param_map is found (caller falls back to the legacy layer broadcast).
    """
    from math import prod

    prefix = "activation:"
    for obj in (getattr(model, "gradiend", None), model):
        param_map = getattr(obj, "param_map", None)
        if not isinstance(param_map, dict) or not param_map:
            continue
        sites = []
        for name, spec in param_map.items():
            if not str(name).startswith(prefix):
                return []  # a gradient (weight-space) param_map, not our signal
            shape = (spec or {}).get("shape")
            if not shape:
                return []
            sites.append((str(name)[len(prefix):], int(prod(tuple(shape)))))
        return sites
    return []


def caga_causal_specs(
    direction: np.ndarray,
    *,
    sites: Optional[Sequence[tuple]] = None,
    layers: Optional[Sequence[int]] = None,
    hf_resid_template: Optional[str] = None,
):
    """Build ``[(module_path, direction_tensor)]`` for activation steering.

    The ``activation_gradient`` signal is extracted over EVERY residual site in
    scope (``SignalScope.layers()``) and flattened by concatenation, so a gpt2
    ``dL/dh`` direction is 9216-dim = 12 layers x 768, NOT a single 768-dim site.
    Steering therefore splits the direction per site and installs each site's own
    slice at its residual module (``h <- h + a*d_L`` at layer L, faithful
    activation-gradient steering).

    Two modes:

    * ``sites`` = ordered ``[(module_path, width), ...]`` from the model's
      ``signal_space.mapping`` (the layout ``direction`` was flattened in). If
      ``len(direction)`` equals the summed widths, the direction is a per-layer
      concatenation and is split so each residual module gets its OWN slice. If it
      equals a single site's width (all widths equal), the same vector is
      broadcast to every site (a genuinely single-site signal).
    * ``layers`` + ``hf_resid_template`` (legacy fallback): broadcast the full
      ``direction`` to ``template.format(layer=L)`` for each L. Only correct when
      the signal is single-site.

    Import torch lazily so the numpy-only parts of this module stay
    GPU-free/importable.
    """
    import torch

    vec = np.asarray(direction, dtype=np.float64).reshape(-1)

    if sites is not None:
        names = [str(name) for name, _w in sites]
        widths = [int(w) for _name, w in sites]
        total = int(sum(widths))
        if vec.size == total:
            specs = []
            offset = 0
            for name, width in zip(names, widths):
                chunk = vec[offset : offset + width]
                offset += width
                specs.append((name, torch.tensor(chunk, dtype=torch.float32)))
            return specs
        if widths and len(set(widths)) == 1 and vec.size == widths[0]:
            tensor = torch.tensor(vec, dtype=torch.float32)
            return [(name, tensor.clone()) for name in names]
        raise ValueError(
            f"CAGA direction dim {vec.size} matches neither the concatenated site "
            f"total {total} nor a single site width "
            f"{widths[0] if widths else 'n/a'}; sites={names}"
        )

    if layers is None or hf_resid_template is None:
        raise ValueError(
            "caga_causal_specs requires either sites=[(module,width),...] or "
            "layers=[...] together with hf_resid_template=..."
        )
    tensor = torch.tensor(vec, dtype=torch.float32)
    return [
        (str(hf_resid_template).format(layer=int(L)), tensor.clone())
        for L in layers
    ]


@dataclass(frozen=True)
class CagaFit:
    """A fitted CAGA direction (activation space) plus provenance."""

    direction: np.ndarray  # unit-normalized, activation-dimensional
    n_used: int
    dim: int
    split: str




def fit_caga_direction(
    trainer: Any,
    *,
    split: str = "train",
    max_size_per_group: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> CagaFit:
    """Extract paired ``dL/dh`` signals from a CAGA trainer and fit the direction.

    The trainer must have been built on ``Signal.activation_gradient`` (backend
    ``caga``/``agiend``); ``extract_paired_signals_from_trainer`` then returns the
    per-example activation-gradient factual/alternative arrays.  No projection is
    needed -- the signal is activation-dimensional.
    """
    from study.signals.package_adapter import extract_paired_signals_from_trainer

    batch = extract_paired_signals_from_trainer(
        trainer,
        split=split,
        max_size_per_group=max_size_per_group,
        max_samples=max_samples,
        projection_dim=None,  # activation-dim signal; keep native
        include_learned=False,
    )
    factual = np.asarray(batch.factual)
    alternative = np.asarray(batch.alternative)
    direction = caga_direction_from_signals(factual, alternative, batch.orientation)
    return CagaFit(
        direction=direction,
        n_used=int(factual.shape[0]),
        dim=int(factual.shape[1]) if factual.ndim == 2 else 0,
        split=str(split),
    )


def caga_fit_and_specs(
    trainer: Any,
    *,
    cfg: Any = None,
    fit: Optional[CagaFit] = None,
    fit_split: str = "train",
):
    """Fit the CAGA direction (if not supplied) and build per-site steering specs.

    Used by the main pipeline's CAGA causal block; splits the concatenated ``dL/dh`` direction per
    residual site identically. Falls back to a single-site broadcast over
    ``resolve_caa_layers(cfg)`` only if no per-site layout is recoverable from the
    model's param_map (should not happen for a real caga trainer).
    """
    if fit is None:
        fit = fit_caga_direction(trainer, split=fit_split)
    sites = _resolve_signal_sites(trainer.get_model())
    if sites:
        return fit, caga_causal_specs(fit.direction, sites=sites)
    from study.stages.caa import resolve_caa_layers

    layers = resolve_caa_layers(cfg)
    template = ((cfg.model if cfg is not None else None) or {}).get(
        "hf_resid_template"
    ) or "transformer.h.{layer}"
    return fit, caga_causal_specs(
        fit.direction, layers=layers, hf_resid_template=str(template)
    )


def caga_direction_from_model(model: Any) -> np.ndarray:
    """Read the CAGA ``dL/dh`` direction persisted in the latent_dim=1 decoder.

    ``_fit_caga_once`` installs the fitted direction into the decoder via
    ``install_cga_direction`` (the same mechanism CGA uses to persist its
    direction in a checkpoint). The main-pipeline causal stage reads it back here
    -- so it consumes a direction fitted ONCE at train time, with no re-fit and no
    dependence on train data being available at causal time (mirroring how the
    weight-rewrite backends read their decoder). The standalone runner does not use
    this; it fits and steers in one pass.
    """
    gm = getattr(model, "gradiend", None)
    if gm is None:
        raise ValueError("model has no gradiend model to read a CAGA direction from")
    weight = gm.decoder[0].linear.weight.detach().reshape(-1)
    return np.asarray(weight.float().cpu().numpy(), dtype=np.float64)


def caga_specs_from_model(model: Any):
    """Per-site steering specs from a caga checkpoint's installed decoder direction.

    Pipeline counterpart to ``caga_fit_and_specs`` that does NOT re-fit: reads the
    persisted direction (``caga_direction_from_model``) and splits it per residual
    site (``_resolve_signal_sites`` + ``caga_causal_specs``).
    """
    direction = caga_direction_from_model(model)
    sites = _resolve_signal_sites(model)
    if not sites:
        raise ValueError(
            "no residual signal sites resolved from the caga model's param_map; "
            "cannot split the dL/dh direction for per-site steering"
        )
    return caga_causal_specs(direction, sites=sites)


def caga_layer_specs(model: Any) -> Dict[int, list]:
    """Single-site steering specs per layer, read only from the checkpoint."""
    grouped: Dict[int, list] = {}
    for module_name, direction in caga_specs_from_model(model):
        layer = _site_layer(module_name)
        if layer in grouped:
            raise ValueError(f"multiple CAGA signal sites resolved for layer {layer}")
        grouped[layer] = [(module_name, direction)]
    return dict(sorted(grouped.items()))


def caga_layer_slices(model: Any) -> Dict[int, Tuple[int, int]]:
    """Per-layer slices in the exact fitted/extracted CAGA signal layout."""
    sites = _resolve_signal_sites(model)
    if not sites:
        raise ValueError("no residual signal sites resolved from the CAGA param_map")
    offset = 0
    out: Dict[int, Tuple[int, int]] = {}
    for module_name, width in sites:
        layer = _site_layer(module_name)
        if layer in out:
            raise ValueError(f"multiple CAGA signal sites resolved for layer {layer}")
        out[layer] = (offset, offset + int(width))
        offset += int(width)
    return dict(sorted(out.items()))


def _caga_extractor(model_with_gradiend: Any):
    """Build the dL/dh extractor for per-row CAGA encoder scoring.

    Same signal/scope the caga fit used (``activation_gradient`` at the prediction
    span, residual layers), so a row's extracted dL/dh lives in the SAME concatenated
    space as the fitted direction and cosines are meaningful.
    """
    from gradiend.trainer.core.signals import ActivationSignalExtractor, Signal, SignalScope

    return ActivationSignalExtractor(
        model_with_gradiend,
        signal=Signal.activation_gradient(token_selector="prediction"),
        scope=SignalScope.layers(),
        tokenizer=model_with_gradiend.tokenizer,
    )


def _caga_dldh_for_item(extractor: Any, item: Mapping[str, Any]):
    """Run the extractor on one filled prediction item -> flat dL/dh (adds batch dim)."""
    import torch

    batched = {
        k: (v.unsqueeze(0) if torch.is_tensor(v) and v.dim() == 1 else v)
        for k, v in dict(item).items()
    }
    return extractor(factual_inputs=batched, requires_alternative=False).factual


def _caga_score_labeled(
    extractor: Any, direction: Any, masked_texts: Sequence[str], label_tokens: Sequence[str],
    *, tokenizer: Any, max_length: int, mask_placeholder: str,
    component_slices: Optional[Mapping[str, Tuple[int, int]]] = None,
):
    """dL/dh cosine for each (masked template, label token); drop rows that fail
    to build/extract (multi-token targets, truncation) -- same tolerant policy as
    the CGA labeled scorer. Returns ``(scores, kept_mask)``."""
    import numpy as np

    from cga_eval import cosine_to_direction
    from gradiend.trainer.text.prediction.dataset import _filled_prediction_from_template

    scores: List[float] = []
    component_scores = {str(part): [] for part in (component_slices or {})}
    kept = np.zeros(len(masked_texts), dtype=bool)
    for i, (tmpl, lab) in enumerate(zip(masked_texts, label_tokens)):
        try:
            item = _filled_prediction_from_template(
                tokenizer, template=str(tmpl), target=str(lab),
                max_length=max_length, mask_placeholder=mask_placeholder,
            )
            signal = _caga_dldh_for_item(extractor, item).reshape(-1)
            flat_direction = direction.reshape(-1)
            scores.append(cosine_to_direction(signal, flat_direction))
            for part, (lo, hi) in (component_slices or {}).items():
                component_scores[str(part)].append(
                    cosine_to_direction(
                        signal[int(lo):int(hi)], flat_direction[int(lo):int(hi)]
                    )
                )
            kept[i] = True
        except Exception:
            continue
    return (
        np.asarray(scores, dtype=np.float64),
        kept,
        {part: np.asarray(values, dtype=np.float64) for part, values in component_scores.items()},
    )


def _caga_score_neutral(
    extractor: Any, direction: Any, neu_texts: Sequence[str],
    *, tokenizer: Any, is_decoder_only: bool, excluded: Optional[Sequence[str]],
    max_length: int, mask_placeholder: str,
    component_slices: Optional[Mapping[str, Tuple[int, int]]] = None,
):
    """dL/dh cosine for auto-masked neutral texts (same masked-pair convention the
    package/CGA neutral scoring uses). Returns ``(scores, n_used)``."""
    import numpy as np

    from cga_eval import cosine_to_direction
    from gradiend.trainer.text.prediction.dataset import (
        _filled_prediction_from_template,
        create_masked_pair_from_text,
    )

    scores: List[float] = []
    component_scores = {str(part): [] for part in (component_slices or {})}
    for text in neu_texts:
        try:
            pair = create_masked_pair_from_text(
                str(text), tokenizer, is_decoder_only_model=is_decoder_only,
                excluded_tokens=list(excluded or []), mask_placeholder=mask_placeholder,
            )
            if not pair:
                continue
            masked_tmpl, target = pair
            item = _filled_prediction_from_template(
                tokenizer, template=masked_tmpl, target=target,
                max_length=max_length, mask_placeholder=mask_placeholder,
            )
            signal = _caga_dldh_for_item(extractor, item).reshape(-1)
            flat_direction = direction.reshape(-1)
            scores.append(cosine_to_direction(signal, flat_direction))
            for part, (lo, hi) in (component_slices or {}).items():
                component_scores[str(part)].append(
                    cosine_to_direction(
                        signal[int(lo):int(hi)], flat_direction[int(lo):int(hi)]
                    )
                )
        except Exception:
            continue
    return (
        np.asarray(scores, dtype=np.float64),
        len(scores),
        {part: np.asarray(values, dtype=np.float64) for part, values in component_scores.items()},
    )


def fair_caga_encoder_eval(
    model_with_gradiend: Any, direction: Any, eval_df: Any, neutral_df: Any,
    *, max_length: int = 128, mask_placeholder: str = "[MASK]",
    layer_slices: Optional[Mapping[int, Tuple[int, int]]] = None, **kwargs,
) -> Dict[str, Any]:
    """CAGA encoding metrics via CAA-style cosine of each row's dL/dh to the direction.

    Reuses ``cga_eval.fair_cga_encoder_eval``'s entire val-fit/test-report +
    ``class_vs_neutral_metrics`` machinery (so CAGA and CGA are metric-comparable),
    swapping ONLY the per-row scorer: instead of CGA's weight-gradient ``model.forward``
    cosine (wrong signal for caga, and crashes on frozen params), it scores each row's
    ``dL/dh`` via the ``ActivationSignalExtractor`` over items built WITH a
    ``prediction_mask`` (the same path the fit uses), then cosines to the caga
    direction. The direction lives in the concatenated per-site dL/dh space; the
    extractor produces a row's dL/dh in the same space.
    """
    import torch

    from cga_eval import fair_cga_encoder_eval
    from gradiend.model.utils import is_decoder_only_model as _is_decoder_only

    direction_t = direction if torch.is_tensor(direction) else torch.as_tensor(direction, dtype=torch.float32)
    tokenizer = model_with_gradiend.tokenizer
    is_decoder_only = _is_decoder_only(tokenizer)
    extractor = _caga_extractor(model_with_gradiend)
    excluded_words = kwargs.get("excluded_words")
    component_slices = {
        f"L{int(layer)}": (int(bounds[0]), int(bounds[1]))
        for layer, bounds in (layer_slices or {}).items()
    }

    def _labeled(masked, labels):
        return _caga_score_labeled(
            extractor, direction_t, masked, labels,
            tokenizer=tokenizer, max_length=max_length, mask_placeholder=mask_placeholder,
            component_slices=component_slices,
        )

    def _neutral(neu_texts):
        return _caga_score_neutral(
            extractor, direction_t, neu_texts,
            tokenizer=tokenizer, is_decoder_only=is_decoder_only, excluded=excluded_words,
            max_length=max_length, mask_placeholder=mask_placeholder,
            component_slices=component_slices,
        )

    return fair_cga_encoder_eval(
        model_with_gradiend, direction_t, eval_df, neutral_df,
        score_labeled_fn=_labeled, score_neutral_fn=_neutral,
        component_slices=component_slices, backend_name="caga", **kwargs,
    )
