"""Post-hoc ridge decoder variants for the E0d causal audit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np

from study.signals.bridge import (
    FrozenEncoderRidgeFit,
    fit_frozen_encoder_ridge_decoder,
)
from study.signals.learned import LearnedIendMeasurements
from study.signals.schema import IendSignalBatch, SourceKind


def _activation(values: np.ndarray, name: str) -> np.ndarray:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return np.tanh(values)
    if normalized in {"id", "identity", "linear"}:
        return values
    raise ValueError(
        "E0d supports only the learned tanh encoder and an explicit identity "
        f"bypass, got {name!r}"
    )


def fit_checkpoint_ridge_decoders(
    train: IendSignalBatch,
    validation: IendSignalBatch,
    test: IendSignalBatch,
    learned: LearnedIendMeasurements,
    *,
    source_kind: SourceKind,
    ridge_grid: Sequence[float],
    relative_ridge: bool = True,
) -> Dict[str, FrozenEncoderRidgeFit]:
    """Fit tanh-latent and raw-preactivation decoders for one checkpoint."""
    if source_kind == "diff":
        raise ValueError("diff is not a trainable E0d checkpoint source")
    if learned.checkpoint_source != source_kind:
        raise ValueError(
            f"Checkpoint source {learned.checkpoint_source!r} does not match "
            f"requested source {source_kind!r}"
        )
    views = {
        "train": train.compile(source_kind),
        "validation": validation.compile(source_kind),
        "test": test.compile(source_kind),
    }
    weight = np.asarray(learned.projected_encoder_weight, dtype=np.float64)
    scores = {
        split: view.source @ weight + float(learned.encoder_bias)
        for split, view in views.items()
    }
    latents = {
        split: _activation(score, learned.activation_encoder)
        for split, score in scores.items()
    }

    def _fit(values: Mapping[str, np.ndarray]) -> FrozenEncoderRidgeFit:
        return fit_frozen_encoder_ridge_decoder(
            values["train"],
            views["train"].target,
            values["validation"],
            views["validation"].target,
            values["test"],
            views["test"].target,
            ridge_grid=ridge_grid,
            relative_ridge=relative_ridge,
        )

    return {"latent": _fit(latents), "preactivation": _fit(scores)}




@dataclass(frozen=True)
class DecoderCausalVariant:
    """One temporary ACTIEND parameterization used by the causal evaluator."""

    variant_id: str
    encoder_activation: str
    decoder_weight: np.ndarray | None
    decoder_bias: np.ndarray | None
    decoder_bias_policy: str
    description: str

    def summary(self) -> Dict[str, Any]:
        return {
            "variant": self.variant_id,
            "encoder_activation": self.encoder_activation,
            "decoder_bias_policy": self.decoder_bias_policy,
            "decoder_weight_norm": (
                None
                if self.decoder_weight is None
                else float(np.linalg.norm(self.decoder_weight))
            ),
            "decoder_bias_norm": (
                None
                if self.decoder_bias is None
                else float(np.linalg.norm(self.decoder_bias))
            ),
            "description": self.description,
        }


def ridge_causal_variants(
    fits: Mapping[str, FrozenEncoderRidgeFit],
    *,
    trained_decoder_weight: Any,
    trained_decoder_bias: Any,
) -> tuple[DecoderCausalVariant, ...]:
    """Return the pre-registered decoder audit variants.

    The affine tanh variant is the exact theorem-matched reconstruction
    predictor and primary replacement for the currently affine decoder.  The
    centered variant isolates the source-dependent slope from the fitted mean
    target.  The identity-centered variant then isolates the inference-time
    tanh cost without changing that intercept policy.
    """
    latent = fits["latent"]
    preactivation = fits["preactivation"]
    trained_weight = np.asarray(trained_decoder_weight, dtype=np.float64).reshape(-1)
    trained_bias = np.asarray(trained_decoder_bias, dtype=np.float64).reshape(-1)
    if trained_weight.shape != latent.decoder_weight.shape:
        raise ValueError(
            "Trained and ridge decoder weights have different signal dimensions"
        )
    if trained_bias.shape != latent.decoder_bias.shape:
        raise ValueError(
            "Trained and ridge decoder biases have different signal dimensions"
        )
    return (
        DecoderCausalVariant(
            variant_id="trained_tanh",
            encoder_activation="checkpoint",
            decoder_weight=None,
            decoder_bias=None,
            decoder_bias_policy="checkpoint",
            description="Unmodified trained ACTIEND checkpoint.",
        ),
        DecoderCausalVariant(
            variant_id="trained_tanh_centered",
            encoder_activation="tanh",
            decoder_weight=trained_weight,
            decoder_bias=np.zeros_like(trained_bias),
            decoder_bias_policy="zero_trained_decoder_bias",
            description=(
                "Trained tanh encoder and decoder slope with the learned "
                "decoder bias removed."
            ),
        ),
        DecoderCausalVariant(
            variant_id="ridge_tanh_affine",
            encoder_activation="tanh",
            decoder_weight=latent.decoder_weight,
            decoder_bias=latent.decoder_bias,
            decoder_bias_policy="posthoc_reconstruction_intercept",
            description=(
                "Exact affine post-hoc reconstruction decoder, including the "
                "unregularized fitted target intercept."
            ),
        ),
        DecoderCausalVariant(
            variant_id="ridge_tanh_centered",
            encoder_activation="tanh",
            decoder_weight=latent.decoder_weight,
            decoder_bias=np.zeros_like(latent.decoder_bias),
            decoder_bias_policy="zero_centered_operator",
            description=(
                "Frozen learned tanh encoder with validation-selected ridge "
                "slope; intercept excluded to isolate source dependence."
            ),
        ),
        DecoderCausalVariant(
            variant_id="ridge_tanh_bias_only",
            encoder_activation="tanh",
            decoder_weight=np.zeros_like(latent.decoder_weight),
            decoder_bias=latent.decoder_bias,
            decoder_bias_policy="posthoc_intercept_only",
            description=(
                "Post-hoc ridge decoder intercept with zero slope; measures "
                "the mean-target baseline under the same encoder gate."
            ),
        ),
        DecoderCausalVariant(
            variant_id="ridge_identity_centered",
            encoder_activation="identity",
            decoder_weight=preactivation.decoder_weight,
            decoder_bias=np.zeros_like(preactivation.decoder_bias),
            decoder_bias_policy="zero_centered_operator",
            description=(
                "Inference-time identity bypass with preactivation ridge "
                "direction; this is not a separately trained identity IEND."
            ),
        ),
    )


@contextmanager
def apply_decoder_causal_variant(
    model_with_iend: Any,
    variant: DecoderCausalVariant,
) -> Iterator[None]:
    """Temporarily apply a decoder audit variant and restore the checkpoint."""
    if variant.decoder_weight is None:
        yield
        return

    import torch
    from torch import nn

    gradiend = getattr(model_with_iend, "gradiend", None)
    if gradiend is None:
        raise ValueError("Expected a ModelWithGradiend instance")
    if int(getattr(gradiend, "latent_dim", 0)) != 1:
        raise ValueError("E0d decoder variants require latent_dim=1")
    gradiend._require_built()
    decoder_linear = gradiend.decoder[0].linear
    if decoder_linear.bias is None and variant.decoder_bias is not None:
        if np.any(np.asarray(variant.decoder_bias) != 0.0):
            raise ValueError("Affine post-hoc decoder requires decoder bias parameters")
    wanted_weight = np.asarray(variant.decoder_weight, dtype=np.float64).reshape(-1)
    if wanted_weight.size != decoder_linear.weight.numel():
        raise ValueError(
            f"Decoder variant has {wanted_weight.size} values, expected "
            f"{decoder_linear.weight.numel()}"
        )
    wanted_bias = (
        None
        if variant.decoder_bias is None
        else np.asarray(variant.decoder_bias, dtype=np.float64).reshape(-1)
    )
    if decoder_linear.bias is not None and (
        wanted_bias is None or wanted_bias.size != decoder_linear.bias.numel()
    ):
        raise ValueError(
            "Decoder variant bias is missing or has a different signal dimension"
        )

    original_encoder_activation = gradiend.activation
    original_encoder_module = gradiend.encoder[1]
    original_decoder_activation = gradiend.activation_decoder
    original_decoder_module = gradiend.decoder[1]
    original_weight = decoder_linear.weight.detach().clone()
    original_bias = (
        decoder_linear.bias.detach().clone()
        if decoder_linear.bias is not None
        else None
    )
    try:
        with torch.no_grad():
            decoder_linear.weight.copy_(
                torch.as_tensor(
                    wanted_weight.reshape(decoder_linear.weight.shape),
                    device=decoder_linear.weight.device,
                    dtype=decoder_linear.weight.dtype,
                )
            )
            if decoder_linear.bias is not None and wanted_bias is not None:
                decoder_linear.bias.copy_(
                    torch.as_tensor(
                        wanted_bias,
                        device=decoder_linear.bias.device,
                        dtype=decoder_linear.bias.dtype,
                    )
                )
        # Use nn.Identity directly.  The package's historical get_activation
        # helper maps encoder ``id`` to LayerNorm(1), which annihilates a scalar
        # latent and is therefore not an identity-link experiment.
        if variant.encoder_activation == "identity":
            gradiend.activation = "id"
            gradiend.encoder[1] = nn.Identity()
        elif variant.encoder_activation == "tanh":
            gradiend.activation = "tanh"
            gradiend.encoder[1] = nn.Tanh()
        else:
            raise ValueError(
                f"Unsupported causal encoder activation {variant.encoder_activation!r}"
            )
        gradiend.activation_decoder = "id"
        gradiend.decoder[1] = nn.Identity()
        yield
    finally:
        with torch.no_grad():
            decoder_linear.weight.copy_(original_weight)
            if decoder_linear.bias is not None and original_bias is not None:
                decoder_linear.bias.copy_(original_bias)
        gradiend.activation = original_encoder_activation
        gradiend.encoder[1] = original_encoder_module
        gradiend.activation_decoder = original_decoder_activation
        gradiend.decoder[1] = original_decoder_module
