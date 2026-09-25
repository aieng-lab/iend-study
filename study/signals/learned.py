"""Measurements of an already-trained rank-one IEND on extracted signal pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from study.signals.moments import safe_cosine, saturation_summary
from study.signals.projection import CountSketchProjector
from study.signals.schema import SourceKind


_ACCUMULATION_KINDS = (
    "factual",
    "alternative",
    "diff",
    "both_factual",
    "both_alternative",
)


def _numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach()
        # numpy has no bfloat16 representation (hit live on a bf16-loaded
        # qwen3.5-9b-base checkpoint: "Got unsupported ScalarType BFloat16");
        # upcast first. Same fix as study/signals/package_adapter.py's _numpy.
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _scalar_activation(value: float, name: str) -> float:
    normalized = str(name).strip().lower()
    if normalized in {"id", "identity", "linear"}:
        return float(value)
    if normalized == "tanh":
        return float(np.tanh(value))
    if normalized == "sigmoid":
        return float(1.0 / (1.0 + np.exp(-value)))
    if normalized == "relu":
        return float(max(0.0, value))
    if normalized in {"smht", "hardtanh"}:
        return float(np.clip(value, -1.0, 1.0))
    raise ValueError(f"Unsupported scalar encoder activation {name!r}")


def _pearson(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


@dataclass
class LearnedIendAccumulator:
    encoder_weight: np.ndarray
    encoder_bias: float
    decoder_weight: np.ndarray
    decoder_bias: np.ndarray
    decoder_weight_norm_sq: float
    decoder_bias_norm_sq: float
    decoder_bias_weight_dot: float
    activation_encoder: str
    activation_decoder: str
    projected_encoder_weight: np.ndarray
    projected_decoder_weight: np.ndarray
    checkpoint_source: Optional[str]
    preactivation: Dict[str, List[float]]
    latent: Dict[str, List[float]]
    reconstruction_mse: Dict[str, List[float]]
    target_zero_mse: Dict[str, List[float]]
    decoder_bias_mse: Dict[str, List[float]]
    decoder_target_cosine_abs: Dict[str, List[float]]

    @classmethod
    def from_model(
        cls,
        gradiend: Any,
        *,
        projector: Optional[CountSketchProjector],
        checkpoint_source: Optional[str] = None,
    ) -> "LearnedIendAccumulator":
        if int(getattr(gradiend, "latent_dim", 0)) != 1:
            raise ValueError("E0 learned diagnostics currently require latent_dim=1")
        gradiend._require_built()
        encoder = gradiend.encoder[0].linear
        decoder = gradiend.decoder[0].linear
        encoder_weight = _numpy(encoder.weight).reshape(-1)
        decoder_weight = _numpy(decoder.weight).reshape(-1)
        encoder_bias_values = _numpy(encoder.bias).reshape(-1)
        decoder_bias_values = _numpy(decoder.bias).reshape(-1)
        encoder_bias = float(encoder_bias_values[0]) if len(encoder_bias_values) else 0.0
        decoder_bias = (
            decoder_bias_values
            if len(decoder_bias_values)
            else np.zeros_like(decoder_weight)
        )
        if len(encoder_weight) != len(decoder_weight):
            raise ValueError("learned encoder and decoder signal dimensions differ")
        if projector is not None:
            projected_encoder = projector.project_vector(encoder_weight)
            projected_decoder = projector.project_vector(decoder_weight)
        else:
            projected_encoder = encoder_weight.copy()
            projected_decoder = decoder_weight.copy()
        kwargs = getattr(gradiend, "kwargs", {}) or {}
        if checkpoint_source is None and kwargs.get("source") is not None:
            checkpoint_source = str(kwargs["source"])
        return cls(
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            decoder_weight_norm_sq=float(np.dot(decoder_weight, decoder_weight)),
            decoder_bias_norm_sq=float(np.dot(decoder_bias, decoder_bias)),
            decoder_bias_weight_dot=float(np.dot(decoder_bias, decoder_weight)),
            activation_encoder=str(getattr(gradiend, "activation", "tanh")),
            activation_decoder=str(getattr(gradiend, "activation_decoder", "id")),
            projected_encoder_weight=projected_encoder,
            projected_decoder_weight=projected_decoder,
            checkpoint_source=checkpoint_source,
            preactivation={kind: [] for kind in _ACCUMULATION_KINDS},
            latent={kind: [] for kind in _ACCUMULATION_KINDS},
            reconstruction_mse={kind: [] for kind in _ACCUMULATION_KINDS},
            target_zero_mse={kind: [] for kind in _ACCUMULATION_KINDS},
            decoder_bias_mse={kind: [] for kind in _ACCUMULATION_KINDS},
            decoder_target_cosine_abs={kind: [] for kind in _ACCUMULATION_KINDS},
        )

    def _mse(self, latent: float, target: np.ndarray) -> float:
        if str(self.activation_decoder).strip().lower() not in {"id", "identity", "linear"}:
            prediction = self.decoder_bias + self.decoder_weight * latent
            if str(self.activation_decoder).strip().lower() == "tanh":
                prediction = np.tanh(prediction)
            else:
                raise ValueError(
                    f"Unsupported decoder activation {self.activation_decoder!r}"
                )
            return float(np.mean((prediction - target) ** 2))
        # Algebraic form avoids allocating another parameter-sized prediction.
        prediction_norm_sq = (
            float(np.dot(self.decoder_bias, self.decoder_bias))
            + 2.0 * latent * float(np.dot(self.decoder_bias, self.decoder_weight))
            + latent**2 * float(np.dot(self.decoder_weight, self.decoder_weight))
        )
        error_sq = (
            prediction_norm_sq
            - 2.0 * float(np.dot(self.decoder_bias, target))
            - 2.0 * latent * float(np.dot(self.decoder_weight, target))
            + float(np.dot(target, target))
        )
        return float(max(error_sq, 0.0) / len(target))

    def _observe(self, kind: str, source: np.ndarray, target: np.ndarray) -> None:
        preactivation = float(np.dot(self.encoder_weight, source) + self.encoder_bias)
        latent = _scalar_activation(preactivation, self.activation_encoder)
        self.preactivation[kind].append(preactivation)
        self.latent[kind].append(latent)
        decoder_is_linear = str(self.activation_decoder).strip().lower() in {
            "id",
            "identity",
            "linear",
        }
        target_norm_sq = float(np.dot(target, target))
        zero_mse = target_norm_sq / len(target)
        if decoder_is_linear:
            bias_target_dot = float(np.dot(self.decoder_bias, target))
            weight_target_dot = float(np.dot(self.decoder_weight, target))
            prediction_norm_sq = (
                self.decoder_bias_norm_sq
                + 2.0 * latent * self.decoder_bias_weight_dot
                + latent**2 * self.decoder_weight_norm_sq
            )
            error_sq = (
                prediction_norm_sq
                - 2.0 * bias_target_dot
                - 2.0 * latent * weight_target_dot
                + target_norm_sq
            )
            reconstruction_mse = float(max(error_sq, 0.0) / len(target))
            bias_error_sq = (
                self.decoder_bias_norm_sq - 2.0 * bias_target_dot + target_norm_sq
            )
            bias_mse = float(max(bias_error_sq, 0.0) / len(target))
            denominator = float(np.sqrt(self.decoder_weight_norm_sq * target_norm_sq))
            cosine = (
                abs(weight_target_dot / denominator)
                if denominator > 0.0
                else float("nan")
            )
        else:
            reconstruction_mse = self._mse(latent, target)
            bias_mse = self._mse(0.0, target)
            cosine = float("nan")
        self.reconstruction_mse[kind].append(reconstruction_mse)
        self.target_zero_mse[kind].append(zero_mse)
        self.decoder_bias_mse[kind].append(bias_mse)
        self.decoder_target_cosine_abs[kind].append(cosine)

    def observe_pair(self, factual: np.ndarray, alternative: np.ndarray) -> None:
        difference = factual - alternative
        self._observe("factual", factual, difference)
        self._observe("alternative", alternative, difference)
        self._observe("diff", difference, difference)
        self._observe("both_factual", factual, difference)
        self._observe("both_alternative", alternative, -difference)


@dataclass(frozen=True)
class LearnedIendMeasurements:
    projected_encoder_weight: np.ndarray
    projected_decoder_weight: np.ndarray
    encoder_bias: float
    activation_encoder: str
    activation_decoder: str
    checkpoint_source: Optional[str]
    preactivation: Dict[SourceKind, np.ndarray]
    latent: Dict[SourceKind, np.ndarray]
    reconstruction_mse: Dict[SourceKind, np.ndarray]
    target_zero_mse: Dict[SourceKind, np.ndarray]
    decoder_bias_mse: Dict[SourceKind, np.ndarray]
    decoder_target_cosine_abs: Dict[SourceKind, np.ndarray]

    @classmethod
    def from_accumulator(cls, value: LearnedIendAccumulator) -> "LearnedIendMeasurements":
        def ordered(rows: Dict[str, List[float]]) -> Dict[SourceKind, np.ndarray]:
            return {
                "factual": np.asarray(rows["factual"]),
                "alternative": np.asarray(rows["alternative"]),
                "diff": np.asarray(rows["diff"]),
                "both": np.concatenate(
                    [rows["both_factual"], rows["both_alternative"]]
                ),
            }

        return cls(
            projected_encoder_weight=value.projected_encoder_weight,
            projected_decoder_weight=value.projected_decoder_weight,
            encoder_bias=value.encoder_bias,
            activation_encoder=value.activation_encoder,
            activation_decoder=value.activation_decoder,
            checkpoint_source=value.checkpoint_source,
            preactivation=ordered(value.preactivation),
            latent=ordered(value.latent),
            reconstruction_mse=ordered(value.reconstruction_mse),
            target_zero_mse=ordered(value.target_zero_mse),
            decoder_bias_mse=ordered(value.decoder_bias_mse),
            decoder_target_cosine_abs=ordered(value.decoder_target_cosine_abs),
        )

    def summary(
        self,
        source_kind: SourceKind,
        *,
        spectral: Any = None,
        spectral_score: Any = None,
        orientation: Any = None,
    ) -> Dict[str, Any]:
        preactivation = self.preactivation[source_kind]
        mse = self.reconstruction_mse[source_kind]
        zero_mse = self.target_zero_mse[source_kind]
        bias_mse = self.decoder_bias_mse[source_kind]
        zero_mean = float(np.mean(zero_mse))
        bias_mean = float(np.mean(bias_mse))
        mse_mean = float(np.mean(mse))
        target_cosines = self.decoder_target_cosine_abs[source_kind]
        finite_target_cosines = target_cosines[np.isfinite(target_cosines)]
        out: Dict[str, Any] = {
            "learned_checkpoint_source": self.checkpoint_source,
            "learned_activation_encoder": self.activation_encoder,
            "learned_activation_decoder": self.activation_decoder,
            "learned_encoder_bias": self.encoder_bias,
            "learned_reconstruction_mse_mean": mse_mean,
            "learned_reconstruction_mse_median": float(np.median(mse)),
            "learned_zero_prediction_mse_mean": zero_mean,
            "learned_decoder_bias_only_mse_mean": bias_mean,
            "learned_reconstruction_explained_vs_zero": (
                float(1.0 - mse_mean / zero_mean) if zero_mean > 0.0 else float("nan")
            ),
            "learned_reconstruction_gain_vs_decoder_bias": (
                float(1.0 - mse_mean / bias_mean) if bias_mean > 0.0 else float("nan")
            ),
            "learned_decoder_target_cosine_abs_mean": (
                float(np.mean(finite_target_cosines))
                if len(finite_target_cosines)
                else float("nan")
            ),
            **{f"learned_{key}": val for key, val in saturation_summary(preactivation).items()},
        }
        if spectral is not None:
            out["learned_spectral_encoder_sketch_cosine_abs"] = safe_cosine(
                self.projected_encoder_weight,
                spectral.encoder_direction,
                absolute=True,
            )
            out["learned_spectral_decoder_sketch_cosine_abs"] = safe_cosine(
                self.projected_decoder_weight,
                spectral.decoder_direction,
                absolute=True,
            )
        if spectral_score is not None:
            scores = np.asarray(spectral_score, dtype=np.float64).reshape(-1)
            if len(scores) != len(preactivation):
                raise ValueError(
                    f"spectral_score has length {len(scores)}, expected {len(preactivation)}"
                )
            out["learned_spectral_preactivation_correlation_abs"] = abs(
                _pearson(preactivation, scores)
            )
            out["learned_spectral_latent_correlation_abs"] = abs(
                _pearson(self.latent[source_kind], scores)
            )
        if orientation is not None:
            labels = np.asarray(orientation, dtype=np.float64).reshape(-1)
            if len(labels) != len(mse):
                raise ValueError(
                    f"orientation has length {len(labels)}, expected {len(mse)}"
                )
            for group, mask in (
                ("positive", labels > 0),
                ("negative", labels < 0),
                ("neutral", labels == 0),
            ):
                if np.any(mask):
                    out[f"learned_reconstruction_mse_{group}"] = float(np.mean(mse[mask]))
                    out[f"learned_latent_mean_{group}"] = float(
                        np.mean(self.latent[source_kind][mask])
                    )
        return out

    def subset(self, mask: Any) -> "LearnedIendMeasurements":
        """Select base paired rows, preserving the doubled order of `both`."""
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        base_n = len(self.preactivation["factual"])
        if len(selected) != base_n:
            raise ValueError(f"subset mask has length {len(selected)}, expected {base_n}")
        both_selected = np.concatenate([selected, selected])
        return LearnedIendMeasurements(
            projected_encoder_weight=self.projected_encoder_weight,
            projected_decoder_weight=self.projected_decoder_weight,
            encoder_bias=self.encoder_bias,
            activation_encoder=self.activation_encoder,
            activation_decoder=self.activation_decoder,
            checkpoint_source=self.checkpoint_source,
            preactivation={
                kind: values[both_selected if kind == "both" else selected]
                for kind, values in self.preactivation.items()
            },
            latent={
                kind: values[both_selected if kind == "both" else selected]
                for kind, values in self.latent.items()
            },
            reconstruction_mse={
                kind: values[both_selected if kind == "both" else selected]
                for kind, values in self.reconstruction_mse.items()
            },
            target_zero_mse={
                kind: values[both_selected if kind == "both" else selected]
                for kind, values in self.target_zero_mse.items()
            },
            decoder_bias_mse={
                kind: values[both_selected if kind == "both" else selected]
                for kind, values in self.decoder_bias_mse.items()
            },
            decoder_target_cosine_abs={
                kind: values[both_selected if kind == "both" else selected]
                for kind, values in self.decoder_target_cosine_abs.items()
            },
        )
