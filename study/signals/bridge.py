"""Held-out bridge diagnostics between learned IEND and spectral rank-one ridge.

The ridge theorem characterizes a population linear rank-one operator.  These
diagnostics test the additional empirical claim needed by the paper: that a
trained, nonlinear, latent-dimension-one IEND learns the same mode.  Model
selection remains confined to train/validation; the learned--spectral bridge is
reported on the frozen test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from study.signals.evaluation import CrossSplitRecovery
from study.signals.learned import LearnedIendMeasurements
from study.signals.moments import safe_cosine
from study.signals.schema import IendSignalBatch, SourceKind


def _pearson(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def _binary_auc(scores: Any, positive: Any) -> float:
    """Dependency-free Mann--Whitney binary AUC with average ranks for ties."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(positive, dtype=bool).reshape(-1)
    if len(values) != len(labels) or len(values) < 2:
        return float("nan")
    n_positive = int(np.sum(labels))
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # One-indexed average rank for [start, end).
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(np.sum(ranks[labels]))
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _orientation_invariant_label_metrics(scores: Any, orientation: Any) -> Dict[str, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(orientation, dtype=np.float64).reshape(-1)
    if len(values) != len(labels):
        raise ValueError(f"score/label lengths differ: {len(values)} vs {len(labels)}")
    mask = labels != 0.0
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) != 2:
        return {"correlation_abs": float("nan"), "auc_orientation_invariant": float("nan")}
    correlation = abs(_pearson(values[mask], labels[mask]))
    auc = _binary_auc(values[mask], labels[mask] > 0.0)
    invariant_auc = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
    return {
        "correlation_abs": correlation,
        "auc_orientation_invariant": invariant_auc,
    }


def _class_mean_difference(target: Any, orientation: Any) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    labels = np.asarray(orientation, dtype=np.float64).reshape(-1)
    positive = values[labels > 0.0]
    negative = values[labels < 0.0]
    if not len(positive) or not len(negative):
        raise ValueError("A cross-fitted target direction requires both label poles")
    return positive.mean(axis=0) - negative.mean(axis=0)


def _apply_activation(value: Any, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    normalized = str(name).strip().lower()
    if normalized in {"id", "identity", "linear"}:
        return values
    if normalized == "tanh":
        return np.tanh(values)
    if normalized == "sigmoid":
        return 1.0 / (1.0 + np.exp(-values))
    if normalized == "relu":
        return np.maximum(values, 0.0)
    if normalized in {"smht", "hardtanh"}:
        return np.clip(values, -1.0, 1.0)
    raise ValueError(f"Unsupported encoder activation {name!r}")


@dataclass(frozen=True)
class FrozenEncoderRidgeFit:
    """Exact ridge decoder for a frozen scalar encoder score.

    ``decoder_bias`` is the affine intercept needed to reconstruct targets:
    ``T_hat = decoder_bias + score * decoder_weight``.  The full affine
    predictor is the exact MSE solution.  A zero-bias causal variant separately
    isolates the source-dependent slope, so experiments can distinguish an
    absolute predicted target from a feature-state contrast.
    """

    selected_ridge: float
    effective_ridge: float
    score_mean: float
    target_mean: np.ndarray
    decoder_weight: np.ndarray
    decoder_bias: np.ndarray
    validation_mse: float
    test_mse: float
    test_constant_train_mean_mse: float
    score_variance: float = float("nan")
    """Var(z) on the fit split -- the denominator of the closed form."""
    cross_covariance_norm: float = float("nan")
    """||Cov(T, z)|| on the fit split -- the numerator.

    Exposed because ``decoder = Cov(T, z) / (Var(z) + lambda_eff)`` exactly, so
    these two quantities fully determine ||d*||. Without them a change in the
    ridge target cannot be attributed to the latent's scale versus its coupling
    to the target -- which is what made the 100-step arm uninterpretable
    (IEND_THEORY_PLAN 2.7d).
    """

    @property
    def decoder_norm_closed_form(self) -> float:
        """||d*|| reconstructed from its two factors.

        Equals ``decoder_norm`` up to floating point; a mismatch means the fit
        did not use the closed form it is documented to use.
        """
        scale = float(self.score_variance) + float(self.effective_ridge)
        if not scale > 0.0:
            return float("nan")
        return float(self.cross_covariance_norm) / scale

    @property
    def test_explained_variance(self) -> float:
        baseline = float(self.test_constant_train_mean_mse)
        return (
            float(1.0 - self.test_mse / baseline)
            if baseline > 0.0
            else float("nan")
        )

    @property
    def decoder_norm(self) -> float:
        return float(np.linalg.norm(self.decoder_weight))

    @property
    def decoder_direction(self) -> np.ndarray:
        norm = self.decoder_norm
        return self.decoder_weight / norm if norm > 0.0 else self.decoder_weight

    def summary(self) -> Dict[str, Any]:
        return {
            "selected_ridge": float(self.selected_ridge),
            "effective_ridge": float(self.effective_ridge),
            "score_mean": float(self.score_mean),
            "target_mean_norm": float(np.linalg.norm(self.target_mean)),
            "decoder_bias_norm": float(np.linalg.norm(self.decoder_bias)),
            "validation_mse": float(self.validation_mse),
            "test_mse": float(self.test_mse),
            "test_constant_train_mean_mse": float(
                self.test_constant_train_mean_mse
            ),
            "test_explained_variance": float(self.test_explained_variance),
            "decoder_norm": float(self.decoder_norm),
        }


def fit_frozen_encoder_ridge_decoder(
    train_score: Any,
    train_target: Any,
    validation_score: Any,
    validation_target: Any,
    test_score: Any,
    test_target: Any,
    *,
    ridge_grid: Any,
    relative_ridge: bool = True,
) -> FrozenEncoderRidgeFit:
    """Fit ``T = intercept + score * decoder`` with scalar ridge.

    Ridge is selected only on validation MSE.  Test arrays are used solely for
    the final held-out reconstruction metrics returned with the fitted
    parameters.
    """
    train_z = np.asarray(train_score, dtype=np.float64).reshape(-1)
    validation_z = np.asarray(validation_score, dtype=np.float64).reshape(-1)
    test_z = np.asarray(test_score, dtype=np.float64).reshape(-1)
    train_t = np.asarray(train_target, dtype=np.float64)
    validation_t = np.asarray(validation_target, dtype=np.float64)
    test_t = np.asarray(test_target, dtype=np.float64)
    if len(train_z) != len(train_t) or len(validation_z) != len(validation_t) or len(test_z) != len(test_t):
        raise ValueError("Fixed-score decoder score/target row counts differ")
    candidates = tuple(float(value) for value in ridge_grid)
    if not candidates or any(value < 0.0 for value in candidates):
        raise ValueError("ridge_grid must contain non-negative values")

    score_mean = float(np.mean(train_z))
    target_mean = train_t.mean(axis=0)
    centered_z = train_z - score_mean
    centered_target = train_t - target_mean
    denominator = float(max(1, len(train_z) - 1))
    score_variance = float(np.dot(centered_z, centered_z) / denominator)
    cross_covariance = centered_target.T @ centered_z / denominator
    fits: Dict[float, tuple[np.ndarray, float, float]] = {}
    for ridge in candidates:
        effective_ridge = ridge * score_variance if relative_ridge else ridge
        scale = score_variance + effective_ridge
        decoder = cross_covariance / scale if scale > 0.0 else np.zeros_like(cross_covariance)
        validation_prediction = target_mean + (validation_z - score_mean)[:, None] * decoder
        validation_mse = float(np.mean((validation_prediction - validation_t) ** 2))
        fits[ridge] = (decoder, validation_mse, effective_ridge)
    selected = min(candidates, key=lambda value: fits[value][1])
    decoder, validation_mse, effective_ridge = fits[selected]
    test_prediction = target_mean + (test_z - score_mean)[:, None] * decoder
    test_mse = float(np.mean((test_prediction - test_t) ** 2))
    test_baseline_mse = float(np.mean((test_t - target_mean) ** 2))
    return FrozenEncoderRidgeFit(
        score_variance=float(score_variance),
        cross_covariance_norm=float(np.linalg.norm(cross_covariance)),
        selected_ridge=float(selected),
        effective_ridge=float(effective_ridge),
        score_mean=score_mean,
        target_mean=np.asarray(target_mean, dtype=np.float64),
        decoder_weight=np.asarray(decoder, dtype=np.float64),
        decoder_bias=np.asarray(target_mean - score_mean * decoder, dtype=np.float64),
        validation_mse=validation_mse,
        test_mse=test_mse,
        test_constant_train_mean_mse=test_baseline_mse,
    )


def _null_summary(observed: float, values: Any) -> Dict[str, float]:
    null = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = null[np.isfinite(null)]
    if not len(finite):
        return {
            "null_mean": float("nan"),
            "null_q95": float("nan"),
            "effect_over_null_mean": float("nan"),
            "permutation_p_upper": float("nan"),
        }
    return {
        "null_mean": float(np.mean(finite)),
        "null_q95": float(np.quantile(finite, 0.95)),
        "effect_over_null_mean": float(observed - np.mean(finite)),
        "permutation_p_upper": float(
            (1 + np.sum(finite >= observed)) / (len(finite) + 1)
        ),
    }


@dataclass(frozen=True)
class LearnedSpectralBridgeResult:
    """Summary plus auditable null rows and directions for one checkpoint."""

    summary_values: Mapping[str, Any]
    permutation_rows: tuple[Mapping[str, Any], ...]
    directions: Mapping[str, np.ndarray]

    def summary(self) -> Dict[str, Any]:
        return dict(self.summary_values)


def evaluate_learned_spectral_bridge(
    recovery: CrossSplitRecovery,
    learned_test: LearnedIendMeasurements,
    train: IendSignalBatch,
    validation: IendSignalBatch,
    test: IendSignalBatch,
    *,
    source_kind: SourceKind,
) -> LearnedSpectralBridgeResult:
    """Compare one trained IEND with a source-matched spectral estimator.

    Ridge is assumed to have been selected by ``recovery`` on validation MSE.
    All learned--spectral score comparisons use the test split.  The null
    spectral models keep that selected ridge fixed and refit after permuting
    train source--target pairing; this isolates alignment beyond finite-sample
    rank-one structure without selecting anything on test.
    """
    if source_kind == "diff":
        raise ValueError("diff is not a trainable source policy for the E0c bridge")
    if recovery.source_kind != source_kind:
        raise ValueError(
            f"recovery source {recovery.source_kind!r} does not match {source_kind!r}"
        )
    checkpoint_source = learned_test.checkpoint_source
    if checkpoint_source is None:
        raise ValueError("Learned checkpoint source is unknown; source matching is not auditable")
    if str(checkpoint_source) != str(source_kind):
        raise ValueError(
            f"Learned checkpoint source {checkpoint_source!r} does not match "
            f"spectral source {source_kind!r}"
        )

    train_view = train.compile(source_kind)
    validation_view = validation.compile(source_kind)
    test_view = test.compile(source_kind)
    if len(learned_test.preactivation[source_kind]) != test_view.n_samples:
        raise ValueError("Learned measurements and compiled test rows are not aligned")

    model = recovery.model
    spectral_test_score = model.score(test_view.source)
    summary = learned_test.summary(
        source_kind,
        spectral=model,
        spectral_score=spectral_test_score,
        orientation=test_view.orientation,
    )
    # Retain the older keys for compatibility, while exposing names that state
    # exactly what is being compared in the E0c table.
    summary.update(
        diagnostic_role="learned_spectral_bridge",
        source_kind=source_kind,
        selected_ridge=float(recovery.selected_ridge),
        ridge_selected_on="validation_reconstruction_mse",
        bridge_evaluated_on="frozen_test",
        permutation_refit_policy="permute_train_targets_keep_observed_validation_ridge",
        permutation_n=int(len(recovery.permutation_models)),
        learned_spectral_encoder_weight_cosine_abs=summary.get(
            "learned_spectral_encoder_sketch_cosine_abs"
        ),
        learned_spectral_decoder_weight_cosine_abs=summary.get(
            "learned_spectral_decoder_sketch_cosine_abs"
        ),
    )

    learned_pre_metrics = _orientation_invariant_label_metrics(
        learned_test.preactivation[source_kind], test_view.orientation
    )
    learned_latent_metrics = _orientation_invariant_label_metrics(
        learned_test.latent[source_kind], test_view.orientation
    )
    spectral_metrics = _orientation_invariant_label_metrics(
        spectral_test_score, test_view.orientation
    )
    for key, value in learned_pre_metrics.items():
        summary[f"learned_preactivation_test_label_{key}"] = value
    for key, value in learned_latent_metrics.items():
        summary[f"learned_latent_test_label_{key}"] = value
    for key, value in spectral_metrics.items():
        summary[f"spectral_score_test_label_{key}"] = value

    target_directions = {
        "train": _class_mean_difference(train_view.target, train_view.orientation),
        "validation": _class_mean_difference(
            validation_view.target, validation_view.orientation
        ),
        "test": _class_mean_difference(test_view.target, test_view.orientation),
    }
    summary["target_direction_train_validation_cosine"] = safe_cosine(
        target_directions["train"], target_directions["validation"]
    )
    summary["target_direction_train_test_cosine"] = safe_cosine(
        target_directions["train"], target_directions["test"]
    )
    for split, direction in target_directions.items():
        summary[f"learned_decoder_{split}_target_direction_cosine_abs"] = safe_cosine(
            learned_test.projected_decoder_weight, direction, absolute=True
        )
        summary[f"spectral_decoder_{split}_target_direction_cosine_abs"] = safe_cosine(
            model.decoder_direction, direction, absolute=True
        )
    summary["learned_decoder_crossfit_target_direction_cosine_abs"] = summary[
        "learned_decoder_test_target_direction_cosine_abs"
    ]

    # Optimization audit: preserve the trained encoder and replace only its
    # decoder by the exact scalar ridge optimum. The tanh-latent fit asks whether
    # checkpoint decoder training/selection failed; the preactivation fit asks
    # how much additional target information tanh discarded.
    split_views = {
        "train": train_view,
        "validation": validation_view,
        "test": test_view,
    }
    learned_scores = {
        split: view.source @ learned_test.projected_encoder_weight
        + float(learned_test.encoder_bias)
        for split, view in split_views.items()
    }
    learned_latents = {
        split: _apply_activation(score, learned_test.activation_encoder)
        for split, score in learned_scores.items()
    }
    for score_name, values in (
        ("latent", learned_latents),
        ("preactivation", learned_scores),
    ):
        posthoc = fit_frozen_encoder_ridge_decoder(
            values["train"],
            train_view.target,
            values["validation"],
            validation_view.target,
            values["test"],
            test_view.target,
            ridge_grid=recovery.ridge_validation_mse.keys(),
            relative_ridge=True,
        )
        decoder_direction = posthoc.decoder_direction
        for key, value in posthoc.summary().items():
            summary[f"posthoc_{score_name}_decoder_{key}"] = value
        summary[
            f"posthoc_{score_name}_decoder_test_target_direction_cosine_abs"
        ] = safe_cosine(
            decoder_direction, target_directions["test"], absolute=True
        )
        summary[
            f"posthoc_{score_name}_decoder_learned_decoder_cosine_abs"
        ] = safe_cosine(
            decoder_direction,
            learned_test.projected_decoder_weight,
            absolute=True,
        )
        summary[
            f"posthoc_{score_name}_decoder_spectral_decoder_cosine_abs"
        ] = safe_cosine(
            decoder_direction, model.decoder_direction, absolute=True
        )
    summary["posthoc_decoder_optimization_gap_test_ev"] = (
        float(summary["posthoc_latent_decoder_test_explained_variance"])
        - float(summary["learned_reconstruction_gain_vs_decoder_bias"])
    )
    summary["posthoc_preactivation_over_latent_test_ev"] = (
        float(summary["posthoc_preactivation_decoder_test_explained_variance"])
        - float(summary["posthoc_latent_decoder_test_explained_variance"])
    )

    observed_by_metric = {
        "learned_spectral_preactivation_correlation_abs": float(
            summary["learned_spectral_preactivation_correlation_abs"]
        ),
        "learned_spectral_latent_correlation_abs": float(
            summary["learned_spectral_latent_correlation_abs"]
        ),
        "learned_spectral_encoder_weight_cosine_abs": float(
            summary["learned_spectral_encoder_weight_cosine_abs"]
        ),
        "learned_spectral_decoder_weight_cosine_abs": float(
            summary["learned_spectral_decoder_weight_cosine_abs"]
        ),
    }
    null_values = {key: [] for key in observed_by_metric}
    permutation_rows = []
    learned_preactivation = learned_test.preactivation[source_kind]
    learned_latent = learned_test.latent[source_kind]
    for index, null_model in enumerate(recovery.permutation_models):
        null_score = null_model.score(test_view.source)
        row = {
            "permutation": index,
            "learned_spectral_preactivation_correlation_abs": abs(
                _pearson(learned_preactivation, null_score)
            ),
            "learned_spectral_latent_correlation_abs": abs(
                _pearson(learned_latent, null_score)
            ),
            "learned_spectral_encoder_weight_cosine_abs": safe_cosine(
                learned_test.projected_encoder_weight,
                null_model.encoder_direction,
                absolute=True,
            ),
            "learned_spectral_decoder_weight_cosine_abs": safe_cosine(
                learned_test.projected_decoder_weight,
                null_model.decoder_direction,
                absolute=True,
            ),
        }
        permutation_rows.append(row)
        for key in null_values:
            null_values[key].append(float(row[key]))
    for metric, observed in observed_by_metric.items():
        for key, value in _null_summary(observed, null_values[metric]).items():
            summary[f"{metric}_{key}"] = value

    projection = dict((test.metadata or {}).get("projection") or {})
    identity_space = str(projection.get("kind") or "") == "identity"
    summary["alignment_space"] = "native" if identity_space else "countsketch"
    summary["parameter_alignment_exact_native_space"] = bool(identity_space)
    summary["projection_mapping_id"] = projection.get("mapping_id")

    directions = {
        "learned_encoder_weight": np.asarray(
            learned_test.projected_encoder_weight, dtype=np.float64
        ),
        "learned_decoder_weight": np.asarray(
            learned_test.projected_decoder_weight, dtype=np.float64
        ),
        "spectral_encoder_direction": np.asarray(
            model.encoder_direction, dtype=np.float64
        ),
        "spectral_decoder_direction": np.asarray(
            model.decoder_direction, dtype=np.float64
        ),
        **{
            f"target_direction_{split}": np.asarray(direction, dtype=np.float64)
            for split, direction in target_directions.items()
        },
    }
    return LearnedSpectralBridgeResult(
        summary_values=summary,
        permutation_rows=tuple(permutation_rows),
        directions=directions,
    )
