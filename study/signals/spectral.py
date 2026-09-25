"""Matrix-free-in-feature-space reduced-rank diagnostics for IEND.

The implementation forms only sample-space Gram matrices.  It still accepts
materialized sample matrices; checkpoint extraction may stream/sketch those
matrices before calling this module when the original signal space is too wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from study.signals.moments import safe_cosine
from study.signals.schema import as_float_matrix


@dataclass(frozen=True)
class SpectralResult:
    n_samples: int
    source_dim: int
    target_dim: int
    sample_rank: int
    ridge: float
    effective_ridge: float
    singular_values: np.ndarray
    rank_one_fraction: float
    spectral_gap_fraction: float
    source_mean: np.ndarray
    target_mean: np.ndarray
    encoder_coefficient: np.ndarray
    decoder_coefficient: np.ndarray
    encoder_direction: np.ndarray
    decoder_direction: np.ndarray
    whitened_source_mode: np.ndarray

    def predict(self, source: Any) -> np.ndarray:
        """Predict targets with the fitted rank-one ridge regression."""
        x = as_float_matrix(source, name="source")
        if x.shape[1] != self.source_dim:
            raise ValueError(
                f"source has dimension {x.shape[1]}, expected {self.source_dim}"
            )
        score = (x - self.source_mean) @ self.encoder_coefficient
        return self.target_mean + score[:, None] * self.decoder_coefficient[None, :]

    def score(self, source: Any) -> np.ndarray:
        """Return the fitted scalar rank-one coordinate before decoding."""
        x = as_float_matrix(source, name="source")
        if x.shape[1] != self.source_dim:
            raise ValueError(
                f"source has dimension {x.shape[1]}, expected {self.source_dim}"
            )
        return (x - self.source_mean) @ self.encoder_coefficient

    def summary(self, *, n_singular_values: int = 10) -> Dict[str, Any]:
        values = self.singular_values[: max(1, int(n_singular_values))]
        return {
            "n_samples": self.n_samples,
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "sample_rank": self.sample_rank,
            "row_span_saturated": bool(self.sample_rank >= self.n_samples - 1),
            "source_dim_to_sample_ratio": float(self.source_dim / self.n_samples),
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "singular_values": [float(value) for value in values],
            "sigma1": float(self.singular_values[0]) if len(self.singular_values) else 0.0,
            "sigma2": float(self.singular_values[1]) if len(self.singular_values) > 1 else 0.0,
            "rank_one_fraction": self.rank_one_fraction,
            "spectral_gap_fraction": self.spectral_gap_fraction,
        }

    def alignments(
        self,
        *,
        learned_encoder: Optional[Any] = None,
        learned_decoder: Optional[Any] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if learned_encoder is not None:
            out["learned_spectral_encoder_cosine_abs"] = safe_cosine(
                learned_encoder, self.encoder_direction, absolute=True
            )
        if learned_decoder is not None:
            out["learned_spectral_decoder_cosine_abs"] = safe_cosine(
                learned_decoder, self.decoder_direction, absolute=True
            )
        return out


@dataclass(frozen=True)
class RankKSpectralResult:
    """Rank-k ridge-regression solution in compact factorized form."""

    n_samples: int
    source_dim: int
    target_dim: int
    sample_rank: int
    requested_rank: int
    fitted_rank: int
    ridge: float
    effective_ridge: float
    singular_values: np.ndarray
    captured_energy_fraction: float
    source_mean: np.ndarray
    target_mean: np.ndarray
    encoder_coefficients: np.ndarray
    decoder_coefficients: np.ndarray
    whitened_source_modes: np.ndarray

    def predict(self, source: Any) -> np.ndarray:
        x = as_float_matrix(source, name="source")
        if x.shape[1] != self.source_dim:
            raise ValueError(
                f"source has dimension {x.shape[1]}, expected {self.source_dim}"
            )
        scores = (x - self.source_mean) @ self.encoder_coefficients
        return self.target_mean + scores @ self.decoder_coefficients.T

    def score(self, source: Any) -> np.ndarray:
        x = as_float_matrix(source, name="source")
        if x.shape[1] != self.source_dim:
            raise ValueError(
                f"source has dimension {x.shape[1]}, expected {self.source_dim}"
            )
        return (x - self.source_mean) @ self.encoder_coefficients

    @property
    def encoder_directions(self) -> np.ndarray:
        norms = np.linalg.norm(self.encoder_coefficients, axis=0)
        return self.encoder_coefficients / norms.reshape(1, -1)

    @property
    def decoder_directions(self) -> np.ndarray:
        norms = np.linalg.norm(self.decoder_coefficients, axis=0)
        return self.decoder_coefficients / norms.reshape(1, -1)

    def summary(self, *, n_singular_values: int = 10) -> Dict[str, Any]:
        values = self.singular_values[: max(1, int(n_singular_values))]
        sigma1 = float(values[0]) if len(values) else 0.0
        sigma2 = float(values[1]) if len(values) > 1 else 0.0
        energy = float(np.sum(self.singular_values**2))
        rank_one_fraction = (
            float(self.singular_values[0] ** 2 / energy) if energy > 0.0 else 0.0
        )
        return {
            "n_samples": self.n_samples,
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "sample_rank": self.sample_rank,
            "row_span_saturated": bool(self.sample_rank >= self.n_samples - 1),
            "source_dim_to_sample_ratio": float(self.source_dim / self.n_samples),
            "requested_rank": self.requested_rank,
            "fitted_rank": self.fitted_rank,
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "singular_values": [float(value) for value in values],
            "sigma1": sigma1,
            "sigma2": sigma2,
            "rank_one_fraction": rank_one_fraction,
            "spectral_gap_fraction": (
                float((sigma1 - sigma2) / sigma1) if sigma1 > 0.0 else 0.0
            ),
            "captured_energy_fraction": self.captured_energy_fraction,
        }


@dataclass(frozen=True)
class SpectralPermutationNull:
    n_permutations: int
    seed: int
    observed_sigma1: float
    null_sigma1: np.ndarray

    def summary(self) -> Dict[str, Any]:
        null = self.null_sigma1
        exceedances = int(np.sum(null >= self.observed_sigma1))
        std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
        mean = float(np.mean(null))
        return {
            "permutation_n": self.n_permutations,
            "permutation_seed": self.seed,
            "permutation_sigma1_mean": mean,
            "permutation_sigma1_std": std,
            "permutation_sigma1_q95": float(np.quantile(null, 0.95)),
            "permutation_sigma1_ratio": float(
                self.observed_sigma1 / mean if mean > 0.0 else np.inf
            ),
            "permutation_sigma1_z": float(
                (self.observed_sigma1 - mean) / std if std > 0.0 else np.inf
            ),
            "permutation_p_upper": float(
                (exceedances + 1) / (self.n_permutations + 1)
            ),
        }


@dataclass(frozen=True)
class _SourceFactorization:
    n_samples: int
    source_dim: int
    centered: bool
    source_mean: np.ndarray
    u: np.ndarray
    singular_values: np.ndarray
    vt: np.ndarray
    covariance_eigenvalues: np.ndarray
    denominator: float


def _factor_source(
    source: Any,
    *,
    center: bool = True,
    svd_tolerance: float = 1e-10,
) -> _SourceFactorization:
    """Factor a source matrix once for several ridges or target permutations."""
    x = as_float_matrix(source, name="source").copy()
    if len(x) < 2:
        raise ValueError("At least two samples are required for covariance diagnostics")
    source_mean = x.mean(axis=0) if center else np.zeros(x.shape[1], dtype=np.float64)
    if center:
        x -= source_mean.reshape(1, -1)
    u, singular_x, vt = np.linalg.svd(x, full_matrices=False)
    if len(singular_x) == 0 or singular_x[0] <= 0.0:
        raise ValueError("Centered source has zero rank")
    cutoff = float(svd_tolerance) * float(singular_x[0])
    keep = singular_x > cutoff
    u = u[:, keep]
    singular_x = singular_x[keep]
    vt = vt[keep, :]
    denominator = float(len(x) - 1)
    return _SourceFactorization(
        n_samples=int(len(x)),
        source_dim=int(x.shape[1]),
        centered=bool(center),
        source_mean=source_mean,
        u=u,
        singular_values=singular_x,
        vt=vt,
        covariance_eigenvalues=(singular_x**2) / denominator,
        denominator=denominator,
    )


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Cannot normalize zero/non-finite {name}")
    return vector / norm


def reduced_rank_spectral(
    source: Any,
    target: Any,
    *,
    ridge: float = 1e-6,
    relative_ridge: bool = True,
    center: bool = True,
    svd_tolerance: float = 1e-10,
    _source_factorization: Optional[_SourceFactorization] = None,
) -> SpectralResult:
    """Solve the spectral part of ridge-regularized rank-one regression.

    After centering (which optimizes an unregularized decoder intercept), the
    population objective

        E[||target - B @ source||^2] + lambda ||B||_F^2,
        rank(B) <= 1,

    is solved by the leading singular triplet of

        Cov(target, source) @ (Cov(source) + lambda I)^(-1/2).

    ``relative_ridge=True`` interprets ``ridge`` as a fraction of the largest
    observed source covariance eigenvalue. The returned ``encoder_direction``
    includes the second inverse-square-root factor required by the regression
    encoder, whereas ``whitened_source_mode`` is the operator's right singular
    vector before that factor.
    """
    x = as_float_matrix(source, name="source")
    y = as_float_matrix(target, name="target").copy()
    if len(x) != len(y):
        raise ValueError(f"source and target sample counts differ: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("At least two samples are required for covariance diagnostics")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    factor = _source_factorization
    if factor is None:
        factor = _factor_source(x, center=center, svd_tolerance=svd_tolerance)
    elif (
        factor.n_samples != len(x)
        or factor.source_dim != x.shape[1]
        or factor.centered != bool(center)
    ):
        raise ValueError("source factorization is incompatible with source matrix")
    source_mean = factor.source_mean
    target_mean = y.mean(axis=0) if center else np.zeros(y.shape[1], dtype=np.float64)
    if center:
        y -= target_mean.reshape(1, -1)

    # Compact source SVD: X = U diag(s) V^T.  The source covariance and every
    # nonzero right mode of the cross-covariance live in span(V).
    u = factor.u
    singular_x = factor.singular_values
    vt = factor.vt
    rank = int(len(singular_x))
    denom = factor.denominator
    covariance_eigenvalues = factor.covariance_eigenvalues
    effective_ridge = float(ridge)
    if relative_ridge:
        effective_ridge *= float(covariance_eigenvalues[0])

    inverse_sqrt = 1.0 / np.sqrt(covariance_eigenvalues + effective_ridge)
    # K = Y^T U diag(s / (n-1) / sqrt(eig + lambda)) V^T.
    scale = (singular_x / denom) * inverse_sqrt

    # K_compact^T K_compact without constructing target_dim x rank.
    y_gram = y @ y.T
    left_sample = u * scale.reshape(1, -1)
    compact_gram = left_sample.T @ y_gram @ left_sample
    compact_gram = (compact_gram + compact_gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(compact_gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    # The Gram eigendecomposition can turn an exact algebraic zero into an
    # O(sqrt(machine-epsilon)) singular value.  Do not report those numerical
    # modes as evidence for additional structure.
    numerical_floor = float(singular_values[0]) * 10.0 * np.sqrt(
        np.finfo(np.float64).eps
    )
    positive = singular_values > max(1e-14, numerical_floor)
    singular_values = singular_values[positive]
    eigenvectors = eigenvectors[:, positive]
    if len(singular_values) == 0:
        raise ValueError("Source-target cross-covariance is numerically zero")

    compact_right = eigenvectors[:, 0]
    whitened_source_mode = _normalize(vt.T @ compact_right, name="source mode")
    encoder_coefficient = vt.T @ (inverse_sqrt * compact_right)
    encoder_direction = _normalize(encoder_coefficient, name="encoder direction")
    decoder_direction = y.T @ (left_sample @ compact_right)
    decoder_direction = _normalize(
        decoder_direction / float(singular_values[0]), name="decoder direction"
    )
    decoder_coefficient = float(singular_values[0]) * decoder_direction

    energy = float(np.sum(singular_values**2))
    rank_one_fraction = float(singular_values[0] ** 2 / energy) if energy > 0 else 0.0
    sigma2 = float(singular_values[1]) if len(singular_values) > 1 else 0.0
    gap = float((singular_values[0] - sigma2) / singular_values[0])

    return SpectralResult(
        n_samples=int(len(x)),
        source_dim=int(x.shape[1]),
        target_dim=int(y.shape[1]),
        sample_rank=rank,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
        singular_values=singular_values,
        rank_one_fraction=rank_one_fraction,
        spectral_gap_fraction=gap,
        source_mean=source_mean,
        target_mean=target_mean,
        encoder_coefficient=encoder_coefficient,
        decoder_coefficient=decoder_coefficient,
        encoder_direction=encoder_direction,
        decoder_direction=decoder_direction,
        whitened_source_mode=whitened_source_mode,
    )


def reduced_rank_spectral_k(
    source: Any,
    target: Any,
    *,
    rank: int,
    ridge: float = 1e-6,
    relative_ridge: bool = True,
    center: bool = True,
    svd_tolerance: float = 1e-10,
    _source_factorization: Optional[_SourceFactorization] = None,
) -> RankKSpectralResult:
    """Solve ridge-regularized reduced-rank regression for arbitrary rank.

    This is the rank-k analogue of :func:`reduced_rank_spectral`. It uses only
    sample-space Gram matrices and returns the factorization

        B_k = decoder_coefficients @ encoder_coefficients.T.
    """
    requested_rank = int(rank)
    if requested_rank <= 0:
        raise ValueError("rank must be positive")
    x = as_float_matrix(source, name="source")
    y = as_float_matrix(target, name="target").copy()
    if len(x) != len(y):
        raise ValueError(f"source and target sample counts differ: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("At least two samples are required for covariance diagnostics")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    factor = _source_factorization
    if factor is None:
        factor = _factor_source(x, center=center, svd_tolerance=svd_tolerance)
    elif (
        factor.n_samples != len(x)
        or factor.source_dim != x.shape[1]
        or factor.centered != bool(center)
    ):
        raise ValueError("source factorization is incompatible with source matrix")

    source_mean = factor.source_mean
    target_mean = y.mean(axis=0) if center else np.zeros(y.shape[1], dtype=np.float64)
    if center:
        y -= target_mean.reshape(1, -1)

    covariance_eigenvalues = factor.covariance_eigenvalues
    effective_ridge = float(ridge)
    if relative_ridge:
        effective_ridge *= float(covariance_eigenvalues[0])
    inverse_sqrt = 1.0 / np.sqrt(covariance_eigenvalues + effective_ridge)
    scale = (factor.singular_values / factor.denominator) * inverse_sqrt
    left_sample = factor.u * scale.reshape(1, -1)
    compact_gram = left_sample.T @ (y @ y.T) @ left_sample
    compact_gram = (compact_gram + compact_gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(compact_gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    numerical_floor = float(singular_values[0]) * 10.0 * np.sqrt(
        np.finfo(np.float64).eps
    )
    positive = singular_values > max(1e-14, numerical_floor)
    singular_values = singular_values[positive]
    eigenvectors = eigenvectors[:, positive]
    if len(singular_values) == 0:
        raise ValueError("Source-target cross-covariance is numerically zero")

    fitted_rank = min(requested_rank, len(singular_values))
    compact_right = eigenvectors[:, :fitted_rank]
    whitened_source_modes = factor.vt.T @ compact_right
    encoder_coefficients = factor.vt.T @ (
        inverse_sqrt.reshape(-1, 1) * compact_right
    )
    decoder_coefficients = y.T @ (left_sample @ compact_right)
    energy = float(np.sum(singular_values**2))
    captured = float(np.sum(singular_values[:fitted_rank] ** 2) / energy)
    return RankKSpectralResult(
        n_samples=int(len(x)),
        source_dim=int(x.shape[1]),
        target_dim=int(y.shape[1]),
        sample_rank=int(len(factor.singular_values)),
        requested_rank=requested_rank,
        fitted_rank=fitted_rank,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
        singular_values=singular_values,
        captured_energy_fraction=captured,
        source_mean=source_mean,
        target_mean=target_mean,
        encoder_coefficients=encoder_coefficients,
        decoder_coefficients=decoder_coefficients,
        whitened_source_modes=whitened_source_modes,
    )


def spectral_permutation_null(
    source: Any,
    target: Any,
    *,
    observed: Optional[SpectralResult] = None,
    n_permutations: int = 100,
    seed: int = 0,
    ridge: float = 1e-6,
    relative_ridge: bool = True,
    center: bool = True,
) -> SpectralPermutationNull:
    """Permutation null for absolute source--target association strength.

    The null statistic is ``sigma1``, not the normalized rank-one fraction: a
    target distribution may be intrinsically one-dimensional even after its
    pairing with the source is destroyed.
    """
    x = as_float_matrix(source, name="source")
    y = as_float_matrix(target, name="target")
    if len(x) != len(y):
        raise ValueError(f"source and target sample counts differ: {len(x)} vs {len(y)}")
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    if observed is None:
        observed = reduced_rank_spectral(
            x,
            y,
            ridge=ridge,
            relative_ridge=relative_ridge,
            center=center,
        )
    source_factorization = _factor_source(x, center=center)
    rng = np.random.default_rng(int(seed))
    null = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        permuted = reduced_rank_spectral(
            x,
            y[rng.permutation(len(y))],
            ridge=ridge,
            relative_ridge=relative_ridge,
            center=center,
            _source_factorization=source_factorization,
        )
        null[index] = permuted.singular_values[0]
    return SpectralPermutationNull(
        n_permutations=n_permutations,
        seed=int(seed),
        observed_sigma1=float(observed.singular_values[0]),
        null_sigma1=null,
    )
