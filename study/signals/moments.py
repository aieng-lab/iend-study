"""Small, task-neutral distribution diagnostics for IEND signal batches."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from study.signals.schema import as_float_matrix, as_orientation




def safe_cosine(left: Any, right: Any, *, absolute: bool = False) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"cosine shapes differ: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan")
    value = float(np.dot(a, b) / denom)
    return abs(value) if absolute else value


def row_cosines(left: Any, right: Any) -> np.ndarray:
    a = as_float_matrix(left, name="left")
    b = as_float_matrix(right, name="right")
    if a.shape != b.shape:
        raise ValueError(f"row cosine shapes differ: {a.shape} vs {b.shape}")
    numer = np.einsum("ij,ij->i", a, b)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan, dtype=np.float64)
    valid = denom > 0.0
    out[valid] = numer[valid] / denom[valid]
    return out


def antisymmetry_error(target: Any, orientation: Any, *, eps: float = 1e-12) -> float:
    """Relative failure of E[target|+] = -E[target|-].

    Neutral orientation-zero rows are ignored. Both signs must be present.
    """
    matrix = as_float_matrix(target, name="target")
    labels = as_orientation(orientation, n_samples=len(matrix))
    positive = matrix[labels > 0]
    negative = matrix[labels < 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    mu_positive = positive.mean(axis=0)
    mu_negative = negative.mean(axis=0)
    numerator = np.linalg.norm(mu_positive + mu_negative)
    denominator = np.linalg.norm(mu_positive - mu_negative)
    return float(numerator / (denominator + float(eps)))






def saturation_summary(
    preactivation: Any,
    *,
    thresholds: Sequence[float] = (2.0, 3.0),
) -> Dict[str, float]:
    values = np.asarray(preactivation, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("preactivation must be a non-empty finite vector")
    out = {
        "preactivation_mean": float(np.mean(values)),
        "preactivation_std": float(np.std(values)),
        "preactivation_abs_mean": float(np.mean(np.abs(values))),
    }
    for threshold in thresholds:
        key = str(float(threshold)).replace(".", "_")
        out[f"preactivation_abs_gt_{key}"] = float(np.mean(np.abs(values) > threshold))
    return out


