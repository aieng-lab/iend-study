"""Shared deterministic projections for high-dimensional IEND signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from study.signals.schema import as_float_matrix


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer with fixed unsigned overflow semantics."""
    z = values.astype(np.uint64, copy=True)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


@dataclass(frozen=True)
class CountSketchProjector:
    """One-nonzero-per-column projection shared across every source view.

    The mapping is generated from coordinate indices and a seed, so it does not
    depend on task labels or source type.  It uses O(input_dim) mapping memory
    and O(input_dim) work per projected observation, but never forms a dense
    ``output_dim x input_dim`` matrix.
    """

    input_dim: int
    output_dim: int
    seed: int = 0
    _buckets: np.ndarray = field(init=False, repr=False, compare=False)
    _signs: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        input_dim = int(self.input_dim)
        output_dim = int(self.output_dim)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        offset = np.uint64(int(self.seed) & 0xFFFFFFFFFFFFFFFF)
        indices = np.arange(input_dim, dtype=np.uint64)
        hashed = _splitmix64(indices + offset + np.uint64(0x9E3779B97F4A7C15))
        buckets = np.asarray(hashed % np.uint64(output_dim), dtype=np.int64)
        signs = np.where((hashed >> np.uint64(63)) == 0, -1.0, 1.0)
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "output_dim", output_dim)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "_buckets", buckets)
        object.__setattr__(self, "_signs", signs.astype(np.float64, copy=False))

    def project_vector(self, value: Any) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(vector) != self.input_dim:
            raise ValueError(
                f"vector has dimension {len(vector)}, expected {self.input_dim}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("vector contains NaN or infinite values")
        return np.bincount(
            self._buckets,
            weights=self._signs * vector,
            minlength=self.output_dim,
        ).astype(np.float64, copy=False)

    def project_matrix(self, value: Any) -> np.ndarray:
        matrix = as_float_matrix(value, name="value")
        if matrix.shape[1] != self.input_dim:
            raise ValueError(
                f"matrix has dimension {matrix.shape[1]}, expected {self.input_dim}"
            )
        return np.stack([self.project_vector(row) for row in matrix], axis=0)

    def metadata(self) -> Dict[str, Any]:
        return {
            "kind": "countsketch",
            "version": 1,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seed": self.seed,
            "mapping_id": (
                f"countsketch-v1:d{self.input_dim}:k{self.output_dim}:seed{self.seed}"
            ),
        }
