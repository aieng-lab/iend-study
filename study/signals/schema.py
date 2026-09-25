"""Array-level contracts and source compilation for IEND theory analyses.

The GRADIEND package owns model signal extraction.  These classes only describe
already-extracted factual/alternative vectors and reproduce the documented
source/target geometry for mathematical diagnostics and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np

SourceKind = Literal["factual", "alternative", "diff", "both"]
SOURCE_KINDS: Tuple[SourceKind, ...] = ("factual", "alternative", "diff", "both")


def as_float_matrix(value: Any, *, name: str) -> np.ndarray:
    """Return a finite, two-dimensional float64 matrix."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(array)


def as_orientation(value: Any, *, n_samples: int, name: str = "orientation") -> np.ndarray:
    """Return a length-n vector with values in {-1, 0, +1}."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) != int(n_samples):
        raise ValueError(f"{name} has length {len(array)}, expected {n_samples}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    rounded = np.rint(array)
    if not np.allclose(array, rounded) or not np.isin(rounded, (-1.0, 0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only -1, 0, or +1")
    return rounded.astype(np.float64, copy=False)


@dataclass(frozen=True)
class CompiledIendBatch:
    """One source/target view in the exact space consumed by an IEND."""

    source: np.ndarray
    target: np.ndarray
    orientation: np.ndarray
    source_kind: SourceKind
    factual_orientation: Optional[np.ndarray] = None
    pole: Optional[np.ndarray] = None
    sample_ids: Optional[Tuple[str, ...]] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        source = as_float_matrix(self.source, name="source")
        target = as_float_matrix(self.target, name="target")
        if source.shape[0] != target.shape[0]:
            raise ValueError(
                f"source and target sample counts differ: {source.shape[0]} vs {target.shape[0]}"
            )
        orientation = as_orientation(self.orientation, n_samples=source.shape[0])
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown source_kind {self.source_kind!r}")
        factual_orientation = None
        if self.factual_orientation is not None:
            factual_orientation = as_orientation(
                self.factual_orientation,
                n_samples=source.shape[0],
                name="factual_orientation",
            )
        pole = None
        if self.pole is not None:
            pole = as_orientation(self.pole, n_samples=source.shape[0], name="pole")
            if np.any(pole == 0):
                raise ValueError("pole must contain only -1 or +1")
        sample_ids = self.sample_ids
        if sample_ids is not None:
            sample_ids = tuple(str(value) for value in sample_ids)
            if len(sample_ids) != source.shape[0]:
                raise ValueError(
                    f"sample_ids has length {len(sample_ids)}, expected {source.shape[0]}"
                )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "factual_orientation", factual_orientation)
        object.__setattr__(self, "pole", pole)
        object.__setattr__(self, "sample_ids", sample_ids)

    @property
    def n_samples(self) -> int:
        return int(self.source.shape[0])


@dataclass(frozen=True)
class IendSignalBatch:
    """Aligned factual/alternative signals before source compilation.

    ``orientation`` describes the factual class under one explicit global
    contrast convention.  It is +1 for the positive class, -1 for the negative
    class, and optionally 0 for neutral identity transitions.
    """

    factual: np.ndarray
    alternative: np.ndarray
    orientation: np.ndarray
    sample_ids: Optional[Tuple[str, ...]] = None
    factual_classes: Optional[Tuple[str, ...]] = None
    alternative_classes: Optional[Tuple[str, ...]] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        factual = as_float_matrix(self.factual, name="factual")
        alternative = as_float_matrix(self.alternative, name="alternative")
        if factual.shape != alternative.shape:
            raise ValueError(
                f"factual and alternative shapes differ: {factual.shape} vs {alternative.shape}"
            )
        orientation = as_orientation(self.orientation, n_samples=factual.shape[0])
        sample_ids = self.sample_ids
        if sample_ids is not None:
            sample_ids = tuple(str(value) for value in sample_ids)
            if len(sample_ids) != factual.shape[0]:
                raise ValueError(
                    f"sample_ids has length {len(sample_ids)}, expected {factual.shape[0]}"
                )
        factual_classes = self._validate_classes(
            self.factual_classes, n_samples=factual.shape[0], name="factual_classes"
        )
        alternative_classes = self._validate_classes(
            self.alternative_classes,
            n_samples=factual.shape[0],
            name="alternative_classes",
        )
        object.__setattr__(self, "factual", factual)
        object.__setattr__(self, "alternative", alternative)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "factual_classes", factual_classes)
        object.__setattr__(self, "alternative_classes", alternative_classes)

    @staticmethod
    def _validate_classes(
        value: Optional[Tuple[str, ...]], *, n_samples: int, name: str
    ) -> Optional[Tuple[str, ...]]:
        if value is None:
            return None
        classes = tuple(str(item) for item in value)
        if len(classes) != n_samples:
            raise ValueError(f"{name} has length {len(classes)}, expected {n_samples}")
        return classes

    @property
    def n_samples(self) -> int:
        return int(self.factual.shape[0])

    @property
    def signal_dim(self) -> int:
        return int(self.factual.shape[1])

    def subset(
        self,
        mask: Any,
        *,
        orientation: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "IendSignalBatch":
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if len(selected) != self.n_samples:
            raise ValueError(
                f"subset mask has length {len(selected)}, expected {self.n_samples}"
            )
        if not np.any(selected):
            raise ValueError("subset mask selects no observations")
        selected_orientation = (
            self.orientation[selected]
            if orientation is None
            else as_orientation(orientation, n_samples=int(np.sum(selected)))
        )
        base_metadata = dict(self.metadata or {})
        primitive = base_metadata.get("primitive_pair_geometry")
        if isinstance(primitive, (list, tuple)) and len(primitive) == self.n_samples:
            base_metadata["primitive_pair_geometry"] = list(
                np.asarray(primitive, dtype=object)[selected]
            )
        merged_metadata = {**base_metadata, **dict(metadata or {})}
        return IendSignalBatch(
            factual=self.factual[selected],
            alternative=self.alternative[selected],
            orientation=selected_orientation,
            sample_ids=(
                tuple(np.asarray(self.sample_ids, dtype=object)[selected].tolist())
                if self.sample_ids is not None
                else None
            ),
            factual_classes=(
                tuple(np.asarray(self.factual_classes, dtype=object)[selected].tolist())
                if self.factual_classes is not None
                else None
            ),
            alternative_classes=(
                tuple(np.asarray(self.alternative_classes, dtype=object)[selected].tolist())
                if self.alternative_classes is not None
                else None
            ),
            metadata=merged_metadata,
        )

    def compile(self, source_kind: SourceKind) -> CompiledIendBatch:
        """Compile factual/alternative/diff/both using package semantics.

        For ``alternative``, the reconstruction target remains factual minus
        alternative, while the exposed source orientation is inverted.  For
        ``both``, every pair is represented in both orientations, matching the
        population augmentation implemented by alternating package batches.
        """
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown source_kind {source_kind!r}")
        diff = self.factual - self.alternative
        y = self.orientation
        ids = self.sample_ids

        if source_kind == "factual":
            return CompiledIendBatch(
                source=self.factual,
                target=diff,
                orientation=y,
                factual_orientation=y,
                pole=np.ones_like(y),
                source_kind=source_kind,
                sample_ids=ids,
                metadata=self.metadata,
            )
        if source_kind == "alternative":
            return CompiledIendBatch(
                source=self.alternative,
                target=diff,
                orientation=-y,
                factual_orientation=y,
                pole=-np.ones_like(y),
                source_kind=source_kind,
                sample_ids=ids,
                metadata=self.metadata,
            )
        if source_kind == "diff":
            return CompiledIendBatch(
                source=diff,
                target=diff,
                orientation=y,
                factual_orientation=y,
                pole=np.ones_like(y),
                source_kind=source_kind,
                sample_ids=ids,
                metadata=self.metadata,
            )

        both_ids: Optional[Tuple[str, ...]] = None
        if ids is not None:
            both_ids = tuple(f"{value}:factual" for value in ids) + tuple(
                f"{value}:alternative" for value in ids
            )
        return CompiledIendBatch(
            source=np.concatenate([self.factual, self.alternative], axis=0),
            target=np.concatenate([diff, -diff], axis=0),
            orientation=np.concatenate([y, -y], axis=0),
            factual_orientation=np.concatenate([y, y], axis=0),
            pole=np.concatenate([np.ones_like(y), -np.ones_like(y)], axis=0),
            source_kind=source_kind,
            sample_ids=both_ids,
            metadata=self.metadata,
        )

    def summary(self) -> Dict[str, Any]:
        nonzero_signs = set(np.sign(self.orientation[self.orientation != 0]).astype(int))
        contrast_mode = (
            "pair" if nonzero_signs == {-1, 1} else "one_pole" if len(nonzero_signs) == 1 else "unlabeled"
        )
        summary = {
            "n_samples": self.n_samples,
            "signal_dim": self.signal_dim,
            "n_positive": int(np.sum(self.orientation > 0)),
            "n_negative": int(np.sum(self.orientation < 0)),
            "n_neutral": int(np.sum(self.orientation == 0)),
            "contrast_mode": contrast_mode,
        }
        if self.factual_classes is not None:
            summary["factual_classes"] = sorted(set(self.factual_classes))
        if self.alternative_classes is not None:
            summary["alternative_classes"] = sorted(set(self.alternative_classes))
        return summary
