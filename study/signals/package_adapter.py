"""Thin adapters from package-compiled signal datasets to theory arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from study.signals.projection import CountSketchProjector
from study.signals.schema import IendSignalBatch
from study.signals.learned import LearnedIendAccumulator, LearnedIendMeasurements


@dataclass(frozen=True)
class PairedSignalExtraction:
    batch: IendSignalBatch
    learned: Optional[LearnedIendMeasurements] = None


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        # numpy has no bfloat16 representation; every caller here immediately
        # widens to float64 anyway (see _one_vector), so upcast first.
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        return value.cpu().numpy()
    return np.asarray(value)


def _one_vector(value: Any, *, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == 1:
        return np.asarray(array, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        return np.asarray(array[0], dtype=np.float64)
    raise ValueError(
        f"{name} must describe exactly one signal vector; create package data with batch_size=1, "
        f"got shape {array.shape}"
    )


def _one_orientation(value: Any) -> float:
    array = _numpy(value).reshape(-1)
    if len(array) != 1:
        raise ValueError(
            f"label must contain one orientation; create package data with batch_size=1, got {array.shape}"
        )
    scalar = float(array[0])
    if scalar > 0:
        return 1.0
    if scalar < 0:
        return -1.0
    return 0.0


def _one_metadata_value(value: Any) -> Any:
    """Unwrap metadata emitted by a batch-size-one package dataset."""
    if hasattr(value, "detach"):
        array = value.detach().cpu().reshape(-1)
        if len(array) == 1:
            return array[0].item()
        return array.tolist()
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if len(flat) == 1:
            return flat[0].item()
        return flat.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _one_metadata_value(value[0])
    return value


def primitive_pair_geometry(
    factual: Any,
    alternative: Any,
) -> Dict[str, float]:
    """Exact native-space geometry for one aligned factual/alternative pair.

    The projection coefficients are the primary E1 quantities.  Under the
    shared-Jacobian binary softmax approximation they obey

    ``<g_f, diff>/||diff||^2 ~= 1-p_f`` and
    ``-<g_a, diff>/||diff||^2 ~= p_f``.

    Norm ratios are retained as secondary diagnostics because components
    orthogonal to ``diff`` can make them larger without changing the focal
    binary prediction.
    """
    f = _one_vector(factual, name="factual")
    a = _one_vector(alternative, name="alternative")
    if f.shape != a.shape:
        raise ValueError(f"factual and alternative shapes differ: {f.shape} vs {a.shape}")
    diff = f - a
    f_norm = float(np.linalg.norm(f))
    a_norm = float(np.linalg.norm(a))
    d_norm = float(np.linalg.norm(diff))
    d_sq = d_norm * d_norm

    def _cosine(x: np.ndarray, x_norm: float) -> float:
        denom = x_norm * d_norm
        return float(np.dot(x, diff) / denom) if denom > 0 else float("nan")

    factual_coefficient = float(np.dot(f, diff) / d_sq) if d_sq > 0 else float("nan")
    alternative_coefficient = float(-np.dot(a, diff) / d_sq) if d_sq > 0 else float("nan")
    return {
        "factual_norm": f_norm,
        "alternative_norm": a_norm,
        "diff_norm": d_norm,
        "factual_norm_over_diff": f_norm / d_norm if d_norm > 0 else float("nan"),
        "alternative_norm_over_diff": a_norm / d_norm if d_norm > 0 else float("nan"),
        "factual_diff_cosine": _cosine(f, f_norm),
        "alternative_diff_cosine": _cosine(a, a_norm),
        "factual_diff_coefficient": factual_coefficient,
        "alternative_diff_coefficient_oriented": alternative_coefficient,
        # This is algebraically one for every non-degenerate pair.  Persisting
        # it is a useful extraction/sign integrity check, not theory evidence.
        "coefficient_sum_error": factual_coefficient + alternative_coefficient - 1.0,
    }




def collect_paired_signal_dataset(
    dataset: Any,
    *,
    max_samples: Optional[int] = None,
    projection_dim: Optional[int] = None,
    projection_seed: int = 0,
    learned_iend: Optional[Any] = None,
    checkpoint_source: Optional[str] = None,
    return_extraction: bool = False,
) -> Any:
    """Collect a package dataset configured as factual-source/alternative-target.

    Construct ``dataset`` through the package with ``source='factual'``,
    ``target='alternative'``, and ``batch_size=1``.  The returned arrays are the
    aligned primitive signals from which this study compiles every experimental
    source.  When ``projection_dim`` is smaller than the original dimension, the
    exact same task-independent CountSketch is applied to both poles.
    """
    factual: List[np.ndarray] = []
    alternative: List[np.ndarray] = []
    orientations: List[float] = []
    sample_ids: List[str] = []
    factual_classes: List[str] = []
    alternative_classes: List[str] = []
    primitive_rows: List[Dict[str, Any]] = []
    projector: Optional[CountSketchProjector] = None
    learned_accumulator: Optional[LearnedIendAccumulator] = None
    original_dim: Optional[int] = None
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    for index in range(limit):
        row = dataset[index]
        if "source" not in row or "target" not in row:
            raise KeyError("Package paired dataset row must contain source and target")
        f_row = _one_vector(row["source"], name="source/factual")
        a_row = _one_vector(row["target"], name="target/alternative")
        if f_row.shape != a_row.shape:
            raise ValueError(
                f"factual and alternative row shapes differ: {f_row.shape} vs {a_row.shape}"
            )
        if original_dim is None:
            original_dim = int(len(f_row))
            if projection_dim is not None and int(projection_dim) < original_dim:
                projector = CountSketchProjector(
                    input_dim=original_dim,
                    output_dim=int(projection_dim),
                    seed=int(projection_seed),
                )
            if learned_iend is not None:
                learned_accumulator = LearnedIendAccumulator.from_model(
                    learned_iend,
                    projector=projector,
                    checkpoint_source=checkpoint_source,
                )
        elif len(f_row) != original_dim:
            raise ValueError(
                f"signal dimension changed at row {index}: {len(f_row)} vs {original_dim}"
            )
        if learned_accumulator is not None:
            learned_accumulator.observe_pair(f_row, a_row)
        geometry = primitive_pair_geometry(f_row, a_row)
        geometry.update(
            {
                "sample_id": str(_one_metadata_value(row.get("sample_id", row.get("pair_id", index)))),
                "orientation": _one_orientation(row.get("label", 0.0)),
                "factual_class": str(_one_metadata_value(row.get("factual_id", "unknown"))),
                "alternative_class": str(_one_metadata_value(row.get("alternative_id", "unknown"))),
                "factual_token": str(_one_metadata_value(row.get("factual_token", ""))),
                "alternative_token": str(_one_metadata_value(row.get("alternative_token", ""))),
            }
        )
        primitive_rows.append(geometry)
        if projector is not None:
            f_row = projector.project_vector(f_row)
            a_row = projector.project_vector(a_row)
        factual.append(f_row)
        alternative.append(a_row)
        orientations.append(_one_orientation(row.get("label", 0.0)))
        raw_id = row.get("sample_id", row.get("pair_id", index))
        if hasattr(raw_id, "detach"):
            raw_id = raw_id.detach().cpu().reshape(-1).tolist()
        sample_ids.append(str(raw_id))
        factual_classes.append(str(row.get("factual_id", "unknown")))
        alternative_classes.append(str(row.get("alternative_id", "unknown")))
    if not factual or original_dim is None:
        raise ValueError("Package paired dataset yielded no signal observations")
    projection_meta = (
        projector.metadata()
        if projector is not None
        else {
            "kind": "identity",
            "version": 1,
            "input_dim": original_dim,
            "output_dim": original_dim,
            "mapping_id": f"identity-v1:d{original_dim}",
        }
    )
    batch = IendSignalBatch(
        factual=np.stack(factual, axis=0),
        alternative=np.stack(alternative, axis=0),
        orientation=np.asarray(orientations, dtype=np.float64),
        sample_ids=tuple(sample_ids),
        factual_classes=tuple(factual_classes),
        alternative_classes=tuple(alternative_classes),
        metadata={
            "adapter": "package_factual_alternative_dataset",
            "max_samples": max_samples,
            "projection": projection_meta,
            # These are computed before any CountSketch, so E1 scale/alignment
            # conclusions do not depend on the diagnostic projection.
            "primitive_pair_geometry": primitive_rows,
        },
    )
    if not return_extraction:
        return batch
    learned = (
        LearnedIendMeasurements.from_accumulator(learned_accumulator)
        if learned_accumulator is not None
        else None
    )
    return PairedSignalExtraction(batch=batch, learned=learned)


def _training_args(trainer: Any):
    return getattr(trainer, "training_args", None) or getattr(trainer, "args", None)


def _sync_activation_scope_from_model(trainer: Any, model: Any) -> None:
    """Align trainer signal_scope with activation sites stored in a loaded checkpoint."""
    args = _training_args(trainer)
    if args is None:
        return
    gradiend = getattr(model, "gradiend", None)
    if gradiend is None:
        return
    kwargs = getattr(gradiend, "kwargs", {}) or {}
    signal_space = kwargs.get("signal_space")
    if not isinstance(signal_space, dict):
        return
    if str(signal_space.get("kind") or "").strip().lower() != "activation":
        return
    from gradiend import SignalScope

    args.signal_scope = SignalScope.from_signal_space(signal_space)


def extract_paired_signals_from_trainer(
    trainer: Any,
    *,
    split: str = "validation",
    max_size_per_group: Optional[int] = None,
    max_samples: Optional[int] = None,
    projection_dim: Optional[int] = 512,
    projection_seed: int = 0,
    use_cached_signals: bool = False,
    cache_dir: Optional[str] = None,
    include_learned: bool = False,
    checkpoint_source: Optional[str] = None,
) -> Any:
    """Use the supported trainer API to extract an aligned diagnostic batch."""
    model = trainer.get_model()
    _sync_activation_scope_from_model(trainer, model)
    raw = trainer.create_training_data(
        model,
        split=split,
        batch_size=1,
        max_size=max_size_per_group,
    )
    paired = trainer.create_gradient_training_dataset(
        raw,
        model,
        source="factual",
        target="alternative",
        cache_dir=cache_dir,
        use_cached_gradients=bool(use_cached_signals),
    )
    return collect_paired_signal_dataset(
        paired,
        max_samples=max_samples,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
        learned_iend=getattr(model, "gradiend", None) if include_learned else None,
        checkpoint_source=(
            checkpoint_source
            if checkpoint_source is not None
            else (
                str(getattr(model, "source"))
                if include_learned and getattr(model, "source", None) is not None
                else None
            )
        ),
        return_extraction=include_learned,
    )
