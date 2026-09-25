"""Split-honest detection-site selection shared by execution and reporting."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from suitability import detection_score
from results_schema import _has_encoder_metrics


def validation_detection(metrics: Mapping[str, Any]) -> Optional[float]:
    """Return a candidate's validation Detection score (with legacy fallback)."""
    val = metrics.get("val_readout")
    if not isinstance(val, Mapping):
        score = metrics.get("val_encoding_E")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        return None
    score = detection_score(val)
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return float(score)


def lock_by_validation_detection(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select one candidate strictly from validation Detection.

    Every candidate must have a score.  A partial candidate pool is not a
    selection problem with a convenient fallback; it is an incomplete result.
    """
    candidates = [dict(row) for row in rows]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    scored: list[Tuple[float, Dict[str, Any]]] = []
    for row in candidates:
        value = validation_detection(row.get("metrics") or {})
        if value is None:
            return None
        scored.append((value, row))

    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for score, row in scored:
        if score > best_score + 1e-15:
            best, best_score = row, score
        elif abs(score - best_score) <= 1e-15 and best is not None:
            # Match the report lock's historical tie break: prefer the row
            # whose encoder metric payload is complete.
            if _has_encoder_metrics(row.get("metrics") or {}) and not _has_encoder_metrics(
                best.get("metrics") or {}
            ):
                best = row
    return best
