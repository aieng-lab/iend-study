"""Single source of truth for every Bayesian ``autorank`` call in the analysis."""

from __future__ import annotations

from typing import Any, Dict

ALPHA = 0.05
BAYESIAN_SAMPLES = 50_000
BAYESIAN_RANDOM_STATE = 42
ROPE_MODE = "absolute"
FORCE_MODE = "nonparametric"
ORDER = "descending"

DEFAULT_ROPES: Dict[str, float] = {
    "detection_score": 0.01,
    "detection_light_score": 0.01,
    "intervention_score": 0.01,
}


def bayesian_autorank_kwargs(
    metric: str,
    *,
    alpha: float = ALPHA,
    rope: float | None = None,
    nsamples: int = BAYESIAN_SAMPLES,
    random_state: int = BAYESIAN_RANDOM_STATE,
) -> Dict[str, Any]:
    return {
        "alpha": float(alpha),
        "order": ORDER,
        "approach": "bayesian",
        "rope": float(DEFAULT_ROPES[metric] if rope is None else rope),
        "rope_mode": ROPE_MODE,
        "nsamples": int(nsamples),
        "force_mode": FORCE_MODE,
        "random_state": int(random_state),
        "verbose": False,
    }
