"""JSON-safe conversion for study payloads (numpy/torch/path/dataclass)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, set):
        return sorted(json_ready(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(obj) and not isinstance(obj, type):
            return json_ready(asdict(obj))
    except Exception:
        pass
    return obj
