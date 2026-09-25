"""Pure resume-policy helpers shared by launch and training stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def completed_result_matches_config(payload: Any, config_hash: str) -> bool:
    """Whether a result is complete for this exact resolved study config."""
    return (
        isinstance(payload, Mapping)
        and payload.get("status") == "ok"
        and str(payload.get("config_hash") or "") == str(config_hash)
    )


def artifact_has_successful_result(
    methods: Sequence[Mapping[str, Any]] | None, artifact_name: str
) -> bool:
    """Whether a prior method row successfully represents an artifact folder."""
    for row in methods or ():
        artifacts = row.get("artifacts") or {}
        experiment_dir = artifacts.get("experiment_dir") if isinstance(artifacts, Mapping) else None
        if Path(str(experiment_dir or "")).name != str(artifact_name):
            continue
        if str(row.get("status") or "").lower() != "error":
            return True
    return False
