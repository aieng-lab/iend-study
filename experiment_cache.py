"""
Study-managed experimental cache (not GRADIEND evaluate_* / TrainingArguments.use_cache).

Package ``evaluate_decoder`` defaults to one ``decoder_grid_cache.json`` per
experiment_dir; none-split ACTIEND token/gate ablations overwrite that file
(see PLANNING.md §8.6). Study call sites hardcode ``use_cache=False`` on
``evaluate_*`` and resume via this module / ``--skip-existing`` instead.

Skip a stage when ``artifacts/<stage>/done.json`` exists and its config_hash
matches the current stage hash. Delete the artifact dir or pass ``force`` to recompute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence




def stable_hash(payload: Any) -> str:
    """SHA1 over canonical JSON (sorted keys)."""
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


@dataclass
class ExperimentCache:
    """File-existence + config-hash resume for study stages."""

    root: Path
    enabled: bool = True
    force_stages: Sequence[str] = field(default_factory=tuple)
    force_all: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.artifacts = self.root / "artifacts"
        self._force = {str(s) for s in self.force_stages}

    def stage_dir(self, stage: str) -> Path:
        return self.artifacts / str(stage)

    def done_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "done.json"

    def should_skip(self, stage: str, config_hash: str) -> bool:
        if not self.enabled or self.force_all or stage in self._force:
            return False
        path = self.done_path(stage)
        if not path.is_file():
            return False
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return str(meta.get("config_hash") or "") == str(config_hash)

    def load_done(self, stage: str) -> Optional[Dict[str, Any]]:
        path = self.done_path(stage)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def mark_done(
        self,
        stage: str,
        *,
        config_hash: str,
        paths: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Path:
        stage_dir = self.stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "stage": stage,
            "config_hash": config_hash,
            "paths": paths or {},
        }
        if extras:
            payload["extras"] = extras
        path = self.done_path(stage)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._update_manifest(stage, config_hash)
        return path

    def _update_manifest(self, stage: str, config_hash: str) -> None:
        manifest_path = self.artifacts / "manifest.json"
        manifest: Dict[str, Any] = {"stages": {}}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {"stages": {}}
        stages = dict(manifest.get("stages") or {})
        stages[stage] = {"config_hash": config_hash}
        manifest["stages"] = stages
        self.artifacts.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def write_json(self, stage: str, name: str, payload: Any) -> Path:
        stage_dir = self.stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def read_json(self, stage: str, name: str) -> Optional[Any]:
        path = self.stage_dir(stage) / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None








