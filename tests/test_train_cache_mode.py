from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.stages.train import _artifact_train_complete

_ROOT = Path("runs") / "_test_train_cache_mode"


def _scratch(name: str) -> Path:
    d = _ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_done_with_checkpoint(d: Path, *, config_hash: str) -> None:
    (d / "done.json").write_text(
        json.dumps({"config_hash": config_hash}), encoding="utf-8"
    )
    (d / "model.safetensors").write_text("stub", encoding="utf-8")


def test_always_mode_reuses_artifact_despite_hash_mismatch():
    d = _scratch("always_mismatch")
    try:
        _write_done_with_checkpoint(d, config_hash="old_hash")
        assert _artifact_train_complete(d, config_hash="new_hash", cache_mode="always") is True
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_hash_match_mode_forces_retrain_on_mismatch():
    d = _scratch("strict_mismatch")
    try:
        _write_done_with_checkpoint(d, config_hash="old_hash")
        assert (
            _artifact_train_complete(d, config_hash="new_hash", cache_mode="hash_match")
            is False
        )
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_hash_match_mode_reuses_artifact_when_hash_matches():
    d = _scratch("strict_match")
    try:
        _write_done_with_checkpoint(d, config_hash="same_hash")
        assert (
            _artifact_train_complete(d, config_hash="same_hash", cache_mode="hash_match")
            is True
        )
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_default_cache_mode_is_always():
    d = _scratch("default_mismatch")
    try:
        _write_done_with_checkpoint(d, config_hash="old_hash")
        assert _artifact_train_complete(d, config_hash="new_hash") is True
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
