from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from study.stages.sae import (
    _load_sae_encode_snapshot,
    _sae_rows_are_substantive,
    _save_sae_encode_snapshot,
)


def _cfg(tmp: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=tmp,
        model_key="gpt2-small",
        task_id="ioi",
        config_hash=lambda: "testhash",
    )


def test_sae_rows_substantive_vs_stub():
    assert not _sae_rows_are_substantive(
        [{"method": "sae", "status": "error", "error": "boom"}]
    )
    assert _sae_rows_are_substantive(
        [{"method": "sae:IO:kstar", "status": "ok", "metrics": {"roc_auc_neutral": 0.9}}]
    )


def test_sae_encode_snapshot_roundtrip():
    root = Path(__file__).resolve().parents[1] / ".test_scratch_sae_snapshot"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        cfg = _cfg(root)
        rows = [
            {
                "method": "sae:IO:k1",
                "model": "gpt2-small",
                "task": "ioi",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.91, "class_exclusivity": 0.8},
                "artifacts": {},
            },
            {
                "method": "sae:IO:kstar",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.95},
                "artifacts": {},
            },
        ]
        path = _save_sae_encode_snapshot(cfg, rows, raw_meta={"sae_top_k": 128})
        assert path is not None and path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["n_methods"] == 2
        assert (root / "artifacts" / "sae" / "done.json").is_file()

        loaded = _load_sae_encode_snapshot(cfg)
        assert len(loaded) == 2
        by = {r["method"]: r for r in loaded}
        assert abs(by["sae:IO:k1"]["metrics"]["roc_auc_neutral"] - 0.91) < 1e-9
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sae_snapshot_not_written_for_stub_only():
    root = Path(__file__).resolve().parents[1] / ".test_scratch_sae_snapshot_stub"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        cfg = _cfg(root)
        assert (
            _save_sae_encode_snapshot(
                cfg, [{"method": "sae", "status": "error", "error": "x"}]
            )
            is None
        )
        assert _load_sae_encode_snapshot(cfg) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
