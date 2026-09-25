from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.config import load_study_config
from study.stages.train import _effective_train_splits, _rows_from_pair_train


def test_core_and_full_never_train_actiend_by_tensor():
    for suite in ("core", "full"):
        cfg = load_study_config(model="gpt2-small", task="ioi", suite=suite)
        assert "tensors" not in _effective_train_splits(cfg, "actiend", one_pole=False)


def test_full_plus_allows_actiend_by_tensor():
    cfg = load_study_config(model="gpt2-small", task="ioi", suite="full_plus")
    assert "tensors" in _effective_train_splits(cfg, "actiend", one_pole=False)


def test_full_plus_allows_gradiend_by_tensor():
    cfg = load_study_config(model="gpt2-small", task="ioi", suite="full_plus")
    assert "tensors" in _effective_train_splits(cfg, "gradiend", one_pole=False)


def test_full_plus_still_skips_actiend_pre_by_tensor():
    cfg = load_study_config(model="gpt2-small", task="ioi", suite="full_plus")
    assert "tensors" not in _effective_train_splits(cfg, "actiend_pre", one_pole=False)


def test_one_pole_never_trains_by_tensor_even_under_full_plus():
    cfg = load_study_config(model="gpt2-small", task="ioi", suite="full_plus")
    assert "tensors" not in _effective_train_splits(cfg, "actiend", one_pole=True)


def test_rows_from_pair_train_handles_tensors_split_mode():
    # Regression: _rows_from_pair_train used TENSORS_SPLIT_TAG without
    # importing it, so full_plus's by-tensor ACTIEND path raised
    # NameError before emitting any method rows (see CLAUDE.md).
    cfg = load_study_config(model="gpt2-small", task="ioi", suite="full_plus")
    raw = {
        "split_mode": "tensors",
        "encoder_metrics": {"correlation": 0.5},
        "readout_metrics": {},
        "per_class_readouts": {},
    }
    rows = _rows_from_pair_train(backend="actiend", a="F", b="M", raw=raw, cfg=cfg)
    assert any(row["metrics"].get("component_part") == "tensors" for row in rows)
