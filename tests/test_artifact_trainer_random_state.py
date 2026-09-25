"""Regression tests for deterministic counterfactual construction."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("random_state", [None, 37])
def test_artifact_trainer_forwards_optional_random_state(monkeypatch, tmp_path: Path, random_state):
    import study.stages.train as train

    captured: dict[str, object] = {}

    class CapturingConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class CapturingTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    leaf = types.ModuleType("gradiend.trainer.text.prediction.trainer")
    leaf.TextPredictionConfig = CapturingConfig
    leaf.TextPredictionTrainer = CapturingTrainer
    monkeypatch.setitem(sys.modules, "gradiend.trainer.text.prediction.trainer", leaf)
    monkeypatch.setattr(train, "build_training_arguments", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train, "_merged_one_pole_frame", lambda bundle: None)
    monkeypatch.setattr(train, "resolve_mask_placeholder", lambda bundle, raw: "[MASK]")

    cfg = SimpleNamespace(
        model_key="gpt2-small",
        hf_model="gpt2",
        task_id="fixture",
        raw={"model": {}},
    )
    bundle = SimpleNamespace(
        classes=["a", "b"],
        data_per_class={"a": [], "b": []},
        neutrals=None,
        excluded_words=[],
        class_merge_map=None,
        class_merge_transition_groups=None,
    )

    train._build_trainer_for_artifact(
        backend="actiend",
        cfg=cfg,
        bundle=bundle,
        target_classes=["a", "b"],
        experiment_dir=tmp_path,
        split_mode="none",
        shared={},
        smoke=False,
        all_classes=["a", "b"],
        random_state=random_state,
    )

    if random_state is None:
        assert "random_state" not in captured
    else:
        assert captured["random_state"] == random_state
