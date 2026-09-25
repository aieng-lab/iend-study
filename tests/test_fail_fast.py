from pathlib import Path

import pytest
import pandas as pd

from study.config import StudyConfig
from study.stages import train
from study.tasks import TaskBundle


def test_fail_fast_reraises_the_first_training_ablation_error(monkeypatch, tmp_path):
    cfg = StudyConfig(
        model_key="test-model",
        task_id="test-task",
        suite_id="test-suite",
        raw={
            "training": {"source": "alternative"},
            "ablations": {"pair": True, "one_pole": False},
            "suite": {"methods": {"gradiend": ["none"]}},
            "model": {"hf_model": "test-model"},
        },
    )
    bundle = TaskBundle(
        task_id="test-task",
        classes=["A", "B"],
        data_per_class={},
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame({"text": []}),
    )
    monkeypatch.setattr(type(cfg), "output_dir", property(lambda _self: tmp_path))
    monkeypatch.setattr(
        train,
        "shared_training_kwargs",
        lambda *_args, **_kwargs: {"source": "alternative", "learning_rate": 1e-5},
    )

    def fail_train(**_kwargs):
        raise RuntimeError("intentional ablation failure")

    monkeypatch.setattr(train, "_train_once", fail_train)

    with pytest.raises(RuntimeError, match="intentional ablation failure"):
        train.run_train_stage(
            cfg,
            bundle,
            enabled_backends=["gradiend"],
            fail_fast=True,
        )


def test_fail_fast_is_plumbed_through_causal_stage_and_study():
    root = Path(__file__).resolve().parents[1]
    stage = (root / "study" / "stages" / "causal.py").read_text(encoding="utf-8")
    study = (root / "causal_study.py").read_text(encoding="utf-8")
    pipeline = (root / "study" / "deep_pipeline.py").read_text(encoding="utf-8")

    assert "fail_fast: bool = False" in stage
    assert "fail_fast=fail_fast" in stage
    assert "if fail_fast:\n            raise" in stage
    assert "fail_fast: bool = False" in study
    # Each per-method causal catch must propagate in early-fail mode instead
    # of producing a partial result and continuing the worker.
    lines = study.splitlines()
    # Some catches checkpoint already-completed causal sweeps before raising.
    # Accept that cleanup statement between the guard and the re-raise instead
    # of requiring ``raise`` to be the immediately following source line.
    reraises = sum(
        line.strip() == "if fail_fast:"
        and any(next_line.strip() == "raise" for next_line in lines[index + 1 : index + 4])
        for index, line in enumerate(lines[:-1])
    )
    assert reraises >= 8
    assert "def record_stage_errors" in pipeline
    # Training must release each completed ablation; causal rehydrates its
    # trainer wrappers only after the train/encode lifecycle boundary.
    assert "retain_trainers = False" in pipeline
    assert "causal: reloading completed trainer artifacts" in pipeline
    compact = "".join(pipeline.split())
    for call in (
        "run_sae_stage(cfg,bundle,train_raw_none,fail_fast=fail_fast",
        "run_caa_stage(cfg,bundle,train_raw_none,fail_fast=fail_fast",
        "sae_raw,fail_fast=fail_fast",
    ):
        assert call in compact
