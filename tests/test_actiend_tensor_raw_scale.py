from __future__ import annotations

from study.config import load_study_config
from study.stages.train import _train_artifact_hash
from study.training_profiles import build_training_arguments, shared_training_kwargs


def _args(tmp_path, split_mode: str):
    return build_training_arguments(
        "actiend",
        experiment_dir=str(tmp_path / split_mode),
        split_mode=split_mode,
        shared={
            "learning_rate": 1.0e-5,
            "activation_site": "prediction",
        },
    )


def test_actiend_by_tensor_uses_raw_activations(tmp_path):
    args = _args(tmp_path, "tensors")

    assert "scale" not in args.signal.options
    assert args.metadata["actiend_signal_scale"] == {"scale": "raw"}


def test_actiend_none_retains_running_rms(tmp_path):
    args = _args(tmp_path, "none")

    assert args.signal.options["scale"] == "running_rms"
    assert args.signal.options["scale_reduce"] == "per_site"
    assert args.signal.options["scale_momentum"] == 0.0
    assert args.metadata["actiend_signal_scale"] == {
        "scale": "running_rms",
        "scale_reduce": "per_site",
        "scale_momentum": 0.0,
        "scale_eps": 1e-6,
    }


def test_actiend_scale_policy_is_part_of_artifact_identity(monkeypatch):
    from study import training_profiles

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full_plus")
    shared = shared_training_kwargs(cfg, backend="actiend")

    raw_hash = _train_artifact_hash(
        cfg,
        backend="actiend",
        shared=shared,
        split_mode="tensors",
        target_classes=["F", "M"],
    )

    monkeypatch.setattr(
        training_profiles,
        "actiend_signal_scale_policy",
        lambda _split: {
            "scale": "running_rms",
            "scale_reduce": "per_site",
            "scale_momentum": 0.0,
            "scale_eps": 1e-6,
        },
    )
    scaled_hash = _train_artifact_hash(
        cfg,
        backend="actiend",
        shared=shared,
        split_mode="tensors",
        target_classes=["F", "M"],
    )

    assert raw_hash != scaled_hash
