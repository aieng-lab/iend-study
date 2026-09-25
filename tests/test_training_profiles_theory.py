from __future__ import annotations

from study.training_profiles import build_training_arguments


def test_study_encoder_bootstrap_control_is_not_forwarded_to_training_arguments(tmp_path):
    args = build_training_arguments(
        "gradiend",
        experiment_dir=str(tmp_path / "run"),
        shared={"learning_rate": 1.0e-5, "encoder_bootstrap_auc": 5},
    )
    assert not hasattr(args, "encoder_bootstrap_auc")
