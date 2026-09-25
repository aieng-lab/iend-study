"""CGA-only model settings must not leak into other training backends."""

from study.config import load_study_config
from study.training_profiles import build_training_arguments, shared_training_kwargs


def test_27b_cga_pre_prune_setting_does_not_leak_into_actiend_arguments():
    cfg = load_study_config(model="gemma-3-27b-pt", task="gender_en", suite="core")
    assert shared_training_kwargs(cfg, backend="cga")["cga_pre_prune"] is True
    shared = shared_training_kwargs(cfg, backend="actiend")
    assert "cga_pre_prune" not in shared

    args = build_training_arguments(
        "actiend",
        experiment_dir="/tmp/actiend-27b",
        shared=shared,
        split_mode="none",
    )

    assert not hasattr(args, "cga_pre_prune")


def test_27b_cga_pre_prune_setting_is_still_consumed_by_cga_profile():
    cfg = load_study_config(model="gemma-3-27b-pt", task="gender_en", suite="core")
    args = build_training_arguments(
        "cga",
        experiment_dir="/tmp/cga-27b",
        shared=shared_training_kwargs(cfg, backend="cga"),
        split_mode="none",
    )

    assert args.pre_prune_config is not None
    assert args.pre_prune_config.topk == 0.01
    assert args.metadata["cga_coordinate_projection"] == "pre_prune"
