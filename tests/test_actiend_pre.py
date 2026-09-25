"""actiend_pre method family (mixed-site) mirrors sae_pre naming."""

from __future__ import annotations

from study.config import list_tasks, load_study_config
from study.method_ids import artifact_dirname, onepole_feature_id, pair_feature_id
from study.results_merge import method_family, should_replace_method
from study.training_profiles import build_training_arguments, is_actiend_backend, shared_training_kwargs


def test_list_tasks_hides_gender_en_pre():
    tasks = list_tasks()
    assert "gender_en" in tasks
    assert "gender_en_pre" not in tasks
    assert "gender_en_pre" in list_tasks(include_hidden=True)


def test_gender_en_pre_still_loadable_for_legacy():
    cfg = load_study_config(model="gpt2-small", task="gender_en_pre", suite="core")
    assert cfg.task_id == "gender_en_pre"
    assert cfg.training.get("activation_site") == "pre_prediction"


def test_actiend_pre_ids_and_family():
    assert onepole_feature_id("actiend_pre", "M") == "actiend_pre:M"
    assert pair_feature_id("actiend_pre", "F", "M", "M") == "actiend_pre:F-M:M"
    assert artifact_dirname("actiend_pre", kind="onepole", key="M").startswith("actiend_pre__")
    assert method_family("actiend_pre:M") == "actiend_pre"
    assert method_family("actiend:M") == "actiend"
    # Opt-in: not replaced when only actiend is enabled.
    assert not should_replace_method("actiend_pre:M", enabled={"actiend"})
    assert should_replace_method("actiend_pre:M", enabled={"actiend_pre"})
    assert not should_replace_method("actiend_pre:M", enabled={"gradiend"})


def test_actiend_pre_off_by_default():
    from study.stages.train import _actiend_pre_enabled

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    assert cfg.training.get("actiend_pre") is False
    assert _actiend_pre_enabled(cfg) is False


def test_actiend_pre_training_forces_mixed_sites():
    assert is_actiend_backend("actiend_pre")
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    shared = shared_training_kwargs(cfg, backend="actiend")
    # Even if task is filled-site, actiend_pre ignores task YAML sites.
    shared["activation_site"] = "prediction"
    args = build_training_arguments(
        "actiend_pre",
        experiment_dir="tmp_actiend_pre",
        shared=shared,
        one_pole=True,
    )
    meta = args.metadata or {}
    assert meta.get("actiend_source_site") == "pre_prediction"
    assert meta.get("actiend_target_site") == "prediction"
    assert meta.get("actiend_mixed_site") is True
    assert meta.get("backend") == "actiend_pre"
