from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.resume_policy import (
    artifact_has_successful_result,
    completed_result_matches_config,
)
from study.config import StudyConfig


def test_config_hash_ignores_operational_resume_switches_but_keeps_suite():
    base = {"suite": {"id": "full"}, "training": {"max_steps": 10}}
    fresh = StudyConfig("gpt2-small", "task", "full", {**base, "skip_existing": False})
    resume = StudyConfig("gpt2-small", "task", "full", {**base, "skip_existing": True})
    full_plus = StudyConfig(
        "gpt2-small",
        "task",
        "full_plus",
        {**base, "suite": {"id": "full_plus"}, "skip_existing": True},
    )
    assert fresh.config_hash() == resume.config_hash()
    assert fresh.config_hash() != full_plus.config_hash()


def test_config_hash_ignores_targeted_train_split_filter():
    base = StudyConfig(
        model_key="gpt2-small",
        task_id="gender_en",
        suite_id="full_plus",
        raw={"suite": {"id": "full_plus"}},
    )
    targeted = StudyConfig(
        model_key="gpt2-small",
        task_id="gender_en",
        suite_id="full_plus",
        raw={
            "suite": {"id": "full_plus"},
            "train_splits": ["tensors"],
            "cli": {"train_splits": ["tensors"]},
        },
    )
    assert base.config_hash() == targeted.config_hash()


def test_ok_result_is_not_current_when_requested_suite_config_changed():
    prior = {"status": "ok", "config_hash": "full-config"}
    assert completed_result_matches_config(prior, "full-config")
    assert not completed_result_matches_config(prior, "full-plus-config")


def test_new_artifact_without_a_result_row_is_not_reused_after_config_change():
    prior = [
        {
            "status": "ok",
            "artifacts": {"experiment_dir": "/runs/task/artifacts/actiend__pair__a-b"},
        }
    ]
    assert artifact_has_successful_result(prior, "actiend__pair__a-b")
    assert not artifact_has_successful_result(prior, "actiend__pair__a-b__tensors")


def test_error_result_does_not_make_an_artifact_current():
    prior = [
        {
            "status": "error",
            "artifacts": {"experiment_dir": "/runs/task/artifacts/actiend__pair__a-b__tensors"},
        }
    ]
    assert not artifact_has_successful_result(prior, "actiend__pair__a-b__tensors")
