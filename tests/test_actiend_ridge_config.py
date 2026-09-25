"""``actiend_ridge`` is a suite config flag, not hardcoded off ACTIEND."""
import pytest

from study.config import load_study_config
from study.deep_pipeline import _default_enabled


def _enabled(suite, cli=None):
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite=suite)
    return _default_enabled(cfg, cli or {})


def test_core_does_not_run_ridge():
    enabled = _enabled("core")
    assert "actiend" in enabled
    assert "actiend_ridge" not in enabled


@pytest.mark.parametrize("suite", ["full", "full_plus"])
def test_full_suites_run_ridge(suite):
    assert "actiend_ridge" in _enabled(suite)


def test_ridge_needs_actiend_even_when_suite_lists_it():
    assert "actiend_ridge" not in _enabled("full", {"methods": ["gradiend"]})


def test_explicit_methods_flag_can_still_request_ridge_on_core():
    assert "actiend_ridge" in _enabled("core", {"methods": ["actiend_ridge"]})


def test_task_specs_mirror_suite_flag():
    from analysis.task_specs import _enabled_from_yaml, _load_suite_cfg

    assert "actiend_ridge" not in _enabled_from_yaml({}, _load_suite_cfg("core"))
    assert "actiend_ridge" in _enabled_from_yaml({}, _load_suite_cfg("full"))
