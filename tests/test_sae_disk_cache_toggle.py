from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.stages.sae import _configure_poc_globals


class _FakePoc:
    """Stub for study/sae_engine.py -- only the globals this
    function touches, so this test needs no torch/gradiend import."""

    def apply_study_model(self, model_key):
        pass


def _cfg(training: dict) -> SimpleNamespace:
    return SimpleNamespace(
        model_key="gpt2-small",
        output_dir=Path("runs/_test_sae_disk_cache"),
        training=training,
        raw={},
    )


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(classes=["F", "M"], neutrals=None, excluded_words=None)


def test_sae_disk_cache_defaults_to_enabled_when_unset():
    poc = _FakePoc()
    _configure_poc_globals(poc, _cfg({}), _bundle())
    assert poc.USE_CACHE is True


def test_sae_disk_cache_can_be_disabled():
    poc = _FakePoc()
    _configure_poc_globals(poc, _cfg({"sae_disk_cache": False}), _bundle())
    assert poc.USE_CACHE is False


def test_sae_disk_cache_explicit_true_is_a_no_op():
    poc = _FakePoc()
    _configure_poc_globals(poc, _cfg({"sae_disk_cache": True}), _bundle())
    assert poc.USE_CACHE is True
