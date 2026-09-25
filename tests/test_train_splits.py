from __future__ import annotations

import sys
import weakref
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.config import StudyConfig
from study.deep_pipeline import _trainers_needed_after_train, _unrequested_train_rows
from study.stages import train as train_stage
from study.stages.train import (
    _ablations,
    _effective_train_splits,
    _release_train_raw_trainer,
    _requested_train_splits,
    _suite_splits,
)


def test_encode_only_train_run_does_not_retain_live_trainers():
    assert _trainers_needed_after_train({"gradiend"}) is False
    assert _trainers_needed_after_train({"gradiend", "actiend_ridge"}) is False
    for consumer in ("sae", "caa", "causal", "localization"):
        assert _trainers_needed_after_train({"gradiend", consumer}) is True


def test_release_train_raw_trainer_drops_cyclic_trainer_and_clears_cache(monkeypatch):
    class Trainer:
        pass

    trainer = Trainer()
    trainer.cycle = trainer
    trainer_ref = weakref.ref(trainer)
    raw = {"trainer": trainer, "encoder_metrics": {"correlation": 0.8}}
    del trainer

    cache_calls = []
    monkeypatch.setattr(train_stage.gc, "collect", lambda: cache_calls.append("gc"))

    assert _release_train_raw_trainer(raw) is True
    assert "trainer" not in raw
    assert raw["encoder_metrics"] == {"correlation": 0.8}
    assert cache_calls == ["gc"]

    # Run a real collection after restoring the patched collector.
    monkeypatch.undo()
    train_stage.gc.collect()
    assert trainer_ref() is None

def _cfg(*, suite_methods: dict, ablations: dict | None = None) -> StudyConfig:
    return StudyConfig(
        model_key="gpt2-small",
        task_id="induction",
        suite_id="full",
        raw={
            "suite": {"methods": suite_methods},
            "ablations": ablations or {"pair": False, "one_pole": True},
        },
    )


def test_suite_full_lists_tensors_but_protocol_skips_trainers():
    cfg = _cfg(suite_methods={"gradiend": ["none", "tensors"], "actiend": ["none", "tensors"]})
    assert _suite_splits(cfg, "gradiend") == ["none", "tensors"]
    assert _effective_train_splits(cfg, "gradiend", one_pole=True) == ["none"]
    assert _effective_train_splits(cfg, "gradiend", one_pole=False) == ["none"]
    assert _effective_train_splits(cfg, "actiend", one_pole=True) == ["none"]
    assert _effective_train_splits(cfg, "actiend", one_pole=False) == ["none"]
    assert _effective_train_splits(cfg, "actiend_pre", one_pole=False) == ["none"]


def test_requested_train_splits_filters_without_fallback():
    assert _requested_train_splits(["none", "tensors"], ["tensors"]) == ["tensors"]
    assert _requested_train_splits(["none"], ["tensors"]) == []
    assert _requested_train_splits(["none", "tensors"], None) == ["none", "tensors"]


def test_tensor_refresh_preserves_scalar_rows_from_same_family():
    previous = {
        "methods": [
            {"method": "actiend:F-M", "metrics": {"gradiend_split": "none"}},
            {
                "method": "actiend:F-M:tensors",
                "metrics": {"gradiend_split": "tensors"},
            },
            {
                "method": "actiend:F-M:tensors:F:tok_all",
                "metrics": {},
            },
            {"method": "gradiend:F-M", "metrics": {"gradiend_split": "none"}},
            {"method": "localization", "metrics": {}},
        ]
    }
    kept = _unrequested_train_rows(
        previous,
        backends={"actiend"},
        requested_splits={"tensors"},
    )
    assert [row["method"] for row in kept] == ["actiend:F-M"]


def test_smoke_drops_one_pole_only_when_pair_trainer_exists():
    """Regression for smoke mode training zero trainers on pair:false tasks.

    ``ioi``/``induction``/``key_value``/``function_composition``/``ioi_mib``/
    ``repetition``/``race_one_pole``/``religion_one_pole`` all set
    ``ablations: {pair: false, one_pole: true}`` — smoke unconditionally
    forcing ``one_pole=False`` left these tasks with neither trainer enabled,
    so ``--smoke`` produced ``methods=0`` and SAE silently fell back to the
    untrained HF backbone (confirmed 2026-08-18 on ``ioi``).
    """
    pair_and_one_pole = _cfg(
        suite_methods={}, ablations={"pair": True, "one_pole": True}
    )
    assert _ablations(pair_and_one_pole, smoke=False) == {
        "pair": True,
        "one_pole": True,
    }
    assert _ablations(pair_and_one_pole, smoke=True) == {
        "pair": True,
        "one_pole": False,
    }

    one_pole_only = _cfg(
        suite_methods={}, ablations={"pair": False, "one_pole": True}
    )
    assert _ablations(one_pole_only, smoke=False) == {
        "pair": False,
        "one_pole": True,
    }
    assert _ablations(one_pole_only, smoke=True) == {
        "pair": False,
        "one_pole": True,
    }
