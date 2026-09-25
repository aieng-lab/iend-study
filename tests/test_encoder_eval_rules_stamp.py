"""The rules-version stamp gates *encode* reuse but must not break the causal reload.

Blobs written before the validation-frozen Spec_n/Excl fix carry no
``encoder_eval_rules_version``. The encode path must recompute them; the causal
reload path (which reads the saved *validation* readouts, unchanged by the fix)
must keep reading them at their existing ``encoder_eval_version``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from results_schema import ENCODER_EVAL_RULES_VERSION, encoder_eval_rules_current
from study.stages.train import (
    _encoder_eval_rules_stamp,
    encoder_eval_from_done,
    load_encoder_eval_for_reload,
    persist_encoder_eval_in_done,
)

BLOB = {
    "encoder_metrics": {},
    "readout_metrics": {},
    "per_class_readouts": {"F": {"roc_auc_neutral": 0.9, "val_readout": {"x": 1}}},
    "per_component_readouts": {},
    "component_keys": [],
}


def _write_done(directory: Path, **extras) -> None:
    (directory / "done.json").write_text(
        json.dumps({"extras": {"encoder_eval": BLOB, **extras}}), encoding="utf-8"
    )


def test_stamp_helper_treats_missing_or_old_as_stale():
    assert not encoder_eval_rules_current(None)
    assert not encoder_eval_rules_current(1)
    assert not encoder_eval_rules_current("garbage")
    assert encoder_eval_rules_current(ENCODER_EVAL_RULES_VERSION)
    assert encoder_eval_rules_current(ENCODER_EVAL_RULES_VERSION + 1)


def test_unstamped_blob_is_rejected_by_encode_guard_but_still_readable():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_done(d, encoder_eval_version=3)
        assert encoder_eval_from_done(d, required_version=3) is not None
        assert encoder_eval_from_done(d, required_version=3, require_current_rules=True) is None


def test_causal_reload_path_still_reads_a_legacy_blob():
    # Mirrors the call at the causal reload: exact version, NO rules guard. Bumping
    # ``*_ENCODER_EVAL_VERSION`` instead would have returned {} here and marked the
    # layer representation unavailable for every not-yet-migrated artifact.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_done(d, encoder_eval_version=3)
        got = load_encoder_eval_for_reload(d, required_version=3)
        assert got is not None
        assert got["per_class_readouts"]["F"]["val_readout"] == {"x": 1}


def test_guarded_load_never_falls_back_to_unstamped_results_rows():
    rows = [
        {
            "artifacts": {"experiment_dir": "cga__pair__F-M"},
            "metrics": {},
            "extras": {
                "encoder_metrics": {"correlation": 0.5},
                "per_class_readouts": {"F": {"roc_auc_neutral": 0.9}},
            },
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "cga__pair__F-M"
        d.mkdir()
        # Unguarded (legacy) load can rebuild from rows; the guarded one must not,
        # since rows carry no stamp and so cannot prove validation-frozen rules.
        assert load_encoder_eval_for_reload(d, previous_methods=rows) is not None
        assert (
            load_encoder_eval_for_reload(d, previous_methods=rows, require_current_rules=True)
            is None
        )


def test_persist_stamps_the_blob_so_the_next_guarded_load_reuses_it():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_done(d, encoder_eval_version=3)
        assert encoder_eval_from_done(d, required_version=3, require_current_rules=True) is None
        persist_encoder_eval_in_done(d, BLOB, version=3)
        assert encoder_eval_from_done(d, required_version=3, require_current_rules=True) is not None
        extras = json.loads((d / "done.json").read_text(encoding="utf-8"))["extras"]
        assert extras["encoder_eval_rules_version"] == ENCODER_EVAL_RULES_VERSION
        assert extras["encoder_eval_version"] == 3


def _readout(source):
    return {"neutral_youden_threshold_source": source, "roc_auc_neutral": 0.9}


def test_stamp_is_withheld_when_any_readout_used_a_test_fit_rule():
    frozen = {"per_class_readouts": {"F": _readout("provided")}, "per_component_readouts": {}}
    assert _encoder_eval_rules_stamp(frozen) == {
        "encoder_eval_rules_version": ENCODER_EVAL_RULES_VERSION
    }
    oracle_class = {"per_class_readouts": {"F": _readout("provided"), "M": _readout("eval")}}
    assert _encoder_eval_rules_stamp(oracle_class) == {}
    oracle_layer = {
        "per_class_readouts": {"F": _readout("provided")},
        "per_component_readouts": {"L3": {"F": _readout("eval")}},
    }
    assert _encoder_eval_rules_stamp(oracle_layer) == {}


def test_persist_clears_a_stale_stamp_for_an_unfrozen_blob():
    oracle = dict(BLOB, per_class_readouts={"F": _readout("eval")})
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_done(d, encoder_eval_version=3, encoder_eval_rules_version=ENCODER_EVAL_RULES_VERSION)
        persist_encoder_eval_in_done(d, oracle, version=3)
        extras = json.loads((d / "done.json").read_text(encoding="utf-8"))["extras"]
        assert "encoder_eval_rules_version" not in extras  # would otherwise be trusted
        assert encoder_eval_from_done(d, required_version=3, require_current_rules=True) is None
