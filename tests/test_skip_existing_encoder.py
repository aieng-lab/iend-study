from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.stages.train import (
    encoder_eval_from_done,
    encoder_eval_from_method_rows,
    load_encoder_eval_for_reload,
    persist_encoder_eval_in_done,
)

_ROOT = Path("runs") / "_test_skip_existing_encoder"


def _payload():
    return {
        "encoder_metrics": {"correlation": 0.93, "mean_by_class": {"1.0": 0.8}},
        "readout_metrics": {"roc_auc": 0.95},
        "per_class_readouts": {"asian": {"roc_auc": 0.95, "specificity": 1.0}},
        "per_component_readouts": {},
        "component_keys": [],
    }


def _scratch(name: str) -> Path:
    d = _ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_restore_encoder_eval_from_done():
    d = _scratch("from_done")
    try:
        (d / "done.json").write_text(
            json.dumps({"extras": {"encoder_eval": _payload()}}),
            encoding="utf-8",
        )
        blob = encoder_eval_from_done(d)
        assert blob is not None
        assert blob["encoder_metrics"]["correlation"] == 0.93
        assert load_encoder_eval_for_reload(d) is not None
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_restore_encoder_eval_from_results_rows():
    exp = _scratch("gradiend__onepole__asian")
    try:
        rows = [
            {
                "method": "gradiend:asian",
                "artifacts": {"experiment_dir": str(exp)},
                "extras": {
                    "encoder_metrics": {"correlation": 0.88},
                    "per_class_readouts": {"asian": {"roc_auc": 0.91}},
                    "readout_metrics": {"roc_auc": 0.91},
                },
            }
        ]
        blob = encoder_eval_from_method_rows(rows, exp)
        assert blob is not None
        assert blob["encoder_metrics"]["correlation"] == 0.88
        assert blob["per_class_readouts"]["asian"]["roc_auc"] == 0.91
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_restore_matches_artifact_dirname_across_hosts():
    exp = Path("/workspace/runs/gpt2-small/race_one_pole/artifacts/gradiend__onepole__asian")
    rows = [
        {
            "method": "gradiend:asian",
            "artifacts": {
                "experiment_dir": "C:/Git/gradiend-sae/runs/gpt2-small/race_one_pole/artifacts/gradiend__onepole__asian"
            },
            "extras": {
                "encoder_metrics": {"correlation": 0.5},
                "per_class_readouts": {"asian": {"roc_auc": 0.7}},
            },
        }
    ]
    blob = encoder_eval_from_method_rows(rows, exp)
    assert blob is not None
    assert blob["encoder_metrics"]["correlation"] == 0.5


def test_missing_encoder_eval_does_not_pretend_ok():
    d = _scratch("legacy_done")
    try:
        (d / "done.json").write_text(
            json.dumps({"extras": {"ablation": "one_pole"}}),
            encoding="utf-8",
        )
        assert encoder_eval_from_done(d) is None
        assert load_encoder_eval_for_reload(d, previous_methods=[]) is None
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_persist_encoder_eval_merges_legacy_done():
    d = _scratch("persist")
    try:
        (d / "done.json").write_text(
            json.dumps({"config_hash": "abc", "extras": {"ablation": "one_pole"}}),
            encoding="utf-8",
        )
        persist_encoder_eval_in_done(d, _payload())
        blob = encoder_eval_from_done(d)
        assert blob is not None
        meta = json.loads((d / "done.json").read_text(encoding="utf-8"))
        assert meta["config_hash"] == "abc"
        assert meta["extras"]["ablation"] == "one_pole"
        assert blob["per_class_readouts"]["asian"]["roc_auc"] == 0.95
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_required_encoder_eval_version_invalidates_only_stale_metrics():
    d = _scratch("versioned")
    try:
        (d / "done.json").write_text(
            json.dumps({"extras": {"encoder_eval": _payload()}}),
            encoding="utf-8",
        )
        # Legacy metrics remain usable for methods without a protocol version,
        # but CGA's corrected objective must not restore them.
        assert encoder_eval_from_done(d) is not None
        assert encoder_eval_from_done(d, required_version=2) is None
        assert load_encoder_eval_for_reload(
            d,
            previous_methods=[
                {
                    "artifacts": {"experiment_dir": str(d)},
                    "extras": _payload(),
                }
            ],
            required_version=2,
        ) is None

        persist_encoder_eval_in_done(d, _payload(), version=2)
        assert encoder_eval_from_done(d, required_version=2) is not None
        meta = json.loads((d / "done.json").read_text(encoding="utf-8"))
        assert meta["extras"]["encoder_eval_version"] == 2
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
