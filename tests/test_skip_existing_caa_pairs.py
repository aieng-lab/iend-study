"""Gate 1 must re-enter an ok task that lacks requested two-pole CAA.

The two-pole CAA regime was enabled by a code fix (CAA-encode reuse), invisible to
config_hash, so an ``ok`` task trained before it would be skipped forever and never
gain ``caa:A-B:*`` rows. ``_should_skip_existing(expects_caa_pairs=True)`` closes
that gap -- automatically, for any newly scheduled job, without a manual reset.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from study.runner import _payload_has_requested_method_families, _should_skip_existing


def _write(payload: dict) -> Path:
    d = Path(tempfile.mkdtemp(prefix="skipcaa_"))
    (d / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


OK_NO_PAIR = {
    "status": "ok",
    "config_hash": "h1",
    "methods": [{"method": "caa:F:act_prediction"}],  # one-pole only
}
OK_WITH_PAIR = {
    "status": "ok",
    "config_hash": "h1",
    "methods": [
        {"method": "caa:F:act_prediction"},
        {"method": "caa:F-M:F:act_prediction"},  # two-pole present
    ],
}


def test_ok_task_without_pairs_reenters_when_pairs_expected():
    d = _write(OK_NO_PAIR)
    skip, changed = _should_skip_existing(
        d, skip_existing=True, config_hash="h1", expects_caa_pairs=True
    )
    assert skip is False and changed is False  # re-enter to fill pairs


def test_ok_task_with_pairs_still_skips():
    d = _write(OK_WITH_PAIR)
    skip, _ = _should_skip_existing(
        d, skip_existing=True, config_hash="h1", expects_caa_pairs=True
    )
    assert skip is True  # already has pairs -> self-limiting, no re-entry


def test_no_perpetual_loop_when_run_does_not_compute_caa():
    # e.g. a --methods gradiend run: expects_caa_pairs=False, so an ok task
    # missing CAA pairs must NOT re-enter (it would loop forever otherwise).
    d = _write(OK_NO_PAIR)
    skip, _ = _should_skip_existing(
        d, skip_existing=True, config_hash="h1", expects_caa_pairs=False
    )
    assert skip is True


def test_pair_detection_reads_raw_caa_metrics_too():
    payload = {
        "status": "ok",
        "config_hash": "h1",
        "methods": [],
        "raw": {"caa": {"method_metrics": {"caa:F-M:F:act_prediction": {}}}},
    }
    d = _write(payload)
    skip, _ = _should_skip_existing(
        d, skip_existing=True, config_hash="h1", expects_caa_pairs=True
    )
    assert skip is True  # pair present in raw.caa -> skip


def test_config_change_still_takes_precedence():
    d = _write(OK_WITH_PAIR)
    skip, changed = _should_skip_existing(
        d, skip_existing=True, config_hash="DIFFERENT", expects_caa_pairs=True
    )
    assert skip is False and changed is True


def test_method_repair_detects_caga_causal_only_one_pole_rows():
    payload = {
        "config": {
            "claim_classes": ["F", "M"],
            "ablations": {"one_pole": True, "pair": True},
        },
        "methods": [
            {
                "method": "caga:F-M",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.9},
            },
            {"method": "caga:F", "status": "ok", "metrics": {"readout_kind": "causal_only"}},
            {"method": "caga:M", "status": "ok", "metrics": {"readout_kind": "causal_only"}},
        ],
    }
    assert not _payload_has_requested_method_families(payload, {"caga"})

    payload["methods"][1]["metrics"] = {"roc_auc_neutral": 0.8}
    payload["methods"][2]["metrics"] = {"roc_auc_neutral": 0.85}
    assert _payload_has_requested_method_families(payload, {"caga"})
