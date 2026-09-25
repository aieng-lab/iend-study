"""CAA site selection ranks on validation Detection, and refuses to guess.

CAA's candidate set is concat / per-layer. Without a validation
score there is nothing to select on, so the group is NaN: substituting a fixed
site would report an unselected readout as though it had been chosen.
"""

from __future__ import annotations

from analysis.method_groups import _lock_caa_site


def _row(method, *, val_det=None, auc=0.9):
    metrics = {
        "roc_auc_neutral": auc,
        "roc_auc_other": auc,
        "class_exclusivity": auc,
    }
    if val_det is not None:
        # Detection is min over its inputs, so a flat readout scores val_det.
        metrics["val_readout"] = {
            "roc_auc_neutral": val_det,
            "neutral_specificity": val_det,
            "roc_auc_other": val_det,
            "class_exclusivity": val_det,
        }
    return {"method": method, "metrics": metrics}


def test_returns_none_when_no_validation_readout():
    members = [
        _row("caa:positive:all_act_prediction"),
        _row("caa:positive:L3_act_prediction"),
        _row("caa:positive:act_prediction"),
        _row("caa:positive:L7_act_prediction"),
    ]
    assert _lock_caa_site(members) is None


def test_two_pole_pair_selects_on_validation_detection():
    members = [
        _row("caa:F-M:F:L1_act_prediction", val_det=0.4),
        _row("caa:F-M:F:act_prediction", val_det=0.6),
        _row("caa:F-M:F:L3_act_prediction", val_det=0.9),
    ]
    locked = _lock_caa_site(members)
    assert locked is not None
    assert locked["method"] == "caa:F-M:F:L3_act_prediction"


def test_validation_detection_still_wins_when_present():
    members = [
        _row("caa:positive:act_prediction", val_det=0.5),
        _row("caa:positive:L3_act_prediction", val_det=0.8),  # best validation Det
    ]
    locked = _lock_caa_site(members)
    assert locked is not None
    assert locked["method"] == "caa:positive:L3_act_prediction"


def test_single_candidate_is_returned():
    members = [_row("caa:positive:act_prediction")]
    assert _lock_caa_site(members)["method"] == "caa:positive:act_prediction"


def test_empty_returns_none():
    assert _lock_caa_site([]) is None
