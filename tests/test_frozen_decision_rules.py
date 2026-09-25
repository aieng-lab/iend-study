"""Spec_n / Excl decision rules must be fit on validation and frozen for test.

``class_vs_neutral_metrics`` used to refit the neutral threshold (Spec_n) and every
per-rival threshold (Excl) on whatever split it was scoring, and callers only
carried the *pooled diagnostic* ``youden_threshold`` from validation to test. The
held-out Spec_n/Excl were therefore oracle (test-optimised) numbers. These tests pin
the corrected contract:

* a validation-fit threshold *and orientation sign* are reused verbatim on test;
* a threshold without its sign, or a rival without a frozen rule, raises instead
  of silently refitting on the evaluated split;
* with nothing supplied, the fit source is recorded as ``"eval"`` (never hidden).
"""

from __future__ import annotations

import numpy as np
import pytest

from sae_eval import (
    FrozenRulesUnavailable,
    class_vs_neutral_metrics,
    frozen_decision_rules,
    require_frozen_rules,
)
from results_schema import unfrozen_rule_readouts


def _val_test(seed: int = 0, n: int = 4000):
    rng = np.random.default_rng(seed)
    val = {
        "pos": rng.normal(1.0, 0.5, n),
        "neu": rng.normal(0.0, 0.5, n),
        "rival": rng.normal(0.3, 0.5, n),
    }
    # Test drifts: neutrals move toward the target, so a rule frozen on validation
    # is no longer the test-optimal cut.
    test = {
        "pos": rng.normal(1.0, 0.5, n),
        "neu": rng.normal(0.6, 0.5, n),
        "rival": rng.normal(0.7, 0.5, n),
    }
    return val, test


def _metrics(split, **kw):
    return class_vs_neutral_metrics(
        split["pos"],
        split["neu"],
        target_class="A",
        scores_other_by_class={"B": split["rival"]},
        n_bootstrap=0,
        **kw,
    )


def test_frozen_neutral_rule_is_applied_to_test_not_refit():
    val, test = _val_test()
    val_out = _metrics(val)
    frozen = frozen_decision_rules(val_out)

    oracle = _metrics(test)  # what the old code reported on test
    held_out = _metrics(test, **frozen)

    tau, sign = val_out["neutral_youden_threshold"], val_out["neutral_youden_sign"]
    assert held_out["neutral_youden_threshold"] == pytest.approx(tau)
    assert held_out["neutral_youden_sign"] == sign
    assert held_out["neutral_youden_threshold_source"] == "provided"

    expected_spec = float((sign * test["neu"] <= tau).mean())
    assert held_out["neutral_specificity"] == pytest.approx(expected_spec)
    assert held_out["specificity"] == pytest.approx(expected_spec)
    # The oracle refit is optimistic on drifted data: this is the bias being removed.
    assert oracle["neutral_specificity"] > held_out["neutral_specificity"] + 0.05


def test_frozen_rival_rule_is_applied_to_test_not_refit():
    val, test = _val_test()
    val_out = _metrics(val)
    frozen = frozen_decision_rules(val_out)

    oracle = _metrics(test)
    held_out = _metrics(test, **frozen)

    tau = val_out["rival_youden_threshold_by_class"]["B"]
    sign = val_out["rival_youden_sign_by_class"]["B"]
    assert held_out["rival_youden_threshold_by_class"]["B"] == pytest.approx(tau)
    assert held_out["rival_youden_threshold_source"] == "provided"

    expected = float((sign * test["rival"] <= tau).mean())
    assert held_out["other_tnr_by_class"]["B"] == pytest.approx(expected)
    assert held_out["class_exclusivity"] == pytest.approx(expected)
    assert oracle["class_exclusivity"] > held_out["class_exclusivity"] + 0.02


def test_orientation_sign_is_frozen_not_rederived_from_test_scores():
    rng = np.random.default_rng(5)
    n = 2000
    # Validation: target BELOW neutral -> sign -1.
    val = {
        "pos": rng.normal(-1.0, 0.2, n),
        "neu": rng.normal(0.0, 0.2, n),
        "rival": rng.normal(-1.0, 0.2, n),
    }
    # Test: polarity flipped (target above neutral). The frozen sign must win.
    test = {
        "pos": rng.normal(1.0, 0.2, n),
        "neu": rng.normal(0.0, 0.2, n),
        "rival": rng.normal(1.0, 0.2, n),
    }
    val_out = _metrics(val)
    assert val_out["neutral_youden_sign"] == -1.0
    held_out = _metrics(test, **frozen_decision_rules(val_out))
    assert held_out["neutral_youden_sign"] == -1.0
    assert held_out["rival_youden_sign_by_class"]["B"] == val_out["rival_youden_sign_by_class"]["B"]


def test_threshold_without_sign_raises():
    val, test = _val_test()
    val_out = _metrics(val)
    with pytest.raises(ValueError, match="without neutral_youden_sign"):
        _metrics(test, neutral_youden_threshold=val_out["neutral_youden_threshold"])


def test_missing_frozen_rival_rule_raises_instead_of_refitting():
    val, test = _val_test()
    frozen = frozen_decision_rules(_metrics(val))
    frozen["rival_youden_threshold_by_class"] = {}
    frozen["rival_youden_sign_by_class"] = {}
    with pytest.raises(ValueError, match="frozen rival rule missing"):
        _metrics(test, **frozen)


def test_no_frozen_rules_records_eval_source():
    val, _ = _val_test()
    out = _metrics(val)
    assert out["neutral_youden_threshold_source"] == "eval"
    assert out["rival_youden_threshold_source"] == "eval"


def test_frozen_decision_rules_degrades_to_empty_without_a_usable_fit():
    assert frozen_decision_rules(None) == {}
    assert frozen_decision_rules({"error": "insufficient samples"}) == {}
    # A rival whose validation threshold is undefined is dropped (so a test call
    # that scores that rival raises rather than refitting).
    partial = frozen_decision_rules(
        {
            "neutral_youden_threshold": 0.1,
            "neutral_youden_sign": 1.0,
            "rival_youden_threshold_by_class": {"B": 0.2, "C": None},
            "rival_youden_sign_by_class": {"B": 1.0, "C": 1.0},
        }
    )
    assert partial["rival_youden_threshold_by_class"] == {"B": 0.2}
    assert partial["rival_youden_sign_by_class"] == {"B": 1.0}


def test_frozen_rule_errors_are_valueerrors_but_a_dedicated_type():
    # Dedicated so broad ``except Exception`` blocks can let it through.
    assert issubclass(FrozenRulesUnavailable, ValueError)


def test_require_frozen_rules_raises_on_a_test_fit_readout():
    val, test = _val_test()
    frozen = frozen_decision_rules(_metrics(val))
    good = _metrics(test, **frozen)
    oracle = _metrics(test)  # no validation rules: fit on test itself
    require_frozen_rules({"A": good}, context="ok")
    with pytest.raises(FrozenRulesUnavailable, match=r"\[B\]"):
        require_frozen_rules({"A": good, "B": oracle}, context="unit")


def test_unfrozen_rule_readouts_ignores_errors_and_non_readouts_and_flags_rivals():
    assert unfrozen_rule_readouts(None) == []
    assert unfrozen_rule_readouts({"X": {"error": "no rows"}, "Y": {"foo": 1}}) == []
    val, test = _val_test()
    frozen = frozen_decision_rules(_metrics(val))
    held = _metrics(test, **frozen)
    assert unfrozen_rule_readouts({"A": held}) == []
    # Neutral rule frozen but a rival refit on test is still flagged.
    mixed = dict(held, rival_youden_threshold_source="eval")
    assert unfrozen_rule_readouts({"A": mixed}) == ["A"]
