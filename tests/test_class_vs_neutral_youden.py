"""Regression tests for class_vs_neutral_metrics' Youden threshold handling.

Covers two related bugs found while investigating why ACTIEND's Excl./Spec.
numbers could read near-zero on tasks where target-vs-neutral separation was
actually near-perfect (e.g. gpt2-small/repetition's actiend:YES row). See
CLAUDE.md for the full writeup.

1. Shared/pooled threshold: specificity and class_exclusivity used to share
   one Youden threshold fit against rivals+neutrals pooled together. When a
   rival collapses onto the target class (encoder can't tell them apart),
   that shared threshold gets dragged into the collapsed cluster and can
   report near-zero neutral specificity even though target-vs-neutral alone
   is cleanly separable.
2. Orientation: `_youden_threshold` used to assume "higher score = target"
   with no self-orientation, unlike `_roc_auc` (which flips to whichever
   direction separates better). A class whose natural polarity runs opposite
   that convention (raw target-class scores below neutral/rival scores) got
   a degenerate threshold search even though roc_auc read correctly.
"""

from __future__ import annotations

import numpy as np

from sae_eval import _orientation_sign, _youden_threshold, class_vs_neutral_metrics


def test_youden_threshold_self_orients_like_roc_auc():
    rng = np.random.default_rng(0)
    n = 500
    # Target scores sit BELOW neutral scores -- natural polarity opposite the
    # naive "higher score = positive" convention.
    pos = -1.0 + rng.normal(0, 0.02, n)
    neg = -0.07 + rng.normal(0, 0.05, n)

    tau, j, sign = _youden_threshold(pos, neg)
    assert sign == -1.0
    assert j is not None and j > 0.9  # near-perfect separation once oriented

    # Predicate is (sign * raw_score) > tau.
    tpr = float((sign * pos > tau).mean())
    tnr = float((sign * neg <= tau).mean())
    assert tpr > 0.95
    assert tnr > 0.95


def test_orientation_sign_matches_youden_threshold():
    rng = np.random.default_rng(1)
    pos = rng.normal(2.0, 0.5, 200)
    neg = rng.normal(-2.0, 0.5, 200)
    _tau, _j, sign = _youden_threshold(pos, neg)
    assert sign == _orientation_sign(pos, neg)

    pos_flipped, neg_flipped = -pos, -neg
    _tau2, _j2, sign2 = _youden_threshold(pos_flipped, neg_flipped)
    assert sign2 == _orientation_sign(pos_flipped, neg_flipped)
    assert sign2 != sign


def test_class_vs_neutral_metrics_rival_collapse_does_not_zero_neutral_specificity():
    """Reproduces the gpt2-small/repetition actiend:YES pattern.

    Target and rival collapse onto the same (negative-polarity) region;
    neutral sits far away in a different region. Before the fix, the shared
    pooled threshold got dragged into the target/rival cluster and reported
    specificity=0.0 despite near-perfect target-vs-neutral separation
    (roc_auc_neutral ~= 1.0). After the fix, specificity should track
    roc_auc_neutral, and class_exclusivity should independently reflect the
    genuine rival collapse (roc_auc_other ~= chance).
    """
    rng = np.random.default_rng(2)
    n = 2000
    pos = -1.0 + rng.normal(0, 0.02, n)
    rival = -1.0 + rng.normal(0, 0.02, n)
    neu = -0.07 + rng.normal(0, 0.05, n)

    out = class_vs_neutral_metrics(
        pos,
        neu,
        target_class="YES",
        scores_other=rival,
        other_class="NOT_YES",
        n_bootstrap=0,
    )

    assert out["roc_auc_neutral"] > 0.99
    assert out["specificity"] > 0.9
    assert out["neutral_specificity"] > 0.9

    # Rival separation genuinely is at chance -- class_exclusivity should
    # stay low, decoupled from the (now-fixed) specificity number.
    assert out["roc_auc_other"] < 0.6
    assert out["class_exclusivity"] < 0.6

    # Diagnostic pooled/joint threshold is still reported, unchanged
    # semantics, just no longer used to compute specificity/exclusivity.
    assert "youden_threshold" in out
    assert "youden_j" in out


def test_class_vs_neutral_metrics_clean_separation_all_metrics_agree():
    """Sanity check: when target/rival/neutral are all well separated, the
    decoupled thresholds agree with each other and with the pooled one."""
    rng = np.random.default_rng(3)
    n = 1000
    pos = rng.normal(2.0, 0.2, n)
    rival = rng.normal(-2.0, 0.2, n)
    neu = rng.normal(0.0, 0.2, n)

    out = class_vs_neutral_metrics(
        pos,
        neu,
        target_class="A",
        scores_other=rival,
        other_class="B",
        n_bootstrap=0,
    )
    assert out["specificity"] > 0.95
    assert out["class_exclusivity"] > 0.95
    assert out["roc_auc_neutral"] > 0.99
    assert out["roc_auc_other"] > 0.99
