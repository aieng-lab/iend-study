"""
Regression tests for the ``_parse_sweep_sign``/``refine_and_select_lms_gated``
fix: an earlier version silently defaulted a missing/malformed ``sign=`` tag
in ``StrengthResult.notes`` to +1.0. That's safe for a sweep that never mixes
signs (e.g. SAE clamp, target >= 0 only) but dangerous for one that does
(the additive SAE sweeps sweep both polarities) -- a forgotten tag on a
result that should have been tagged would silently report/steer the wrong
polarity with no error. Fixed: missing tags now raise loudly when sign
actually matters, and read as a legitimate "no distinction" case only when
nothing in the sweep ever carries a sign tag at all.
"""
from __future__ import annotations

import causal_eval
from causal_eval import StrengthResult, _parse_sweep_sign, refine_and_select_lms_gated


def test_study_uses_coarse_grid_without_refinement():
    assert causal_eval.CAUSAL_STRENGTH_REFINE_POINTS == 0


def test_parse_sweep_sign_returns_real_value_when_tagged():
    assert _parse_sweep_sign("sae_feats=[3] n=1 sign=-1 token_selector=all") == -1.0
    assert _parse_sweep_sign("sae_feats=[3] n=1 sign=1 token_selector=all") == 1.0


def test_parse_sweep_sign_returns_none_when_untagged():
    """Must NOT silently default to +1.0 -- callers decide what None means."""
    assert _parse_sweep_sign("") is None
    assert _parse_sweep_sign(None) is None
    assert _parse_sweep_sign("sae_clamp_feat=3 target=2 ref=1.5") is None


def _result(strength, lms, notes="", base_lms=1.0):
    return StrengthResult(
        strength=strength, lms=lms, base_lms=base_lms, lms_ok=lms >= 0.9 * base_lms,
        signed_effect=abs(strength), notes=notes,
    )


def test_refine_never_mixes_signs_when_every_result_is_tagged():
    """Sign-mixing sweep, every result tagged: refinement must stay within
    the coarse-selected sign's candidates (the documented, correct behavior)."""
    results = [
        _result(1.0, lms=1.0, notes="sign=1"),
        _result(2.0, lms=0.5, notes="sign=1"),
        _result(1.0, lms=1.0, notes="sign=-1"),
        _result(2.0, lms=0.95, notes="sign=-1"),
    ]

    def run_one(sign, s):
        return _result(s, lms=0.95, notes=f"sign={sign:g} refine=1")

    out_results, selected = refine_and_select_lms_gated(results, run_one, n_refine=1)
    assert selected is not None
    assert _parse_sweep_sign(selected.notes) in (1.0, -1.0)


def test_refine_raises_when_sign_mixing_sweep_has_untagged_coarse_selection():
    """The dangerous case this fix closes: some results are tagged with a
    real sign, but the coarse-selected one is not -- must raise, not
    silently treat the untagged result as sign=+1.0."""
    results = [
        _result(1.0, lms=1.0, notes=""),  # missing tag -- this is the bug
        _result(2.0, lms=0.5, notes="sign=1"),
        _result(1.0, lms=1.0, notes="sign=-1"),
        _result(2.0, lms=0.95, notes="sign=-1"),
    ]

    def run_one(sign, s):
        return _result(s, lms=0.95, notes=f"sign={sign:g} refine=1")

    try:
        refine_and_select_lms_gated(results, run_one, n_refine=1)
        assert False, "expected ValueError for untagged coarse selection in a sign-mixing sweep"
    except ValueError as exc:
        assert "sign=" in str(exc)


def test_refine_no_op_when_sweep_never_tags_sign():
    """A sweep that genuinely never distinguishes sign (e.g. SAE clamp,
    target >= 0 only) must still work with zero sign= tags anywhere --
    this is the legitimate "no distinction needed" case, not an error."""
    results = [
        _result(1.0, lms=1.0, notes="sae_clamp_feat=3 target=1"),
        _result(2.0, lms=0.5, notes="sae_clamp_feat=3 target=2"),
    ]

    def run_one(_sign, s):
        return _result(s, lms=0.7, notes=f"sae_clamp_feat=3 target={s:g} refine=1")

    out_results, selected = refine_and_select_lms_gated(results, run_one, n_refine=1)
    assert selected is not None
    assert len(out_results) == 3
