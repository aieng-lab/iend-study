"""``should_reuse_caa_encode``: --skip-existing must recompute CAA when a
requested pole regime (e.g. newly-wired two-pole ``caa:F-M:C:*``) is absent from
the prior rows. A brand-new id cannot be "already done", so the coarse
all-or-nothing reuse would otherwise never train it.
"""

from __future__ import annotations

from study.deep_pipeline import _caa_row_is_pair, should_reuse_caa_encode


def _row(method: str, status: str = "ok"):
    return {"method": method, "status": status}


ONE_POLE_ROWS = [
    _row("caa:negative:act_prediction"),
    _row("caa:positive:act_prediction"),
    _row("caa:negative:L3_act_last"),
]
PAIR_ROWS = [
    _row("caa:negative-positive:negative:act_prediction"),
    _row("caa:negative-positive:positive:act_prediction"),
]


def test_pair_detection_covers_per_layer_and_plain_ids():
    assert _caa_row_is_pair("caa:negative-positive:negative:act_prediction") is True
    assert _caa_row_is_pair("caa:asia-europe:asia:L1_act_prediction") is True
    assert _caa_row_is_pair("caa:negative:act_prediction") is False
    assert _caa_row_is_pair("caa:united_states:L10_act_prediction") is False


def test_recompute_when_pair_requested_but_only_one_pole_present():
    # The exact suite_full2 situation: prior rows are one-pole only, full_plus
    # requests pair -> must recompute (not reuse).
    assert (
        should_reuse_caa_encode(
            ONE_POLE_ROWS, {}, want_pair=True, want_one_pole=True
        )
        is False
    )


def test_reuse_when_both_regimes_already_present():
    assert (
        should_reuse_caa_encode(
            ONE_POLE_ROWS + PAIR_ROWS, {}, want_pair=True, want_one_pole=True
        )
        is True
    )


def test_reuse_when_pair_not_requested():
    # one-pole-only config against one-pole-only rows -> reuse is fine.
    assert (
        should_reuse_caa_encode(
            ONE_POLE_ROWS, {}, want_pair=False, want_one_pole=True
        )
        is True
    )


def test_recompute_when_one_pole_requested_but_only_pair_present():
    assert (
        should_reuse_caa_encode(
            PAIR_ROWS, {}, want_pair=True, want_one_pole=True
        )
        is False
    )


def test_no_reuse_without_prior_rows_or_on_error():
    assert should_reuse_caa_encode([], {}, want_pair=False, want_one_pole=True) is False
    assert (
        should_reuse_caa_encode(
            ONE_POLE_ROWS, {"error": "boom"}, want_pair=False, want_one_pole=True
        )
        is False
    )
    errored = ONE_POLE_ROWS + [_row("caa:neutral:act_prediction", status="error")]
    assert (
        should_reuse_caa_encode(errored, {}, want_pair=False, want_one_pole=True)
        is False
    )
