from __future__ import annotations

from suitability import compute_feature_suitability


ENCODER = {
    "roc_auc_neutral": 0.95,
    "roc_auc_other": 0.90,
    "class_exclusivity": 0.85,
}


def _causal(effect: float, **overrides):
    payload = {
        "causal_signed_effect": effect,
        "causal_lms_ok": True,
        "causal_beats_random_control": True,
    }
    payload.update(overrides)
    return payload


def test_suitability_accepts_positive_direction_with_protocol_gates():
    out = compute_feature_suitability(ENCODER, _causal(0.10), tau_c=0.05)

    assert out["suitability_E"] == 0.85
    assert out["suitability_G"] == 1.0
    assert out["suitability"] == 0.85
    assert out["suitability_reason"] == "ok"
    assert out["suitability_gate_reason"] is None


def test_suitability_rejects_large_wrong_direction_effect():
    out = compute_feature_suitability(ENCODER, _causal(-0.20), tau_c=0.05)

    assert out["suitability_G"] == 0.0
    assert out["suitability"] == 0.0
    assert out["suitability_reason"] == "G"
    assert out["suitability_gate_reason"] == "wrong_direction"


def test_suitability_rejects_effect_that_fails_random_control():
    out = compute_feature_suitability(
        ENCODER,
        _causal(0.20, causal_beats_random_control=False),
        tau_c=0.05,
    )

    assert out["suitability_G"] == 0.0
    assert out["suitability_gate_reason"] == "random_control_failed"


def test_suitability_rejects_explicit_lms_failure():
    out = compute_feature_suitability(
        ENCODER,
        _causal(0.20, causal_lms_ok=False),
        tau_c=0.05,
    )

    assert out["suitability_G"] == 0.0
    assert out["suitability_gate_reason"] == "lms_failed"


def test_legacy_missing_control_fields_remain_compatible():
    out = compute_feature_suitability(
        ENCODER,
        {"causal_signed_effect": 0.10},
        tau_c=0.05,
    )

    assert out["suitability_G"] == 1.0
    assert out["suitability_gate_reason"] is None


def test_soft_gate_is_directional():
    positive = compute_feature_suitability(
        ENCODER,
        _causal(0.025),
        tau_c=0.05,
        soft_causal=True,
    )
    negative = compute_feature_suitability(
        ENCODER,
        _causal(-0.025),
        tau_c=0.05,
        soft_causal=True,
    )

    assert positive["suitability_G"] == 0.5
    assert negative["suitability_G"] == 0.0
    assert negative["suitability_gate_reason"] == "wrong_direction"
