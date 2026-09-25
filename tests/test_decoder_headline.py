"""Decoder-plot LR is the GRADIEND/ACTIEND causal headline (no ΔP re-select)."""

from __future__ import annotations

import pytest

from causal_eval import (
    CausalMethodResult,
    StrengthResult,
    _alias_legacy_factual_metrics,
    _coerce_probs_by_dataset_for_target,
    apply_package_decoder_headlines,
    class_names_from_probs,
    headline_metrics,
    meta_rows_to_frame,
    other_class,
    select_decoder_headline,
    select_lms_gated,
    signed_effect_from_group_summary,
    summary_by_group_from_probs,
    strengthen_panel,
    strength_result_from_decoder_cell,
    weaken_headline_from_decoder_results,
)


def test_weaken_selection_is_frozen_on_validation_and_reported_from_test():
    def cell(p, lms=1.0):
        return {
            "probs_by_dataset": {
                "C": {"C": p},
                "neutral": {"C": 0.1},
            },
            "lms": lms,
        }

    frozen = {
        "summary": {
            "C_weaken": {"feature_factor": -1.0, "learning_rate": 1.0}
        },
        "selection_grid": {
            "base": cell(0.8),
            (-1.0, 1.0): cell(0.6),
            (-1.0, 2.0): cell(0.7),
        },
        "grid": {
            "base": cell(0.8),
            (-1.0, 1.0): cell(0.5),
            # Test would prefer this, but test is not allowed to select.
            (-1.0, 2.0): cell(0.1),
        },
    }

    curve, test_selected, validation_selected, ff = (
        weaken_headline_from_decoder_results(frozen, target_class="C")
    )
    assert ff == -1.0
    assert validation_selected.strength == 1.0
    assert validation_selected.signed_effect == pytest.approx(0.2)
    assert test_selected.strength == 1.0
    assert test_selected.signed_effect == pytest.approx(0.3)
    assert len(curve) == 2


def _sr(
    lr: float,
    *,
    signed: float = 0.0,
    sel: float | None = None,
    lms: float = 1.0,
    base: float = 1.0,
) -> StrengthResult:
    return StrengthResult(
        strength=lr,
        lms=lms,
        base_lms=base,
        lms_ok=True,
        signed_effect=signed,
        selection_metric=sel,
    )


def test_other_class_requires_explicit_classes():
    assert other_class("F", ("M", "F")) == "M"
    assert other_class("negative", ("negative", "positive")) == "positive"
    assert other_class("IO", ("IO",)) is None
    with pytest.raises(KeyError):
        other_class("negative", ("M", "F"))
    with pytest.raises(KeyError):
        other_class("IO", ())


def test_class_names_from_probs_skips_neutral_and_fails_empty():
    names = class_names_from_probs(
        {"negative": {"negative": 0.1}, "positive": {"negative": 0.2}, "neutral": {}}
    )
    assert names == ("negative", "positive")
    with pytest.raises(KeyError):
        class_names_from_probs({"neutral": {"x": 1.0}})


def test_alias_nested_factual_keys_in_probs_by_dataset():
    aliased = _alias_legacy_factual_metrics(
        {
            "probs_by_dataset": {
                "MATCH": {"MATCH_factual": 0.4},
                "neutral": {"lms_only": 1.0},
            }
        }
    )
    assert aliased["probs_by_dataset"]["MATCH"]["MATCH"] == 0.4


def test_strengthen_panel_one_pole_is_same_panel():
    assert strengthen_panel("IO", ("IO",)) == "IO"
    assert strengthen_panel("negative", ("negative", "positive")) == "positive"


def test_headline_metrics_rival_only_summary_does_not_keyerror_target():
    """P(target) scored only on rival panels — ravel_language SAE used to set error='english'."""
    from causal_eval import strengthen_panel_from_summary

    summary = {
        "portuguese": {"mean_delta_p_target": 0.18, "n": 34},
        "spanish": {"mean_delta_p_target": 0.05, "n": 39},
        "neutral": {"mean_delta_p_target": -0.02, "n": 100},
    }
    assert strengthen_panel_from_summary("english", summary) == "portuguese"
    sr = StrengthResult(
        strength=20.0,
        lms=0.08,
        base_lms=0.09,
        lms_ok=True,
        signed_effect=0.18,
        summary_by_group=summary,
    )
    out = headline_metrics(
        CausalMethodResult(
            method="sae:english:k1",
            backend="sae",
            target_class="english",
            strengths=[sr],
            selected_strength=20.0,
            selected=sr,
        )
    )
    assert out.get("causal_error") is None
    assert abs(out["causal_signed_effect"] - 0.18) < 1e-12
    assert out["causal_delta_target"] is None
    assert abs(out["causal_delta_other"] - 0.18) < 1e-12


def test_strength_result_emotion_delta_is_real():
    base = {
        "lms": 0.1,
        "probs_by_dataset": {
            "negative": {"negative": 0.14, "positive": 0.01},
            "positive": {"negative": 0.008, "positive": 0.2},
        },
    }
    cell = {
        "lms": 0.099,
        "probs_by_dataset": {
            "negative": {"negative": 0.11, "positive": 0.02},
            "positive": {"negative": 0.03, "positive": 0.18},
        },
    }
    sr = strength_result_from_decoder_cell(
        target_class="negative",
        strength=0.5,
        base_entry=base,
        cell=cell,
    )
    assert sr.selection_metric == 0.03
    assert abs(sr.signed_effect - (0.03 - 0.008)) < 1e-12


def test_strength_result_one_pole_same_panel_delta():
    """One-pole grids only have the claim dataset — ΔP must not become 0.0."""
    base = {
        "lms": 0.089,
        "probs_by_dataset": {"IO": {"IO": 0.2275}},
    }
    cell = {
        "lms": 0.0889,
        "probs_by_dataset": {"IO": {"IO": 0.3713}},
    }
    sr = strength_result_from_decoder_cell(
        target_class="IO",
        strength=0.02,
        base_entry=base,
        cell=cell,
    )
    assert abs(sr.selection_metric - 0.3713) < 1e-12
    assert abs(sr.signed_effect - (0.3713 - 0.2275)) < 1e-12
    assert abs(sr.summary_by_group["IO"]["mean_delta_p_target"] - sr.signed_effect) < 1e-12
    assert abs(signed_effect_from_group_summary(sr.summary_by_group, target_class="IO") - sr.signed_effect) < 1e-12


def test_strength_result_one_pole_skips_neutral_without_claim():
    """Neutral blocks often omit claim keys — must not KeyError; CF panel is rival."""
    base = {
        "lms": 0.089,
        "probs_by_dataset": {
            "IO": {"IO": 0.23, "DISTRACTOR": 0.4},
            "DISTRACTOR": {"IO": 0.10, "DISTRACTOR": 0.5},
            "neutral": {"lms_only": 1.0},
        },
    }
    cell = {
        "lms": 0.088,
        "probs_by_dataset": {
            "IO": {"IO": 0.25, "DISTRACTOR": 0.35},
            "DISTRACTOR": {"IO": 0.24, "DISTRACTOR": 0.45},
            "neutral": {"lms_only": 1.0},
        },
    }
    sr = strength_result_from_decoder_cell(
        target_class="IO",
        strength=0.02,
        base_entry=base,
        cell=cell,
    )
    # Strengthen panel = rival DISTRACTOR → ΔP(IO) on DISTRACTOR texts.
    assert abs(sr.selection_metric - 0.24) < 1e-12
    assert abs(sr.signed_effect - (0.24 - 0.10)) < 1e-12
    assert "neutral" not in sr.summary_by_group
    assert "IO" in sr.summary_by_group and "DISTRACTOR" in sr.summary_by_group


def test_meta_rows_to_frame_maps_row_wise_columns():
    rows = [
        {
            "sample_id": "MATCH:0",
            "group": "MATCH",
            "text": "dog w dog [MASK]",
            "label": "w",
            "label_class": "MATCH",
            "alternative": "dog",
            "alternative_class": "DISTRACTOR",
        }
    ]
    frame = meta_rows_to_frame(rows)
    assert frame.loc[0, "factual"] == "w"
    assert frame.loc[0, "factual_id"] == "MATCH"
    assert frame.loc[0, "alternative_id"] == "DISTRACTOR"


def test_strengthen_panel_from_probs_rival_only_outer_panels():
    """MIB/ravel: P(english) only under portuguese/spanish outer keys."""
    pbd = {
        "portuguese": {"english": 0.2, "portuguese": 0.8},
        "spanish": {"english": 0.15, "spanish": 0.85},
    }
    from causal_eval import strengthen_panel_from_probs

    assert strengthen_panel_from_probs(pbd, "english") == "portuguese"


def test_strength_result_ravel_rival_only_panels():
    pbd = {
        "portuguese": {"english": 0.2, "portuguese": 0.8},
        "spanish": {"english": 0.15, "spanish": 0.85},
    }
    base = {"lms": 0.1, "probs_by_dataset": pbd}
    cell = {
        "lms": 0.099,
        "probs_by_dataset": {
            "portuguese": {"english": 0.25, "portuguese": 0.75},
            "spanish": {"english": 0.15, "spanish": 0.85},
        },
    }
    sr = strength_result_from_decoder_cell(
        target_class="english",
        strength=1.0,
        base_entry=base,
        cell=cell,
    )
    assert abs(sr.signed_effect - 0.05) < 1e-12


def test_summary_by_group_accepts_factual_suffix_keys():
    base = {
        "christian": {"christian_factual": 0.20},
        "muslim": {"christian_factual": 0.10},
    }
    mod = {
        "christian": {"christian_factual": 0.25},
        "muslim": {"christian_factual": 0.11},
    }
    summ = summary_by_group_from_probs(base, mod, target_class="christian")
    assert abs(summ["christian"]["mean_delta_p_target"] - 0.05) < 1e-12
    assert abs(summ["muslim"]["mean_delta_p_target"] - 0.01) < 1e-12


def test_summary_by_group_accepts_case_mismatch_keys():
    base = {"Portuguese": {"English": 0.20}, "Spanish": {"English": 0.10}}
    mod = {"Portuguese": {"English": 0.24}, "Spanish": {"English": 0.11}}
    summ = summary_by_group_from_probs(base, mod, target_class="english")
    assert abs(summ["Portuguese"]["mean_delta_p_target"] - 0.04) < 1e-12


def test_coerce_probs_by_dataset_from_same_panel_probs():
    pbd = _coerce_probs_by_dataset_for_target(
        {"probs": {"MATCH": 0.42}, "probs_by_dataset": {"MATCH": {"M": 0.1}}},
        "MATCH",
    )
    assert pbd["MATCH"]["MATCH"] == 0.42


def test_coerce_fills_claim_panel_from_probs_when_inner_missing():
    """Grid refresh can leave rival inner keys without the claim key on the claim panel."""
    pbd = _coerce_probs_by_dataset_for_target(
        {
            "probs": {"MATCH": 0.42},
            "probs_by_dataset": {"MATCH": {"DISTRACTOR": 0.1}, "DISTRACTOR": {"DISTRACTOR": 0.5}},
        },
        "MATCH",
    )
    assert pbd["MATCH"]["MATCH"] == 0.42


def test_strength_result_one_pole_two_outer_panels_same_panel_strengthen():
    """Expanded CF rows add a DISTRACTOR panel; strengthen stays on MATCH."""
    base = {
        "lms": 0.089,
        "probs_by_dataset": {
            "MATCH": {"MATCH": 0.23, "DISTRACTOR": 0.40},
            "DISTRACTOR": {"DISTRACTOR": 0.50},
        },
    }
    cell = {
        "lms": 0.088,
        "probs_by_dataset": {
            "MATCH": {"MATCH": 0.37, "DISTRACTOR": 0.35},
            "DISTRACTOR": {"DISTRACTOR": 0.45},
        },
    }
    sr = strength_result_from_decoder_cell(
        target_class="MATCH",
        strength=0.02,
        base_entry=base,
        cell=cell,
    )
    assert abs(sr.selection_metric - 0.37) < 1e-12
    assert abs(sr.signed_effect - (0.37 - 0.23)) < 1e-12


def test_strength_result_claim_scalar_without_inner_key():
    base = {"lms": 0.1, "probs": {"MATCH": 0.2}, "probs_by_dataset": {"MATCH": {"DISTRACTOR": 0.1}}}
    cell = {"lms": 0.1, "probs": {"MATCH": 0.35}, "probs_by_dataset": {"MATCH": {"DISTRACTOR": 0.08}}}
    sr = strength_result_from_decoder_cell(
        target_class="MATCH",
        strength=0.02,
        base_entry=base,
        cell=cell,
    )
    assert abs(sr.signed_effect - 0.15) < 1e-12


def test_strength_result_missing_probs_keyerror():
    with pytest.raises(KeyError):
        strength_result_from_decoder_cell(
            target_class="IO",
            strength=0.02,
            base_entry={"lms": 0.1},
            cell={"lms": 0.1, "probs_by_dataset": {"IO": {"IO": 0.3}}},
        )


def test_select_decoder_headline_uses_package_lr_not_delta_p():
    floor = _sr(1e-5, signed=0.0, sel=0.01)
    starred = _sr(50.0, signed=0.0, sel=0.03)
    results = [starred, floor]
    by_lr = {r.strength: r for r in results}

    delta_pick = select_lms_gated(results, prefer_signed_effect=True)
    assert delta_pick is not None
    assert delta_pick.strength == 1e-5

    headline = select_decoder_headline(results, by_lr, package_lr=50.0)
    assert headline.strength == 50.0
    with pytest.raises(KeyError):
        select_decoder_headline(results, by_lr, package_lr=7.0)


def test_headline_metrics_one_pole_uses_panel_delta():
    sr = StrengthResult(
        strength=0.02,
        lms=0.0889,
        base_lms=0.0894,
        lms_ok=True,
        signed_effect=0.0,  # stale wrong field — panel summary wins
        summary_by_group={"IO": {"mean_delta_p_target": 0.1438}},
    )
    out = headline_metrics(
        CausalMethodResult(
            method="gradiend:IO",
            backend="gradiend",
            target_class="IO",
            strengths=[sr],
            selected_strength=0.02,
            selected=sr,
        )
    )
    assert abs(out["causal_signed_effect"] - 0.1438) < 1e-12
    assert out["causal_null_effect"] is False


def test_headline_metrics_reports_weaken_and_base_p():
    sr_up = StrengthResult(
        strength=0.02,
        lms=0.09,
        base_lms=0.09,
        lms_ok=True,
        signed_effect=0.2,
        summary_by_group={
            "IO": {
                "mean_delta_p_target": 0.2,
                "mean_p_target_base": 0.4,
                "mean_p_target_mod": 0.6,
                "n": 10,
            }
        },
    )
    sr_down = StrengthResult(
        strength=0.05,
        lms=0.09,
        base_lms=0.09,
        lms_ok=True,
        signed_effect=0.3,
        summary_by_group={
            "IO": {
                "mean_delta_p_target": -0.3,
                "mean_p_target_base": 0.4,
                "mean_p_target_mod": 0.1,
                "n": 10,
            }
        },
    )
    out = headline_metrics(
        CausalMethodResult(
            method="gradiend:IO",
            backend="gradiend",
            target_class="IO",
            strengths=[sr_up, sr_down],
            selected_strength=0.02,
            selected=sr_up,
            weaken_strengths=[sr_down],
            weaken_selected_strength=0.05,
            weaken_selected=sr_down,
        )
    )
    assert abs(out["causal_signed_effect"] - 0.2) < 1e-12
    assert abs(out["causal_signed_effect_weaken"] - 0.3) < 1e-12
    assert abs(out["causal_base_p"] - 0.4) < 1e-12


def test_headline_metrics_keep_validation_selection_separate_from_test_report():
    test_sr = StrengthResult(
        strength=2.0,
        lms=0.087,
        base_lms=0.09,
        lms_ok=False,
        signed_effect=0.25,
        summary_by_group={
            "IO": {
                "mean_delta_p_target": 0.25,
                "mean_p_target_base": 0.4,
                "mean_p_target_mod": 0.65,
                "n": 10,
            }
        },
    )
    out = headline_metrics(
        CausalMethodResult(
            method="gradiend:IO",
            backend="gradiend",
            target_class="IO",
            strengths=[],
            selected_strength=2.0,
            selected=test_sr,
            meta={
                "selection_split": "validation",
                "report_split": "test",
                "selection_strength": 2.0,
                "selection_signed_effect": 0.2,
                "selection_lms": 0.0892,
                "selection_base_lms": 0.09,
                "selection_lms_ratio_to_base": 0.9911111111111112,
                "selection_lms_ok": True,
            },
        )
    )

    assert out["causal_selection_split"] == "validation"
    assert out["causal_report_split"] == "test"
    assert out["causal_selection_lms_ok"] is True
    assert out["causal_selection_lms_ratio_to_base"] > 0.99
    assert out["causal_lms_ok"] is False
    assert out["causal_selection_signed_effect"] == 0.2
    assert out["causal_signed_effect"] == 0.25


def test_apply_package_decoder_headlines_does_not_rewrite_legacy_floor_pick(capsys):
    results = {
        "methods": [
            {
                "method": "gradiend:negative",
                "target_class": "negative",
                "metrics": {
                    "causal_selected_strength": 1e-5,
                    "causal_signed_effect": 0.0,
                    "causal_grid_floor": True,
                    "causal_null_effect": True,
                },
                "extras": {
                    "causal": {
                        "method": "gradiend:negative",
                        "backend": "gradiend",
                        "target_class": "negative",
                        "selected_strength": 1e-5,
                        "meta": {
                            "source": "decoder_grid_cache",
                            "package_learning_rate": 0.5,
                            "study_learning_rate": 1e-5,
                        },
                        "strengths": [
                            {
                                "strength": 1e-5,
                                "lms": 0.1,
                                "base_lms": 0.1,
                                "lms_ok": True,
                                "signed_effect": 0.0,
                                "summary_by_group": {
                                    "negative": {"mean_delta_p_target": 0.0},
                                    "positive": {"mean_delta_p_target": 0.0},
                                },
                            },
                            {
                                "strength": 0.5,
                                "lms": 0.098,
                                "base_lms": 0.099,
                                "lms_ok": True,
                                "signed_effect": 0.0,
                                "summary_by_group": {
                                    "negative": {"mean_delta_p_target": -0.02},
                                    "positive": {"mean_delta_p_target": 0.0218},
                                },
                            },
                        ],
                    }
                },
            }
        ]
    }
    apply_package_decoder_headlines(results)
    m = results["methods"][0]["metrics"]
    assert m["causal_selected_strength"] == 1e-5
    assert m["causal_signed_effect"] == 0.0
    assert m["causal_grid_floor"] is True
    assert m["causal_null_effect"] is True
    assert results["methods"][0]["extras"]["causal"]["meta"]["study_learning_rate"] == 1e-5
    assert capsys.readouterr().out == ""


def test_apply_package_preserves_frozen_test_headline(capsys):
    """Validation curves must never overwrite a selected-LR test report."""
    results = {
        "methods": [
            {
                "method": "cga:africa",
                "target_class": "africa",
                "metrics": {
                    "causal_selected_strength": 0.2,
                    "causal_signed_effect": 0.3355,
                },
                "extras": {
                    "causal": {
                        "method": "cga:africa",
                        "backend": "cga",
                        "target_class": "africa",
                        "selected_strength": 0.2,
                        "selected": {
                            "strength": 0.2,
                            "signed_effect": 0.3355,
                        },
                        "meta": {
                            "source": "decoder_grid_cache",
                            "package_learning_rate": 0.2,
                            "selection_split": "validation",
                            "report_split": "test",
                        },
                        "strengths": [
                            {
                                "strength": 0.2,
                                "lms": 0.1,
                                "base_lms": 0.1,
                                "lms_ok": True,
                                "signed_effect": 0.7049,
                                "summary_by_group": {
                                    "africa": {"mean_delta_p_target": 0.7049},
                                },
                            },
                        ],
                    }
                },
            }
        ]
    }

    apply_package_decoder_headlines(results)

    row = results["methods"][0]
    assert row["metrics"]["causal_signed_effect"] == pytest.approx(0.3355)
    assert row["extras"]["causal"]["selected"]["signed_effect"] == pytest.approx(0.3355)
    assert row["extras"]["causal"]["strengths"][0]["signed_effect"] == pytest.approx(0.7049)
    assert capsys.readouterr().out == ""


def test_apply_package_ignores_legacy_rows_without_mutating_them(capsys):
    """Legacy rows are not even inspected during report loading."""
    results = {
        "methods": [
            {
                "method": "gradiend:IO",
                "target_class": "IO",
                "metrics": {
                    "causal_selected_strength": 0.02,
                    "causal_signed_effect": 0.0,
                    "causal_random_signed_effect": 0.0,
                    "causal_null_effect": True,
                },
                "extras": {
                    "causal": {
                        "method": "gradiend:IO",
                        "backend": "gradiend",
                        "target_class": "IO",
                        "selected_strength": 0.02,
                        "meta": {
                            "source": "decoder_grid_cache",
                            "package_learning_rate": 0.02,
                        },
                        "strengths": [
                            {
                                "strength": 0.02,
                                "lms": 0.0889,
                                "base_lms": 0.0894,
                                "lms_ok": True,
                                "signed_effect": 0.0,
                                "summary_by_group": {
                                    "IO": {"mean_delta_p_target": 0.1438},
                                },
                            },
                        ],
                        "random_control": {
                            "strength": 0.02,
                            "lms": 0.09,
                            "base_lms": 0.0894,
                            "lms_ok": True,
                            "signed_effect": 0.0,
                            "summary_by_group": {},
                            "is_random_control": True,
                            "notes": "opposite feature_factor polarity control",
                        },
                    }
                },
            }
        ]
    }
    apply_package_decoder_headlines(results)
    row = results["methods"][0]
    m = row["metrics"]
    causal = row["extras"]["causal"]
    assert m["causal_signed_effect"] == 0.0
    assert m["causal_random_signed_effect"] == 0.0
    assert m["causal_null_effect"] is True
    assert causal["random_control"]["summary_by_group"] == {}
    assert capsys.readouterr().out == ""
