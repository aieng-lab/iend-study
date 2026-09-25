"""``causal_random_signed_effect`` used to sit next to ``causal_signed_effect``
in results.json/REPORT.md/TABLES.txt with nothing ever computed from the
pair anywhere downstream (checked: zero consumption in report_study.py or
analysis/*.py before this fix). Since the compute for the random control is
small (~2-5% of the dominant decoder-grid sweep, see the 2026-08-21 CLAUDE.md
entries) but non-zero, and it's explicitly a planned part of the causal
protocol (STUDY.md: "matched random controls"), the fix is to make it earn
its compute rather than cut it: turn the pair into a reportable per-row
specificity flag/ratio (causal_eval.py::specificity_vs_random_control) and a
paper-citable aggregate stat (causal_eval.py::specificity_control_summary).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_eval import (
    CausalMethodResult,
    StrengthResult,
    headline_metrics,
    specificity_control_summary,
    specificity_vs_random_control,
)


def test_real_effect_larger_than_random_control():
    ratio, beats = specificity_vs_random_control(0.20, 0.02)
    assert beats is True
    assert abs(ratio - 10.0) < 1e-9


def test_real_effect_smaller_than_random_control_flagged_false():
    ratio, beats = specificity_vs_random_control(0.01, 0.05)
    assert beats is False
    assert abs(ratio - 0.2) < 1e-9


def test_near_zero_random_control_ratio_is_none_but_beats_is_defined():
    ratio, beats = specificity_vs_random_control(0.10, 1e-9)
    assert ratio is None
    assert beats is True


def test_missing_inputs_return_none_none():
    assert specificity_vs_random_control(None, 0.1) == (None, None)
    assert specificity_vs_random_control(0.1, None) == (None, None)


def test_headline_metrics_reports_specificity_fields():
    sr = StrengthResult(
        strength=10.0,
        lms=0.09,
        base_lms=0.09,
        lms_ok=True,
        signed_effect=0.20,
        summary_by_group={"cls_a": {"mean_delta_p_target": 0.20}},
    )
    rnd = StrengthResult(
        strength=10.0,
        lms=0.09,
        base_lms=0.09,
        lms_ok=True,
        signed_effect=0.02,
        is_random_control=True,
    )
    out = headline_metrics(
        CausalMethodResult(
            method="gradiend:cls_a",
            backend="gradiend",
            target_class="cls_a",
            strengths=[sr],
            selected_strength=10.0,
            selected=sr,
            random_control=rnd,
        )
    )
    assert out["causal_beats_random_control"] is True
    assert abs(out["causal_specificity_ratio"] - 10.0) < 1e-9


def test_headline_metrics_specificity_fields_none_without_random_control():
    sr = StrengthResult(
        strength=10.0,
        lms=0.09,
        base_lms=0.09,
        lms_ok=True,
        signed_effect=0.20,
        summary_by_group={"cls_a": {"mean_delta_p_target": 0.20}},
    )
    out = headline_metrics(
        CausalMethodResult(
            method="gradiend:cls_a",
            backend="gradiend",
            target_class="cls_a",
            strengths=[sr],
            selected_strength=10.0,
            selected=sr,
        )
    )
    assert out["causal_beats_random_control"] is None
    assert out["causal_specificity_ratio"] is None


def test_specificity_control_summary_aggregates_across_rows():
    rows = [
        {"causal_beats_random_control": True, "causal_specificity_ratio": 4.0},
        {"causal_beats_random_control": True, "causal_specificity_ratio": 2.0},
        {"causal_beats_random_control": False, "causal_specificity_ratio": 0.5},
        {"causal_beats_random_control": None},  # no random control computed
        {"causal_error": "boom"},  # unrelated row shape, ignored
    ]

    summary = specificity_control_summary(rows)

    assert summary["n_total"] == 3
    assert summary["n_exceeds_random_control"] == 2
    assert abs(summary["fraction_exceeds_random_control"] - 2 / 3) < 1e-9
    assert abs(summary["median_specificity_ratio"] - 2.0) < 1e-9


def test_specificity_control_summary_empty_when_nothing_to_aggregate():
    summary = specificity_control_summary([{"causal_signed_effect": 0.1}])
    assert summary["n_total"] == 0
    assert summary["fraction_exceeds_random_control"] is None
    assert summary["median_specificity_ratio"] is None
