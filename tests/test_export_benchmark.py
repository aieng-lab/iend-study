"""The class-level export must stay consistent with the group tables.

``collect_group_rows_for_results`` (paper group tables) and
``collect_class_rows_for_results`` (benchmark CSV, paper artifact #1) share
one bucketing path, so the per-class rows must average back to the exact
group-row value. If that invariant breaks, the appendix per-class numbers
and the main-text macro table would silently disagree.
"""

from __future__ import annotations

import math

from analysis.method_groups import (
    collect_class_rows_for_results,
    collect_group_rows_for_results,
)


def _payload():
    def _row(method, ablation, auc_n, causal):
        return {
            "method": method,
            "status": "ok",
            "metrics": {
                "ablation": ablation,
                "roc_auc_neutral": auc_n,
                "roc_auc": auc_n,
                "causal_signed_effect": causal,
            },
        }

    return {
        "model": "gpt2-small",
        "task": "gender_en",
        "suite": "full_plus",
        "status": "ok",
        "config": {"claim_classes": ["F", "M"], "target_classes": ["F", "M"],
                   "ablations": {"pair": True, "one_pole": True}},
        "raw": {},
        "methods": [
            _row("gradiend:F-M:F", "pair", 0.9, 0.4),
            _row("gradiend:F-M:M", "pair", 0.8, 0.6),
            _row("gradiend:F", "one_pole", 0.7, 0.2),
            _row("gradiend:M", "one_pole", 0.6, 0.1),
        ],
    }


def test_class_rows_average_back_to_group_rows():
    payload = _payload()
    groups = collect_group_rows_for_results(payload, results_path="runs/_no_such_run_/gender_en/results.json")
    classes = collect_class_rows_for_results(payload, results_path="runs/_no_such_run_/gender_en/results.json")

    assert {g["method_group"] for g in groups} == {"gradiend:two_pole", "gradiend:one_pole"}
    # 2 classes per construction -> 4 class rows, none pooled here.
    assert len(classes) == 4
    assert all(r["target_class"] in {"F", "M"} for r in classes)

    for g in groups:
        grp = g["method_group"]
        members = [r["causal_signed_effect"] for r in classes if r["method_group"] == grp]
        assert math.isclose(sum(members) / len(members), g["causal_signed_effect"], rel_tol=1e-9)


def test_class_rows_carry_construction_and_class_labels():
    classes = collect_class_rows_for_results(_payload(), results_path="")
    two_pole = {r["target_class"] for r in classes if r["method_group"] == "gradiend:two_pole"}
    assert two_pole == {"F", "M"}
    assert all(r["construction"] in {"two_pole", "one_pole"} for r in classes)
    assert all(r["backend"] == "gradiend" for r in classes)
