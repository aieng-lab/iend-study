from __future__ import annotations

from analysis.recipe_overview import (
    recipe_for_method,
    recipe_per_class_for_results,
    format_recipe_text,
)
from analysis.family_overview import aggregate_family_cells


def test_recipe_for_method_strips_class_keeps_config():
    known = ["M", "F", "negative", "positive", "asian"]
    assert recipe_for_method("sae:M:kstar", known_classes=known) == "sae:kstar"
    assert recipe_for_method("sae:asian:kstar", known_classes=known) == "sae:kstar"
    assert recipe_for_method("sae:F:k1", known_classes=known) == "sae:k1"
    assert recipe_for_method("sae:M:L8", known_classes=known) == "sae:L8"
    assert recipe_for_method("sae:M:L3_k1", known_classes=known) == "sae:L3_k1"
    assert (
        recipe_for_method("caa:M:act_prediction", known_classes=known)
        == "caa:act_prediction"
    )
    assert (
        recipe_for_method("caa:M:all_act_prediction", known_classes=known)
        == "caa:all_act_prediction"
    )
    assert (
        recipe_for_method("caa:F:L5_act_prediction", known_classes=known)
        == "caa:L5_act_prediction"
    )
    assert (
        recipe_for_method(
            "gradiend:negative-positive:negative",
            known_classes=known,
            metrics={"ablation": "pair"},
        )
        == "gradiend:two_pole"
    )
    assert (
        recipe_for_method(
            "gradiend:M",
            known_classes=known,
            metrics={"ablation": "one_pole"},
        )
        == "gradiend:one_pole"
    )
    assert (
        recipe_for_method(
            "actiend:M:L11",
            known_classes=known,
            metrics={"ablation": "pair"},
        )
        == "actiend:L11"
    )
    assert (
        recipe_for_method(
            "actiend:M:tensors",
            known_classes=known,
            metrics={"ablation": "pair"},
        )
        == "actiend:tensors"
    )


def test_same_instantiation_for_all_metrics():
    payload = {
        "model": "gpt2-small",
        "task": "gender_en",
        "status": "ok",
        "config": {"target_classes": ["M", "F"], "claim_classes": ["M", "F"]},
        "methods": [
            {
                "method": "sae:M:kstar",
                "status": "ok",
                "metrics": {
                    "roc_auc_neutral": 0.80,
                    "roc_auc_other": 0.70,
                    "class_exclusivity": 0.60,
                    "neutral_specificity": 0.90,
                },
            },
            {
                "method": "sae:M:L8",
                "status": "ok",
                "metrics": {
                    "roc_auc_neutral": 0.99,
                    "roc_auc_other": 0.99,
                    "class_exclusivity": 0.99,
                    "neutral_specificity": 0.99,
                },
            },
            {
                "method": "sae:F:kstar",
                "status": "ok",
                "metrics": {
                    "roc_auc_neutral": 0.70,
                    "roc_auc_other": 0.65,
                    "class_exclusivity": 0.50,
                    "neutral_specificity": 0.80,
                },
            },
            {
                "method": "caa:M:all_act_prediction",
                "status": "ok",
                "metrics": {
                    "roc_auc_neutral": 0.95,
                    "neutral_specificity": 0.94,
                },
            },
            {
                "method": "caa:F:all_act_prediction",
                "status": "ok",
                "metrics": {
                    "roc_auc_neutral": 0.97,
                    "neutral_specificity": 0.96,
                },
            },
            {
                "method": "caa:M:L3_act_prediction",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 1.0, "neutral_specificity": 1.0},
            },
        ],
    }
    per = recipe_per_class_for_results(
        payload, metrics=["encoding_E", "roc_auc_neutral"]
    )
    sae_e = [r for r in per if r["family"] == "sae:kstar" and r["metric"] == "encoding_E"]
    sae_n = [
        r for r in per if r["family"] == "sae:kstar" and r["metric"] == "roc_auc_neutral"
    ]
    assert {r["winner_method"] for r in sae_e} == {"sae:M:kstar", "sae:F:kstar"}
    assert {r["winner_method"] for r in sae_n} == {"sae:M:kstar", "sae:F:kstar"}
    # Layer configs are their own rows, not mixed into k*.
    assert {r["family"] for r in per if "L8" in r["winner_method"]} == {"sae:L8"}
    assert {r["family"] for r in per if "L3_act_prediction" in r["winner_method"]} == {
        "caa:L3_act_prediction"
    }

    cells = aggregate_family_cells(per)
    kstar_e = next(
        c for c in cells if c["family"] == "sae:kstar" and c["metric"] == "encoding_E"
    )
    # encoding_E = min(auc_n, auc_o, excl) → M=0.60, F=0.50
    assert abs(kstar_e["mean"] - 0.55) < 1e-9
    assert abs(kstar_e["min"] - 0.50) < 1e-9
    assert abs(kstar_e["max"] - 0.60) < 1e-9

    text = format_recipe_text(cells, metric="encoding_E", model="gpt2-small")
    assert "sae:kstar" in text
    assert "sae:L8" in text
    assert "caa:all_act_prediction" in text
    assert "caa:L3_act_prediction" in text
    assert "class token stripped" in text
