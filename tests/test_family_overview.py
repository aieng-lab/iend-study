from __future__ import annotations

from analysis.family_overview import (
    aggregate_family_cells,
    best_per_class_for_results,
    feature_class_for_method,
    format_cell,
    format_overview_latex,
    format_overview_text,
)


def test_feature_class_parsing():
    known = ["M", "F", "asian", "white"]
    assert feature_class_for_method("gradiend:M", known_classes=known) == "M"
    assert feature_class_for_method("gradiend:asian-white:white", known_classes=known) == "white"
    assert feature_class_for_method("actiend:asian-white:tensors:asian", known_classes=known) == "asian"
    assert feature_class_for_method("sae:M:kstar", known_classes=known) == "M"
    assert feature_class_for_method("sae:M:L3_k1", known_classes=known) == "M"
    assert feature_class_for_method("caa:F:act_prediction", known_classes=known) == "F"
    assert feature_class_for_method("caa:F:L2_act_prediction", known_classes=known) == "F"
    assert feature_class_for_method("gradiend", known_classes=known) is None
    assert feature_class_for_method("sae", known_classes=known) is None


def test_best_per_class_and_mean_minmax():
    payload = {
        "model": "gpt2-small",
        "task": "gender_en",
        "status": "ok",
        "config": {
            "target_classes": ["M", "F"],
            "claim_classes": ["M", "F"],
        },
        "methods": [
            {
                "method": "gradiend:M",
                "status": "ok",
                "metrics": {
                    "ablation": "one_pole",
                    "roc_auc_neutral": 0.80,
                    "neutral_specificity": 0.70,
                },
            },
            {
                "method": "gradiend:M",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["M", "F"],
                    "roc_auc_neutral": 0.95,
                    "roc_auc_other": 0.90,
                    "class_exclusivity": 0.88,
                },
            },
            {
                "method": "gradiend:F",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["M", "F"],
                    "roc_auc_neutral": 0.70,
                    "roc_auc_other": 0.65,
                    "class_exclusivity": 0.60,
                },
            },
            {
                "method": "sae:M:k1",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.50, "neutral_specificity": 0.40},
            },
            {
                "method": "sae:M:kstar",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.92, "neutral_specificity": 0.85},
            },
            {
                "method": "sae:F:kstar",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.84, "neutral_specificity": 0.80},
            },
            {
                "method": "caa:M:L0_act_prediction",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.60, "neutral_specificity": 0.55},
            },
            {
                "method": "caa:M:act_prediction",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.99, "neutral_specificity": 0.90},
            },
            {
                "method": "caa:F:act_prediction",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.97, "neutral_specificity": 0.91},
            },
        ],
    }

    per = best_per_class_for_results(payload, metric="roc_auc_neutral")
    by = {(r["family"], r["feature_class"]): r for r in per}
    assert abs(by[("gradiend", "M")]["value"] - 0.95) < 1e-9
    assert "asian-white" in by[("gradiend", "M")]["winner_method"] or by[("gradiend", "M")]["winner_method"].endswith(":M")
    assert abs(by[("gradiend", "F")]["value"] - 0.70) < 1e-9
    assert by[("sae", "M")]["winner_method"] == "sae:M:kstar"
    assert abs(by[("caa", "M")]["value"] - 0.99) < 1e-9

    cells = aggregate_family_cells(per)
    g = next(c for c in cells if c["family"] == "gradiend")
    assert abs(g["mean"] - (0.95 + 0.70) / 2) < 1e-9
    assert abs(g["min"] - 0.70) < 1e-9
    assert abs(g["max"] - 0.95) < 1e-9

    text = format_overview_text(cells, metric="roc_auc_neutral", model="gpt2-small")
    assert "gradiend" in text
    assert "0.825" in text
    assert "Family x task" in text

    tex = format_overview_latex(cells, metric="roc_auc_neutral", model="gpt2-small")
    assert r"\begin{tabular}" in tex
    assert "gradiend" in tex
    assert "--" in tex  # en-dash range


def test_format_cell_single_and_range():
    assert format_cell(0.9, 0.9, 0.9) == "0.900"
    assert format_cell(0.85, 0.7, 0.95) == "0.850 [0.700-0.950]"
    assert format_cell(None, None, None) == "NaN"
    assert format_cell(None, None, None, expected=False) == "-"
    assert format_cell(float("nan"), float("nan"), float("nan")) == "NaN"


def test_family_missing_is_nan_structural_gap_is_dash():
    cells = [
        {
            "model": "gpt2-small",
            "task": "gender_en",
            "family": "gradiend",
            "metric": "roc_auc_neutral",
            "mean": 0.9,
            "min": 0.9,
            "max": 0.9,
            "n_classes": 1,
        },
        {
            "model": "gpt2-small",
            "task": "race_one_pole",
            "family": "gradiend",
            "metric": "roc_auc_neutral",
            "mean": 0.8,
            "min": 0.8,
            "max": 0.8,
            "n_classes": 1,
        },
    ]
    text = format_overview_text(cells, metric="roc_auc_neutral", model="gpt2-small")
    # SAE is enabled on gender_en but absent from the dump → missing.
    assert "NaN" in text
    # race_one_pole disables SAE/CAA → expected gap.
    sae_line = next(ln for ln in text.splitlines() if ln.startswith("sae"))
    caa_line = next(ln for ln in text.splitlines() if ln.startswith("caa"))
    assert "-" in sae_line
    assert "-" in caa_line


def test_pair_encoder_joins_short_causal_for_suitability():
    payload = {
        "model": "gpt2-small",
        "task": "emotion",
        "status": "ok",
        "config": {
            "target_classes": ["positive", "negative"],
            "claim_classes": ["positive", "negative"],
            "ablations": {"pair": True, "one_pole": True},
        },
        "methods": [
            {
                "method": "gradiend:negative-positive:negative",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["negative", "positive"],
                    "roc_auc_neutral": 0.95,
                    "roc_auc_other": 0.99,
                    "class_exclusivity": 1.0,
                    "suitability_E": 0.95,
                    "suitability_reason": "no_causal",
                },
            },
            {
                "method": "gradiend:negative-positive:positive",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["negative", "positive"],
                    "roc_auc_neutral": 0.99,
                    "roc_auc_other": 0.99,
                    "class_exclusivity": 1.0,
                    "suitability_E": 0.99,
                    "suitability_reason": "no_causal",
                },
            },
            {
                "method": "gradiend:negative",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.0, "causal_null_effect": True},
            },
            {
                "method": "gradiend:positive",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.0, "causal_null_effect": True},
            },
        ],
    }
    per = best_per_class_for_results(payload, metric="suitability")
    by = {(r["family"], r["feature_class"]): r for r in per}
    assert by[("gradiend", "negative")]["value"] == 0.0
    assert by[("gradiend", "positive")]["value"] == 0.0
    cells = aggregate_family_cells(per)
    g = next(c for c in cells if c["family"] == "gradiend")
    assert g["mean"] == 0.0
    text = format_overview_text(cells, metric="suitability", model="gpt2-small")
    g_line = next(ln for ln in text.splitlines() if ln.startswith("gradiend"))
    assert "0.000" in g_line
    assert "NaN" not in g_line.split("gradiend", 1)[1]
