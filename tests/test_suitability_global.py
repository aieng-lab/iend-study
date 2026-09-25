from __future__ import annotations

import json
import shutil
from pathlib import Path

from suitability import (
    MEAN_LABEL,
    collect_suitability_method_rows,
    format_model_task_matrix,
    matrix_with_mean_margins,
    model_task_cell_means,
    intervention_score,
    detection_score,
    refresh_global_suitability,
    resolve_causal_metrics,
)


def test_detection_score_includes_specificity_without_changing_legacy_e():
    metrics = {
        "roc_auc_neutral": 0.95,
        "roc_auc_other": 0.92,
        "neutral_specificity": 0.61,
        "class_exclusivity": 0.88,
    }
    assert detection_score(metrics) == 0.61
    # Historical E remains the site/checkpoint selector used by old runs.
    from suitability import encoding_e

    assert encoding_e(metrics) == 0.88


def test_detection_score_one_pole_uses_auc_and_specificity():
    assert detection_score(
        {"roc_auc_neutral": 0.91, "neutral_specificity": 0.72}
    ) == 0.72


def test_intervention_score_is_bidirectional_mean_and_requires_both_directions():
    assert intervention_score(
        {"causal_signed_effect": 0.3, "causal_signed_effect_weaken": 0.1}
    ) == 0.2
    assert intervention_score({"causal_signed_effect": 0.3}) is None


def test_matrix_with_mean_margins():
    cells = {
        ("gpt2-small", "gender_en"): 0.8,
        ("gpt2-small", "ioi"): 0.4,
        ("pythia-70m", "gender_en"): 0.6,
        ("pythia-70m", "ioi"): 0.2,
    }
    models, tasks, out = matrix_with_mean_margins(cells)
    assert models[-1] == MEAN_LABEL
    assert tasks[-1] == MEAN_LABEL
    assert abs(out[("gpt2-small", MEAN_LABEL)] - 0.6) < 1e-12
    assert abs(out[(MEAN_LABEL, "gender_en")] - 0.7) < 1e-12
    assert abs(out[(MEAN_LABEL, MEAN_LABEL)] - 0.5) < 1e-12


def test_format_model_task_matrix_includes_mean():
    cells = {("m1", "t1"): 1.0, ("m1", "t2"): 0.0, ("m2", "t1"): 0.5}
    models, tasks, margined = matrix_with_mean_margins(cells)
    text = format_model_task_matrix(margined, models=models, tasks=tasks)
    assert "mean" in text
    assert "m1" in text
    assert "t1" in text


def test_refresh_global_suitability_writes_tables():
    # In-repo scratch: system Temp can be permission-locked on this host.
    root = Path(__file__).resolve().parents[1] / ".test_scratch_suitability"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    runs = root / "runs"
    out = root / "tables"
    try:
        for model, task, s_m, s_f in (
            ("gpt2-small", "gender_en", 0.9, 0.7),
            ("gpt2-small", "ioi", 0.1, 0.0),
            ("pythia-70m", "gender_en", 0.5, 0.5),
        ):
            path = runs / model / task / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "model": model,
                        "task": task,
                        "status": "ok",
                        "config": {
                            "target_classes": ["M", "F"],
                            "enabled_methods": ["actiend"],
                        },
                        "methods": [
                            {
                                "method": "actiend:M",
                                "status": "ok",
                                "metrics": {
                                    "roc_auc_neutral": 0.9,
                                    "roc_auc_other": 0.9,
                                    "class_exclusivity": 0.9,
                                    "suitability": s_m,
                                    "suitability_E": 0.9,
                                    "suitability_G": 1.0 if s_m else 0.0,
                                    "suitability_reason": "ok" if s_m else "G",
                                    "suitability_scope": "full",
                                    "causal_signed_effect": 0.1 if s_m else 0.0,
                                },
                            },
                            {
                                "method": "actiend:F",
                                "status": "ok",
                                "metrics": {
                                    "roc_auc_neutral": 0.9,
                                    "roc_auc_other": 0.9,
                                    "class_exclusivity": 0.9,
                                    "suitability": s_f,
                                    "suitability_E": 0.9,
                                    "suitability_G": 1.0 if s_f else 0.0,
                                    "suitability_reason": "ok" if s_f else "G",
                                    "suitability_scope": "full",
                                    "causal_signed_effect": 0.1 if s_f else 0.0,
                                },
                            },
                        ],
                        "raw": {
                            "suitability": {
                                "tau_c": 0.05,
                                "soft_causal": False,
                                "e_ok": 0.8,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

        result = refresh_global_suitability(runs, out, primary_only=True)
        assert Path(result["paths"]["model_task_csv"]).is_file()
        assert Path(result["paths"]["long_csv"]).is_file()
        assert Path(result["paths"]["model_task_txt"]).is_file()
        assert "gpt2-small" in result["overall_text"]
        assert MEAN_LABEL in result["overall_text"]

        cells = model_task_cell_means(
            collect_suitability_method_rows(runs, primary_only=True)
        )
        assert abs(cells[("gpt2-small", "gender_en")] - 0.8) < 1e-9
        assert abs(cells[("gpt2-small", "ioi")] - 0.05) < 1e-9
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resolve_causal_metrics_reads_inherited_raw_by_method():
    results = {
        "methods": [
            {
                "method": "sae:IO:kstar",
                "status": "ok",
                "metrics": {
                    "roc_auc_neutral": 0.9,
                    "roc_auc_other": 0.8,
                    "class_exclusivity": 0.7,
                },
            }
        ],
        "raw": {
            "causal": {
                "by_method": {
                    "sae:IO:kstar_tok_prediction": {
                        "selected_strength": 0.1,
                        "selected": {"signed_effect": 0.02, "lms": 0.09, "lms_ok": True},
                    }
                }
            }
        },
    }
    m, src = resolve_causal_metrics(results, "sae:IO:kstar")
    assert src == "sae:IO:kstar_tok_prediction"
    assert abs(float(m.get("causal_signed_effect")) - 0.02) < 1e-12


def test_resolve_causal_metrics_does_not_invent_weaken_from_legacy_strengthen_curve():
    results = {
        "methods": [
            {
                "method": "sae:positive:kstar",
                "metrics": {"roc_auc_neutral": 0.9, "causal_signed_effect": 0.01},
            }
        ],
        "raw": {
            "causal": {
                "by_method": {
                    "sae:positive:kstar": {
                        "target_class": "positive",
                        "strengths": [
                            {
                                "strength": 1.0,
                                "lms": 0.99,
                                "base_lms": 1.0,
                                "summary_by_group": {
                                    "positive": {"mean_p_target_base": 0.8, "mean_p_target_mod": 0.6}
                                },
                            },
                            {
                                "strength": 2.0,
                                "lms": 0.9,
                                "base_lms": 1.0,
                                "summary_by_group": {
                                    "positive": {"mean_p_target_base": 0.8, "mean_p_target_mod": 0.1}
                                },
                            },
                        ],
                    }
                }
            }
        },
    }
    metrics, _src = resolve_causal_metrics(results, "sae:positive:kstar")
    assert metrics.get("causal_signed_effect_weaken") is None


def test_resolve_causal_metrics_keeps_separately_selected_weaken():
    results = {
        "methods": [
            {
                "method": "sae:positive:kstar",
                "metrics": {
                    "causal_signed_effect": 0.3,
                    "causal_signed_effect_weaken": 0.2,
                },
            }
        ],
        "raw": {
            "causal": {
                "by_method": {
                    "sae:positive:kstar": {
                        "weaken_selected_strength": 2.0,
                        "weaken_selected": {"signed_effect": 0.2},
                    }
                }
            }
        },
    }
    metrics, _src = resolve_causal_metrics(results, "sae:positive:kstar")
    assert metrics["causal_signed_effect_weaken"] == 0.2


def test_resolve_causal_metrics_promotes_weaken_from_method_extras():
    results = {
        "methods": [
            {
                "method": "actiend:F-M:F:tok_prediction",
                "metrics": {},
                "extras": {
                    "causal": {
                        "selected": {"signed_effect": 0.3, "lms": 0.09, "lms_ok": True},
                        "weaken_selected_strength": 0.5,
                        "weaken_selected": {"signed_effect": 0.2, "lms": 0.08, "lms_ok": True},
                    }
                },
            }
        ]
    }
    metrics, _src = resolve_causal_metrics(
        results, "actiend:F-M:F:tok_prediction"
    )
    assert metrics["causal_signed_effect"] == 0.3
    assert metrics["causal_lms"] == 0.09
    assert metrics["causal_lms_ok"] is True
    assert metrics["causal_signed_effect_weaken"] == 0.2
    assert metrics["causal_weaken_lms"] == 0.08
    assert metrics["causal_weaken_lms_ok"] is True
