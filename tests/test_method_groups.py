from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest


def test_causal_table_policy_prefers_test_then_validation_and_hides_known_bad_cga():
    from analysis.method_groups import _apply_causal_table_policy

    validation = {
        "causal_selection_split": "validation",
        "causal_selection_signed_effect": 0.4,
        "causal_selection_lms": 0.9,
    }
    base_results = {"raw": {"causal": {"by_method": {}}}}
    kept = _apply_causal_table_policy(
        base_results,
        "gradiend:F",
        {"causal_signed_effect": 0.3},
        validation,
    )
    assert kept["causal_signed_effect"] == 0.3
    assert kept["causal_score_source"] == "test"

    fallback = _apply_causal_table_policy(
        {**base_results, "_causal_validation_fallback": True},
        "gradiend:F",
        {},
        validation,
    )
    assert fallback["causal_signed_effect"] == 0.4
    assert fallback["causal_lms"] == 0.9
    assert fallback["causal_score_source"] == "validation_fallback"

    bad_cga = {
        "raw": {
            "causal": {
                "by_method": {
                    "cga:F": {
                        "meta": {"polarity_reselected_from_bidirectional_grid": True}
                    }
                }
            }
        },
        "_causal_validation_fallback": True,
    }
    hidden = _apply_causal_table_policy(
        bad_cga,
        "cga:F",
        {"causal_signed_effect": 0.01, "causal_delta_mean": 0.01},
        validation,
    )
    assert hidden["causal_signed_effect"] == 0.4
    assert hidden["causal_delta_mean"] == 0.4
    assert hidden["causal_score_source"] == "validation_fallback_invalid_test"

    unversioned_cga = _apply_causal_table_policy(
        {**base_results, "_causal_validation_fallback": True},
        "cga:F",
        {"causal_signed_effect": 0.01},
        validation,
    )
    assert unversioned_cga["causal_signed_effect"] == 0.4
    assert unversioned_cga["causal_score_source"] == "validation_fallback_invalid_test"

    for stale_method in ("agiend:F", "caga:F"):
        stale = _apply_causal_table_policy(
            {**base_results, "_causal_validation_fallback": True},
            stale_method,
            {"causal_signed_effect": 0.01},
            validation,
        )
        assert stale["causal_signed_effect"] == 0.4
        assert stale["causal_score_source"] == "validation_fallback_invalid_test"


def test_causal_table_policy_keeps_legacy_scores_visible_unless_strict(monkeypatch):
    from analysis.method_groups import _apply_causal_table_policy

    results = {"raw": {"causal": {"by_method": {}}}}
    metrics = {"causal_signed_effect": 0.3, "causal_delta_mean": 0.3}

    visible = _apply_causal_table_policy(results, "cga:F", metrics, {})
    assert visible["causal_signed_effect"] == 0.3
    assert visible["causal_score_source"] == "legacy_unvalidated_artifact_grid_provenance"

    monkeypatch.setenv("STRICT_CAUSAL_PROVENANCE", "1")
    suppressed = _apply_causal_table_policy(results, "cga:F", metrics, {})
    assert suppressed["causal_signed_effect"] is None
    assert suppressed["causal_score_source"] == "invalid_artifact_grid_provenance"

from analysis.method_groups import (
    _collect_buckets,
    collect_group_rows,
    collect_group_rows_for_results,
    format_overview_text,
    format_pivot_cell,
    pivot_group_task,
)


def _write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cga_and_caga_layer_candidates_are_validation_locked_into_headlines():
    def _row(method: str, backend: str, test_auc: float, val_auc: float) -> dict:
        return {
            "method": method,
            "status": "ok",
            "metrics": {
                "backend": backend,
                "ablation": "pair",
                "target_class": "F",
                "roc_auc_neutral": test_auc,
                "roc_auc_other": test_auc,
                "val_readout": {
                    "roc_auc_neutral": val_auc,
                    "roc_auc_other": val_auc,
                    "neutral_specificity": val_auc,
                    "class_exclusivity": val_auc,
                },
            },
        }

    payload = {
        "model": "gpt2-small",
        "task": "gender_en",
        "status": "ok",
        "config": {
            "target_classes": ["F", "M"],
            "claim_classes": ["F"],
            "enabled_methods": ["cga", "caga"],
            "ablations": {"pair": True, "one_pole": False},
        },
        "methods": [
            _row("cga:F-M:F", "cga", 0.99, 0.60),
            _row("cga:F-M:F:L1", "cga", 0.80, 0.90),
            _row("caga:F-M:F", "caga", 0.98, 0.55),
            _row("caga:F-M:F:L2", "caga", 0.82, 0.92),
        ],
    }
    groups = {
        row["method_group"]: row for row in collect_group_rows_for_results(payload)
    }
    assert groups["cga:two_pole"]["roc_auc_neutral"] == 0.80
    assert groups["caga:two_pole"]["roc_auc_neutral"] == 0.82
    assert groups["cga:two_pole"]["source_methods"] == "cga:F-M:F:L1"
    assert groups["caga:two_pole"]["source_methods"] == "caga:F-M:F:L2"

    all_layer = {
        row["method_group"]: row
        for row in collect_group_rows_for_results(payload, representation_view="all_layer")
    }
    single_layer = {
        row["method_group"]: row
        for row in collect_group_rows_for_results(payload, representation_view="best_single_layer")
    }
    assert all_layer["cga:two_pole"]["source_methods"] == "cga:F-M:F"
    assert all_layer["cga:two_pole"]["roc_auc_neutral"] == 0.99
    assert single_layer["caga:two_pole"]["source_methods"] == "caga:F-M:F:L2"
    assert single_layer["caga:two_pole"]["roc_auc_neutral"] == 0.82


def test_aggregate_only_cga_caga_are_unavailable_for_layerwise_headline():
    """Old core rows must not masquerade as all-vs-layer-selected headlines."""
    def _row(method: str, backend: str) -> dict:
        return {
            "method": method,
            "status": "ok",
            "metrics": {
                "backend": backend,
                "ablation": "pair",
                "target_class": "F",
                "roc_auc_neutral": 0.9,
                "roc_auc_other": 0.9,
                "val_readout": {
                    "roc_auc_neutral": 0.9,
                    "roc_auc_other": 0.9,
                    "neutral_specificity": 0.9,
                    "class_exclusivity": 0.9,
                },
            },
        }

    payload = {
        "model": "llama-3.1-8b",
        "task": "gender_en",
        "status": "ok",
        "config": {
            "target_classes": ["F", "M"],
            "claim_classes": ["F"],
            "enabled_methods": ["cga", "caga"],
            "ablations": {"pair": True, "one_pole": False},
        },
        "methods": [_row("cga:F-M:F", "cga"), _row("caga:F-M:F", "caga")],
    }
    groups = collect_group_rows_for_results(payload)
    names = {str(row["method_group"]) for row in groups}
    assert "cga:two_pole" not in names
    assert "caga:two_pole" not in names


@pytest.mark.parametrize("backend", ["agiend"])
def test_pair_base_readouts_join_to_split_causal_rows(backend: str):
    """AGIEND stores encoder and causal payloads in separate rows.

    CGA/CAGA use per-class aggregate + per-layer candidates instead (see
    ``test_aggregate_only_cga_caga_are_unavailable_for_layerwise_headline``).
    """
    def readout(cls: str, auc: float) -> dict:
        return {
            "target_class": cls,
            "roc_auc": auc,
            "roc_auc_neutral": auc,
            "roc_auc_other": auc - 0.01,
            "neutral_specificity": auc - 0.02,
            "class_exclusivity": auc - 0.03,
            "val_readout": {
                "roc_auc_neutral": auc - 0.04,
                "roc_auc_other": auc - 0.05,
                "neutral_specificity": auc - 0.06,
                "class_exclusivity": auc - 0.07,
            },
        }

    def causal(method: str, effect: float) -> dict:
        protocol_meta = (
            {
                "agiend_causal_grid_protocol_version": 2,
                "direction_polarity_protocol_version": 2,
            }
            if backend == "agiend"
            else {"direction_polarity_protocol_version": 2}
        )
        return {
            "method": method,
            "status": "ok",
            "metrics": {"readout_kind": "causal_only"},
            "extras": {
                "causal": {
                    "selected": {"signed_effect": effect, "strength": 1.0},
                    "meta": protocol_meta,
                }
            },
        }

    payload = {
        "model": "gpt2-small",
        "task": "gender_en",
        "status": "ok",
        "config": {
            "target_classes": ["F", "M"],
            "claim_classes": ["F", "M"],
            "enabled_methods": [backend],
            "ablations": {"pair": True, "one_pole": False},
        },
        "methods": [
            {
                "method": f"{backend}:F-M",
                "status": "ok",
                "metrics": {"correlation": 0.8},
                "extras": {
                    "per_class_readouts": {
                        "F": readout("F", 0.95),
                        "M": readout("M", 0.90),
                    }
                },
            },
            {
                "method": f"{backend}:F-M|proxy=her_his",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.99},
            },
            causal(f"{backend}:F-M:F", 0.2),
            causal(f"{backend}:F-M:M", 0.4),
        ],
    }

    collected = _collect_buckets(payload, "")
    assert collected is not None
    assert collected[1]
    groups = {
        row["method_group"]: row
        for row in collect_group_rows_for_results(payload)
    }
    assert f"{backend}:two_pole" in groups
    row = groups[f"{backend}:two_pole"]
    assert row["detection_score"] is not None
    assert abs(row["causal_signed_effect"] - 0.3) < 1e-12
    assert "proxy=" not in row["source_methods"]


def test_gradiend_two_pole_and_one_pole_groups():
    root = Path(__file__).resolve().parents[1] / ".test_scratch_method_groups"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    runs = root / "runs"
    try:
        payload = {
            "model": "gpt2-small",
            "task": "gender_en",
            "status": "ok",
            "config": {
                "target_classes": ["M", "F"],
                "claim_classes": ["M", "F"],
                "ablations": {"pair": True, "one_pole": True},
            },
            "methods": [
                {
                    "method": "gradiend",
                    "status": "ok",
                    "metrics": {
                        "ablation": "pair",
                        "roc_auc_neutral": 1.0,
                        "roc_auc_other": 0.9,
                        "class_exclusivity": 0.9,
                    },
                },
                {
                    "method": "gradiend:M",
                    "status": "ok",
                    "metrics": {
                        "ablation": "pair",
                        "roc_auc_neutral": 0.8,
                        "roc_auc_other": 0.8,
                        "class_exclusivity": 0.8,
                    },
                },
                {
                    "method": "gradiend:F",
                    "status": "ok",
                    "metrics": {
                        "ablation": "pair",
                        "roc_auc_neutral": 0.6,
                        "roc_auc_other": 0.6,
                        "class_exclusivity": 0.6,
                    },
                },
                {
                    "method": "gradiend:M",
                    "status": "ok",
                    "metrics": {
                        "ablation": "one_pole",
                        "roc_auc_neutral": 0.95,
                        "neutral_specificity": 0.9,
                    },
                },
                {
                    "method": "gradiend:F",
                    "status": "ok",
                    "metrics": {
                        "ablation": "one_pole",
                        "roc_auc_neutral": 0.85,
                        "neutral_specificity": 0.8,
                    },
                },
                {
                    "method": "gradiend:M:all",
                    "status": "ok",
                    "metrics": {"ablation": "pair", "roc_auc_neutral": 0.99},
                },
            ],
        }
        path = runs / "gpt2-small" / "gender_en" / "results.json"
        _write_results(path, payload)

        rows = collect_group_rows_for_results(payload, results_path=str(path))
        groups = {r["method_group"]: r for r in rows}
        assert "gradiend:two_pole" in groups
        assert "gradiend:one_pole" in groups
        assert abs(groups["gradiend:two_pole"]["roc_auc_neutral"] - 0.8) < 1e-9
        assert abs(groups["gradiend:one_pole"]["roc_auc_neutral"] - 0.9) < 1e-9
        assert groups["gradiend:two_pole"]["n_sources"] == 3
        assert "gradiend:M:all" not in groups["gradiend:two_pole"]["source_methods"]

        df = collect_group_rows(runs)
        assert len(df) >= 2
        assert set(df["method_group"]) >= {"gradiend:two_pole", "gradiend:one_pole"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_one_pole_task_uses_claim_class_only():
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "partial",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "gradiend:MATCH",
                "status": "ok",
                "metrics": {"ablation": "one_pole", "roc_auc_neutral": 0.9},
            },
            {
                "method": "caa:DISTRACTOR:act_prediction",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.5},
            },
            {
                "method": "caa:MATCH:act_prediction",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 0.7},
            },
        ],
    }
    rows = collect_group_rows_for_results(payload)
    groups = {r["method_group"]: r for r in rows}
    assert groups["gradiend:one_pole"]["n_sources"] == 1
    assert groups["caa:one_pole"]["roc_auc_neutral"] == 0.7
    assert "DISTRACTOR" not in groups["caa:one_pole"]["source_methods"]


def test_group_rows_expose_complete_detection_and_intervention_headlines():
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "ok",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "gradiend:MATCH",
                "status": "ok",
                "metrics": {
                    "ablation": "one_pole",
                    "roc_auc_neutral": 0.9,
                    "roc_auc_other": 0.8,
                    "neutral_specificity": 0.75,
                    "class_exclusivity": 0.7,
                    "causal_signed_effect": 0.3,
                    "causal_signed_effect_weaken": 0.1,
                },
            }
        ],
        "raw": {
            "causal": {
                "by_method": {
                    "gradiend:MATCH": {
                        "weaken_selected_strength": 1.0,
                        "weaken_selected": {"signed_effect": 0.1},
                    }
                }
            }
        },
    }
    group = collect_group_rows_for_results(payload)[0]
    assert group["detection_score"] == 0.7
    assert group["intervention_score"] == 0.2
    assert group["detection_complete"] is True
    assert group["intervention_complete"] is True


def test_actiend_group_pulls_tok_causal():
    """Bare ``actiend:{cls}`` is encode-only; causal lives on ``:tok_*`` children."""
    payload = {
        "model": "gpt2-small",
        "task": "gender_en",
        "status": "partial",
        "config": {
            "target_classes": ["M", "F"],
            "claim_classes": ["M", "F"],
            "ablations": {"pair": True, "one_pole": False},
            "suitability": {"causal_effect_threshold": 0.05, "soft_causal": False},
        },
        "methods": [
            {
                "method": "actiend:M",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "roc_auc_neutral": 0.99,
                    "roc_auc_other": 1.0,
                    "class_exclusivity": 0.95,
                    "suitability": 0.0,
                    "suitability_E": 0.95,
                    "suitability_G": 0.0,
                },
            },
            {
                "method": "actiend:F",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "roc_auc_neutral": 0.98,
                    "roc_auc_other": 1.0,
                    "class_exclusivity": 0.9,
                    "suitability": 0.0,
                    "suitability_E": 0.9,
                    "suitability_G": 0.0,
                },
            },
            {
                "method": "actiend:M:tok_all_gate_encoder_direction",
                "status": "ok",
                "metrics": {
                    "causal_signed_effect": 0.15,
                    "causal_lms": 0.09,
                    "causal_lms_ok": True,
                    "causal_selected_strength": 50.0,
                },
            },
            {
                "method": "actiend:F:tok_all",
                "status": "ok",
                "metrics": {
                    "causal_signed_effect": 0.16,
                    "causal_lms": 0.09,
                    "causal_lms_ok": True,
                    "causal_selected_strength": 20.0,
                },
            },
            {
                "method": "actiend:M:tok_prediction",
                "status": "ok",
                "metrics": {
                    "causal_signed_effect": 0.01,
                    "causal_lms_ok": True,
                },
            },
        ],
    }
    rows = collect_group_rows_for_results(payload)
    groups = {r["method_group"]: r for r in rows}
    g = groups["actiend:two_pole"]
    assert g["causal_signed_effect"] is not None
    # Prefer gate for M (0.15) and tok_all for F (0.16) → mean ~0.155
    assert abs(g["causal_signed_effect"] - 0.155) < 1e-9
    assert g["suitability_G"] == 1.0
    assert g["suitability"] is not None and g["suitability"] > 0.8


def test_actiend_ridge_causal_only_rows_are_included_in_headline_groups():
    payload = {
        "model": "gpt2-small",
        "task": "emotion",
        "status": "ok",
        "config": {
            "target_classes": ["negative", "positive"],
            "claim_classes": ["negative", "positive"],
            "ablations": {"pair": True, "one_pole": False},
        },
        "methods": [
            {
                "method": "actiend_ridge:negative-positive:negative:tok_all_gate_encoder_direction",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.02, "causal_lms": 0.09},
            },
            {
                "method": "actiend_ridge:negative-positive:positive:tok_all_gate_encoder_direction",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.04, "causal_lms": 0.10},
            },
        ],
    }

    groups = {row["method_group"]: row for row in collect_group_rows_for_results(payload)}

    ridge = groups["actiend_ridge:two_pole"]
    assert ridge["encoding_E"] is None
    assert abs(ridge["causal_signed_effect"] - 0.03) < 1e-12
    assert abs(ridge["causal_lms"] - 0.095) < 1e-12


def test_caa_encode_payload_restores_rows_overwritten_by_legacy_causal_refresh():
    payload = {
        "model": "gpt2-small",
        "task": "emotion",
        "status": "ok",
        "config": {"target_classes": ["negative", "positive"], "claim_classes": ["negative", "positive"]},
        "methods": [
            {
                "method": "caa:negative:act_prediction",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.01},
            },
            {
                "method": "caa:positive:act_prediction",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.03},
            },
        ],
        "raw": {
            "caa": {
                "method_metrics": {
                    "caa:negative:act_prediction": {
                        "target_class": "negative", "roc_auc_neutral": 0.8, "specificity": 0.75
                    },
                    "caa:positive:act_prediction": {
                        "target_class": "positive", "roc_auc_neutral": 0.9, "specificity": 0.85
                    },
                }
            }
        },
    }
    groups = {row["method_group"]: row for row in collect_group_rows_for_results(payload)}
    caa = groups["caa:one_pole"]
    assert abs(caa["encoding_E"] - 0.8) < 1e-12
    assert abs(caa["causal_signed_effect"] - 0.02) < 1e-12


def test_actiend_ridge_is_hidden_from_encoder_pivots_but_present_in_causal_ones():
    df = pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "task": "emotion",
                "method_group": "actiend_ridge:two_pole",
                "encoding_E": None,
                "causal_signed_effect": 0.03,
            }
        ]
    )

    encoder = pivot_group_task(df, metric="encoding_E", model="gpt2-small")
    causal = pivot_group_task(df, metric="causal_signed_effect", model="gpt2-small")

    assert "actiend_ridge:two_pole" not in encoder.index
    assert causal.loc["actiend_ridge:two_pole", "emotion"] == 0.03


def test_missing_pivot_cells_nan_structural_gap_dash():
    assert format_pivot_cell(float("nan")) == "NaN"
    assert format_pivot_cell(None) == "NaN"
    assert format_pivot_cell(float("nan"), expected=False) == "-"
    assert format_pivot_cell(None, expected=False) == "-"

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "task": "gender_en",
                "method_group": "gradiend:two_pole",
                "roc_auc_neutral": 1.0,
            },
            {
                "model": "gpt2-small",
                "task": "induction",
                "method_group": "gradiend:one_pole",
                "roc_auc_neutral": 0.9,
            },
        ]
    )
    text = format_overview_text(df, metric="roc_auc_neutral", model="gpt2-small")
    two_pole = next(ln for ln in text.splitlines() if ln.startswith("gradiend:two_pole"))
    one_pole = next(ln for ln in text.splitlines() if ln.startswith("gradiend:one_pole"))
    # induction is pair:false → two_pole is an expected gap.
    assert "-" in two_pole
    # gender_en enables one_pole but the dump has no value → missing.
    assert "NaN" in one_pole
    # Present values stay numeric.
    assert "1.0000" in two_pole
    assert "0.9000" in one_pole

    from analysis.method_groups import format_pivot_display
    from analysis.task_specs import spec_for_task

    pivot = pivot_group_task(df, metric="roc_auc_neutral", model="gpt2-small")
    specs = {
        ("gpt2-small", "gender_en"): spec_for_task("gender_en", model="gpt2-small"),
        ("gpt2-small", "induction"): spec_for_task("induction", model="gpt2-small"),
    }
    rendered = format_pivot_display(pivot, specs=specs, model="gpt2-small").to_csv()
    assert "NaN" in rendered
    assert ",-," in rendered or ",-\n" in rendered or rendered.count("-") >= 1


def test_emotion_pair_rows_get_suitability_from_short_causal():
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
                },
            },
            {
                "method": "gradiend:negative",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.0, "causal_null_effect": True},
            },
        ],
    }
    rows = collect_group_rows_for_results(payload, results_path="runs/gpt2-small/emotion/results.json")
    groups = {r["method_group"]: r for r in rows}
    assert "gradiend:two_pole" in groups
    assert groups["gradiend:two_pole"]["suitability"] == 0.0
    assert groups["gradiend:two_pole"]["suitability_G"] == 0.0
    assert "gradiend:one_pole" not in groups


def test_sae_error_status_still_enters_group_when_metrics_present():
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "partial",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "sae:MATCH:k1",
                "status": "error",
                "error": "no probs_by_dataset panels containing 'MATCH'",
                "metrics": {
                    "roc_auc_neutral": 0.98,
                    "roc_auc_other": 0.57,
                    "class_exclusivity": 0.10,
                    "causal_signed_effect": 0.197,
                    "causal_lms": 0.107,
                },
            },
        ],
    }
    rows = collect_group_rows_for_results(payload)
    groups = {r["method_group"]: r for r in rows}
    assert "sae:k1" in groups
    assert abs(float(groups["sae:k1"]["causal_signed_effect"]) - 0.197) < 1e-9


def _enc(val_det=None, **kwargs):
    base = {
        "roc_auc_neutral": 0.5,
        "roc_auc_other": 0.5,
        "class_exclusivity": 0.5,
        "neutral_specificity": 0.5,
    }
    base.update(kwargs)
    if val_det is not None:
        # The site lock ranks on validation Detection = min over its inputs, so
        # a flat validation readout scores exactly ``val_det``.
        base["val_readout"] = {
            "roc_auc_neutral": val_det,
            "neutral_specificity": val_det,
            "roc_auc_other": val_det,
            "class_exclusivity": val_det,
        }
    return base


def test_sae_k1_locks_all_k1_when_e_higher():
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "ok",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "sae:MATCH:k1",
                "status": "ok",
                "metrics": _enc(val_det=0.6, roc_auc_neutral=0.6, roc_auc_other=0.6, class_exclusivity=0.6),
            },
            {
                "method": "sae:MATCH:all_k1",
                "status": "ok",
                "metrics": _enc(val_det=0.88, roc_auc_neutral=0.91, roc_auc_other=0.9, class_exclusivity=0.88),
            },
            {
                "method": "sae:MATCH:L3_k1",
                "status": "ok",
                "metrics": _enc(val_det=0.4, roc_auc_neutral=0.4, roc_auc_other=0.4, class_exclusivity=0.4),
            },
        ],
    }
    rows = collect_group_rows_for_results(payload)
    groups = {r["method_group"]: r for r in rows}
    assert groups["sae:k1"]["source_methods"] == "sae:MATCH:all_k1"
    assert abs(float(groups["sae:k1"]["roc_auc_neutral"]) - 0.91) < 1e-9


def test_caa_lock_ignores_mean_of_layer_all_act_diagnostic():
    # Candidates are concat (act_prediction) and per-layer only; all_act_* is an
    # appendix diagnostic even when its validation Detection is higher.
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "ok",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "caa:MATCH:act_prediction",
                "status": "ok",
                "metrics": _enc(val_det=0.7, roc_auc_neutral=0.7, roc_auc_other=0.7, class_exclusivity=0.7),
            },
            {
                "method": "caa:MATCH:all_act_prediction",
                "status": "ok",
                "metrics": _enc(val_det=0.83, roc_auc_neutral=0.85, roc_auc_other=0.84, class_exclusivity=0.83),
            },
            {
                "method": "caa:MATCH:L2_act_prediction",
                "status": "ok",
                "metrics": _enc(val_det=0.4, roc_auc_neutral=0.4, roc_auc_other=0.4, class_exclusivity=0.4),
            },
        ],
    }
    rows = collect_group_rows_for_results(payload)
    groups = {r["method_group"]: r for r in rows}
    assert set(groups) == {"caa:one_pole"}
    assert groups["caa:one_pole"]["source_methods"] == "caa:MATCH:act_prediction"
    assert abs(float(groups["caa:one_pole"]["roc_auc_neutral"]) - 0.7) < 1e-9


def test_legacy_caa_raw_keeps_validation_scalar_for_site_lock():
    def raw(auc: float, val_e: float) -> dict:
        return {
            "roc_auc": auc,
            "roc_auc_neutral": auc,
            "neutral_specificity": auc,
            "target_class": "MATCH",
            "ablation": "one_pole",
            "val_encoding_E": val_e,
        }

    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "ok",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [],
        "raw": {
            "caa": {
                "method_metrics": {
                    "caa:MATCH:act_prediction": raw(0.70, 0.90),
                    "caa:MATCH:L2_act_prediction": raw(0.99, 0.20),
                }
            }
        },
    }

    groups = {r["method_group"]: r for r in collect_group_rows_for_results(payload)}
    assert groups["caa:one_pole"]["source_methods"] == "caa:MATCH:act_prediction"
    assert groups["caa:one_pole"]["roc_auc_neutral"] == 0.70


def test_caga_encoder_rows_restore_from_done_json(tmp_path: Path):
    results_path = tmp_path / "gpt2-small" / "suite_full2" / "race" / "results.json"
    done = (
        results_path.parent
        / "artifacts"
        / "caga__onepole__asian"
        / "done.json"
    )
    done.parent.mkdir(parents=True)
    done.write_text(
        json.dumps(
            {
                "extras": {
                    "encoder_eval": {
                        "encoder_metrics": {"correlation": 0.8},
                        "per_class_readouts": {
                            "asian": _enc(
                                val_det=0.60,
                                roc_auc_neutral=0.95,
                                roc_auc_other=0.94,
                                class_exclusivity=0.93,
                            ),
                            # The encoder evaluator records distractor
                            # readouts too, but a one-pole artifact has no
                            # causal sweep for them.  Restoring this row would
                            # make the complete, valid asian group look
                            # causally incomplete and render it as NaN.
                            "black": _enc(
                                val_det=0.99,
                                roc_auc_neutral=0.99,
                                roc_auc_other=0.99,
                                class_exclusivity=0.99,
                            ),
                        },
                        "per_component_readouts": {
                            "L2": {
                                "asian": _enc(
                                    val_det=0.90,
                                    roc_auc_neutral=0.98,
                                    roc_auc_other=0.97,
                                    class_exclusivity=0.96,
                                ),
                                "black": _enc(
                                    val_det=1.00,
                                    roc_auc_neutral=1.00,
                                    roc_auc_other=1.00,
                                    class_exclusivity=1.00,
                                ),
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "model": "gpt2-small",
        "task": "race",
        "status": "ok",
        "config": {
            "target_classes": ["asian", "black", "white"],
            "claim_classes": ["asian"],
            "ablations": {"pair": True, "one_pole": True},
        },
        "methods": [
            {
                "method": "caga:asian",
                "status": "ok",
                "metrics": {
                    "target_class": "asian",
                    "readout_kind": "causal_only",
                },
                "extras": {
                    "causal": {
                        "selected_strength": 2.0,
                        "selected": {"strength": 2.0, "signed_effect": 0.2},
                        "weaken_selected_strength": 3.0,
                        "weaken_selected": {"strength": 3.0, "signed_effect": 0.1},
                        "meta": {"direction_polarity_protocol_version": 2},
                    }
                },
            },
            {
                "method": "caga:asian:L2",
                "status": "ok",
                "metrics": {
                    "target_class": "asian",
                    "readout_kind": "causal_only",
                },
                "extras": {
                    "causal": {
                        "selected_strength": 2.0,
                        "selected": {"strength": 2.0, "signed_effect": 0.2},
                        "weaken_selected_strength": 3.0,
                        "weaken_selected": {"strength": 3.0, "signed_effect": 0.1},
                        "meta": {"direction_polarity_protocol_version": 2},
                    }
                },
            },
        ],
    }
    _write_results(results_path, payload)

    groups = {
        row["method_group"]: row
        for row in collect_group_rows_for_results(
            payload, results_path=str(results_path)
        )
    }
    restored = groups["caga:one_pole"]
    assert restored["source_methods"] == "caga:asian:L2"
    assert restored["roc_auc_neutral"] == 0.98
    assert restored["causal_signed_effect"] == 0.2
    assert restored["causal_signed_effect_weaken"] == 0.1


def test_multi_candidate_lock_never_falls_back_to_test_e():
    # Multiple CAA sites, none carrying a validation readout, and the per-layer
    # candidate has the HIGHER test roc_auc_neutral. A test-E oracle would pick
    # L2 (0.99). There is nothing honest to select on, so the group must be
    # absent (NaN) rather than reporting either candidate.
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "ok",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "caa:MATCH:act_prediction",
                "status": "ok",
                "metrics": _enc(roc_auc_neutral=0.50),
            },
            {
                "method": "caa:MATCH:L2_act_prediction",
                "status": "ok",
                "metrics": _enc(roc_auc_neutral=0.99),
            },
        ],
    }
    groups = {r["method_group"]: r for r in collect_group_rows_for_results(payload)}
    assert "caa:one_pole" not in groups


def test_multi_candidate_lock_selects_on_validation_detection():
    # Same shape, but every candidate carries a validation readout: selection is
    # now possible and must follow validation, not the test metric.
    payload = {
        "model": "gpt2-small",
        "task": "induction",
        "status": "ok",
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "ablations": {"pair": False, "one_pole": True},
        },
        "methods": [
            {
                "method": "caa:MATCH:act_prediction",
                "status": "ok",
                "metrics": _enc(val_det=0.9, roc_auc_neutral=0.50),
            },
            {
                "method": "caa:MATCH:L2_act_prediction",
                "status": "ok",
                "metrics": _enc(val_det=0.2, roc_auc_neutral=0.99),
            },
        ],
    }
    groups = {r["method_group"]: r for r in collect_group_rows_for_results(payload)}
    assert groups["caa:one_pole"]["source_methods"] == "caa:MATCH:act_prediction"
    assert abs(float(groups["caa:one_pole"]["roc_auc_neutral"]) - 0.50) < 1e-9


def test_caa_pair_and_one_pole_are_separate_headline_groups():
    payload = {
        "model": "gpt2-small",
        "task": "three_class",
        "status": "ok",
        "config": {
            "target_classes": ["A", "B", "C"],
            "claim_classes": ["A", "B", "C"],
            "ablations": {"pair": True, "one_pole": True},
        },
        "methods": [
            {
                "method": "caa:A-B:A:act_prediction",
                "status": "ok",
                "metrics": {
                    **_enc(roc_auc_neutral=0.7, roc_auc_other=0.6),
                    "target_class": "A",
                    "ablation": "pair",
                    "pair": ["A", "B"],
                },
            },
            {
                "method": "caa:A-C:A:act_prediction",
                "status": "ok",
                "metrics": {
                    **_enc(roc_auc_neutral=0.9, roc_auc_other=0.8),
                    "target_class": "A",
                    "ablation": "pair",
                    "pair": ["A", "C"],
                },
            },
            {
                "method": "caa:A:act_prediction",
                "status": "ok",
                "metrics": {
                    **_enc(roc_auc_neutral=0.75, roc_auc_other=0.7),
                    "target_class": "A",
                    "ablation": "one_pole",
                },
            },
        ],
    }

    groups = {r["method_group"]: r for r in collect_group_rows_for_results(payload)}
    assert set(groups) == {"caa:two_pole", "caa:one_pole"}
    assert abs(float(groups["caa:two_pole"]["roc_auc_neutral"]) - 0.8) < 1e-9
    assert abs(float(groups["caa:one_pole"]["roc_auc_neutral"]) - 0.75) < 1e-9
