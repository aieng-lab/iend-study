"""Pair-qualified ids, SAE layer encoder inherit, decoder metric retry, causal notes."""

from __future__ import annotations

from causal_eval import evaluate_decoder_for_classes
from results_schema import (
    causal_table,
    encoder_ablation_methods,
    encoder_inherit_source,
    resolve_encoder_metrics,
)
from study.method_ids import feature_or_causal_id, pair_feature_id, pair_trainer_id
from study.stages import sae as sae_stage


def test_pair_feature_ids():
    assert pair_trainer_id("gradiend", "white", "asian") == "gradiend:asian-white"
    assert pair_feature_id("gradiend", "white", "asian", "white") == "gradiend:asian-white:white"
    assert (
        pair_feature_id("actiend", "white", "asian", "asian", split_mode="tensors")
        == "actiend:asian-white:tensors:asian"
    )
    raw = {"ablation": "pair", "pair": ["asian", "white"]}
    assert (
        feature_or_causal_id("gradiend", "white", raw)
        == "gradiend:asian-white:white"
    )
    assert (
        feature_or_causal_id(
            "actiend",
            "white",
            raw,
            token_selector="all",
            activation_gate="encoder_direction",
        )
        == "actiend:asian-white:white:tok_all_gate_encoder_direction"
    )
    assert (
        feature_or_causal_id("gradiend", "RESULT", {"ablation": "one_pole"})
        == "gradiend:RESULT"
    )


def test_inherit_pair_tok_and_sae_layer_k1():
    assert encoder_inherit_source("sae:africa:L0_k1") == "sae:africa:L0"
    assert (
        encoder_inherit_source("actiend:asian-white:white:tok_all_gate_encoder_direction")
        == "actiend:asian-white:white"
    )
    assert encoder_inherit_source("sae:africa:k1_tok_prediction") == "sae:africa:k1"
    assert encoder_inherit_source("sae:africa:k1_pre_tok_prediction") == "sae_pre:africa:k1"


def test_sae_layer_k1_hidden_without_encoder_data():
    results = {
        "config": {
            "target_classes": ["africa"],
            "claim_classes": ["africa"],
            "enabled_methods": ["sae"],
        },
        "methods": [
            {
                "method": "sae:africa:L0_k1",
                "status": "ok",
                "metrics": {
                    "causal_selected_strength": 50.0,
                    "causal_signed_effect": 0.001,
                    "readout_kind": "causal_only",
                },
            }
        ],
    }
    assert "sae:africa:L0_k1" not in encoder_ablation_methods(results)


def test_sae_layer_k1_shows_by_layer_readouts():
    results = {
        "config": {
            "target_classes": ["africa"],
            "claim_classes": ["africa"],
            "enabled_methods": ["sae"],
        },
        "methods": [
            {
                "method": "sae:africa:L0_k1",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.001, "readout_kind": "causal_only"},
            }
        ],
        "raw": {
            "sae": {
                "by_layer_readouts": {
                    "0": {
                        "africa": {
                            "roc_auc_neutral": 1.0,
                            "roc_auc_other": 1.0,
                            "class_exclusivity": 1.0,
                            "balanced_accuracy": 1.0,
                        }
                    }
                }
            },
            "causal": {"by_method": {"sae:africa:L0_k1": {}}},
        },
    }
    m, src, inherited = resolve_encoder_metrics(results, "sae:africa:L0_k1")
    assert inherited is True
    assert src == "sae:africa:L0"
    assert m.get("roc_auc_neutral") == 1.0
    assert "sae:africa:L0_k1" in encoder_ablation_methods(results)


def test_display_method_id_uncollapses_legacy_pair_rows():
    results = {
        "config": {
            "target_classes": ["asian", "white"],
            "claim_classes": ["asian", "white"],
            "enabled_methods": ["gradiend"],
        },
        "methods": [
            {
                "method": "gradiend:white",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["asian", "white"],
                    "roc_auc_neutral": 0.9,
                    "roc_auc_other": 0.8,
                    "class_exclusivity": 0.7,
                },
            },
            {
                "method": "gradiend:white",
                "status": "ok",
                "metrics": {
                    "ablation": "one_pole",
                    "roc_auc_neutral": 0.4,
                    "roc_auc_other": 0.5,
                    "class_exclusivity": 0.6,
                },
            },
        ],
    }
    from results_schema import display_method_id, encoder_ablation_methods, primary_fair_methods

    assert display_method_id(results["methods"][0]) == "gradiend:asian-white:white"
    assert display_method_id(results["methods"][1]) == "gradiend:white"
    methods = encoder_ablation_methods(results)
    assert "gradiend:asian-white:white" in methods
    assert "gradiend:white" in methods
    primary = primary_fair_methods(results)
    assert "gradiend:asian-white:white" in primary
    assert "gradiend:white" in primary
    results = {
        "config": {
            "target_classes": ["asian", "black", "white"],
            "claim_classes": ["asian", "black", "white"],
            "enabled_methods": ["gradiend", "actiend"],
        },
        "methods": [
            {"method": "gradiend:asian-white:white", "metrics": {"ablation": "pair"}},
            {"method": "gradiend:white", "metrics": {"ablation": "one_pole"}},
            {"method": "actiend:asian-white:white", "metrics": {"ablation": "pair"}},
        ],
    }
    primary = primary_fair_methods(results)
    assert "gradiend:asian-white:white" in primary
    assert "gradiend:white" in primary
    assert "actiend:asian-white:white" in primary


def test_onepole_decoder_pins_claim_not_cf():
    """One-pole: never request CF names even if caller passes full task classes."""
    calls = []

    class Trainer:
        config = type("C", (), {"target_classes": ["MATCH"]})()

        def evaluate_decoder(self, **kw):
            calls.append(kw)
            sm = list(kw.get("summary_metrics") or [])
            if "DISTRACTOR" in sm:
                raise ValueError(
                    "Requested metrics not present in decoder results: ['DISTRACTOR']. "
                    "Available metrics: ['MATCH', 'MATCH_weaken']"
                )
            return {"MATCH": {"feature_factor": 1.0, "learning_rate": 1.0}, "grid": {}}

    out = evaluate_decoder_for_classes(
        Trainer(), ["MATCH", "DISTRACTOR"],
        split="validation", max_size=64, use_cache=False
    )
    assert "MATCH" in out
    assert calls[0]["target_class"] == ["MATCH"]
    assert calls[0]["summary_metrics"] == ["MATCH"]
    assert len(calls) == 1


def test_decoder_metric_error_raises_immediately_no_retry():
    """
    A "requested metrics not present" error must raise immediately, not be
    silently retried with a narrowed class list. An earlier version caught
    this exact error, parsed which metrics the grid actually scored, and
    retried with those -- provably dead in practice (the pre-filter already
    narrows to claim classes before the first call) and a real risk if
    ``_trainer_claim_classes`` were ever wrong for some trainer shape.
    Removed entirely; this test locks in that removal.
    """
    msg = (
        "Requested metrics not present in decoder results: ['1']. "
        "Available metrics: ['1_weaken', '2', '3']"
    )
    calls = []

    class Trainer:
        args = type("A", (), {"experiment_dir": None})()
        config = type("C", (), {"target_classes": ["1"]})()

        def evaluate_decoder(self, **kw):
            calls.append(kw)
            raise ValueError(msg)

    try:
        evaluate_decoder_for_classes(
            Trainer(), ["1"], split="validation", max_size=64, use_cache=False
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "['1']" in str(exc)
    assert len(calls) == 1, "must not retry -- exactly one call"
    assert calls[0].get("summary_metrics") == ["1"]


def test_missing_probability_panel_fails_without_learning_rate_retry():
    """A missing rival factual panel is a data-contract failure, not a bad LR."""
    msg = (
        "Missing decoder probability for selected metric in feature_factor=1.0, "
        "learning_rate=0.001: probs_by_dataset['M']['F'] is absent. "
        "Do not interpret missing decoder panels as probability 0.0; re-run "
        "decoder evaluation with full factual-class coverage."
    )
    calls = []

    class Trainer:
        args = type("A", (), {"experiment_dir": None})()
        config = type("C", (), {"target_classes": ["F"]})()

        def evaluate_decoder(self, **kw):
            calls.append(kw)
            raise ValueError(msg)

    try:
        evaluate_decoder_for_classes(
            Trainer(), ["F"], split="validation", lrs=[0.001, 0.01],
            max_size=64, use_cache=False
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert str(exc) == msg
    assert len(calls) == 1
    assert calls[0]["lrs"] == [0.001, 0.01]


def test_causal_table_shows_decoder_error_instead_of_blank_flag():
    results = {
        "config": {
            "target_classes": ["RESULT", "DISTRACTOR"],
            "claim_classes": ["RESULT"],
            "enabled_methods": ["actiend"],
        },
        "methods": [
            {
                "method": "actiend:RESULT:tok_all_gate_encoder_direction",
                "status": "error",
                "error": "Requested metrics not present in decoder results: ['DISTRACTOR']. Available metrics: ['RESULT', 'RESULT_weaken']",
                "metrics": {
                    "roc_auc_neutral": 0.98,
                    "causal_error": "Requested metrics not present in decoder results: ['DISTRACTOR']. Available metrics: ['RESULT', 'RESULT_weaken']",
                },
            }
        ],
        "raw": {"causal": {"by_method": {"actiend:RESULT:tok_all_gate_encoder_direction": {}}}},
    }
    text = causal_table(results)
    assert "actiend:RESULT:tok_all_gate_encoder_direction" in text
    assert "DISTRACTOR" in text
    assert "no causal" not in text.split("actiend:RESULT:tok_all_gate_encoder_direction", 1)[1][:200]


def test_sae_stage_imports_label_tokens_from_config():
    assert hasattr(sae_stage, "label_tokens_from_config")
    assert callable(sae_stage.label_tokens_from_config)


def test_short_class_method_aliases():
    from results_schema import short_class_method_aliases

    assert short_class_method_aliases("gradiend:negative-positive:negative") == (
        "gradiend:negative",
    )
    assert short_class_method_aliases(
        "actiend:negative-positive:negative:tok_all_gate_encoder_direction"
    ) == ("actiend:negative:tok_all_gate_encoder_direction",)
    assert short_class_method_aliases("gradiend:F-M:tensors:F") == ("gradiend:F:tensors",)
    assert short_class_method_aliases("gradiend:negative") == ()
    assert short_class_method_aliases("sae:positive:k1") == ()


def test_resolve_causal_joins_pair_encoder_to_short_class_id():
    from suitability import compute_feature_suitability, resolve_causal_metrics

    results = {
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
                "method": "gradiend:negative",
                "status": "ok",
                "metrics": {
                    "causal_signed_effect": 0.0,
                    "causal_lms": 0.1,
                    "causal_null_effect": True,
                },
            },
            {
                "method": "actiend:negative-positive:negative",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["negative", "positive"],
                    "roc_auc_neutral": 0.93,
                    "roc_auc_other": 0.99,
                    "class_exclusivity": 1.0,
                },
            },
            {
                "method": "actiend:negative:tok_all_gate_encoder_direction",
                "status": "ok",
                "metrics": {"causal_signed_effect": 0.0, "causal_lms": 0.1},
            },
        ]
    }
    g_cau, g_src = resolve_causal_metrics(results, "gradiend:negative-positive:negative")
    assert g_src == "gradiend:negative"
    assert g_cau.get("causal_signed_effect") == 0.0
    g_s = compute_feature_suitability(
        results["methods"][0]["metrics"], g_cau
    )
    assert g_s["suitability"] == 0.0
    assert g_s["suitability_G"] == 0.0
    assert g_s["suitability_reason"] == "G"

    a_cau, a_src = resolve_causal_metrics(results, "actiend:negative-positive:negative")
    assert a_src == "actiend:negative:tok_all_gate_encoder_direction"
    a_s = compute_feature_suitability(results["methods"][2]["metrics"], a_cau)
    assert a_s["suitability"] == 0.0
    assert a_s["suitability_reason"] == "G"


def test_causal_table_prints_pair_qualified_id_for_legacy_short_causal():
    from results_schema import pair_qualify_method_id

    results = {
        "config": {
            "target_classes": ["positive", "negative"],
            "claim_classes": ["positive", "negative"],
            "enabled_methods": ["gradiend", "actiend"],
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
                "method": "gradiend:negative",
                "status": "ok",
                "metrics": {
                    "causal_signed_effect": 0.0,
                    "causal_selected_strength": 1e-5,
                    "causal_lms": 0.1,
                    "causal_lms_ok": True,
                    "causal_null_effect": True,
                    "causal_grid_floor": True,
                },
            },
            {
                "method": "actiend:negative-positive:negative",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["negative", "positive"],
                    "roc_auc_neutral": 0.93,
                    "roc_auc_other": 0.99,
                    "class_exclusivity": 1.0,
                },
            },
            {
                "method": "actiend:negative:tok_all_gate_encoder_direction",
                "status": "ok",
                "metrics": {
                    "causal_signed_effect": 0.0,
                    "causal_lms": 0.1,
                    "causal_lms_ok": True,
                    "causal_null_effect": True,
                },
            },
        ],
    }
    assert (
        pair_qualify_method_id(results, "gradiend:negative")
        == "gradiend:negative-positive:negative"
    )
    assert (
        pair_qualify_method_id(results, "actiend:negative:tok_all_gate_encoder_direction")
        == "actiend:negative-positive:negative:tok_all_gate_encoder_direction"
    )
    text = causal_table(results)
    assert "gradiend:negative-positive:negative" in text
    assert "actiend:negative-positive:negative:tok_all_gate_encoder_direction" in text
    causal_block = text.split("Causal (LMS", 1)[1]
    assert "gradiend:negative " not in causal_block
    assert "actiend:negative:tok_all_gate_encoder_direction" not in causal_block


def test_pair_qualify_keeps_short_id_when_one_pole_encoder_exists():
    from results_schema import pair_qualify_method_id

    results = {
        "methods": [
            {
                "method": "gradiend:F-M:F",
                "status": "ok",
                "metrics": {
                    "ablation": "pair",
                    "pair": ["F", "M"],
                    "roc_auc_neutral": 1.0,
                    "roc_auc_other": 1.0,
                    "class_exclusivity": 1.0,
                },
            },
            {
                "method": "gradiend:F",
                "status": "ok",
                "metrics": {
                    "ablation": "one_pole",
                    "target_class": "F",
                    "roc_auc_neutral": 0.9,
                    "neutral_specificity": 0.8,
                    "causal_signed_effect": 0.2,
                },
            },
        ]
    }
    assert pair_qualify_method_id(results, "gradiend:F") == "gradiend:F"
