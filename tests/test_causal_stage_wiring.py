from __future__ import annotations

import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.config import StudyConfig
from study.stages.causal import build_causal_study_kwargs
from study.stages.sae import sae_suite_params
from study.causal_policies import (
    ACTIEND_CAUSAL_POLICY_DEFAULTS,
    actiend_ablation_policy_specs,
    actiend_default_policy_enabled,
    normalize_actiend_causal_policies,
)


def test_sae_suite_params_core_fixed_ks():
    params = sae_suite_params(["k1", "kstar", "all_k1"])
    assert params["fixed_ks"] == (1,)
    assert params["opp_fire"] is False
    assert params["arad"] is False
    assert params["jh_f1"] is False


def test_sae_suite_params_full_ablations():
    params = sae_suite_params(["k1", "opp_fire", "arad_out", "jh_f1", "k4"])
    assert params["fixed_ks"] == (1, 4)
    assert params["opp_fire"] is True
    assert params["arad"] is True
    assert params["jh_f1"] is True


def test_full_suite_yaml_enables_arad_and_jh():
    from study.config import CONFIGS, _load_yaml

    raw = _load_yaml(CONFIGS / "suites" / "full.yaml")
    params = sae_suite_params(raw["methods"]["sae"])
    assert params["opp_fire"] is True
    assert params["arad"] is True
    assert params["jh_f1"] is True
    core = _load_yaml(CONFIGS / "suites" / "core.yaml")
    core_params = sae_suite_params(core["methods"]["sae"])
    assert core_params["arad"] is False
    assert core_params["jh_f1"] is False


def test_build_causal_study_kwargs_from_config():
    cfg = StudyConfig(
        model_key="pythia-70m-deduped",
        task_id="induction",
        suite_id="core",
        raw={
            "model": {
                "sae_release": "pythia-70m-deduped-res-jb",
                "sae_layers": [5],
                "n_layers": 6,
            },
            "suite": {
                "methods": {
                    "sae": ["k1", "kstar", "all_k1"],
                    "caa": ["act_prediction"],
                }
            },
            "causal": {
                "n_per_group": 50,
                "decoder_max_size": 500,
                "primary_only": True,
            },
            "cli": {"save_modified_models": True},
        },
    )
    kw = build_causal_study_kwargs(
        cfg,
        target_classes=["MATCH"],
        enabled={"gradiend", "actiend", "causal"},
        skip_causal_methods={"gradiend:MATCH"},
    )
    assert kw["target_classes"] == ["MATCH"]
    assert kw["sae_fixed_ks"] == (1,)
    assert kw["sae_opp_fire_enabled"] is False
    assert kw["n_per_group"] == 50
    assert kw["decoder_max_size"] == 500
    assert kw["save_modified_models"] is True
    assert kw["skip_causal_methods"] == {"gradiend:MATCH"}
    assert kw["caa_causal_policies"] == ["prediction"]
    assert kw["primary_only"] is True
    assert kw["actiend_causal_policies"] == ACTIEND_CAUSAL_POLICY_DEFAULTS
    assert kw["direction_polarity_protocol_version"] == 2
    assert kw["cga_causal_grid_protocol_version"] == 2
    assert kw["agiend_causal_grid_protocol_version"] == 2


def test_suite_yaml_selects_only_needed_actiend_causal_policies():
    from study.config import CONFIGS, _load_yaml

    core = _load_yaml(CONFIGS / "suites" / "core.yaml")
    full = _load_yaml(CONFIGS / "suites" / "full.yaml")
    full_plus = _load_yaml(CONFIGS / "suites" / "full_plus.yaml")

    # core runs only the ACTIEND default policy, which is now ungated all-token.
    assert normalize_actiend_causal_policies(
        core["causal"]["actiend_policies"]
    ) == ("all",)
    assert normalize_actiend_causal_policies(
        full["causal"]["actiend_policies"]
    ) == ACTIEND_CAUSAL_POLICY_DEFAULTS
    assert normalize_actiend_causal_policies(
        full_plus["causal"]["actiend_policies"]
    ) == ACTIEND_CAUSAL_POLICY_DEFAULTS


def test_loaded_suite_passes_actiend_policy_subset_to_causal_runner():
    from study.config import load_study_config

    core = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    full = load_study_config(model="gpt2-small", task="gender_en", suite="full")

    core_kw = build_causal_study_kwargs(
        core, target_classes=["F", "M"], enabled={"actiend"}
    )
    full_kw = build_causal_study_kwargs(
        full, target_classes=["F", "M"], enabled={"actiend"}
    )
    assert core_kw["actiend_causal_policies"] == ("all",)
    assert full_kw["actiend_causal_policies"] == ACTIEND_CAUSAL_POLICY_DEFAULTS


def test_actiend_causal_policy_normalization_and_ablation_specs():
    assert normalize_actiend_causal_policies(
        ["default", "tok_all", "tok_prediction", "tok_all"]
    ) == ("gated_all", "all", "prediction")

    # 2026-09-03 swap: ungated all-token (``all``) is the default; the gated policy
    # (``gated_all``) and ``prediction`` are the ablations. Requesting ``all`` alone
    # enables the default and emits no ablations (so no id collision).
    assert actiend_default_policy_enabled(["all"]) is True
    assert actiend_default_policy_enabled(["gated_all"]) is False
    assert actiend_default_policy_enabled(["gated_all", "all"]) is True

    assert actiend_ablation_policy_specs(["all"]) == ()
    assert actiend_ablation_policy_specs(["gated_all"]) == (("all", "encoder_direction"),)
    assert actiend_ablation_policy_specs(["gated_all", "all", "prediction"]) == (
        ("all", "encoder_direction"),
        ("prediction", None),
    )


def test_unknown_actiend_causal_policy_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown ACTIEND causal policy"):
        normalize_actiend_causal_policies(["alll"])


def test_causal_stage_calls_causal_study_not_poc():
    from study.stages import causal as causal_stage

    src = inspect.getsource(causal_stage.run_causal_stage)
    assert "causal_study.run_study_causal" in src or "from causal_study import run_study_causal" in src
    assert "sae_engine" not in src


def test_causal_study_accepts_skip_and_sae_flags():
    import causal_study

    sig = inspect.signature(causal_study.run_study_causal)
    assert "skip_causal_methods" in sig.parameters
    assert "sae_opp_fire_enabled" in sig.parameters
    assert "actiend_causal_policies" in sig.parameters
    assert "reuse_strengthen_by_method" in sig.parameters
    assert "primary_only" in sig.parameters


def test_primary_caa_site_uses_the_same_validation_lock_as_headline_tables():
    from caa_eval import caa_encoder_id
    from causal_study import selected_caa_part_by_validation_detection

    def metrics(score):
        return {
            "val_readout": {
                "roc_auc_neutral": score,
                "neutral_specificity": score,
            }
        }

    payload = {
        caa_encoder_id("MATCH", policy="prediction", part=None): metrics(0.70),
        caa_encoder_id("MATCH", policy="prediction", part="all"): metrics(0.80),
        caa_encoder_id("MATCH", policy="prediction", part="L0"): metrics(0.95),
        caa_encoder_id("MATCH", policy="prediction", part="L1"): metrics(0.60),
    }
    assert selected_caa_part_by_validation_detection(
        payload, cls="MATCH", policy="prediction", pair=None, layers=[0, 1]
    ) == "L0"


def test_primary_caa_site_refuses_an_incomplete_detection_pool():
    from caa_eval import caa_encoder_id
    from causal_study import selected_caa_part_by_validation_detection

    payload = {
        caa_encoder_id("MATCH", policy="prediction", part=None): {
            "val_readout": {"roc_auc_neutral": 0.8, "neutral_specificity": 0.8}
        }
    }
    assert selected_caa_part_by_validation_detection(
        payload, cls="MATCH", policy="prediction", pair=None, layers=[0]
    ) is None


def test_core_layers_materialize_cga_caga_candidates_for_validation_lock():
    def cfg(primary_only):
        return StudyConfig(
            model_key="gpt2-small",
            task_id="induction",
            suite_id="test",
            raw={
                "model": {"n_layers": 2, "sae_layers": [0, 1]},
                "suite": {"methods": {"cga": ["plain", "layers"], "caga": ["plain", "layers"]}},
                "causal": {"primary_only": primary_only},
            },
        )

    core_kw = build_causal_study_kwargs(cfg(True), target_classes=["MATCH"], enabled={"cga", "caga"})
    full_kw = build_causal_study_kwargs(cfg(False), target_classes=["MATCH"], enabled={"cga", "caga"})
    assert core_kw["layerwise_cga"] is True
    assert core_kw["layerwise_caga"] is True
    assert full_kw["layerwise_cga"] is True
    assert full_kw["layerwise_caga"] is True


def test_primary_cga_caga_lock_uses_validation_detection_for_aggregate_or_layer():
    from causal_study import selected_direction_part_by_validation_detection

    def metrics(score):
        return {
            "val_readout": {
                "roc_auc_neutral": score,
                "neutral_specificity": score,
            }
        }

    raw = {
        "per_class_readouts": {"MATCH": metrics(0.70)},
        "per_component_readouts": {
            "L0": {"MATCH": metrics(0.95)},
            "L1": {"MATCH": metrics(0.60)},
        },
    }
    assert selected_direction_part_by_validation_detection(raw, cls="MATCH") == "L0"


def test_v5_strengthen_is_reused_for_component_repair():
    from study.stages.causal import reusable_strengthen_entries

    good = {
        "method": "gradiend:MATCH",
        "strengths": [{"strength": 1.0}],
        "selected": {"strength": 1.0},
        "meta": {},
    }
    incomplete = {"method": "sae:MATCH:k1", "strengths": [], "selected": None}
    previous = {
        "protocol_version": "decoder-val-select-test-report-v5-caa-poles",
        "actiend_ridge_id_protocol_version": 1,
        "by_method": {
            "gradiend:MATCH": good,
            "sae:MATCH:k1": incomplete,
            "actiend_ridge:MATCH": {**good, "method": "actiend_ridge:MATCH"},
        },
    }

    assert reusable_strengthen_entries(previous, skip_existing=True) == {
        "gradiend:MATCH": good,
    }
    assert reusable_strengthen_entries(previous, skip_existing=False) == {}
    assert reusable_strengthen_entries(
        {**previous, "protocol_version": "decoder-full-factual-v2"},
        skip_existing=True,
    ) == {}


def test_causal_group_already_complete():
    from causal_study import causal_group_already_complete

    skip = {"gradiend:MATCH", "actiend:MATCH"}
    assert causal_group_already_complete(["gradiend:MATCH"], skip) is True
    assert causal_group_already_complete(["gradiend:MATCH", "actiend:MATCH"], skip) is True
    assert causal_group_already_complete(["gradiend:MATCH", "gradiend:DISTRACTOR"], skip) is False
    assert causal_group_already_complete(["gradiend:MATCH"], set()) is False
    assert causal_group_already_complete([], skip) is False


def test_direction_polarity_protocol_invalidates_only_affected_families():
    from study.stages.causal import invalid_direction_polarity_method_ids

    completed = {
        "agiend:M",
        "caga:M",
        "cga:M",
        "cga_tensor_norm:F-M:F",
        "gradiend:M",
        "actiend:M",
        "sae:M:k1",
        "caa:M:prediction",
    }
    assert invalid_direction_polarity_method_ids(completed, None) == {
        "agiend:M",
        "caga:M",
        "cga:M",
        "cga_tensor_norm:F-M:F",
    }
    assert invalid_direction_polarity_method_ids(completed, 2) == set()
    by_method = {
        "caga:M": {"meta": {"direction_polarity_protocol_version": 2}},
        "agiend:M": {"meta": {"polarity_reselected_from_bidirectional_grid": True}},
    }
    assert invalid_direction_polarity_method_ids(completed, 2, by_method) == {
        "cga:M",
        "cga_tensor_norm:F-M:F",
    }


def test_cga_grid_protocol_invalidates_only_cga_families():
    from study.stages.causal import invalid_cga_causal_grid_method_ids

    completed = {
        "agiend:M",
        "caga:M",
        "cga:M",
        "cga:M:L0",
        "cga_tensor_norm:F-M:F",
        "gradiend:M",
    }
    assert invalid_cga_causal_grid_method_ids(completed, None) == {
        "cga:M",
        "cga:M:L0",
        "cga_tensor_norm:F-M:F",
    }
    assert invalid_cga_causal_grid_method_ids(completed, 2) == set()


def test_agiend_grid_protocol_invalidates_only_agiend():
    from study.stages.causal import invalid_agiend_causal_grid_method_ids

    completed = {"agiend:M", "agiend:F-M:F", "caga:M", "cga:M", "gradiend:M"}
    assert invalid_agiend_causal_grid_method_ids(completed, None) == {
        "agiend:M", "agiend:F-M:F"
    }
    assert invalid_agiend_causal_grid_method_ids(completed, 2) == set()


def test_previous_causal_entries_recovers_method_row_grid_when_raw_family_was_replaced():
    from study.stages.causal import previous_causal_entries

    complete = {
        "method": "cga:M",
        "selected_strength": 1.0,
        "selected": {"strength": 1.0},
        "strengths": [{"strength": 1.0}],
        "weaken_selected_strength": 1.0,
        "weaken_selected": {"strength": 1.0},
        "weaken_strengths": [{"strength": 1.0}],
        "meta": {},
    }
    previous = {
        "raw": {"causal": {"by_method": {}}},
        "methods": [{"method": "cga:M", "extras": {"causal": complete}}],
    }
    assert previous_causal_entries(previous) == {"cga:M": complete}


def test_previous_causal_entries_keys_proxy_extras_by_causal_method_id():
    from study.stages.causal import previous_causal_entries

    complete = {
        "method": "caga:M",
        "selected_strength": 1.0,
        "selected": {"strength": 1.0},
        "strengths": [{"strength": 1.0}],
        "weaken_selected_strength": 1.0,
        "weaken_selected": {"strength": 1.0},
        "weaken_strengths": [{"strength": 1.0}],
        "meta": {},
    }
    previous = {
        "methods": [
            {
                "method": "caga:M|proxy=her_his",
                "extras": {"causal": complete},
            }
        ]
    }

    assert previous_causal_entries(previous) == {"caga:M": complete}


def test_apply_causal_crow_clears_stale_keyerror_and_keeps_encoder_on_failure():
    from study.stages.causal import (
        apply_causal_crow_to_row,
        clear_stale_causal_error_on_encode_row,
    )

    poisoned = {
        "method": "sae:english:k1",
        "status": "error",
        "error": "'english'",
        "metrics": {"roc_auc_neutral": 0.999, "causal_error": "'english'"},
    }
    cleaned = clear_stale_causal_error_on_encode_row(poisoned)
    assert cleaned["status"] == "ok"
    assert "error" not in cleaned
    assert "causal_error" not in (cleaned.get("metrics") or {})

    row = dict(cleaned)
    apply_causal_crow_to_row(
        row,
        {
            "status": "ok",
            "error": None,
            "metrics": {"causal_signed_effect": 0.18},
            "extras": {"causal": {"method": "sae:english:k1"}},
            "artifacts": {},
        },
    )
    assert row["status"] == "ok"
    assert row.get("error") is None
    assert abs(row["metrics"]["causal_signed_effect"] - 0.18) < 1e-12
    assert abs(row["metrics"]["roc_auc_neutral"] - 0.999) < 1e-12

    failed = {
        "method": "sae:english:k1",
        "status": "ok",
        "metrics": {"roc_auc_neutral": 0.999},
    }
    apply_causal_crow_to_row(
        failed,
        {"status": "error", "error": "'english'", "metrics": {}, "extras": {}, "artifacts": {}},
    )
    assert failed["status"] == "partial"
    assert failed["error"] == "'english'"
    assert abs(failed["metrics"]["roc_auc_neutral"] - 0.999) < 1e-12


def test_causal_summary_preserves_skipped_completed_methods(tmp_path):
    from causal_eval import persist_causal_summary
    from study.stages.causal import _merged_causal_summary_entries

    previous = [
        {
            "method": "gradiend:MATCH",
            "target_class": "MATCH",
            "headline_metrics": {"causal_signed_effect": 0.7},
        }
    ]
    current = [
        {
            "method": "sae:MATCH:k1",
            "target_class": "MATCH",
            "headline_metrics": {"causal_error": "retry failed"},
        }
    ]
    serial_by = {
        "gradiend:MATCH": {"method": "gradiend:MATCH", "target_class": "MATCH"},
        "sae:MATCH:k1": {"method": "sae:MATCH:k1", "target_class": "MATCH"},
    }

    merged = _merged_causal_summary_entries(
        serial_by,
        current=current,
        previous=previous,
    )
    persist_causal_summary(tmp_path, merged)

    import json

    saved = json.loads((tmp_path / "causal" / "summary.json").read_text())
    assert [row["method"] for row in saved] == [
        "gradiend:MATCH",
        "sae:MATCH:k1",
    ]
    assert saved[0]["headline_metrics"]["causal_signed_effect"] == 0.7
    assert saved[1]["headline_metrics"]["causal_error"] == "retry failed"
    assert not (tmp_path / "causal" / "summary.json.tmp").exists()


def test_sae_only_causal_does_not_drop_backbone_trainers():
    from study.stages import causal as causal_stage

    src = inspect.getsource(causal_stage.run_causal_stage)
    assert "train_none = dict(train_raw_none or {})" in src
    assert "backbone=" in src


def test_resolve_causal_backbone_prefers_trainer_then_hf(monkeypatch):
    import causal_study

    class _Tok:
        pad_token = None
        eos_token = "<eos>"

    class _Model:
        pass

    class _Mwg:
        base_model = _Model()

    class _Trainer:
        tokenizer = _Tok()

        def get_model(self):
            return _Mwg()

    live_m, live_t = object(), object()
    got = causal_study.resolve_causal_backbone(
        live_model=live_m,
        live_tokenizer=live_t,
        train_raw_none={"gradiend": {"trainer": _Trainer()}},
        load_hf=False,
    )
    assert got == (live_m, live_t)

    model, tok = causal_study.resolve_causal_backbone(
        train_raw_none={"gradiend": {"trainer": _Trainer()}},
        load_hf=False,
    )
    assert model is _Mwg.base_model
    assert tok is _Trainer.tokenizer

    loaded = []

    def _fake_hf(key=None):
        loaded.append(key)
        return _Model(), _Tok()

    monkeypatch.setattr(causal_study, "load_hf_backbone_for_causal", _fake_hf)
    model, tok = causal_study.resolve_causal_backbone(
        train_raw_none={},
        study_model_key="gpt2-small",
        load_hf=True,
    )
    assert loaded == ["gpt2-small"]
    assert tok.pad_token is None  # stub; real load_hf sets pad from eos

    src = inspect.getsource(causal_study.run_study_causal)
    assert "no model/tokenizer; skip" not in src
    assert "SAE causal needs a HF model/tokenizer" in src


def test_load_hf_backbone_sets_pad_token(monkeypatch):
    import causal_study

    class _Tok:
        pad_token = None
        eos_token = "<eos>"

    class _Model:
        device = "cpu"
        eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *_a, **_k: _Model(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: _Tok(),
    )
    model, tok = causal_study.load_hf_backbone_for_causal("gpt2-small")
    assert isinstance(model, _Model)
    assert str(model.device) in {"cpu", "cuda"}
    assert model.eval_called is True
    assert tok.pad_token == "<eos>"


def test_direct_causal_pool_honors_decoder_max_size():
    import causal_study

    assert causal_study.causal_eval_n_per_group(
        n_per_group=1000, decoder_max_size=64
    ) == 64
    assert causal_study.causal_eval_n_per_group(
        n_per_group=64, decoder_max_size=1000
    ) == 64


def test_causal_only_sae_rows_do_not_block_encode_snapshot_recovery():
    from study.stages.sae import _sae_rows_are_substantive

    causal_only = {
        "method": "sae:RESULT:kstar",
        "status": "ok",
        "metrics": {"causal_signed_effect": 0.05, "readout_kind": "causal_only"},
    }
    encoded = {
        "method": "sae:RESULT:kstar",
        "status": "ok",
        "metrics": {"roc_auc_neutral": 0.99, "balanced_accuracy": 0.95},
    }
    assert _sae_rows_are_substantive([causal_only]) is False
    assert _sae_rows_are_substantive([encoded]) is True
