"""Classical vs filled SAE selection sites."""

from __future__ import annotations

from results_schema import encoder_inherit_source, fair_comparison_methods
from study.method_ids import normalize_sae_method_id
from sae_eval import (
    iter_sae_site_payloads,
    normalize_sae_select_sites,
    sae_backend_for_site,
    sae_causal_id,
    sae_encoder_all_k_id,
    sae_encoder_kstar_id,
    sae_joint_method_id,
    sae_part_for_site,
    strip_sae_raw_for_json,
)
from study.stages.sae import _sae_method_allowed


def test_sae_pre_backend_ids():
    assert sae_backend_for_site("prediction") == "sae"
    assert sae_backend_for_site("pre_prediction") == "sae_pre"
    assert sae_part_for_site("k1", "prediction") == "k1"
    assert sae_part_for_site("k1", "pre_prediction") == "k1"
    assert sae_encoder_kstar_id("M") == "sae:M:kstar"
    assert sae_encoder_kstar_id("M", site="pre_prediction") == "sae_pre:M:kstar"
    assert sae_encoder_all_k_id("F", 1, site="pre_prediction") == "sae_pre:F:all_k1"
    assert sae_causal_id("M", part="k1", site="pre_prediction") == "sae_pre:M:k1"
    assert (
        sae_causal_id("M", part="k1", token_selector="prediction", site="pre_prediction")
        == "sae_pre:M:k1_tok_prediction"
    )
    assert sae_joint_method_id("M", site="pre_prediction") == "sae_pre:joint:M"
    assert (
        sae_causal_id("M", part="sel_opp_fire", site="pre_prediction")
        == "sae_pre:M:sel_opp_fire"
    )


def test_normalize_legacy_pre_suffix():
    assert normalize_sae_method_id("sae:M:k1_pre") == "sae_pre:M:k1"
    assert normalize_sae_method_id("sae:M:kstar_pre") == "sae_pre:M:kstar"
    assert normalize_sae_method_id("sae:M:L11_k1_pre") == "sae_pre:M:L11_k1"
    assert normalize_sae_method_id("sae:joint_pre:M") == "sae_pre:joint:M"
    assert normalize_sae_method_id("sae:M:k1_pre_tok_prediction") == "sae_pre:M:k1_tok_prediction"
    assert normalize_sae_method_id("sae:M:k1") == "sae:M:k1"


def test_normalize_sae_select_sites_default_both():
    assert normalize_sae_select_sites() == ("prediction", "pre_prediction")
    assert normalize_sae_select_sites(["pre_prediction"]) == ("pre_prediction",)


def test_suite_filter_keeps_classical_when_k1_enabled():
    tags = {"k1", "kstar", "all_k1"}
    assert _sae_method_allowed("sae:M:k1", tags)
    assert _sae_method_allowed("sae_pre:M:k1", tags)
    assert _sae_method_allowed("sae_pre:M:kstar", tags)
    assert _sae_method_allowed("sae_pre:F:all_k1", tags)
    assert _sae_method_allowed("sae:M:k1_pre", tags)  # legacy id
    assert not _sae_method_allowed("sae_pre:M:sel_opp_fire", tags)


def test_suite_filter_keeps_opp_fire_pre_when_enabled():
    tags = {"opp_fire"}
    assert _sae_method_allowed("sae:M:sel_opp_fire", tags)
    assert _sae_method_allowed("sae_pre:M:sel_opp_fire", tags)


def test_fair_comparison_includes_sae_pre():
    methods = fair_comparison_methods(["M", "F"])
    assert "sae:M:k1" in methods
    assert "sae_pre:M:k1" in methods
    assert "sae_pre:F:kstar" in methods
    assert "sae_pre:F:all_k1" in methods
    assert "sae_pre:joint" in methods


def test_inherit_pre_tok_prediction():
    assert encoder_inherit_source("sae_pre:M:k1_tok_prediction") == "sae_pre:M:k1"
    assert encoder_inherit_source("sae:M:k1_pre_tok_prediction") == "sae_pre:M:k1"
    assert encoder_inherit_source("sae_pre:M:L3_k1") == "sae_pre:M:L3"


def test_iter_and_strip_by_site():
    raw = {
        "act_policy": "prediction",
        "_model": object(),
        "by_site": {
            "prediction": {"act_policy": "prediction", "_model": object(), "selections": {}},
            "pre_prediction": {
                "act_policy": "pre_prediction",
                "_sae_by_layer": {0: object()},
                "selections": {"per_class": {"features_by_class": {"M": [1]}}},
            },
        },
    }
    sites = [s for s, _ in iter_sae_site_payloads(raw)]
    assert sites == ["prediction", "pre_prediction"]
    dumped = strip_sae_raw_for_json(raw)
    assert "_model" not in dumped
    assert "_model" not in dumped["by_site"]["prediction"]
    assert "_sae_by_layer" not in dumped["by_site"]["pre_prediction"]
    assert dumped["by_site"]["pre_prediction"]["selections"]["per_class"]["features_by_class"]["M"] == [1]
