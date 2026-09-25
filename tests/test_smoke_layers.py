from __future__ import annotations

from study.config import StudyConfig
from study.stages.caa import resolve_caa_layers
from study.stages.causal import build_causal_study_kwargs
from study.training_profiles import SMOKE_LAYER_CAP, apply_smoke_model_layers, resolve_study_layers


def _gpt2_cfg(*, smoke: bool = False) -> StudyConfig:
    raw = {
        "model": {"n_layers": 12, "hf_resid_template": "transformer.h.{layer}"},
        "suite": {"methods": {"sae": ["k1"], "caa": ["act_prediction", "act_mean"]}},
        "causal": {"n_per_group": 50, "decoder_max_size": 500},
    }
    if smoke:
        raw["_smoke"] = True
    return StudyConfig(
        model_key="gpt2-small",
        task_id="induction",
        suite_id="full",
        raw=raw,
    )


def test_smoke_layer_cap_is_two():
    assert SMOKE_LAYER_CAP == 2


def test_resolve_study_layers_smoke_uses_first_two_of_n_layers():
    assert resolve_study_layers({"n_layers": 12}, smoke=True) == [0, 1]
    assert resolve_study_layers({"n_layers": 12}, smoke=False) == list(range(12))
    assert resolve_study_layers({"sae_layers": [5, 6, 7, 8]}, smoke=True) == [5, 6]


def test_apply_smoke_model_layers_writes_into_raw_not_copy():
    cfg = _gpt2_cfg(smoke=True)
    capped = apply_smoke_model_layers(cfg)
    assert capped == [0, 1]
    assert cfg.raw["model"]["sae_layers"] == [0, 1]
    assert cfg.model["sae_layers"] == [0, 1]


def test_caa_smoke_does_not_use_all_12_gpt2_layers():
    cfg = _gpt2_cfg(smoke=True)
    assert resolve_caa_layers(cfg) == [0, 1]
    assert resolve_caa_layers(cfg, layers=list(range(12))) == [0, 1]


def test_caa_non_smoke_keeps_explicit_layers():
    cfg = _gpt2_cfg(smoke=False)
    cfg.raw["model"]["sae_layers"] = [3, 4, 5]
    assert resolve_caa_layers(cfg) == [3, 4, 5]


def test_causal_smoke_kwargs_cap_sae_layers():
    cfg = _gpt2_cfg(smoke=True)
    kw = build_causal_study_kwargs(
        cfg,
        target_classes=["MATCH"],
        enabled={"causal"},
    )
    assert kw["sae_layers"] == (0, 1)


def test_causal_non_smoke_without_sae_layers_stays_empty():
    cfg = _gpt2_cfg(smoke=False)
    kw = build_causal_study_kwargs(
        cfg,
        target_classes=["MATCH"],
        enabled={"causal"},
    )
    assert kw["sae_layers"] == ()
    assert kw["n_model_layers"] == 12
