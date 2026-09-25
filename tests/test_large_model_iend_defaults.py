"""Keep the successful large-model LR settings in the resolved core config."""

import pytest

from study.config import load_study_config, training_kwargs_from_config


MODELS = ("llama-3.1-8b", "qwen3.5-9b-base")


@pytest.mark.parametrize(
    ("model", "expected_lr", "expected_eval_steps"),
    (("llama-3.1-8b", 2e-6, 50), ("qwen3.5-9b-base", 1e-5, 25)),
)
def test_large_model_gradiend_defaults(model, expected_lr, expected_eval_steps):
    cfg = load_study_config(model=model, task="gender_en", suite="core")
    kwargs = training_kwargs_from_config(cfg, backend="gradiend")

    assert kwargs["learning_rate"] == pytest.approx(expected_lr)
    assert kwargs["max_steps"] == 500
    assert kwargs["eval_steps"] == expected_eval_steps
    assert "learning_rate_decoder" not in kwargs
    assert cfg.training["torch_dtype"] == "bfloat16"
    assert cfg.training["pre_prune"]["topk"] == pytest.approx(0.01)


@pytest.mark.parametrize("model", MODELS)
def test_large_model_actiend_keeps_validated_encoder_and_decoder_lrs(model):
    cfg = load_study_config(model=model, task="gender_en", suite="core")
    kwargs = training_kwargs_from_config(cfg, backend="actiend")

    assert kwargs["learning_rate"] == pytest.approx(1e-5)
    assert kwargs["max_steps"] == 100
    assert kwargs["eval_steps"] == 20
    assert kwargs["learning_rate_decoder"] == pytest.approx(1.15e-2)


@pytest.mark.parametrize("model", MODELS)
def test_explicit_cli_overrides_still_win(model):
    cfg = load_study_config(
        model=model,
        task="gender_en",
        suite="core",
        cli_overrides={
            "training": {
                "learning_rate_gradiend": 2e-6,
                "learning_rate_decoder_actiend": 3e-5,
                "max_steps": 200,
            }
        },
    )

    assert cfg.learning_rate(backend="gradiend") == pytest.approx(2e-6)
    assert cfg.learning_rate_decoder(backend="actiend") == pytest.approx(3e-5)
    assert cfg.max_steps(backend="gradiend") == 200
    assert cfg.max_steps(backend="actiend") == 200


@pytest.mark.parametrize("model", MODELS)
def test_core_includes_all_main_method_families_and_causal(model):
    cfg = load_study_config(model=model, task="gender_en", suite="core")
    methods = cfg.raw["suite"]["methods"]

    assert methods["gradiend"] == ["none"]
    assert methods["actiend"] == ["none"]
    assert methods["sae"] == ["k1", "kstar", "all_k1"]
    assert methods["caa"] == ["act_prediction"]
    assert methods["cga"] == ["plain", "layers"]
    assert methods["caga"] == ["plain", "layers"]
    assert methods["agiend"] == ["plain"]
    assert cfg.raw["causal"]["enabled"] is True
