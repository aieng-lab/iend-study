from study.config import load_study_config
from study.training_profiles import shared_training_kwargs


def test_encoder_bias_is_enabled_across_study_defaults():
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")

    # Keep the code-level architecture default outside resolved StudyConfig so
    # completed task-level config hashes remain reusable.
    assert "bias_encoder" not in cfg.training
    assert shared_training_kwargs(cfg, backend="gradiend")["bias_encoder"] is True
    assert shared_training_kwargs(cfg, backend="actiend")["bias_encoder"] is True
