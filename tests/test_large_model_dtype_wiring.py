"""torch_dtype / device_map plumbing for large models.

The package has had ``TrainingArguments.torch_dtype`` and
``base_model_device_map`` all along; the study never set either, so every run
loaded float32 on a single device. That is fine at 124M and impossible at 8B --
float32 weights alone are ~32GB before gradients or activations.

Same load-bearing guarantee as the other opt-in keys: an unconfigured run must
produce a byte-identical kwargs dict and an unchanged artifact hash.
"""

from __future__ import annotations

import pytest

from study.config import (
    load_study_config,
    resolve_torch_dtype,
    training_kwargs_from_config,
)
from study.training_profiles import shared_training_kwargs


def _cfg(**training):
    return load_study_config(
        model="gpt2-small",
        task="gender_en",
        cli_overrides={"training": dict(training)} if training else None,
    )


class TestUnconfiguredIsUnchanged:
    def test_both_keys_absent(self):
        kwargs = training_kwargs_from_config(_cfg(), backend="actiend")
        assert "torch_dtype" not in kwargs
        assert "base_model_device_map" not in kwargs

    def test_shared_training_kwargs_also_omits_them(self):
        kwargs = shared_training_kwargs(_cfg(), backend="actiend")
        assert "torch_dtype" not in kwargs
        assert "base_model_device_map" not in kwargs


class TestResolveTorchDtype:
    def test_resolves_names(self):
        import torch

        assert resolve_torch_dtype("bfloat16") is torch.bfloat16
        assert resolve_torch_dtype("float32") is torch.float32

    def test_accepts_the_torch_prefix(self):
        import torch

        assert resolve_torch_dtype("torch.bfloat16") is torch.bfloat16

    def test_passes_through_an_actual_dtype(self):
        import torch

        assert resolve_torch_dtype(torch.float16) is torch.float16

    def test_unknown_name_raises_rather_than_defaulting(self):
        """A typo must not silently train an 8B model in float32."""
        with pytest.raises(ValueError, match="Unknown training.torch_dtype"):
            resolve_torch_dtype("bflaot16")


class TestConfiguredPathIsForwarded:
    def test_dtype_is_resolved_into_kwargs(self):
        import torch

        kwargs = training_kwargs_from_config(
            _cfg(torch_dtype="bfloat16"), backend="actiend"
        )
        assert kwargs["torch_dtype"] is torch.bfloat16

    def test_device_map_passes_through_unchanged(self):
        for value in ("auto", False, {"": 0}):
            kwargs = training_kwargs_from_config(
                _cfg(base_model_device_map=value), backend="actiend"
            )
            assert kwargs["base_model_device_map"] == value


class TestArtifactHash:
    @staticmethod
    def _hash(**extra):
        from study.stages.train import _train_artifact_hash

        cfg = _cfg()
        shared = dict(shared_training_kwargs(cfg, backend="actiend"), **extra)
        return _train_artifact_hash(
            shared=shared,
            backend="actiend",
            cfg=cfg,
            split_mode="none",
            target_classes=["F", "M"],
        )

    def test_float32_keeps_the_historical_hash(self):
        import torch

        assert self._hash() == self._hash(torch_dtype=torch.float32)

    def test_bfloat16_is_a_distinct_protocol(self):
        """A bfloat16 run must not silently reuse a float32 checkpoint."""
        import torch

        assert self._hash() != self._hash(torch_dtype=torch.bfloat16)


class TestModelConfigsCarryTheDtype:
    """Large models must load bfloat16; small models must not change.

    Adding a dtype to gpt2-small or pythia would shift their train-artifact
    hashes and the numerics behind every existing result.
    """

    LARGE = (
        "llama-3.1-8b",
        "llama-3.3-70b-instruct",
        "qwen3.5-9b-base",
        "qwen3.5-35b-a3b-base",
        "gemma-3-4b-pt",
        "gemma-3-27b-pt",
    )
    SMALL = ("gpt2-small", "pythia-70m-deduped", "gemma-3-270m")

    def test_large_models_resolve_to_bfloat16(self):
        import torch

        for model in self.LARGE:
            cfg = load_study_config(model=model, task="gender_en")
            kwargs = training_kwargs_from_config(cfg, backend="actiend")
            assert kwargs.get("torch_dtype") is torch.bfloat16, model

    def test_small_models_stay_float32_by_omission(self):
        for model in self.SMALL:
            cfg = load_study_config(model=model, task="gender_en")
            kwargs = training_kwargs_from_config(cfg, backend="actiend")
            assert "torch_dtype" not in kwargs, model
