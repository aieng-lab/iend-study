"""Shared activation-site protocol (ACTIEND/CAA/SAE)."""

from __future__ import annotations

import torch

from activation_protocol import (
    ACTIVATION_PROTOCOL_VERSION,
    normalize_activation_site,
    protocol_metadata,
)
from caa_eval import resolve_act_selector


def test_activation_protocol_version_bumped():
    assert ACTIVATION_PROTOCOL_VERSION >= 4


def test_normalize_pre_prediction_aliases():
    assert normalize_activation_site("unfilled_prediction") == "pre_prediction"
    assert normalize_activation_site("pre_prediction") == "pre_prediction"
    assert normalize_activation_site("prediction") == "prediction"


def test_resolve_act_selector_pre_prediction():
    assert resolve_act_selector("pre_prediction") == "pre_prediction"
    assert resolve_act_selector("unfilled_prediction") == "pre_prediction"


def test_pre_prediction_selector_gathers_one_before_span():
    from gradiend.trainer.core.signals import ActivationSignalExtractor

    # batch=1, seq=4, d=3 — prediction span at indices 2..3 → pre = index 1
    act = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    pred_mask = torch.tensor([[False, False, True, True]])
    out = ActivationSignalExtractor._gather_pre_prediction_tokens(
        act, pred_mask, selector_name="pre_prediction"
    )
    assert out.shape == (1, 3)
    assert torch.equal(out[0], act[0, 1])


def test_protocol_metadata_lists_shared_methods():
    meta = protocol_metadata("prediction")
    assert meta["activation_site"] == "prediction"
    assert meta["actiend_mixed_site"] is False
    assert set(meta["shared_h_methods"]) == {"actiend", "caa", "sae"}
    assert "left context" in meta["one_pole_unfilled_note"]
    assert meta["neutral_unit"] == "token"
    assert meta["youden_split"] == "validation"
    assert meta["causal_select_split"] == "validation"
    assert protocol_metadata("mean")["neutral_unit"] == "text"
    assert protocol_metadata("last")["neutral_unit"] == "text"
    pre = protocol_metadata("pre_prediction")
    assert pre["actiend_source_site"] == "pre_prediction"
    assert pre["actiend_target_site"] == "prediction"
    assert pre["actiend_mixed_site"] is True
