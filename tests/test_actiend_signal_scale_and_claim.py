"""Causal claim-class pinning + Signal.activation running_rms scale."""

from __future__ import annotations

import torch

from causal_eval import _trainer_claim_classes, evaluate_decoder_for_classes
from gradiend.trainer.core.signals import ActivationRunningRms, Signal


def test_trainer_claim_prefers_one_pole_positive():
    class T:
        def _one_pole_positive_class(self):
            return "europe"

        config = type("C", (), {"target_classes": ["europe", "africa", "asia"]})()

    assert _trainer_claim_classes(T()) == ["europe"]


def test_evaluate_decoder_pins_claim_not_cf_rivals():
    calls = []

    class Trainer:
        config = type("C", (), {"target_classes": ["europe"]})()

        def _one_pole_positive_class(self):
            return "europe"

        def evaluate_decoder(self, **kw):
            calls.append(kw)
            return {"europe": {"value": 0.1, "feature_factor": 1.0, "learning_rate": 1.0}}

    evaluate_decoder_for_classes(
        Trainer(),
        ["africa", "asia", "europe"],  # task vocab — must be filtered to claim
        split="validation",
        max_size=8,
        use_cache=False,
    )
    assert calls[0]["target_class"] == ["europe"]
    assert calls[0]["summary_metrics"] == ["europe"]


def test_signal_activation_scale_option():
    sig = Signal.activation(token_selector="prediction", scale="running_rms")
    assert sig.options["scale"] == "running_rms"
    assert sig.options["scale_reduce"] == "per_site"
    raw = Signal.activation(token_selector="prediction")
    assert "scale" not in raw.options


def test_running_rms_memory_and_roundtrip():
    scaler = ActivationRunningRms(reduce="per_site", momentum=0.0, eps=1e-6)
    sites = [
        torch.ones(2, 4) * 3.0,
        torch.ones(2, 4) * 4.0,
    ]
    scaled = scaler.update_and_scale(sites)
    assert abs(float(scaled[0].pow(2).mean().sqrt()) - 1.0) < 1e-4
    assert abs(float(scaled[1].pow(2).mean().sqrt()) - 1.0) < 1e-4
    # O(n_sites) state only
    state = scaler.state_dict()
    assert len(state["ms"]) == 2
    # Use one row worth of widths for unscale demo
    widths = [4, 4]
    # Build a scaled concat from first batch row mean
    scaled_flat = torch.cat([scaled[0][0], scaled[1][0]], dim=0)
    raw = scaler.unscale_flat(scaled_flat, widths)
    assert abs(float(raw[:4].mean()) - 3.0) < 1e-3
    assert abs(float(raw[4:].mean()) - 4.0) < 1e-3
