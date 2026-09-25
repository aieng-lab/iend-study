"""Unit tests for the CAGA closed-form direction (numpy core, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from caga_eval import (
    caga_causal_id,
    caga_causal_specs,
    caga_direction_from_signals,
    caga_layer_slices,
    caga_layer_specs,
    normalize_vec,
)


def test_one_pole_direction_is_normalized_mean_transition():
    # All labels +1 (one-pole): direction = normalized mean(F - A).
    F = np.array([[2.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
    A = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    lab = np.array([1.0, 1.0, 1.0])
    d = caga_direction_from_signals(F, A, lab)
    assert d == pytest.approx([1.0, 0.0])  # mean is (2,0), normalized


def test_labels_flip_the_contributing_sign():
    F = np.array([[1.0, 0.0], [1.0, 0.0]])
    A = np.array([[0.0, 0.0], [0.0, 0.0]])
    # opposite labels cancel -> degenerate -> zeros
    d = caga_direction_from_signals(F, A, np.array([1.0, -1.0]))
    assert d == pytest.approx([0.0, 0.0])


def test_uses_transition_not_factual_alone():
    F = np.array([[3.0, 0.0], [3.0, 0.0]])
    A = np.array([[3.0, 0.0], [3.0, 0.0]])  # F - A == 0
    d = caga_direction_from_signals(F, A, np.array([1.0, 1.0]))
    assert d == pytest.approx([0.0, 0.0])  # transition is zero despite large F


def test_supervised_decoder_formula_scaling_is_normalized_away():
    # sum(label*target)/sum(label**2) then unit-norm: overall scale irrelevant.
    F = np.array([[1.0, 1.0], [1.0, 1.0]])
    A = np.zeros((2, 2))
    d = caga_direction_from_signals(F, A, np.array([1.0, 1.0]))
    assert d == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_normalize_vec_zero_stays_zero():
    assert normalize_vec(np.zeros(4)) == pytest.approx(np.zeros(4))


def test_shape_validation():
    with pytest.raises(ValueError):
        caga_direction_from_signals(np.ones((3, 2)), np.ones((3, 3)), np.ones(3))
    with pytest.raises(ValueError):
        caga_direction_from_signals(np.ones((3, 2)), np.ones((3, 2)), np.ones(2))


def test_caga_causal_id_pole_shapes():
    # One-pole and two-pole ids mirror CGA so analysis grouping buckets them.
    assert caga_causal_id("VALUE") == "caga:VALUE"
    assert caga_causal_id("F", pair="F-M") == "caga:F-M:F"
    assert caga_causal_id("F", layer=3) == "caga:F:L3"
    assert caga_causal_id("F", pair="F-M", layer=3) == "caga:F-M:F:L3"


def test_caga_checkpoint_layer_specs_and_slices_round_trip():
    direction = torch.arange(1, 6, dtype=torch.float32)
    linear = torch.nn.Linear(1, 5, bias=False)
    with torch.no_grad():
        linear.weight.copy_(direction[:, None])
    gm = SimpleNamespace(
        param_map={
            "activation:gpt_neox.layers.0": {"shape": (2,), "repr": "all"},
            "activation:gpt_neox.layers.1": {"shape": (3,), "repr": "all"},
        },
        decoder=[SimpleNamespace(linear=linear)],
    )
    model = SimpleNamespace(gradiend=gm)
    slices = caga_layer_slices(model)
    specs = caga_layer_specs(model)
    assert slices == {0: (0, 2), 1: (2, 5)}
    assert list(specs) == [0, 1]
    rebuilt = torch.cat([specs[layer][0][1] for layer in specs])
    assert torch.equal(rebuilt, direction)


def test_caga_causal_specs_legacy_broadcasts_direction_to_each_layer():
    d = np.array([1.0, 0.0, 0.0])
    specs = caga_causal_specs(d, layers=[6, 9], hf_resid_template="transformer.h.{layer}")
    assert [m for m, _ in specs] == ["transformer.h.6", "transformer.h.9"]
    for _mod, vec in specs:
        assert np.allclose(vec.numpy(), d)  # same shared direction at each site


def test_caga_causal_specs_splits_concatenated_direction_per_site():
    # dL/dh is extracted over every residual site and concatenated: a 3-layer x
    # width-2 direction must be split so each site's module gets its OWN slice.
    sites = [("transformer.h.0", 2), ("transformer.h.1", 2), ("transformer.h.2", 2)]
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    specs = caga_causal_specs(d, sites=sites)
    assert [m for m, _ in specs] == ["transformer.h.0", "transformer.h.1", "transformer.h.2"]
    assert np.allclose(specs[0][1].numpy(), [1.0, 2.0])
    assert np.allclose(specs[1][1].numpy(), [3.0, 4.0])
    assert np.allclose(specs[2][1].numpy(), [5.0, 6.0])


def test_caga_causal_specs_broadcasts_a_single_site_direction():
    # If the direction is exactly one site's width (all widths equal), broadcast it.
    sites = [("transformer.h.0", 2), ("transformer.h.1", 2)]
    d = np.array([1.0, 2.0])
    specs = caga_causal_specs(d, sites=sites)
    assert [m for m, _ in specs] == ["transformer.h.0", "transformer.h.1"]
    for _mod, vec in specs:
        assert np.allclose(vec.numpy(), d)


def test_caga_causal_specs_rejects_dim_that_matches_no_layout():
    sites = [("transformer.h.0", 2), ("transformer.h.1", 2)]
    with pytest.raises(ValueError, match="matches neither"):
        caga_causal_specs(np.ones(3), sites=sites)


def test_caga_causal_specs_requires_sites_or_layers():
    with pytest.raises(ValueError, match="requires either"):
        caga_causal_specs(np.ones(4))
