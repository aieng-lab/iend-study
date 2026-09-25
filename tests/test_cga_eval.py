"""CGA — Contrastive Gradient Addition: the estimator itself (``cga_eval.py``).

Covers the closed-form direction, the streaming accumulator's memory contract,
the per-tensor-normalized ablation, how a fitted direction is installed into
a ``latent_dim=1`` GRADIEND *decoder* (never the encoder), and the CAA-style
cosine scoring path used for encoding metrics instead.

The final test runs the real path end to end against gpt2 (real trainer, real
paired gradients, ~4 minutes on CPU) and is opt-in via ``GRADIEND_RUN_CGA_E2E=1``
— it is what actually proves the wiring, so run it after touching any of the
fit / install / scoring / training-argument code, not only the fast tests
above it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cga_eval import (
    _SINGLE_LABEL_TOKEN_ERROR,
    _score_one_row_with_components,
    ContrastiveGradientAccumulator,
    cga_causal_id,
    cga_layer_slices,
    cosine_to_direction,
    direction_from_decoder,
    fair_cga_encoder_eval,
    fit_cga_direction,
    install_cga_direction,
    neutral_masked_pairs,
    normalize_direction,
    param_segment_bounds,
    score_labeled_rows,
    score_labeled_rows_tolerant,
    score_neutral_texts,
    tensor_normalized_direction,
)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def test_direction_is_the_signed_mean_of_paired_gradients():
    # Two rows of the same pair seen from both poles must reinforce, not cancel:
    # the -1 row's target is already the negated contrast.
    acc = ContrastiveGradientAccumulator(3, device="cpu")
    acc.update(torch.tensor([2.0, 0.0, 0.0]), 1.0)
    acc.update(torch.tensor([-4.0, 0.0, 0.0]), -1.0)
    assert torch.allclose(acc.direction(), torch.tensor([3.0, 0.0, 0.0]))
    assert acc.n_used == 2


def test_matches_the_least_squares_closed_form():
    # sum(label*target) / sum(label**2) -- i.e. what supervised_decoder's MSE
    # objective converges to, computed directly.
    torch.manual_seed(0)
    targets = [torch.randn(6) for _ in range(11)]
    labels = [1.0, -1.0] * 5 + [1.0]
    acc = ContrastiveGradientAccumulator(6, device="cpu")
    for target, label in zip(targets, labels):
        acc.update(target, label)
    num = sum(t * l for t, l in zip(targets, labels))
    den = sum(l * l for l in labels)
    assert torch.allclose(acc.direction(), num / den, atol=1e-5)


def test_neutral_rows_drop_out_of_both_sums():
    acc = ContrastiveGradientAccumulator(2, device="cpu")
    acc.update(torch.tensor([1.0, 0.0]), 1.0)
    # Identity transitions carry label 0 and must not shift the mean or the count.
    assert acc.update(torch.tensor([99.0, 99.0]), 0.0) is False
    assert torch.allclose(acc.direction(), torch.tensor([1.0, 0.0]))
    assert acc.n_used == 1
    assert acc.n_skipped_zero_label == 1


def test_non_finite_gradients_are_dropped_and_counted_not_propagated():
    acc = ContrastiveGradientAccumulator(2, device="cpu")
    acc.update(torch.tensor([1.0, 2.0]), 1.0)
    assert acc.update(torch.tensor([float("nan"), 1.0]), 1.0) is False
    assert acc.update(torch.tensor([float("inf"), 1.0]), -1.0) is False
    assert acc.n_skipped_nonfinite == 2
    assert torch.isfinite(acc.direction()).all()


def test_accumulator_holds_one_buffer_regardless_of_row_count():
    # The whole point of streaming: fitting over many rows must not retain them.
    # (85M-entry gradients on gpt2-small -- 340MB each in fp32.)
    acc = ContrastiveGradientAccumulator(1024, device="cpu")
    tensors = [torch.ones(1024) for _ in range(50)]
    for i, t in enumerate(tensors):
        acc.update(t, 1.0 if i % 2 == 0 else -1.0)
    held = [
        v
        for v in vars(acc).values()
        if isinstance(v, torch.Tensor)
    ]
    assert len(held) == 1
    assert held[0].numel() == 1024


def test_running_mean_stays_bounded_over_many_rows():
    # A plain running *sum* would grow with n; the Welford update must not.
    acc = ContrastiveGradientAccumulator(4, device="cpu")
    for _ in range(5000):
        acc.update(torch.full((4,), 3.0), 1.0)
    assert torch.allclose(acc.direction(), torch.full((4,), 3.0), atol=1e-4)


def test_empty_fit_raises_instead_of_returning_a_zero_direction():
    acc = ContrastiveGradientAccumulator(3, device="cpu")
    acc.update(torch.ones(3), 0.0)
    with pytest.raises(ValueError, match="no labelled rows"):
        acc.direction()


def test_wrong_sized_target_is_rejected():
    acc = ContrastiveGradientAccumulator(3, device="cpu")
    with pytest.raises(ValueError, match="accumulator expects"):
        acc.update(torch.ones(4), 1.0)


def test_accumulator_can_reuse_preallocated_decoder_storage_in_place():
    decoder = torch.full((3, 1), 99.0)
    acc = ContrastiveGradientAccumulator(3, buffer=decoder)
    assert acc._mean.data_ptr() == decoder.data_ptr()
    assert torch.equal(decoder, torch.zeros_like(decoder))

    acc.update(torch.tensor([1.0, 3.0, 5.0]), 1.0)
    acc.update(torch.tensor([3.0, 5.0, 7.0]), 1.0)
    direction = acc.direction()

    assert direction.data_ptr() == decoder.data_ptr()
    assert torch.allclose(decoder.reshape(-1), torch.tensor([2.0, 4.0, 6.0]))


def test_accumulator_rejects_wrong_sized_reuse_buffer():
    with pytest.raises(ValueError, match="buffer has"):
        ContrastiveGradientAccumulator(3, buffer=torch.zeros(4))


# --------------------------------------------------------------------------
# Normalization and the tensor-norm ablation
# --------------------------------------------------------------------------


def test_normalize_matches_caa_convention():
    from caa_eval import normalize_vec

    vec = torch.tensor([3.0, 4.0, 0.0])
    assert torch.allclose(
        normalize_direction(vec),
        torch.as_tensor(normalize_vec(vec.numpy())).float(),
        atol=1e-6,
    )


def test_normalize_leaves_a_zero_vector_alone():
    assert torch.equal(normalize_direction(torch.zeros(4)), torch.zeros(4))


class _FakeGradiend:
    """Minimal stand-in for a ParamMappedGradiendModel's mapping + layers."""

    def __init__(self, param_map, input_dim, latent_dim=1, bias=False):
        self.param_map = param_map
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.torch_dtype = torch.float32
        self.device_encoder = torch.device("cpu")
        self.encoder = [_FakeLinearHolder(torch.nn.Linear(input_dim, latent_dim, bias=bias))]
        self.decoder = [_FakeLinearHolder(torch.nn.Linear(latent_dim, input_dim, bias=bias))]


class _FakeLinearHolder:
    def __init__(self, linear):
        self.linear = linear


class _FakeMwg:
    def __init__(self, gradiend):
        self.gradiend = gradiend


def _param_map_all(shapes):
    return {name: {"repr": "all", "shape": shape} for name, shape in shapes}


def test_param_segments_follow_param_map_insertion_order():
    pm = _param_map_all([("a.weight", (2, 3)), ("a.bias", (2,)), ("b.weight", (4,))])
    bounds = param_segment_bounds(_FakeGradiend(pm, 12))
    assert bounds == [("a.weight", 0, 6), ("a.bias", 6, 8), ("b.weight", 8, 12)]


def test_param_segments_count_masked_and_indexed_selections():
    pm = {
        "m": {"repr": "mask", "shape": (4,), "mask": torch.tensor([True, False, True, True])},
        "i": {"repr": "indices", "shape": (10,), "indices": torch.tensor([0, 5])},
    }
    assert param_segment_bounds(_FakeGradiend(pm, 5)) == [("m", 0, 3), ("i", 3, 5)]


def test_param_segments_reject_a_map_that_disagrees_with_input_dim():
    pm = _param_map_all([("a", (3,))])
    with pytest.raises(ValueError, match="input_dim"):
        param_segment_bounds(_FakeGradiend(pm, 99))


def test_cga_layer_slices_follow_persisted_param_map_and_round_trip():
    pm = _param_map_all(
        [
            ("transformer.h.0.attn.weight", (2,)),
            ("transformer.h.0.mlp.weight", (3,)),
            ("transformer.h.1.attn.weight", (4,)),
        ]
    )
    model = _FakeMwg(_FakeGradiend(pm, 9))
    slices = cga_layer_slices(model)
    assert slices == {0: (0, 5), 1: (5, 9)}
    direction = torch.arange(9)
    rebuilt = torch.cat([direction[lo:hi] for lo, hi in slices.values()])
    assert torch.equal(rebuilt, direction)
    assert cga_causal_id("F", layer=1) == "cga:F:L1"
    assert cga_causal_id("F", pair="F-M", layer=1) == "cga:F-M:F:L1"


def test_tensor_norm_equalizes_blocks_that_differ_by_orders_of_magnitude():
    # The ablation's purpose: without it a single high-gradient tensor owns the
    # direction. Block A is 1000x block B before normalization.
    pm = _param_map_all([("big", (2,)), ("small", (2,))])
    vec = torch.tensor([1000.0, 0.0, 1.0, 0.0])
    bounds = param_segment_bounds(_FakeGradiend(pm, 4))
    out = tensor_normalized_direction(vec, bounds)
    assert out[0] == pytest.approx(out[2], abs=1e-6)
    assert float(torch.linalg.vector_norm(out)) == pytest.approx(1.0, abs=1e-6)
    # ...whereas the plain estimator keeps the imbalance.
    plain = normalize_direction(vec)
    assert float(plain[0]) > 100 * float(plain[2])


def test_tensor_norm_leaves_an_all_zero_block_at_zero():
    pm = _param_map_all([("a", (2,)), ("dead", (2,))])
    bounds = param_segment_bounds(_FakeGradiend(pm, 4))
    out = tensor_normalized_direction(torch.tensor([3.0, 4.0, 0.0, 0.0]), bounds)
    assert torch.allclose(out[2:], torch.zeros(2))


# --------------------------------------------------------------------------
# Installing the direction
# --------------------------------------------------------------------------


def test_install_writes_only_the_decoder():
    # The encoder half is never used for anything (scoring is direct cosine,
    # not through the package's encoder module) -- install must leave it alone.
    mwg = _FakeMwg(_FakeGradiend(_param_map_all([("a", (4,))]), 4))
    before_enc = mwg.gradiend.encoder[0].linear.weight.detach().clone()
    direction = normalize_direction(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    install_cga_direction(mwg, direction)
    dec = mwg.gradiend.decoder[0].linear.weight.detach().reshape(-1)
    enc = mwg.gradiend.encoder[0].linear.weight.detach()
    # The decoder is the intervention: theta + lr*ff*direction, unscaled.
    assert torch.allclose(dec, direction, atol=1e-6)
    assert torch.equal(enc, before_enc)


def test_install_records_cga_factual_diff_polarity_instead_of_trainer_source():
    mwg = _FakeMwg(_FakeGradiend(_param_map_all([("a", (3,))]), 3))
    # Reproduce the task-level source that used to leak into a CGA checkpoint.
    mwg._source = "alternative"
    mwg._target = "diff"
    install_cga_direction(mwg, torch.tensor([0.0, 1.0, 0.0]))
    assert mwg._source == "factual"
    assert mwg._target == "diff"


def test_install_zeroes_the_decoder_bias_so_ff0_is_the_unmodified_model():
    mwg = _FakeMwg(_FakeGradiend(_param_map_all([("a", (3,))]), 3, bias=True))
    with torch.no_grad():
        mwg.gradiend.decoder[0].linear.bias.fill_(5.0)
    install_cga_direction(mwg, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(mwg.gradiend.decoder[0].linear.bias.detach(), torch.zeros(3))


def test_install_rejects_a_mismatched_direction_or_latent_dim():
    mwg = _FakeMwg(_FakeGradiend(_param_map_all([("a", (4,))]), 4))
    with pytest.raises(ValueError, match="input_dim"):
        install_cga_direction(mwg, torch.ones(3))
    wide = _FakeMwg(_FakeGradiend(_param_map_all([("a", (4,))]), 4, latent_dim=2))
    with pytest.raises(ValueError, match="latent_dim=1"):
        install_cga_direction(wide, torch.ones(4))


def test_direction_from_decoder_round_trips_through_install():
    # A CGA checkpoint's direction is never persisted separately -- a reload
    # path that needs to recompute encoding metrics must recover it from the
    # installed decoder weight exactly.
    mwg = _FakeMwg(_FakeGradiend(_param_map_all([("a", (5,))]), 5))
    direction = normalize_direction(torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0]))
    install_cga_direction(mwg, direction)
    recovered = direction_from_decoder(mwg)
    assert torch.allclose(recovered, direction, atol=1e-6)


# --------------------------------------------------------------------------
# fit_cga_direction against a stubbed trainer (no model, exact expectations)
# --------------------------------------------------------------------------


class _StubDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


class _StubTrainer:
    """Records what ``fit_cga_direction`` asks the package for."""

    def __init__(self, items, n_rows=None):
        self.items = items
        self.n_rows = len(items) if n_rows is None else n_rows
        self.create_data_kwargs = None
        self.dataset_kwargs = None

    def create_training_data(self, model, **kwargs):
        self.create_data_kwargs = dict(kwargs)
        return list(range(self.n_rows))

    def create_gradient_training_dataset(self, training_data, model, **kwargs):
        self.dataset_kwargs = dict(kwargs)
        return _StubDataset(self.items)


def _stub_mwg(input_dim=3):
    return _FakeMwg(_FakeGradiend(_param_map_all([("a", (input_dim,))]), input_dim))


def _items(pairs):
    return [{"target": t, "label": l, "feature_class_id": "1" if l > 0 else "0"} for t, l in pairs]


def test_fit_returns_the_unit_normalized_signed_mean():
    items = _items([(torch.tensor([2.0, 0.0, 0.0]), 1.0), (torch.tensor([0.0, -2.0, 0.0]), -1.0)])
    trainer = _StubTrainer(items)
    fit = fit_cga_direction(trainer, _stub_mwg(), progress_every=0)
    expected = normalize_direction(torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(fit.direction, expected, atol=1e-6)
    assert float(torch.linalg.vector_norm(fit.direction)) == pytest.approx(1.0, abs=1e-6)
    assert fit.n_used == 2
    assert fit.variant == "plain"


def test_fit_pulls_one_row_at_a_time_with_the_paired_diff_target():
    # A merged multi-row batch would give one mixed gradient and a *list* of
    # labels, which has no well-defined contrastive sign.
    trainer = _StubTrainer(_items([(torch.ones(3), 1.0), (-torch.ones(3), -1.0)]))
    fit_cga_direction(trainer, _stub_mwg(), split="train", progress_every=0)
    assert trainer.create_data_kwargs["batch_size"] == 1
    assert trainer.create_data_kwargs["split"] == "train"
    assert trainer.dataset_kwargs["target"] == "diff"
    assert trainer.dataset_kwargs["source"] == "factual"
    assert trainer.dataset_kwargs["use_cached_gradients"] is False
    assert trainer.dataset_kwargs["combine_diff_in_place"] is True


def test_fit_reports_class_counts():
    trainer = _StubTrainer(
        _items([(torch.tensor([3.0, 4.0, 0.0]), 1.0), (torch.tensor([0.0, 0.0, -5.0]), -1.0)])
    )
    fit = fit_cga_direction(trainer, _stub_mwg(), progress_every=0)
    assert fit.class_counts == {"1": 1, "0": 1}
    stats = fit.stats()
    assert stats["n_used"] == 2 and stats["variant"] == "plain"


def test_fit_tensor_norm_variant_differs_from_plain_on_unbalanced_blocks():
    pm = _param_map_all([("big", (2,)), ("small", (2,))])
    mwg = _FakeMwg(_FakeGradiend(pm, 4))
    items = _items([(torch.tensor([500.0, 0.0, 1.0, 0.0]), 1.0)])
    plain = fit_cga_direction(_StubTrainer(items), mwg, variant="plain", progress_every=0)
    tn = fit_cga_direction(
        _StubTrainer(items), mwg, variant="tensor_norm", progress_every=0
    )
    assert tn.variant == "tensor_norm"
    assert not torch.allclose(plain.direction, tn.direction, atol=1e-3)
    assert tn.direction[0] == pytest.approx(float(tn.direction[2]), abs=1e-6)


def test_fit_rejects_an_unknown_variant_and_a_batched_label():
    with pytest.raises(ValueError, match="Unknown CGA variant"):
        fit_cga_direction(_StubTrainer(_items([(torch.ones(3), 1.0)])), _stub_mwg(), variant="l2")
    batched = [{"target": torch.ones(3), "label": [1.0, -1.0]}]
    with pytest.raises(ValueError, match="batch_size=1"):
        fit_cga_direction(_StubTrainer(batched), _stub_mwg(), progress_every=0)


def test_fit_fails_loudly_when_every_row_cancels():
    # Exactly cancelling rows mean the estimator has no direction to report;
    # returning a zero vector would silently produce a no-op intervention.
    items = _items([(torch.ones(3), 1.0), (torch.ones(3), -1.0)])
    with pytest.raises(ValueError, match="all zeros"):
        fit_cga_direction(_StubTrainer(items), _stub_mwg(), progress_every=0)


def test_fit_fails_loudly_on_an_empty_split():
    with pytest.raises(ValueError, match="no 'train' rows"):
        fit_cga_direction(_StubTrainer([], n_rows=0), _stub_mwg(), progress_every=0)


def test_fit_requires_paired_targets():
    trainer = _StubTrainer([{"target": None, "label": 1.0}])
    with pytest.raises(ValueError, match="target='diff'"):
        fit_cga_direction(trainer, _stub_mwg(), progress_every=0)


# --------------------------------------------------------------------------
# CAA-style cosine scoring -- the corrected design. An earlier version routed
# CGA's encoding evaluation through the package's own ``trainer.
# evaluate_encoder()`` (tanh through a real installed encoder module), which
# does not discard per-row gradient magnitude the way cosine does and so
# reintroduces the exact confound CAA's cosine similarity exists to remove.
# These functions never touch ``ModelWithGradiend``'s encoder at all.
# --------------------------------------------------------------------------


def test_cosine_to_direction_matches_manual_computation():
    g = torch.tensor([3.0, 4.0, 0.0])
    d = torch.tensor([0.0, 1.0, 0.0])
    assert cosine_to_direction(g, d) == pytest.approx(4.0 / 5.0)


def test_cosine_to_direction_handles_a_zero_vector():
    assert cosine_to_direction(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0])) == 0.0


class _StubGradientModel:
    """Deterministic ``create_inputs``/``forward`` keyed by label -- no real
    backward pass.

    F-token rows score +1 cosine to ``direction=[1,0]``, M-token rows -1, and
    anything else (neutral auto-picked tokens) is orthogonal (0) -- a clean,
    well-separated case for exercising the metrics plumbing. Deliberately
    implements the same two-call shape ``score_labeled_rows`` actually uses
    (``create_inputs`` then ``forward``), not the single-call
    ``create_gradients`` it replaced -- see cga_eval.py's
    ``score_labeled_rows`` docstring for why.
    """

    def __init__(self, tokenizer, encoder_device="cuda:0"):
        self.tokenizer = tokenizer
        self.calls: List[Tuple[str, str]] = []
        self.to_calls: List[Any] = []
        self.gradiend = _StubGradiendHolder(encoder_device)

    def create_inputs(self, text, label):
        return {"text": str(text), "label": str(label)}

    def forward(self, item, return_dict=False, **kwargs):
        text, label = item["text"], item["label"]
        self.calls.append((text, label))
        if label == "she":
            return torch.tensor([1.0, 0.0])
        if label == "he":
            return torch.tensor([-1.0, 0.0])
        return torch.tensor([0.0, 1.0])

    def to(self, device):
        self.to_calls.append(device)
        return self

    def _get_base_forward_device(self):
        return self.gradiend.device_encoder


class _StubGradiendHolder:
    def __init__(self, device_encoder):
        self.device_encoder = device_encoder


def test_score_labeled_rows_computes_cosine_per_row():
    mwg = _StubGradientModel(tokenizer=None)
    direction = torch.tensor([1.0, 0.0])
    scores = score_labeled_rows(
        mwg, direction, ["t1", "t2", "t3"], ["she", "he", "neutral_tok"]
    )
    assert scores.tolist() == pytest.approx([1.0, -1.0, 0.0])
    assert mwg.calls == [("t1", "she"), ("t2", "he"), ("t3", "neutral_tok")]


def test_real_decoder_scoring_matches_training_prefix_and_first_target_token():
    class Tokenizer:
        vocab = {}

        def __init__(self):
            self.calls = []

        def __call__(self, text, **kwargs):
            self.calls.append((text, kwargs))
            if kwargs.get("return_tensors") == "pt":
                assert text == "The answer is "
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }
            assert text == "United States"
            return {"input_ids": [77, 88]}

    class Model:
        is_decoder_only_model = True

        def __init__(self):
            self.tokenizer = Tokenizer()
            self.base_model = SimpleNamespace()

        def _get_base_forward_device(self):
            return torch.device("cpu")

        def create_inputs(self, *_args, **_kwargs):
            raise AssertionError("decoder-only CGA must use the training objective")

        def forward(self, item, return_dict=False):
            assert item["input_ids"].tolist() == [[11, 12]]
            assert item["labels"].tolist() == [[-100, 77]]
            return torch.tensor([1.0, 0.0])

    model = Model()
    scores = score_labeled_rows(
        model,
        torch.tensor([1.0, 0.0]),
        ["The answer is [MASK] in this template suffix"],
        ["United States"],
    )
    assert scores.tolist() == pytest.approx([1.0])


def test_empty_prefix_row_scores_zero_without_touching_the_model():
    """A target at position 0 (mask at the start, e.g. race/religion's
    "[MASK] Cup.") has an empty prefix -> zero-length ``input_ids``. Nothing
    precedes the target, so there is no gradient signal: score it 0.0, the same
    zero-alignment contribution the other methods get from the extractor's
    ``allow_empty_rows`` zero vector -- never let the empty tensor reach the
    model (``input_ids.view(-1, 0)`` raises an uncatchable RuntimeError)."""

    class Tokenizer:
        vocab = {}

        def __call__(self, text, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                assert text == ""  # empty prefix before the leading [MASK]
                return {
                    "input_ids": torch.zeros((1, 0), dtype=torch.long),
                    "attention_mask": torch.zeros((1, 0), dtype=torch.long),
                }
            return {"input_ids": [77]}

    class Model:
        is_decoder_only_model = True

        def __init__(self):
            self.tokenizer = Tokenizer()
            self.base_model = SimpleNamespace()

        def _get_base_forward_device(self):
            return torch.device("cpu")

        def create_inputs(self, *_args, **_kwargs):
            raise AssertionError("decoder-only CGA must use the training objective")

        def forward(self, *_args, **_kwargs):
            raise AssertionError("empty-prefix row must short-circuit before the model")

        def forward_gradient_cosine(self, *_args, **_kwargs):
            raise AssertionError("empty-prefix row must short-circuit before the model")

    scores, kept = score_labeled_rows_tolerant(
        Model(),
        torch.tensor([1.0, 0.0]),
        ["[MASK] Cup."],
        ["China"],
    )
    assert kept.tolist() == [True]
    assert scores.tolist() == pytest.approx([0.0])


def test_score_labeled_rows_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        score_labeled_rows(_StubGradientModel(None), torch.tensor([1.0]), ["a", "b"], ["x"])


@pytest.fixture(scope="module")
def _gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def test_neutral_masked_pairs_uses_the_packages_own_auto_mask_convention(_gpt2_tokenizer):
    texts = [
        "The weather is quite cold outside today.",
        "hi",  # too short for min_prefix_tokens=5 in decoder-only mode -- dropped
    ]
    pairs = neutral_masked_pairs(texts, _gpt2_tokenizer, is_decoder_only_model=True, seed=0)
    # Only the first text can yield a valid (prefix, next-token) pair.
    assert len(pairs) == 1
    masked_text, target_token = pairs[0]
    assert "[MASK]" in masked_text
    assert target_token


def test_neutral_masked_pairs_is_seed_deterministic(_gpt2_tokenizer):
    texts = ["The weather is quite cold outside today, they said."]
    a = neutral_masked_pairs(texts, _gpt2_tokenizer, is_decoder_only_model=True, seed=1)
    b = neutral_masked_pairs(texts, _gpt2_tokenizer, is_decoder_only_model=True, seed=1)
    assert a == b


def test_score_neutral_texts_scores_only_valid_pairs(_gpt2_tokenizer):
    mwg = _StubGradientModel(_gpt2_tokenizer)
    direction = torch.tensor([1.0, 0.0])
    texts = ["The weather is quite cold outside today.", "hi"]
    scores, n_used = score_neutral_texts(
        mwg,
        direction,
        texts,
        tokenizer=_gpt2_tokenizer,
        is_decoder_only_model=True,
        seed=0,
    )
    assert n_used == 1
    assert scores.shape[0] == 1


class _StubGradientModelRaisesOnLabel(_StubGradientModel):
    """Like ``_StubGradientModel``, but raises the package's real "multi-token
    label" ``ValueError`` for one specific label -- reproducing the live
    2026-08-27 cluster failure (an auto-picked neutral token, a BPE sub-word
    fragment, re-tokenizes to more than one token) without needing a real
    tokenization edge case to land on by chance.
    """

    def __init__(self, tokenizer, raise_for_label, encoder_device="cuda:0"):
        super().__init__(tokenizer, encoder_device=encoder_device)
        self._raise_for_label = raise_for_label

    def forward(self, item, return_dict=False, **kwargs):
        if item["label"] == self._raise_for_label:
            raise ValueError(_SINGLE_LABEL_TOKEN_ERROR, [705, 83], self._raise_for_label)
        return super().forward(item, return_dict=return_dict, **kwargs)


def test_score_neutral_texts_skips_a_multi_token_auto_picked_label(monkeypatch):
    # Force neutral_masked_pairs to hand back two candidate pairs, one of
    # which the (stubbed) package rejects as multi-token on re-tokenization.
    monkeypatch.setattr(
        "cga_eval.neutral_masked_pairs",
        lambda texts, tokenizer, is_decoder_only_model, **kw: [
            ("ok text [MASK]", "she"),
            ("bad text [MASK]", "'t"),
        ],
    )
    mwg = _StubGradientModelRaisesOnLabel(tokenizer=None, raise_for_label="'t")
    scores, n_used = score_neutral_texts(
        mwg,
        torch.tensor([1.0, 0.0]),
        ["irrelevant, patched above", "irrelevant, patched above"],
        tokenizer=None,
        is_decoder_only_model=True,
    )
    # The bad row is dropped, not padded or propagated as a crash.
    assert n_used == 1
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(1.0)


def test_score_labeled_rows_does_not_swallow_the_same_error():
    # score_labeled_rows is the strict, hard-guarantee primitive: any per-row
    # tokenization failure raises. Production scoring does NOT use this
    # function for exactly that reason -- see score_labeled_rows_tolerant,
    # since real cluster data (2026-08-27) showed class labels can genuinely
    # be multi-token ("handicaps", "IMAM", "United States").
    mwg = _StubGradientModelRaisesOnLabel(tokenizer=None, raise_for_label="'t")
    with pytest.raises(ValueError, match=_SINGLE_LABEL_TOKEN_ERROR):
        score_labeled_rows(mwg, torch.tensor([1.0, 0.0]), ["t1"], ["'t"])


def test_score_labeled_rows_tolerant_drops_multi_token_labels_and_keeps_the_rest():
    # Real cluster failure (2026-08-27): "handicaps"/"IMAM"/"United States"
    # are genuine class-label completions that split into >1 GPT-2 BPE token
    # regardless of create_inputs's leading-space convention -- confirmed
    # directly (tok(' handicaps') -> 2 tokens either way). This is not a
    # neutral-only edge case; production scoring must not crash the whole
    # task over one bad label.
    mwg = _StubGradientModelRaisesOnLabel(tokenizer=None, raise_for_label="handicaps")
    scores, kept = score_labeled_rows_tolerant(
        mwg, torch.tensor([1.0, 0.0]), ["t1", "t2", "t3"], ["she", "handicaps", "he"]
    )
    assert kept.tolist() == [True, False, True]
    assert scores.tolist() == pytest.approx([1.0, -1.0])


def test_score_labeled_rows_tolerant_still_raises_on_other_errors():
    class _StubRaisesTypeError(_StubGradientModel):
        def forward(self, item, return_dict=False, **kwargs):
            raise TypeError("some unrelated bug")

    with pytest.raises(TypeError, match="unrelated bug"):
        score_labeled_rows_tolerant(
            _StubRaisesTypeError(None), torch.tensor([1.0, 0.0]), ["t1"], ["she"]
        )


def test_score_labeled_rows_uses_streaming_cosine_when_available():
    class _StreamingStub(_StubGradientModel):
        def forward_gradient_cosine(self, item, direction):
            assert direction.tolist() == [1.0, 0.0]
            return 0.375

        def forward(self, item, return_dict=False, **kwargs):
            raise AssertionError("flattened-gradient fallback must not run")

    scores, kept = score_labeled_rows_tolerant(
        _StreamingStub(None), torch.tensor([1.0, 0.0]), ["t1"], ["she"]
    )
    assert kept.tolist() == [True]
    assert scores.tolist() == pytest.approx([0.375])


def test_component_cosines_stream_one_backward_without_flattening():
    class _Selector:
        def __init__(self, name):
            self.name = name
            self.num_selected = 2

        def select_from_param_grad(self, grad):
            return grad.reshape(-1)

    class _TinyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p0 = torch.nn.Parameter(torch.zeros(2))
            self.p1 = torch.nn.Parameter(torch.zeros(2))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            x = x.reshape(-1)
            return SimpleNamespace(
                loss=(self.p0 * x[:2]).sum() + (self.p1 * x[2:]).sum()
            )

    class _TinyWrapper:
        def __init__(self):
            self.base = _TinyBase()
            self.gradiend = SimpleNamespace(
                _get_compiled_param_selectors=lambda: [_Selector("p0"), _Selector("p1")]
            )

        def create_inputs(self, text, label):
            return {"x": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        @contextmanager
        def exclusive_base_gradient_access(self):
            yield

        def _place_inputs_for_base_forward(self, inputs):
            return inputs

        def _get_base_forward_model(self):
            return self.base

        def _zero_base_grad(self, set_to_none=True):
            self.base.zero_grad(set_to_none=set_to_none)

        def _backward_through_base_model(self, loss):
            loss.backward()

        def forward(self, *args, **kwargs):
            raise AssertionError("the flattened-gradient fallback must not run")

    wrapper = _TinyWrapper()
    aggregate, layers = _score_one_row_with_components(
        wrapper,
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
        {"L0": (0, 2), "L1": (2, 4)},
        "text",
        "label",
        device=torch.device("cpu"),
    )
    assert wrapper.base.calls == 1
    assert aggregate == pytest.approx(5.0 / np.sqrt(60.0))
    assert layers["L0"] == pytest.approx(1.0 / np.sqrt(5.0))
    assert layers["L1"] == pytest.approx(0.8)


def test_score_neutral_texts_empty_when_nothing_valid(_gpt2_tokenizer):
    mwg = _StubGradientModel(_gpt2_tokenizer)
    scores, n_used = score_neutral_texts(
        mwg,
        torch.tensor([1.0, 0.0]),
        ["hi", "no"],
        tokenizer=_gpt2_tokenizer,
        is_decoder_only_model=True,
    )
    assert n_used == 0
    assert scores.shape == (0,)


def test_fair_cga_encoder_eval_resyncs_base_model_device_before_scoring(_gpt2_tokenizer):
    # First real GPU run (2026-08-26) crashed 3/3 CGA trainers identically, at
    # the first row scored here, right after a fit that ran many successful
    # forward/backward passes -- see cga_eval.py's _resync_base_model_device
    # docstring. This pins that the defensive re-sync actually fires before
    # any row is scored, and specifically targets the gradiend model's own
    # device_encoder (not some other device).
    mwg = _StubGradientModel(_gpt2_tokenizer, encoder_device="cuda:0")
    fair_cga_encoder_eval(
        mwg,
        torch.tensor([1.0, 0.0]),
        _labeled_eval_df(),
        _neutral_df(),
        target_classes=["F", "M"],
        n_bootstrap=0,
    )
    assert mwg.to_calls == ["cuda:0"]


def _labeled_eval_df():
    import pandas as pd

    rows = []
    names_f = ["Mary", "Anna", "Emma", "Julia", "Sarah", "Laura"]
    names_m = ["John", "Peter", "Thomas", "Daniel", "Michael", "Robert"]
    splits = ["train"] * 4 + ["validation"] + ["test"]
    for f, m, split in zip(names_f, names_m, splits):
        rows.append({"masked": f"{f} said [MASK] was happy.", "label": "she", "label_class": "F", "split": split})
        rows.append({"masked": f"{m} said [MASK] was happy.", "label": "he", "label_class": "M", "split": split})
    return pd.DataFrame(rows)


def _neutral_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "text": [
                "The weather is quite cold outside today.",
                "A train arrived at the station this morning.",
                "The old bridge collapsed during the storm last year.",
            ],
            "split": ["train", "validation", "test"],
        }
    )


def test_fair_cga_encoder_eval_separates_classes_from_neutral(_gpt2_tokenizer):
    mwg = _StubGradientModel(_gpt2_tokenizer)
    direction = torch.tensor([1.0, 0.0])
    fair = fair_cga_encoder_eval(
        mwg,
        direction,
        _labeled_eval_df(),
        _neutral_df(),
        target_classes=["F", "M"],
        class_encoding_direction={"F": 1, "M": -1},
        n_bootstrap=0,
    )
    assert not fair.get("error"), fair.get("error")
    per_class = fair["per_class_readouts"]
    assert set(per_class) == {"F", "M"}
    # F scores +1 cosine, neutrals score 0 -- perfectly separable either sign.
    assert per_class["F"]["roc_auc_neutral"] == pytest.approx(1.0, abs=1e-6)
    assert per_class["M"]["roc_auc_neutral"] == pytest.approx(1.0, abs=1e-6)
    assert per_class["F"]["val_encoding_E"] == pytest.approx(1.0, abs=1e-6)
    assert per_class["M"]["val_encoding_E"] == pytest.approx(1.0, abs=1e-6)


def test_layer_readouts_share_each_rows_single_gradient_extraction(_gpt2_tokenizer):
    mwg = _StubGradientModel(_gpt2_tokenizer)
    fair = fair_cga_encoder_eval(
        mwg,
        torch.tensor([1.0, 0.0]),
        _labeled_eval_df(),
        _neutral_df(),
        target_classes=["F", "M"],
        class_encoding_direction={"F": 1, "M": -1},
        n_bootstrap=0,
        component_slices={"L0": (0, 1), "L1": (1, 2)},
    )
    assert fair["component_keys"] == ["L0", "L1"]
    assert set(fair["per_component_readouts"]) == {"L0", "L1"}
    # validation/test each score two labeled rows plus one neutral row.  The
    # layer readouts do not multiply that extraction count.
    assert len(mwg.calls) == 6
    assert fair["encoder_metrics"]["mean_by_class"]["F"] == pytest.approx(1.0)
    assert fair["encoder_metrics"]["mean_by_class"]["M"] == pytest.approx(-1.0)
    # Correlation is against the UNFLIPPED score vs. the encoding-direction
    # sign, so F (+1, score +1) and M (-1, score -1) must correlate positively.
    assert fair["encoder_metrics"]["correlation"] == pytest.approx(1.0, abs=1e-6)


def _prior_eval_kwargs():
    return dict(
        target_classes=["F", "M"],
        class_encoding_direction={"F": 1, "M": -1},
        n_bootstrap=0,
        component_slices={"L0": (0, 1), "L1": (1, 2)},
    )


def test_prior_readouts_skip_the_validation_pass_and_reproduce_test_metrics(_gpt2_tokenizer):
    """Test-only migration: reusing stored validation readouts must give the same
    test metrics as a full recompute while scoring only the test split."""
    direction = torch.tensor([1.0, 0.0])
    full_mwg = _StubGradientModel(_gpt2_tokenizer)
    full = fair_cga_encoder_eval(
        full_mwg, direction, _labeled_eval_df(), _neutral_df(), **_prior_eval_kwargs()
    )
    assert len(full_mwg.calls) == 6  # val (2 labeled + 1 neutral) + test (same)

    prior_mwg = _StubGradientModel(_gpt2_tokenizer)
    migrated = fair_cga_encoder_eval(
        prior_mwg,
        direction,
        _labeled_eval_df(),
        _neutral_df(),
        prior_readouts=full,
        **_prior_eval_kwargs(),
    )
    assert len(prior_mwg.calls) == 3  # test only: validation never scored

    metrics = (
        "roc_auc_neutral",
        "roc_auc_other",
        "neutral_specificity",
        "class_exclusivity",
        "neutral_youden_threshold",
    )
    for cls in ("F", "M"):
        assert migrated["per_class_readouts"][cls]["val_readout_source"] == "prior"
        assert full["per_class_readouts"][cls]["val_readout_source"] == "scored"
        assert (
            migrated["per_class_readouts"][cls]["val_readout"]
            == full["per_class_readouts"][cls]["val_readout"]
        )
        assert migrated["per_class_readouts"][cls]["val_encoding_E"] == pytest.approx(
            full["per_class_readouts"][cls]["val_encoding_E"]
        )
        for part in ("L0", "L1"):
            got = migrated["per_component_readouts"][part][cls]
            want = full["per_component_readouts"][part][cls]
            assert got["neutral_youden_threshold_source"] == "provided"
            for key in metrics:
                assert got[key] == pytest.approx(want[key]), (part, cls, key)
        for key in metrics:
            assert migrated["per_class_readouts"][cls][key] == pytest.approx(
                full["per_class_readouts"][cls][key]
            ), (cls, key)


def test_prior_readouts_missing_one_component_falls_back_to_full_validation(_gpt2_tokenizer):
    direction = torch.tensor([1.0, 0.0])
    full = fair_cga_encoder_eval(
        _StubGradientModel(_gpt2_tokenizer),
        direction,
        _labeled_eval_df(),
        _neutral_df(),
        **_prior_eval_kwargs(),
    )
    # One (component, class) without a usable validation readout: reusing the rest
    # would leave that cell without a frozen rule, so the whole val pass must run.
    del full["per_component_readouts"]["L1"]["M"]["val_readout"]
    mwg = _StubGradientModel(_gpt2_tokenizer)
    redone = fair_cga_encoder_eval(
        mwg,
        direction,
        _labeled_eval_df(),
        _neutral_df(),
        prior_readouts=full,
        **_prior_eval_kwargs(),
    )
    assert len(mwg.calls) == 6
    assert redone["per_class_readouts"]["F"]["val_readout_source"] == "scored"


def test_missing_validation_for_a_class_raises_instead_of_reporting_an_oracle(_gpt2_tokenizer):
    import pandas as pd

    from sae_eval import FrozenRulesUnavailable

    df = _labeled_eval_df()
    # Validation has no M rows: M's Spec_n/Excl could only be fit on test.
    df = df[~((df["split"] == "validation") & (df["label_class"] == "M"))]
    with pytest.raises(FrozenRulesUnavailable, match="M"):
        fair_cga_encoder_eval(
            _StubGradientModel(_gpt2_tokenizer),
            torch.tensor([1.0, 0.0]),
            pd.DataFrame(df),
            _neutral_df(),
            **_prior_eval_kwargs(),
        )


def test_no_validation_split_at_all_raises(_gpt2_tokenizer):
    from sae_eval import FrozenRulesUnavailable

    df = _labeled_eval_df()
    df = df[df["split"] != "validation"]
    with pytest.raises(FrozenRulesUnavailable):
        fair_cga_encoder_eval(
            _StubGradientModel(_gpt2_tokenizer),
            torch.tensor([1.0, 0.0]),
            df,
            _neutral_df(),
            **_prior_eval_kwargs(),
        )


def test_fair_cga_encoder_eval_survives_a_multi_token_class_label(_gpt2_tokenizer):
    # Real cluster failure (2026-08-27): a task's own class-label completion
    # ("handicaps", "IMAM", "United States") can be multi-token, which used to
    # crash the whole task via score_labeled_rows's strict path. This mixes
    # one such row into an otherwise-good frame and confirms the row is
    # dropped (not the whole eval), with label_class staying aligned to the
    # scores that actually got computed.
    mwg = _StubGradientModelRaisesOnLabel(_gpt2_tokenizer, raise_for_label="handicaps")
    df = _labeled_eval_df()
    import pandas as pd

    bad_row = pd.DataFrame(
        [{"masked": "Support for people with [MASK] improved.", "label": "handicaps", "label_class": "F", "split": "test"}]
    )
    df = pd.concat([df, bad_row], ignore_index=True)
    fair = fair_cga_encoder_eval(
        mwg,
        torch.tensor([1.0, 0.0]),
        df,
        _neutral_df(),
        target_classes=["F", "M"],
        class_encoding_direction={"F": 1, "M": -1},
        n_bootstrap=0,
    )
    assert not fair.get("error"), fair.get("error")
    # The bad row must not have poisoned F's readout (still perfectly
    # separable on the rows that survived).
    assert fair["per_class_readouts"]["F"]["roc_auc_neutral"] == pytest.approx(1.0, abs=1e-6)


def test_fair_cga_encoder_eval_drops_absent_classes_instead_of_erroring(_gpt2_tokenizer):
    mwg = _StubGradientModel(_gpt2_tokenizer)
    fair = fair_cga_encoder_eval(
        mwg,
        torch.tensor([1.0, 0.0]),
        _labeled_eval_df(),
        _neutral_df(),
        target_classes=["F", "M", "nonexistent"],
        n_bootstrap=0,
    )
    assert set(fair["per_class_readouts"]) == {"F", "M"}


def test_fair_cga_encoder_eval_reports_error_on_empty_input():
    import pandas as pd

    mwg = _StubGradientModel(None)
    fair = fair_cga_encoder_eval(
        mwg, torch.tensor([1.0, 0.0]), pd.DataFrame(), _neutral_df(), target_classes=["F", "M"]
    )
    assert fair.get("error")
    assert fair["per_class_readouts"] == {}


# --------------------------------------------------------------------------
# Real gpt2, real gradients (opt-in; ~4 minutes on CPU)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("GRADIEND_RUN_CGA_E2E", "").strip().lower() not in {"1", "true", "yes"},
    reason="set GRADIEND_RUN_CGA_E2E=1 (real gpt2 gradients, ~4 min on CPU)",
)
def test_end_to_end_against_a_real_gpt2_trainer(tmp_path):
    import pandas as pd

    from study.config import StudyConfig
    from study.stages.train import _build_trainer
    from study.tasks import TaskBundle
    from study.training_profiles import shared_training_kwargs

    rows = []
    pairs = list(zip(["Mary", "Anna", "Emma", "Julia"], ["John", "Peter", "Thomas", "Daniel"]))
    for i, (fem, masc) in enumerate(pairs):
        split = "train" if i < 3 else "test"
        for name, label, alt, lc, ac in (
            (fem, "she", "he", "F", "M"),
            (masc, "he", "she", "M", "F"),
        ):
            rows.append(
                {
                    "masked": f"{name} went to the store because [MASK] needed milk.",
                    "label": label,
                    "label_class": lc,
                    "alternative": alt,
                    "alternative_class": ac,
                    "split": split,
                    "pair_id": f"{name}{i}",
                }
            )
    merged = pd.DataFrame(rows)
    neutrals = pd.DataFrame(
        {
            "text": ["The weather is cold today.", "A train arrived at the station."],
            "split": ["train", "test"],
        }
    )
    bundle = TaskBundle(
        task_id="cga_e2e",
        classes=["F", "M"],
        data_per_class={
            "F": merged[merged.label_class == "F"],
            "M": merged[merged.label_class == "M"],
        },
        merged_df=merged,
        neutrals=neutrals,
    )
    cfg = StudyConfig(
        model_key="gpt2-small",
        task_id="cga_e2e",
        suite_id="full",
        raw={
            "model": {"hf_model": "gpt2", "arch": "gpt2", "n_layers": 12},
            "suite": {"id": "full", "methods": {"cga": ["plain"]}},
            "training": {
                "source": "alternative",
                "target": "diff",
                "max_steps": 2,
                "train_batch_size": 1,
                "max_length": 32,
                "seed": 0,
                "gradiend_exclude_embeddings": True,
                "encoder_eval_max_size": 4,
            },
            "ablations": {"pair": True, "one_pole": False},
        },
    )
    built = _build_trainer(
        backend="cga",
        cfg=cfg,
        bundle=bundle,
        target_classes=["F", "M"],
        experiment_dir=tmp_path,
        split_mode="none",
        shared=shared_training_kwargs(cfg, backend="cga"),
        smoke=False,
        all_classes=["F", "M"],
    )
    trainer = built.trainer
    mwg = trainer.get_model()
    gradiend = mwg.gradiend

    # Same scope as GRADIEND: residual blocks only, no embeddings and no ln_f.
    names = list(gradiend.param_map.keys())
    assert names, "no mapped parameters"
    assert all(".h." in n for n in names), sorted({n for n in names if ".h." not in n})
    assert gradiend.latent_dim == 1

    fit = fit_cga_direction(
        trainer, mwg, split="train", accumulate_device="cpu", progress_every=0
    )
    assert fit.n_used > 0
    # Neutral identity rows exist in the frame and must have been skipped.
    assert fit.n_skipped_zero_label > 0
    assert fit.n_skipped_nonfinite == 0
    assert fit.input_dim == gradiend.input_dim
    assert float(torch.linalg.vector_norm(fit.direction)) == pytest.approx(1.0, abs=1e-3)

    direction = fit.direction.clone()
    install_cga_direction(mwg, direction)

    # The intervention is exactly theta + lr*ff*delta, on every mapped tensor.
    # Compared in the same float32 arithmetic the rewrite performs: a direction
    # entry is ~1e-4 against weights of order 1, so ``(theta + d) - theta`` is
    # quantized by float32 itself. Comparing against the bare ``d`` would be
    # testing rounding, not the update rule.
    base_named = dict(mwg.base_model.named_parameters())
    modified = dict(mwg.rewrite_base_model(1.0, 1.0).named_parameters())
    for name, start, end in param_segment_bounds(gradiend):
        theta = base_named[name].detach()
        delta = direction[start:end].reshape(theta.shape).to(theta.dtype)
        expected = (theta + delta) - theta
        observed = modified[name].detach() - theta
        assert torch.equal(observed, expected), f"rewrite deviates on {name}"

    # feature_factor=0 must leave the model untouched (zero decoder bias).
    unchanged = dict(mwg.rewrite_base_model(1.0, 0.0).named_parameters())
    assert torch.equal(unchanged[names[0]].detach(), base_named[names[0]].detach())

    # The readout is direct cosine, never the package's encoder module (which
    # is never touched/installed at all -- see cga_eval.py's module docstring
    # for why routing through trainer.evaluate_encoder()'s tanh was the actual
    # bug this design replaces).
    from study.tasks import labeled_df_for_eval

    eval_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    fair = fair_cga_encoder_eval(
        mwg,
        direction,
        eval_df,
        bundle.neutrals,
        target_classes=["F", "M"],
        class_encoding_direction=getattr(mwg, "feature_class_encoding_direction", None),
        max_size=4,
        max_neutral=4,
        n_bootstrap=0,
    )
    assert not fair.get("error"), fair.get("error")
    per_class = fair["per_class_readouts"]
    assert set(per_class) == {"F", "M"}
    for cls, rd in per_class.items():
        assert rd.get("roc_auc_neutral") is not None, f"{cls}: no roc_auc_neutral"
    # Cosine scores are bounded in [-1, 1] by construction -- no saturation
    # concept applies at all, unlike the tanh-through-encoder design this
    # replaces.
    assert fair["encoder_metrics"].get("mean_by_class")


@pytest.mark.skipif(
    os.environ.get("GRADIEND_RUN_CGA_E2E", "").strip().lower() not in {"1", "true", "yes"},
    reason="set GRADIEND_RUN_CGA_E2E=1 (real gpt2 gradients, ~1 min on CPU)",
)
def test_fit_cga_once_respects_the_configured_row_cap(tmp_path):
    # Real bug, hit live on gender_en's 74,060-row train split (2026-08-26,
    # first real cluster run): _fit_cga_once used to read training.
    # train_max_size -- a key nothing ever sets -- so the fit silently scanned
    # every row in the split. This builds a bundle with far more rows than the
    # configured cap and confirms the fit actually stops well short of all of
    # them, through the real orchestration function (not fit_cga_direction
    # called directly, which every other test here uses and which would not
    # have caught this -- the bug was in what _fit_cga_once passed in).
    import pandas as pd

    from study.config import StudyConfig
    from study.stages.train import _fit_cga_once
    from study.tasks import TaskBundle

    names_f = [f"Person{i}F" for i in range(30)]
    names_m = [f"Person{i}M" for i in range(30)]
    rows = []
    for i, (f, m) in enumerate(zip(names_f, names_m)):
        split = "train" if i < 25 else "test"
        rows.append(
            {
                "masked": f"{f} said [MASK] would arrive soon.",
                "label": "she",
                "label_class": "F",
                "alternative": "he",
                "alternative_class": "M",
                "split": split,
                "pair_id": f"p{i}",
            }
        )
        rows.append(
            {
                "masked": f"{m} said [MASK] would arrive soon.",
                "label": "he",
                "label_class": "M",
                "alternative": "she",
                "alternative_class": "F",
                "split": split,
                "pair_id": f"q{i}",
            }
        )
    merged = pd.DataFrame(rows)
    neutrals = pd.DataFrame(
        {
            "text": ["The weather is cold today.", "A train arrived at the station."],
            "split": ["train", "train"],
        }
    )
    bundle = TaskBundle(
        task_id="cga_cap",
        classes=["F", "M"],
        data_per_class={
            "F": merged[merged.label_class == "F"],
            "M": merged[merged.label_class == "M"],
        },
        merged_df=merged,
        neutrals=neutrals,
    )
    cap = 5
    cfg = StudyConfig(
        model_key="gpt2-small",
        task_id="cga_cap",
        suite_id="full",
        raw={
            "model": {"hf_model": "gpt2", "arch": "gpt2", "n_layers": 12},
            "suite": {"id": "full", "methods": {"cga": ["plain"]}},
            "training": {
                "source": "alternative",
                "target": "diff",
                "max_steps": 2,
                "train_batch_size": 1,
                "max_length": 32,
                "seed": 0,
                "gradiend_exclude_embeddings": True,
                "encoder_eval_max_size": 4,
                "cga_max_size": cap,
            },
            "ablations": {"pair": True, "one_pole": False},
        },
    )
    from study.training_profiles import shared_training_kwargs

    raw = _fit_cga_once(
        backend="cga",
        cfg=cfg,
        bundle=bundle,
        target_classes=["F", "M"],
        experiment_dir=tmp_path,
        split_mode="none",
        shared=shared_training_kwargs(cfg, backend="cga"),
        smoke=False,
        all_classes=["F", "M"],
    )
    n_rows_seen = raw["cga_fit"]["n_rows_seen"]
    # 25 pairs x 2 classes = 50 raw rows available; the cap is per balance
    # group (>=2 groups here), so an uncapped fit would see all 50+.
    assert n_rows_seen <= cap * 3, (
        f"fit scanned {n_rows_seen} rows against a configured cap of {cap} "
        "(per balance group) -- cga_max_size was not applied"
    )
