"""SAE/CAA causal sweeps recompute an *unsteered* base_probs/base_lms baseline
on every call even though, for a fixed (val_rows, meta_rows) pair, that
baseline only depends on the unsteered model + rows — not on which
feature/layer/class/policy is being swept. ``run_sae_causal_sweep``,
``run_sae_multi_layer_causal_sweep`` and ``run_caa_causal_sweep`` now accept
optional ``base_probs``/``base_lms``/``r_base_probs``/``r_base_lms`` so a
caller sweeping many methods over the same rows can compute the baseline once
and skip the redundant unsteered forward pass on every subsequent call.

These tests stub out the actual model/steering machinery (no real LM
required) and assert on call counts of ``score_class_probs_by_dataset`` /
``compute_lms_safe`` to pin the caching contract, not the causal math itself
(covered elsewhere, e.g. test_decoder_headline.py).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import causal_eval
import caa_eval


def _meta_rows():
    return [
        {"sample_id": "cls_a:0", "group": "cls_a", "text": "t1"},
        {"sample_id": "cls_a:1", "group": "cls_a", "text": "t2"},
        {"sample_id": "neutral:0", "group": "neutral", "text": "n1"},
    ]


class _FakeSAE:
    W_dec = torch.eye(4)[:, :4]


def _patch_common(monkeypatch, module):
    calls = {"probs": 0, "lms": 0}

    def _fake_probs(model, tokenizer, frame, **kw):
        calls["probs"] += 1
        return {"cls_a": {"cls_a": 0.5}}

    def _fake_lms(model, tokenizer, texts):
        calls["lms"] += 1
        return 0.1

    def _fake_strength(model, tokenizer, rows, *, target_class, strength, base_probs, base_lms, lms_texts, **kw):
        assert base_probs is not None
        assert base_lms is not None
        return causal_eval.StrengthResult(
            strength=strength, lms=0.1, base_lms=0.1, lms_ok=True, signed_effect=0.0,
            notes=kw.get("notes", ""),
        )

    monkeypatch.setattr(module, "score_class_probs_by_dataset", _fake_probs)
    monkeypatch.setattr(module, "compute_lms_safe", _fake_lms)
    monkeypatch.setattr(module, "run_strength_on_model", _fake_strength)
    return calls


def test_run_sae_causal_sweep_skips_recompute_when_base_stats_given(monkeypatch):
    calls = _patch_common(monkeypatch, causal_eval)

    @contextmanager
    def _noop(*a, **k):
        yield None

    monkeypatch.setattr(causal_eval, "sae_steering_context", _noop)
    monkeypatch.setattr(causal_eval, "sae_decoder_direction", lambda sae, idx: torch.zeros(4))

    result = causal_eval.run_sae_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        sae=_FakeSAE(),
        module_path="dummy",
        feature_index=0,
        target_class="cls_a",
        strengths=[1.0],
        include_random_control=False,
        report_rows=_meta_rows(),
        base_probs={"cls_a": {"cls_a": 0.5}},
        base_lms=0.1,
        r_base_probs={"cls_a": {"cls_a": 0.5}},
        r_base_lms=0.1,
    )

    assert calls["probs"] == 0
    assert calls["lms"] == 0
    assert result.selected is not None


def test_run_sae_causal_sweep_computes_base_stats_when_not_given(monkeypatch):
    calls = _patch_common(monkeypatch, causal_eval)

    @contextmanager
    def _noop(*a, **k):
        yield None

    monkeypatch.setattr(causal_eval, "sae_steering_context", _noop)
    monkeypatch.setattr(causal_eval, "sae_decoder_direction", lambda sae, idx: torch.zeros(4))

    causal_eval.run_sae_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        sae=_FakeSAE(),
        module_path="dummy",
        feature_index=0,
        target_class="cls_a",
        strengths=[1.0],
        include_random_control=False,
    )

    assert calls["probs"] == 1
    assert calls["lms"] == 1


def test_run_sae_multi_layer_causal_sweep_skips_recompute_when_base_stats_given(monkeypatch):
    calls = _patch_common(monkeypatch, causal_eval)

    @contextmanager
    def _noop(*a, **k):
        yield None

    monkeypatch.setattr(causal_eval, "multi_site_steering_context", _noop)
    monkeypatch.setattr(causal_eval, "sae_decoder_direction", lambda sae, idx: torch.zeros(4))

    layer_specs = [
        {"layer": 3, "module_path": "dummy.L3", "sae": _FakeSAE(), "feature_indices": [0]},
    ]

    result = causal_eval.run_sae_multi_layer_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        layer_specs=layer_specs,
        target_class="cls_a",
        strengths=[1.0],
        include_random_control=False,
        report_rows=_meta_rows(),
        base_probs={"cls_a": {"cls_a": 0.5}},
        base_lms=0.1,
        r_base_probs={"cls_a": {"cls_a": 0.5}},
        r_base_lms=0.1,
    )

    assert calls["probs"] == 0
    assert calls["lms"] == 0
    assert result.selected is not None


def test_run_sae_multi_layer_causal_sweep_accepts_precomputed_directions(monkeypatch):
    """All-layer callers need not retain full SAE objects on the GPU."""
    _patch_common(monkeypatch, causal_eval)

    @contextmanager
    def _noop(*a, **k):
        yield None

    monkeypatch.setattr(causal_eval, "multi_site_steering_context", _noop)

    def _must_not_read_sae(*args, **kwargs):
        raise AssertionError("precomputed direction unexpectedly read an SAE")

    monkeypatch.setattr(causal_eval, "sae_decoder_direction", _must_not_read_sae)
    monkeypatch.setattr(causal_eval, "sae_decoder_bag_direction", _must_not_read_sae)

    result = causal_eval.run_sae_multi_layer_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        layer_specs=[
            {
                "layer": 3,
                "module_path": "dummy.L3",
                "direction": torch.ones(4),
                "feature_indices": [7],
            },
            {
                "layer": 4,
                "module_path": "dummy.L4",
                "direction": torch.ones(4) * 2,
                "feature_indices": [8, 9],
            },
        ],
        target_class="cls_a",
        strengths=[1.0],
        include_random_control=False,
        base_probs={"cls_a": {"cls_a": 0.5}},
        base_lms=0.1,
    )

    assert result.selected is not None
    assert result.meta["n_sites"] == 2


def test_run_caa_causal_sweep_skips_recompute_when_base_stats_given(monkeypatch):
    calls = _patch_common(monkeypatch, caa_eval)

    @contextmanager
    def _noop(*a, **k):
        yield None

    monkeypatch.setattr(caa_eval, "caa_steering_context", _noop)

    specs = [("dummy.L3", torch.ones(4))]

    result = caa_eval.run_caa_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        specs=specs,
        target_class="cls_a",
        strengths=[1.0],
        include_random_control=False,
        report_rows=_meta_rows(),
        base_probs={"cls_a": {"cls_a": 0.5}},
        base_lms=0.1,
        r_base_probs={"cls_a": {"cls_a": 0.5}},
        r_base_lms=0.1,
    )

    assert calls["probs"] == 0
    assert calls["lms"] == 0
    assert result.selected is not None


def test_run_study_causal_wires_cached_base_stats_into_sae_and_caa_sweeps():
    """causal_study.run_study_causal must reuse one cached baseline per section
    rather than letting each sae:*/sae_pre:*/caa:* method call recompute its
    own unsteered base_probs/base_lms."""
    import inspect

    import causal_study

    src = inspect.getsource(causal_study.run_study_causal)
    assert "_cached_base_stats" in src
    assert "_sae_base_probs, _sae_base_lms = _cached_base_stats" in src
    assert "_caa_base_probs, _caa_base_lms = _cached_base_stats" in src
    assert "base_probs=_sae_base_probs" in src
    assert "base_probs=_caa_base_probs" in src
