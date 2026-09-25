"""Activation-clamping SAE causal ablation (Templeton et al. 2024 style).

Mirrors ``tests/test_causal_base_stats_cache.py``'s stub pattern: no real
model/SAE required, ``run_sae_clamp_causal_sweep`` is exercised with the same
kind of no-op context manager + fake strength scorer used there for the
additive ``run_sae_causal_sweep``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import causal_eval
from study.stages.sae import sae_suite_params


def _meta_rows():
    return [
        {"sample_id": "cls_a:0", "group": "cls_a", "text": "t1"},
        {"sample_id": "cls_a:1", "group": "cls_a", "text": "t2"},
        {"sample_id": "neutral:0", "group": "neutral", "text": "n1"},
    ]


class _FakeSAE:
    W_dec = torch.eye(4)[:, :4]


def _patch_common(monkeypatch):
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
            strength=strength, lms=0.1, base_lms=0.1, lms_ok=True, signed_effect=0.0
        )

    @contextmanager
    def _noop_clamp_ctx(*a, **k):
        yield None

    monkeypatch.setattr(causal_eval, "score_class_probs_by_dataset", _fake_probs)
    monkeypatch.setattr(causal_eval, "compute_lms_safe", _fake_lms)
    monkeypatch.setattr(causal_eval, "run_strength_on_model", _fake_strength)
    monkeypatch.setattr(causal_eval, "sae_clamp_steering_context", _noop_clamp_ctx)
    monkeypatch.setattr(causal_eval, "sae_decoder_direction", lambda sae, idx: torch.zeros(4))
    monkeypatch.setattr(causal_eval, "sae_feature_max_activation", lambda *a, **k: 1.0)
    return calls


def test_run_sae_clamp_causal_sweep_skips_recompute_when_base_stats_given(monkeypatch):
    calls = _patch_common(monkeypatch)

    result = causal_eval.run_sae_clamp_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        sae=_FakeSAE(),
        module_path="dummy",
        feature_index=0,
        target_class="cls_a",
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
    assert result.meta["mode"] == "clamp"


def test_run_sae_clamp_causal_sweep_computes_base_stats_when_not_given(monkeypatch):
    calls = _patch_common(monkeypatch)

    causal_eval.run_sae_clamp_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        sae=_FakeSAE(),
        module_path="dummy",
        feature_index=0,
        target_class="cls_a",
        include_random_control=False,
    )

    assert calls["probs"] == 1
    assert calls["lms"] == 1


def test_run_sae_clamp_causal_sweep_covers_target_multiplier_grid(monkeypatch):
    _patch_common(monkeypatch)

    result = causal_eval.run_sae_clamp_causal_sweep(
        object(),
        object(),
        _meta_rows(),
        sae=_FakeSAE(),
        module_path="dummy",
        feature_index=0,
        target_class="cls_a",
        target_multipliers=(0.0, 1.0, 2.0),
        include_random_control=False,
        base_probs={"cls_a": {"cls_a": 0.5}},
        base_lms=0.1,
    )

    # sae_feature_max_activation is stubbed to 1.0, so the coarse grid is
    # exactly the multipliers themselves (0, 1, 2); refine adds more points
    # around the (stub-selected) coarse pick, so len(results) >= 3.
    coarse_targets = {0.0, 1.0, 2.0}
    seen_targets = {r.strength for r in result.strengths}
    assert coarse_targets.issubset(seen_targets)


def test_sae_suite_params_parses_clamp_tag():
    assert sae_suite_params(["clamp"])["clamp"] is True
    assert sae_suite_params(["k1", "kstar"])["clamp"] is False
    assert sae_suite_params([])["clamp"] is False


def test_full_suite_yaml_enables_sae_clamp_ablation():
    path = Path(__file__).resolve().parents[1] / "configs" / "suites" / "full.yaml"
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "clamp" in (suite.get("methods") or {}).get("sae", [])


def test_full_suite_resolves_sae_clamp_enabled_but_core_does_not():
    from study.config import load_study_config

    full_cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full")
    core_cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    assert sae_suite_params((full_cfg.suite.get("methods") or {}).get("sae") or [])["clamp"] is True
    assert sae_suite_params((core_cfg.suite.get("methods") or {}).get("sae") or [])["clamp"] is False


def test_runtime_benchmark_sae_k1_is_one_site_k1_only():
    from study.config import load_study_config

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="runtime_benchmark_sae_k1")
    params = sae_suite_params((cfg.suite.get("methods") or {}).get("sae") or [])
    assert "kstar" not in params["tags"]
    assert params["readout_ks"] == (1,)
    assert list(cfg.training["sae_select_sites"]) == ["prediction"]

    full = load_study_config(model="gpt2-small", task="gender_en", suite="runtime_benchmark")
    assert "kstar" in sae_suite_params((full.suite.get("methods") or {}).get("sae") or [])["tags"]
    assert list(full.training["sae_select_sites"]) == ["prediction", "pre_prediction"]


def test_causal_kstar_is_gated_on_suite_tag():
    src = (Path(__file__).resolve().parents[1] / "causal_study.py").read_text(encoding="utf-8")
    assert "sae_kstar_enabled: bool = True" in src
    assert src.count("SAE_KSTAR_ENABLED") == 3  # assignment + all-layers gate + per-layer gate
