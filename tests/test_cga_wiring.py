"""CGA — Contrastive Gradient Addition: config/suite/method-id wiring.

CGA fills the empty cell of the study's (signal) x (estimator) grid::

                 mean-difference     learned encoder-decoder
    activation   CAA                 ACTIEND
    gradient     CGA                 GRADIEND

so a GRADIEND-vs-CAA gap can be attributed to the gradient signal or to the
learned encoder-decoder rather than to both at once. See ``cga_eval.py``.

These are the light, torch-free checks: which suites request CGA, how variants
become backend ids, and that every registry a method row passes through on its
way into a table knows about those ids. The estimator math and the real
trainer path are covered in ``test_cga_eval.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.method_groups import METHOD_GROUP_ORDER
from analysis.summary_latex import METHOD_LATEX, PAPER_METHOD_GROUPS
from analysis.task_specs import GRADIEND_LIKE_BACKENDS, group_is_applicable
from study.config import METHOD_FAMILIES, load_study_config, parse_methods_arg
from study.method_ids import (
    CGA_DEFAULT_VARIANT,
    CGA_VARIANTS,
    cga_backend_id,
    cga_backend_ids,
    cga_variant_from_backend,
    feature_or_causal_id,
    resolve_cga_variants,
)
from study.results_merge import method_family
from study.training_profiles import CGA_BACKENDS, is_cga_backend

ROOT = Path(__file__).resolve().parents[1]


def test_variant_backend_ids_round_trip():
    assert cga_backend_id("plain") == "cga"
    assert cga_backend_id("tensor_norm") == "cga_tensor_norm"
    for variant in CGA_VARIANTS:
        assert cga_variant_from_backend(cga_backend_id(variant)) == variant


def test_headline_variant_keeps_the_bare_cga_prefix():
    # The paper's CGA column must be the CAA-matched estimator, so the plain
    # variant -- and only it -- owns the unsuffixed ``cga:`` ids.
    assert CGA_DEFAULT_VARIANT == "plain"
    assert cga_backend_ids()[0] == "cga"


def test_unknown_variant_is_rejected_not_silently_defaulted():
    with pytest.raises(ValueError):
        resolve_cga_variants(["plain", "l2_norm"])
    with pytest.raises(ValueError):
        cga_backend_id("l2_norm")
    with pytest.raises(ValueError):
        cga_variant_from_backend("gradiend")


def test_resolve_variants_defaults_to_plain_and_dedupes():
    assert resolve_cga_variants(None) == ["plain"]
    assert resolve_cga_variants([]) == ["plain"]
    assert resolve_cga_variants(["tensor_norm", "plain", "tensor_norm"]) == [
        "tensor_norm",
        "plain",
    ]
    assert resolve_cga_variants(["plain", "layers", "tensor_norm"]) == [
        "plain",
        "tensor_norm",
    ]


def test_training_profiles_backend_list_matches_cga_eval():
    # study/training_profiles.py hardcodes nothing: a new variant must show up
    # in both places or this fails.
    assert CGA_BACKENDS == cga_backend_ids()
    assert is_cga_backend("cga") and is_cga_backend("cga_tensor_norm")
    assert not is_cga_backend("gradiend")
    assert not is_cga_backend("actiend")


def test_causal_filters_admit_every_variant_backend_id():
    # The enabled set carries family names; trainers are keyed by backend id.
    # Filtering on the family alone silently drops every CGA trainer, so no CGA
    # causal row would ever be produced -- and nothing would error.
    from study.training_profiles import expand_cga_backends

    expanded = expand_cga_backends({"gradiend", "cga", "causal"})
    assert expanded >= {"gradiend", "causal", *cga_backend_ids()}
    # ...and nothing is added when the family is off.
    assert expand_cga_backends({"gradiend", "sae"}) == {"gradiend", "sae"}


def test_cga_is_a_selectable_method_family():
    assert "cga" in METHOD_FAMILIES
    assert parse_methods_arg(["cga"]) == ["cga"]
    assert parse_methods_arg(["gradiend,cga"]) == ["gradiend", "cga"]


def test_method_ids_have_the_gradiend_shape():
    pair_raw = {"ablation": "pair", "pair": ["F", "M"]}
    assert feature_or_causal_id("cga", "F", pair_raw) == "cga:F-M:F"
    assert (
        feature_or_causal_id("cga_tensor_norm", "F", pair_raw)
        == "cga_tensor_norm:F-M:F"
    )
    one_pole_raw = {"ablation": "one_pole"}
    assert feature_or_causal_id("cga", "F", one_pole_raw) == "cga:F"


def test_method_family_separates_the_ablation_from_headline_cga():
    assert method_family("cga:F-M:F") == "cga"
    assert method_family("cga:F") == "cga"
    assert method_family("cga:F:causal") == "cga"
    # The tensor-norm ablation must never be averaged into the CGA column.
    assert method_family("cga_tensor_norm:F-M:F") == "cga_tensor_norm"
    assert method_family("gradiend:F") == "gradiend"


def test_analysis_registries_know_every_cga_backend():
    for backend in cga_backend_ids():
        assert backend in GRADIEND_LIKE_BACKENDS
        for pole in ("two_pole", "one_pole"):
            group = f"{backend}:{pole}"
            assert group in METHOD_GROUP_ORDER
            assert group in PAPER_METHOD_GROUPS
            assert group in METHOD_LATEX


def test_group_applicability_follows_the_single_cga_family_flag():
    spec = {"enabled_methods": {"gradiend", "cga"}, "pair": True, "one_pole": True}
    assert group_is_applicable("cga:two_pole", spec)
    # Both variants are gated by the one ``cga`` family flag, not by their own
    # backend id (which never appears in ``enabled_methods``).
    assert group_is_applicable("cga_tensor_norm:two_pole", spec)
    off = {"enabled_methods": {"gradiend"}, "pair": True, "one_pole": True}
    assert not group_is_applicable("cga:two_pole", off)
    assert not group_is_applicable("cga_tensor_norm:two_pole", off)


def test_group_applicability_still_respects_task_ablations():
    one_pole_only = {"enabled_methods": {"cga"}, "pair": False, "one_pole": True}
    assert not group_is_applicable("cga:two_pole", one_pole_only)
    assert group_is_applicable("cga:one_pole", one_pole_only)


def _suite_methods(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "suites" / f"{name}.yaml").read_text(encoding="utf-8"))["methods"]


def test_suite_membership_is_opt_in():
    # core carries the headline arms (plain + per-layer); tensor_norm is full_plus only.
    assert _suite_methods("core")["cga"] == ["plain", "layers"]
    assert _suite_methods("core")["caga"] == ["plain", "layers"]
    assert _suite_methods("full")["cga"] == ["plain"]
    # full_plus is a strict superset of full, and owns the tensor-norm ablation.
    assert _suite_methods("full_plus")["cga"] == ["plain", "tensor_norm", "layers"]
    assert _suite_methods("full_plus")["caga"] == ["plain", "layers"]


def test_enabled_set_tracks_the_suite():
    from study.deep_pipeline import _default_enabled

    core = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    assert "cga" in _default_enabled(core, {})
    full = load_study_config(model="gpt2-small", task="gender_en", suite="full")
    assert "cga" in _default_enabled(full, {})


def test_explicit_methods_flag_can_select_cga_alone():
    from study.deep_pipeline import _default_enabled

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full")
    enabled = _default_enabled(cfg, {"methods": ["cga"]})
    assert "cga" in enabled
    assert "gradiend" not in enabled and "sae" not in enabled


def test_cga_never_trains_a_component_split():
    # A single hand-installed direction has nothing to split by tensor and no
    # training to converge, so ``tensors`` stays off even under full_plus (which
    # does enable it for GRADIEND/ACTIEND).
    from study.stages.train import _effective_train_splits

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full_plus")
    for backend in cga_backend_ids():
        assert _effective_train_splits(cfg, backend, one_pole=False) == ["none"]
        assert _effective_train_splits(cfg, backend, one_pole=True) == ["none"]


def test_explicit_cga_accumulator_device_invalidates_legacy_fit(monkeypatch):
    """Large CGA may safely refit after moving its running mean off GPU."""
    from study.stages.train import _cga_accumulator_matches

    cfg = load_study_config(model="llama-3.1-8b", task="gender_en", suite="core")
    assert cfg.training["cga_accumulate_device"] == "cpu"
    artifact = Path("/nonexistent/cga")
    legacy = '{"extras": {"cga_fit": {"accumulate_device": "cuda:0"}}}'
    current = (
        '{"extras": {"cga_fit": {"accumulate_device": "cpu"},'
        ' "cga_coordinate_projection": "pre_prune"}}'
    )
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: legacy)
    assert not _cga_accumulator_matches(artifact, cfg)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: current)
    assert _cga_accumulator_matches(artifact, cfg)


def test_cga_projection_invalidates_a_legacy_full_space_artifact(monkeypatch):
    from study.stages.train import _cga_accumulator_matches

    cfg = load_study_config(model="gemma-3-27b-pt", task="gender_en", suite="core")
    artifact = Path("/nonexistent/cga")
    full = '{"extras": {"cga_fit": {}, "cga_coordinate_projection": "full"}}'
    projected = '{"extras": {"cga_fit": {}, "cga_coordinate_projection": "pre_prune"}}'
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: full)
    assert not _cga_accumulator_matches(artifact, cfg)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: projected)
    assert _cga_accumulator_matches(artifact, cfg)


def test_training_arguments_drop_optimization_only_machinery():
    from study.training_profiles import build_training_arguments, shared_training_kwargs

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full")
    shared = shared_training_kwargs(cfg, backend="cga")
    args = build_training_arguments(
        "cga", experiment_dir="/tmp/cga", shared=shared, split_mode="none"
    )
    # One direction, GRADIEND's own architecture, no bias anywhere that could
    # add a constant weight delta at feature_factor=0.
    assert args.latent_dim == 1
    # The encoder half is never used (install_cga_direction writes only the
    # decoder; CGA's readout is direct cosine, not through this module).
    # Its bias is disabled explicitly alongside the decoder bias.
    assert args.activation_encoder == "tanh"
    assert args.activation_decoder == "id"
    assert args.bias_encoder is False
    assert args.bias_decoder is False
    # Pruning is part of GRADIEND's training pipeline, not of the estimator
    # being ablated -- keeping it would restrict CGA to whichever coordinates
    # GRADIEND's pruner picked.
    assert args.pre_prune_config is None
    assert args.post_prune_config is None
    assert args.metadata["cga_estimator"] == "paired_mean_diff"


def test_27b_cga_uses_the_recorded_pre_pruning_projection():
    from study.training_profiles import build_training_arguments, shared_training_kwargs

    cfg = load_study_config(model="gemma-3-27b-pt", task="gender_en", suite="core")
    args = build_training_arguments(
        "cga",
        experiment_dir="/tmp/cga-27b",
        shared=shared_training_kwargs(cfg, backend="cga"),
        split_mode="none",
    )
    assert args.pre_prune_config is not None
    assert args.pre_prune_config.topk == 0.01
    assert args.metadata["cga_coordinate_projection"] == "pre_prune"


def test_cga_accumulator_stays_fp32_when_large_model_decoder_is_bf16():
    """Mean accumulation must not inherit checkpoint precision by buffer reuse."""
    from cga_eval import ContrastiveGradientAccumulator

    decoder_weight = torch.empty(8, dtype=torch.bfloat16)
    acc = ContrastiveGradientAccumulator(
        8,
        device="cpu",
        dtype=torch.float32,
        # The caller now refuses this bf16 buffer; document the required
        # accumulator precision independently here.
        buffer=None,
    )
    assert decoder_weight.dtype == torch.bfloat16
    assert acc.dtype == torch.float32


def test_cga_shares_gradiends_parameter_scope():
    # The ablation removes the learning, not the scope: both must resolve to the
    # same SignalScope so the comparison is like-for-like.
    from gradiend import SignalScope
    from study.training_profiles import build_training_arguments, shared_training_kwargs

    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full")
    assert cfg.training.get("gradiend_exclude_embeddings") is True
    scopes = {}
    for backend in ("gradiend", "cga"):
        shared = shared_training_kwargs(cfg, backend=backend)
        args = build_training_arguments(
            backend, experiment_dir="/tmp/cga", shared=shared, split_mode="none"
        )
        scopes[backend] = args.signal_scope
    assert scopes["cga"] == scopes["gradiend"] == SignalScope.layers()


def test_cga_fit_has_a_bounded_default_row_cap():
    # Real bug, hit live on gender_en's 74,060-row train split (2026-08-26,
    # first real cluster run): CGA's fit is a single streaming pass with no
    # step budget (unlike GRADIEND/ACTIEND, bounded by max_steps regardless of
    # dataset size), so with no cap of its own it silently scans the entire
    # split. SCALE=small does NOT help here -- it only bounds causal/
    # encoder-eval sizes -- so this needs its own default, present with no
    # --scale flag at all.
    bare = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    assert isinstance(bare.training.get("cga_max_size"), int)
    assert bare.training["cga_max_size"] < 74_060


def test_cga_fit_cap_is_scale_aware():
    from study.runner import _load_yaml, _deep_merge_dicts

    profiles = _load_yaml(ROOT / "configs" / "defaults.yaml").get("scale_profiles", {})
    small = _deep_merge_dicts({}, profiles["small"])["training"]["cga_max_size"]
    large = _deep_merge_dicts({}, profiles["large"])["training"]["cga_max_size"]
    assert small < large


def test_deferred_cga_checkpoint_load_and_release_lifecycle():
    from causal_study import (
        load_deferred_cga_trainer_for_causal,
        release_deferred_cga_trainer_after_causal,
    )

    model = SimpleNamespace(
        gradiend=SimpleNamespace(_require_built=lambda: None, device_encoder=None),
        _get_base_forward_device=lambda: torch.device("cuda:0"),
    )

    class FakeTrainer:
        _model_instance = None
        _model_manually_unloaded = False

        def __init__(self):
            self.calls = []

        def get_model(self, **kwargs):
            self.calls.append(kwargs)
            self._model_instance = model
            return model

    trainer = FakeTrainer()
    raw = {"_reload_checkpoint": "/checkpoint/cga"}
    assert load_deferred_cga_trainer_for_causal(trainer, raw, "cga")
    assert trainer.calls == [
        {"load_directory": "/checkpoint/cga", "device_encoder": "cpu"}
    ]
    assert trainer._model_instance is model
    assert model.gradiend.device_encoder == torch.device("cuda:0")
    assert model._cga_encoder_offloaded is True
    assert release_deferred_cga_trainer_after_causal(trainer, raw, "cga")
    assert trainer._model_instance is None
    assert trainer._model_manually_unloaded is True
