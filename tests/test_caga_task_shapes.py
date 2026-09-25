"""CAGA across task shapes: one-pole, binary, multi-class, and their one-pole
forms, plus answer-only circuit tasks (IOI/key_value) whose CF-only pole has no
factual data.

Two layers:

* Fast, no-model tests of the two functions that actually differ by task shape:
  ``caga_direction_from_signals`` (the estimator; shape enters via the label
  vector and how many rivals collapse into the alternative pole) and the runner's
  claim-class target selection + no-data skip (the fix for the IOI ``SUBJECT``
  crash -- the runner used to try every ``bundle.classes`` entry; it now restricts
  to claim classes exactly like the main causal pipeline).
* Opt-in end-to-end tests (``GRADIEND_RUN_CGA_E2E=1``) that build a real CAGA
  trainer per task shape and assert nonzero contrast, a unit per-layer-concatenated
  direction, and a per-site steering split onto real residual modules.

The estimator is shape-agnostic by construction (``sum(label*(F-A))/sum(label**2)``)
-- these tests pin that each shape's label/rival layout produces the intended
direction, and that answer-only circuit tasks never even attempt the CF-only pole.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from caga_eval import caga_direction_from_signals


# --------------------------------------------------------------------------- #
# Estimator across task shapes (fast, no model)
# --------------------------------------------------------------------------- #
def test_binary_one_pole_single_rival_all_positive():
    # claim=M vs one rival=F, one-pole -> every label +1; direction = mean(gM - gF).
    F = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # gM
    A = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # gF
    lab = np.array([1.0, 1.0, 1.0])
    d = caga_direction_from_signals(F, A, lab)
    assert d == pytest.approx([1.0, 0.0])
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_multiclass_one_pole_collapses_multiple_rivals():
    # claim=asia vs {europe, africa}: the alternative pole averages several rivals.
    # Rivals that disagree (europe up, africa down on axis 1) cancel there, leaving
    # the claim's own direction -- the rival-collapse behavior one-pole relies on.
    ga = np.array([2.0, 0.0])
    A = np.array([[0.0, 2.0], [0.0, -2.0]])  # europe, africa: cancel on axis 1
    F = np.array([ga, ga])
    lab = np.array([1.0, 1.0])
    d = caga_direction_from_signals(F, A, lab)
    assert d == pytest.approx([1.0, 0.0])
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_multiclass_one_pole_anti_aligned_rival_still_recovers_claim():
    # If a rival gradient is anti-aligned with the claim, mean(F-A) still points
    # toward the claim contrast (does not flip) -- direction magnitude just grows.
    ga = np.array([1.0, 0.0])
    A = np.array([[-1.0, 0.0], [-1.0, 0.0]])  # rivals point opposite the claim
    F = np.array([ga, ga])
    d = caga_direction_from_signals(F, A, np.array([1.0, 1.0]))
    assert d == pytest.approx([1.0, 0.0])  # +2 mean transition, normalized


def test_two_pole_pair_uses_signed_transitions():
    # Two-pole: prompt1 factual=gA/alt=gB (label +1), prompt2 swapped (label -1).
    # sum(label*(F-A)) = (gA-gB) - (gB-gA) = 2(gA-gB) -> normalized (gA-gB).
    gA, gB = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    F = np.array([gA, gB])
    A = np.array([gB, gA])
    lab = np.array([1.0, -1.0])
    d = caga_direction_from_signals(F, A, lab)
    assert d == pytest.approx([1.0, 0.0])


def test_source_both_one_pole_mixed_labels_align():
    # source=both one-pole: half the prompts are swapped with label -1. The signed
    # sum still aligns with the claim-vs-rival contrast.
    claim, rival = np.array([0.0, 3.0]), np.array([0.0, 0.0])
    F = np.array([claim, rival, claim, rival])
    A = np.array([rival, claim, rival, claim])
    lab = np.array([1.0, -1.0, 1.0, -1.0])
    d = caga_direction_from_signals(F, A, lab)
    assert d == pytest.approx([0.0, 1.0])


def test_degenerate_when_signed_transitions_cancel():
    # Identical transition under opposite labels -> zero direction (not a crash).
    F = np.array([[1.0, 0.0], [1.0, 0.0]])
    A = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = caga_direction_from_signals(F, A, np.array([1.0, -1.0]))
    assert d == pytest.approx([0.0, 0.0])


def test_all_zero_labels_are_degenerate_not_a_divide_by_zero():
    F = np.array([[1.0, 2.0]])
    A = np.array([[0.0, 0.0]])
    d = caga_direction_from_signals(F, A, np.array([0.0]))
    assert d == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------- #
# Claim-class target selection (fast, no model)
# --------------------------------------------------------------------------- #
class _StubBundle:
    def __init__(self, classes):
        self.classes = list(classes)


class _StubCfg:
    def __init__(self, ablations, training):
        self.raw = {"ablations": ablations}
        self.training = training


def _claim_classes(cfg, bundle):
    """Claim classes exactly as the main causal stage derives them."""
    from study.tasks import claim_classes_for_study

    abl = dict(cfg.raw.get("ablations") or {})
    one_pole = bool(abl.get("one_pole")) and abl.get("pair") is False
    opc = cfg.training.get("one_pole_classes")
    return claim_classes_for_study(
        bundle,
        one_pole=one_pole,
        one_pole_classes=opc if isinstance(opc, (list, tuple)) else None,
    )


def _claim(ablations, training, classes):
    return _claim_classes(_StubCfg(ablations, training), _StubBundle(classes))


def test_claim_classes_binary_pair_keeps_both():
    # A non-one-pole binary task: both poles are real targets.
    assert _claim({"one_pole": False, "pair": True}, {}, ["M", "F"]) == ["M", "F"]


def test_claim_classes_multiclass_pair_keeps_all():
    got = _claim({"one_pole": False, "pair": True}, {}, ["asia", "europe", "africa"])
    assert got == ["asia", "europe", "africa"]


def test_claim_classes_answer_only_circuit_drops_cf_only_pole():
    # IOI-style: one_pole with one_pole_classes=[IO] -> SUBJECT (CF-only) is dropped,
    # so the runner never builds a trainer for a class with no factual transitions.
    got = _claim(
        {"one_pole": True, "pair": False},
        {"one_pole_classes": ["IO"]},
        ["IO", "SUBJECT"],
    )
    assert got == ["IO"]


def test_claim_classes_answer_only_key_value_drops_other():
    got = _claim(
        {"one_pole": True, "pair": False},
        {"one_pole_classes": ["VALUE"]},
        ["VALUE", "OTHER"],
    )
    assert got == ["VALUE"]


# --------------------------------------------------------------------------- #
# End-to-end per task shape (opt-in: needs torch + gpt2)
# --------------------------------------------------------------------------- #
E2E = pytest.mark.skipif(
    os.environ.get("GRADIEND_RUN_CGA_E2E") != "1",
    reason="set GRADIEND_RUN_CGA_E2E=1 to build a real gpt2 CAGA trainer",
)


def _fit_and_specs(task, target):
    import tempfile
    from pathlib import Path

    from study.config import load_study_config
    from study.registry import build_task
    from study.stages.train import _build_trainer_for_artifact
    from study.training_profiles import shared_training_kwargs
    from study.signals.package_adapter import extract_paired_signals_from_trainer
    import caga_eval

    cfg = load_study_config(model="gpt2-small", task=task, suite="full_plus")
    bundle = build_task(task, cfg.raw, smoke=True)
    all_classes = [str(c) for c in bundle.classes]
    shared = shared_training_kwargs(cfg, backend="caga")
    import random

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="caga_shape_") as tmp:
        trainer, *_ = _build_trainer_for_artifact(
            backend="caga", cfg=cfg, bundle=bundle,
            target_classes=[target], experiment_dir=Path(tmp),
            split_mode="none", shared=shared, smoke=True,
            feature_class=target, counterfactual_classes="all",
            all_classes=all_classes,
        )
        batch = extract_paired_signals_from_trainer(
            trainer, split="train", projection_dim=None, include_learned=False,
        )
        fit = caga_eval.fit_caga_direction(trainer)
        sites = caga_eval._resolve_signal_sites(trainer.get_model())
        specs = caga_eval.caga_causal_specs(fit.direction, sites=sites)
        base_mods = dict(trainer.get_model().base_model.named_modules())
    F = np.asarray(batch.factual)
    A = np.asarray(batch.alternative)
    return F, A, fit, sites, specs, base_mods


@E2E
@pytest.mark.parametrize(
    "task, target",
    [
        ("gender_en", "M"),         # binary
        ("ravel_country", "china"), # multi-class
    ],
)
def test_e2e_direction_and_split_per_task_shape(task, target):
    F, A, fit, sites, specs, base_mods = _fit_and_specs(task, target)
    # nonzero contrast and a unit, per-layer-concatenated direction
    assert np.linalg.norm((F - A).mean(axis=0)) > 1e-4
    assert np.isclose(np.linalg.norm(fit.direction), 1.0)
    # one steering spec per residual site, each mapping to a REAL module
    assert len(specs) == len(sites) > 1
    per_widths = {tuple(v.shape) for _m, v in specs}
    assert len(per_widths) == 1  # every site the same width
    assert all(m in base_mods for m, _v in specs)


@E2E
def test_e2e_answer_only_circuit_claim_excludes_cf_pole():
    # IOI: claim selection must yield [IO] only; building SUBJECT would raise the
    # package's no-data refusal.
    from study.config import load_study_config
    from study.registry import build_task

    cfg = load_study_config(model="gpt2-small", task="ioi", suite="full_plus")
    bundle = build_task("ioi", cfg.raw, smoke=True)
    assert _claim_classes(cfg, bundle) == ["IO"]
