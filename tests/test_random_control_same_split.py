"""IEND controls are genuinely random, L2-matched package interventions."""

from __future__ import annotations

from pathlib import Path
import sys
from contextlib import contextmanager

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import causal_eval


def _meta_rows():
    return [
        {"sample_id": "cls_a:0", "group": "cls_a", "text": "t1", "label": "x", "label_class": "cls_a"},
        {"sample_id": "neutral:0", "group": "neutral", "text": "n1"},
    ]


def _grid_entry(p_target: float, p_neutral: float, lms: float):
    return {
        "probs_by_dataset": {
            "cls_a": {"cls_a": p_target},
            "neutral": {"cls_a": p_neutral},
        },
        "lms": lms,
    }


def _decoder_results():
    out = {
        "summary": {"cls_a": {"feature_factor": 1.0, "learning_rate": 500.0}},
        "grid": {
            "base": _grid_entry(0.4, 0.1, 1.0),
            (1.0, 500.0): _grid_entry(0.9, 0.15, 0.99),
        },
    }
    out["weaken_results"] = {
        "summary": {
            "cls_a_weaken": {"feature_factor": -1.0, "learning_rate": 500.0}
        },
        "grid": {
            "base": _grid_entry(0.4, 0.1, 1.0),
            (-1.0, 500.0): _grid_entry(0.05, 0.08, 0.99),
        },
    }
    return out


def _decoder_results_calibrated_ff():
    out = {
        "summary": {"cls_a": {"feature_factor": 0.3, "learning_rate": 500.0}},
        "grid": {
            "base": _grid_entry(0.4, 0.1, 1.0),
            (0.3, 500.0): _grid_entry(0.9, 0.15, 0.99),
            (-0.3, 500.0): _grid_entry(0.05, 0.08, 0.97),
        },
    }
    out["weaken_results"] = {
        "summary": {
            "cls_a_weaken": {"feature_factor": -0.3, "learning_rate": 500.0}
        },
        "grid": out["grid"],
    }
    return out


class _FakeModelWith:
    def __init__(self):
        self.base_model = object()
        self.updates = []

    def intervention_update_vector(self, **_kw):
        return torch.tensor([3.0, 4.0])

    @contextmanager
    def intervene(self, **kw):
        self.updates.append(kw["update_vector"].detach().clone())
        yield {}


class _FakeTrainer:
    tokenizer = object()

    def __init__(self):
        self.model_with = _FakeModelWith()
        self.eval_calls = []

    def get_model(self):
        return self.model_with

    def evaluate_base_model(self, _model, _tokenizer, **kw):
        self.eval_calls.append(kw)
        return _grid_entry(0.45, 0.1, 0.99)


def test_non_unit_feature_factor_is_preserved_not_snapped_to_unit():
    """A calibrated (non-+-1) feature_factor used to be silently canonicalized
    to +-1.0 before matching grid cells (``feature_factor = 1.0 if x >= 0
    else -1.0``). A grid genuinely built at ff=0.3/-0.3 (e.g. an
    encoder-mean-calibrated sweep) would then never match its own summary
    entry (ff_match snapped to 1.0) -- `results`/`by_lr` end up empty and
    `select_decoder_headline` raises `KeyError: package_learning_rate=...
    not in decoder grid`, uncaught, right out of this function. Found while
    designing scripts/ablation_actiend_calibrated_ff.py (that ablation's own
    design sidesteps this path by rescaling `lrs` instead of `ff`, but the
    bug is real for any future caller that passes a non-unit
    feature_factor). See CLAUDE.md's 2026-08-27 entry.
    """
    trainer = _FakeTrainer()
    result = causal_eval.run_encoder_causal_for_class(
        trainer,
        _meta_rows(),
        target_class="cls_a",
        backend="actiend",
        decoder_results=_decoder_results_calibrated_ff(),
        include_random_control=True,
        method_id="actiend:cls_a",
    )

    assert result.meta.get("error") is None
    assert result.selected is not None
    assert result.selected.strength == 500.0
    assert result.random_control is not None
    assert result.random_control.is_random_control is True
    assert "genuine random update" in (
        result.random_control.notes or ""
    )
    assert torch.isclose(trainer.model_with.updates[0].norm(), torch.tensor(5.0))


def test_random_control_reuses_same_split_frames_via_package_api():
    training_like_df = pd.DataFrame({"masked": ["a"], "label": ["x"], "label_class": ["cls_a"]})
    neutral_df = pd.DataFrame({"text": ["n"]})
    trainer = _FakeTrainer()
    result = causal_eval.run_encoder_causal_for_class(
        trainer,
        _meta_rows(),
        target_class="cls_a",
        backend="gradiend",
        decoder_results=_decoder_results(),
        include_random_control=True,
        method_id="gradiend:cls_a",
        training_like_df=training_like_df,
        neutral_df=neutral_df,
        decoder_split="test",
    )
    assert trainer.eval_calls[0]["training_like_df"] is training_like_df
    assert trainer.eval_calls[0]["neutral_df"] is neutral_df
    assert result.random_control is not None
    assert "split=test" in (result.random_control.notes or "")


def test_random_control_derives_frame_from_meta_rows_when_not_supplied():
    trainer = _FakeTrainer()
    result = causal_eval.run_encoder_causal_for_class(
        trainer,
        _meta_rows(),
        target_class="cls_a",
        backend="gradiend",
        decoder_results=_decoder_results(),
        include_random_control=True,
        method_id="gradiend:cls_a",
    )

    assert isinstance(trainer.eval_calls[0]["training_like_df"], pd.DataFrame)
    assert result.random_control is not None
    assert result.random_control.is_random_control is True


def test_component_repair_reuses_strengthen_without_evaluating_it(monkeypatch):
    original = causal_eval.run_encoder_causal_for_class(
        _FakeTrainer(),
        _meta_rows(),
        target_class="cls_a",
        backend="gradiend",
        decoder_results=_decoder_results(),
        include_random_control=False,
        method_id="gradiend:cls_a",
    )
    persisted = original.to_dict()
    weaken_only = {"weaken_results": _decoder_results()["weaken_results"]}

    def _forbid_decoder_eval(*_args, **_kwargs):
        raise AssertionError("strengthen decoder evaluation must not run during repair")

    monkeypatch.setattr(
        causal_eval, "evaluate_decoder_for_classes_refined", _forbid_decoder_eval
    )
    trainer = _FakeTrainer()
    repaired = causal_eval.run_encoder_causal_for_class(
        trainer,
        _meta_rows(),
        target_class="cls_a",
        backend="gradiend",
        decoder_results=weaken_only,
        reuse_strengthen=persisted,
        include_random_control=True,
        method_id="gradiend:cls_a",
        decoder_split="test",
    )

    assert [row.to_dict() for row in repaired.strengths] == [
        row.to_dict() for row in original.strengths
    ]
    assert repaired.selected.to_dict() == original.selected.to_dict()
    assert repaired.meta["strengthen_reused"] is True
    assert repaired.weaken_selected is not None
    assert repaired.random_control is not None


def test_inverted_bidirectional_repair_reselects_saved_grids_without_validation_eval():
    def _stored(strength, delta, modified, *, weaken_role):
        return {
            "strength": float(strength),
            "lms": 1.0,
            "base_lms": 1.0,
            "lms_ok": True,
            # The old weaken role stored factual drop, while the group summary
            # retains the raw probability delta needed after swapping roles.
            "signed_effect": float(-delta if weaken_role else delta),
            "selection_metric": None if weaken_role else float(modified),
            "summary_by_group": {
                "cls_a": {
                    "mean_delta_p_target": float(delta),
                    "mean_p_target_base": 0.4,
                    "mean_p_target_mod": float(modified),
                }
            },
        }

    inverted = {
        "method": "agiend:cls_a",
        "backend": "agiend",
        "target_class": "cls_a",
        "selected_strength": 1.0,
        "selected": _stored(1.0, -0.1, 0.3, weaken_role=False),
        "strengths": [
            _stored(1.0, -0.1, 0.3, weaken_role=False),
            _stored(2.0, -0.3, 0.1, weaken_role=False),
        ],
        "weaken_selected_strength": 1.0,
        "weaken_selected": _stored(1.0, -0.1, 0.3, weaken_role=True),
        "weaken_strengths": [
            _stored(1.0, 0.2, 0.6, weaken_role=True),
            _stored(2.0, 0.5, 0.9, weaken_role=True),
        ],
        "meta": {"feature_factor": 1.0, "weaken_feature_factor": -1.0},
    }

    strengthen_curve, strengthen, weaken_curve, weaken = (
        causal_eval.reselect_inverted_bidirectional_curves(
            inverted, target_class="cls_a"
        )
    )
    assert len(strengthen_curve) == len(weaken_curve) == 2
    assert strengthen.strength == 2.0
    assert strengthen.signed_effect == 0.5
    assert strengthen.selection_metric == 0.9
    assert weaken.strength == 2.0
    assert abs(weaken.signed_effect - 0.3) < 1e-12


def test_direct_bidirectional_repair_reselects_rival_panel_from_saved_grid():
    def _stored(strength, *, factual_delta, rival_delta, sign):
        return {
            "strength": float(strength),
            "lms": 1.0,
            "base_lms": 1.0,
            "lms_ok": True,
            # Legacy direct sweeps incorrectly stored the factual-panel delta.
            "signed_effect": float(factual_delta),
            "notes": f"caa sign={sign:g}",
            "summary_by_group": {
                "A": {
                    "mean_delta_p_target": float(factual_delta),
                    "mean_p_target_base": 0.4,
                    "mean_p_target_mod": 0.4 + float(factual_delta),
                },
                "B": {
                    "mean_delta_p_target": float(rival_delta),
                    "mean_p_target_base": 0.2,
                    "mean_p_target_mod": 0.2 + float(rival_delta),
                },
            },
        }

    legacy = {
        "method": "caga:A",
        "backend": "caga",
        "target_class": "A",
        "strengths": [
            _stored(1.0, factual_delta=0.30, rival_delta=0.05, sign=1),
            _stored(2.0, factual_delta=-0.20, rival_delta=0.40, sign=-1),
        ],
        "selected_strength": 1.0,
        "selected": _stored(
            1.0, factual_delta=0.30, rival_delta=0.05, sign=1
        ),
    }

    strengthen_curve, strengthen, weaken_curve, weaken = (
        causal_eval.reselect_direct_bidirectional_curves(
            legacy, target_class="A"
        )
    )
    assert len(strengthen_curve) == len(weaken_curve) == 2
    assert strengthen.strength == 2.0
    assert strengthen.signed_effect == 0.40
    assert strengthen.notes.endswith("sign=-1")
    assert weaken.strength == 2.0
    assert abs(weaken.signed_effect - 0.20) < 1e-12


def test_polarity_repair_runs_only_reselected_test_candidates(monkeypatch):
    original = causal_eval.run_encoder_causal_for_class(
        _FakeTrainer(),
        _meta_rows(),
        target_class="cls_a",
        backend="agiend",
        decoder_results=_decoder_results(),
        include_random_control=False,
        method_id="agiend:cls_a",
    )
    calls = []

    def _one_candidate(_trainer, _classes, **kw):
        calls.append(dict(kw))
        if kw.get("increase_target_probabilities") is False:
            return _decoder_results()["weaken_results"]
        return _decoder_results()

    monkeypatch.setattr(
        causal_eval, "evaluate_decoder_for_classes_refined", _one_candidate
    )
    training_like_df = pd.DataFrame(
        {"masked": ["a"], "label": ["x"], "label_class": ["cls_a"]}
    )
    neutral_df = pd.DataFrame({"text": ["n"]})
    repaired = causal_eval.run_encoder_causal_for_class(
        _FakeTrainer(),
        _meta_rows(),
        target_class="cls_a",
        backend="agiend",
        include_random_control=False,
        method_id="agiend:cls_a",
        decoder_split="test",
        training_like_df=training_like_df,
        neutral_df=neutral_df,
        reuse_inverted_bidirectional=original.to_dict(),
    )

    assert len(calls) == 2
    assert all(call["split"] == "test" for call in calls)
    assert all(call["refine_points"] == 0 for call in calls)
    assert all(len(call["lrs"]) == 1 for call in calls)
    assert all(call["training_like_df"] is training_like_df for call in calls)
    assert all(call["neutral_df"] is neutral_df for call in calls)
    assert repaired.meta["polarity_reselected_from_bidirectional_grid"] is True
    assert len(repaired.strengths) == len(original.weaken_strengths)
    assert len(repaired.weaken_strengths) == len(original.strengths)
