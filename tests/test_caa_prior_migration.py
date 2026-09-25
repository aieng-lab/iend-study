"""Test-only migration of CAA encoder metrics onto validation-frozen Spec_n/Excl.

A stored CAA run already holds its fitted vectors (``raw.caa.vectors``) and each
method row's ``val_readout`` (which carries the frozen threshold + sign). The
migration must reproduce a full recompute exactly while (a) not refitting the
vectors and (b) not scoring the validation split at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import caa_eval
from caa_eval import CaaVectors, encode_caa_scores, run_caa_study
from sae_eval import FrozenRulesUnavailable
from study.stages.caa import prior_caa_val_readouts

LAYERS = (0, 1)
POLICY = "prediction"


def _vectors() -> CaaVectors:
    def unit(v):
        v = np.asarray(v, dtype=np.float64)
        return v / np.linalg.norm(v)

    by_layer = {
        "F": {0: unit([1, 0]), 1: unit([1, 0])},
        "M": {0: unit([-1, 0]), 1: unit([-1, 0])},
    }
    by_concat = {c: np.concatenate([by_layer[c][0], by_layer[c][1]]) for c in by_layer}
    meta = {
        c: {"target_class": c, "rival_classes": [o for o in by_layer if o != c],
            "ablation": "one_pole", "pair": None}
        for c in by_layer
    }
    return CaaVectors(
        policy=POLICY, layers=LAYERS, by_layer=by_layer, by_concat=by_concat,
        module_by_layer={0: "m0", 1: "m1"}, contrast_meta=meta,
    )


def _gender_df() -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        for i in range(30):
            for cls, tok in (("F", "she"), ("M", "he")):
                rows.append({"masked": f"s{i} [MASK]", "label": tok,
                             "label_class": cls, "split": split})
    return pd.DataFrame(rows)


def _neutral_df() -> pd.DataFrame:
    splits = ["train"] * 3 + ["validation"] * 3 + ["test"] * 3
    return pd.DataFrame({
        "text": [f"{sp} neutral text number {i}" for i, sp in enumerate(splits)],
        "split": splits,
    })


@pytest.fixture
def fake_extractors(monkeypatch):
    """Deterministic activations; records which splits were actually scored."""
    calls = {"labeled": [], "neutral": 0}
    dist = {"validation": 0.15, "test": 0.55}  # neutrals drift toward the target on test

    def labeled(model, tokenizer, df, *, policy, layers, **kw):
        split = str(df["split"].iloc[0])
        calls["labeled"].append(split)
        rng = np.random.default_rng({"train": 3, "validation": 1, "test": 2}[split])
        sign = np.where(df["label_class"].astype(str).to_numpy() == "F", 1.0, -1.0)
        base = np.stack([sign, np.zeros_like(sign)], axis=1)
        return {L: base + rng.normal(0, 0.6, base.shape) for L in layers}

    def neutral(model, tokenizer, chunk, *, policy, layers, **kw):
        calls["neutral"] += 1
        split = "validation" if str(chunk[0]).startswith("validation") else "test"
        rng = np.random.default_rng({"validation": 11, "test": 12}[split])
        base = np.tile([dist[split], 1.0], (len(chunk) * 4, 1))
        return {L: base + rng.normal(0, 0.6, base.shape) for L in layers}

    monkeypatch.setattr(caa_eval, "_extract_gender_all_layer_acts", labeled)
    monkeypatch.setattr(caa_eval, "_extract_neutral_all_layer_acts", neutral)
    return calls


def _encode(prior=None):
    return encode_caa_scores(
        None, None, _gender_df(), _neutral_df(), _vectors(),
        target_classes=["F", "M"], n_bootstrap=0, batch_size=8,
        prior_val_readouts=prior,
    )


def test_prior_val_readouts_reproduce_full_recompute_without_scoring_validation(
    fake_extractors,
):
    full = _encode()
    assert "validation" in fake_extractors["labeled"]
    full_labeled_calls = list(fake_extractors["labeled"])
    full_neutral_calls = fake_extractors["neutral"]
    assert full_neutral_calls == 2  # one validation chunk + one test chunk

    prior = {mid: m["val_readout"] for mid, m in full["method_metrics"].items()}
    fake_extractors["labeled"].clear()
    fake_extractors["neutral"] = 0
    migrated = _encode(prior)

    assert fake_extractors["labeled"] == ["test"]  # validation never scored
    assert fake_extractors["neutral"] == 1
    assert len(full_labeled_calls) == 2

    assert set(migrated["method_metrics"]) == set(full["method_metrics"])
    keys = ("neutral_specificity", "class_exclusivity", "roc_auc_neutral",
            "roc_auc_other", "neutral_youden_threshold")
    for mid, m in migrated["method_metrics"].items():
        assert m["val_readout_source"] == "prior", mid
        assert m["val_readout"] == full["method_metrics"][mid]["val_readout"], mid
        assert m["neutral_youden_threshold_source"] == "provided", mid
        for k in keys:
            assert m[k] == pytest.approx(full["method_metrics"][mid][k]), (mid, k)


def test_incomplete_prior_falls_back_to_scoring_validation(fake_extractors):
    full = _encode()
    prior = {mid: m["val_readout"] for mid, m in full["method_metrics"].items()}
    prior.pop(next(iter(prior)))  # one method without a stored validation readout
    fake_extractors["labeled"].clear()
    redone = _encode(prior)
    assert "validation" in fake_extractors["labeled"]
    assert all(m["val_readout_source"] == "scored" for m in redone["method_metrics"].values())


def _rows(vr):
    return [
        {"method": "caa:F:prediction", "status": "ok", "metrics": {"val_readout": vr}},
        {"method": "caa:M:prediction", "status": "ok", "metrics": {},
         "extras": {"readout_metrics": {"val_readout": vr}}},
        {"method": "caa:F:prediction:L0", "status": "error", "error": "boom",
         "metrics": {"val_readout": vr}},
        {"method": "caa:M:prediction:L0", "status": "ok", "metrics": {}},
    ]


def test_prior_caa_val_readouts_collects_from_metrics_or_extras_and_skips_errors():
    vr = {"neutral_youden_threshold": 0.1, "neutral_youden_sign": 1.0}
    got = prior_caa_val_readouts(_rows(vr))
    assert set(got) == {"caa:F:prediction", "caa:M:prediction"}




def _payload(layers=LAYERS):
    v = _vectors()
    return {
        POLICY: {
            "layers": list(layers),
            "module_by_layer": {str(k): m for k, m in v.module_by_layer.items()},
            "by_layer": {c: {str(L): v.by_layer[c][L].tolist() for L in v.layers}
                         for c in v.by_layer},
            "by_concat": {c: v.by_concat[c].tolist() for c in v.by_concat},
            "contrast_meta": v.contrast_meta,
        }
    }


def _run(monkeypatch, *, layers, prior_vectors, prior_val):
    seen = {}

    def no_fit(*a, **k):
        seen["fit"] = True
        return {POLICY: _vectors()}

    def fake_encode(*a, **k):
        seen["prior_val"] = k.get("prior_val_readouts")
        return {"method_metrics": {}}

    monkeypatch.setattr(caa_eval, "fit_caa_vectors", no_fit)
    monkeypatch.setattr(caa_eval, "encode_caa_scores", fake_encode)
    run_caa_study(
        None, None, _gender_df(), _neutral_df(), target_classes=["F", "M"],
        layers=list(layers), policies=[POLICY], hf_resid_template="m{layer}",
        prior_vectors_payload=prior_vectors, prior_val_readouts=prior_val,
    )
    return seen


def test_run_caa_study_reuses_stored_vectors_and_skips_the_fit(monkeypatch):
    seen = _run(monkeypatch, layers=LAYERS, prior_vectors=_payload(), prior_val={"x": {}})
    assert "fit" not in seen
    assert seen["prior_val"] == {"x": {}}


def test_run_caa_study_refits_and_drops_prior_val_when_layers_differ(monkeypatch):
    seen = _run(monkeypatch, layers=(0, 1, 2), prior_vectors=_payload(), prior_val={"x": {}})
    assert seen.get("fit") is True
    assert seen["prior_val"] is None  # stored val was for different vectors


def test_missing_validation_split_raises_instead_of_reporting_an_oracle(fake_extractors):
    df = _gender_df()
    df = df[df["split"] != "validation"]
    with pytest.raises(FrozenRulesUnavailable):
        encode_caa_scores(
            None, None, df, _neutral_df(), _vectors(),
            target_classes=["F", "M"], n_bootstrap=0, batch_size=8,
        )
