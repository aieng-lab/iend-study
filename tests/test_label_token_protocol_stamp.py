"""Artifacts remember which decoder-only label-token protocol trained them.

New runs use the package default ("canonical"); a reload must follow the artifact's
own stamp so nothing already trained is re-interpreted (unstamped == "legacy").
"""

import json
from pathlib import Path

import pytest


def _done(tmp_path: Path, extras) -> Path:
    d = tmp_path / "gradiend__pair__F-M"
    d.mkdir()
    payload = {"stage": d.name, "config_hash": "x"}
    if extras is not None:
        payload["extras"] = extras
    (d / "done.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_unstamped_artifact_is_legacy(tmp_path):
    from study.stages.train import artifact_label_token_protocol

    assert artifact_label_token_protocol(_done(tmp_path, {"learning_rate": 1e-5})) == "legacy"


def test_artifact_without_extras_is_legacy(tmp_path):
    from study.stages.train import artifact_label_token_protocol

    assert artifact_label_token_protocol(_done(tmp_path, None)) == "legacy"


def test_missing_or_unreadable_done_json_is_legacy(tmp_path):
    from study.stages.train import artifact_label_token_protocol

    assert artifact_label_token_protocol(tmp_path / "does-not-exist") == "legacy"
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "done.json").write_text("{not json", encoding="utf-8")
    assert artifact_label_token_protocol(bad) == "legacy"


@pytest.mark.parametrize("stamp", ["canonical", "legacy"])
def test_stamp_round_trips_through_done_json(tmp_path, stamp):
    from study.stages.train import _mark_train_artifact_done, artifact_label_token_protocol

    d = tmp_path / "actiend__pair__F-M"
    _mark_train_artifact_done(d, config_hash="h", extras={"label_token_protocol": stamp})
    assert artifact_label_token_protocol(d) == stamp


def test_unknown_stamp_is_treated_as_legacy(tmp_path):
    from study.stages.train import artifact_label_token_protocol

    assert artifact_label_token_protocol(_done(tmp_path, {"label_token_protocol": "v9"})) == "legacy"


def test_config_forwards_the_protocol_only_when_set():
    from study.config import load_study_config, training_kwargs_from_config

    cfg = load_study_config(model="gpt2-small", task="gender_en")
    assert "label_token_protocol" not in training_kwargs_from_config(cfg, backend="gradiend")

    cfg.raw["training"]["label_token_protocol"] = "legacy"
    assert training_kwargs_from_config(cfg, backend="gradiend")["label_token_protocol"] == "legacy"
