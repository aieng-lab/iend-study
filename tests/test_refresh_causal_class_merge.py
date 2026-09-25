"""Regression coverage for merged-class checkpoint reloads."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from study.method_ids import artifact_dirname, pair_key, sorted_pair
from study.stages import train as train_stage
from study.tasks import TaskBundle


@pytest.mark.parametrize(
    ("task_id", "classes", "merge_map", "transition_groups"),
    [
        (
            "pronoun_number",
            ["singular", "plural"],
            {"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            [["1SG", "1PL"], ["3SG", "3PL"]],
        ),
        (
            "pronoun_person",
            ["1", "2", "3"],
            {"1": ["1SG", "1PL"], "2": ["2SGPL"], "3": ["3SG", "3PL"]},
            None,
        ),
    ],
)
def test_bulk_artifact_reload_preserves_pronoun_class_merge_config(
    tmp_path,
    monkeypatch,
    task_id,
    classes,
    merge_map,
    transition_groups,
):
    """``--refresh-causal`` must rebuild the same class view as training."""

    class FakeConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class FakeTrainer:
        configs = []

        def __init__(self, *, config, **_kwargs):
            self.config = config
            self.configs.append(config)

        def get_model(self, **_kwargs):
            return SimpleNamespace(gradiend=None)

    import gradiend.trainer.text.prediction.trainer as prediction_trainer

    monkeypatch.setattr(prediction_trainer, "TextPredictionConfig", FakeConfig)
    monkeypatch.setattr(prediction_trainer, "TextPredictionTrainer", FakeTrainer)
    monkeypatch.setattr(
        train_stage,
        "build_training_arguments",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(train_stage, "shared_training_kwargs", lambda *_args, **_kwargs: {})

    cfg = SimpleNamespace(
        model_key="gpt2-small",
        task_id=task_id,
        hf_model="gpt2",
        output_dir=tmp_path,
        training={},
        raw={
            "model": {"hf_model": "gpt2", "arch": "gpt2"},
            "ablations": {"pair": True, "one_pole": False},
        },
    )
    raw_classes = sorted({raw for members in merge_map.values() for raw in members})
    data_per_class = {
        raw: pd.DataFrame({"masked": ["x [MASK]"], "split": ["train"], raw: [raw]})
        for raw in raw_classes
    }
    bundle = TaskBundle(
        task_id=task_id,
        classes=list(classes),
        data_per_class=data_per_class,
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame(),
        class_merge_map=merge_map,
        class_merge_transition_groups=transition_groups,
    )

    a, b = sorted_pair(classes[0], classes[1])
    artifact = tmp_path / "artifacts" / artifact_dirname(
        "gradiend", kind="pair", key=pair_key(a, b), split_mode="none"
    )
    artifact.mkdir(parents=True)
    (artifact / "config.json").write_text("{}", encoding="utf-8")

    result = train_stage.reload_train_raw_from_artifacts(
        cfg, bundle, backends=["gradiend"]
    )

    assert result["train_raw_none"]["gradiend"]["trainer"].config.class_merge_map == merge_map
    assert (
        getattr(
            result["train_raw_none"]["gradiend"]["trainer"].config,
            "class_merge_transition_groups",
            None,
        )
        == transition_groups
    )


def test_bulk_artifact_reload_keeps_every_pair_and_one_pole_with_pair_metadata(
    tmp_path,
    monkeypatch,
):
    """A causal refresh must neither overwrite trainers nor erase pair IDs."""

    class FakeConfig(SimpleNamespace):
        pass

    class FakeTrainer:
        def __init__(self, *, config, **_kwargs):
            self.config = config

        def get_model(self, **_kwargs):
            return SimpleNamespace(gradiend=None)

    import gradiend.trainer.text.prediction.trainer as prediction_trainer

    monkeypatch.setattr(prediction_trainer, "TextPredictionConfig", FakeConfig)
    monkeypatch.setattr(prediction_trainer, "TextPredictionTrainer", FakeTrainer)
    monkeypatch.setattr(
        train_stage,
        "build_training_arguments",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(train_stage, "shared_training_kwargs", lambda *_args, **_kwargs: {})

    classes = ["A", "B", "C"]
    cfg = SimpleNamespace(
        model_key="gpt2-small",
        task_id="synthetic_multiclass",
        hf_model="gpt2",
        output_dir=tmp_path,
        training={},
        raw={
            "model": {"hf_model": "gpt2", "arch": "gpt2"},
            "ablations": {"pair": True, "one_pole": True},
        },
    )
    bundle = TaskBundle(
        task_id=cfg.task_id,
        classes=classes,
        data_per_class={
            cls: pd.DataFrame({"masked": ["x [MASK]"], "split": ["train"], cls: [cls]})
            for cls in classes
        },
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame(),
    )

    artifacts = tmp_path / "artifacts"
    for cls in classes:
        artifact = artifacts / artifact_dirname(
            "actiend", kind="onepole", key=cls, split_mode="none"
        )
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        artifact = artifacts / artifact_dirname(
            "actiend", kind="pair", key=pair_key(left, right), split_mode="none"
        )
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")

    result = train_stage.reload_train_raw_from_artifacts(
        cfg, bundle, backends=["actiend"]
    )

    by_class = result["train_raw_none_by_class"]["actiend"]
    for cls in classes:
        raws = by_class[cls] if isinstance(by_class[cls], list) else [by_class[cls]]
        assert len(raws) == 3  # one one-pole plus the two pairs containing cls
        pair_raws = [raw for raw in raws if raw["ablation"] == "pair"]
        assert len(pair_raws) == 2
        assert all(raw["pair"] and cls in raw["pair"] for raw in pair_raws)


def test_bulk_cga_reload_defers_every_checkpoint(tmp_path, monkeypatch):
    """Multiclass CGA reload must not materialize all model-width decoders."""

    class FakeConfig(SimpleNamespace):
        pass

    class FakeTrainer:
        instances = []

        def __init__(self, *, config, **_kwargs):
            self.config = config
            self.get_model_calls = []
            self.instances.append(self)

        def get_model(self, **kwargs):
            self.get_model_calls.append(kwargs)
            return SimpleNamespace(gradiend=None)

    import gradiend.trainer.text.prediction.trainer as prediction_trainer

    monkeypatch.setattr(prediction_trainer, "TextPredictionConfig", FakeConfig)
    monkeypatch.setattr(prediction_trainer, "TextPredictionTrainer", FakeTrainer)
    monkeypatch.setattr(
        train_stage,
        "build_training_arguments",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(train_stage, "shared_training_kwargs", lambda *_args, **_kwargs: {})

    classes = ["A", "B", "C"]
    cfg = SimpleNamespace(
        model_key="llama-3.1-8b",
        task_id="synthetic_multiclass",
        hf_model="llama",
        output_dir=tmp_path,
        training={},
        raw={
            "model": {"hf_model": "llama", "arch": "llama"},
            "ablations": {"pair": True, "one_pole": True},
        },
    )
    bundle = TaskBundle(
        task_id=cfg.task_id,
        classes=classes,
        data_per_class={
            cls: pd.DataFrame({"masked": ["x [MASK]"], "split": ["train"], cls: [cls]})
            for cls in classes
        },
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame(),
    )

    artifacts = tmp_path / "artifacts"
    for cls in classes:
        artifact = artifacts / artifact_dirname(
            "cga", kind="onepole", key=cls, split_mode="none"
        )
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        artifact = artifacts / artifact_dirname(
            "cga", kind="pair", key=pair_key(left, right), split_mode="none"
        )
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")

    result = train_stage.reload_train_raw_from_artifacts(
        cfg, bundle, backends=["cga"]
    )

    assert len(FakeTrainer.instances) == 6
    assert all(not trainer.get_model_calls for trainer in FakeTrainer.instances)
    by_class = result["train_raw_none_by_class"]["cga"]
    unique_raws = {
        id(raw): raw
        for raw_or_list in by_class.values()
        for raw in (raw_or_list if isinstance(raw_or_list, list) else [raw_or_list])
    }
    assert len(unique_raws) == 6
    assert all(Path(raw["_reload_checkpoint"]).is_dir() for raw in unique_raws.values())
