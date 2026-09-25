from pathlib import Path
import sys

from gradiend import PrePruneConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.config import StudyConfig, training_kwargs_from_config
from study.training_profiles import build_training_arguments


def test_config_forces_alternative_preprune_source_for_both_training_source():
    cfg = StudyConfig(
        model_key="test-model",
        task_id="test-task",
        suite_id="test-suite",
        raw={
            "model": {"learning_rate": 1e-5},
            "training": {
                "source": "both",
                "pre_prune": {"n_samples": 4, "topk": 0.1, "source": "both"},
                "post_prune": False,
            },
        },
    )

    kwargs = training_kwargs_from_config(cfg)

    assert kwargs["source"] == "both"
    assert kwargs["pre_prune_config"].source == "alternative"


def test_training_profile_normalizes_existing_preprune_source(tmp_path):
    args = build_training_arguments(
        "gradiend",
        experiment_dir=str(tmp_path),
        shared={
            "learning_rate": 1e-5,
            "source": "both",
            "pre_prune_config": PrePruneConfig(source="diff"),
        },
    )

    assert args.source == "both"
    assert args.pre_prune_config.source == "alternative"
