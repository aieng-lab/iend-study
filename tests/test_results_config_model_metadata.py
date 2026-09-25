from __future__ import annotations

import pandas as pd

from study.config import StudyConfig
from study.deep_pipeline import _results_config_block
from study.tasks import TaskBundle


def test_results_config_uses_resolved_model_sae_metadata(monkeypatch):
    import study.deep_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "_poc_sae_report_config",
        lambda: {
            "sae_release": "wrong-release",
            "sae_layer": 99,
            "sae_layers": [98, 99],
            "sae_top_k": 128,
        },
    )
    cfg = StudyConfig(
        model_key="gpt2-small",
        task_id="gender_en",
        suite_id="full_plus",
        raw={
            "model": {
                "hf_model": "gpt2",
                "sae_release": "gpt2-small-resid-post-v5-32k",
                "n_layers": 12,
            },
            "ablations": {"pair": True, "one_pole": True},
            "training": {},
        },
    )
    empty = pd.DataFrame()
    bundle = TaskBundle(
        task_id="gender_en",
        classes=["F", "M"],
        data_per_class={},
        merged_df=empty,
        neutrals=empty,
    )

    block = _results_config_block(cfg, bundle, enabled={"sae"})

    assert block["sae_release"] == "gpt2-small-resid-post-v5-32k"
    assert block["sae_layer"] == 11
    assert block["sae_layers"] == list(range(12))
    assert block["sae_top_k"] == 128
