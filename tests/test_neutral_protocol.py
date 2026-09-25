"""Neutral split, excluded tokens, validation Youden."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from neutral_protocol import (
    excluded_token_id_set,
    make_nonpad_nonexcluded_selector,
    sample_and_split_neutrals,
    texts_for_split,
)
from sae_eval import (
    class_vs_neutral_metrics,
    encoder_class_vs_neutral_metrics,
    encoder_per_class_metrics,
    infer_class_encoding_directions,
)


def test_sample_and_split_uses_explicit_counts_not_fractions():
    df = pd.DataFrame({"text": [f"row{i}" for i in range(50)]})
    a = sample_and_split_neutrals(
        df, test_rows=10, val_rows=5, train_rows=8, seed=0
    )
    b = sample_and_split_neutrals(
        df, test_rows=10, val_rows=5, train_rows=8, seed=0
    )
    c = sample_and_split_neutrals(
        df, test_rows=10, val_rows=5, train_rows=8, seed=1
    )
    assert a["text"].tolist() == b["text"].tolist()
    assert a["text"].tolist() != df["text"].head(10).tolist()
    assert a["text"].tolist() != c["text"].tolist()
    assert (a["split"] == "test").sum() == 10
    assert (a["split"] == "validation").sum() == 5
    assert (a["split"] == "train").sum() == 8


def test_sample_and_split_requires_all_counts():
    df = pd.DataFrame({"text": [f"row{i}" for i in range(10)]})
    try:
        sample_and_split_neutrals(df, max_rows=10)
    except ValueError as exc:
        assert "val_rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sample_and_split_can_explicitly_replace_incomplete_published_split():
    df = pd.DataFrame(
        {"text": [f"row{i}" for i in range(30)], "split": ["test"] * 30}
    )
    preserved = sample_and_split_neutrals(
        df, test_rows=10, val_rows=5, train_rows=8, seed=0
    )
    assert set(preserved["split"]) == {"test"}

    resplit = sample_and_split_neutrals(
        df,
        test_rows=10,
        val_rows=5,
        train_rows=8,
        seed=0,
        respect_existing_splits=False,
    )
    assert resplit["split"].value_counts().to_dict() == {
        "test": 10,
        "train": 8,
        "validation": 5,
    }

    small = sample_and_split_neutrals(
        df.head(10),
        test_rows=10,
        val_rows=5,
        train_rows=8,
        seed=0,
        respect_existing_splits=False,
    )
    counts = small["split"].value_counts().to_dict()
    assert sum(counts.values()) == 10
    assert counts["train"] > 0
    assert counts["validation"] > 0
    assert counts["test"] > 0


def test_split_kwargs_from_cfg_reads_neutral_block():
    from neutral_protocol import split_kwargs_from_cfg

    kw = split_kwargs_from_cfg(
        {"neutral": {"max_rows": 100, "val_rows": 20, "train_rows": 50}}
    )
    assert kw == {"test_rows": 100, "val_rows": 20, "train_rows": 50}


def test_split_kwargs_from_cfg_requires_all_keys():
    from neutral_protocol import split_kwargs_from_cfg

    try:
        split_kwargs_from_cfg({"neutral": {"max_rows": 100}})
    except ValueError as exc:
        assert "val_rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_split_kwargs_smoke_uses_yaml_overlay():
    from neutral_protocol import split_kwargs_from_cfg

    cfg = {
        "neutral": {"max_rows": 1000, "val_rows": 300, "train_rows": 1000},
        "smoke": {"neutral": {"max_rows": 80, "val_rows": 40, "train_rows": 80}},
    }
    kw = split_kwargs_from_cfg(cfg, smoke=True)
    assert kw == {"test_rows": 80, "val_rows": 40, "train_rows": 80}


def test_defaults_yaml_declares_neutral_split_counts():
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "defaults.yaml").read_text(
            encoding="utf-8"
        )
    )
    neu = raw["neutral"]
    assert neu["max_rows"] == 1000
    assert neu["val_rows"] == 300
    assert neu["train_rows"] == 1000
    smoke = raw["smoke"]["neutral"]
    assert smoke["max_rows"] == 80
    assert smoke["val_rows"] == 40
    assert smoke["train_rows"] == 80


def test_texts_for_split_uses_split_column():
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "split": ["train", "validation", "test"],
        }
    )
    assert texts_for_split(df, "validation") == ["b"]
    assert texts_for_split(df, "test") == ["c"]


class _Tok:
    def encode(self, s, add_special_tokens=False):
        table = {"he": [11], " she": [12], "she": [12], " him": [13]}
        return table.get(s, [99])


def test_excluded_token_ids_include_space_prefix():
    ids = excluded_token_id_set(_Tok(), ["he", "she"])
    assert 11 in ids and 12 in ids


def test_nonpad_nonexcluded_drops_class_token_ids():
    act = torch.arange(18, dtype=torch.float32).view(1, 6, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    ids = torch.tensor([[1, 11, 2, 12, 3, 0]])
    from caa_eval import nonpad_token_selector

    sel = make_nonpad_nonexcluded_selector(
        tokenizer=_Tok(),
        excluded_words=["he", "she"],
        fallback_selector=nonpad_token_selector,
    )
    out = sel(act, {"attention_mask": mask, "input_ids": ids})
    # tokens 1,2,3 kept (ids 1,2,3); 11 and 12 dropped
    assert out.shape == (3, 3)
    assert torch.equal(out[0], act[0, 0])
    assert torch.equal(out[1], act[0, 2])
    assert torch.equal(out[2], act[0, 4])


def test_youden_threshold_from_val_applied_to_test():
    val = class_vs_neutral_metrics(
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        target_class="M",
        scores_other=[0.0, 0.0],
        other_class="F",
        n_bootstrap=0,
    )
    tau = val["youden_threshold"]
    assert tau is not None
    test = class_vs_neutral_metrics(
        [2.0, 2.0],
        [1.2, 1.2, 1.2],
        target_class="M",
        scores_other=[1.2],
        other_class="F",
        n_bootstrap=0,
        youden_threshold=tau,
    )
    same = class_vs_neutral_metrics(
        [2.0, 2.0],
        [1.2, 1.2, 1.2],
        target_class="M",
        scores_other=[1.2],
        other_class="F",
        n_bootstrap=0,
    )
    assert test["youden_threshold_source"] == "provided"
    assert same["youden_threshold_source"] == "eval"
    assert test["youden_threshold"] == tau
    # The pooled ``youden_threshold`` is a diagnostic only: it must NOT drive
    # Spec_n (that used to be the stale assertion here), so supplying it alone
    # leaves specificity a test-fit number.
    assert test["specificity"] == same["specificity"]

    # Spec_n / Excl are frozen through the neutral/per-rival rules (threshold AND
    # orientation sign). Val-fitted neutral τ (~1.0) sits below 1.2, so the test
    # neutrals are false positives under the frozen rule, while the test-fit
    # (oracle) refit moves the cut above 1.2 and reports perfect specificity.
    from sae_eval import frozen_decision_rules

    held_out = class_vs_neutral_metrics(
        [2.0, 2.0],
        [1.2, 1.2, 1.2],
        target_class="M",
        scores_other=[1.2],
        other_class="F",
        n_bootstrap=0,
        **frozen_decision_rules(val),
    )
    assert held_out["neutral_youden_threshold_source"] == "provided"
    assert held_out["specificity"] < same["specificity"]


def test_infer_encoding_direction_uses_provided_means_not_chance_flip():
    df = pd.DataFrame(
        {
            "encoded": [-0.1, -0.2, 0.1, 0.2],
            "label_class": ["M", "M", "F", "F"],
        }
    )
    dirs = infer_class_encoding_directions(df, ["M", "F"])
    assert dirs["M"] == -1.0
    assert dirs["F"] == 1.0
    forced = infer_class_encoding_directions(
        df, ["M", "F"], encoding_direction={"M": 1.0, "F": -1.0}
    )
    assert forced == {"M": 1.0, "F": -1.0}


def test_polarity_from_val_not_test_means():
    val = pd.DataFrame(
        {
            "encoded": [-1.0, -1.0, 1.0, 1.0],
            "label_class": ["M", "M", "F", "F"],
            "split": ["validation"] * 4,
        }
    )
    test = pd.DataFrame(
        {
            "encoded": [0.05, 0.04, -0.05, -0.04],
            "label_class": ["M", "M", "F", "F"],
            "split": ["test"] * 4,
        }
    )
    neu = [0.0, 0.0, 0.0]
    leaked = encoder_class_vs_neutral_metrics(
        test,
        target_class="M",
        target_classes=["M", "F"],
        overlay_neutral_scores=neu,
        polarity_df=test,
        n_bootstrap=0,
        split="test",
    )
    val_fit = encoder_class_vs_neutral_metrics(
        test,
        target_class="M",
        target_classes=["M", "F"],
        overlay_neutral_scores=neu,
        polarity_df=val,
        n_bootstrap=0,
        split="test",
    )
    no_fit = encoder_class_vs_neutral_metrics(
        test,
        target_class="M",
        target_classes=["M", "F"],
        overlay_neutral_scores=neu,
        polarity_df=None,
        n_bootstrap=0,
        split="test",
    )
    trainer = encoder_class_vs_neutral_metrics(
        test,
        target_class="M",
        target_classes=["M", "F"],
        overlay_neutral_scores=neu,
        encoding_direction={"M": -1.0, "F": 1.0},
        polarity_df=None,
        n_bootstrap=0,
        split="test",
    )
    assert leaked["encoding_direction"] == 1.0
    assert val_fit["encoding_direction"] == -1.0
    assert no_fit["encoding_direction"] == 1.0
    assert trainer["encoding_direction"] == -1.0
    assert leaked["mean_target"] > 0
    assert val_fit["mean_target"] < 0
    assert trainer["mean_target"] < 0


def test_per_class_metrics_use_own_pair_not_full_task_class_list():
    """A 3-class task's pair-trained (two-pole) encoder only knows 2 of the
    3 classes. Evaluating it against the full task class list (done so the
    off-pair "does this leak?" class can be checked too, see train.py's
    ``_fair_encoder_eval`` "Fair metrics need the full task class set"
    comment) used to resolve ``class_a``/``class_b`` positionally from that
    full list instead of from the encoder's own trained poles -- silently
    dropping whichever real class wasn't first/second in the task's
    ``classes:`` yaml order, and fabricating a spurious result for whatever
    off-pair class happened to land in that position instead. Reproduces
    the RAVEL country bug (config.classes = [united_states, china, russia];
    a china-russia pair encoder always errored "no rows for class 'russia'"
    while a phantom "united_states" result appeared) using a numeric
    +1/-1/0 label column, which is what actually triggered it.
    """
    n = 20
    rng = np.random.default_rng(0)
    china_scores = rng.normal(0.9, 0.05, n)
    russia_scores = rng.normal(-0.9, 0.05, n)
    neutral_scores = rng.normal(0.0, 0.05, n)

    train = pd.DataFrame(
        {
            "label": ["1.0"] * n + ["-1.0"] * n,
            "encoded": np.concatenate([china_scores, russia_scores]),
            "split": ["test"] * (2 * n),
            "type": ["training"] * (2 * n),
        }
    )
    neutral = pd.DataFrame(
        {
            "label": ["0.0"] * n,
            "encoded": neutral_scores,
            "split": ["test"] * n,
            "type": ["neutral_dataset"] * n,
        }
    )
    df = pd.concat([train, neutral], ignore_index=True)

    # This trainer's own poles (china-russia pair); yaml classes order puts
    # the off-pair class ("united_states") first, "china" second.
    result = encoder_per_class_metrics(
        df,
        ["united_states", "china", "russia"],
        full_encoder_df=df,
        encoding_direction={"china": 1.0, "russia": -1.0},
        split="test",
        n_bootstrap=0,
    )

    assert result["china"].get("error") is None
    assert result["russia"].get("error") is None, result["russia"]
    assert result["china"]["roc_auc_neutral"] > 0.9
    assert result["russia"]["roc_auc_neutral"] > 0.9
    # The off-pair class has no real rows in this pair's own data --
    # must report that honestly rather than fabricate a chance-level result.
    assert result["united_states"].get("error") == "no rows for class 'united_states'"
