"""Emotion/polarity task backed by frozen aieng-lab Hub datasets."""

from __future__ import annotations

from typing import Any, Dict, List

from study.tasks import TaskBundle, load_hf_splits, load_neutral_from_cfg, per_class_from_merged

DEFAULT_DATASET = "aieng-lab/en-sentiment-nrc"
DEFAULT_CONFIG = "split"
DEFAULT_NEUTRAL = "aieng-lab/en-sentiment-nrc-neutral"


def build_emotion(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    hf_id = data_cfg.get("hf_dataset", DEFAULT_DATASET)
    hf_config = data_cfg.get("hf_config", DEFAULT_CONFIG)

    df = load_hf_splits(hf_id, config_name=hf_config)
    required = {"masked", "label", "label_class", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Emotion dataset {hf_id!r} missing columns: {sorted(missing)}")

    classes = list(task.get("classes") or ["positive", "negative"])
    if smoke:
        df = df.groupby(["split", "label_class"], group_keys=False).head(40)

    bad = df["masked"].astype(str).str.match(r"^\s*\[MASK\]")
    if bad.any():
        print(f"Warning: dropping {int(bad.sum())} emotion rows with leading [MASK]")
        df = df.loc[~bad].copy()

    data_per_class = per_class_from_merged(df, classes=classes)
    neutral_hf = data_cfg.get("neutral_hf", DEFAULT_NEUTRAL)
    # The published emotion-neutral corpus currently exposes one source split.
    # It is a neutral control pool, so make the study's disjoint train/val/test
    # partition explicitly instead of preserving a split that leaves identity
    # training without rows.
    neutrals = load_neutral_from_cfg(
        neutral_hf,
        cfg,
        smoke=smoke,
        respect_existing_splits=False,
    )

    excluded: List[str] = sorted(
        df["label"].dropna().astype(str).str.lower().unique().tolist()
    )
    return TaskBundle(
        task_id="emotion",
        classes=classes,
        data_per_class=data_per_class,
        merged_df=df,
        neutrals=neutrals,
        eval_spec={"causal_score": "class_token_logit"},
        excluded_words=excluded,
        notes=f"{hf_id} config={hf_config} + {neutral_hf}",
    )
