"""Race / religion / pronoun / emotion HF adapters."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from study.tasks import (
    TaskBundle,
    assign_splits,
    load_hf_splits,
    load_neutral_from_cfg,
    per_class_from_merged,
)

DEFAULT_PRONOUN_DATASET = "aieng-lab/en-pronouns"
DEFAULT_PRONOUN_NEUTRAL = "aieng-lab/en-pronoun-neutral"


def _load_hf_per_class(hf_id: str, classes: List[str]) -> Dict[str, pd.DataFrame]:
    from gradiend.trainer.core.unified_data import load_hf_per_class

    return load_hf_per_class(hf_id, classes=classes)


def _merged_from_per_class(data: Dict[str, pd.DataFrame], classes: List[str]) -> pd.DataFrame:
    rows = []
    for cls in classes:
        df = data[cls].copy()
        label_col = cls if cls in df.columns else ("label" if "label" in df.columns else None)
        if label_col is None:
            raise ValueError(f"Class frame {cls} missing token column")
        for _, r in df.iterrows():
            rows.append(
                {
                    "masked": r["masked"],
                    "split": r.get("split", "train"),
                    "label": r[label_col],
                    "label_class": cls,
                }
            )
    return assign_splits(pd.DataFrame(rows))


def build_race(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    task_id = str(cfg.get("task_id") or task.get("id") or "race")
    classes = list(task.get("classes") or ["asian", "black", "white"])
    data_cfg = task.get("data") or {}
    hf = data_cfg.get("hf_dataset", "aieng-lab/gradiend_race_data")
    data_per_class = _load_hf_per_class(hf, classes)
    if smoke:
        # Group by split before capping — .head(64) alone (pre-2026-08-18)
        # grabbed only 'train' rows whenever a class's frame was split-blocked,
        # leaving do_eval=True with no validation/test split at all. Confirmed
        # on race/race_one_pole 2026-08-18.
        data_per_class = {
            k: (
                v.groupby("split", group_keys=False).head(20)
                if "split" in v.columns
                else v.head(64)
            )
            for k, v in data_per_class.items()
        }
    merged = _merged_from_per_class(data_per_class, classes)
    neutrals = load_neutral_from_cfg(
        data_cfg.get("neutral_hf", "aieng-lab/biasneutral-ajibawa"),
        cfg,
        smoke=smoke,
    )
    return TaskBundle(
        task_id=task_id,
        classes=classes,
        data_per_class=data_per_class,
        merged_df=merged,
        neutrals=neutrals,
        eval_spec={"causal_score": "class_token_logit", "multiclass": True},
        excluded_words=classes,
    )


def build_religion(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    task_id = str(cfg.get("task_id") or task.get("id") or "religion")
    classes = list(task.get("classes") or ["christian", "muslim", "jewish"])
    data_cfg = task.get("data") or {}
    hf = data_cfg.get("hf_dataset", "aieng-lab/gradiend_religion_data")
    data_per_class = _load_hf_per_class(hf, classes)
    if smoke:
        data_per_class = {
            k: (
                v.groupby("split", group_keys=False).head(20)
                if "split" in v.columns
                else v.head(64)
            )
            for k, v in data_per_class.items()
        }
    merged = _merged_from_per_class(data_per_class, classes)
    neutrals = load_neutral_from_cfg(
        data_cfg.get("neutral_hf", "aieng-lab/biasneutral-ajibawa"),
        cfg,
        smoke=smoke,
    )
    return TaskBundle(
        task_id=task_id,
        classes=classes,
        data_per_class=data_per_class,
        merged_df=merged,
        neutrals=neutrals,
        eval_spec={"causal_score": "class_token_logit", "multiclass": True},
        excluded_words=classes,
    )


def _load_pronoun_data(data_cfg: Dict[str, Any], *, smoke: bool) -> pd.DataFrame:
    hf_id = data_cfg.get("hf_dataset", DEFAULT_PRONOUN_DATASET)
    df = load_hf_splits(hf_id)
    required = {"masked", "label", "label_class", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pronoun dataset {hf_id!r} missing columns: {sorted(missing)}")
    max_rows = 40 if smoke else data_cfg.get("max_rows_per_split")
    if max_rows is not None:
        # Hub rows are class-blocked; sample after grouping so every class and
        # every published split remains represented.
        df = df.groupby(["split", "label_class"], group_keys=False).head(int(max_rows))
    return df


def build_pronoun_person(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    df = _load_pronoun_data(data_cfg, smoke=smoke)
    df = assign_splits(df)
    classes = list(task.get("classes") or ["1", "2", "3"])
    merge = dict(task.get("class_merge_map") or {})
    neutral_hf = data_cfg.get("neutral_hf", DEFAULT_PRONOUN_NEUTRAL)
    neutrals = load_neutral_from_cfg(neutral_hf, cfg, smoke=smoke)
    # Keep raw-class frames; trainer + labeled_df_for_eval apply class_merge_map.
    raw_classes = sorted(df["label_class"].astype(str).unique())
    data_per_class = per_class_from_merged(df, classes=raw_classes)
    return TaskBundle(
        task_id="pronoun_person",
        classes=classes,
        data_per_class=data_per_class,
        merged_df=df,
        neutrals=neutrals,
        class_merge_map=merge,
        eval_spec={"causal_score": "class_token_logit", "multiclass": True},
        excluded_words=["i", "we", "you", "he", "she", "it", "they"],
        notes=f"{data_cfg.get('hf_dataset', DEFAULT_PRONOUN_DATASET)} + {neutral_hf}",
    )


def build_pronoun_number(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    df = _load_pronoun_data(data_cfg, smoke=smoke)
    df = assign_splits(df)
    classes = list(task.get("classes") or ["singular", "plural"])
    merge = dict(task.get("class_merge_map") or {})
    groups = task.get("class_merge_transition_groups")
    neutral_hf = data_cfg.get("neutral_hf", DEFAULT_PRONOUN_NEUTRAL)
    neutrals = load_neutral_from_cfg(neutral_hf, cfg, smoke=smoke)
    raw_classes = sorted(df["label_class"].astype(str).unique())
    data_per_class = per_class_from_merged(df, classes=raw_classes)
    return TaskBundle(
        task_id="pronoun_number",
        classes=classes,
        data_per_class=data_per_class,
        merged_df=df,
        neutrals=neutrals,
        class_merge_map=merge,
        class_merge_transition_groups=groups,
        eval_spec={"causal_score": "class_token_logit"},
        excluded_words=["i", "we", "you", "he", "she", "it", "they"],
        notes=f"{data_cfg.get('hf_dataset', DEFAULT_PRONOUN_DATASET)} + {neutral_hf}",
    )
