"""Gender EN task: ``genter-ajibawa-name-filled`` + her/his proxy banks."""

from __future__ import annotations

from typing import Any, Dict

from gender_en_data import (
    BIASNEUTRAL_AJIBAWA,
    GENTER_AJIBAWA,
    GENTER_AJIBAWA_SUBSET,
    build_her_his_proxy,
    load_ajibawa_name_filled,
)
from study.tasks import TaskBundle, per_class_from_merged


def build_task(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    hf = data_cfg.get("hf_dataset", GENTER_AJIBAWA)
    hf_subset = data_cfg.get("hf_subset", GENTER_AJIBAWA_SUBSET)
    neutral_hf = data_cfg.get("neutral_hf") or (cfg.get("neutral") or {}).get(
        "hf_dataset", BIASNEUTRAL_AJIBAWA
    )
    max_rows = 200 if smoke else data_cfg.get("max_rows_per_split")
    if smoke and max_rows is None:
        max_rows = 200

    merged = load_ajibawa_name_filled(
        hf_dataset=hf, hf_subset=hf_subset, max_rows_per_split=max_rows
    )
    classes = list(task.get("classes") or ["M", "F"])
    data_per_class = per_class_from_merged(merged, classes=classes)

    proxies = {"he_she": merged}
    proxy_ids = list(task.get("eval_proxies") or ["he_she", "her_his"])
    if "her_his" in proxy_ids:
        proxies["her_his"] = build_her_his_proxy(
            max_rows_per_split=max_rows,
            hf_dataset=hf,
            hf_subset=hf_subset,
        )

    from study.tasks import load_neutral_from_cfg

    neutrals = load_neutral_from_cfg(neutral_hf, cfg, smoke=smoke)

    task_id = str(task.get("id") or cfg.get("task_id") or "gender_en")
    return TaskBundle(
        task_id=task_id,
        classes=classes,
        data_per_class=data_per_class,
        merged_df=merged,
        neutrals=neutrals,
        proxies=proxies,
        eval_spec={
            "causal_score": (task.get("causal") or {}).get("score", "he_she_margin"),
            "target_tokens": (task.get("causal") or {}).get("target_tokens"),
            "eval_proxies": proxy_ids,
        },
        excluded_words=["he", "she", "him", "her", "his", "hers"],
        notes="genter-ajibawa-name-filled + biasneutral-ajibawa; proxies he_she + her_his",
    )
