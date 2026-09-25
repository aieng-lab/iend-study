"""Localization comparison stage."""

from __future__ import annotations

from typing import Any, Dict, List

from localization_eval import run_localization_analysis
from results_schema import method_result
from study.config import StudyConfig
from study.tasks import TaskBundle, encode_target_classes, labeled_df_for_eval


def run_localization_stage(
    cfg: StudyConfig,
    bundle: TaskBundle,
    train_raw_none: Dict[str, Dict[str, Any]],
    sae_raw: Dict[str, Any] | None,
    *,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    # SAE selections only exist for factual label_class poles on one-pole tasks.
    classes = encode_target_classes(
        bundle, labeled_df_for_eval(bundle, expand_one_pole=False)
    )
    method_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        raw = run_localization_analysis(
            train_raw_none,
            sae_raw,
            output_dir=cfg.output_dir,
            target_classes=classes,
        )
        method_rows.append(
            method_result(
                method="localization",
                model=cfg.model_key,
                task=cfg.task_id,
                metrics={
                    "n_classes": len(classes),
                    "has_gradiend": "gradiend" in train_raw_none,
                    "has_actiend": "actiend" in train_raw_none,
                    "has_sae": bool(sae_raw) and not (sae_raw or {}).get("error"),
                },
                extras={"summary_keys": list(raw.keys()) if isinstance(raw, dict) else []},
                artifacts={"localization_dir": str(cfg.output_dir / "localization")},
            )
        )
    except Exception as exc:
        if fail_fast:
            raise
        errors.append(f"localization: {exc}")
        raw = {"error": str(exc)}
        method_rows.append(
            method_result(
                method="localization",
                model=cfg.model_key,
                task=cfg.task_id,
                status="error",
                error=str(exc),
            )
        )
    return {"methods": method_rows, "raw": raw, "errors": errors}
