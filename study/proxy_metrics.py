"""Proxy transfer metrics for gender_en (he/she → her/his)."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd




def transfer_summary(
    *,
    in_domain: Dict[str, Any],
    transfer: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare in-domain (he/she) vs transfer (her/his) scalar metrics."""
    out: Dict[str, Any] = {"proxy_in_domain": "he_she", "proxy_transfer": "her_his"}
    for key in ("roc_auc", "balanced_accuracy", "cohens_d"):
        a = in_domain.get(key)
        b = transfer.get(key)
        out[f"{key}_he_she"] = a
        out[f"{key}_her_his"] = b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[f"{key}_delta"] = float(b) - float(a)
    return out


def bank_stats(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n": int(len(df)),
        "labels": sorted(df["label"].astype(str).unique().tolist()) if "label" in df.columns else [],
        "classes": sorted(df["label_class"].astype(str).unique().tolist())
        if "label_class" in df.columns
        else [],
    }
