"""Gender EN data from Hugging Face ajibawa datasets only.

Training / eval: ``aieng-lab/genter-ajibawa-name-filled`` (config ``n1``)
Neutrals: ``aieng-lab/biasneutral-ajibawa``

See the dataset README for the canonical loader::

    from datasets import load_dataset
    ds = load_dataset("aieng-lab/genter-ajibawa-name-filled", "n1")
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

GENTER_AJIBAWA = "aieng-lab/genter-ajibawa-name-filled"
GENTER_AJIBAWA_SUBSET = "n1"
BIASNEUTRAL_AJIBAWA = "aieng-lab/biasneutral-ajibawa"

PRONOUN_HE_SHE = {"M": "he", "F": "she"}
PRONOUN_HER_HIS = {"M": "his", "F": "her"}


def _normalize_split(value) -> str:
    s = str(value).strip().lower()
    if s in {"val", "validation"}:
        return "validation"
    if s in {"train", "test"}:
        return s
    return "train"


def geneutral_split_kwargs(*, smoke: bool = False) -> dict:
    """``neutral.*`` / ``smoke.neutral`` counts from ``configs/defaults.yaml``."""
    import yaml
    from pathlib import Path

    from neutral_protocol import split_kwargs_from_cfg

    path = Path(__file__).resolve().parent / "configs" / "defaults.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return split_kwargs_from_cfg(raw, smoke=smoke)


def read_geneutral(
    max_size: Optional[int] = None,
    path: Optional[str] = None,
    *,
    val_rows: Optional[int] = None,
    train_rows: Optional[int] = None,
    test_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load ``aieng-lab/biasneutral-ajibawa`` (``path`` ignored; kept for call-site compat)."""
    del path
    from study.tasks import load_neutral

    hf_id = os.environ.get("GENDER_NEUTRAL_HF", BIASNEUTRAL_AJIBAWA)
    if "biasneutral" in hf_id and "ajibawa" not in hf_id:
        raise ValueError(
            f"Refusing plain biasneutral id {hf_id!r}; use {BIASNEUTRAL_AJIBAWA!r}"
        )
    return load_neutral(
        hf_id=hf_id,
        max_rows=max_size,
        test_rows=test_rows,
        val_rows=val_rows,
        train_rows=train_rows,
    )


def load_ajibawa_name_filled(
    *,
    hf_dataset: str = GENTER_AJIBAWA,
    hf_subset: str = GENTER_AJIBAWA_SUBSET,
    max_rows_per_split: Optional[int] = None,
    pronoun_map: Optional[dict[str, str]] = None,
    shuffle: bool = True,
    shuffle_seed: int = 0,
) -> pd.DataFrame:
    """Load name-filled GENTER via ``load_dataset(repo, config)``."""
    from study.tasks import load_hf_splits

    if "genter" in hf_dataset and "ajibawa" not in hf_dataset:
        raise ValueError(
            f"Refusing plain genter id {hf_dataset!r}; use {GENTER_AJIBAWA!r}"
        )
    pronoun_map = pronoun_map or PRONOUN_HE_SHE
    subset = (hf_subset or GENTER_AJIBAWA_SUBSET).strip() or GENTER_AJIBAWA_SUBSET
    raw = load_hf_splits(
        hf_dataset,
        config_name=subset,
        max_rows_per_split=max_rows_per_split,
    )
    if shuffle and len(raw) > 1:
        raw = raw.sample(frac=1.0, random_state=int(shuffle_seed)).reset_index(drop=True)

    rows = []
    for _, row in raw.iterrows():
        label_class = str(row.get("label", row.get("label_class", ""))).strip().upper()
        if label_class not in ("M", "F"):
            continue
        masked = str(row.get("masked") or "")
        name = str(row.get("name", "")).strip()
        if "[NAME]" in masked:
            if not name:
                continue
            masked = masked.replace("[NAME]", name)
        if "[MASK]" not in masked:
            continue
        label_token = str(row.get("pronoun") or pronoun_map[label_class])
        if pronoun_map is not PRONOUN_HE_SHE:
            label_token = pronoun_map[label_class]
        rows.append(
            {
                "masked": masked,
                "label": label_token,
                "label_class": label_class,
                "split": _normalize_split(row.get("split", "train")),
                "pair_id": row.get("template_id", row.get("id")),
            }
        )
    if not rows:
        raise RuntimeError(f"No usable gender rows from {hf_dataset} subset={subset!r}")
    return pd.DataFrame(rows)


def build_her_his_proxy(
    *,
    max_rows_per_split: Optional[int] = None,
    hf_dataset: str = GENTER_AJIBAWA,
    hf_subset: str = GENTER_AJIBAWA_SUBSET,
) -> pd.DataFrame:
    return load_ajibawa_name_filled(
        hf_dataset=hf_dataset,
        hf_subset=hf_subset,
        max_rows_per_split=max_rows_per_split,
        pronoun_map=PRONOUN_HER_HIS,
    )


