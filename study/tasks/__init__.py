"""Task adapter protocol and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TaskBundle:
    """Standardized task payload for the study runner."""

    task_id: str
    classes: List[str]
    data_per_class: Dict[str, pd.DataFrame]
    merged_df: pd.DataFrame
    neutrals: pd.DataFrame
    eval_spec: Dict[str, Any] = field(default_factory=dict)
    proxies: Dict[str, pd.DataFrame] = field(default_factory=dict)
    excluded_words: List[str] = field(default_factory=list)
    class_merge_map: Optional[Dict[str, List[str]]] = None
    class_merge_transition_groups: Optional[List[List[str]]] = None
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


DEFAULT_MASK_PLACEHOLDER = "[MASK]"


def resolve_mask_placeholder(
    bundle: TaskBundle,
    cfg: Optional[Mapping[str, Any]] = None,
) -> str:
    """Dataset prediction-slot marker for ``TextPredictionConfig.mask_placeholder``.

    Prior HF tasks (religion, race, pronoun, gender) use ``[MASK]`` in ``masked``
    templates. Gender name-filled data may still contain ``[PRONOUN]`` in raw hub
    rows, but loaders normalize to ``[MASK]`` before training. Only set a
    non-default placeholder in task YAML or ``TaskBundle.training_overrides``.
    """
    cfg = cfg or {}
    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    for source in (
        bundle.training_overrides.get("mask_placeholder"),
        data_cfg.get("mask_placeholder"),
        task.get("mask_placeholder"),
    ):
        if source is not None and str(source).strip():
            return str(source).strip()
    return DEFAULT_MASK_PLACEHOLDER


def per_class_from_merged(
    df: pd.DataFrame,
    *,
    classes: Sequence[str],
    label_class_col: str = "label_class",
    label_col: str = "label",
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for cls in classes:
        sub = df[df[label_class_col].astype(str) == str(cls)].copy()
        if sub.empty:
            raise ValueError(f"No rows for class {cls!r}")
        frame = sub[["masked", "split", label_col]].copy()
        frame = frame.rename(columns={label_col: str(cls)})
        out[str(cls)] = frame
    return out


def subset_task_bundle(
    bundle: TaskBundle,
    *,
    column: str,
    value: Any,
) -> TaskBundle:
    """Return a task bundle restricted by one merged-data metadata value.

    This is intentionally task-neutral.  It supports confidence sweeps and
    other controlled strata without placing theory metrics in task adapters.
    One-pole alternative classes are reconstructed from ``alternative_*``.
    """
    if column not in bundle.merged_df.columns:
        raise KeyError(
            f"Cannot subset task {bundle.task_id!r}: missing column {column!r}; "
            f"available={sorted(map(str, bundle.merged_df.columns))}"
        )
    wanted = str(value)
    merged = bundle.merged_df[
        bundle.merged_df[column].astype(str) == wanted
    ].copy().reset_index(drop=True)
    if merged.empty:
        available = sorted(set(bundle.merged_df[column].astype(str)))
        raise ValueError(
            f"Task subset {column}={wanted!r} is empty; available={available}"
        )

    data_per_class: Dict[str, pd.DataFrame] = {}
    for cls in map(str, bundle.classes):
        factual = merged[merged["label_class"].astype(str) == cls]
        if not factual.empty:
            data_per_class[cls] = factual[["masked", "split", "label"]].rename(
                columns={"label": cls}
            )
            continue
        if {"alternative", "alternative_class"}.issubset(merged.columns):
            alternative = merged[merged["alternative_class"].astype(str) == cls]
            if not alternative.empty:
                data_per_class[cls] = alternative[
                    ["masked", "split", "alternative"]
                ].rename(columns={"alternative": cls})

    missing = [str(cls) for cls in bundle.classes if str(cls) not in data_per_class]
    if missing:
        raise ValueError(
            f"Task subset {column}={wanted!r} lacks classes {missing}"
        )
    return TaskBundle(
        task_id=bundle.task_id,
        classes=list(bundle.classes),
        data_per_class=data_per_class,
        merged_df=merged,
        neutrals=bundle.neutrals,
        eval_spec=dict(bundle.eval_spec),
        proxies=dict(bundle.proxies),
        excluded_words=list(bundle.excluded_words),
        class_merge_map=bundle.class_merge_map,
        class_merge_transition_groups=bundle.class_merge_transition_groups,
        training_overrides=dict(bundle.training_overrides),
        notes=f"{bundle.notes}; subset {column}={wanted}",
    )


def labeled_df_for_eval(
    bundle: TaskBundle,
    *,
    expand_one_pole: bool = False,
) -> pd.DataFrame:
    """Return ``merged_df`` ready for SAE / CAA / causal encode eval.

    Pronoun tasks keep raw HF ids (``1SG``, ``3PL``, …) on ``merged_df`` and
    only apply ``class_merge_map`` inside the trainer. Eval stages must remap
    here. Unmapped raw ids (e.g. ``2SGPL`` on number) are dropped.

    Circuit one-pole frames (IOI / induction / …) store the CF only on
    ``alternative`` / ``alternative_class``. Set ``expand_one_pole=True`` for
    CAA (filled-token policies need both poles). SAE should keep
    ``expand_one_pole=False`` and restrict ``TARGET_CLASSES`` to factual
    ``label_class`` values — same masked left-context would make duplicated
    CF rows identical under SAE hooks.
    """
    df = bundle.merged_df
    merge_map = bundle.class_merge_map
    if merge_map:
        base_to_merged: Dict[str, str] = {}
        for merged, bases in merge_map.items():
            for base in bases:
                key = str(base)
                if key in base_to_merged and base_to_merged[key] != str(merged):
                    raise ValueError(
                        f"class_merge_map: base class {key!r} appears in multiple "
                        f"merged classes ({base_to_merged[key]!r} and {merged!r})"
                    )
                base_to_merged[key] = str(merged)

        out = df.copy()
        label_col = "label_class"
        if label_col not in out.columns:
            raise ValueError("merged_df missing label_class for class_merge_map eval remap")
        if "raw_label_class" not in out.columns:
            out["raw_label_class"] = out[label_col].astype(str)
        out[label_col] = out[label_col].astype(str).map(lambda x: base_to_merged.get(x, x))

        alt_col = "alternative_class"
        if alt_col in out.columns:
            if "raw_alternative_class" not in out.columns:
                out["raw_alternative_class"] = out[alt_col].astype(str)
            out[alt_col] = out[alt_col].astype(str).map(lambda x: base_to_merged.get(x, x))

        targets = {str(c) for c in bundle.classes}
        out = out[out[label_col].isin(targets)].copy()
        if out.empty:
            raise ValueError(
                f"class_merge_map remapped to empty frame for classes={sorted(targets)}; "
                f"raw ids present={sorted(set(df[label_col].astype(str)))}"
            )
        missing = targets - set(out[label_col].astype(str))
        if missing:
            raise ValueError(
                f"After class_merge_map remap, missing classes {sorted(missing)}; "
                f"present={sorted(set(out[label_col].astype(str)))}"
            )
        df = out

    if expand_one_pole:
        df = expand_one_pole_cf_rows(df, classes=bundle.classes)
    return df


def expand_one_pole_cf_rows(
    df: pd.DataFrame,
    *,
    classes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Duplicate factual rows as CF-labeled rows (``alternative`` → ``label``).

    Used when ``label_class`` only has the factual pole (e.g. MATCH) but
    ``bundle.classes`` also lists the CF (DISTRACTOR). CAA prediction fill
    then sees both tokens; SAE should not use this (identical left-context).
    """
    need = {"masked", "label", "label_class", "alternative", "alternative_class"}
    if df is None or getattr(df, "empty", True) or not need.issubset(df.columns):
        return df
    present = {str(c) for c in df["label_class"].astype(str).unique()}
    alt_present = {str(c) for c in df["alternative_class"].astype(str).unique()}
    wanted = {str(c) for c in (classes or [])} if classes else (present | alt_present)
    missing = wanted - present
    if not (missing & alt_present):
        return df

    fac = df.copy()
    cf = df.copy()
    cf["label"] = cf["alternative"]
    cf["label_class"] = cf["alternative_class"].astype(str)
    # Swap rival token/class so row-wise decoder eval sees distinct factual vs alt.
    cf["alternative"] = df["label"].astype(str)
    cf["alternative_class"] = df["label_class"].astype(str)
    if "pair_id" in cf.columns:
        cf["pair_id"] = cf["pair_id"].astype(str) + "__cf"
    else:
        cf["pair_id"] = [f"cf_{i}" for i in range(len(cf))]
    cf["one_pole_cf_row"] = True
    fac["one_pole_cf_row"] = False
    out = pd.concat([fac, cf], ignore_index=True)
    if wanted:
        out = out[out["label_class"].astype(str).isin(wanted)].copy()
    return out.reset_index(drop=True)


def encode_target_classes(
    bundle: TaskBundle,
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Classes present on ``label_class`` for encode stages (order = bundle.classes)."""
    frame = bundle.merged_df if df is None else df
    if frame is None or getattr(frame, "empty", True) or "label_class" not in frame.columns:
        return [str(c) for c in bundle.classes]
    present = {str(c) for c in frame["label_class"].astype(str).unique()}
    ordered = [str(c) for c in bundle.classes if str(c) in present]
    if ordered:
        return ordered
    return sorted(present)


def claim_classes_for_study(
    bundle: TaskBundle,
    *,
    one_pole: bool = False,
    one_pole_classes: Optional[Sequence[str]] = None,
) -> List[str]:
    """Classes that belong in primary / fair claim tables.

    One-pole circuit tasks train and read out factual poles only
    (``training.one_pole_classes``); CF partners stay on ``alternative_*`` and
    must not reserve empty bipolar fair IDs (e.g. ``sae:DISTRACTOR``).
    """
    if one_pole:
        if one_pole_classes:
            wanted = [str(c) for c in one_pole_classes if str(c).strip()]
            allowed = {str(c) for c in bundle.classes}
            ordered = [c for c in wanted if c in allowed]
            if ordered:
                return ordered
            if wanted:
                return wanted
        return encode_target_classes(
            bundle, labeled_df_for_eval(bundle, expand_one_pole=False)
        )
    return [str(c) for c in bundle.classes]


def load_neutral(
    hf_id: str = "aieng-lab/biasneutral-ajibawa",
    *,
    max_rows: Optional[int] = None,
    test_rows: Optional[int] = None,
    val_rows: Optional[int] = None,
    train_rows: Optional[int] = None,
    seed: int = 0,
    respect_existing_splits: bool = True,
) -> pd.DataFrame:
    """Load neutrals; sample disjoint train/val/test counts (not file head).

    ``max_rows`` aliases ``test_rows`` (report pool). Validation and train are
    extra sampled rows — they are not carved out of the test pool.

    Tries the checked-in ``data/neutral/biasneutral-ajibawa.csv`` snapshot
    first (fast, no network), but ONLY when it actually has enough rows for
    this call's ``test_rows+val_rows+train_rows`` -- otherwise falls back to
    a live Hub download. The naive "use it whenever present" version of this
    check silently produced an EMPTY train split (root cause of a real,
    reproducible failure, not model- or platform-specific): the snapshot is
    1000 rows, under configs/defaults.yaml's own max_rows+val_rows+
    train_rows=2100, and since ``sample_and_split_neutrals`` fills test first,
    test consumed all 1000 rows and train/val got 0. A fresh clone of this
    repo with no local snapshot at all already falls straight to the Hub path
    below and just works -- that path is deliberately the same one taken here
    on insufficiency, not a separate code path. The Hub load itself is fully
    deterministic (``seed`` is threaded through ``sample_and_split_neutrals``,
    fixed at 0 by every caller in this codebase), so a snapshot too small for
    the request is the ONLY thing that can make this call fail or diverge --
    checking sufficiency up front removes that risk without giving up the
    snapshot's speed for calls it can actually satisfy.
    """
    from neutral_protocol import sample_and_split_neutrals

    # Mirror resolve_neutral_split_counts' own aliasing exactly (max_rows
    # aliases test_rows when test_rows is unset) so this sufficiency check
    # can't under-count what a call actually needs.
    effective_test_rows = test_rows if test_rows is not None else max_rows
    needed = [n for n in (effective_test_rows, val_rows, train_rows) if n is not None]
    needed_total = sum(needed) if needed else None

    local_neutral = ROOT / "data" / "neutral" / "biasneutral-ajibawa.csv"
    if "biasneutral-ajibawa" in str(hf_id) and local_neutral.is_file():
        local_df = pd.read_csv(local_neutral)
        if "text" not in local_df.columns:
            raise ValueError(f"Local neutral cache {local_neutral} needs a text column")
        if needed_total is None or len(local_df) >= needed_total:
            return sample_and_split_neutrals(
                local_df,
                test_rows=test_rows,
                val_rows=val_rows,
                train_rows=train_rows,
                max_rows=max_rows,
                seed=int(seed),
                respect_existing_splits=respect_existing_splits,
            )
        # Cache too small for this request -- fall through to the live Hub
        # load below rather than silently truncating a required split.

    from datasets import load_dataset

    if "biasneutral" in hf_id and "ajibawa" not in hf_id:
        raise ValueError(
            f"Refusing plain biasneutral id {hf_id!r}; use aieng-lab/biasneutral-ajibawa"
        )
    df = load_dataset(hf_id, split="train").to_pandas()
    if "text" not in df.columns and "masked" in df.columns:
        df = df.rename(columns={"masked": "text"})
    if "text" not in df.columns:
        raise ValueError(f"Neutral dataset {hf_id} needs a text column")
    return sample_and_split_neutrals(
        df,
        test_rows=test_rows,
        val_rows=val_rows,
        train_rows=train_rows,
        max_rows=max_rows,
        seed=int(seed),
        respect_existing_splits=respect_existing_splits,
    )


def load_neutral_from_cfg(
    hf_id: str,
    cfg: Dict[str, Any],
    *,
    smoke: bool = False,
    seed: int = 0,
    respect_existing_splits: bool = True,
) -> pd.DataFrame:
    """Load neutrals using ``neutral.max_rows`` / ``val_rows`` / ``train_rows``."""
    from neutral_protocol import split_kwargs_from_cfg

    return load_neutral(
        hf_id,
        **split_kwargs_from_cfg(cfg, smoke=smoke),
        seed=int(seed),
        respect_existing_splits=respect_existing_splits,
    )


def load_hf_splits(
    hf_id: str,
    *,
    config_name: Optional[str] = None,
    max_rows_per_split: Optional[int] = None,
) -> pd.DataFrame:
    """Load every published split of a Hub dataset into one data frame.

    The source split is authoritative. We write it into the ``split`` column
    even when an uploaded artifact also contains that column, preventing stale
    embedded labels from disagreeing with the Hub layout.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_id, config_name) if config_name else load_dataset(hf_id)
    if not hasattr(ds, "items"):
        raise ValueError(f"Expected DatasetDict from {hf_id!r}, got {type(ds).__name__}")

    frames = []
    for split_name, split_ds in ds.items():
        frame = split_ds.to_pandas()
        if max_rows_per_split is not None:
            frame = frame.head(int(max_rows_per_split))
        frame["split"] = "validation" if split_name == "val" else str(split_name)
        frames.append(frame)
    if not frames:
        raise ValueError(f"Dataset {hf_id!r} has no splits")
    return pd.concat(frames, ignore_index=True)


def assign_splits(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    if "split" in out.columns and out["split"].notna().any():
        out["split"] = out["split"].replace({"val": "validation"})
        return out
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(out)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    splits = np.array(["test"] * n, dtype=object)
    splits[idx[:n_train]] = "train"
    splits[idx[n_train : n_train + n_val]] = "validation"
    out["split"] = splits
    return out
