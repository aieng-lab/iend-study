"""Convert MIB/RAVEL HuggingFace datasets into GRADIEND/study CSV schema.

Outputs under ``data/mib/``:

- ``ioi_mib.csv`` — IO vs SUBJECT (same protocol as synthetic ``ioi``)
- ``ravel_{continent,country,language}.csv`` — balanced, entity-disjoint top-K
  attribute classes derived from the original disambiguated RAVEL records
- ``ravel_*_neutral.csv`` — off-query + wikipedia distractors

Schema (training rows):
  masked, label, label_class, alternative, alternative_class, pair_id, split
  (+ entity / attribute / query_attr metadata where applicable)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIB_DIR = ROOT / "data" / "mib"
DATA_VERSION = 1
RAVEL_DATA_VERSION = 2
RAVEL_HF_ID = "hij/ravel"
# Pin the source revision used to build the paper datasets. The Hub dataset
# contains stable entity IDs and URLs that the flattened MIB adapter omits.
RAVEL_REVISION = "2c45eb232b9944c235509db5320345eaba563e07"

# Frequency-based defaults from mib-bench/ravel train split.
DEFAULT_TOP_K = 3
RAVEL_TOP: Dict[str, Tuple[str, ...]] = {
    "Continent": ("Asia", "Europe", "Africa"),
    "Country": ("United States", "China", "Russia"),
    "Language": ("English", "Portuguese", "Spanish"),
}

ATTR_TO_TASK = {
    "Continent": "ravel_continent",
    "Country": "ravel_country",
    "Language": "ravel_language",
}


def _slug_class(value: str) -> str:
    """Stable class id: lowercase, spaces → underscore."""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unk"


def _mask_prompt(prompt: str) -> str:
    p = str(prompt).rstrip()
    if "[MASK]" in p:
        return p
    # No leading space when the answer continues a token/string (… "continent": ").
    if p.endswith((" ", "\t", '"', "'", "=", ":", "/", "(")):
        return p + "[MASK]"
    return p + " [MASK]"


def _write_csv(
    path: Path,
    df: pd.DataFrame,
    *,
    meta: Optional[Dict[str, Any]] = None,
    version: int = DATA_VERSION,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    payload = {
        "version": version,
        "n": len(df),
        "classes": sorted(df["label_class"].astype(str).unique().tolist())
        if "label_class" in df.columns
        else [],
        **(meta or {}),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_neutral(
    path: Path,
    texts: Sequence[str],
    *,
    meta: Optional[Dict[str, Any]] = None,
    version: int = DATA_VERSION,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"text": list(texts)})
    df.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(
        json.dumps({"version": version, "n": len(df), **(meta or {})}, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# IOI (mib-bench/ioi)
# ---------------------------------------------------------------------------

def build_ioi_mib_df(
    *,
    max_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Map MIB IOI → one-pole factual=IO / alternative=SUBJECT rows."""
    from datasets import load_dataset

    ds = load_dataset("mib-bench/ioi")
    rows: List[dict] = []
    rng = np.random.default_rng(seed)
    for split_name, split in ds.items():
        out_split = "validation" if split_name in ("val", "validation") else split_name
        idxs = list(range(len(split)))
        lim = None if max_per_split is None else max_per_split.get(out_split)
        if lim is not None and lim < len(idxs):
            idxs = list(rng.choice(idxs, size=int(lim), replace=False))
        for i in idxs:
            r = split[int(i)]
            meta = r["metadata"] or {}
            io = str(meta.get("indirect_object") or r["choices"][int(r["answerKey"])])
            subj = str(meta.get("subject") or r["choices"][1 - int(r["answerKey"])])
            cf = r.get("s2_io_flip_counterfactual") or {}
            alt_prompt = cf.get("prompt") if isinstance(cf, dict) else None
            rows.append(
                {
                    "masked": _mask_prompt(r["prompt"]),
                    "label": io,
                    "label_class": "IO",
                    "alternative": subj,
                    "alternative_class": "SUBJECT",
                    "pair_id": f"ioi_mib:{out_split}:{i}",
                    "split": out_split,
                    "template": r.get("template"),
                    "cf_masked": _mask_prompt(alt_prompt) if alt_prompt else "",
                    "source": "mib-bench/ioi",
                }
            )
    return pd.DataFrame(rows)


def generate_ioi_mib(*, force: bool = False, seed: int = 0) -> Path:
    path = MIB_DIR / "ioi_mib.csv"
    if path.is_file() and not force:
        return path
    df = build_ioi_mib_df(seed=seed)
    return _write_csv(
        path,
        df,
        meta={
            "hf": "mib-bench/ioi",
            "protocol": "IO vs SUBJECT (factual=IO)",
            "counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        },
    )


# ---------------------------------------------------------------------------
# RAVEL (hij/ravel)
# ---------------------------------------------------------------------------


def _load_ravel_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pinned original city entities and prompt templates."""
    from datasets import load_dataset

    entity_ds = load_dataset(
        RAVEL_HF_ID,
        "city_entity",
        revision=RAVEL_REVISION,
    )
    prompt_ds = load_dataset(
        RAVEL_HF_ID,
        "city_prompt",
        revision=RAVEL_REVISION,
    )

    entities = []
    for source_split, split in entity_ds.items():
        frame = split.to_pandas()
        frame["source_entity_split"] = (
            "validation" if source_split in {"val", "validation"} else source_split
        )
        entities.append(frame)
    prompts = []
    for source_split, split in prompt_ds.items():
        frame = split.to_pandas()
        frame["source_prompt_split"] = (
            "validation" if source_split in {"val", "validation"} else source_split
        )
        prompts.append(frame)
    return pd.concat(entities, ignore_index=True), pd.concat(prompts, ignore_index=True)


def _surface_key(value: Any) -> str:
    """Canonicalize only presentation-irrelevant whitespace/case."""
    return " ".join(str(value).split()).casefold()


def _canonical_ravel_entities(
    entities: pd.DataFrame,
    attr: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return one prompt-identifiable entity per surface form.

    RAVEL deliberately disambiguates homonymous cities with ``ID`` and ``URL``.
    Our cloze prompt displays only ``City``. If the same displayed city has
    conflicting values for the queried attribute, the prompt cannot determine
    which source entity is intended, so all such rows are excluded. Repeated
    IDs with the same displayed city and value are collapsed to one row rather
    than treated as independent observations.
    """
    required = {"ID", "City", "URL", attr}
    missing = sorted(required.difference(entities.columns))
    if missing:
        raise ValueError(f"{RAVEL_HF_ID}/city_entity is missing columns: {missing}")

    frame = entities.dropna(subset=["ID", "City", attr]).copy()
    frame["surface_key"] = frame["City"].map(_surface_key)
    value_counts = frame.groupby("surface_key", sort=False)[attr].nunique(dropna=True)
    ambiguous_keys = set(value_counts[value_counts > 1].index)
    ambiguous = frame[frame["surface_key"].isin(ambiguous_keys)]

    clean = frame[~frame["surface_key"].isin(ambiguous_keys)].copy()
    clean = clean.sort_values(["surface_key", "ID"], kind="stable")
    duplicate_surface_rows = int(clean.duplicated(["surface_key"], keep="first").sum())
    clean = clean.drop_duplicates(["surface_key"], keep="first").reset_index(drop=True)
    audit = {
        "ambiguous_surface_count": int(len(ambiguous_keys)),
        "ambiguous_entity_count": int(len(ambiguous)),
        "same_value_duplicate_entity_count": duplicate_surface_rows,
        "ambiguous_surface_examples": sorted(
            ambiguous.groupby("surface_key")["City"].first().astype(str).tolist()
        )[:20],
    }
    return clean, audit


def _ravel_templates_by_split(prompts: pd.DataFrame, attr: str) -> Dict[str, List[str]]:
    required = {"Template", "Attribute", "Source", "Entity", "source_prompt_split"}
    missing = sorted(required.difference(prompts.columns))
    if missing:
        raise ValueError(f"{RAVEL_HF_ID}/city_prompt is missing columns: {missing}")
    generic = prompts[
        prompts["Attribute"].astype(str).eq(attr)
        & prompts["Source"].astype(str).str.casefold().eq("ravel")
        & prompts["Entity"].fillna("").astype(str).str.strip().eq("")
    ].copy()
    generic["Template"] = generic["Template"].astype(str)
    generic = generic[generic["Template"].str.count("%s").eq(1)]
    out: Dict[str, List[str]] = {}
    for split in ("train", "validation", "test"):
        values = generic.loc[
            generic["source_prompt_split"].eq(split), "Template"
        ].drop_duplicates().tolist()
        if not values:
            raise ValueError(f"No generic RAVEL {attr} templates for split={split}")
        out[split] = values
    return out


def _balanced_entity_split(
    frame: pd.DataFrame,
    *,
    attr: str,
    values: Sequence[str],
    seed: int,
) -> Tuple[pd.DataFrame, int]:
    """Balance classes, then assign exact 80/10/10 entity-level splits."""
    counts = frame[attr].value_counts()
    missing = [value for value in values if int(counts.get(value, 0)) < 10]
    if missing:
        raise ValueError(f"RAVEL {attr} has fewer than 10 unambiguous entities for {missing}")
    # A multiple of ten gives exact, equal 80/10/10 counts for every class.
    per_class = (min(int(counts[value]) for value in values) // 10) * 10
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for value in values:
        cls = frame[frame[attr].eq(value)].copy()
        cls = cls.iloc[rng.permutation(len(cls))[:per_class]].reset_index(drop=True)
        n_train = per_class * 8 // 10
        n_val = per_class // 10
        cls["split"] = (
            ["train"] * n_train
            + ["validation"] * n_val
            + ["test"] * (per_class - n_train - n_val)
        )
        parts.append(cls)
    return pd.concat(parts, ignore_index=True), per_class

def _ravel_top_values(attr: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, ...]:
    preset = RAVEL_TOP.get(attr)
    if preset and int(top_k) == len(preset):
        return preset
    if preset and int(top_k) < len(preset):
        return preset[: int(top_k)]
    # top_k exceeds the preset: derive it from unique, prompt-identifiable
    # entities rather than the repeated intervention pairs in mib-bench/ravel.
    entities, _ = _load_ravel_sources()
    clean, _audit = _canonical_ravel_entities(entities, attr)
    values = tuple(v for v, _ in Counter(clean[attr].astype(str)).most_common(int(top_k)))
    if len(values) < int(top_k):
        raise ValueError(
            f"RAVEL attribute {attr!r} has only {len(values)} distinct values; "
            f"top_k={top_k} cannot be satisfied. Lower top_k rather than "
            "training on fewer classes than the task config declares."
        )
    return values


def build_ravel_attribute_df(
    attr: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    top_values: Optional[Sequence[str]] = None,
    max_per_class_per_split: Optional[int] = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a balanced, entity-disjoint derivative of original RAVEL."""
    allowed_surface = tuple(top_values) if top_values else _ravel_top_values(attr, top_k=top_k)
    surface_to_slug = {v: _slug_class(v) for v in allowed_surface}
    entities, prompts = _load_ravel_sources()
    canonical, ambiguity_audit = _canonical_ravel_entities(entities, attr)
    selected = canonical[canonical[attr].astype(str).isin(allowed_surface)].copy()
    selected, per_class = _balanced_entity_split(
        selected,
        attr=attr,
        values=allowed_surface,
        seed=seed,
    )
    templates = _ravel_templates_by_split(prompts, attr)
    rng = np.random.default_rng(seed)
    train_rows: List[dict] = []
    for split in ("train", "validation", "test"):
        split_rows = selected[selected["split"].eq(split)].copy()
        if max_per_class_per_split is not None:
            split_rows = split_rows.groupby(attr, sort=False, group_keys=False).head(
                int(max_per_class_per_split)
            )
        split_templates = list(templates[split])
        rng.shuffle(split_templates)
        pools = {
            value: split_rows[split_rows[attr].astype(str).eq(value)].reset_index(drop=True)
            for value in allowed_surface
        }
        for row_index, (_, entity) in enumerate(split_rows.iterrows()):
            label = str(entity[attr])
            rivals = [value for value in allowed_surface if value != label]
            alternative = rivals[row_index % len(rivals)]
            rival_pool = pools[alternative]
            rival = rival_pool.iloc[row_index % len(rival_pool)]
            template = split_templates[row_index % len(split_templates)]
            prompt = template % str(entity["City"])
            cf_prompt = template % str(rival["City"])
            train_rows.append(
                {
                    "masked": _mask_prompt(prompt),
                    "label": label,
                    "label_class": surface_to_slug[label],
                    "alternative": alternative,
                    "alternative_class": surface_to_slug[alternative],
                    "pair_id": f"ravel:{attr}:{split}:{entity['ID']}",
                    "split": split,
                    "entity": str(entity["City"]),
                    "entity_id": str(entity["ID"]),
                    "entity_url": str(entity["URL"]),
                    "source_entity_split": str(entity["source_entity_split"]),
                    "query_attr": attr,
                    "template": template,
                    "source_prompt_split": split,
                    "cf_masked": _mask_prompt(cf_prompt),
                    "alternative_entity": str(rival["City"]),
                    "alternative_entity_id": str(rival["ID"]),
                    "source": RAVEL_HF_ID,
                    "source_revision": RAVEL_REVISION,
                }
            )

    train_df = pd.DataFrame(train_rows)
    if train_df["masked"].duplicated().any():
        raise RuntimeError(f"RAVEL {attr} build produced duplicate displayed prompts")
    entity_split_counts = train_df.groupby("entity_id")["split"].nunique()
    if int(entity_split_counts.max()) != 1:
        raise RuntimeError(f"RAVEL {attr} entity IDs cross data splits")

    # Off-query RAVEL templates provide a neutral pool with the same entity
    # domain and formatting, but no query for the target attribute.
    neutral_templates = prompts[
        prompts["Attribute"].astype(str).ne(attr)
        & prompts["Attribute"].fillna("").astype(str).str.strip().ne("")
        & prompts["Source"].astype(str).str.casefold().eq("ravel")
        & prompts["Entity"].fillna("").astype(str).str.strip().eq("")
    ]["Template"].astype(str).drop_duplicates()
    neutral_templates = neutral_templates[neutral_templates.str.count("%s").eq(1)].tolist()
    neutral_entities = canonical.iloc[rng.permutation(len(canonical))].reset_index(drop=True)
    neutral_texts: List[str] = []
    seen_neutral = set()
    if neutral_templates:
        for i in range(5000):
            entity = neutral_entities.iloc[i % len(neutral_entities)]
            text = (neutral_templates[i % len(neutral_templates)] % str(entity["City"])).strip()
            if text and text not in seen_neutral:
                seen_neutral.add(text)
                neutral_texts.append(text)
    train_df.attrs["ravel_audit"] = {
        **ambiguity_audit,
        "balanced_entities_per_class": per_class,
    }
    return train_df, pd.DataFrame({"text": neutral_texts})


def ravel_top_k_suffix(top_k) -> str:
    """Filename suffix distinguishing a larger-K RAVEL build from the base."""
    value = int(top_k)
    return "" if value == DEFAULT_TOP_K else f"_k{value}"


def _ravel_artifacts_current(train_path: Path, neutral_path: Path) -> bool:
    if not train_path.is_file() or not neutral_path.is_file():
        return False
    try:
        train_meta = json.loads(train_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        neutral_meta = json.loads(neutral_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        int(train_meta.get("version", 0)) == RAVEL_DATA_VERSION
        and int(neutral_meta.get("version", 0)) == RAVEL_DATA_VERSION
    )


def generate_ravel_attribute(
    attr: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    force: bool = False,
    seed: int = 0,
    max_per_class_per_split: Optional[int] = None,
) -> Tuple[Path, Path]:
    task_id = ATTR_TO_TASK[attr]
    # A non-default top_k is a different dataset; writing it to the base path
    # would silently overwrite the K=3 CSV every other task depends on.
    suffix = ravel_top_k_suffix(top_k)
    train_path = MIB_DIR / f"{task_id}{suffix}.csv"
    neu_path = MIB_DIR / f"{task_id}{suffix}_neutral.csv"
    if not force and _ravel_artifacts_current(train_path, neu_path):
        return train_path, neu_path

    top_values = _ravel_top_values(attr, top_k=top_k)
    train_df, neu_df = build_ravel_attribute_df(
        attr,
        top_k=top_k,
        top_values=top_values,
        max_per_class_per_split=max_per_class_per_split,
        seed=seed,
    )
    if train_df.empty:
        raise RuntimeError(f"No RAVEL rows for attribute={attr} top={top_values}")

    _write_csv(
        train_path,
        train_df,
        meta={
            "hf": RAVEL_HF_ID,
            "revision": RAVEL_REVISION,
            "attribute": attr,
            "top_values": list(top_values),
            "class_ids": sorted(train_df["label_class"].unique().tolist()),
            "counts_by_class_and_split": {
                str(class_id): {
                    str(split): int(n)
                    for split, n in group["split"].value_counts().sort_index().items()
                }
                for class_id, group in train_df.groupby("label_class", sort=True)
            },
            "counts_by_split": {
                str(k): int(v) for k, v in train_df["split"].value_counts().items()
            },
            "split_policy": (
                "balanced 80/10/10 by prompt-identifiable city surface; entity IDs and "
                "prompt templates are split-disjoint"
            ),
            "ambiguity_policy": (
                "exclude every displayed city surface that maps to multiple values for the queried attribute; "
                "collapse same-value duplicate surfaces to one canonical source ID"
            ),
            **dict(train_df.attrs.get("ravel_audit") or {}),
        },
        version=RAVEL_DATA_VERSION,
    )
    _write_neutral(
        neu_path,
        neu_df["text"].tolist(),
        meta={
            "hf": RAVEL_HF_ID,
            "revision": RAVEL_REVISION,
            "attribute": attr,
            "kind": "off-query RAVEL city prompts",
        },
        version=RAVEL_DATA_VERSION,
    )
    return train_path, neu_path


def generate_all_mib(
    *,
    force: bool = True,
    seed: int = 0,
    ravel_top_k: int = DEFAULT_TOP_K,
    ravel_max_per_class_per_split: Optional[int] = None,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    out["ioi_mib"] = generate_ioi_mib(force=force, seed=seed)
    for attr in ("Continent", "Country", "Language"):
        train_p, neu_p = generate_ravel_attribute(
            attr,
            top_k=ravel_top_k,
            force=force,
            seed=seed,
            max_per_class_per_split=ravel_max_per_class_per_split,
        )
        out[ATTR_TO_TASK[attr]] = train_p
        out[ATTR_TO_TASK[attr] + "_neutral"] = neu_p
    return out


# ---------------------------------------------------------------------------
# TaskBundle builders
# ---------------------------------------------------------------------------

def _load_local_neutral(path: Path, cfg: Dict[str, Any], *, smoke: bool = False) -> pd.DataFrame:
    from neutral_protocol import sample_and_split_neutrals, split_kwargs_from_cfg

    df = pd.read_csv(path)
    if "text" not in df.columns and "masked" in df.columns:
        df = df.rename(columns={"masked": "text"})
    if "text" not in df.columns:
        raise ValueError(f"Neutral file {path} needs a text column")
    return sample_and_split_neutrals(df, **split_kwargs_from_cfg(cfg, smoke=smoke))


def build_ioi_mib(cfg: Dict[str, Any], *, smoke: bool = False):
    from study.tasks import TaskBundle, load_neutral_from_cfg

    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    path = Path(data_cfg.get("local_path") or (MIB_DIR / "ioi_mib.csv"))
    if not path.is_file():
        path = generate_ioi_mib(force=True)
    df = pd.read_csv(path)
    if smoke:
        df = df.groupby("split", group_keys=False).head(32)
    classes = list(task.get("classes") or ["IO", "SUBJECT"])
    pos, neg = classes[0], classes[1]
    data_per_class = {
        pos: df[["masked", "split", "label"]].rename(columns={"label": pos}),
        neg: df.assign(label=df["alternative"])[["masked", "split", "label"]].rename(
            columns={"label": neg}
        ),
    }
    neutrals = load_neutral_from_cfg(
        data_cfg.get("neutral_hf", "aieng-lab/biasneutral-ajibawa"),
        cfg,
        smoke=smoke,
    )
    return TaskBundle(
        task_id="ioi_mib",
        classes=classes,
        data_per_class=data_per_class,
        merged_df=df,
        neutrals=neutrals,
        eval_spec={
            "causal_score": (task.get("causal") or {}).get("score", "mask_argmax_among"),
            "source_both": True,
            "min_base_accuracy": (task.get("eval") or {}).get("min_base_accuracy"),
        },
        training_overrides=dict(task.get("training") or {}),
        notes="MIB mib-bench/ioi → IO vs SUBJECT (compare to synthetic ioi)",
    )


def build_ravel_task(task_id: str, cfg: Dict[str, Any], *, smoke: bool = False):
    from study.tasks import TaskBundle, per_class_from_merged

    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    # Read attribute/top_k from the task config. A generated larger-K variant
    # resolves to this builder, so deriving them from task_id alone silently
    # served the base K=3 CSV and failed with "No rows for class ...".
    attr = str(
        data_cfg.get("attribute")
        or {
            "ravel_continent": "Continent",
            "ravel_country": "Country",
            "ravel_language": "Language",
        }[task_id]
    )
    top_k = int(data_cfg.get("top_k") or DEFAULT_TOP_K)
    base_id = ATTR_TO_TASK[attr]
    suffix = ravel_top_k_suffix(top_k)
    train_path = Path(
        data_cfg.get("local_path") or (MIB_DIR / f"{base_id}{suffix}.csv")
    )
    neu_path = Path(
        data_cfg.get("neutral_path") or (MIB_DIR / f"{base_id}{suffix}_neutral.csv")
    )
    if not _ravel_artifacts_current(train_path, neu_path):
        train_path, neu_path = generate_ravel_attribute(
            attr, top_k=top_k, force=True
        )

    df = pd.read_csv(train_path)
    if smoke:
        # Group by split *and* class — grouping by class alone (pre-2026-08-18)
        # let .head(24) grab only 'train' rows for a class whenever the CSV's
        # rows happen to be split-blocked, leaving do_eval=True with no
        # validation/test split at all. Confirmed on ravel_country 2026-08-18.
        df = df.groupby(["split", "label_class"], group_keys=False).head(24)
    classes = list(task.get("classes") or sorted(df["label_class"].astype(str).unique()))
    data_per_class = per_class_from_merged(df, classes=classes)
    neutrals = _load_local_neutral(neu_path, cfg, smoke=smoke)
    return TaskBundle(
        task_id=task_id,
        classes=classes,
        data_per_class=data_per_class,
        merged_df=df,
        neutrals=neutrals,
        eval_spec={
            "causal_score": (task.get("causal") or {}).get("score", "class_token_logit"),
            "multiclass": True,
            "min_base_accuracy": (task.get("eval") or {}).get("min_base_accuracy", 0.4),
            "attribute": attr,
        },
        training_overrides=dict(task.get("training") or {}),
        excluded_words=[],
        notes=(
            f"RAVEL {RAVEL_HF_ID}@{RAVEL_REVISION[:8]} attribute={attr}; "
            f"balanced entity/template-disjoint 80/10/10; classes={classes}"
        ),
    )


def build_ravel_continent(cfg: Dict[str, Any], *, smoke: bool = False):
    return build_ravel_task("ravel_continent", cfg, smoke=smoke)


def build_ravel_country(cfg: Dict[str, Any], *, smoke: bool = False):
    return build_ravel_task("ravel_country", cfg, smoke=smoke)


def build_ravel_language(cfg: Dict[str, Any], *, smoke: bool = False):
    return build_ravel_task("ravel_language", cfg, smoke=smoke)
