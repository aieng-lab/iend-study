"""Shared neutral-document protocol (split, sampling, excluded tokens).

Counts are **absolute** and come from config (``neutral.max_rows`` /
``neutral.val_rows`` / ``neutral.train_rows``), not fractions of one pool.

- ``test_rows``  ← ``neutral.max_rows`` — report spec / auc_n / ΔP
- ``val_rows``   ← ``neutral.val_rows`` — Youden τ, SAE rank, causal strength
- ``train_rows`` ← ``neutral.train_rows`` — GRADIEND cloze identity

Shuffle with a fixed seed, then take ``train + val + test`` disjoint rows.
Encode methods that score every non-pad token skip pad **and** class-token ids
(``excluded_words``). GRADIEND uses the train split (do not shrink ``train_rows``).
"""

from __future__ import annotations

from typing import Any, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

NEUTRAL_SPLIT_SEED = 0
NEUTRAL_TEXT_COLUMNS: Tuple[str, ...] = ("text", "masked", "prompt")
_NEUTRAL_COUNT_KEYS = ("max_rows", "val_rows", "train_rows")


def resolve_neutral_split_counts(
    *,
    test_rows: Optional[int] = None,
    val_rows: Optional[int] = None,
    train_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Return ``(train, val, test)`` counts. ``max_rows`` aliases ``test_rows``.

    All three sizes must be passed (from config). There is no coded default.
    """
    test = test_rows if test_rows is not None else max_rows
    if test is None or val_rows is None or train_rows is None:
        raise ValueError(
            "neutral split needs test_rows (or max_rows), val_rows, and train_rows "
            "from config (neutral.max_rows / val_rows / train_rows)"
        )
    test_n, val_n, train_n = int(test), int(val_rows), int(train_rows)
    if test_n < 1:
        raise ValueError(f"test_rows must be ≥ 1, got {test_n}")
    if val_n < 0 or train_n < 0:
        raise ValueError(f"val_rows/train_rows must be ≥ 0, got val={val_n} train={train_n}")
    return train_n, val_n, test_n


def split_kwargs_from_cfg(
    cfg: Optional[Mapping[str, Any]] = None,
    *,
    smoke: bool = False,
) -> dict:
    """Read ``neutral.max_rows`` / ``val_rows`` / ``train_rows`` from merged config.

    ``--smoke`` applies ``smoke.neutral`` from the same config (also YAML).
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("split_kwargs_from_cfg needs the merged study config mapping")
    neu = dict(cfg.get("neutral") or {})
    if smoke:
        smoke_neu = (cfg.get("smoke") or {}).get("neutral")
        if not isinstance(smoke_neu, Mapping):
            raise ValueError(
                "smoke neutral sizes need smoke.neutral in config "
                "(max_rows, val_rows, train_rows)"
            )
        neu.update(dict(smoke_neu))
    missing = [k for k in _NEUTRAL_COUNT_KEYS if neu.get(k) is None]
    if missing:
        raise ValueError(
            f"config.neutral missing {missing}; set max_rows, val_rows, train_rows"
        )
    return {
        "test_rows": int(neu["max_rows"]),
        "val_rows": int(neu["val_rows"]),
        "train_rows": int(neu["train_rows"]),
    }


def sample_and_split_neutrals(
    df: pd.DataFrame,
    *,
    test_rows: Optional[int] = None,
    val_rows: Optional[int] = None,
    train_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    seed: int = NEUTRAL_SPLIT_SEED,
    respect_existing_splits: bool = True,
) -> pd.DataFrame:
    """Shuffle with ``seed``, take train+val+test rows, assign splits.

    Test and validation are filled first so report/select sets are not empty
    when the corpus is smaller than the requested total. All four size args
    ``None`` means shuffle only (no cap); used by regen scripts.
    """
    if df is None or getattr(df, "empty", True):
        return df
    unlimited = all(v is None for v in (test_rows, val_rows, train_rows, max_rows))
    if unlimited:
        out = df.copy().reset_index(drop=True)
        rng = np.random.default_rng(int(seed))
        return out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    train_n, val_n, test_n = resolve_neutral_split_counts(
        test_rows=test_rows,
        val_rows=val_rows,
        train_rows=train_rows,
        max_rows=max_rows,
    )
    out = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(out))
    out = out.iloc[order].reset_index(drop=True)
    if (
        respect_existing_splits
        and "split" in out.columns
        and out["split"].notna().any()
    ):
        out["split"] = out["split"].replace({"val": "validation"})
        parts = []
        for name, cap in (
            ("train", train_n),
            ("validation", val_n),
            ("test", test_n),
        ):
            sub = out.loc[_split_mask(out["split"], name)]
            if cap > 0:
                parts.append(sub.head(int(cap)))
        if parts:
            out = pd.concat(parts, ignore_index=True)
        return out.reset_index(drop=True)
    if not respect_existing_splits and "split" in out.columns:
        out = out.drop(columns=["split"])
    need = train_n + val_n + test_n
    out = out.head(int(need)).reset_index(drop=True)
    n = len(out)
    if not respect_existing_splits and n < need:
        requested = np.asarray([test_n, val_n, train_n], dtype=np.float64)
        positive = requested > 0
        counts = np.zeros(3, dtype=np.int64)
        if n >= int(np.sum(positive)):
            counts[positive] = 1
            remaining = int(n - np.sum(counts))
            residual_weights = np.maximum(requested - counts, 0.0)
            if remaining > 0 and float(np.sum(residual_weights)) > 0.0:
                quotas = residual_weights / np.sum(residual_weights) * remaining
                additions = np.floor(quotas).astype(np.int64)
                counts += additions
                leftover = int(remaining - np.sum(additions))
                order = np.argsort(-(quotas - additions), kind="stable")
                counts[order[:leftover]] += 1
        else:
            # Degenerate tiny pools cannot populate every requested split.
            counts[np.flatnonzero(positive)[:n]] = 1
        n_test, n_val, n_train = map(int, counts)
    else:
        n_test = min(test_n, n)
        n_val = min(val_n, max(0, n - n_test))
        n_train = max(0, n - n_test - n_val)
    splits = np.array(
        (["test"] * n_test) + (["validation"] * n_val) + (["train"] * n_train),
        dtype=object,
    )
    out["split"] = splits
    return out


def _split_mask(series: pd.Series, split: str) -> pd.Series:
    want = str(split)
    vals = series.astype(str)
    if want == "validation":
        return vals.isin(["validation", "val"])
    return vals == want


def frame_for_split(
    df: Optional[pd.DataFrame],
    split: str,
    *,
    fallback: str = "all",
) -> pd.DataFrame:
    """Rows for ``split``. If missing/empty: ``all`` (full frame) or empty."""
    if df is None or getattr(df, "empty", True):
        return df if df is not None else pd.DataFrame()
    if "split" not in df.columns:
        return df if fallback == "all" else df.iloc[0:0].copy()
    sub = df.loc[_split_mask(df["split"], split)].copy()
    if sub.empty and fallback == "all":
        return df
    return sub.reset_index(drop=True)


def neutral_text_column(df: pd.DataFrame) -> Optional[str]:
    for col in NEUTRAL_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None


def texts_for_split(
    df: Optional[pd.DataFrame],
    split: str,
    *,
    max_rows: Optional[int] = None,
    fallback: str = "all",
) -> list[str]:
    sub = frame_for_split(df, split, fallback=fallback)
    if sub is None or getattr(sub, "empty", True):
        return []
    col = neutral_text_column(sub)
    if col is None:
        return []
    texts = sub[col].astype(str).tolist()
    if max_rows is not None:
        texts = texts[: int(max_rows)]
    return texts


def excluded_token_id_set(tokenizer: Any, words: Optional[Iterable[str]]) -> FrozenSet[int]:
    """Token ids for class words, with/without a leading space (BPE)."""
    ids: set[int] = set()
    for raw in words or ():
        w = str(raw).strip()
        if not w:
            continue
        variants = {
            w,
            w.lower(),
            w.capitalize(),
            w.upper(),
            f" {w}",
            f" {w.lower()}",
            f" {w.capitalize()}",
        }
        for variant in variants:
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except TypeError:
                enc = tokenizer.encode(variant)
            except Exception:
                continue
            if enc is None:
                continue
            ids.update(int(i) for i in list(enc))
    return frozenset(ids)


def make_nonpad_nonexcluded_selector(
    *,
    tokenizer: Any,
    excluded_words: Optional[Sequence[str]],
    fallback_selector: Any,
):
    """Wrap a per-token selector so excluded class-token ids are dropped."""
    if not excluded_words or tokenizer is None:
        return fallback_selector
    excluded_ids = excluded_token_id_set(tokenizer, excluded_words)
    if not excluded_ids:
        return fallback_selector

    def selector(activation: torch.Tensor, inputs: Any) -> torch.Tensor:
        if activation.dim() < 3:
            return fallback_selector(activation, inputs)
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, Mapping) else None
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        d = int(activation.shape[-1])
        if not torch.is_tensor(attention_mask):
            return activation.reshape(-1, d)
        mask = attention_mask.to(device=activation.device, dtype=torch.bool)
        if mask.shape[:2] != activation.shape[:2]:
            return activation.reshape(-1, d)
        keep = mask
        if torch.is_tensor(input_ids) and input_ids.shape[:2] == activation.shape[:2]:
            ids = input_ids.to(device=activation.device)
            excl = torch.tensor(
                sorted(excluded_ids), device=ids.device, dtype=ids.dtype
            )
            keep = mask & ~torch.isin(ids, excl)
        if not bool(keep.any()):
            keep = mask
        return activation[keep]

    return selector
