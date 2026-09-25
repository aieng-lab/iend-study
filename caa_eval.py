"""
CAA / ActAdd baseline for the gender-EN study (experimental, study-repo only).

Mean-difference steering vectors from the same gender templates used by
ACTIEND/GRADIEND. Encoding readout = cosine similarity (SV-Detect style):

    s(x) = <a(x), v> / ||a(x)||_2     (v unit-normalized)

Activation-pooling ablations (construction + encoding):
  - ``prediction`` — mean over filled prediction span (ACTIEND-matched)
  - ``pre_prediction`` — one token before that span (shared optional site)
  - ``mean``       — mean over all non-pad tokens (classical full-text)
  - ``last``       — last non-pad token (common steering default)

Pole regimes match IEND/CGA:
  - pair: one direction per unordered pair and oriented endpoint, e.g.
    ``caa:A-B:A:act_prediction``;
  - one-pole: one target-vs-rest direction, e.g. ``caa:A:act_prediction``.

Site variants append ``act_{policy}``, ``all_act_{policy}``, or
``L{n}_act_{policy}`` for concat, mean-of-layer scores/multi-layer steering,
and a single layer respectively.

No logistic regression: vectors are mean-diff only; scores are cosine only.

Implementation note on the shared contrastive-mean estimator
-------------------------------------------------------------
CAA receives materialized activations together with semantic class labels, so
it forms the contrast directly as ``mean(positive) - mean(negative)``.  CAGA
and CGA receive paired factual/alternative signals from the gradient-training
API and therefore use the algebraically equivalent orientation-corrected form
``mean(label * (factual - alternative))``.  The two forms agree for the
balanced paired data used by the study; the different code paths reflect their
input APIs (and CGA's streaming memory constraint), not different estimators.

There is one weighting caveat.  For a multiclass one-pole contrast, CAA pools
all observed rival rows before taking their mean.  A paired implementation
instead inherits the multiplicity of the constructed target/alternative
pairs.  These agree when that construction induces the same class/row weights,
as intended by the balanced study data, but need not agree for arbitrarily
unbalanced input data.

Neutrals have no prediction span. Do **not** mean-pool the whole sequence into
one vector: cosine(mean(a_t), v) washes out token-level firing and makes
class-vs-neutral AUROC easy. Score **each non-pad token** as its own sample
when the labeled site is a span/token (``prediction`` / ``pre_prediction``).
``last`` stays last-token (matched). ``mean`` stays document-mean on both
sides (that ablation is document-level by construction).

Shared site with ACTIEND/SAE (see ``activation_protocol.py``):
  - ``prediction`` — filled prediction span (default)
  - ``pre_prediction`` — one token before that span
  - ``mean`` / ``last`` — ablations (still same extractor family)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from gradiend.trainer.core.signals import ActivationSignalExtractor, Signal, SignalScope
from gradiend.trainer.text.prediction.dataset import _filled_prediction_from_template
from gradiend.util.positions import last_real_token_positions

from causal_eval import (
    DEFAULT_CAUSAL_STRENGTHS,
    CausalMethodResult,
    StrengthResult,
    causal_method_result_from_dict,
    compute_lms_safe,
    reselect_direct_bidirectional_curves,
    as_weaken_strength,
    meta_rows_to_frame,
    refine_and_select_lms_gated,
    refine_and_select_weaken_lms_gated,
    run_strength_on_model,
    score_class_probs_by_dataset,
    selection_lms_metadata,
)
from sae_eval import (
    class_vs_neutral_metrics,
    frozen_decision_rules,
    method_part_id,
    MAX_NEUTRAL_TOKEN_ROWS,
    require_frozen_rules,
    subsample_row_indices,
)
from study.method_ids import pair_key, sorted_pair

# ``pre_prediction`` = one token before the prediction span (shared with ACTIEND/SAE).
CAA_ACT_POLICIES: Tuple[str, ...] = ("prediction", "pre_prediction", "mean", "last")
CAA_DEFAULT_MAX_LENGTH = 256


def last_token_selector(activation: torch.Tensor, inputs: Any) -> torch.Tensor:
    """Mean-free last non-pad token; callable for ``Signal.activation(token_selector=…)``."""
    if activation.dim() < 3:
        return activation
    attention_mask = inputs.get("attention_mask") if isinstance(inputs, Mapping) else None
    if not torch.is_tensor(attention_mask):
        return activation[:, -1, ...]
    # Last REAL token for left and right padding alike. Neutral batches come from
    # ``tokenizer(padding=True)`` and Gemma tokenizers pad left, where
    # ``sum() - 1`` would read a different token for every row shorter than the batch.
    pos = last_real_token_positions(attention_mask.to(device=activation.device), allow_empty=True)
    b, _, d = activation.shape[0], activation.shape[1], activation.shape[-1]
    idx = pos.view(b, 1, 1).expand(b, 1, d).to(dtype=torch.long)
    return activation.gather(1, idx).squeeze(1)


def nonpad_token_selector(activation: torch.Tensor, inputs: Any) -> torch.Tensor:
    """Flatten every non-pad token to ``(n_tok, d)`` — one score site per token.

    Used for neutrals when the labeled site is a single span/token. Mean-pooling
    first would cancel directional components and inflate class-vs-neutral AUROC.
    """
    if activation.dim() < 3:
        if activation.dim() == 1:
            return activation.unsqueeze(0)
        return activation
    attention_mask = inputs.get("attention_mask") if isinstance(inputs, Mapping) else None
    d = int(activation.shape[-1])
    if not torch.is_tensor(attention_mask):
        return activation.reshape(-1, d)
    mask = attention_mask.to(device=activation.device, dtype=torch.bool)
    if mask.shape[:2] != activation.shape[:2]:
        return activation.reshape(-1, d)
    return activation[mask]


def resolve_neutral_act_selector(policy: str) -> Any:
    """Token selector for unlabeled neutrals (no ``prediction_mask``).

    ``prediction`` / ``pre_prediction`` → every non-pad token (not document mean).
    ``mean`` / ``last`` keep the labeled-side pooling so that ablation stays matched.
    """
    p = str(policy).strip().lower()
    if p in {
        "prediction",
        "pre_prediction",
        "unfilled_prediction",
        "pre-prediction",
        "pre_fill",
        "prefill",
        "tokens",
        "per_token",
        "all_tokens",
    }:
        return nonpad_token_selector
    return resolve_act_selector(policy)


def resolve_act_selector(policy: str) -> Any:
    """Map study act-policy name → ActivationSignalExtractor token_selector."""
    p = str(policy).strip().lower()
    if p in {"unfilled_prediction", "pre-prediction", "pre_fill", "prefill"}:
        p = "pre_prediction"
    if p == "prediction":
        return "prediction"
    if p == "pre_prediction":
        # Package string selector (same geometry as ACTIEND Signal.activation).
        return "pre_prediction"
    if p in {"mean", "all"}:
        return "all"
    if p == "last":
        return last_token_selector
    raise ValueError(f"Unknown CAA act policy {policy!r}; expected one of {CAA_ACT_POLICIES}")


def cosine_score(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine / projection onto unit ``direction``: <a,v> / ||a||."""
    a = np.asarray(acts, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64).reshape(-1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[-1] != v.shape[0]:
        raise ValueError(f"act dim {a.shape[-1]} != direction dim {v.shape[0]}")
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros(a.shape[0], dtype=np.float64)
    v = v / v_norm
    a_norm = np.linalg.norm(a, axis=1).clip(min=1e-12)
    return (a @ v) / a_norm


def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def caa_encoder_id(
    class_id: str,
    *,
    policy: str,
    part: Optional[str] = None,
    pair: Optional[Sequence[str]] = None,
) -> str:
    """Stable CAA id for one-pole or pair-trained directions.

    One-pole: ``caa:M:act_prediction``. Pair: ``caa:F-M:M:act_prediction``.
    """
    pol = str(policy).strip().lower()
    if pol == "all":
        pol = "mean"
    suffix = f"act_{pol}"
    class_token = str(class_id)
    if pair is not None:
        if len(pair) != 2:
            raise ValueError(f"CAA pair must contain two classes, got {pair!r}")
        class_token = f"{pair_key(str(pair[0]), str(pair[1]))}:{class_token}"
    if part is None:
        return method_part_id("caa", class_token, suffix)
    return method_part_id("caa", class_token, f"{part}_{suffix}")


def _stack_items(items: List[dict]) -> dict:
    if len(items) == 1:
        out = {k: v.unsqueeze(0) if torch.is_tensor(v) and v.dim() == 1 else v for k, v in items[0].items()}
        return out
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _neutral_batch(tokenizer, texts: Sequence[str], *, max_length: int) -> dict:
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _gender_fill_batch(
    tokenizer,
    templates: Sequence[str],
    targets: Sequence[str],
    *,
    max_length: int,
) -> dict:
    items = [
        _filled_prediction_from_template(
            tokenizer,
            template=str(t),
            target=str(y),
            max_length=int(max_length),
        )
        for t, y in zip(templates, targets)
    ]
    return _stack_items(items)


def _make_extractor(
    model: nn.Module,
    tokenizer,
    *,
    policy: str,
    layer: Optional[int] = None,
    all_layers: bool = False,
    scope: Optional[Any] = None,
) -> ActivationSignalExtractor:
    selector = resolve_act_selector(policy)
    signal = Signal.activation(token_selector=selector)
    if scope is not None:
        resolved_scope = scope
    elif layer is not None:
        scope = SignalScope.layer(int(layer))
        resolved_scope = scope
    elif all_layers:
        resolved_scope = SignalScope.layers()
    else:
        resolved_scope = SignalScope.layers()
    return ActivationSignalExtractor(
        model,
        signal=signal,
        scope=resolved_scope,
        tokenizer=tokenizer,
        aggregate_batch="none",
    )


@torch.no_grad()
def extract_activations_batched(
    extractor: ActivationSignalExtractor,
    batch_inputs: Sequence[dict] | dict,
    *,
    batch_size: int = 8,
    desc: str = "caa acts",
) -> np.ndarray:
    """Extract (N, d) activations. ``batch_inputs`` is a stacked dict or list of per-row dicts."""
    if isinstance(batch_inputs, Mapping) and torch.is_tensor(batch_inputs.get("input_ids")):
        n = int(batch_inputs["input_ids"].shape[0])
        rows: List[np.ndarray] = []
        for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
            sl = slice(start, min(start + batch_size, n))
            sub = {k: v[sl] for k, v in batch_inputs.items() if torch.is_tensor(v)}
            sig = extractor._extract(sub, validate_width=False)
            if sig.dim() == 1:
                sig = sig.unsqueeze(0)
            rows.append(sig.detach().float().cpu().numpy())
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float64)

    items = list(batch_inputs)
    rows = []
    for start in tqdm(range(0, len(items), batch_size), desc=desc, leave=False):
        chunk = items[start : start + batch_size]
        sub = _stack_items(chunk)
        sig = extractor._extract(sub, validate_width=False)
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)
        rows.append(sig.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float64)


@torch.no_grad()
def extract_activations_by_layer_batched(
    extractor: ActivationSignalExtractor,
    batch_inputs: Sequence[dict] | dict,
    *,
    layers: Sequence[int],
    batch_size: int = 8,
    desc: str = "caa acts",
) -> Dict[int, np.ndarray]:
    """Capture all requested layers in one backbone forward per batch.

    The old CAA path constructed one extractor per layer, causing the complete
    9B backbone to be evaluated once *for every layer*.  A layers-scoped
    extractor already installs hooks for all sites; split its captured tensors
    here instead of replaying the model.
    """
    layer_list = [int(x) for x in layers]
    names = [name for name, _module in extractor._module_items]
    if len(names) != len(layer_list):
        raise ValueError(f"layer/extractor mismatch: {layer_list!r} vs {names!r}")
    buckets: Dict[int, List[np.ndarray]] = {L: [] for L in layer_list}
    if isinstance(batch_inputs, Mapping) and torch.is_tensor(batch_inputs.get("input_ids")):
        n = int(batch_inputs["input_ids"].shape[0])
        chunks = (slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size))
    else:
        items = list(batch_inputs)
        n = len(items)
        chunks = (items[i : i + batch_size] for i in range(0, n, batch_size))
    for chunk in tqdm(chunks, desc=desc, leave=False, total=(n + batch_size - 1) // batch_size):
        if isinstance(chunk, slice):
            sub = {k: v[chunk] for k, v in batch_inputs.items() if torch.is_tensor(v)}
        else:
            sub = _stack_items(chunk)
        captured, prepared = extractor._capture(sub)
        selector = (extractor.signal.options or {}).get("token_selector")
        for L, name in zip(layer_list, names):
            selected = extractor._flatten_site(captured[name], prepared, selector=selector)
            if selected.dim() == 1:
                selected = selected.unsqueeze(0)
            buckets[L].append(selected.detach().float().cpu().numpy())
    return {
        L: (np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float64))
        for L, parts in buckets.items()
    }


@torch.no_grad()
def stream_caa_contrast_vectors(
    extractor: ActivationSignalExtractor,
    batch_inputs: Sequence[dict] | dict,
    *,
    layers: Sequence[int],
    labels: Sequence[str],
    pair_ids: Optional[Sequence[Any]],
    contrast_defs: Sequence[Tuple[str, str, List[str], str, Optional[List[str]]]],
    batch_size: int = 8,
    desc: str = "caa fit",
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, np.ndarray]]:
    """Fit every CAA contrast in ONE streaming pass, without ever holding the
    full ``(N, d)`` activation matrix.

    The non-streaming path (``extract_activations_by_layer_batched`` +
    ``_paired_mean_diff``) materializes ``{layer: (N, d)}`` in host float32 --
    for an 8B model with 32 layers that is ~5GB per split before the model,
    concat arrays and other splits, which is what pushes large-model CAA over a
    64G host-RAM budget. But every CAA direction is a *mean*, so it needs only
    O(d) running state, not O(N*d): fold the reduction into the capture loop.

    This reproduces ``_paired_mean_diff`` exactly at the level that matters for
    the result -- the same per-pair first-``pos``/first-``neg`` row selection for
    the paired path, and the same pooled ``mean(pos) - mean(all rival rows)`` for
    the fallback path -- and returns per-layer + concat unit vectors identical to
    the non-streaming build. It is not bit-identical: a running sum reduces in a
    different order than ``np.mean`` over the materialized matrix (accumulation is
    in float64 here, so if anything it is more precise). Directions match to
    floating-point reduction tolerance; steering/detection are unaffected.

    Memory is bounded by the number of *simultaneously incomplete* pairs (one at
    a time when a frame's pair rows are contiguous, as the study's are), plus one
    O(num_classes * num_layers * d) table of running class sums -- both tiny next
    to the (N, d) matrix this avoids.
    """
    layer_list = [int(x) for x in layers]
    names = [name for name, _module in extractor._module_items]
    if len(names) != len(layer_list):
        raise ValueError(f"layer/extractor mismatch: {layer_list!r} vs {names!r}")
    labels_arr = [str(x) for x in labels]
    n = len(labels_arr)
    use_pairs = pair_ids is not None and len(pair_ids) == n
    pair_arr = [pair_ids[i] for i in range(n)] if use_pairs else None

    # Per-contrast plan: paired (single rival, valid pair_ids) vs pooled fallback,
    # matching ``_paired_mean_diff``'s own branch condition exactly.
    plans: Dict[str, Dict[str, Any]] = {}
    for vector_key, cls, rivals, _ablation, _pair in contrast_defs:
        neg_classes = [str(rivals[0])] if len(rivals) == 1 else [str(r) for r in rivals]
        paired = use_pairs and len(neg_classes) == 1
        plans[vector_key] = {
            "pos": str(cls),
            "neg": neg_classes,
            "paired": paired,
            # paired accumulators: running sum of (pos_row - neg_row) per layer + count
            "diff_sum": {L: None for L in layer_list} if paired else None,
            "diff_count": 0,
        }
    # Which (pos, neg) class pairs need paired accumulation, and per pair_id the
    # first-seen row vector per involved class per layer (dropped once used).
    paired_pairs = {
        (p["pos"], p["neg"][0]) for p in plans.values() if p["paired"]
    }
    paired_classes = {c for pr in paired_pairs for c in pr}
    pair_first: Dict[Any, Dict[str, Dict[int, np.ndarray]]] = {}

    # Fallback accumulators: per-class running sum per layer + per-class count.
    class_sum: Dict[int, Dict[str, np.ndarray]] = {L: {} for L in layer_list}
    class_count: Dict[str, int] = {}

    selector = (extractor.signal.options or {}).get("token_selector")
    if isinstance(batch_inputs, Mapping) and torch.is_tensor(batch_inputs.get("input_ids")):
        total = int(batch_inputs["input_ids"].shape[0])
        chunks = ((i, slice(i, min(i + batch_size, total))) for i in range(0, total, batch_size))
        n_batches = (total + batch_size - 1) // batch_size
        as_slice = True
    else:
        items = list(batch_inputs)
        total = len(items)
        chunks = ((i, items[i : i + batch_size]) for i in range(0, total, batch_size))
        n_batches = (total + batch_size - 1) // batch_size
        as_slice = False

    def _finalize_pair(pid: Any) -> None:
        store = pair_first.get(pid)
        if store is None:
            return
        completed = False
        for (pos, neg) in paired_pairs:
            if pos in store and neg in store:
                completed = True
                for vk, p in plans.items():
                    if p["paired"] and p["pos"] == pos and p["neg"][0] == neg:
                        for L in layer_list:
                            d = store[pos][L] - store[neg][L]
                            if p["diff_sum"][L] is None:
                                p["diff_sum"][L] = d.astype(np.float64)
                            else:
                                p["diff_sum"][L] += d
                        p["diff_count"] += 1
        # Only drop a pair once it has actually contributed. Popping an
        # still-incomplete pair (its second class not yet seen -- e.g. a pair
        # straddling a chunk boundary) would silently discard the first class's
        # activation and lose that pair's diff entirely.
        if completed:
            pair_first.pop(pid, None)

    for start, chunk in tqdm(chunks, desc=desc, leave=False, total=n_batches):
        if as_slice:
            sub = {k: v[chunk] for k, v in batch_inputs.items() if torch.is_tensor(v)}
            chunk_n = int(sub["input_ids"].shape[0])
        else:
            sub = _stack_items(chunk)
            chunk_n = len(chunk)
        captured, prepared = extractor._capture(sub)
        # Per layer, the (chunk_n, d) block for this chunk.
        blocks: Dict[int, np.ndarray] = {}
        for L, name in zip(layer_list, names):
            selected = extractor._flatten_site(captured[name], prepared, selector=selector)
            if selected.dim() == 1:
                selected = selected.unsqueeze(0)
            blocks[L] = selected.detach().float().cpu().numpy()
        del captured, prepared
        touched_pids: List[Any] = []
        for i in range(chunk_n):
            g = start + i
            cls = labels_arr[g]
            # Fallback: running per-class sums (float64) + count once per row.
            for L in layer_list:
                row = blocks[L][i].astype(np.float64)
                acc = class_sum[L].get(cls)
                class_sum[L][cls] = row if acc is None else acc + row
            class_count[cls] = class_count.get(cls, 0) + 1
            # Paired: record first-seen row per (pid, class) for classes in a pair.
            if use_pairs and cls in paired_classes:
                pid = pair_arr[g]
                store = pair_first.setdefault(pid, {})
                if cls not in store:
                    store[cls] = {L: blocks[L][i].astype(np.float64) for L in layer_list}
                    touched_pids.append(pid)
        # Finalize any pair whose endpoints are now both present.
        for pid in dict.fromkeys(touched_pids):
            _finalize_pair(pid)
        del blocks
    # Any still-open pairs (non-contiguous frames): finalize what completed.
    for pid in list(pair_first.keys()):
        _finalize_pair(pid)

    d_dim = next(
        (v.shape[-1] for L in layer_list for v in class_sum[L].values()),
        0,
    )
    layer_map_by_key: Dict[str, Dict[int, np.ndarray]] = {}
    concat_by_key: Dict[str, np.ndarray] = {}
    for vk, p in plans.items():
        pos, neg = p["pos"], p["neg"]
        layer_map: Dict[int, np.ndarray] = {}
        concat_parts: List[np.ndarray] = []
        for L in layer_list:
            if p["paired"] and p["diff_count"] > 0:
                v = normalize_vec(p["diff_sum"][L] / float(p["diff_count"]))
            else:
                pos_sum = class_sum[L].get(pos)
                neg_sum = None
                neg_n = 0
                for c in neg:
                    cs = class_sum[L].get(c)
                    if cs is not None:
                        neg_sum = cs if neg_sum is None else neg_sum + cs
                        neg_n += class_count.get(c, 0)
                pos_n = class_count.get(pos, 0)
                if pos_sum is None or neg_sum is None or pos_n == 0 or neg_n == 0:
                    v = np.zeros(d_dim, dtype=np.float64)
                else:
                    v = normalize_vec(pos_sum / pos_n - neg_sum / neg_n)
            layer_map[L] = v
            concat_parts.append(v)
        layer_map_by_key[vk] = layer_map
        concat_by_key[vk] = normalize_vec(np.concatenate(concat_parts, axis=0))
    return layer_map_by_key, concat_by_key


def _subset_df(
    df: pd.DataFrame,
    split: str,
    max_size: Optional[int],
    *,
    label_col: str = "label_class",
) -> pd.DataFrame:
    """Split filter + optional size cap.

    When ``label_col`` is present, the cap is **class-stratified** (groupby head).
    Plain ``iloc[:max_size]`` on class-blocked religion/race frames otherwise keeps
    only the first class and zeroes the rest (``insufficient samples``).
    """
    out = df
    if "split" in df.columns:
        want = str(split)
        if want == "validation":
            mask = df["split"].astype(str).isin(["validation", "val"])
        else:
            mask = df["split"].astype(str) == want
        out = df.loc[mask].copy()
    if max_size is not None and len(out) > int(max_size):
        cap = int(max_size)
        if label_col in out.columns and out[label_col].nunique() > 1:
            n_cls = int(out[label_col].nunique())
            per = max(1, cap // n_cls)
            out = out.groupby(label_col, group_keys=False).head(per)
            if len(out) > cap:
                # Trim leftovers while keeping class order stable.
                out = (
                    out.groupby(label_col, group_keys=False, sort=False)
                    .head(max(1, cap // max(1, out[label_col].nunique())))
                )
            if len(out) > cap:
                out = out.iloc[:cap].copy()
        else:
            out = out.iloc[:cap].copy()
    return out.reset_index(drop=True)


def _paired_mean_diff(
    acts: np.ndarray,
    labels: Sequence[str],
    pair_ids: Optional[Sequence[Any]],
    *,
    class_pos: str,
    class_neg: str | Sequence[str],
) -> np.ndarray:
    """Return CAA's semantic positive-minus-negative mean direction.

    With one rival and usable ``pair_ids``, every pair is oriented explicitly
    as ``positive - negative`` before averaging.  This is equivalent to the
    orientation-corrected ``mean(label * (factual - alternative))`` used by
    CAGA/CGA on balanced paired data; it is not an unoriented mean of raw
    factual-minus-alternative differences.

    ``class_neg`` may also be a list for a multiclass positive-vs-rest
    contrast.  In that case this function deliberately falls back to the two
    empirical class pools, ``mean(pos) - mean(all negative rows)``.  Thus its
    rival weighting follows row frequencies, whereas CAGA/CGA inherit the
    multiplicities of their constructed pairs.  The results coincide only
    when those weighting schemes coincide.
    """
    labels_arr = np.asarray(labels).astype(str)
    neg_classes = (
        [str(class_neg)]
        if isinstance(class_neg, str)
        else [str(c) for c in class_neg]
    )
    if pair_ids is not None and len(pair_ids) == len(labels_arr) and len(neg_classes) == 1:
        pair_arr = np.asarray(pair_ids)
        neg = neg_classes[0]
        diffs: List[np.ndarray] = []
        for pid in pd.unique(pair_arr):
            idx = np.where(pair_arr == pid)[0]
            labs = labels_arr[idx]
            if class_pos not in labs or neg not in labs:
                continue
            a_pos = acts[idx[labs == class_pos][0]]
            a_neg = acts[idx[labs == neg][0]]
            diffs.append(a_pos - a_neg)
        if diffs:
            return normalize_vec(np.mean(np.stack(diffs, axis=0), axis=0))
    pos = acts[labels_arr == str(class_pos)]
    neg_mask = np.isin(labels_arr, neg_classes)
    neg = acts[neg_mask]
    if pos.size == 0 or neg.size == 0:
        return np.zeros(acts.shape[-1] if acts.ndim == 2 else 0, dtype=np.float64)
    return normalize_vec(pos.mean(axis=0) - neg.mean(axis=0))


@dataclass
class CaaVectors:
    """Per-policy CAA directions."""

    policy: str
    layers: Tuple[int, ...]
    # class -> layer -> unit vector
    by_layer: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    # class -> concat unit vector (layer order)
    by_concat: Dict[str, np.ndarray] = field(default_factory=dict)
    module_by_layer: Dict[int, str] = field(default_factory=dict)
    # vector key -> target/rival/pole metadata. Legacy payloads omit this and
    # are interpreted as one-pole class keys.
    contrast_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def layer_vec(self, class_id: str, layer: int) -> np.ndarray:
        return self.by_layer[str(class_id)][int(layer)]

    def concat_vec(self, class_id: str) -> np.ndarray:
        return self.by_concat[str(class_id)]

    def contrasts(self) -> List[Tuple[str, Dict[str, Any]]]:
        if self.contrast_meta:
            return [(str(k), dict(v)) for k, v in self.contrast_meta.items()]
        return [
            (
                str(cls),
                {
                    "target_class": str(cls),
                    "rival_classes": [],
                    "ablation": "one_pole",
                    "pair": None,
                },
            )
            for cls in self.by_layer
        ]


def fit_caa_vectors(
    model: nn.Module,
    tokenizer,
    gender_df: pd.DataFrame,
    *,
    target_classes: Sequence[str] = ("M", "F"),
    layers: Sequence[int],
    policies: Sequence[str] = CAA_ACT_POLICIES,
    train_split: str = "train",
    max_train: Optional[int] = None,
    batch_size: int = 8,
    max_length: int = CAA_DEFAULT_MAX_LENGTH,
    hf_resid_template: str,
    label_tokens: Optional[Mapping[str, str]] = None,
    output_dir: Optional[Any] = None,
    include_pair: bool = True,
    include_one_pole: bool = True,
    stream_fit: bool = False,
) -> Dict[str, CaaVectors]:
    """Fit pairwise and/or one-pole mean-diff CAA vectors.

    Pair mode fits an independent direction for every unordered class pair and
    exposes both oriented endpoints. One-pole mode fits each class against the
    pooled remainder. This mirrors the IEND/CGA pole regimes.
    """
    label_tokens = dict(label_tokens or {"M": "he", "F": "she"})
    classes = [str(c) for c in target_classes]
    if len(classes) < 2:
        raise ValueError("CAA requires at least two target classes")
    if not include_pair and not include_one_pole:
        raise ValueError("CAA requires include_pair and/or include_one_pole")
    train_df = _subset_df(gender_df, train_split, max_train)
    if train_df.empty:
        raise ValueError(f"No gender rows for CAA train split={train_split!r}")
    present = {
        str(c) for c in train_df["label_class"].astype(str).unique()
    } if "label_class" in train_df.columns else set()
    missing_fit = [c for c in classes if c not in present]
    if missing_fit:
        print(
            f"  CAA fit: missing classes in train subset after cap: {missing_fit} "
            f"(present={sorted(present)}); vectors for those classes will be zero",
            flush=True,
        )

    templates = train_df["masked"].astype(str).tolist()
    labels = train_df["label_class"].astype(str).tolist()
    targets = [label_tokens.get(lab, train_df["label"].astype(str).iloc[i]) for i, lab in enumerate(labels)]
    pair_ids = train_df["pair_id"].tolist() if "pair_id" in train_df.columns else None
    layers_t = tuple(int(L) for L in layers)
    module_by_layer = {
        int(L): str(hf_resid_template).format(layer=int(L)) for L in layers_t
    }

    out: Dict[str, CaaVectors] = {}
    for policy in policies:
        pol = "mean" if str(policy) == "all" else str(policy)
        # One forward captures every residual layer; do not replay a 9B
        # backbone once per layer.
        extractor = _make_extractor(model, tokenizer, policy=pol, scope=SignalScope.layers(*layers_t))
        batch = _gender_fill_batch(tokenizer, templates, targets, max_length=max_length)

        vecs = CaaVectors(policy=pol, layers=layers_t, module_by_layer=dict(module_by_layer))
        contrast_defs: List[Tuple[str, str, List[str], str, Optional[List[str]]]] = []
        if include_pair:
            for a, b in combinations(classes, 2):
                a, b = sorted_pair(a, b)
                contrast_defs.extend(
                    [
                        (f"pair:{pair_key(a, b)}:{a}", a, [b], "pair", [a, b]),
                        (f"pair:{pair_key(a, b)}:{b}", b, [a], "pair", [a, b]),
                    ]
                )
        if include_one_pole:
            for cls in classes:
                contrast_defs.append(
                    (
                        f"one_pole:{cls}",
                        cls,
                        [c for c in classes if c != cls],
                        "one_pole",
                        None,
                    )
                )

        # Large models stream the mean-diff into O(d) accumulators instead of
        # materializing the full (N, d) host matrix -- CAA's direction is a mean,
        # so this is mathematically equivalent (see stream_caa_contrast_vectors)
        # and the only way it fits a 64G host budget at 8B+.
        stream = bool(stream_fit) and hasattr(extractor, "_module_items")
        by_layer_acts: Dict[int, np.ndarray] = {}
        if stream:
            print(f"  CAA fit act={pol}: streaming train activations (low-memory) …", flush=True)
            layer_map_by_key, concat_by_key = stream_caa_contrast_vectors(
                extractor, batch, layers=layers_t, labels=labels, pair_ids=pair_ids,
                contrast_defs=contrast_defs, batch_size=batch_size,
                desc=f"caa fit all layers {pol}",
            )
        else:
            print(f"  CAA fit act={pol}: extracting train activations …", flush=True)
            if hasattr(extractor, "_module_items"):
                by_layer_acts = extract_activations_by_layer_batched(
                    extractor, batch, layers=layers_t, batch_size=batch_size, desc=f"caa fit all layers {pol}"
                )
            else:  # compatibility for lightweight callers/tests providing a stub extractor
                for L in layers_t:
                    by_layer_acts[L] = extract_activations_batched(
                        _make_extractor(model, tokenizer, policy=pol, layer=L), batch,
                        batch_size=batch_size, desc=f"caa fit L{L} {pol}"
                    )
        for L in layers_t:
            try:
                from study.progress import write_pipeline_progress
                write_pipeline_progress(output_dir, stage="caa_fit", policy=pol, layer=int(L))
            except Exception:
                pass

        for vector_key, cls, rivals, ablation, pair in contrast_defs:
            if stream:
                vecs.by_layer[vector_key] = layer_map_by_key[vector_key]
                vecs.by_concat[vector_key] = concat_by_key[vector_key]
            else:
                other: str | list = rivals[0] if len(rivals) == 1 else rivals
                layer_map: Dict[int, np.ndarray] = {}
                concat_parts: List[np.ndarray] = []
                for L in layers_t:
                    v = _paired_mean_diff(
                        by_layer_acts[L],
                        labels,
                        pair_ids,
                        class_pos=cls,
                        class_neg=other,
                    )
                    layer_map[L] = v
                    concat_parts.append(v)
                vecs.by_layer[vector_key] = layer_map
                vecs.by_concat[vector_key] = normalize_vec(np.concatenate(concat_parts, axis=0))
            vecs.contrast_meta[vector_key] = {
                "target_class": cls,
                "rival_classes": list(rivals),
                "ablation": ablation,
                "pair": pair,
            }
        out[pol] = vecs
    return out


def _extract_gender_layer_acts(
    model,
    tokenizer,
    df: pd.DataFrame,
    *,
    policy: str,
    layer: int,
    label_tokens: Mapping[str, str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    templates = df["masked"].astype(str).tolist()
    labels = df["label_class"].astype(str).tolist()
    targets = [
        label_tokens.get(lab, df["label"].astype(str).iloc[i] if "label" in df.columns else "he")
        for i, lab in enumerate(labels)
    ]
    extractor = _make_extractor(model, tokenizer, policy=policy, layer=layer)
    batch = _gender_fill_batch(tokenizer, templates, targets, max_length=max_length)
    return extract_activations_batched(
        extractor, batch, batch_size=batch_size, desc=f"caa enc L{layer} {policy}"
    )


def _extract_gender_all_layer_acts(
    model, tokenizer, df: pd.DataFrame, *, policy: str, layers: Sequence[int],
    label_tokens: Mapping[str, str], batch_size: int, max_length: int,
) -> Dict[int, np.ndarray]:
    templates = df["masked"].astype(str).tolist()
    labels = df["label_class"].astype(str).tolist()
    targets = [label_tokens.get(lab, df["label"].astype(str).iloc[i] if "label" in df.columns else "he")
               for i, lab in enumerate(labels)]
    extractor = _make_extractor(model, tokenizer, policy=policy, scope=SignalScope.layers(*[int(x) for x in layers]))
    batch = _gender_fill_batch(tokenizer, templates, targets, max_length=max_length)
    return extract_activations_by_layer_batched(
        extractor, batch, layers=layers, batch_size=batch_size, desc=f"caa enc all layers {policy}"
    )


def _make_neutral_extractor(
    model: nn.Module,
    tokenizer,
    *,
    policy: str,
    layer: Optional[int] = None,
    all_layers: bool = False,
    excluded_words: Optional[Sequence[str]] = None,
    scope: Optional[Any] = None,
) -> ActivationSignalExtractor:
    from neutral_protocol import make_nonpad_nonexcluded_selector

    selector = resolve_neutral_act_selector(policy)
    if selector is nonpad_token_selector:
        selector = make_nonpad_nonexcluded_selector(
            tokenizer=tokenizer,
            excluded_words=excluded_words,
            fallback_selector=nonpad_token_selector,
        )
    signal = Signal.activation(token_selector=selector)
    if scope is None:
        if layer is not None:
            scope = SignalScope.layer(int(layer))
        else:
            scope = SignalScope.layers()
    return ActivationSignalExtractor(
        model,
        signal=signal,
        scope=scope,
        tokenizer=tokenizer,
        aggregate_batch="none",
    )


def _extract_neutral_layer_acts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    policy: str,
    layer: int,
    batch_size: int,
    max_length: int,
    excluded_words: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Residual rows for neutrals.

    ``prediction`` / ``pre_prediction``: one row per non-pad token (not a
    document mean), with excluded class-token ids dropped. ``mean`` / ``last``:
    one row per text, matched to labeled pooling.
    """
    extractor = _make_neutral_extractor(
        model,
        tokenizer,
        policy=policy,
        layer=layer,
        excluded_words=excluded_words,
    )
    batch = _neutral_batch(tokenizer, texts, max_length=max_length)
    return extract_activations_batched(
        extractor, batch, batch_size=batch_size, desc=f"caa neu L{layer} {policy}"
    )


def _extract_neutral_all_layer_acts(
    model, tokenizer, texts: Sequence[str], *, policy: str, layers: Sequence[int],
    batch_size: int, max_length: int, excluded_words: Optional[Sequence[str]] = None,
) -> Dict[int, np.ndarray]:
    extractor = _make_neutral_extractor(
        model, tokenizer, policy=policy, all_layers=True, excluded_words=excluded_words,
    )
    batch = _neutral_batch(tokenizer, texts, max_length=max_length)
    return extract_activations_by_layer_batched(
        extractor, batch, layers=layers, batch_size=batch_size, desc=f"caa neu all layers {policy}"
    )


def encode_caa_scores(
    model: nn.Module,
    tokenizer,
    gender_df: pd.DataFrame,
    neutral_df: pd.DataFrame,
    vectors: CaaVectors,
    *,
    target_classes: Sequence[str],
    eval_split: str = "test",
    val_split: str = "validation",
    max_eval: Optional[int] = None,
    max_neutral: Optional[int] = None,
    batch_size: int = 8,
    max_length: int = CAA_DEFAULT_MAX_LENGTH,
    label_tokens: Optional[Mapping[str, str]] = None,
    n_bootstrap: int = 100,
    excluded_words: Optional[Sequence[str]] = None,
    prior_val_readouts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Cosine encoding metrics for one act policy (concat / :all / L*).

    Youden τ is fit on ``val_split`` labeled + val neutrals and applied to
    ``eval_split`` (test) spec/excl/J.

    ``prior_val_readouts`` (test-only migration): ``{method_id: val_readout}`` from
    an earlier encode of the *same vectors*. When every method scored on test has a
    usable one, the validation split is not scored at all -- the stored readout is
    deterministic given the vectors, so its frozen Spec_n/Excl rules and the
    ``val_readout`` the site lock ranks on are reused verbatim. If any method lacks
    one, validation is scored in full as usual.
    """
    from neutral_protocol import texts_for_split

    label_tokens = dict(label_tokens or {"M": "he", "F": "she"})
    classes = [str(c) for c in target_classes]
    eval_df = _subset_df(gender_df, eval_split, max_eval, label_col="label_class")
    val_df = _subset_df(gender_df, val_split, max_eval, label_col="label_class")
    neu_test = texts_for_split(neutral_df, eval_split, max_rows=max_neutral, fallback="all")
    neu_val = texts_for_split(neutral_df, val_split, max_rows=max_neutral, fallback="all")
    if eval_df.empty or not neu_test:
        return {"error": "empty eval or neutral set", "method_metrics": {}}

    labels = eval_df["label_class"].astype(str).to_numpy()
    present = set(labels.tolist())
    missing = [c for c in classes if c not in present]
    if missing:
        print(
            f"  CAA: skip absent label_class {missing} "
            f"(present={sorted(present)}; expand one-pole CF rows for CF poles)",
            flush=True,
        )
        classes = [c for c in classes if c in present]
    if not classes:
        return {
            "error": f"no target classes present after subset (had {list(target_classes)})",
            "method_metrics": {},
        }
    policy = vectors.policy
    layers = vectors.layers
    contrasts = vectors.contrasts()
    contrasts = [
        (key, meta)
        for key, meta in contrasts
        if str(meta.get("target_class")) in classes
    ]
    if not contrasts:
        return {"error": "no CAA contrasts remain after class filtering", "method_metrics": {}}
    per_token_neutral = resolve_neutral_act_selector(policy) is nonpad_token_selector

    def _score_side(df: pd.DataFrame, neu_texts: Sequence[str], *, desc: str):
        if df is None or df.empty or not neu_texts:
            return None
        side_labels = df["label_class"].astype(str).to_numpy()
        layer_acts = _extract_gender_all_layer_acts(
            model, tokenizer, df, policy=policy, layers=layers,
            label_tokens=label_tokens, batch_size=batch_size, max_length=max_length,
        )
        concat_acts = np.concatenate([layer_acts[L] for L in layers], axis=1)
        neu_concat_parts: Dict[str, List[np.ndarray]] = {key: [] for key, _ in contrasts}
        neu_layer_parts: Dict[str, Dict[int, List[np.ndarray]]] = {
            key: {L: [] for L in layers} for key, _ in contrasts
        }
        n_tok = 0
        bs = max(1, int(batch_size))
        n_chunks = (len(neu_texts) + bs - 1) // bs
        for start in tqdm(
            range(0, len(neu_texts), bs),
            desc=desc,
            leave=False,
            total=n_chunks,
        ):
            if per_token_neutral and n_tok >= MAX_NEUTRAL_TOKEN_ROWS:
                break
            chunk = list(neu_texts[start : start + bs])
            layer_tok: Dict[int, np.ndarray] = {}
            layer_tok = _extract_neutral_all_layer_acts(
                model, tokenizer, chunk, policy=policy, layers=layers,
                batch_size=bs, max_length=max_length, excluded_words=excluded_words,
            )
            n_rows = int(next(iter(layer_tok.values())).shape[0]) if layer_tok else 0
            if n_rows == 0:
                continue
            if per_token_neutral:
                remaining = int(MAX_NEUTRAL_TOKEN_ROWS) - int(n_tok)
                if remaining <= 0:
                    break
                if n_rows > remaining:
                    idx = subsample_row_indices(n_rows, remaining)
                    if idx is not None:
                        for L in layers:
                            layer_tok[L] = layer_tok[L][idx]
                        n_rows = int(idx.shape[0])
            n_tok += n_rows
            concat_neu = np.concatenate([layer_tok[L] for L in layers], axis=1)
            for vector_key, _meta in contrasts:
                neu_concat_parts[vector_key].append(
                    cosine_score(concat_neu, vectors.concat_vec(vector_key))
                )
                for L in layers:
                    neu_layer_parts[vector_key][L].append(
                        cosine_score(layer_tok[L], vectors.layer_vec(vector_key, L))
                    )
            del layer_tok, concat_neu

        def _cat(parts: List[np.ndarray]) -> np.ndarray:
            if not parts:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(parts, axis=0)

        scored: Dict[str, Dict[str, Any]] = {}
        for vector_key, meta in contrasts:
            cls = str(meta["target_class"])
            pair = meta.get("pair")
            mid = caa_encoder_id(cls, policy=policy, part=None, pair=pair)
            s_cls = cosine_score(concat_acts, vectors.concat_vec(vector_key))
            scored[mid] = {
                "cls": cls,
                "vector_key": vector_key,
                "ablation": meta.get("ablation"),
                "pair": pair,
                "part": None,
                "score_name": "cosine",
                "pos": s_cls[side_labels == cls],
                # Neutrals are keyed by vector_key, not cls: for a two-pole pair
                # the vector_key is the pair ("F-M") while cls is one pole ("F").
                # The mean-path below already uses vector_key; match it here.
                "neu": _cat(neu_concat_parts[vector_key]),
                "other_by": {o: s_cls[side_labels == o] for o in classes if o != cls},
            }
            layer_scores = [
                cosine_score(layer_acts[L], vectors.layer_vec(vector_key, L)) for L in layers
            ]
            layer_neu_scores = [_cat(neu_layer_parts[vector_key][L]) for L in layers]
            s_mean = np.mean(np.stack(layer_scores, axis=0), axis=0)
            s_neu_mean = np.mean(np.stack(layer_neu_scores, axis=0), axis=0)
            scored[caa_encoder_id(cls, policy=policy, part="all", pair=pair)] = {
                "cls": cls,
                "vector_key": vector_key,
                "ablation": meta.get("ablation"),
                "pair": pair,
                "part": "all",
                "score_name": "cosine_mean_layers",
                "pos": s_mean[side_labels == cls],
                "neu": s_neu_mean,
                "other_by": {o: s_mean[side_labels == o] for o in classes if o != cls},
            }
            for L in layers:
                s_L = cosine_score(layer_acts[L], vectors.layer_vec(vector_key, L))
                scored[caa_encoder_id(cls, policy=policy, part=f"L{L}", pair=pair)] = {
                    "cls": cls,
                    "vector_key": vector_key,
                    "ablation": meta.get("ablation"),
                    "pair": pair,
                    "part": f"L{L}",
                    "score_name": "cosine",
                    "pos": s_L[side_labels == cls],
                    "neu": _cat(neu_layer_parts[vector_key][L]),
                    "other_by": {o: s_L[side_labels == o] for o in classes if o != cls},
                }
        del layer_acts, concat_acts
        return scored, n_tok, len(neu_texts)

    test_pack = _score_side(eval_df, neu_test, desc=f"caa neu {policy}")
    if test_pack is None:
        return {"error": "empty eval or neutral set", "method_metrics": {}}
    test_scored, n_neutral_tokens, n_neu_texts = test_pack

    def _prior_val_readout(mid: str) -> Optional[Mapping[str, Any]]:
        vr = (prior_val_readouts or {}).get(mid)
        # Usable = carries a frozen neutral rule; anything else must be rescored.
        return vr if isinstance(vr, Mapping) and frozen_decision_rules(vr) else None

    use_prior_val = bool(prior_val_readouts) and all(
        _prior_val_readout(mid) is not None for mid in test_scored
    )
    val_pack = None
    if use_prior_val:
        print(
            f"  CAA act={policy}: reusing prior validation readouts for "
            f"{len(test_scored)} method(s); validation split not scored",
            flush=True,
        )
    elif not val_df.empty and neu_val:
        val_pack = _score_side(val_df, neu_val, desc=f"caa neu val {policy}")
    val_scored = val_pack[0] if val_pack is not None else {}

    method_metrics: Dict[str, Dict[str, Any]] = {}

    def _metrics_for(mid: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        cls = payload["cls"]
        other_by = payload["other_by"]
        other = next(iter(other_by), None)
        tau = None
        frozen: Dict[str, Any] = {}
        val_readout = None
        prior_vr = _prior_val_readout(mid) if use_prior_val else None
        if prior_vr is not None:
            tau = prior_vr.get("youden_threshold")
            frozen = frozen_decision_rules(prior_vr)
            val_readout = dict(prior_vr)
        elif mid in val_scored:
            vp = val_scored[mid]
            v_other = vp["other_by"]
            v_o = next(iter(v_other), None)
            v_rd = class_vs_neutral_metrics(
                vp["pos"],
                vp["neu"],
                target_class=cls,
                scores_other=v_other.get(v_o) if v_o else None,
                other_class=v_o,
                scores_other_by_class=v_other,
                n_bootstrap=0,
            )
            tau = v_rd.get("youden_threshold")
            # Spec_n / Excl decision rules (threshold + orientation) are frozen on
            # validation and applied unchanged to test; ``tau`` above is only the
            # pooled diagnostic and does not feed either metric.
            frozen = frozen_decision_rules(v_rd)
            # Use the exact same validation-only bottleneck used to lock SAE
            # sites.  Previously CAA kept only the validation Youden threshold,
            # so the headline selector silently fell back to test E.
            # Keep the whole validation readout, not one derived scalar: any
            # selector is then computable from it without a GPU re-score, which
            # is what persisting a single derived number previously prevented.
            val_readout = dict(v_rd)
        rd = class_vs_neutral_metrics(
            payload["pos"],
            payload["neu"],
            target_class=cls,
            scores_other=other_by.get(other) if other else None,
            other_class=other,
            scores_other_by_class=other_by,
            n_bootstrap=n_bootstrap,
            youden_threshold=tau,
            **frozen,
            extras={
                "backend": "caa",
                "act_policy": policy,
                "score": payload["score_name"],
                "component_part": payload["part"],
                "ablation": payload.get("ablation"),
                "pair": payload.get("pair"),
                "vector_key": payload.get("vector_key"),
            },
        )
        rd["act_policy"] = policy
        rd["score"] = payload["score_name"]
        rd["component_part"] = payload["part"]
        rd["ablation"] = payload.get("ablation")
        rd["pair"] = payload.get("pair")
        rd["vector_key"] = payload.get("vector_key")
        rd["val_readout"] = val_readout
        rd["val_readout_source"] = "prior" if prior_vr is not None else "scored"
        rd["n_neutral_texts"] = int(n_neu_texts)
        rd["neutral_unit"] = (
            "token"
            if resolve_neutral_act_selector(policy) is nonpad_token_selector
            else "text"
        )
        return rd

    for mid, payload in test_scored.items():
        method_metrics[mid] = _metrics_for(mid, payload)
    # A method without a usable validation readout would silently get a test-fit
    # (oracle) Spec_n/Excl; that is an error, not a fallback.
    require_frozen_rules(method_metrics, context=f"CAA act={policy} encode (test)")

    return {
        "policy": policy,
        "layers": list(layers),
        "n_eval": int(len(eval_df)),
        "n_neutral": int(n_neu_texts),
        "n_neutral_tokens": int(n_neutral_tokens),
        "method_metrics": method_metrics,
    }


def make_multi_activation_intervention(
    specs: Sequence[Tuple[str, torch.Tensor]],
    strength: float,
    *,
    token_selector: str = "all",
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Build multi-site residual steering (classic multi-layer CAA)."""
    interventions: List[Dict[str, Any]] = []
    tensors: Dict[str, torch.Tensor] = {}
    for i, (module_path, direction) in enumerate(specs):
        key = f"steering_{i}"
        vec = (float(strength) * direction.flatten()).detach().float().cpu()
        interventions.append(
            {
                "module": module_path,
                "tensor_key": key,
                "application": {"axis": "last_dim", "token_selector": token_selector},
            }
        )
        tensors[key] = vec
    return interventions, tensors


@contextmanager
def caa_steering_context(
    model,
    specs: Sequence[Tuple[str, torch.Tensor]],
    strength: float,
    *,
    token_selector: str = "all",
) -> Iterator[Any]:
    from gradiend.model.modified import apply_activation_steering, remove_hook_handles

    interventions, tensors = make_multi_activation_intervention(
        specs, strength, token_selector=token_selector
    )
    apply_activation_steering(model, interventions=interventions, tensors=tensors)
    handles = getattr(model, "_gradiend_modified_hook_handles", None) or []
    try:
        yield model
    finally:
        remove_hook_handles(handles)
        for attr in (
            "_gradiend_modified_config",
            "_gradiend_modified_tensors",
            "_gradiend_modified_hook_handles",
        ):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception:
                    pass


def run_caa_causal_sweep(
    model,
    tokenizer,
    meta_rows: Sequence[Dict[str, str]],
    *,
    specs: Sequence[Tuple[str, torch.Tensor]],
    target_class: str,
    strengths: Sequence[float] = DEFAULT_CAUSAL_STRENGTHS,
    lms_texts: Optional[Sequence[str]] = None,
    include_random_control: bool = True,
    random_seed: int = 0,
    method_id: Optional[str] = None,
    token_selector: str = "all",
    act_policy: str = "prediction",
    part: Optional[str] = None,
    report_rows: Optional[Sequence[Dict[str, str]]] = None,
    base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    base_lms: Optional[Any] = None,
    r_base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    r_base_lms: Optional[Any] = None,
    reuse_strengthen: Optional[CausalMethodResult | Mapping[str, Any]] = None,
    reuse_reselect_bidirectional: Optional[
        CausalMethodResult | Mapping[str, Any]
    ] = None,
) -> CausalMethodResult:
    """LMS-gated CAA strength sweep (single- or multi-layer residual add).

    ``base_probs``/``base_lms`` (and ``r_base_probs``/``r_base_lms`` for the
    ``report_rows`` split) are the unsteered model's scores on ``meta_rows``/
    ``report_rows`` — identical across every part/policy/class swept against
    the same rows. Callers looping many CAA methods over one fixed
    ``meta_rows``/``report_rows`` pair should compute these once and pass
    them in instead of paying a fresh unsteered forward pass per call.
    """
    texts = [r["text"] for r in meta_rows]
    lms_texts = list(
        lms_texts or [r["text"] for r in meta_rows if r["group"] == "neutral"] or texts
    )
    if reuse_strengthen is not None and reuse_reselect_bidirectional is not None:
        raise ValueError(
            "reuse_strengthen and reuse_reselect_bidirectional are mutually exclusive"
        )
    if (
        reuse_reselect_bidirectional is None
        and (base_probs is None or base_lms is None)
    ):
        frame = meta_rows_to_frame(meta_rows)
        base_probs = score_class_probs_by_dataset(model, tokenizer, frame)
        base_lms = compute_lms_safe(model, tokenizer, lms_texts)

    dirs = [(m, d.detach().float().cpu().flatten()) for m, d in specs]
    reused = (
        causal_method_result_from_dict(reuse_strengthen)
        if isinstance(reuse_strengthen, Mapping)
        else reuse_strengthen
    )
    repair_source = (
        causal_method_result_from_dict(reuse_reselect_bidirectional)
        if isinstance(reuse_reselect_bidirectional, Mapping)
        else reuse_reselect_bidirectional
    )
    results = []
    validation_selected = None
    weaken_results = []
    weaken_validation_selected = None
    if repair_source is not None:
        (
            results,
            selected,
            weaken_results,
            weaken_validation_selected,
        ) = reselect_direct_bidirectional_curves(
            repair_source,
            target_class=target_class,
        )
        validation_selected = selected
    elif reused is None:
        for sign in (1.0, -1.0):
            signed = [(m, d * float(sign)) for m, d in dirs]
            for s in strengths:
                with caa_steering_context(
                    model, signed, float(s), token_selector=token_selector
                ):
                    results.append(
                        run_strength_on_model(
                            model,
                            tokenizer,
                            meta_rows,
                            target_class=target_class,
                            strength=float(s),
                            base_probs=base_probs,
                            base_lms=base_lms,
                            lms_texts=lms_texts,
                            notes=(
                                f"caa act={act_policy} part={part} sign={sign:g} "
                                f"token_selector={token_selector} n_sites={len(dirs)}"
                            ),
                        )
                    )

    def _run_one(sign: float, s: float) -> StrengthResult:
        signed_s = [(m, d * float(sign)) for m, d in dirs]
        with caa_steering_context(
            model, signed_s, float(s), token_selector=token_selector
        ):
            return run_strength_on_model(
                model,
                tokenizer,
                meta_rows,
                target_class=target_class,
                strength=float(s),
                base_probs=base_probs,
                base_lms=base_lms,
                lms_texts=lms_texts,
                notes=(
                    f"caa act={act_policy} part={part} sign={sign:g} "
                    f"token_selector={token_selector} n_sites={len(dirs)} refine=1"
                ),
            )

    if repair_source is not None:
        pass
    elif reused is None:
        results, selected = refine_and_select_lms_gated(results, _run_one)
        validation_selected = selected
    else:
        if reused.selected is None or not reused.strengths:
            raise ValueError("cannot reuse incomplete CAA strengthen result")
        results = list(reused.strengths)
        selected = reused.selected
        validation_selected = None
    if repair_source is None:
        weaken_results, weaken_validation_selected = refine_and_select_weaken_lms_gated(
            results,
            _run_one,
            target_class=target_class,
        )
    selected_sign = 1.0
    if selected is not None and "sign=" in (selected.notes or ""):
        try:
            selected_sign = float(selected.notes.split("sign=")[1].split()[0])
        except Exception:
            selected_sign = 1.0

    random_ctrl = None
    if include_random_control and selected is not None and dirs and not report_rows:
        rng = np.random.default_rng(random_seed)
        rand_specs = []
        for m, d in dirs:
            r = torch.tensor(rng.normal(size=d.numel()), dtype=torch.float32)
            r = r * (d.norm() / (r.norm() + 1e-8))
            rand_specs.append((m, r * float(selected_sign)))
        with caa_steering_context(
            model, rand_specs, float(selected.strength), token_selector=token_selector
        ):
            random_ctrl = run_strength_on_model(
                model,
                tokenizer,
                meta_rows,
                target_class=target_class,
                strength=float(selected.strength),
                base_probs=base_probs,
                base_lms=base_lms,
                lms_texts=lms_texts,
                is_random_control=True,
                notes="matched-L2 random CAA direction(s)",
            )

    if report_rows and (selected is not None or weaken_validation_selected is not None):
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        if r_base_probs is None or r_base_lms is None:
            r_frame = meta_rows_to_frame(report_rows)
            r_base_probs = score_class_probs_by_dataset(model, tokenizer, r_frame)
            r_base_lms = compute_lms_safe(model, tokenizer, r_lms)
    if (
        report_rows
        and selected is not None
        and dirs
        and (reused is None or repair_source is not None)
    ):
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        signed_sel = [(m, d * float(selected_sign)) for m, d in dirs]
        with caa_steering_context(
            model, signed_sel, float(selected.strength), token_selector=token_selector
        ):
            selected = run_strength_on_model(
                model,
                tokenizer,
                report_rows,
                target_class=target_class,
                strength=float(selected.strength),
                base_probs=r_base_probs,
                base_lms=r_base_lms,
                lms_texts=r_lms,
                notes=(selected.notes or "") + " report=test",
            )
        if include_random_control:
            rng = np.random.default_rng(random_seed)
            rand_specs = []
            for m, d in dirs:
                r = torch.tensor(rng.normal(size=d.numel()), dtype=torch.float32)
                r = r * (d.norm() / (r.norm() + 1e-8))
                rand_specs.append((m, r * float(selected_sign)))
            with caa_steering_context(
                model, rand_specs, float(selected.strength), token_selector=token_selector
            ):
                random_ctrl = run_strength_on_model(
                    model,
                    tokenizer,
                    report_rows,
                    target_class=target_class,
                    strength=float(selected.strength),
                    base_probs=r_base_probs,
                    base_lms=r_base_lms,
                    lms_texts=r_lms,
                    is_random_control=True,
                    notes="matched-L2 random CAA direction(s) report=test",
                )
    elif report_rows and selected is not None and dirs and include_random_control:
        rng = np.random.default_rng(random_seed)
        rand_specs = []
        for m, d in dirs:
            r = torch.tensor(rng.normal(size=d.numel()), dtype=torch.float32)
            r = r * (d.norm() / (r.norm() + 1e-8))
            rand_specs.append((m, r * float(selected_sign)))
        with caa_steering_context(
            model, rand_specs, float(selected.strength), token_selector=token_selector
        ):
            random_ctrl = run_strength_on_model(
                model,
                tokenizer,
                report_rows,
                target_class=target_class,
                strength=float(selected.strength),
                base_probs=r_base_probs,
                base_lms=r_base_lms,
                lms_texts=r_lms,
                is_random_control=True,
                notes="matched-L2 random CAA direction(s) report=test repair=1",
            )

    weaken_selected = weaken_validation_selected
    if report_rows and weaken_validation_selected is not None and dirs:
        weaken_sign = 1.0
        if "sign=" in (weaken_validation_selected.notes or ""):
            weaken_sign = float(
                weaken_validation_selected.notes.split("sign=")[1].split()[0]
            )
        weaken_specs = [(m, d * weaken_sign) for m, d in dirs]
        with caa_steering_context(
            model,
            weaken_specs,
            float(weaken_validation_selected.strength),
            token_selector=token_selector,
        ):
            weaken_raw = run_strength_on_model(
                model,
                tokenizer,
                report_rows,
                target_class=target_class,
                strength=float(weaken_validation_selected.strength),
                base_probs=r_base_probs,
                base_lms=r_base_lms,
                lms_texts=r_lms,
                notes=(weaken_validation_selected.notes or "") + " report=test",
            )
        weaken_selected = as_weaken_strength(
            weaken_raw, target_class=target_class
        )

    mid = method_id or f"caa:{target_class}"
    return CausalMethodResult(
        method=mid,
        backend="caa",
        target_class=str(target_class),
        strengths=results,
        selected_strength=None if selected is None else selected.strength,
        selected=selected,
        modified_model_path=None,
        random_control=random_ctrl,
        weaken_strengths=weaken_results,
        weaken_selected_strength=(
            None if weaken_selected is None else weaken_selected.strength
        ),
        weaken_selected=weaken_selected,
        meta={
            **(dict(reused.meta) if reused is not None else {}),
            "act_policy": act_policy,
            "part": part,
            "token_selector": token_selector,
            "sign": selected_sign,
            "n_sites": len(dirs),
            "modules": [m for m, _ in dirs],
            "source": "caa_mean_diff",
            "metric": "delta_P(target)_on_other_dataset",
            "score": "cosine",
            "selection_split": "validation" if report_rows else "eval",
            "report_split": "test" if report_rows else "eval",
            "strengthen_reused": reused is not None,
            "validation_grid_reused": repair_source is not None,
            **(
                {}
                if reused is not None and repair_source is None
                else selection_lms_metadata(validation_selected)
            ),
            **{
                f"weaken_{k}": v
                for k, v in selection_lms_metadata(
                    weaken_validation_selected
                ).items()
            },
        },
    )


def run_caa_study(
    model: nn.Module,
    tokenizer,
    gender_df: pd.DataFrame,
    neutral_df: pd.DataFrame,
    *,
    target_classes: Sequence[str] = ("M", "F"),
    layers: Sequence[int],
    policies: Sequence[str] = CAA_ACT_POLICIES,
    hf_resid_template: str,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
    max_neutral: Optional[int] = None,
    batch_size: int = 8,
    n_bootstrap: int = 100,
    label_tokens: Optional[Mapping[str, str]] = None,
    excluded_words: Optional[Sequence[str]] = None,
    output_dir: Optional[Any] = None,
    include_pair: bool = True,
    include_one_pole: bool = True,
    stream_fit: bool = False,
    prior_vectors_payload: Optional[Mapping[str, Any]] = None,
    prior_val_readouts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fit all policies + encode; returns vectors + per-method metrics.

    ``prior_vectors_payload`` / ``prior_val_readouts`` (test-only migration): the
    ``vectors`` payload and ``{method_id: val_readout}`` of an earlier run over the
    same layers/policies. The vectors are reloaded instead of refit (no pass over
    the training rows) and validation is not rescored. Any policy or layer set that
    the payload does not cover falls back to a full fit + encode.
    """
    from cost_timer import cost_timer

    fitted = None
    if prior_vectors_payload:
        loaded = load_caa_vectors_payload(prior_vectors_payload)
        want_layers = tuple(int(L) for L in layers)
        if all(
            str(p) in loaded and tuple(loaded[str(p)].layers) == want_layers
            for p in policies
        ):
            fitted = {str(p): loaded[str(p)] for p in policies}
            print(
                f"  CAA: reusing {len(fitted)} stored policy vector set(s); fit skipped",
                flush=True,
            )
    if fitted is None:
        prior_val_readouts = None  # vectors changed or absent: stored val is not comparable
        with cost_timer(
            "caa_fit",
            phase="feature_select",
            backend="caa",
            ledger_unit="caa_sweep",
            policies=list(policies),
        ):
            fitted = fit_caa_vectors(
                model,
                tokenizer,
                gender_df,
                target_classes=target_classes,
                layers=layers,
                policies=policies,
                max_train=max_train,
                batch_size=batch_size,
                hf_resid_template=hf_resid_template,
                label_tokens=label_tokens,
                output_dir=output_dir,
                include_pair=include_pair,
                include_one_pole=include_one_pole,
                stream_fit=stream_fit,
            )
    all_metrics: Dict[str, Any] = {}
    per_policy_encode: Dict[str, Any] = {}
    for pol, vecs in fitted.items():
        print(f"  CAA encode act={pol} …", flush=True)
        with cost_timer(
            "caa_encode",
            phase="encode",
            backend="caa",
            ledger_unit="caa_sweep",
            policy=str(pol),
        ):
            enc = encode_caa_scores(
                model,
                tokenizer,
                gender_df,
                neutral_df,
                vecs,
                target_classes=target_classes,
                max_eval=max_eval,
                max_neutral=max_neutral,
                batch_size=batch_size,
                n_bootstrap=n_bootstrap,
                label_tokens=label_tokens,
                excluded_words=excluded_words,
                prior_val_readouts=prior_val_readouts,
            )
        per_policy_encode[pol] = {
            k: v for k, v in enc.items() if k != "method_metrics"
        }
        all_metrics.update(enc.get("method_metrics") or {})
        try:
            from study.progress import write_pipeline_progress

            write_pipeline_progress(
                output_dir,
                stage="caa_encode",
                policy=str(pol),
            )
        except Exception:
            pass

    # Serialize vectors lightly for causal stage.
    vectors_payload: Dict[str, Any] = {}
    for pol, vecs in fitted.items():
        vectors_payload[pol] = {
            "layers": list(vecs.layers),
            "module_by_layer": {str(k): v for k, v in vecs.module_by_layer.items()},
            "by_layer": {
                cls: {str(L): vecs.by_layer[cls][L].tolist() for L in vecs.layers}
                for cls in vecs.by_layer
            },
            "by_concat": {cls: vecs.by_concat[cls].tolist() for cls in vecs.by_concat},
            "contrast_meta": vecs.contrast_meta,
        }

    return {
        "method_metrics": all_metrics,
        "per_policy": per_policy_encode,
        "vectors": vectors_payload,
        "layers": list(layers),
        "policies": [str(p) for p in policies],
        "pole_regimes": [
            name
            for name, enabled in (("pair", include_pair), ("one_pole", include_one_pole))
            if enabled
        ],
    }


def load_caa_vectors_payload(payload: Mapping[str, Any]) -> Dict[str, CaaVectors]:
    """Rehydrate :class:`CaaVectors` from ``run_caa_study`` JSON-friendly payload."""
    out: Dict[str, CaaVectors] = {}
    for pol, blob in (payload or {}).items():
        layers = tuple(int(L) for L in (blob.get("layers") or ()))
        module_by_layer = {
            int(k): str(v) for k, v in (blob.get("module_by_layer") or {}).items()
        }
        by_layer: Dict[str, Dict[int, np.ndarray]] = {}
        for cls, layer_map in (blob.get("by_layer") or {}).items():
            by_layer[str(cls)] = {
                int(L): np.asarray(vec, dtype=np.float64) for L, vec in layer_map.items()
            }
        by_concat = {
            str(cls): np.asarray(vec, dtype=np.float64)
            for cls, vec in (blob.get("by_concat") or {}).items()
        }
        contrast_meta = {
            str(k): dict(v) for k, v in (blob.get("contrast_meta") or {}).items()
        }
        out[str(pol)] = CaaVectors(
            policy=str(pol),
            layers=layers,
            by_layer=by_layer,
            by_concat=by_concat,
            module_by_layer=module_by_layer,
            contrast_meta=contrast_meta,
        )
    return out


def specs_for_part(
    vectors: CaaVectors,
    vector_key: str,
    part: Optional[str],
) -> List[Tuple[str, torch.Tensor]]:
    """Build ``(module_path, direction)`` list for causal steering."""
    key = str(vector_key)
    if part is None or part == "all":
        return [
            (
                vectors.module_by_layer[L],
                torch.tensor(vectors.layer_vec(key, L), dtype=torch.float32),
            )
            for L in vectors.layers
        ]
    if str(part).startswith("L") and str(part)[1:].isdigit():
        L = int(str(part)[1:])
        return [
            (
                vectors.module_by_layer[L],
                torch.tensor(vectors.layer_vec(key, L), dtype=torch.float32),
            )
        ]
    raise ValueError(f"Unknown CAA causal part {part!r}")
