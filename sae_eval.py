"""
SAE feature identification, specificity, and causal steering.

Primary study design (gender EN):
  - GRADIEND/ACTIEND: one joint bipolar gender feature
  - SAE ``per_class``: one (or k) feature(s) per class; readout = z_M − z_F
  - SAE ``joint``: single best contrastive feature (ablation / closest 1-feature analogue)

Additional ablation ranking modes (top-k bags): pairwise, one_vs_rest, class_diff,
pair_aware, dense_probe — see SELECTION_MODES.

Shared: min firing rate, optional neutral specificity, validation selection,
test-set fair metrics (+ k-curve / bootstrap).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from results_schema import unfrozen_rule_readouts

GPT2_SMALL_SAE_RELEASE = "gpt2-small-resid-post-v5-32k"






def resid_sites(layer: int) -> tuple[str, str]:
    """Active study-model resid hook sites (see ``study_models``)."""
    from study_models import resid_sites as _sites

    return _sites(int(layer))


def active_sae_release() -> str:
    try:
        from study_models import active_model

        return active_model().sae_release
    except Exception:
        return GPT2_SMALL_SAE_RELEASE


def sae_encode_classes(sae_raw: Mapping[str, Any] | None) -> List[str]:
    """Class ids SAE encode actually scored (non-empty feature selections).

    One-pole circuit tasks only encode factual poles (e.g. MATCH); CF partners
    such as DISTRACTOR share the masked left-context and must not get causal rows.
    """
    if not sae_raw or sae_raw.get("error"):
        return []
    keys: set[str] = set()
    sel = sae_raw.get("selections") or {}
    per_class = sel.get("per_class") or {}
    for branch_name in (
        "features_by_class",
        "features_by_class_opp_fire",
        "features_by_class_arad_out",
        "features_by_class_jh_f1",
    ):
        branch = per_class.get(branch_name) or {}
        if isinstance(branch, dict):
            for k, v in branch.items():
                if v:
                    keys.add(str(k))
    slbc = sae_raw.get("selected_layer_by_class") or {}
    if isinstance(slbc, dict):
        keys.update(str(k) for k in slbc)
    for layer_raw in (sae_raw.get("by_layer") or {}).values():
        if not isinstance(layer_raw, dict):
            continue
        fbc = (
            ((layer_raw.get("selections") or {}).get("per_class") or {}).get(
                "features_by_class"
            )
            or {}
        )
        for k, v in fbc.items():
            if v:
                keys.add(str(k))
    return sorted(keys)


def load_sae(release: str, sae_id: str, device: str = "cpu"):
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise ImportError("Install sae-lens: pip install sae-lens") from exc
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def sae_decoder_direction(sae, feature_index: int) -> torch.Tensor:
    if hasattr(sae, "W_dec"):
        w = sae.W_dec
        if w.ndim == 2 and w.shape[0] > w.shape[1]:
            return w[feature_index].detach()
        if w.ndim == 2:
            return w[:, feature_index].detach()
    raise AttributeError("Could not read SAE decoder direction from W_dec")












def collect_study_activations(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    *,
    layer: int,
    act_policy: str = "prediction",
    label_tokens: Optional[Mapping[str, str]] = None,
    batch_size: int = 8,
    max_length: int = 256,
) -> np.ndarray:
    """Residual extract matching CAA/ACTIEND (filled prediction / pre_prediction / …)."""
    from caa_eval import _extract_gender_layer_acts

    return _extract_gender_layer_acts(
        model,
        tokenizer,
        df,
        policy=str(act_policy),
        layer=int(layer),
        label_tokens=dict(label_tokens or {}),
        batch_size=int(batch_size),
        max_length=int(max_length),
    )


def collect_study_neutral_activations(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    layer: int,
    act_policy: str = "prediction",
    batch_size: int = 8,
    max_length: int = 256,
    excluded_words: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Neutral residuals. ``prediction`` / ``pre_prediction`` → one row per non-pad token."""
    from caa_eval import _extract_neutral_layer_acts

    return _extract_neutral_layer_acts(
        model,
        tokenizer,
        texts,
        policy=str(act_policy),
        layer=int(layer),
        batch_size=int(batch_size),
        max_length=int(max_length),
        excluded_words=excluded_words,
    )


def encode_sae(sae, hidden: np.ndarray, batch_size: int = 64) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, hidden.shape[0], batch_size):
            batch = torch.tensor(hidden[start : start + batch_size], device=sae.device, dtype=sae.dtype)
            parts.append(sae.encode(batch).detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


# Per-token neutrals can be tens of thousands of rows. Cap stored SAE latent matrices
# and CAA neutral scoring rows. Selection/metrics still use token rows, not a
# document mean. Seed is shared across layers so ``all_k`` sums the same token
# indices (not a per-layer shuffle).
MAX_NEUTRAL_TOKEN_ROWS = 16384
NEUTRAL_TOKEN_SUBSAMPLE_SEED = 0


def subsample_row_indices(
    n_rows: int, max_rows: int, *, seed: int = NEUTRAL_TOKEN_SUBSAMPLE_SEED
) -> Optional[np.ndarray]:
    """Sorted keep-indices, or None when no subsample is needed."""
    n = int(n_rows)
    cap = int(max_rows)
    if cap < 1 or n <= cap:
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=cap, replace=False))


def subsample_rows(
    x: np.ndarray,
    max_rows: int,
    *,
    seed: int = NEUTRAL_TOKEN_SUBSAMPLE_SEED,
) -> np.ndarray:
    """Deterministic row subsample; no-op when already small."""
    arr = np.asarray(x)
    idx = subsample_row_indices(arr.shape[0], max_rows, seed=seed)
    return arr if idx is None else arr[idx]


_SAE_LAYER_HEAVY_KEYS = (
    "_sae",
    "_test_latents",
    "_neutral_latents",
    "_val_neutral_latents",
    "_val_latents",
    "_val_labels",
    "_test_labels",
    "_all_k_compact",
)


def release_sae_layer_device_memory(
    layer_raw: Optional[Mapping[str, Any]], *, empty_cuda_cache: bool = False
) -> None:
    """Release a completed layer's SAE module while keeping CPU selection data.

    Cross-layer selection still needs ``_all_k_compact`` until every layer has
    been processed, but it never needs the live SAE module again.  Holding one
    module per layer is especially costly for large SAEs and used to make the
    layer sweep OOM progressively.
    """
    import gc

    if not isinstance(layer_raw, MutableMapping):
        return
    layer_raw.pop("_sae", None)
    gc.collect()
    if empty_cuda_cache:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Cleanup is best-effort and must never invalidate completed data.
            pass


def compact_all_k_layer_slices(
    *,
    test_latents: Optional[np.ndarray],
    val_latents: Optional[np.ndarray],
    neutral_test: Optional[np.ndarray],
    neutral_val: Optional[np.ndarray],
    test_labels: Sequence[str],
    val_labels: Sequence[str],
    features_by_class: Mapping[str, Sequence[int]],
    classes: Sequence[str],
    max_k: int,
    rival_test_latents_by_class: Optional[Mapping[str, np.ndarray]] = None,
    rival_val_latents_by_class: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Keep only top-``max_k`` feature columns per class (for all_k metrics).

    One-pole tasks have only the claimed class in their factual frame.  Their
    counterfactual rival activations therefore have to travel separately into
    the all-layer sum; otherwise that sum can report AUC_n but silently loses
    AUC_o / Excl despite the per-layer evaluator having computed them.
    """
    kcap = max(1, int(max_k))
    by_class: Dict[str, Dict[str, Any]] = {}
    for cls in classes:
        cls = str(cls)
        feats = [int(f) for f in (features_by_class.get(cls) or [])]
        if not feats:
            continue
        n_keep = min(kcap, len(feats))
        idx = feats[:n_keep]
        entry: Dict[str, Any] = {"feature_indices": idx}
        if test_latents is not None:
            entry["test_cols"] = np.asarray(test_latents, dtype=np.float32)[:, idx]
            entry["test_labels"] = [str(x) for x in test_labels]
        if val_latents is not None:
            entry["val_cols"] = np.asarray(val_latents, dtype=np.float32)[:, idx]
            entry["val_labels"] = [str(x) for x in val_labels]
        if neutral_test is not None:
            entry["neutral_test_cols"] = np.asarray(neutral_test, dtype=np.float32)[:, idx]
        if neutral_val is not None:
            entry["neutral_val_cols"] = np.asarray(neutral_val, dtype=np.float32)[:, idx]
        if rival_test_latents_by_class:
            entry["rival_test_cols_by_class"] = {
                str(other): np.asarray(latents, dtype=np.float32)[:, idx]
                for other, latents in rival_test_latents_by_class.items()
                if latents is not None and len(latents)
            }
        if rival_val_latents_by_class:
            entry["rival_val_cols_by_class"] = {
                str(other): np.asarray(latents, dtype=np.float32)[:, idx]
                for other, latents in rival_val_latents_by_class.items()
                if latents is not None and len(latents)
            }
        by_class[cls] = entry
    return {"by_class": by_class}


def release_sae_layer_heavy_memory(layer_raw: Optional[Mapping[str, Any]]) -> None:
    """Drop per-layer SAE weights and latent matrices after cache/metrics are done."""
    if not isinstance(layer_raw, MutableMapping):
        return
    for key in _SAE_LAYER_HEAVY_KEYS:
        layer_raw.pop(key, None)


def release_sae_raw_heavy_memory(sae_raw: Optional[Mapping[str, Any]]) -> None:
    """Drop live SAE handles and any leftover layer tensors from ``sae_raw``."""
    import gc

    if not isinstance(sae_raw, MutableMapping):
        return
    sae_raw.pop("_sae_by_layer", None)
    nested = sae_raw.get("by_site")
    if isinstance(nested, MutableMapping):
        for payload in nested.values():
            if isinstance(payload, MutableMapping):
                payload.pop("_sae_by_layer", None)
    gc.collect()


SELECTION_MODES = (
    "pairwise",           # |μ_a − μ_b|
    "one_vs_rest",        # max_c |μ_c − μ_{¬c}|
    "class_diff",         # signed μ_a − μ_b (features favoring a over b)
    "pair_aware",         # mean |z_a − z_b| on factual/counterfactual pairs (pair_id)
    "dense_probe",        # |coef| of linear probe on SAE latents
)


@dataclass
class SAEFeatureSelection:
    feature_indices: List[int]
    scores: Dict[int, float]
    mean_by_class: Dict[str, Dict[int, float]]
    contrast_pair: Tuple[str, str]
    mode: str
    layer: int
    sae_id: str
    selection_split: str = "validation"
    used_neutral_specificity: bool = False
    neutral_means: Dict[int, float] = field(default_factory=dict)
    bootstrap_stability: Dict[int, float] = field(default_factory=dict)
    # Populated for mode="per_class": features owned by each class (ordered).
    features_by_class: Dict[str, List[int]] = field(default_factory=dict)
    scores_by_class: Dict[str, Dict[int, float]] = field(default_factory=dict)


def _resolve_pair(
    labels: Sequence[str],
    class_a: Optional[str],
    class_b: Optional[str],
) -> Tuple[str, str]:
    labels_arr = np.asarray(labels)
    classes = list(pd.Series(labels_arr).value_counts().index)
    if class_a is None or class_b is None:
        if len(classes) < 2:
            raise ValueError(f"Need ≥2 classes; got {classes}")
        return str(classes[0]), str(classes[1])
    return class_a, class_b


def _class_means(latents: np.ndarray, labels: Sequence[str], classes: Sequence[str]) -> Dict[str, np.ndarray]:
    labels_arr = np.asarray(labels)
    means = {}
    for c in classes:
        mask = labels_arr == c
        if not mask.any():
            raise ValueError(f"No examples for class {c!r}")
        means[c] = latents[mask].mean(axis=0)
    return means


def score_pairwise(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, str]]:
    pair = _resolve_pair(labels, class_a, class_b)
    means = _class_means(latents, labels, pair)
    return np.abs(means[pair[0]] - means[pair[1]]), means, pair


def score_one_vs_rest(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, str]]:
    """Max over classes of |μ_c − μ_{rest}| (binary reduces to pairwise)."""
    labels_arr = np.asarray(labels)
    classes = [str(c) for c in pd.Series(labels_arr).value_counts().index]
    means = _class_means(latents, labels, classes)
    scores = np.zeros(latents.shape[1], dtype=np.float64)
    for c in classes:
        rest = labels_arr != c
        mu_rest = latents[rest].mean(axis=0) if rest.any() else np.zeros_like(means[c])
        scores = np.maximum(scores, np.abs(means[c] - mu_rest))
    pair = _resolve_pair(labels, class_a, class_b)
    return scores, means, pair


def score_class_diff(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, str]]:
    """Signed μ_a − μ_b: positive ⇒ prefers class_a. Rank by absolute value for top-k."""
    pair = _resolve_pair(labels, class_a, class_b)
    means = _class_means(latents, labels, pair)
    signed = means[pair[0]] - means[pair[1]]
    return signed, means, pair


def score_pair_aware(
    latents: np.ndarray,
    labels: Sequence[str],
    pair_ids: Sequence[Any],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, str]]:
    """Mean |z(c_a) − z(c_b)| over rows sharing pair_id (factual/counterfactual pairs)."""
    pair = _resolve_pair(labels, class_a, class_b)
    labels_arr = np.asarray(labels)
    pair_arr = np.asarray(pair_ids)
    means = _class_means(latents, labels, pair)

    diffs = []
    for pid in pd.unique(pair_arr):
        idx = np.where(pair_arr == pid)[0]
        a_idx = [i for i in idx if labels_arr[i] == pair[0]]
        b_idx = [i for i in idx if labels_arr[i] == pair[1]]
        if not a_idx or not b_idx:
            continue
        # average within class then absolute diff (handles multiple names per pair)
        za = latents[a_idx].mean(axis=0)
        zb = latents[b_idx].mean(axis=0)
        diffs.append(np.abs(za - zb))
    if not diffs:
        raise ValueError("pair_aware scoring needs matching pair_id rows for both classes")
    return np.mean(np.stack(diffs, axis=0), axis=0), means, pair


def score_dense_probe(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, str]]:
    """
    Dense linear probe: least-squares y∈{+1,-1} on SAE latents; rank by |coef|.
    (Simple dense readout over all SAE features.)
    """
    pair = _resolve_pair(labels, class_a, class_b)
    labels_arr = np.asarray(labels)
    mask = (labels_arr == pair[0]) | (labels_arr == pair[1])
    x = latents[mask]
    y = np.where(labels_arr[mask] == pair[0], 1.0, -1.0)
    # ridge-ish: (X'X + εI)^{-1} X'y
    d = x.shape[1]
    xtx = x.T @ x + 1e-3 * np.eye(d)
    coef = np.linalg.solve(xtx, x.T @ y)
    means = _class_means(latents, labels, pair)
    return coef, means, pair


_SCORE_FNS = {
    "pairwise": score_pairwise,
    "one_vs_rest": score_one_vs_rest,
    "class_diff": score_class_diff,
    "dense_probe": score_dense_probe,
}


def _apply_filters(
    raw_scores: np.ndarray,
    latents: np.ndarray,
    *,
    min_firing_rate: float,
    rank_by_abs: bool,
    neutral_latents: Optional[np.ndarray],
    neutral_specificity: bool,
    neutral_weight: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    firing = (latents > 0).mean(axis=0)
    scores = np.abs(raw_scores) if rank_by_abs else raw_scores.astype(np.float64).copy()
    scores[firing < min_firing_rate] = -np.inf
    neutral_mean = None
    used_neutral = False
    if neutral_latents is not None and len(neutral_latents) and neutral_specificity:
        neutral_mean = np.abs(neutral_latents.mean(axis=0))
        scores = scores - float(neutral_weight) * neutral_mean
        used_neutral = True
    return scores, neutral_mean, used_neutral


def select_features(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    mode: str = "pairwise",
    top_k: int = 20,
    min_firing_rate: float = 0.01,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
    pair_ids: Optional[Sequence[Any]] = None,
    neutral_latents: Optional[np.ndarray] = None,
    neutral_specificity: bool = False,
    neutral_weight: float = 1.0,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> SAEFeatureSelection:
    """
    Rank SAE features with a selectable scoring mode (see SELECTION_MODES).

    Shared options: min firing rate; optional neutral specificity penalty;
    optional bootstrap stability (fraction of resamples where feature is in top-k).
    """
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {SELECTION_MODES}")

    if mode == "pair_aware":
        if pair_ids is None:
            raise ValueError("mode='pair_aware' requires pair_ids")
        raw, means, pair = score_pair_aware(
            latents, labels, pair_ids, class_a=class_a, class_b=class_b
        )
        rank_by_abs = True
    else:
        raw, means, pair = _SCORE_FNS[mode](latents, labels, class_a=class_a, class_b=class_b)
        rank_by_abs = mode in {"pairwise", "one_vs_rest", "class_diff", "dense_probe"}

    scores, neutral_mean, used_neutral = _apply_filters(
        raw,
        latents,
        min_firing_rate=min_firing_rate,
        rank_by_abs=rank_by_abs,
        neutral_latents=neutral_latents,
        neutral_specificity=neutral_specificity,
        neutral_weight=neutral_weight,
    )
    ranked = np.argsort(scores)[::-1]
    selected = [int(i) for i in ranked[:top_k] if np.isfinite(scores[i])]

    stability: Dict[int, float] = {}
    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        n = latents.shape[0]
        hits = {i: 0 for i in selected}
        labels_arr = np.asarray(labels)
        pair_arr = np.asarray(pair_ids) if pair_ids is not None else None
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, n, size=n)
            sub_lat = latents[idx]
            sub_lab = labels_arr[idx]
            sub_pair = pair_arr[idx] if pair_arr is not None else None
            try:
                boot = select_features(
                    sub_lat,
                    sub_lab,
                    mode=mode,
                    top_k=top_k,
                    min_firing_rate=min_firing_rate,
                    class_a=pair[0],
                    class_b=pair[1],
                    pair_ids=sub_pair,
                    neutral_latents=neutral_latents,
                    neutral_specificity=neutral_specificity,
                    neutral_weight=neutral_weight,
                    bootstrap_samples=0,
                )
            except ValueError:
                continue
            for i in boot.feature_indices:
                if i in hits:
                    hits[i] += 1
        stability = {i: hits[i] / bootstrap_samples for i in selected}

    mean_by_class = {c: {i: float(means[c][i]) for i in selected} for c in pair if c in means}
    # keep all class means present
    for c, mu in means.items():
        mean_by_class.setdefault(c, {i: float(mu[i]) for i in selected})

    return SAEFeatureSelection(
        feature_indices=selected,
        scores={i: float(scores[i]) for i in selected},
        mean_by_class=mean_by_class,
        contrast_pair=pair,
        mode=mode,
        layer=-1,
        sae_id="",
        used_neutral_specificity=used_neutral,
        neutral_means={i: float(neutral_mean[i]) for i in selected} if neutral_mean is not None else {},
        bootstrap_stability=stability,
    )


def select_per_class_features(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
    target_classes: Optional[Sequence[str]] = None,
    top_k_per_class: int = 1,
    min_firing_rate: float = 0.01,
    neutral_latents: Optional[np.ndarray] = None,
    neutral_specificity: bool = True,
    neutral_weight: float = 1.0,
    score_mode: str = "mean_diff_neutral",
    opposite_fire_weight: float = 1.0,
) -> SAEFeatureSelection:
    """
    Select top-k SAE features **per class** (study primary SAE baseline).

    Default ``score_mode="mean_diff_neutral"``:
      score = (μ_c − μ_rest) − λ_neu |μ_neutral|
    ``score_mode="opp_fire_penalty"`` (exclusivity ablation):
      score = (μ_c − μ_rest) − λ_neu |μ_neu| − λ_opp P(z>0 | opposite class)
    Soft mean-diff already uses μ_rest (includes rivals); opp_fire is an explicit
    fire-rate penalty on the opposite labeled class.

    Study layer selection uses this **same** score: per layer take ordered top-k
    (in-layer only), then layer* = argmax_L score(top-1 @ L).
    """
    labels_arr = np.asarray(labels).astype(str)
    if target_classes is not None and len(target_classes) >= 1:
        classes = [str(c) for c in target_classes]
    else:
        # Infer from labels only — never invent gender M/F defaults.
        present = [str(c) for c in pd.Series(labels_arr).value_counts().index]
        if len(present) >= 2 and class_a and class_b:
            classes = [str(class_a), str(class_b)]
        elif len(present) >= 1:
            if class_a and str(class_a) in present:
                classes = [str(class_a)] + [c for c in present if c != str(class_a)]
            else:
                classes = present
        else:
            raise ValueError("select_per_class_features: no class labels in data")
    means = _class_means(latents, labels, tuple(classes))
    firing = (latents > 0).mean(axis=0)
    neutral_mean = None
    used_neutral = False
    if neutral_latents is not None and len(neutral_latents) and neutral_specificity:
        neutral_mean = np.abs(neutral_latents.mean(axis=0))
        used_neutral = True

    mode = str(score_mode or "mean_diff_neutral").strip().lower()
    scores_by_class: Dict[str, np.ndarray] = {}
    for c in classes:
        rest = labels_arr != c
        mu_rest = (
            latents[rest].mean(axis=0)
            if rest.any()
            else np.zeros(latents.shape[1], dtype=np.float64)
        )
        raw = means[c] - mu_rest
        sc = raw.astype(np.float64).copy()
        sc[firing < min_firing_rate] = -np.inf
        if neutral_mean is not None:
            sc = sc - float(neutral_weight) * neutral_mean
        if mode in {"opp_fire_penalty", "opp_fire", "opposite_fire"}:
            rivals = [x for x in classes if x != c]
            if rivals:
                opp_mask = np.zeros(labels_arr.shape[0], dtype=bool)
                for r in rivals:
                    opp_mask |= labels_arr == r
                if opp_mask.any():
                    fire_opp = (latents[opp_mask] > 0).mean(axis=0)
                    sc = sc - float(opposite_fire_weight) * fire_opp
        scores_by_class[c] = sc

    k = max(1, int(top_k_per_class))
    taken: set = set()
    features_by_class: Dict[str, List[int]] = {c: [] for c in classes}
    for _ in range(k):
        picks: Dict[str, int] = {}
        for c in classes:
            order = [
                int(i)
                for i in np.argsort(scores_by_class[c])[::-1]
                if i not in taken and np.isfinite(scores_by_class[c][i])
            ]
            if order:
                picks[c] = order[0]
        if not picks:
            break
        claimed: Dict[int, str] = {}
        for c, idx in sorted(picks.items(), key=lambda kv: -float(scores_by_class[kv[0]][kv[1]])):
            if idx in claimed or idx in taken:
                order = [
                    int(i)
                    for i in np.argsort(scores_by_class[c])[::-1]
                    if i not in taken and i not in claimed and np.isfinite(scores_by_class[c][i])
                ]
                if not order:
                    continue
                idx = order[0]
            claimed[idx] = c
        if not claimed:
            break
        for idx, c in claimed.items():
            features_by_class[c].append(idx)
            taken.add(idx)

    selected: List[int] = []
    for c in classes:
        selected.extend(features_by_class[c])
    scores = {}
    for c, feats in features_by_class.items():
        for i in feats:
            scores[i] = float(scores_by_class[c][i])
    mean_by_class = {
        c: {i: float(means[c][i]) for i in selected} for c in classes
    }
    contrast = (classes[0], classes[1]) if len(classes) >= 2 else (classes[0], classes[0])
    out_mode = "per_class" if mode in {"mean_diff_neutral", "mean_diff", ""} else f"per_class_{mode}"
    return SAEFeatureSelection(
        feature_indices=selected,
        scores=scores,
        mean_by_class=mean_by_class,
        contrast_pair=contrast,
        mode=out_mode,
        layer=-1,
        sae_id="",
        used_neutral_specificity=used_neutral,
        neutral_means=(
            {i: float(neutral_mean[i]) for i in selected} if neutral_mean is not None else {}
        ),
        features_by_class=features_by_class,
        scores_by_class={
            c: {i: float(scores_by_class[c][i]) for i in feats}
            for c, feats in features_by_class.items()
        },
    )


def per_class_readout_scores(
    latents: np.ndarray,
    features_by_class: Dict[str, Sequence[int]],
    *,
    class_a: str,
    class_b: str,
    k: Optional[int] = None,
) -> np.ndarray:
    """Scalar SAE readout aligned with GRADIEND polarity: sum(z_a[:k]) − sum(z_b[:k])."""
    fa = list(features_by_class.get(class_a) or [])
    fb = list(features_by_class.get(class_b) or [])
    if k is not None:
        fa, fb = fa[: int(k)], fb[: int(k)]
    if not fa or not fb:
        raise ValueError("per_class readout needs features for both classes")
    return latents[:, fa].sum(axis=1) - latents[:, fb].sum(axis=1)


def select_joint_feature(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    class_a: str = "M",
    class_b: str = "F",
    min_firing_rate: float = 0.01,
    neutral_latents: Optional[np.ndarray] = None,
    neutral_specificity: bool = True,
    neutral_weight: float = 1.0,
) -> SAEFeatureSelection:
    """Single best contrastive SAE feature (joint gender analogue; ablation)."""
    sel = select_features(
        latents,
        labels,
        mode="pairwise",
        top_k=1,
        min_firing_rate=min_firing_rate,
        class_a=class_a,
        class_b=class_b,
        neutral_latents=neutral_latents,
        neutral_specificity=neutral_specificity,
        neutral_weight=neutral_weight,
    )
    sel.mode = "joint"
    return sel


def joint_readout_scores(
    latents: np.ndarray,
    feature_index: int,
    labels: Sequence[str],
    *,
    class_a: str,
    class_b: str,
) -> np.ndarray:
    """Orient a single feature so higher score prefers ``class_a``."""
    labels_arr = np.asarray(labels)
    scores = latents[:, int(feature_index)].astype(np.float64)
    mean_a = float(scores[labels_arr == class_a].mean()) if (labels_arr == class_a).any() else 0.0
    mean_b = float(scores[labels_arr == class_b].mean()) if (labels_arr == class_b).any() else 0.0
    return scores if mean_a >= mean_b else -scores


def identify_sae_features(
    model,
    tokenizer,
    labeled_df: pd.DataFrame,
    *,
    text_col: str = "masked",
    label_col: str = "label_class",
    pair_id_col: str = "pair_id",
    layer: int = 11,
    release: Optional[str] = None,
    selection_split: str = "validation",
    top_k: int = 20,
    modes: Optional[Sequence[str]] = None,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
    target_classes: Optional[Sequence[str]] = None,
    neutral_texts: Optional[Sequence[str]] = None,
    neutral_specificity: bool = True,
    min_firing_rate: float = 0.01,
    bootstrap_samples: int = 0,
    progress: Optional[Any] = None,
    act_policy: str = "prediction",
    label_tokens: Optional[Mapping[str, str]] = None,
    max_length: int = 256,
    batch_size: int = 8,
    excluded_words: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, SAEFeatureSelection], Any, str, np.ndarray, pd.DataFrame, Optional[np.ndarray]]:
    """
    Run one or more selection modes on GRADIEND-aligned labeled data.

    Activations use the shared ACTIEND/CAA site (``act_policy``), not legacy
    last-token hooks on raw ``[MASK]`` strings.

    Returns (selections_by_mode, sae, hf_module, val_latents, split_df, neutral_latents).
    """
    def _prog(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(msg, flush=True)

    modes = list(modes or ("pairwise",))
    split_df = labeled_df[labeled_df["split"] == selection_split]
    if split_df.empty:
        split_df = labeled_df[labeled_df["split"] == "train"]
    if split_df.empty:
        raise ValueError(f"No rows for selection_split={selection_split!r}")

    release = release or active_sae_release()
    hf_module, sae_id = resid_sites(layer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _prog(f"  SAE: loading pretrained SAE {release!r} / {sae_id!r} on {device} …")
    sae = load_sae(release, sae_id, device=device)
    _prog(
        f"  SAE: loaded; encoding selection split ({len(split_df)} texts) "
        f"act_policy={act_policy!r} …"
    )

    # Shared H with CAA/ACTIEND (filled prediction / pre_prediction / …).
    del text_col  # texts come from masked+label fills via shared extractor
    hidden = collect_study_activations(
        model,
        tokenizer,
        split_df,
        layer=int(layer),
        act_policy=str(act_policy),
        label_tokens=label_tokens,
        batch_size=int(batch_size),
        max_length=int(max_length),
    )
    _prog(f"  SAE: encoding selection activations → latents (hidden={hidden.shape}) …")
    latents = encode_sae(sae, hidden)
    _prog(f"  SAE: selection latents shape={tuple(latents.shape)}")

    neutral_latents = None
    if neutral_texts:
        _prog(f"  SAE: encoding neutrals ({len(neutral_texts)} texts) …")
        neutral_hidden = collect_study_neutral_activations(
            model,
            tokenizer,
            list(neutral_texts),
            layer=int(layer),
            act_policy=str(act_policy),
            batch_size=int(batch_size),
            max_length=int(max_length),
            excluded_words=excluded_words,
        )
        n_tok = int(neutral_hidden.shape[0])
        if n_tok > MAX_NEUTRAL_TOKEN_ROWS:
            _prog(
                f"  SAE: {n_tok} non-pad neutral tokens; subsample {MAX_NEUTRAL_TOKEN_ROWS} "
                f"for latent store (still per-token, not document-mean)"
            )
            # Same indices at every layer (all_k sums aligned tokens).
            neutral_hidden = subsample_rows(
                neutral_hidden,
                MAX_NEUTRAL_TOKEN_ROWS,
                seed=NEUTRAL_TOKEN_SUBSAMPLE_SEED,
            )
        else:
            _prog(f"  SAE: {n_tok} non-pad neutral tokens (per-token, not document-mean)")
        neutral_latents = encode_sae(sae, neutral_hidden)
        _prog(f"  SAE: neutral latents shape={tuple(neutral_latents.shape)}")

    pair_ids = split_df[pair_id_col].tolist() if pair_id_col in split_df.columns else None
    labels = split_df[label_col].tolist()
    present_labels = {str(x) for x in labels}
    # One-pole circuit tasks (IOI / induction): CF partners live on alternative_*
    # and must never be scored as factual SAE classes.
    if target_classes is not None:
        want = [str(c) for c in target_classes]
        kept = [c for c in want if c in present_labels]
        dropped = [c for c in want if c not in present_labels]
        if dropped:
            _prog(
                f"  SAE: drop absent label_class {dropped} "
                f"(one-pole CF poles are not separate SAE contexts; present={sorted(present_labels)})"
            )
        target_classes = kept if kept else None
    if class_a is not None and str(class_a) not in present_labels:
        class_a = (list(target_classes)[0] if target_classes else None) or next(
            iter(present_labels), None
        )
    if class_b is not None and str(class_b) not in present_labels:
        class_b = (
            list(target_classes)[1]
            if target_classes and len(target_classes) > 1
            else None
        )
    bipolar_modes = {"joint", "pairwise", "pair_aware", "dense_probe"}
    selections: Dict[str, SAEFeatureSelection] = {}
    for mode in modes:
        _prog(f"  SAE: selecting features mode={mode!r} …")
        try:
            if mode in bipolar_modes and not (class_a and class_b):
                _prog(
                    f"  SAE: skip mode={mode!r} "
                    f"(needs two factual label_class values; have "
                    f"class_a={class_a!r} class_b={class_b!r})"
                )
                continue
            if mode == "pair_aware" and pair_ids is None:
                _prog(
                    f"  SAE: skip mode={mode!r} "
                    f"(no {pair_id_col!r} column on selection frame)"
                )
                continue
            if mode == "per_class":
                classes = list(target_classes) if target_classes else None
                if classes is None and class_a and class_b:
                    classes = [class_a, class_b]
                elif classes is None and class_a:
                    classes = [class_a]
                if not classes:
                    classes = [str(c) for c in pd.Series(np.asarray(labels).astype(str)).value_counts().index]
                if not classes:
                    raise ValueError("per_class SAE selection: no target classes")
                classes = [c for c in classes if str(c) in present_labels]
                if not classes:
                    raise ValueError(
                        f"per_class SAE selection: no target classes present on "
                        f"label_class (present={sorted(present_labels)})"
                    )
                # One-pole / single factual class is valid (score vs neutrals / empty rest).
                sel_kw: Dict[str, Any] = {
                    "target_classes": classes,
                    "top_k_per_class": top_k,
                    "min_firing_rate": min_firing_rate,
                    "neutral_latents": neutral_latents,
                    "neutral_specificity": neutral_specificity,
                }
                if len(classes) >= 2:
                    sel_kw["class_a"] = classes[0]
                    sel_kw["class_b"] = classes[1]
                elif class_a:
                    sel_kw["class_a"] = class_a
                sel = select_per_class_features(latents, labels, **sel_kw)
            elif mode == "joint":
                if not (class_a and class_b):
                    raise ValueError(
                        "joint SAE selection requires class_a and class_b "
                        "(refusing silent gender M/F defaults)"
                    )
                sel = select_joint_feature(
                    latents,
                    labels,
                    class_a=class_a,
                    class_b=class_b,
                    min_firing_rate=min_firing_rate,
                    neutral_latents=neutral_latents,
                    neutral_specificity=neutral_specificity,
                )
            else:
                sel = select_features(
                    latents,
                    labels,
                    mode=mode,
                    top_k=top_k,
                    min_firing_rate=min_firing_rate,
                    class_a=class_a,
                    class_b=class_b,
                    pair_ids=pair_ids,
                    neutral_latents=neutral_latents,
                    neutral_specificity=neutral_specificity,
                    bootstrap_samples=bootstrap_samples,
                )
        except ValueError as exc:
            # Soft-skip modes that need gender-style pair structure (religion/race).
            if mode in {"pair_aware", "pairwise", "joint"}:
                _prog(f"  SAE: skip mode={mode!r}: {exc}")
                continue
            raise
        sel.layer = layer
        sel.sae_id = sae_id
        sel.selection_split = selection_split
        selections[mode] = sel
    if not selections:
        raise ValueError(
            f"SAE feature selection produced no modes "
            f"(requested={modes}; pair_ids={'yes' if pair_ids is not None else 'no'})"
        )
    _prog(f"  SAE: feature selection done ({len(selections)} modes)")
    return selections, sae, hf_module, latents, split_df, neutral_latents


# ---------------------------------------------------------------------------
# SAE run cache (latents + selections), analogous to GRADIEND encoder CSV cache.
# ---------------------------------------------------------------------------

# v5: neutrals scored per non-pad token (not document-mean) for prediction/pre_prediction.
# v6: val/test neutral split, excluded class tokens, Youden τ from validation.
SAE_CACHE_VERSION = 6


def _texts_fingerprint(texts: Sequence[str]) -> str:
    import hashlib

    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def sae_cache_fingerprint(
    *,
    release: str,
    layer: int,
    modes: Sequence[str],
        top_k: int,
        readout_ks: Sequence[int],
        selection_split: str,
    class_a: str,
    class_b: str,
    neutral_specificity: bool,
    bootstrap_samples: int,
    min_firing_rate: float,
    val_texts: Sequence[str],
    test_texts: Sequence[str],
    neutral_texts: Sequence[str],
    causal_strengths: Sequence[float],
    causal_n: int,
    act_policy: str = "prediction",
    val_neutral_texts: Optional[Sequence[str]] = None,
    test_neutral_texts: Optional[Sequence[str]] = None,
    excluded_words: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    from activation_protocol import ACTIVATION_PROTOCOL_VERSION

    hf_module, sae_id = resid_sites(layer)
    val_neu = list(val_neutral_texts) if val_neutral_texts is not None else list(neutral_texts)
    test_neu = list(test_neutral_texts) if test_neutral_texts is not None else list(neutral_texts)
    return {
        "cache_version": SAE_CACHE_VERSION,
        "activation_protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "act_policy": str(act_policy),
        "release": release,
        "layer": int(layer),
        "sae_id": sae_id,
        "hf_module": hf_module,
        "modes": list(modes),
        "top_k": int(top_k),
        "readout_ks": [int(k) for k in readout_ks],
        "selection_split": selection_split,
        "class_a": class_a,
        "class_b": class_b,
        "neutral_specificity": bool(neutral_specificity),
        "bootstrap_samples": int(bootstrap_samples),
        "min_firing_rate": float(min_firing_rate),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "n_neutral": len(neutral_texts),
        "n_val_neutral": len(val_neu),
        "n_test_neutral": len(test_neu),
        "val_text_fp": _texts_fingerprint(val_texts),
        "test_text_fp": _texts_fingerprint(test_texts),
        "neutral_text_fp": _texts_fingerprint(neutral_texts),
        "val_neutral_text_fp": _texts_fingerprint(val_neu),
        "test_neutral_text_fp": _texts_fingerprint(test_neu),
        "excluded_words": [str(w) for w in (excluded_words or [])],
        "causal_strengths": [float(s) for s in causal_strengths],
        "causal_n": int(causal_n),
    }


def selection_from_dict(payload: Dict[str, Any]) -> SAEFeatureSelection:
    """Inverse of selection_to_dict / asdict(SAEFeatureSelection)."""
    contrast = payload.get("contrast_pair") or ("M", "F")
    if isinstance(contrast, list):
        contrast = tuple(contrast)
    scores = {int(k): float(v) for k, v in (payload.get("scores") or {}).items()}
    mean_by_class = {
        str(c): {int(k): float(v) for k, v in feats.items()}
        for c, feats in (payload.get("mean_by_class") or {}).items()
    }
    return SAEFeatureSelection(
        feature_indices=[int(i) for i in payload.get("feature_indices") or []],
        scores=scores,
        mean_by_class=mean_by_class,
        contrast_pair=(str(contrast[0]), str(contrast[1])),
        mode=str(payload.get("mode") or "pairwise"),
        layer=int(payload.get("layer") or 0),
        sae_id=str(payload.get("sae_id") or ""),
        selection_split=str(payload.get("selection_split") or "validation"),
        used_neutral_specificity=bool(payload.get("used_neutral_specificity")),
        neutral_means={int(k): float(v) for k, v in (payload.get("neutral_means") or {}).items()},
        bootstrap_stability={
            int(k): float(v) for k, v in (payload.get("bootstrap_stability") or {}).items()
        },
        features_by_class={
            str(c): [int(i) for i in feats]
            for c, feats in (payload.get("features_by_class") or {}).items()
        },
        scores_by_class={
            str(c): {int(k): float(v) for k, v in sc.items()}
            for c, sc in (payload.get("scores_by_class") or {}).items()
        },
    )


def _sae_cache_act_slug(act_policy: Optional[str]) -> str:
    s = str(act_policy or "prediction").strip() or "prediction"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _save_sparse_latents(path: Path, arr: np.ndarray) -> None:
    """Persist a [n_rows, d_sae] latent matrix as nonzero (row, col, value) triples.

    SAE latents are ReLU activations, so the large majority of entries are
    exactly zero; storing them densely (the original ``np.save``) wasted
    10-100x disk space (e.g. a 16k-row x 32k-feature neutral cache alone was
    ~2GB dense). This is a pure on-disk format change — callers still get a
    dense array back from the matching loader.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        np.savez_compressed(path, dense=arr)
        return
    n_rows, n_cols = arr.shape
    rows, cols = np.nonzero(arr)
    data = arr[rows, cols]
    counts = np.bincount(rows, minlength=n_rows)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    np.savez_compressed(
        path,
        data=data,
        indices=cols.astype(np.int32),
        indptr=indptr,
        shape=np.asarray([n_rows, n_cols], dtype=np.int64),
    )


def _load_sparse_latents(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        if "dense" in npz.files:
            return npz["dense"]
        shape = tuple(int(x) for x in npz["shape"])
        out = np.zeros(shape, dtype=np.float32)
        indptr = npz["indptr"]
        row_lengths = np.diff(indptr)
        row_ids = np.repeat(np.arange(shape[0]), row_lengths)
        out[row_ids, npz["indices"]] = npz["data"]
        return out


def sae_cache_dir(
    output_dir: Any,
    layer: Optional[int] = None,
    act_policy: Optional[str] = None,
) -> Path:
    from pathlib import Path

    root = Path(output_dir) / "sae_cache"
    if layer is None:
        return root
    name = f"L{int(layer)}"
    if act_policy is not None:
        name = f"{name}_{_sae_cache_act_slug(act_policy)}"
    return root / name


def save_sae_cache(
    output_dir: Any,
    *,
    fingerprint: Dict[str, Any],
    val_latents: np.ndarray,
    test_latents: np.ndarray,
    neutral_latents: Optional[np.ndarray],
    val_labels: Sequence[str],
    test_labels: Sequence[str],
    selections: Dict[str, SAEFeatureSelection],
    causal: Optional[List[Dict[str, Any]]] = None,
    test_neutral_latents: Optional[np.ndarray] = None,
) -> Path:
    """Persist SAE latents + selections (+ optional causal) for skip-on-recompute."""
    import json

    layer = fingerprint.get("layer")
    root = sae_cache_dir(
        output_dir,
        layer=int(layer) if layer is not None else None,
        act_policy=fingerprint.get("act_policy"),
    )
    root.mkdir(parents=True, exist_ok=True)
    _save_sparse_latents(root / "val_latents.npz", val_latents)
    _save_sparse_latents(root / "test_latents.npz", test_latents)
    # Drop legacy dense shards from before the sparse-cache fix, if present,
    # so a re-save doesn't leave a stale multi-GB dense copy alongside.
    (root / "val_latents.npy").unlink(missing_ok=True)
    (root / "test_latents.npy").unlink(missing_ok=True)
    if neutral_latents is not None:
        _save_sparse_latents(root / "neutral_latents.npz", neutral_latents)
    else:
        (root / "neutral_latents.npz").unlink(missing_ok=True)
    (root / "neutral_latents.npy").unlink(missing_ok=True)
    if test_neutral_latents is not None:
        _save_sparse_latents(root / "test_neutral_latents.npz", test_neutral_latents)
    else:
        (root / "test_neutral_latents.npz").unlink(missing_ok=True)
    (root / "test_neutral_latents.npy").unlink(missing_ok=True)
    meta = {
        "fingerprint": fingerprint,
        "val_labels": [str(x) for x in val_labels],
        "test_labels": [str(x) for x in test_labels],
        "selections": {mode: asdict(sel) for mode, sel in selections.items()},
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if causal is not None:
        (root / "causal.json").write_text(json.dumps(causal, indent=2), encoding="utf-8")
    return root


def load_sae_cache(
    output_dir: Any,
    fingerprint: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Load SAE cache when fingerprint matches.

    Returns dict with val/test/neutral latents, labels, selections, optional causal;
    or None if missing / mismatched.
    """
    import json

    layer = fingerprint.get("layer")
    layer_i = int(layer) if layer is not None else None
    root = sae_cache_dir(
        output_dir, layer=layer_i, act_policy=fingerprint.get("act_policy")
    )
    def _latents_exist(base: Path) -> bool:
        # New sparse .npz, or a legacy dense .npy from before the sparse-cache fix.
        return base.with_suffix(".npz").is_file() or base.is_file()

    meta_path = root / "meta.json"
    val_path = root / "val_latents.npy"
    test_path = root / "test_latents.npy"
    if not (meta_path.is_file() and _latents_exist(val_path) and _latents_exist(test_path)):
        # Legacy path: both SAE sites used to share sae_cache/L{n} and clobber.
        legacy = sae_cache_dir(output_dir, layer=layer_i)
        if legacy != root:
            meta_path = legacy / "meta.json"
            val_path = legacy / "val_latents.npy"
            test_path = legacy / "test_latents.npy"
            root = legacy
        if not (meta_path.is_file() and _latents_exist(val_path) and _latents_exist(test_path)):
            return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    saved_fp = meta.get("fingerprint") or {}
    if saved_fp != fingerprint:
        mismatched = sorted(k for k in set(saved_fp) | set(fingerprint) if saved_fp.get(k) != fingerprint.get(k))
        return {"_mismatch": mismatched, "_saved": saved_fp}
    neutral_path = root / "neutral_latents.npy"
    test_neu_path = root / "test_neutral_latents.npy"
    def _safe_load(base_path: Path) -> Optional[np.ndarray]:
        # Prefer the sparse .npz (current format); fall back to a legacy dense
        # .npy shard saved before the sparse-cache fix.
        npz_path = base_path.with_suffix(".npz")
        if npz_path.is_file():
            try:
                return _load_sparse_latents(npz_path)
            except Exception:
                # Corrupt/truncated cache shards can happen after interrupted writes.
                try:
                    npz_path.unlink()
                except Exception:
                    pass
        if base_path.is_file():
            try:
                return np.load(base_path)
            except Exception:
                try:
                    base_path.unlink()
                except Exception:
                    pass
        return None

    neutral = _safe_load(neutral_path)
    test_neutral = _safe_load(test_neu_path)
    selections = {
        mode: selection_from_dict(payload)
        for mode, payload in (meta.get("selections") or {}).items()
    }
    causal = None
    causal_path = root / "causal.json"
    if causal_path.is_file():
        try:
            causal = json.loads(causal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            causal = None
    val_latents = _safe_load(val_path)
    test_latents = _safe_load(test_path)
    if val_latents is None or test_latents is None:
        return None
    return {
        "cache_dir": root,
        "val_latents": val_latents,
        "test_latents": test_latents,
        "neutral_latents": neutral,
        "test_neutral_latents": test_neutral,
        "val_labels": list(meta.get("val_labels") or []),
        "test_labels": list(meta.get("test_labels") or []),
        "selections": selections,
        "causal": causal,
        "hf_module": fingerprint["hf_module"],
        "sae_id": fingerprint["sae_id"],
    }




# Prefer human-readable class ids; fall back to GRADIEND signed ``label`` (±1).
# ``factual_id`` is last: on full encoder caches it is often NaN/neutral even when
# ``source_id`` / ``label`` correctly mark M/F.
_CLASS_LABEL_COLUMNS = (
    "label_class",
    "feature_class",
    "factual_class",
    "source_id",
    "factual_id",
    "label",
)
_SPLIT_COLUMNS = ("data_split", "split")
_USELESS_LABEL_TOKENS = {"", "nan", "none", "neutral", "0", "0.0"}


def encoder_df_split_subset(
    encoder_df: Optional[pd.DataFrame],
    split: str = "test",
) -> Optional[pd.DataFrame]:
    """Labeled training rows on one split (drops package neutral variants)."""
    if encoder_df is None or encoder_df.empty:
        return encoder_df
    out = encoder_df
    if "type" in out.columns:
        training = out[out["type"].astype(str) == "training"]
        if not training.empty:
            out = training
    want = {str(split)}
    if str(split) == "validation":
        want = {"validation", "val"}
    for col in _SPLIT_COLUMNS:
        if col not in out.columns:
            continue
        rows = out[out[col].astype(str).isin(want)]
        if not rows.empty:
            return rows
    return out




def _norm_class_token(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _labels_from_encoder_df(
    encoder_df: pd.DataFrame,
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Pick the first label column that yields usable binary class ids."""
    want_a = _norm_class_token(class_a) if class_a is not None else None
    want_b = _norm_class_token(class_b) if class_b is not None else None
    for col in _CLASS_LABEL_COLUMNS:
        if col not in encoder_df.columns:
            continue
        raw = encoder_df[col]
        # Skip all-null columns (common for factual_id on training rows).
        if raw.isna().all():
            continue
        labels = raw.astype(str).to_numpy()
        if class_a is not None and class_b is not None:
            # Case/space-insensitive match onto requested class ids.
            normed = np.asarray([_norm_class_token(x) for x in labels])
            mapped = labels.copy()
            if want_a:
                mapped[normed == want_a] = str(class_a)
            if want_b and want_b != want_a:
                mapped[normed == want_b] = str(class_b)
            canon = _canonicalize_binary_labels(
                mapped, class_a=str(class_a), class_b=str(class_b)
            )
            has_a = (canon == str(class_a)).any()
            has_b = (canon == str(class_b)).any()
            # One-pole eval passes class_a == class_b; only need the claim class.
            if want_a == want_b:
                if has_a:
                    return canon.astype(str), col
            elif (
                np.isin(canon, [str(class_a), str(class_b)]).sum() >= 2
                and has_a
                and has_b
            ):
                return canon.astype(str), col
            continue
        usable = [
            v
            for v in pd.unique(labels)
            if str(v).strip().lower() not in _USELESS_LABEL_TOKENS
        ]
        if len(usable) >= 2:
            return labels, col
    return None, None


def _labels_for_target_class(
    encoder_df: pd.DataFrame,
    *,
    target_class: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Fallback when binary label extraction fails: any column mentioning ``target_class``."""
    want = _norm_class_token(target_class)
    for col in _CLASS_LABEL_COLUMNS:
        if col not in encoder_df.columns:
            continue
        raw = encoder_df[col]
        if raw.isna().all():
            continue
        labels = raw.astype(str).to_numpy()
        normed = np.asarray([_norm_class_token(x) for x in labels])
        if (normed == want).any():
            mapped = labels.copy()
            mapped[normed == want] = str(target_class)
            # Map signed labels onto the claim class vs "other".
            mapped = _canonicalize_binary_labels(
                mapped, class_a=str(target_class), class_b="__other__"
            )
            return mapped, col
    return None, None


def _canonicalize_binary_labels(
    labels: Sequence[str],
    *,
    class_a: str,
    class_b: str,
) -> np.ndarray:
    """
    Normalize labels to class_a/class_b when cached encoder data uses numeric labels.

    Known fallback: label values in {1, +1, 1.0} map to class_a and
    {-1, -1.0} map to class_b.
    """
    # object dtype: assigning long class ids into ``<U3`` from ``"1.0"`` truncates.
    arr = np.asarray(labels).astype(str)
    out = np.array(arr, dtype=object)
    pos = {"1", "+1", "1.0", "+1.0"}
    neg = {"-1", "-1.0"}
    out[np.isin(arr, list(pos))] = str(class_a)
    out[np.isin(arr, list(neg))] = str(class_b)
    return out


def _resolve_pair_classes(
    encoding_direction: Optional[Mapping[str, float]],
    candidates: Sequence[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the (positive-pole, negative-pole) classes a bipolar encoder
    was actually trained on, from its own ``encoding_direction`` map.

    Callers evaluating a pair-trained encoder often pass the *full* task
    class list (e.g. all 3 RAVEL classes) as ``candidates`` so off-pair
    "does this leak?" checks can run against every class -- but a pair
    encoder's raw label column only ever carries two poles, and their
    identity is a property of *this* trainer, not of ``candidates``' list
    order. Resolving them as ``candidates[0]``/``candidates[1]`` positionally
    silently mislabels whichever real pole isn't first/second in that list:
    e.g. for RAVEL country (``classes: [united_states, china, russia]``), a
    china-russia pair trainer's own poles are +1=china/-1=russia, but
    ``candidates[0]/[1]`` = united_states/china -- so -1 rows (real russia)
    get relabeled "china" and the encoder looks like it has zero russia
    rows at all, while +1 rows (real china) get relabeled "united_states"
    and produce a spurious result for a class this encoder never saw.
    ``encoding_direction`` -- when it has exactly one +1 and one -1 entry
    among ``candidates`` -- is the authoritative source for which two
    classes those poles really are; fall back to positional resolution
    (the old behavior, still correct when ``candidates`` already *is* the
    pair, e.g. any genuinely 2-class task) only when it doesn't resolve.
    """
    if not encoding_direction:
        return None, None
    signed = {
        str(c): float(encoding_direction[str(c)])
        for c in candidates
        if str(c) in encoding_direction
    }
    pos = [c for c, d in signed.items() if d > 0]
    neg = [c for c, d in signed.items() if d < 0]
    if len(pos) == 1 and len(neg) == 1:
        return pos[0], neg[0]
    return None, None


def _binary_mask_and_scores(
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    class_a: str,
    class_b: str,
) -> Tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = _canonicalize_binary_labels(labels, class_a=class_a, class_b=class_b)
    mask = np.isin(labels_arr, [class_a, class_b])
    return scores_arr[mask], labels_arr[mask]


def _roc_auc(scores: np.ndarray, labels: np.ndarray, *, class_a: str) -> Optional[float]:
    """ROC-AUC with polarity flip so higher score prefers ``class_a``.

    When class means are equal the mean-based flip is undefined; report
    ``max(auc, 1-auc)`` so equal-mean anti-ranks do not look like 0.28.
    """
    if scores.size < 2 or len(np.unique(labels)) < 2:
        return None
    y = (labels == class_a).astype(int)
    mean_a = float(scores[labels == class_a].mean())
    mean_b = float(scores[labels != class_a].mean())
    oriented = scores if mean_a >= mean_b else -scores
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, oriented))
    except Exception:
        # Rank-based fallback without sklearn.
        order = np.argsort(oriented)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(oriented) + 1)
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    if abs(mean_a - mean_b) < 1e-12:
        return float(max(auc, 1.0 - auc))
    return auc


def _balanced_accuracy_at_midpoint(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    class_a: str,
    class_b: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Decision-boundary accuracy: threshold = midpoint of class means
    (no held-out threshold fit; same rule for every method).
    """
    a = scores[labels == class_a]
    b = scores[labels == class_b]
    if a.size == 0 or b.size == 0:
        return None, None
    mean_a, mean_b = float(a.mean()), float(b.mean())
    mid = 0.5 * (mean_a + mean_b)
    if mean_a >= mean_b:
        pred_a = scores >= mid
    else:
        pred_a = scores <= mid
    pred = np.where(pred_a, class_a, class_b)
    sens = float((pred[labels == class_a] == class_a).mean())
    spec_cls = float((pred[labels == class_b] == class_b).mean())
    return float(0.5 * (sens + spec_cls)), float(mid)


def _youden_threshold(
    pos: np.ndarray,
    neg: np.ndarray,
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Youden threshold for ``(sign * score) > τ ⇒ positive``.

    Self-orienting like ``_roc_auc``: picks whichever direction (raw score or
    its negation) better separates ``pos`` from ``neg``, based on which group
    has the higher raw mean — same rule ``_roc_auc`` already uses. Without
    this, a class whose natural polarity happens to run opposite the naive
    "higher score = positive" convention (raw ``pos`` mean below ``neg``
    mean — e.g. an encoder that saturates negative for its target class) gets
    a degenerate threshold search: ``roc_auc`` (which does self-orient) can
    read near 1.0 while this search, fixed to "greater than", finds no good
    cut and reports near-zero TNR. See CLAUDE.md,
    "class_exclusivity/specificity ... Youden threshold orientation".

    Maximizes J = TPR + TNR − 1 over midpoints between sorted unique
    *oriented* scores. Returns ``(threshold, J, sign)`` — ``threshold`` is in
    oriented units; callers must compare ``sign * raw_score`` against it, not
    ``raw_score`` directly. Returns ``(None, None, 1.0)`` if undefined.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None, None, 1.0
    sign = 1.0 if float(pos.mean()) >= float(neg.mean()) else -1.0
    pos_o = sign * pos
    neg_o = sign * neg
    all_scores = np.concatenate([pos_o, neg_o])
    uniq = np.unique(all_scores)
    if uniq.size == 1:
        # Degenerate: everything equal — any cut is vacuous.
        tau = float(uniq[0])
        tpr = float((pos_o > tau).mean())
        tnr = float((neg_o <= tau).mean())
        return tau, float(tpr + tnr - 1.0), sign
    # Candidate cuts: midpoints between consecutive unique values, plus
    # below-min / above-max so extreme policies are considered.
    mids = 0.5 * (uniq[:-1] + uniq[1:])
    candidates = np.concatenate(
        [[uniq[0] - 1.0], mids, [uniq[-1]]]
    )
    best_tau = float(candidates[0])
    best_j = -np.inf
    for tau in candidates:
        tpr = float((pos_o > tau).mean())
        tnr = float((neg_o <= tau).mean())
        j = tpr + tnr - 1.0
        if j > best_j:
            best_j = j
            best_tau = float(tau)
    return best_tau, float(best_j), sign


def _orientation_sign(pos: np.ndarray, neg: np.ndarray) -> float:
    """Same mean-based orientation rule as ``_roc_auc``/``_youden_threshold``.

    Used to re-derive the sign for a *provided* (e.g. validation-fit)
    threshold, since only the numeric ``τ`` is carried across splits, not the
    orientation it was fit under. Assumes the checkpoint's polarity doesn't
    flip between validation and test — true unless separation is near-chance
    on one of the two splits, in which case the reported numbers are
    near-chance either way.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    return 1.0 if float(pos.mean()) >= float(neg.mean()) else -1.0


def specificity_from_scores(
    class_scores: Sequence[float],
    labels: Sequence[str],
    neutral_scores: Sequence[float],
    *,
    class_a: str = "M",
    class_b: str = "F",
) -> Dict[str, Any]:
    """
    Unified specificity for a scalar readout (GRADIEND / ACTIEND / SAE).

    Neutrals should sit near the **inactive** anchor:

    - opposite-sign class means (GRADIEND-like bipolar): anchor = midpoint
    - same-sign / non-negative activations (typical SAE): anchor = 0

    Midpoint-only specificity under-rates SAE features that fire on both
    classes but stay off on neutrals (0 is far from a positive midpoint).

    ``specificity`` ∈ (0, 1]: half_sep / (half_sep + mean_|neutral - anchor|)
    """
    scores_arr, labels_arr = _binary_mask_and_scores(
        class_scores, labels, class_a=class_a, class_b=class_b
    )
    neutrals = np.asarray(neutral_scores, dtype=float)
    neutrals = neutrals[np.isfinite(neutrals)]
    if scores_arr.size < 2 or neutrals.size == 0:
        return {"error": "insufficient samples for specificity"}
    a = scores_arr[labels_arr == class_a]
    b = scores_arr[labels_arr == class_b]
    if a.size == 0 or b.size == 0:
        return {"error": "need both classes for specificity"}
    mean_a, mean_b = float(a.mean()), float(b.mean())
    mid = 0.5 * (mean_a + mean_b)
    half_sep = 0.5 * abs(mean_a - mean_b)
    if (mean_a >= 0 and mean_b >= 0) or (mean_a <= 0 and mean_b <= 0):
        anchor = 0.0
        anchor_mode = "inactive_zero"
    else:
        anchor = mid
        anchor_mode = "midpoint"
    neutral_mean = float(neutrals.mean())
    neutral_abs_mean = float(np.abs(neutrals).mean())
    neutral_gap = float(abs(neutral_mean - mid))
    neutral_spread = float(np.abs(neutrals - anchor).mean())
    neutral_spread_mid = float(np.abs(neutrals - mid).mean())
    specificity = float(half_sep / (half_sep + neutral_spread + 1e-8))
    specificity_mid = float(half_sep / (half_sep + neutral_spread_mid + 1e-8))
    return {
        "specificity": specificity,
        "specificity_mid": specificity_mid,
        "specificity_anchor": anchor,
        "specificity_anchor_mode": anchor_mode,
        "neutral_mean": neutral_mean,
        "neutral_abs_mean": neutral_abs_mean,
        "neutral_gap": neutral_gap,
        "neutral_spread": neutral_spread,
        "decision_midpoint": mid,
        "n_neutral": int(neutrals.size),
    }


def readout_metrics(
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    class_a: str = "M",
    class_b: str = "F",
    neutral_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Fair cross-method readout metrics (same protocol for GRADIEND, ACTIEND, SAE).

    Primary (scale-free / decision-based):
      - roc_auc: binary separability (threshold-free)
      - balanced_accuracy: midpoint decision boundary
      - cohens_d: effect size
      - specificity: neutrals near midpoint vs class half-separation
        (requires ``neutral_scores``)

    Diagnostic / scale-dependent:
      - encoder_correlation: Pearson vs ±1 (favors calibrated ±1 readouts)
      - class_separation: raw |Δmean|
    """
    scores_arr, labels_arr = _binary_mask_and_scores(
        scores, labels, class_a=class_a, class_b=class_b
    )
    if scores_arr.size < 2:
        return {"error": "insufficient samples"}
    a = scores_arr[labels_arr == class_a]
    b = scores_arr[labels_arr == class_b]
    if a.size == 0 or b.size == 0:
        return {"error": "need both classes"}
    sep = float(abs(a.mean() - b.mean()))
    signed = np.where(labels_arr == class_a, 1.0, -1.0)
    if scores_arr.std() > 0 and signed.std() > 0:
        corr = float(np.corrcoef(scores_arr, signed)[0, 1])
    else:
        corr = None
    if a.size + b.size > 2:
        pooled_var = (
            (a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)
        ) / (a.size + b.size - 2)
        cohens_d = float(sep / np.sqrt(pooled_var)) if pooled_var > 0 else None
    else:
        cohens_d = None
    roc_auc = _roc_auc(scores_arr, labels_arr, class_a=class_a)
    bal_acc, mid = _balanced_accuracy_at_midpoint(
        scores_arr, labels_arr, class_a=class_a, class_b=class_b
    )
    out: Dict[str, Any] = {
        "class_a": class_a,
        "class_b": class_b,
        f"mean_{class_a}": float(a.mean()),
        f"mean_{class_b}": float(b.mean()),
        "class_separation": sep,
        "separation": sep,
        "encoder_correlation": corr,
        "cohens_d": cohens_d,
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "decision_midpoint": mid,
        f"n_{class_a}": int(a.size),
        f"n_{class_b}": int(b.size),
    }
    if neutral_scores is not None:
        spec = specificity_from_scores(
            scores_arr,
            labels_arr,
            neutral_scores,
            class_a=class_a,
            class_b=class_b,
        )
        if "error" not in spec:
            out.update(spec)
        else:
            out["specificity_error"] = spec["error"]
    return out


def encoder_df_neutral_scores(
    encoder_df: Optional[pd.DataFrame],
    *,
    split: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Collect neutral variant encoder scores from a full encoder analysis DataFrame."""
    if encoder_df is None or encoder_df.empty or "encoded" not in encoder_df.columns:
        return None
    if "type" not in encoder_df.columns:
        return None
    types = encoder_df["type"].astype(str)
    # Prefer true neutral dataset; fall back to masked-training neutrals.
    rows = None
    for preferred in ("neutral_dataset", "neutral_training_masked"):
        cand = encoder_df[types == preferred]
        if not cand.empty:
            rows = cand
            break
    if rows is None:
        rows = encoder_df[types.str.startswith("neutral")]
    if rows.empty:
        return None
    if split is not None:
        want = {str(split)}
        if str(split) == "validation":
            want = {"validation", "val"}
        for col in _SPLIT_COLUMNS:
            if col not in rows.columns:
                continue
            sub = rows[rows[col].astype(str).isin(want)]
            if not sub.empty:
                rows = sub
                break
    return rows["encoded"].to_numpy(dtype=float)


def encoder_specificity(
    encoder_df: pd.DataFrame,
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
    neutral_scores: Optional[Sequence[float]] = None,
    full_encoder_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Fair metrics for GRADIEND/ACTIEND encoder readouts.

    ``encoder_df`` should be the labeled test training subset. Neutrals are taken
    from ``neutral_scores``, else from ``full_encoder_df`` / ``encoder_df``.
    """
    if encoder_df is None or encoder_df.empty:
        return {"error": "empty"}
    if "encoded" not in encoder_df.columns:
        return {"error": "missing encoded column"}
    labels, class_col = _labels_from_encoder_df(
        encoder_df, class_a=class_a, class_b=class_b
    )
    if labels is None:
        return {"error": f"missing class column (tried {_CLASS_LABEL_COLUMNS})"}
    classes = list(pd.Series(labels).value_counts().index)
    if class_a is None or class_b is None:
        usable = [
            str(c) for c in classes
            if str(c).strip().lower() not in _USELESS_LABEL_TOKENS
        ]
        if len(usable) < 2:
            if any(str(c) in {"1", "1.0", "+1", "+1.0", "-1", "-1.0"} for c in classes):
                class_a, class_b = "M", "F"
            else:
                return {"error": "need both classes"}
        else:
            class_a, class_b = usable[0], usable[1]
    if neutral_scores is None:
        neutral_scores = encoder_df_neutral_scores(full_encoder_df)
    if neutral_scores is None:
        neutral_scores = encoder_df_neutral_scores(encoder_df)
    out = readout_metrics(
        encoder_df["encoded"].to_numpy(),
        labels,
        class_a=class_a,
        class_b=class_b,
        neutral_scores=neutral_scores,
    )
    if "error" not in out:
        out["class_column"] = class_col
    return out


def infer_class_encoding_directions(
    encoder_df: pd.DataFrame,
    target_classes: Sequence[str],
    *,
    encoding_direction: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Map each feature class to a polarity so ``direction[c] * encoded`` is high on class ``c``.

    Prefer an explicit ``encoding_direction`` dict from the trainer; otherwise infer
    from class means of ``encoded`` (sign of mean). Callers must pass **validation**
    labeled rows for that fallback — never the test scores being reported.
    Empty ``encoder_df`` without a trainer dict returns +1 (no eval-split inference).
    """
    if encoding_direction:
        out = {str(c): float(encoding_direction[c]) for c in target_classes if c in encoding_direction}
        if len(out) == len(target_classes):
            return out
    pair_a, pair_b = _resolve_pair_classes(encoding_direction, target_classes)
    if pair_a is None or pair_b is None:
        pair_a = str(target_classes[0]) if target_classes else None
        pair_b = str(target_classes[1]) if len(target_classes) > 1 else None
    labels, _ = _labels_from_encoder_df(
        encoder_df,
        class_a=pair_a,
        class_b=pair_b,
    )
    if labels is None or "encoded" not in encoder_df.columns:
        return {str(c): 1.0 for c in target_classes}
    encoded = encoder_df["encoded"].to_numpy(dtype=float)
    canon = _canonicalize_binary_labels(
        labels,
        class_a=pair_a,
        class_b=pair_b if pair_b is not None else pair_a,
    ) if len(target_classes) >= 2 else np.asarray(labels).astype(str)
    directions: Dict[str, float] = {}
    for c in target_classes:
        c = str(c)
        mask = canon == c if len(target_classes) >= 2 else (np.asarray(labels).astype(str) == c)
        if not mask.any():
            directions[c] = 1.0
            continue
        mean_c = float(encoded[mask].mean())
        directions[c] = 1.0 if mean_c >= 0 else -1.0
    return directions


def encoder_class_vs_neutral_metrics(
    encoder_df: pd.DataFrame,
    *,
    target_class: str,
    target_classes: Sequence[str],
    full_encoder_df: Optional[pd.DataFrame] = None,
    encoding_direction: Optional[Dict[str, float]] = None,
    polarity_df: Optional[pd.DataFrame] = None,
    youden_threshold: Optional[float] = None,
    overlay_neutral_scores: Optional[Sequence[float]] = None,
    split: str = "test",
    n_bootstrap: int = 100,
    frozen: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    GRADIEND/ACTIEND per-class analogue of SAE class features.

    ``frozen``: validation-fit Spec_n/Excl rules from :func:`frozen_decision_rules`.

    Orients the joint encoder so higher score means ``target_class``, then
    evaluates **target class vs neutral** (same protocol as SAE class features).
    """
    target_class = str(target_class)
    labeled = (
        encoder_df_split_subset(encoder_df, split=split)
        if encoder_df is not None
        else None
    )
    if labeled is None or labeled.empty or "encoded" not in labeled.columns:
        return {"error": "empty encoder df"}
    # This encoder's own two trained poles -- NOT target_classes[0]/[1], which
    # is the caller's full task class list and need not be this pair (see
    # _resolve_pair_classes docstring / CLAUDE.md's RAVEL pole-mislabeling note).
    pair_a, pair_b = _resolve_pair_classes(encoding_direction, target_classes)
    if pair_a is None or pair_b is None:
        pair_a = str(target_classes[0]) if target_classes else target_class
        pair_b = str(target_classes[1]) if len(target_classes) > 1 else target_class
    labels, _ = _labels_from_encoder_df(
        labeled,
        class_a=pair_a,
        class_b=pair_b,
    )
    if labels is None:
        labels, _ = _labels_for_target_class(labeled, target_class=target_class)
    if labels is None:
        return {"error": "missing class labels"}
    if encoding_direction:
        directions = infer_class_encoding_directions(
            polarity_df if polarity_df is not None else labeled,
            target_classes,
            encoding_direction=encoding_direction,
        )
    elif polarity_df is not None and not getattr(polarity_df, "empty", True):
        directions = infer_class_encoding_directions(
            polarity_df, target_classes, encoding_direction=None
        )
    else:
        directions = {str(c): 1.0 for c in target_classes}
    direction = float(directions.get(target_class, 1.0))
    encoded = labeled["encoded"].to_numpy(dtype=float) * direction
    if pair_a != pair_b:
        canon = _canonicalize_binary_labels(
            labels,
            class_a=pair_a,
            class_b=pair_b,
        )
    else:
        canon = np.asarray(labels).astype(str)
    target_scores = encoded[canon == target_class]
    if target_scores.size == 0:
        return {"error": f"no rows for class {target_class!r}"}

    if overlay_neutral_scores is not None:
        neu = np.asarray(overlay_neutral_scores, dtype=float)
    else:
        neu = encoder_df_neutral_scores(full_encoder_df, split=split)
        if neu is None:
            neu = encoder_df_neutral_scores(encoder_df, split=split)
    if neu is None or len(neu) == 0:
        return {"error": "no neutral encoder scores"}
    neutral_scores = np.asarray(neu, dtype=float) * direction

    other_scores = None
    other_class = None
    scores_other_by_class: Dict[str, np.ndarray] = {}
    others = [str(c) for c in target_classes if str(c) != target_class]
    if others:
        other_class = others[0]
        for oc in others:
            scores_other_by_class[oc] = encoded[canon == oc]
        other_scores = scores_other_by_class.get(other_class)

    out = class_vs_neutral_metrics(
        target_scores,
        neutral_scores,
        target_class=target_class,
        scores_other=other_scores,
        other_class=other_class,
        scores_other_by_class=scores_other_by_class or None,
        n_bootstrap=n_bootstrap,
        youden_threshold=youden_threshold,
        extras={
            "encoding_direction": direction,
            "readout_kind": "class_vs_neutral",
            "backend": "encoder",
        },
        **dict(frozen or {}),
    )
    return out


def encoder_per_class_metrics(
    encoder_df: pd.DataFrame,
    target_classes: Sequence[str],
    *,
    full_encoder_df: Optional[pd.DataFrame] = None,
    encoding_direction: Optional[Dict[str, float]] = None,
    polarity_df: Optional[pd.DataFrame] = None,
    youden_thresholds: Optional[Dict[str, float]] = None,
    overlay_neutral_scores: Optional[Sequence[float]] = None,
    split: str = "test",
    n_bootstrap: int = 100,
    frozen_by_class: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute class-vs-neutral metrics for every class in ``target_classes``.

    ``frozen_by_class``: per-class :func:`frozen_decision_rules` output from the
    validation split (for the held-out report).
    """
    return {
        str(c): encoder_class_vs_neutral_metrics(
            encoder_df,
            target_class=str(c),
            target_classes=target_classes,
            full_encoder_df=full_encoder_df,
            encoding_direction=encoding_direction,
            polarity_df=polarity_df,
            youden_threshold=None if not youden_thresholds else youden_thresholds.get(str(c)),
            overlay_neutral_scores=overlay_neutral_scores,
            split=split,
            n_bootstrap=n_bootstrap,
            frozen=(frozen_by_class or {}).get(str(c)),
        )
        for c in target_classes
    }


def score_encoder_hidden(
    trainer,
    hidden: np.ndarray,
    *,
    batch_size: int = 64,
) -> Optional[np.ndarray]:
    """Run the trained bipolar encoder on residual rows ``(n, d)`` → ``(n,)``."""
    if hidden is None or getattr(hidden, "size", 0) == 0:
        return None
    arr = np.asarray(hidden, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mwg = trainer.get_model()
    g = mwg.gradiend
    input_dim = int(getattr(g, "input_dim", None) or g.encoder.in_features)
    if int(arr.shape[-1]) != input_dim:
        raise RuntimeError(
            f"ACTIEND residual dim {int(arr.shape[-1])} != encoder input_dim "
            f"{input_dim}; overlay must use the trainer signal_scope, not a "
            f"last-layer slice"
        )
    device = getattr(g, "device_encoder", None)
    if device is None:
        device = next(g.encoder.parameters()).device
    dtype = getattr(g, "torch_dtype", torch.float32)
    parts = []
    with torch.no_grad():
        for start in range(0, arr.shape[0], int(batch_size)):
            batch = torch.tensor(
                arr[start : start + int(batch_size)], device=device, dtype=dtype
            )
            try:
                enc = g.encoder(batch)
            except Exception:
                enc = torch.stack([g.forward_encoder(row) for row in batch], dim=0)
            parts.append(enc.detach().float().cpu().numpy())
    out = np.concatenate(parts, axis=0).reshape(len(arr), -1)
    return out[:, 0]


def _actiend_signal_scope(trainer):
    """Layer concat the encoder was trained on (not a last-layer slice)."""
    from gradiend.trainer.core.signals import SignalScope

    args = getattr(trainer, "args", None)
    scope = getattr(args, "signal_scope", None) if args is not None else None
    return scope if scope is not None else SignalScope.layers()


def actiend_shared_neutral_scores(
    trainer,
    texts: Sequence[str],
    *,
    activation_site: str = "prediction",
    excluded_words: Optional[Sequence[str]] = None,
    max_length: int = 256,
    batch_size: int = 8,
) -> Optional[np.ndarray]:
    """Score ACTIEND on the same non-pad non-excluded tokens as CAA/SAE.

    Token selection matches CAA/SAE. Residual layout is the trainer's
    ``signal_scope`` (the concat the encoder was trained on), so last-dim
    equals ``encoder.input_dim`` by construction.
    """
    if not texts:
        return None
    from caa_eval import (
        CAA_DEFAULT_MAX_LENGTH,
        _make_neutral_extractor,
        _neutral_batch,
        extract_activations_batched,
    )

    mwg = trainer.get_model()
    model = mwg.base_model
    tokenizer = trainer.tokenizer
    g = mwg.gradiend
    input_dim = int(getattr(g, "input_dim", None) or g.encoder.in_features)
    max_length = int(max_length or CAA_DEFAULT_MAX_LENGTH)
    extractor = _make_neutral_extractor(
        model,
        tokenizer,
        policy=str(activation_site),
        excluded_words=excluded_words,
        scope=_actiend_signal_scope(trainer),
    )
    batch = _neutral_batch(tokenizer, list(texts), max_length=max_length)
    hidden = extract_activations_batched(
        extractor, batch, batch_size=int(batch_size), desc="actiend neu tokens"
    )
    if hidden.size == 0:
        return None
    got = int(hidden.shape[-1])
    if got != input_dim:
        raise RuntimeError(
            f"ACTIEND token overlay: residual dim {got} != encoder input_dim "
            f"{input_dim}. Same signal_scope as training must match; this is a bug."
        )
    return score_encoder_hidden(trainer, hidden)


def fair_encoder_eval_from_trainer(
    trainer,
    *,
    target_classes: Sequence[str],
    max_size: Optional[int] = None,
    n_bootstrap: int = 100,
    backend: Optional[str] = None,
    excluded_words: Optional[Sequence[str]] = None,
    activation_site: Optional[str] = None,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Val-fit / test-report encoder eval (Youden, polarity, optional ACTIEND overlay).

    ``use_cache``: let the package reuse its cached per-row ``encoded_values_*.csv``
    (and component sidecar) for each split instead of re-encoding. Only safe for an
    already-trained checkpoint being *reloaded*: a fresh fit writes new weights into
    the same experiment dir, and a CSV left from the previous weights would then be
    read back as if it belonged to them. Default False for that reason.
    """
    from activation_protocol import normalize_activation_site
    from error_tracker import track_error
    from neutral_protocol import frame_for_split, texts_for_split

    classes = [str(c) for c in target_classes]
    if len(classes) < 1:
        raise ValueError("target_classes must be non-empty for encoder eval")
    site = normalize_activation_site(activation_site)

    _neutral_pool = getattr(getattr(trainer, "config", None), "neutral_data", None)

    def _neutral_slice(split: str) -> Optional[pd.DataFrame]:
        if not isinstance(_neutral_pool, pd.DataFrame) or max_size is None:
            return None
        return frame_for_split(_neutral_pool, split, fallback="all").head(int(max_size))

    def _eval(split: str, *, plot: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "plot": plot,
            "split": split,
            "return_df": True,
            "use_cache": bool(use_cache),
        }
        neutral_slice = _neutral_slice(split)
        if neutral_slice is not None:
            kw["neutral_data_df"] = neutral_slice
        if max_size is not None and str(backend) == "gradiend":
            kw["max_size"] = int(max_size)
        elif max_size is not None and str(backend) != "actiend":
            kw["max_size"] = int(max_size)
        elif max_size is not None and str(backend) == "actiend":
            # Token scoring multiplies rows; labeled templates can use a smaller cap.
            kw["max_size"] = int(max_size)
        return trainer.evaluate_encoder(**kw)

    val_enc = _eval("validation", plot=False)
    test_enc = _eval("test", plot=True)
    val_df = val_enc.get("encoder_df")
    test_df = test_enc.get("encoder_df")
    val_labeled = encoder_df_split_subset(val_df, split="validation")
    test_labeled = encoder_df_split_subset(test_df, split="test")

    encoding_direction = None
    try:
        mwg = trainer.get_model()
        encoding_direction = getattr(mwg, "feature_class_encoding_direction", None) or (
            mwg.gradiend.kwargs.get("feature_class_encoding_direction")
            if getattr(mwg, "gradiend", None) is not None
            else None
        )
    except Exception:
        encoding_direction = None
    polarity_df = (
        val_labeled
        if val_labeled is not None and not getattr(val_labeled, "empty", True)
        else None
    )
    if encoding_direction:
        polarity_src = "trainer"
    elif polarity_df is not None:
        polarity_src = "validation_means"
    else:
        polarity_src = "default_+1"
    print(f"  encoder polarity: {polarity_src}", flush=True)

    overlay_val = overlay_test = None
    if str(backend) == "actiend":
        neu_df = getattr(getattr(trainer, "config", None), "neutral_data", None)
        if isinstance(neu_df, pd.DataFrame):
            val_texts = texts_for_split(
                neu_df, "validation", max_rows=max_size, fallback="all"
            )
            test_texts = texts_for_split(neu_df, "test", max_rows=max_size, fallback="all")
            overlay_val = actiend_shared_neutral_scores(
                trainer,
                val_texts,
                activation_site=site,
                excluded_words=excluded_words,
            )
            overlay_test = actiend_shared_neutral_scores(
                trainer,
                test_texts,
                activation_site=site,
                excluded_words=excluded_words,
            )
            if overlay_val is not None:
                print(
                    f"  ACTIEND neutrals: shared token H "
                    f"(val_tok={overlay_val.size} test_tok="
                    f"{0 if overlay_test is None else overlay_test.size})",
                    flush=True,
                )

    val_metrics: Dict[str, Any] = {}
    if val_labeled is not None and not getattr(val_labeled, "empty", True):
        val_metrics = encoder_per_class_metrics(
            val_labeled,
            classes,
            full_encoder_df=val_df,
            encoding_direction=encoding_direction,
            polarity_df=polarity_df,
            overlay_neutral_scores=overlay_val,
            split="validation",
            n_bootstrap=0,
        )
    youden_thresholds = {
        str(c): m.get("youden_threshold")
        for c, m in val_metrics.items()
        if isinstance(m, dict) and m.get("youden_threshold") is not None
    } or None
    frozen_by_class = {
        str(c): frozen_decision_rules(m)
        for c, m in val_metrics.items()
        if isinstance(m, dict)
    }

    package_metrics = trainer.get_encoder_metrics(encoder_df=test_df) if test_df is not None else {}
    if not package_metrics:
        package_metrics = {
            k: test_enc.get(k)
            for k in ("correlation", "mean_by_class", "roc_auc", "balanced_accuracy", "cohens_d")
            if k in test_enc
        }

    joint: Dict[str, Any] = {}
    if len(classes) >= 2 and test_labeled is not None:
        # Prefer this encoder's own trained poles over classes[0]/[1]
        # (the full task class list), which need not be this pair --
        # see _resolve_pair_classes.
        joint_a, joint_b = _resolve_pair_classes(encoding_direction, classes)
        if joint_a is None or joint_b is None:
            joint_a, joint_b = classes[0], classes[1]
        joint = encoder_specificity(
            test_labeled,
            class_a=joint_a,
            class_b=joint_b,
            full_encoder_df=test_df,
        )

    per_class: Dict[str, Any] = {}
    if test_labeled is not None:
        per_class = encoder_per_class_metrics(
            test_labeled,
            classes,
            full_encoder_df=test_df,
            encoding_direction=encoding_direction,
            polarity_df=polarity_df,
            youden_thresholds=youden_thresholds,
            overlay_neutral_scores=overlay_test,
            split="test",
            n_bootstrap=int(n_bootstrap),
            frozen_by_class=frozen_by_class,
        )

    per_component: Dict[str, Any] = {}
    component_keys: list = []
    component_df = test_enc.get("component_df")
    val_component_df = val_enc.get("component_df")
    try:
        component_keys = list_component_keys(component_df) if component_df is not None else []
        if component_df is not None and not getattr(component_df, "empty", True):
            print(f"  components: {len(component_keys)} tensor parts", flush=True)
            for ck in component_keys:
                part = ck["part"]
                val_comp = {}
                if val_component_df is not None and not getattr(val_component_df, "empty", True):
                    val_comp = encoder_component_per_class_metrics(
                        val_component_df,
                        classes,
                        component_id=ck.get("component_id"),
                        component_label=ck.get("component_label"),
                        encoding_direction=encoding_direction,
                        polarity_df=polarity_df,
                        split="validation",
                        n_bootstrap=0,
                    )
                taus = {
                    str(c): m.get("youden_threshold")
                    for c, m in (val_comp or {}).items()
                    if isinstance(m, dict) and m.get("youden_threshold") is not None
                } or None
                frozen_comp = {
                    str(c): frozen_decision_rules(m)
                    for c, m in (val_comp or {}).items()
                    if isinstance(m, dict)
                }
                per_component[part] = encoder_component_per_class_metrics(
                    component_df,
                    classes,
                    component_id=ck.get("component_id"),
                    component_label=ck.get("component_label"),
                    encoding_direction=encoding_direction,
                    polarity_df=polarity_df,
                    youden_thresholds=taus,
                    split="test",
                    n_bootstrap=int(n_bootstrap),
                    frozen_by_class=frozen_comp,
                )
    except FrozenRulesUnavailable:
        raise
    except Exception as exc:
        track_error(exc, context="component encode")

    # Held-out Spec_n/Excl must come from validation-frozen rules; never report an
    # oracle (test-fit) number because validation was missing for some class.
    require_frozen_rules(per_class, context="encoder eval (test)")
    for _part, _comp in per_component.items():
        require_frozen_rules(_comp, context=f"encoder eval component {_part!r} (test)")

    return {
        "encoder_metrics": package_metrics,
        "readout_metrics": joint,
        "per_class_readouts": per_class,
        "per_component_readouts": per_component,
        "component_keys": component_keys,
        "encoder_df": test_df,
        "val_encoder_df": val_df,
        "youden_thresholds": youden_thresholds,
        "encoding_direction": encoding_direction,
    }






def method_part_id(backend: str, class_id: str, part: str) -> str:
    """``actiend:M:L11_tok_all_gate_encoder_direction``, ``sae:F:k1``, …"""
    return f"{backend}:{class_id}:{part}"


# SAE feature-selection sites (independent of ACTIEND/CAA ``activation_site``).
# ``prediction`` = filled span (fair encode vs ACTIEND/CAA) → ``sae:{cls}:…``.
# ``pre_prediction`` = last-context residual → ``sae_pre:{cls}:…`` (separate family).
SAE_DEFAULT_SELECT_SITES: Tuple[str, ...] = ("prediction", "pre_prediction")
SAE_BACKEND_FILLED = "sae"
SAE_BACKEND_PRE = "sae_pre"
SAE_BACKEND_BY_SITE: Dict[str, str] = {
    "prediction": SAE_BACKEND_FILLED,
    "pre_prediction": SAE_BACKEND_PRE,
}
# Legacy suffix map (old ``sae:M:k1_pre`` dumps); do not emit new ids with these.


def sae_backend_for_site(site: Optional[str] = None) -> str:
    """Method-id backend for a selection site: ``sae`` vs ``sae_pre``."""
    from activation_protocol import normalize_activation_site

    s = normalize_activation_site(site or "prediction")
    return SAE_BACKEND_BY_SITE.get(s, f"sae_{s}")




def normalize_sae_select_sites(sites: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    from activation_protocol import normalize_activation_site

    raw = list(sites) if sites else list(SAE_DEFAULT_SELECT_SITES)
    out: List[str] = []
    seen = set()
    for item in raw:
        site = normalize_activation_site(item)
        if site not in seen:
            seen.add(site)
            out.append(site)
    if not out:
        out = ["prediction"]
    return tuple(out)




def sae_part_for_site(part: str, site: Optional[str] = None) -> str:
    """Readout tail only — site is encoded in the backend (``sae`` vs ``sae_pre``)."""
    return str(part)




def sae_joint_method_id(class_id: Optional[str] = None, *, site: Optional[str] = None) -> str:
    """``sae:joint`` / ``sae:joint:M``; pre site → ``sae_pre:joint`` / ``sae_pre:joint:M``."""
    backend = sae_backend_for_site(site)
    if class_id is None:
        return f"{backend}:joint"
    return f"{backend}:joint:{class_id}"


def iter_sae_site_payloads(sae_raw: Optional[Mapping[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Yield ``(site, site_raw)`` for every SAE encode site.

    Causal must run the **same** method suite on each payload (k1 / kstar /
    fixed-k / opp-fire / Arad / JH / per-layer / all_k / joint). Top-level
    ``sae_raw`` is the primary site (usually filled ``prediction``); extra
    sites live under ``by_site``. Live ``_model`` handles inherit from parent.
    """
    if not isinstance(sae_raw, Mapping) or sae_raw.get("error"):
        return []
    by_site = sae_raw.get("by_site")
    inherit_keys = ("_model", "_tokenizer", "_sae_by_layer", "_hf_module_by_layer")
    if isinstance(by_site, Mapping) and by_site:
        out: List[Tuple[str, Dict[str, Any]]] = []
        for site, payload in by_site.items():
            if not isinstance(payload, Mapping) or payload.get("error"):
                continue
            merged = dict(payload)
            for key in inherit_keys:
                if merged.get(key) is None and sae_raw.get(key) is not None:
                    merged[key] = sae_raw[key]
            out.append((str(site), merged))
        if out:
            return out
    site = str(sae_raw.get("act_policy") or sae_raw.get("activation_site") or "prediction")
    return [(site, dict(sae_raw))]


def strip_sae_raw_for_json(sae_raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Drop live model/SAE handles (including nested ``by_site``) for results.json."""
    skip = {
        "_model",
        "_tokenizer",
        "_sae",
        "_sae_by_layer",
        "_hf_module_by_layer",
        "_test_latents",
        "_neutral_latents",
        "_val_neutral_latents",
        "_val_latents",
        "_val_labels",
        "_test_labels",
        "_all_k_compact",
    }
    if not isinstance(sae_raw, Mapping):
        return {}
    out = {k: v for k, v in sae_raw.items() if k not in skip}
    nested = out.get("by_site")
    if isinstance(nested, Mapping):
        out["by_site"] = {
            str(site): {k: v for k, v in payload.items() if k not in skip}
            if isinstance(payload, Mapping)
            else payload
            for site, payload in nested.items()
        }
    return out


# Canonical suffixes so defaults are never silent in method ids.
# ACTIEND causal-policy defaults live in study/causal_policies.py (their real home;
# they never had anything to do with SAE). Re-exported here for back-compat imports.
SAE_KSTAR_PART = "kstar"  # val-selected k* bag (encoder + causal)




def sae_causal_id(
    class_id: str,
    *,
    part: str = "k1",
    token_selector: str = "all",
    site: Optional[str] = None,
) -> str:
    """SAE causal id aligned with encoder id when token_selector is the default ``all``.

    Examples: ``sae:M:k1``, ``sae:F:kstar``, ``sae:M:all_k1``, ``sae:M:k1_tok_prediction``,
    ``sae_pre:M:k1`` (classical last-context selection).
    """
    backend = sae_backend_for_site(site)
    part = sae_part_for_site(part, site)
    if str(token_selector) == "all":
        return method_part_id(backend, class_id, part)
    return method_part_id(backend, class_id, f"{part}_tok_{token_selector}")


def sae_encoder_kstar_id(class_id: str, *, site: Optional[str] = None) -> str:
    """Encoder fair-comparison id for val-selected k* bag: ``sae:M:kstar`` / ``sae_pre:M:kstar``."""
    return method_part_id(
        sae_backend_for_site(site), class_id, sae_part_for_site(SAE_KSTAR_PART, site)
    )


def sae_encoder_all_k_id(class_id: str, k: int, *, site: Optional[str] = None) -> str:
    """Multi-layer SAE bag id: ``sae:M:all_k1`` / ``sae_pre:M:all_k1``."""
    return method_part_id(
        sae_backend_for_site(site), class_id, sae_part_for_site(f"all_k{int(k)}", site)
    )


def sae_all_layers_class_vs_neutral_metrics(
    layer_payloads: Sequence[Dict[str, Any]],
    *,
    target_class: str,
    k: int,
    target_classes: Sequence[str],
    n_bootstrap: int = 100,
    youden_threshold: Optional[float] = None,
    frozen: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Sum top-k class-feature activations across layers, then class-vs-neutral metrics.

    ``frozen``: validation-fit Spec_n/Excl rules (:func:`frozen_decision_rules`).

    Each ``layer_payloads`` entry needs:
      ``latents`` (N×d), ``neutral_latents`` (Nn×d), ``labels`` (N,),
      ``features`` (ordered feature indices for ``target_class`` at that layer).
    """
    k = max(1, int(k))
    target_class = str(target_class)
    # Do not rely solely on the configured target-class list.  Several
    # one-target SAE tasks still contain rival labelled rows in ``labels``;
    # omitting them made ``roc_auc_other`` disappear even though the other
    # methods computed it from the same data.
    if not layer_payloads:
        return {"error": "no layer payloads"}

    scores_pos_parts: List[np.ndarray] = []
    scores_neu_parts: List[np.ndarray] = []
    scores_other_parts: Dict[str, List[np.ndarray]] = {}
    feats_by_layer: Dict[str, List[int]] = {}
    n_layers_used = 0
    labels_ref = None
    neu_len_ref: Optional[int] = None

    # One-pole factual labels can contain only the target; rivals then arrive
    # in separately encoded counterfactual columns.
    other_order: List[str] = []
    for payload in layer_payloads:
        payload_labels = payload.get("labels")
        for candidate in np.asarray(
            payload_labels if payload_labels is not None else []
        ).astype(str):
            if candidate != target_class and candidate not in other_order:
                other_order.append(candidate)
        for candidate in (payload.get("rival_cols_by_class") or {}):
            candidate = str(candidate)
            if candidate != target_class and candidate not in other_order:
                other_order.append(candidate)
    scores_other_parts = {other: [] for other in other_order}

    for payload in layer_payloads:
        labels = np.asarray(payload.get("labels")).astype(str)
        layer = payload.get("layer")
        latent_cols = payload.get("latent_cols")
        neu_cols = payload.get("neutral_cols")
        latents = payload.get("latents")
        neu = payload.get("neutral_latents")
        feats = list(payload.get("features") or [])
        feature_indices = payload.get("feature_indices")
        if latent_cols is not None and neu_cols is not None:
            k_use = min(int(k), int(latent_cols.shape[1]))
            if k_use < 1:
                continue
        elif latents is None or neu is None or not feats[:k]:
            continue
        else:
            feats = feats[:k]
            latents = np.asarray(latents)
            neu = np.asarray(neu)
            k_use = int(k)
        if labels_ref is None:
            labels_ref = labels
            # Do not rely solely on the configured target-class list. Several
            # one-target SAE tasks still contain rival labelled rows.
            observed_classes = [str(c) for c in np.unique(labels_ref)]
            others = list(other_order) or [
                c for c in observed_classes if c != target_class
            ]
        elif len(labels) != len(labels_ref) or not np.array_equal(labels, labels_ref):
            return {"error": "mismatched labels across SAE layers"}
        # Neutral rows come from a separate extraction/subsample pass per layer
        # (not covered by the ``labels_ref`` check above) — a layer whose
        # neutral row count drifts would otherwise blow up the ``np.stack``
        # below with an opaque "all input arrays must have the same shape".
        neu_len = int((neu_cols if neu_cols is not None else neu).shape[0])
        if neu_len_ref is None:
            neu_len_ref = neu_len
        elif neu_len != neu_len_ref:
            return {
                "error": (
                    f"mismatched neutral row count across SAE layers "
                    f"(layer {layer}: {neu_len} vs {neu_len_ref})"
                )
            }
        if latent_cols is not None and neu_cols is not None:
            scores_pos_parts.append(latent_cols[:, :k_use].sum(axis=1))
            scores_neu_parts.append(neu_cols[:, :k_use].sum(axis=1))
            for oc in others:
                rival_cols = (payload.get("rival_cols_by_class") or {}).get(oc)
                if rival_cols is not None:
                    scores_other_parts[oc].append(
                        np.asarray(rival_cols)[:, :k_use].sum(axis=1)
                    )
                elif np.any(labels == oc):
                    scores_other_parts[oc].append(
                        latent_cols[labels == oc][:, :k_use].sum(axis=1)
                    )
            if feature_indices is not None:
                feats_by_layer[str(layer)] = [int(f) for f in feature_indices[:k_use]]
        else:
            scores_pos_parts.append(latents[:, feats].sum(axis=1))
            scores_neu_parts.append(neu[:, feats].sum(axis=1))
            for oc in others:
                rival_cols = (payload.get("rival_cols_by_class") or {}).get(oc)
                if rival_cols is not None:
                    scores_other_parts[oc].append(np.asarray(rival_cols)[:, feats].sum(axis=1))
                elif np.any(labels == oc):
                    scores_other_parts[oc].append(latents[labels == oc][:, feats].sum(axis=1))
            feats_by_layer[str(layer)] = [int(f) for f in feats]
        n_layers_used += 1

    if n_layers_used == 0 or labels_ref is None:
        return {"error": "no usable layer features/latents"}

    scores_all = np.sum(np.stack(scores_pos_parts, axis=0), axis=0)
    scores_neu = np.sum(np.stack(scores_neu_parts, axis=0), axis=0)
    scores_target = scores_all[labels_ref == target_class]
    scores_other_by = {
        oc: np.sum(np.stack(parts, axis=0), axis=0)
        for oc, parts in scores_other_parts.items()
        if len(parts) == n_layers_used
    }
    other = others[0] if others else None
    out = class_vs_neutral_metrics(
        scores_target,
        scores_neu,
        target_class=target_class,
        scores_other=scores_other_by.get(other) if other else None,
        other_class=other,
        scores_other_by_class=scores_other_by or None,
        n_bootstrap=n_bootstrap,
        youden_threshold=youden_threshold,
        **dict(frozen or {}),
        extras={
            "readout_k": k,
            "readout_kind": "all_layers_sum_topk",
            "features_by_layer": feats_by_layer,
            "n_layers": n_layers_used,
        },
    )
    out["readout_k"] = int(k)
    out["component_part"] = f"all_k{int(k)}"
    out["n_layers"] = int(n_layers_used)
    out["features_by_layer"] = feats_by_layer
    out["readout_features"] = [
        f for feats in feats_by_layer.values() for f in feats
    ]
    return out


def short_component_part(component_id: Any, component_label: Any = None) -> str:
    """
    Map a GRADIEND/ACTIEND component id/label to a short method suffix.

    Examples:
      ``activation:transformer.h.11`` / ``transformer.h.11`` → ``L11``
      ``transformer.h.11.attn.c_attn.weight`` → ``h11.attn.c_attn.weight``
    """
    import re

    raw = str(component_label or component_id or "").strip()
    if raw.startswith("activation:"):
        raw = raw[len("activation:") :]
    # Whole GPT-2 block / resid site → L{n}
    m = re.fullmatch(r"(?:transformer\.)?h\.(\d+)", raw)
    if m:
        return f"L{int(m.group(1))}"
    m = re.fullmatch(r".*\.(?:layer|layers)\.(\d+)", raw)
    if m:
        return f"L{int(m.group(1))}"
    if re.fullmatch(r"L?\d+", raw, flags=re.IGNORECASE):
        return f"L{int(raw.lstrip('Ll'))}"
    # Nested GPT-2 module under a block
    raw = re.sub(r"transformer\.h\.(\d+)", r"h\1", raw)
    for prefix in ("base_model.", "model.", "transformer."):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    return raw.replace("/", ".")[:80] or "comp"


def list_component_keys(component_df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Unique components as ``{component_id, component_label, part}``."""
    if component_df is None or getattr(component_df, "empty", True):
        return []
    id_col = "component_id" if "component_id" in component_df.columns else None
    label_col = "component_label" if "component_label" in component_df.columns else None
    if id_col is None and label_col is None:
        return []
    keys = []
    seen = set()
    for _, row in component_df.iterrows():
        cid = row[id_col] if id_col else None
        lab = row[label_col] if label_col else None
        part = short_component_part(cid, lab)
        key = (str(cid), part)
        if key in seen:
            continue
        seen.add(key)
        keys.append(
            {
                "component_id": None if cid is None or (isinstance(cid, float) and np.isnan(cid)) else str(cid),
                "component_label": None if lab is None or (isinstance(lab, float) and np.isnan(lab)) else str(lab),
                "part": part,
            }
        )
    return keys


def filter_component_df(
    component_df: pd.DataFrame,
    *,
    component_id: Optional[str] = None,
    component_label: Optional[str] = None,
) -> pd.DataFrame:
    """Rows for one component (id and/or label match)."""
    mask = pd.Series(False, index=component_df.index)
    if component_id is not None and "component_id" in component_df.columns:
        mask = mask | (component_df["component_id"].astype(str) == str(component_id))
    if component_label is not None and "component_label" in component_df.columns:
        mask = mask | (component_df["component_label"].astype(str) == str(component_label))
    return component_df.loc[mask].copy()


def encoder_component_per_class_metrics(
    component_df: pd.DataFrame,
    target_classes: Sequence[str],
    *,
    component_id: Optional[str] = None,
    component_label: Optional[str] = None,
    encoding_direction: Optional[Dict[str, float]] = None,
    polarity_df: Optional[pd.DataFrame] = None,
    youden_thresholds: Optional[Dict[str, float]] = None,
    split: str = "test",
    n_bootstrap: int = 100,
    frozen_by_class: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Class-vs-neutral metrics for one tensor/layer component."""
    subset = filter_component_df(
        component_df, component_id=component_id, component_label=component_label
    )
    if subset.empty:
        return {
            str(c): {"error": "no component rows"}
            for c in target_classes
        }
    split_subset = encoder_df_split_subset(subset, split=split)
    return encoder_per_class_metrics(
        split_subset if split_subset is not None else subset,
        target_classes,
        full_encoder_df=subset,
        encoding_direction=encoding_direction,
        polarity_df=polarity_df,
        youden_thresholds=youden_thresholds,
        split=split,
        n_bootstrap=n_bootstrap,
        frozen_by_class=frozen_by_class,
    )


def sae_sparse_readout_metrics(
    latents: np.ndarray,
    labels: Sequence[str],
    feature_indices: Sequence[int],
    *,
    k: int = 5,
    class_a: str = "M",
    class_b: str = "F",
    neutral_latents: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Scalar SAE readout = sum of top-k selected feature activations (test-set protocol)."""
    if not len(feature_indices):
        return {"error": "no features"}
    k = max(1, min(int(k), len(feature_indices)))
    feats = list(feature_indices[:k])
    scores = latents[:, feats].sum(axis=1)
    neutral_scores = None
    if neutral_latents is not None and len(neutral_latents):
        neutral_scores = neutral_latents[:, feats].sum(axis=1)
    out = readout_metrics(
        scores,
        labels,
        class_a=class_a,
        class_b=class_b,
        neutral_scores=neutral_scores,
    )
    if "error" not in out:
        out["readout_k"] = k
        out["readout_features"] = feats
    return out


def class_vs_neutral_metrics(
    scores_target: Sequence[float],
    scores_neutral: Sequence[float],
    *,
    target_class: str,
    scores_other: Optional[Sequence[float]] = None,
    other_class: Optional[str] = None,
    scores_other_by_class: Optional[Dict[str, Sequence[float]]] = None,
    n_bootstrap: int = 100,
    seed: int = 0,
    extras: Optional[Dict[str, Any]] = None,
    youden_threshold: Optional[float] = None,
    neutral_youden_threshold: Optional[float] = None,
    rival_youden_threshold_by_class: Optional[Dict[str, float]] = None,
    neutral_youden_sign: Optional[float] = None,
    rival_youden_sign_by_class: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generic class-vs-neutral readout metrics (any backend / any feature class).

    **Frozen (validation-fit) decision rules.** For held-out reporting pass the
    dict from :func:`frozen_decision_rules` (fit on validation) as
    ``neutral_youden_threshold``/``neutral_youden_sign`` and
    ``rival_youden_threshold_by_class``/``rival_youden_sign_by_class``. A
    threshold is only meaningful together with the orientation sign it was fit
    under, so supplying a threshold without its sign raises instead of
    re-deriving the sign from the (test) scores. In frozen-rival mode every
    rival present here must have a frozen rule; a missing one raises rather than
    silently refitting on the evaluated split. With nothing supplied both rules
    are fit on the scores handed in (``*_youden_threshold_source == "eval"``) —
    correct for validation, an oracle (optimistic) for a test report.

    ``scores_target``: scalar feature response on target-class examples
    ``scores_neutral``: same feature on neutral / off-distribution examples
    ``scores_other`` / ``scores_other_by_class``: optional off-target labeled classes

    Reports (scale-fair primary metrics):
      - ``roc_auc`` / ``roc_auc_neutral``: AUROC(target vs neutral)
      - ``roc_auc_other`` / ``min_pairwise_auroc``: min AUROC(target vs each rival)
      - ``balanced_accuracy``, ``cohens_d``: target vs neutral
      - ``neutral_specificity`` / ``specificity``: TNR of neutrals at a
        threshold fit **only** on target-vs-neutral (``neutral_youden_threshold``,
        overridable e.g. with a validation-fit value).
      - ``class_exclusivity``: min rival TNR, each rival evaluated at **its own**
        target-vs-rival threshold (``rival_youden_threshold_by_class``), not a
        threshold shared with the neutral comparison.
      - ``youden_threshold`` / ``youden_j``: kept as a diagnostic — the single
        threshold that would be picked against rivals+neutrals *pooled*
        together. Do not use this for specificity/exclusivity: when a rival
        collapses onto the target (encoder can't tell the two classes apart),
        this pooled threshold gets dragged to the target/rival cluster and can
        report near-zero neutral specificity even when target-vs-neutral is
        cleanly separable (``roc_auc_neutral`` near 1). See CLAUDE.md,
        "class_exclusivity/specificity shared one Youden threshold...".

    Magnitude soft-scores are kept as ``*_mag`` diagnostics only.
    """
    pos = np.asarray(scores_target, dtype=float)
    neu = np.asarray(scores_neutral, dtype=float)
    pos = pos[np.isfinite(pos)]
    neu = neu[np.isfinite(neu)]
    if pos.size == 0 or neu.size == 0:
        return {"error": "insufficient samples for class-vs-neutral"}
    scores = np.concatenate([pos, neu], axis=0)
    labels = [str(target_class)] * int(pos.size) + ["neutral"] * int(neu.size)
    out = readout_metrics(
        scores,
        labels,
        class_a=str(target_class),
        class_b="neutral",
        neutral_scores=None,
    )
    if "error" in out:
        return out
    boot = bootstrap_roc_auc(
        scores,
        labels,
        class_a=str(target_class),
        class_b="neutral",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    out["roc_auc_boot_mean"] = boot.get("roc_auc_boot_mean")
    out["roc_auc_boot_std"] = boot.get("roc_auc_boot_std")
    out["readout_kind"] = "class_vs_neutral"
    out["target_class"] = str(target_class)
    out["mean_target"] = float(pos.mean())
    out["mean_neutral"] = float(neu.mean())
    out[f"n_{target_class}"] = int(pos.size)
    out["n_neutral"] = int(neu.size)
    # Explicit aliases: headline roc_auc is always target vs neutral.
    out["roc_auc_neutral"] = out.get("roc_auc")

    half = 0.5 * abs(float(pos.mean()) - float(neu.mean()))
    # Legacy magnitude soft-score (not cross-method fair; diagnostic only).
    neutral_spread = float(np.abs(neu).mean())  # inactive anchor 0
    out["neutral_spread"] = neutral_spread
    out["neutral_specificity_mag"] = float(half / (half + neutral_spread + 1e-8))

    others: Dict[str, np.ndarray] = {}
    if scores_other_by_class:
        for k, v in scores_other_by_class.items():
            arr = np.asarray(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                others[str(k)] = arr
    elif scores_other is not None:
        arr = np.asarray(scores_other, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            others[str(other_class) if other_class is not None else "other"] = arr

    # Diagnostic-only pooled τ: positives = target; negatives = rivals ∪
    # neutrals together. NOT used for specificity/exclusivity below (see
    # docstring) — kept so youden_j/youden_threshold remain inspectable.
    neg_parts: List[np.ndarray] = [neu]
    if others:
        neg_parts.extend(others.values())
    neg_all = np.concatenate(neg_parts, axis=0)
    if youden_threshold is None:
        tau_joint, _fitted_j, sign_joint = _youden_threshold(pos, neg_all)
        out["youden_threshold_source"] = "eval"
    else:
        tau_joint = float(youden_threshold)
        sign_joint = _orientation_sign(pos, neg_all)
        out["youden_threshold_source"] = "provided"
    if tau_joint is not None:
        tpr_joint = float((sign_joint * pos > tau_joint).mean())
        tnr_joint = float((sign_joint * neg_all <= tau_joint).mean())
        youden_j_joint = float(tpr_joint + tnr_joint - 1.0)
    else:
        youden_j_joint = None
    out["youden_threshold"] = tau_joint
    out["youden_j"] = youden_j_joint
    out["youden_negatives"] = "other+neutral" if others else "neutral"

    # Neutral-only τ: decoupled from rival separation. This is what
    # specificity/neutral_specificity are actually computed from.
    if neutral_youden_threshold is None:
        tau_neu, _j_neu, sign_neu = _youden_threshold(pos, neu)
        out["neutral_youden_threshold_source"] = "eval"
    else:
        if neutral_youden_sign is None:
            raise FrozenRulesUnavailable(
                "neutral_youden_threshold was provided without neutral_youden_sign; "
                "the orientation must be frozen with the threshold, not re-derived "
                "from the evaluated split"
            )
        tau_neu = float(neutral_youden_threshold)
        sign_neu = 1.0 if float(neutral_youden_sign) >= 0 else -1.0
        out["neutral_youden_threshold_source"] = "provided"
    out["neutral_youden_threshold"] = tau_neu
    out["neutral_youden_sign"] = sign_neu
    if tau_neu is not None:
        out["neutral_tnr"] = float((sign_neu * neu <= tau_neu).mean())
        out["neutral_specificity"] = out["neutral_tnr"]
        out["specificity"] = out["neutral_tnr"]
        out["target_tpr"] = float((sign_neu * pos > tau_neu).mean())
        out["neutral_youden_j"] = float(out["target_tpr"] + out["neutral_tnr"] - 1.0)
    else:
        out["neutral_tnr"] = None
        out["neutral_specificity"] = out["neutral_specificity_mag"]
        out["specificity"] = out["neutral_specificity"]
        out["target_tpr"] = None
        out["neutral_youden_j"] = None

    if others:
        exclusivities_mag = []
        exclusivities_tnr = []
        pairwise_aucs = []
        off_by = {}
        tnr_by = {}
        rival_tau_by: Dict[str, Optional[float]] = {}
        rival_sign_by: Dict[str, float] = {}
        for oc, oarr in others.items():
            out[f"mean_{oc}"] = float(oarr.mean())
            off = float(np.maximum(oarr, 0.0).mean())
            off_by[oc] = off
            exclusivities_mag.append(float(half / (half + off + 1e-8)))
            # Rival-only τ, one per rival class: decoupled from neutral
            # separation and from other rivals, so one collapsed rival can't
            # silently zero out an otherwise-clean neutral specificity (or
            # vice versa). Overridable per class for a validation-fit τ.
            if rival_youden_threshold_by_class is not None:
                provided_tau = rival_youden_threshold_by_class.get(oc)
                provided_sign = (rival_youden_sign_by_class or {}).get(oc)
                if provided_tau is None or provided_sign is None:
                    raise FrozenRulesUnavailable(
                        f"frozen rival rule missing for class {oc!r} (target "
                        f"{target_class!r}): validation produced threshold="
                        f"{provided_tau!r}, sign={provided_sign!r}. Refusing to "
                        "refit the threshold on the evaluated split."
                    )
                tau_oc: Optional[float] = float(provided_tau)
                sign_oc = 1.0 if float(provided_sign) >= 0 else -1.0
                out["rival_youden_threshold_source"] = "provided"
            else:
                tau_oc, _j_oc, sign_oc = _youden_threshold(pos, oarr)
                out["rival_youden_threshold_source"] = "eval"
            rival_tau_by[oc] = tau_oc
            rival_sign_by[oc] = sign_oc
            if tau_oc is not None:
                tnr = float((sign_oc * oarr <= tau_oc).mean())
                tnr_by[oc] = tnr
                exclusivities_tnr.append(tnr)
            # Target vs rival AUROC on labeled classes only.
            s = np.concatenate([pos, oarr], axis=0)
            lab = [str(target_class)] * int(pos.size) + [oc] * int(oarr.size)
            auc = _roc_auc(s, np.asarray(lab), class_a=str(target_class))
            if isinstance(auc, (int, float)):
                pairwise_aucs.append(float(auc))
        out["other_classes"] = list(others.keys())
        out["offtarget_positive_mean_by_class"] = off_by
        out["other_tnr_by_class"] = tnr_by
        out["rival_youden_threshold_by_class"] = rival_tau_by
        out["rival_youden_sign_by_class"] = rival_sign_by
        out["class_exclusivity_mag"] = (
            float(min(exclusivities_mag)) if exclusivities_mag else None
        )
        # Primary exclusivity: min rival TNR, each rival at its own τ.
        out["class_exclusivity"] = (
            float(min(exclusivities_tnr)) if exclusivities_tnr else out["class_exclusivity_mag"]
        )
        out["roc_auc_other"] = float(min(pairwise_aucs)) if pairwise_aucs else None
        out["min_pairwise_auroc"] = out["roc_auc_other"]
        out["mean_other"] = float(np.mean([float(a.mean()) for a in others.values()]))
        if other_class is not None:
            out["other_class"] = str(other_class)
        out["offtarget_positive_mean"] = float(max(off_by.values())) if off_by else None
    else:
        out["class_exclusivity"] = None
        out["class_exclusivity_mag"] = None
        out["roc_auc_other"] = None
        out["min_pairwise_auroc"] = None

    if extras:
        out.update(extras)
    return out


class FrozenRulesUnavailable(ValueError):
    """A held-out Spec_n/Excl would have been fit on the split it reports."""


def require_frozen_rules(readouts: Any, *, context: str) -> None:
    """Raise unless every class-vs-neutral readout used validation-frozen rules.

    The backstop for the whole held-out report: whichever caller lost or never had
    its validation readout (empty validation split, a class absent from validation,
    an errored validation readout, a forgotten call site) ends up here with
    ``*_youden_threshold_source == "eval"``. Reporting that would silently hand out
    a test-optimised (oracle) Spec_n/Excl, so it is an error, not a fallback.
    """
    bad = unfrozen_rule_readouts(readouts)
    if bad:
        shown = ", ".join(bad[:6]) + (f", ... (+{len(bad) - 6})" if len(bad) > 6 else "")
        raise FrozenRulesUnavailable(
            f"{context}: {len(bad)} readout(s) have no validation-frozen Spec_n/Excl "
            f"rule [{shown}]. Validation produced no usable readout for them (empty "
            "validation split, class/rival absent from validation, or a validation "
            "error), and refitting the threshold on test would report an oracle "
            "number. Fix the validation data/config, do not fall back."
        )


def frozen_decision_rules(val_metrics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keyword arguments that freeze validation-fit Spec_n / Excl decision rules.

    ``val_metrics`` is the :func:`class_vs_neutral_metrics` output computed on the
    **validation** split. The returned dict is splatted into a test-split call so
    the neutral and per-rival thresholds *and their orientation signs* are reused
    instead of refit on test. Returns ``{}`` when validation produced no usable
    rule (error readout / no neutral fit); the test call then reports
    ``*_youden_threshold_source == "eval"``, which stays visible in the output.
    """
    if not isinstance(val_metrics, Mapping):
        return {}
    tau = val_metrics.get("neutral_youden_threshold")
    sign = val_metrics.get("neutral_youden_sign")
    if tau is None or sign is None:
        return {}
    out: Dict[str, Any] = {
        "neutral_youden_threshold": float(tau),
        "neutral_youden_sign": float(sign),
    }
    rival_tau = val_metrics.get("rival_youden_threshold_by_class")
    if rival_tau is not None:
        rival_sign = val_metrics.get("rival_youden_sign_by_class") or {}
        keep = {k for k, v in rival_tau.items() if v is not None and k in rival_sign}
        out["rival_youden_threshold_by_class"] = {str(k): float(rival_tau[k]) for k in keep}
        out["rival_youden_sign_by_class"] = {str(k): float(rival_sign[k]) for k in keep}
    return out


def sae_class_vs_neutral_metrics(
    class_latents: np.ndarray,
    neutral_latents: np.ndarray,
    feature_indices: Sequence[int],
    *,
    target_class: str,
    k: int = 1,
    other_latents: Optional[np.ndarray] = None,
    other_class: Optional[str] = None,
    other_latents_by_class: Optional[Dict[str, np.ndarray]] = None,
    n_bootstrap: int = 100,
    seed: int = 0,
    youden_threshold: Optional[float] = None,
    frozen: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """SAE adapter: sum of top-k class features, then :func:`class_vs_neutral_metrics`.

    ``frozen``: validation-fit Spec_n/Excl rules (:func:`frozen_decision_rules`)."""
    if not len(feature_indices):
        return {"error": "no features"}
    if neutral_latents is None or len(neutral_latents) == 0:
        return {"error": "neutral latents required for class-vs-neutral metrics"}
    k = max(1, min(int(k), len(feature_indices)))
    feats = list(feature_indices[:k])
    scores_pos = class_latents[:, feats].sum(axis=1)
    scores_neu = neutral_latents[:, feats].sum(axis=1)
    scores_other = None
    scores_other_by_class: Dict[str, np.ndarray] = {}
    if other_latents_by_class:
        for oc, lat in other_latents_by_class.items():
            if lat is not None and len(lat):
                scores_other_by_class[str(oc)] = np.asarray(lat)[:, feats].sum(axis=1)
    elif other_latents is not None and len(other_latents):
        scores_other = other_latents[:, feats].sum(axis=1)
        if other_class is not None:
            scores_other_by_class[str(other_class)] = scores_other
    out = class_vs_neutral_metrics(
        scores_pos,
        scores_neu,
        target_class=target_class,
        scores_other=scores_other,
        other_class=other_class,
        scores_other_by_class=scores_other_by_class or None,
        n_bootstrap=n_bootstrap,
        seed=seed,
        youden_threshold=youden_threshold,
        extras={"readout_k": k, "readout_features": feats},
        **dict(frozen or {}),
    )
    return out


def sae_class_vs_neutral_curve(
    class_latents: np.ndarray,
    neutral_latents: np.ndarray,
    feature_indices: Sequence[int],
    *,
    target_class: str,
    ks: Optional[Sequence[int]] = None,
    other_latents: Optional[np.ndarray] = None,
    other_class: Optional[str] = None,
    n_bootstrap: int = 100,
    youden_thresholds: Optional[Mapping[int, float]] = None,
    frozen_by_k: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """k-curve for class-vs-neutral metrics (k = # features of that class).

    ``frozen_by_k``: per-k :func:`frozen_decision_rules` from the validation curve.
    """
    max_k = len(feature_indices)
    if max_k < 1:
        return []
    ks = list(ks or range(1, max_k + 1))
    rows = []
    for k in ks:
        if int(k) < 1 or int(k) > max_k:
            continue
        out = sae_class_vs_neutral_metrics(
            class_latents,
            neutral_latents,
            feature_indices,
            target_class=target_class,
            k=int(k),
            other_latents=other_latents,
            other_class=other_class,
            n_bootstrap=n_bootstrap,
            seed=int(k),
            youden_threshold=(
                None
                if not youden_thresholds
                else youden_thresholds.get(int(k))
            ),
            frozen=(frozen_by_k or {}).get(int(k)),
        )
        if "error" in out:
            rows.append({"k": int(k), "error": out["error"]})
        else:
            rows.append({"k": int(k), **{key: out[key] for key in out if key != "error"}})
    return rows


def sae_per_class_readout_metrics(
    latents: np.ndarray,
    labels: Sequence[str],
    features_by_class: Dict[str, Sequence[int]],
    *,
    k: int = 1,
    class_a: str = "M",
    class_b: str = "F",
    neutral_latents: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fair metrics for per-class SAE readout score = sum(z_a[:k]) − sum(z_b[:k])."""
    try:
        scores = per_class_readout_scores(
            latents, features_by_class, class_a=class_a, class_b=class_b, k=k
        )
    except ValueError as exc:
        return {"error": str(exc)}
    neutral_scores = None
    if neutral_latents is not None and len(neutral_latents):
        neutral_scores = per_class_readout_scores(
            neutral_latents, features_by_class, class_a=class_a, class_b=class_b, k=k
        )
    out = readout_metrics(
        scores,
        labels,
        class_a=class_a,
        class_b=class_b,
        neutral_scores=neutral_scores,
    )
    if "error" not in out:
        out["readout_k"] = int(k)
        out["readout_kind"] = "per_class_diff"
        out["features_by_class"] = {
            class_a: list(features_by_class.get(class_a, []))[: int(k)],
            class_b: list(features_by_class.get(class_b, []))[: int(k)],
        }
    return out


def sae_joint_readout_metrics(
    latents: np.ndarray,
    labels: Sequence[str],
    feature_index: int,
    *,
    class_a: str = "M",
    class_b: str = "F",
    neutral_latents: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fair metrics for a single oriented SAE feature (joint ablation)."""
    scores = joint_readout_scores(
        latents, feature_index, labels, class_a=class_a, class_b=class_b
    )
    neutral_scores = None
    if neutral_latents is not None and len(neutral_latents):
        labels_arr = np.asarray(labels)
        raw = latents[:, int(feature_index)]
        mean_a = float(raw[labels_arr == class_a].mean()) if (labels_arr == class_a).any() else 0.0
        mean_b = float(raw[labels_arr == class_b].mean()) if (labels_arr == class_b).any() else 0.0
        sign = 1.0 if mean_a >= mean_b else -1.0
        neutral_scores = sign * neutral_latents[:, int(feature_index)]
    out = readout_metrics(
        scores,
        labels,
        class_a=class_a,
        class_b=class_b,
        neutral_scores=neutral_scores,
    )
    if "error" not in out:
        out["readout_k"] = 1
        out["readout_kind"] = "joint_single"
        out["readout_features"] = [int(feature_index)]
    return out


def per_class_fair_curve(
    latents: np.ndarray,
    labels: Sequence[str],
    features_by_class: Dict[str, Sequence[int]],
    *,
    class_a: str,
    class_b: str,
    ks: Optional[Sequence[int]] = None,
    neutral_latents: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Fair metrics for k features per class (readout z_a[:k] − z_b[:k])."""
    max_k = min(
        len(features_by_class.get(class_a) or []),
        len(features_by_class.get(class_b) or []),
        20,
    )
    if max_k < 1:
        return []
    ks = list(ks or range(1, max_k + 1))
    rows: List[Dict[str, Any]] = []
    for k in ks:
        k = int(k)
        if k < 1 or k > max_k:
            continue
        out = sae_per_class_readout_metrics(
            latents,
            labels,
            features_by_class,
            k=k,
            class_a=class_a,
            class_b=class_b,
            neutral_latents=neutral_latents,
        )
        if "error" in out:
            rows.append({"k": k, "error": out["error"]})
            continue
        scores = per_class_readout_scores(
            latents, features_by_class, class_a=class_a, class_b=class_b, k=k
        )
        boot = bootstrap_roc_auc(
            scores, labels, class_a=class_a, class_b=class_b, n_bootstrap=n_bootstrap, seed=seed + k
        )
        rows.append(
            {
                "k": k,
                "roc_auc": out.get("roc_auc"),
                "roc_auc_boot_mean": boot.get("roc_auc_boot_mean"),
                "roc_auc_boot_std": boot.get("roc_auc_boot_std"),
                "balanced_accuracy": out.get("balanced_accuracy"),
                "cohens_d": out.get("cohens_d"),
                "specificity": out.get("specificity"),
                "specificity_mid": out.get("specificity_mid"),
                "class_separation": out.get("class_separation"),
                "encoder_correlation": out.get("encoder_correlation"),
                "features_by_class": out.get("features_by_class"),
            }
        )
    return rows


K_SELECTION_BALANCED = "validation_mean_auc_n_auc_o_tie_smaller_k"


def k_selection_score(row: Dict[str, Any]) -> Optional[Tuple[float, int]]:
    """Score for k*: maximize 0.5*(auc_n+auc_o) when both exist; else auc_n; tie → smaller k.

    Returns ``(score, -k)`` for lexicographic max, or None if unscored.
    """
    if "error" in row:
        return None
    k = row.get("k")
    if not isinstance(k, int) or k < 1:
        return None
    auc_n = row.get("roc_auc_neutral", row.get("roc_auc"))
    auc_o = row.get("roc_auc_other", row.get("min_pairwise_auroc"))
    if isinstance(auc_n, (int, float)) and isinstance(auc_o, (int, float)):
        score = 0.5 * (float(auc_n) + float(auc_o))
    elif isinstance(auc_n, (int, float)):
        score = float(auc_n)
    else:
        return None
    return (score, -int(k))


def select_k_from_curve_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    metric: Optional[str] = None,
) -> int:
    """Pick best k from curve rows. Default: balanced auc_n/auc_o + smaller-k tie-break."""
    best_k, best_key = 1, None
    for row in rows:
        if metric:
            if "error" in row:
                continue
            val = row.get(metric)
            k = row.get("k")
            if not isinstance(k, int) or not isinstance(val, (int, float)):
                continue
            key = (float(val), -int(k))
        else:
            key = k_selection_score(row)
            if key is None:
                continue
            k = int(row["k"])
        if best_key is None or key > best_key:
            best_key = key
            best_k = int(k)
    return best_k


def select_per_class_k_on_split(
    latents: np.ndarray,
    labels: Sequence[str],
    features_by_class: Dict[str, Sequence[int]],
    *,
    class_a: str,
    class_b: str,
    ks: Optional[Sequence[int]] = None,
    metric: Optional[str] = None,
) -> int:
    """Pick k* on a split. ``metric=None`` → balanced auc_n/auc_o + smaller-k tie-break."""
    curve = per_class_fair_curve(
        latents,
        labels,
        features_by_class,
        class_a=class_a,
        class_b=class_b,
        ks=ks,
        neutral_latents=None,
        n_bootstrap=0,
    )
    return select_k_from_curve_rows(curve, metric=metric)


def bootstrap_roc_auc(
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    class_a: str = "M",
    class_b: str = "F",
    n_bootstrap: int = 100,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Bootstrap mean/std of ROC-AUC over labeled examples."""
    scores_arr, labels_arr = _binary_mask_and_scores(
        scores, labels, class_a=class_a, class_b=class_b
    )
    point = _roc_auc(scores_arr, labels_arr, class_a=class_a)
    if point is None or n_bootstrap <= 0 or scores_arr.size < 4:
        return {"roc_auc": point, "roc_auc_boot_mean": point, "roc_auc_boot_std": None}
    rng = np.random.default_rng(seed)
    n = scores_arr.size
    vals = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        auc = _roc_auc(scores_arr[idx], labels_arr[idx], class_a=class_a)
        if auc is not None:
            vals.append(auc)
    if not vals:
        return {"roc_auc": point, "roc_auc_boot_mean": point, "roc_auc_boot_std": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "roc_auc": point,
        "roc_auc_boot_mean": float(arr.mean()),
        "roc_auc_boot_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def sparse_fair_curve(
    latents: np.ndarray,
    labels: Sequence[str],
    feature_indices: Sequence[int],
    *,
    class_a: str,
    class_b: str,
    ks: Optional[Sequence[int]] = None,
    neutral_latents: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Fair readout metrics for k=1..K sparse SAE sums, with optional AUC bootstrap std."""
    if not len(feature_indices):
        return []
    max_k = len(feature_indices)
    ks = list(ks or range(1, max_k + 1))
    rows: List[Dict[str, Any]] = []
    for k in ks:
        k = int(k)
        if k < 1 or k > len(feature_indices):
            continue
        out = sae_sparse_readout_metrics(
            latents,
            labels,
            feature_indices,
            k=k,
            class_a=class_a,
            class_b=class_b,
            neutral_latents=neutral_latents,
        )
        if "error" in out:
            rows.append({"k": k, "error": out["error"]})
            continue
        feats = list(feature_indices[:k])
        scores = latents[:, feats].sum(axis=1)
        boot = bootstrap_roc_auc(
            scores,
            labels,
            class_a=class_a,
            class_b=class_b,
            n_bootstrap=n_bootstrap,
            seed=seed + k,
        )
        rows.append(
            {
                "k": k,
                "roc_auc": out.get("roc_auc"),
                "roc_auc_boot_mean": boot.get("roc_auc_boot_mean"),
                "roc_auc_boot_std": boot.get("roc_auc_boot_std"),
                "balanced_accuracy": out.get("balanced_accuracy"),
                "cohens_d": out.get("cohens_d"),
                "specificity": out.get("specificity"),
                "specificity_mid": out.get("specificity_mid"),
                "class_separation": out.get("class_separation"),
                "encoder_correlation": out.get("encoder_correlation"),
                "features": feats,
            }
        )
    return rows


def select_readout_k_on_split(
    latents: np.ndarray,
    labels: Sequence[str],
    feature_indices: Sequence[int],
    *,
    class_a: str,
    class_b: str,
    ks: Optional[Sequence[int]] = None,
    metric: Optional[str] = None,
    neutral_latents: Optional[np.ndarray] = None,
) -> int:
    """Pick k* on a held-out split. Default: balanced auc_n/auc_o + smaller-k tie-break."""
    curve = sparse_fair_curve(
        latents,
        labels,
        feature_indices,
        class_a=class_a,
        class_b=class_b,
        ks=ks,
        neutral_latents=neutral_latents,
        n_bootstrap=0,
    )
    return select_k_from_curve_rows(curve, metric=metric)


def sae_specificity(
    latents: np.ndarray,
    labels: Sequence[str],
    feature_indices: Sequence[int],
    *,
    class_a: Optional[str] = None,
    class_b: Optional[str] = None,
    neutral_latents: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Per-feature SAE specificity rows (selection diagnostics; not the scalar readout)."""
    labels_arr = np.asarray(labels)
    classes = list(pd.Series(labels_arr).value_counts().index)
    if class_a is None or class_b is None:
        class_a, class_b = str(classes[0]), str(classes[1])
    mask_a, mask_b = labels_arr == class_a, labels_arr == class_b
    neutral_mean = (
        neutral_latents.mean(axis=0) if neutral_latents is not None and len(neutral_latents) else None
    )
    rows = []
    for idx in feature_indices:
        mean_a = float(latents[mask_a, idx].mean()) if mask_a.any() else 0.0
        mean_b = float(latents[mask_b, idx].mean()) if mask_b.any() else 0.0
        row: Dict[str, Any] = {
            "feature_index": int(idx),
            f"mean_{class_a}": mean_a,
            f"mean_{class_b}": mean_b,
            "separation": abs(mean_a - mean_b),
        }
        if neutral_mean is not None:
            mid = 0.5 * (mean_a + mean_b)
            half_sep = 0.5 * abs(mean_a - mean_b)
            n_mean = float(neutral_mean[idx])
            row["neutral_mean"] = n_mean
            row["neutral_gap"] = abs(mid - n_mean)
            row["specificity"] = float(half_sep / (half_sep + abs(n_mean - mid) + 1e-8))
        rows.append(row)
    return rows


def selection_to_dict(selection: SAEFeatureSelection) -> Dict[str, Any]:
    return asdict(selection)
