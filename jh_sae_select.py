"""
Jørgensen & Hansen (2026) supervised SAE feature labelling (calibrated F1).

Adapted from https://github.com/MikkelGodsk/SAE-labelling
Paper: "Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines"
  (arXiv:2605.31183) — AxBench re-evaluation: SAEs become competitive under
  *supervised* feature–label matching (calibrated F1), optionally filtered by
  Arad et al. output score.

Study ablation id: ``sae:{cls}:sel_jh_f1``

Adaptation notes
----------------
Their pipeline scores fire-*frequency* over tokens vs multi-label StackExchange
tags, then denoises (support / Arad output-score / top-K). Our gender study uses
prediction-position latents and class ids (M/F/…): we score each feature by
max calibrated F1 of ``1[z > τ]`` vs class membership over a τ grid (quantiles
+ fire threshold), then exclusive top-k per class — same spirit as their
supervised probe match, without StackExchange / AxBench LLM-judge eval.

Arad output-score filtering is a separate study ablation (``sel_arad_out``);
their paper stacks both. Set ``apply_arad_mask`` if precomputed Arad scores are
passed in.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from sae_eval import SAEFeatureSelection

# Paper defaults (π0 tuned on hold-out; support threshold 50 for multi-label fora).
JH_PI0 = 1e-3
JH_SUPPORT_MIN = 50
JH_QUANTILE_GRID = tuple(np.linspace(0.05, 0.95, 19).tolist())


def calibrated_f1_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    *,
    pi: float,
    pi0: float = JH_PI0,
) -> np.ndarray:
    """
    Calibrated F1 (Siblini et al.; Jørgensen & Hansen eq. 11).

    ``F1^c = 2 TP / (2 TP + c·FP + FN)`` with
    ``c = π(1-π0) / (π0(1-π))``.
    """
    pi = float(pi)
    pi0 = float(pi0)
    if not (0.0 < pi < 1.0) or not (0.0 < pi0 < 1.0):
        return np.full_like(tp, np.nan, dtype=np.float64)
    c = (pi * (1.0 - pi0)) / (pi0 * (1.0 - pi))
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)
    denom = 2.0 * tp + c * fp + fn
    out = np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)
    out[denom <= 0] = np.nan
    return out


def max_calibrated_f1_per_feature(
    latents: np.ndarray,
    y_positive: np.ndarray,
    *,
    pi0: float = JH_PI0,
    quantile_grid: Sequence[float] = JH_QUANTILE_GRID,
) -> np.ndarray:
    """
    For each feature, max calibrated F1 over activation thresholds τ.

    ``ŷ = 1[z > τ]``; τ ∈ quantile grid of that feature's activations, plus 0
    (fire). Returns shape ``[n_features]`` (NaN → -inf for ranking).
    """
    z = np.asarray(latents, dtype=np.float64)
    y = np.asarray(y_positive).astype(bool)
    n, n_feat = z.shape
    if y.shape[0] != n:
        raise ValueError("y_positive length must match latents rows")
    pi = float(y.mean()) if n else 0.0
    best = np.full(n_feat, -np.inf, dtype=np.float64)
    if n == 0 or not (0.0 < pi < 1.0):
        return best

    thr_list: List[np.ndarray] = [np.zeros(n_feat, dtype=np.float64)]
    qs = [float(q) for q in quantile_grid]
    if qs:
        thr_list.append(np.quantile(z, qs, axis=0).astype(np.float64))  # [Q, F]

    y_col = y[:, None]
    for thr in thr_list:
        if thr.ndim == 1:
            thresholds = thr[None, :]
        else:
            thresholds = thr
        for t_row in thresholds:
            pred = z > t_row[None, :]
            tp = (pred & y_col).sum(axis=0)
            fp = (pred & ~y_col).sum(axis=0)
            fn = (~pred & y_col).sum(axis=0)
            f1c = calibrated_f1_from_counts(tp, fp, fn, pi=pi, pi0=pi0)
            f1c = np.where(np.isfinite(f1c), f1c, -np.inf)
            best = np.maximum(best, f1c)
    return best


def select_per_class_jh_f1(
    latents: np.ndarray,
    labels: Sequence[str],
    *,
    target_classes: Optional[Sequence[str]] = None,
    top_k_per_class: int = 1,
    min_firing_rate: float = 0.01,
    pi0: float = JH_PI0,
    support_min: int = JH_SUPPORT_MIN,
    arad_scores_by_feature: Optional[Dict[int, float]] = None,
    arad_score_min: float = 1e-3,
) -> SAEFeatureSelection:
    """
    Supervised per-class SAE selection via calibrated F1 (JH labelling core).

    Optional ``arad_scores_by_feature`` applies their output-score denoise
    (threshold ``arad_score_min``, paper default 1e-3).
    """
    labels_arr = np.asarray(labels).astype(str)
    if target_classes is not None and len(target_classes) >= 2:
        classes = [str(c) for c in target_classes]
    else:
        classes = sorted(set(labels_arr.tolist()))
    firing = (latents > 0).mean(axis=0)
    scores_by_class: Dict[str, np.ndarray] = {}
    for c in classes:
        y = labels_arr == c
        n_pos = int(y.sum())
        sc = max_calibrated_f1_per_feature(latents, y, pi0=pi0)
        sc[firing < float(min_firing_rate)] = -np.inf
        if n_pos < int(support_min):
            # Paper support denoise: too few positives → unreliable F1^c.
            sc[:] = -np.inf
        if arad_scores_by_feature:
            for i in range(sc.shape[0]):
                a = arad_scores_by_feature.get(int(i))
                if a is None or float(a) < float(arad_score_min):
                    sc[i] = -np.inf
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
                and scores_by_class[c][i] > float("-inf")
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
                    if i not in taken
                    and i not in claimed
                    and np.isfinite(scores_by_class[c][i])
                    and scores_by_class[c][i] > float("-inf")
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
    scores: Dict[int, float] = {}
    for c in classes:
        for i in features_by_class[c]:
            selected.append(i)
            scores[i] = float(scores_by_class[c][i])

    contrast = (classes[0], classes[1]) if len(classes) >= 2 else (classes[0], classes[0])
    means = {
        c: latents[labels_arr == c].mean(axis=0) if (labels_arr == c).any() else np.zeros(latents.shape[1])
        for c in classes
    }
    return SAEFeatureSelection(
        feature_indices=selected,
        scores=scores,
        mean_by_class={
            c: {i: float(means[c][i]) for i in selected} for c in classes
        },
        contrast_pair=contrast,
        mode="per_class_jh_f1",
        layer=-1,
        sae_id="",
        used_neutral_specificity=False,
        features_by_class=features_by_class,
        scores_by_class={
            c: {i: float(scores_by_class[c][i]) for i in feats}
            for c, feats in features_by_class.items()
        },
    )
