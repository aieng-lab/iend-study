#!/usr/bin/env python
"""Main-study vs appendix ablation retention.

This is **not** a winner picker. Headline settings are fixed a priori
(``configs/suites/core.yaml`` + fair encode↔causal pairing). Full-suite
results on small models are used only to test whether any extra setting
*falsifies* that those headlines are sufficient.

Decision tiers
--------------
``main``
    Run on every model (core suite). Protocol default, or a distinct
    scientific family (e.g. ``sae_pre`` vs filled ``sae``).
``appendix_small``
    Keep on gpt2 / pythia-70m full suite; do not promote to Llama/Qwen.
``review_promote``
    Empirical test fired (ranking flip or large paired gain). Inspect
    before changing ``core.yaml`` — do not auto-promote.
``oracle``
    Per-layer / extra-k shopping. Appendix diagnostic only; promoting
    would be test-set selection.

Paired tests (pre-specified, not tuned on this dump)
----------------------------------------------------
* ΔE τ = 0.02 on encoding bottleneck E.
* Δcausal τ = 0.02 on LMS-gated signed ΔP.
* Ranking flip: replacing the family default in the locked 4-way
  (GRADIEND, ACTIEND, SAE k=1, CAA act_prediction) changes the task winner.
  This test is scored on ``encoding_E`` only, not causal ΔP. Reason:
  ``encoding_E`` is computed for every recipe/task cell already gathered
  (no LMS-gated intervention rollout needed), so it has full coverage —
  ``n_paired_causal`` is frequently smaller than ``n_paired_E`` in the
  output because causal wasn't run for every cell. Causal ΔP is still
  checked as its own paired-delta gate (see τ above); it just isn't the
  ranking-flip metric.
* Promote-review if ranking flips on ≥3 tasks, or mean |ΔE|≥0.05, or
  mean |Δcausal|≥0.05 — **and** the setting is not an oracle.
* Bootstrap 95% CIs and a Wilcoxon signed-rank p-value are computed on
  every paired-delta sample and reported alongside (``*_ci_lo``/``*_ci_hi``/
  ``*_wilcoxon_p`` in the CSV, ``pE`` in the text table). These are
  **diagnostic only** — they do not gate any tier decision. Folding them
  into ``decide_retention`` now, after already having looked at this
  dump's promotion outcomes (e.g. CAA ``all_act_prediction`` firing the
  flip test), would itself be post-hoc tuning of the frozen rule. They
  exist so a human reviewing a ``review_promote`` row can see whether the
  mean delta is actually distinguishable from paired noise before editing
  ``core.yaml`` — n as low as ~10-15 tasks is not enough to treat a mean
  crossing τ as automatically real. `p` is ``None`` below n=8 (normal
  approximation unreliable) or n=0 paired.

Usage::

    python analysis/ablation_retention.py
    python analysis/ablation_retention.py --cells analysis/tables/recipe/recipe_cells_long.csv
    python analysis/ablation_retention.py --study-tasks --cells <canonical recipe cells>
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DELTA_E_TAU = 0.02
DELTA_CAUSAL_TAU = 0.02
LARGE_DELTA = 0.05
MIN_FLIP_TASKS = 3
MIN_PAIRED = 3

HEADLINE_METRICS = ("encoding_E", "causal_signed_effect", "suitability")

# Locked 4-way comparison used for ranking-flip tests (core protocol).
HEADLINE_FAMILY_RECIPES: Dict[str, Tuple[str, ...]] = {
    "gradiend": ("gradiend:two_pole", "gradiend:one_pole"),
    "actiend": ("actiend:two_pole", "actiend:one_pole"),
    "sae": ("sae:k1",),
    "caa": ("caa:act_prediction",),
}

_LAYER = re.compile(r"^L\d+")
_K_FIXED = re.compile(r"^k(\d+)$")
_ALL_K = re.compile(r"^all_k(\d+)$")


@dataclass(frozen=True)
class AxisSpec:
    """One scientific question; default is the core/main setting."""

    axis: str
    default: str
    hypothesis: str
    family: str  # which headline family this variant substitutes


# Pre-registered questions. Defaults match core.yaml / protocol, not dump winners.
AXES: Tuple[AxisSpec, ...] = (
    AxisSpec(
        axis="sae_bag",
        default="sae:k1",
        hypothesis=(
            "k=1 is the fair 1-feature analogue of a 1-D IEND latent. kstar picks "
            "k by max(0.5*(auc_n+auc_o)) on the VALIDATION split only, ties toward "
            "smaller k (sae_eval.py:3350-3371; k is fixed before scoring the TEST "
            "split -- no test-set leakage). all_k1 sums every layer's top-1 feature. "
            "Both test extra SAE capacity (bag size / every layer), not a different "
            "selection rule -- see sae_select for that axis."
        ),
        family="sae",
    ),
    AxisSpec(
        axis="sae_select",
        default="sae:k1",
        hypothesis=(
            "Four DIFFERENT feature-ranking formulas, not variants of one idea -- "
            "none is redundant with encoding_E (E is a downstream eval metric "
            "computed on whatever features a rule picked, not a selection rule "
            "itself). Default (k1/kstar): (mean_c - mean_rest) - lambda_neu*|mean_neu| "
            "(sae_eval.py:684-693). opp_fire: same, minus an extra penalty on the "
            "RIVAL class's binary firing rate P(activation>0) -- catches features "
            "that fire rarely-but-nonzero on the rival, which a continuous mean-diff "
            "barely penalizes (sae_eval.py:694-702). arad_out: re-ranks top mean-diff "
            "candidates by an output-side causal probe -- amplify the feature on a "
            "fixed neutral prompt and score by how far it pushes the model's own "
            "logit-lens top tokens toward the class (arad_sae_select.py:171-213). "
            "jh_f1: max calibrated-F1 of 1[activation>tau] vs class label, swept over "
            "tau, prevalence-corrected (Jorgensen & Hansen eq. 11; jh_sae_select.py:"
            "40-110). joint: single feature maximizing |mean_a - mean_b| between the "
            "two poles directly, no neutral/exclusivity term (sae_eval.py:398-407)."
        ),
        family="sae",
    ),
    AxisSpec(
        axis="sae_pre_bag",
        default="sae_pre:k1",
        hypothesis=(
            "Classical last-context SAE (sae_pre, activation_site=pre_prediction -- "
            "one token before the filled span) is a separate family, not an "
            "ablation of filled-span sae. Bag-size question mirrors sae_bag; kstar "
            "here goes through the identical validation-selection code path as "
            "sae:kstar (study/sae_engine.py loops the same "
            "run_sae_layer per site)."
        ),
        family="sae",
    ),
    AxisSpec(
        axis="caa_pool",
        default="caa:act_prediction",
        hypothesis=(
            "Token-position pooling within one fixed layer/site (caa_eval.py:64, "
            "123-137): act_prediction = mean over the filled prediction span "
            "(matches ACTIEND/SAE's encode site); act_mean = attention-weighted "
            "mean over EVERY non-pad token ('mean' and 'all' are literal synonyms "
            "here -- there is no missing 4th 'pool over everything' option, "
            "act_mean already is that); act_last = last non-pad token. A fourth "
            "site, act_pre_prediction (one token before the span), exists in code "
            "but isn't requested by any suite here -- it's the same site "
            "sae_pre/actiend_pre are built on, tracked as a separate family "
            "instead of a caa_pool variant."
        ),
        family="caa",
    ),
    AxisSpec(
        axis="caa_scope",
        default="caa:act_prediction",
        hypothesis=(
            "Layer aggregation, orthogonal to caa_pool's token-position choice. "
            "Default (no 'all_' prefix) concatenates each layer's mean-diff vector "
            "into one long vector and scores COSINE against the concatenated "
            "activation (caa_eval.py:676-684). all_act_* scores a cosine AT EACH "
            "LAYER separately, then averages those cosines (caa_eval.py:685-698) -- "
            "a genuinely different encoding_E-style number. IMPORTANT: the CAUSAL "
            "steering intervention is identical for both -- causal_study.py:2137 "
            "explicitly reuses the :all specs (\"multi-layer steer (=:all; concat "
            "encode analogue)\") for the default id too, since a concatenated "
            "vector spans multiple layers' dims and can't be added to one residual "
            "stream. So caa:act_prediction and caa:all_act_prediction's "
            "causal_signed_effect numbers are NOT independent measurements -- only "
            "their encoding_E differs."
        ),
        family="caa",
    ),
    AxisSpec(
        axis="actiend_tok",
        default="actiend:tok_all_gate_encoder_direction",
        hypothesis=(
            "Package default is all-tokens + encoder-direction gate. The 'gate' is "
            "NOT a data-fit hyperparameter (no validation/test selection involved): "
            "a fixed per-token rule applied at steering time, projecting each "
            "token's live encoder output onto the target direction and thresholding "
            "(gradiend/model/modified.py:240-243). tok_all / tok_prediction are "
            "ungated scope ablations -- same encoder, gate removed."
        ),
        family="actiend",
    ),
    AxisSpec(
        axis="iend_split",
        default="gradiend:two_pole",
        hypothesis="None-split vs :tensors aggregate (matched causal when run).",
        family="gradiend",
    ),
)

# Pole axes (one-pole vs two-pole) are kept OUT of `AXES`/`decide_retention` on
# purpose: both `two_pole` and `one_pole` are individually in `_CORE_HEADLINES`
# (one_pole is this paper's own contribution, not an ablation of two_pole -- see
# CLAUDE.md's "Method-id naming" section), so the promotion-only pipeline
# above always scores either one as tier "main" regardless of outcome, by
# design. `suggest_core_settings` below asks the narrower, genuinely open
# question instead: for tasks with >2 classes, does training one one-pole
# encoder per class actually cover what the bipolar pairwise contrast gives
# you, or is the pairwise contrast still buying something one-pole training
# alone doesn't? For binary tasks the two are structurally close to the same
# computation, so this comparison is restricted to n_classes > 2 tasks only.
POLE_AXES: Tuple[AxisSpec, ...] = (
    AxisSpec(
        axis="gradiend_pole",
        default="gradiend:two_pole",
        hypothesis=(
            "For >2-class tasks: does one one-pole GRADIEND encoder per class "
            "(this paper's contribution) match/beat the bipolar M-F:M-style "
            "pairwise contrast (prior-work behavior), or do you still need both? "
            "Restricted to n_classes > 2 -- binary tasks make the two nearly "
            "the same computation."
        ),
        family="gradiend",
    ),
    AxisSpec(
        axis="actiend_pole",
        default="actiend:two_pole",
        hypothesis=(
            "Same question as gradiend_pole, for ACTIEND: one-pole vs bipolar "
            "pairwise, restricted to n_classes > 2 tasks."
        ),
        family="actiend",
    ),
)
_POLE_AXIS_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "gradiend_pole": ("gradiend:one_pole",),
    "actiend_pole": ("actiend:one_pole",),
}


def _finite(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


# Diagnostic-only paired stats (do not gate tiers -- see module docstring).
_BOOTSTRAP_N = 2000
_BOOTSTRAP_SEED = 12345
_MIN_N_WILCOXON = 8


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _rank_abs(diffs: Sequence[float]) -> List[float]:
    """Average ranks over |diffs|, ties get the mean of their rank span."""
    order = sorted(range(len(diffs)), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * len(diffs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _tie_correction(diffs: Sequence[float]) -> float:
    counts = Counter(abs(d) for d in diffs)
    return sum((t**3 - t) for t in counts.values()) / 48.0


def wilcoxon_signed_rank_p(deltas: Sequence[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value, normal approximation.

    Pure-stdlib (no scipy) -- sandbox here has no network access to
    install it. Returns None below ``_MIN_N_WILCOXON`` non-zero pairs,
    where the normal approximation is unreliable.
    """
    diffs = [float(d) for d in deltas if d != 0]
    n = len(diffs)
    if n < _MIN_N_WILCOXON:
        return None
    ranks = _rank_abs(diffs)
    w_pos = sum(r for d, r in zip(diffs, ranks) if d > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - _tie_correction(diffs)
    if var_w <= 0:
        return None
    cc = 0.5 if w_pos >= mean_w else -0.5
    z = (w_pos - mean_w - cc) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return max(0.0, min(1.0, p))


def bootstrap_ci(
    deltas: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = _BOOTSTRAP_N,
    seed: int = _BOOTSTRAP_SEED,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap 95% CI on the paired-delta mean. Deterministic seed."""
    vals = [float(d) for d in deltas]
    n = len(vals)
    if n < 3:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = max(0, min(int((alpha / 2) * n_boot), n_boot - 1))
    hi_idx = max(0, min(int((1 - alpha / 2) * n_boot) - 1, n_boot - 1))
    return (means[lo_idx], means[hi_idx])


_CORE_HEADLINES = frozenset(
    {
        "gradiend:two_pole",
        "gradiend:one_pole",
        "actiend:two_pole",
        "actiend:one_pole",
        "actiend:tok_all_gate_encoder_direction",
        "sae:k1",
        "sae:kstar",
        "sae:all_k1",
        "sae_pre:k1",
        "sae_pre:kstar",
        "sae_pre:all_k1",
        "caa:act_prediction",
    }
)


def classify_recipe(name: str) -> Dict[str, Any]:
    """Map a locked recipe id to axis / oracle / core-default flags."""
    rec = str(name or "")
    fam, _, rest = rec.partition(":")
    oracle = False
    axis = "other"
    core = rec in _CORE_HEADLINES
    role = "main_default" if core else "variant"

    if rec in {"gradiend:two_pole", "gradiend:one_pole", "actiend:two_pole", "actiend:one_pole"}:
        axis = "pole"
    elif rec in {"actiend_pre:two_pole", "actiend_pre:one_pole"}:
        axis = "actiend_pre"
        role = "opt_in"
    elif rest.startswith("tok_"):
        axis = "actiend_tok"
    elif fam in {"sae", "sae_pre"} and rest in {"k1", "kstar", "all_k1"}:
        axis = "sae_bag" if fam == "sae" else "sae_pre_bag"
    elif rest in {"sel_opp_fire", "sel_arad_out", "sel_jh_f1", "joint"}:
        axis = "sae_select"
    elif rest in {"all_act_prediction", "all_act_mean", "all_act_last"}:
        axis = "caa_scope"
    elif rest in {"act_prediction", "act_mean", "act_last"}:
        axis = "caa_pool"
    elif rest in {"tensors"} or rec.endswith(":tensors"):
        axis = "iend_split"
    elif _LAYER.match(rest):
        axis = f"{fam}_layer"
        oracle = True
    elif _K_FIXED.match(rest) or _ALL_K.match(rest):
        axis = "sae_bag" if fam == "sae" else "sae_pre_bag"
        m = _K_FIXED.match(rest) or _ALL_K.match(rest)
        k = int(m.group(1)) if m else 0
        if rest not in {"k1", "all_k1"} and k != 1:
            oracle = True

    if oracle:
        role = "oracle"
        core = False

    return {
        "recipe": rec,
        "family": fam,
        "axis": axis,
        "core": core,
        "oracle": oracle,
        "role": role,
    }


def load_recipe_cells(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            mean = _finite(raw.get("mean"))
            if mean is None:
                continue
            n_classes = _finite(raw.get("n_classes"))
            rows.append(
                {
                    "model": str(raw.get("model") or ""),
                    "task": str(raw.get("task") or ""),
                    "recipe": str(raw.get("family") or raw.get("recipe") or ""),
                    "metric": str(raw.get("metric") or ""),
                    "mean": mean,
                    "n_classes": int(n_classes) if n_classes is not None else None,
                }
            )
    return rows


def restrict_to_study_tasks(cells: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Drop hidden recipe artifacts, retaining only visible ``TASKS=all`` tasks."""
    # Imported lazily because this file is also used as a standalone analysis
    # script, whose ``sys.path`` normally begins at ``analysis/``.
    from study.config import list_tasks

    visible_tasks = set(list_tasks())
    return [dict(cell) for cell in cells if str(cell.get("task") or "") in visible_tasks]


def multi_class_tasks(
    cells: Sequence[Mapping[str, Any]],
    *,
    min_classes: int = 3,
    model: Optional[str] = None,
) -> set:
    """(model, task) pairs with strictly more than a binary class set.

    Used to scope the one-pole-vs-two-pole comparison to tasks where the two
    training modes actually pose a different question -- for a 2-class task
    the bipolar pairwise contrast and one encoder-per-class are structurally
    close to the same computation.
    """
    out = set()
    for row in cells:
        n = row.get("n_classes")
        if n is None:
            continue
        if model is not None and str(row.get("model") or "") != model:
            continue
        if int(n) >= min_classes:
            out.add((str(row.get("model") or ""), str(row.get("task") or "")))
    return out


def _index_cells(
    cells: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str], float]:
    out: Dict[Tuple[str, str, str, str], float] = {}
    for row in cells:
        rec = str(row.get("recipe") or row.get("family") or "")
        key = (
            str(row.get("model") or ""),
            str(row.get("task") or ""),
            rec,
            str(row.get("metric") or ""),
        )
        val = _finite(row.get("mean", row.get("value")))
        if rec and val is not None:
            out[key] = val
    return out


def _headline_score(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    model: str,
    task: str,
    family: str,
    metric: str,
    override: Optional[str] = None,
) -> Optional[float]:
    if override:
        return idx.get((model, task, override, metric))
    for rec in HEADLINE_FAMILY_RECIPES[family]:
        v = idx.get((model, task, rec, metric))
        if v is not None:
            return v
    return None


def _winner(
    scores: Mapping[str, Optional[float]],
) -> Optional[str]:
    best_name: Optional[str] = None
    best_val = float("-inf")
    for name, val in scores.items():
        if val is None:
            continue
        if val > best_val + 1e-12:
            best_val = val
            best_name = name
    return best_name


def paired_deltas(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    variant: str,
    default: str,
    metric: str,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tasks = sorted(
        {
            (m, t)
            for (m, t, rec, met) in idx
            if rec == variant and met == metric and (model is None or m == model)
        }
    )
    out: List[Dict[str, Any]] = []
    for m, t in tasks:
        v = idx.get((m, t, variant, metric))
        d = idx.get((m, t, default, metric))
        if v is None or d is None:
            continue
        out.append(
            {
                "model": m,
                "task": t,
                "variant": variant,
                "default": default,
                "metric": metric,
                "variant_mean": v,
                "default_mean": d,
                "delta": v - d,
            }
        )
    return out


def ranking_flips(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    variant: str,
    family: str,
    metric: str = "encoding_E",
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tasks where swapping ``family``'s headline recipe for ``variant`` changes the 4-way winner."""
    tasks = sorted(
        {
            (m, t)
            for (m, t, rec, met) in idx
            if rec == variant and met == metric and (model is None or m == model)
        }
    )
    flips: List[Dict[str, Any]] = []
    for m, t in tasks:
        base = {
            fam: _headline_score(idx, model=m, task=t, family=fam, metric=metric)
            for fam in HEADLINE_FAMILY_RECIPES
        }
        alt = dict(base)
        alt[family] = idx.get((m, t, variant, metric))
        w0, w1 = _winner(base), _winner(alt)
        if w0 is None or w1 is None:
            continue
        if w0 != w1:
            flips.append(
                {
                    "model": m,
                    "task": t,
                    "metric": metric,
                    "base_winner": w0,
                    "alt_winner": w1,
                    "variant": variant,
                    "family": family,
                }
            )
    return flips


def _summarize_deltas(rows: Sequence[Mapping[str, Any]], tau: float) -> Dict[str, Any]:
    deltas = [float(r["delta"]) for r in rows]
    if not deltas:
        return {
            "n_paired": 0,
            "mean_delta": None,
            "median_delta": None,
            "frac_beat_tau": None,
            "frac_lose_tau": None,
            "ci_lo": None,
            "ci_hi": None,
            "wilcoxon_p": None,
        }
    ordered = sorted(deltas)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    ci_lo, ci_hi = bootstrap_ci(deltas)
    return {
        "n_paired": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "median_delta": median,
        "frac_beat_tau": sum(1 for d in deltas if d >= tau) / len(deltas),
        "frac_lose_tau": sum(1 for d in deltas if d <= -tau) / len(deltas),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": wilcoxon_signed_rank_p(deltas),
    }


# caa_scope's default (concat) and all_act_* ids report DIFFERENT encoding_E
# numbers but share the IDENTICAL causal-steering mechanism (a concatenated
# vector spans multiple layers and can't be added to one residual stream, so
# the default id's causal sweep reuses the :all specs -- causal_study.py:2137,
# `"multi-layer steer (=:all; concat encode analogue)"`). Appended to every
# caa_scope reason so this shows up in the report itself, not just the axis
# hypothesis text, since it changes how a causal_signed_effect delta on this
# axis should be read.
_CAA_SCOPE_CAVEAT = (
    " NOTE: causal_signed_effect is NOT independently measured here -- "
    "caa_scope's default and all_act_* ids share the same multi-layer "
    "steering intervention; only encoding_E differs between them."
)


def decide_retention(
    *,
    recipe: str,
    axis: str,
    core: bool,
    oracle: bool,
    n_paired_e: int,
    n_paired_c: int,
    mean_dE: Optional[float],
    mean_dC: Optional[float],
    n_flips: int,
    n_flip_tasks_scored: int,
) -> Tuple[str, str]:
    """Return ``(tier, reason)``. Never auto-promotes into core."""
    tier, reason = _decide_retention_core(
        recipe=recipe,
        axis=axis,
        core=core,
        oracle=oracle,
        n_paired_e=n_paired_e,
        n_paired_c=n_paired_c,
        mean_dE=mean_dE,
        mean_dC=mean_dC,
        n_flips=n_flips,
        n_flip_tasks_scored=n_flip_tasks_scored,
    )
    if axis == "caa_scope":
        reason = reason + _CAA_SCOPE_CAVEAT
    return tier, reason


def _decide_retention_core(
    *,
    recipe: str,
    axis: str,
    core: bool,
    oracle: bool,
    n_paired_e: int,
    n_paired_c: int,
    mean_dE: Optional[float],
    mean_dC: Optional[float],
    n_flips: int,
    n_flip_tasks_scored: int,
) -> Tuple[str, str]:
    meta = classify_recipe(recipe)
    n_paired = max(int(n_paired_e), int(n_paired_c))
    if meta["core"] and meta["role"] == "main_default":
        return (
            "main",
            "Protocol default (core.yaml / fair encode-causal pairing). Not chosen by outcome.",
        )
    if recipe in {"sae_pre:k1", "sae_pre:kstar", "sae_pre:all_k1"}:
        return (
            "main",
            "Separate SAE family (last-context vs filled span), not a ranking ablation.",
        )
    if oracle:
        return (
            "oracle",
            "Per-layer or extra-k setting. Appendix diagnostic; promoting is test-set selection.",
        )
    if n_paired < MIN_PAIRED:
        return (
            "appendix_small",
            f"Too few paired tasks (n={n_paired}) to justify a core change.",
        )
    large_e = (
        mean_dE is not None
        and abs(mean_dE) >= LARGE_DELTA
        and int(n_paired_e) >= MIN_PAIRED
    )
    large_c = (
        mean_dC is not None
        and abs(mean_dC) >= LARGE_DELTA
        and int(n_paired_c) >= MIN_PAIRED
    )
    if n_flips >= MIN_FLIP_TASKS and n_flip_tasks_scored >= MIN_PAIRED:
        return (
            "review_promote",
            f"Changes the locked 4-family winner on {n_flips}/{n_flip_tasks_scored} tasks. Inspect before editing core.yaml.",
        )
    if large_e or large_c:
        return (
            "review_promote",
            "Large mean paired gain vs default. Confirm it is not extra SAE/CAA capacity before promoting.",
        )
    if (
        (mean_dE is None or abs(mean_dE) < DELTA_E_TAU)
        and (mean_dC is None or abs(mean_dC) < DELTA_CAUSAL_TAU)
    ):
        return (
            "appendix_small",
            "Redundant with the default on E and causal dP (below pre-specified tau). Keep on small models only.",
        )
    return (
        "appendix_small",
        "Distinct hypothesis, but does not change headline family ranking. Full suite / small models.",
    )


def evaluate_variant(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    variant: str,
    spec: AxisSpec,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    meta = classify_recipe(variant)
    default = spec.default
    if spec.axis == "iend_split":
        default = f"{variant.split(':', 1)[0]}:two_pole"
    e_rows = paired_deltas(idx, variant=variant, default=default, metric="encoding_E", model=model)
    c_rows = paired_deltas(
        idx, variant=variant, default=default, metric="causal_signed_effect", model=model
    )
    s_rows = paired_deltas(idx, variant=variant, default=default, metric="suitability", model=model)
    e_sum = _summarize_deltas(e_rows, DELTA_E_TAU)
    c_sum = _summarize_deltas(c_rows, DELTA_CAUSAL_TAU)
    s_sum = _summarize_deltas(s_rows, DELTA_E_TAU)
    flips = ranking_flips(idx, variant=variant, family=spec.family, metric="encoding_E", model=model)
    n_scored = len(
        {
            (m, t)
            for (m, t, rec, met) in idx
            if rec == variant and met == "encoding_E" and (model is None or m == model)
        }
    )
    tier, reason = decide_retention(
        recipe=variant,
        axis=spec.axis,
        core=bool(meta["core"]),
        oracle=bool(meta["oracle"]),
        n_paired_e=int(e_sum["n_paired"]),
        n_paired_c=int(c_sum["n_paired"]),
        mean_dE=e_sum["mean_delta"],
        mean_dC=c_sum["mean_delta"],
        n_flips=len(flips),
        n_flip_tasks_scored=n_scored,
    )
    return {
        "axis": spec.axis,
        "hypothesis": spec.hypothesis,
        "default": default,
        "variant": variant,
        "family": spec.family,
        "tier": tier,
        "reason": reason,
        "oracle": meta["oracle"],
        "core": meta["core"],
        "n_tasks_variant": n_scored,
        "n_paired_E": e_sum["n_paired"],
        "mean_dE": e_sum["mean_delta"],
        "median_dE": e_sum["median_delta"],
        "frac_E_beat_tau": e_sum["frac_beat_tau"],
        "dE_ci_lo": e_sum["ci_lo"],
        "dE_ci_hi": e_sum["ci_hi"],
        "dE_wilcoxon_p": e_sum["wilcoxon_p"],
        "n_paired_causal": c_sum["n_paired"],
        "mean_dCausal": c_sum["mean_delta"],
        "frac_causal_beat_tau": c_sum["frac_beat_tau"],
        "dCausal_ci_lo": c_sum["ci_lo"],
        "dCausal_ci_hi": c_sum["ci_hi"],
        "dCausal_wilcoxon_p": c_sum["wilcoxon_p"],
        "n_paired_S": s_sum["n_paired"],
        "mean_dS": s_sum["mean_delta"],
        "n_ranking_flips": len(flips),
        "flip_tasks": ",".join(f"{r['task']}:{r['base_winner']}->{r['alt_winner']}" for r in flips),
    }


_AXIS_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "sae_bag": ("sae:kstar", "sae:all_k1"),
    "sae_pre_bag": ("sae_pre:kstar", "sae_pre:all_k1"),
    "sae_select": ("sae:sel_opp_fire", "sae:sel_arad_out", "sae:sel_jh_f1", "sae:joint"),
    "caa_pool": ("caa:act_mean", "caa:act_last"),
    "caa_scope": ("caa:all_act_prediction", "caa:all_act_mean", "caa:all_act_last"),
    "actiend_tok": ("actiend:tok_all", "actiend:tok_prediction"),
}


def variants_for_axis(
    recipes: Iterable[str],
    spec: AxisSpec,
) -> List[str]:
    present = set(recipes)
    listed = _AXIS_VARIANTS.get(spec.axis, ())
    out = [r for r in listed if r in present and r != spec.default]
    if spec.axis == "iend_split":
        out.extend(
            r for r in present if r.endswith("tensors") and r != spec.default and r not in out
        )
    return sorted(set(out))


def evaluate_all(
    cells: Sequence[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    idx = _index_cells(cells)
    recipes = sorted({rec for (_m, _t, rec, _met) in idx if model is None or _m == model})
    rows: List[Dict[str, Any]] = []
    for spec in AXES:
        for variant in variants_for_axis(recipes, spec):
            rows.append(evaluate_variant(idx, variant=variant, spec=spec, model=model))
    rows.sort(key=lambda r: (r["axis"], r["variant"]))
    return rows


def oracle_inventory(cells: Sequence[Mapping[str, Any]], *, model: Optional[str] = None) -> Dict[str, int]:
    idx = _index_cells(cells)
    recipes = sorted({rec for (m, _t, rec, _met) in idx if model is None or m == model})
    counts: Dict[str, int] = {}
    for rec in recipes:
        meta = classify_recipe(rec)
        if meta["oracle"]:
            counts[meta["axis"]] = counts.get(meta["axis"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Symmetric core-setting suggestions.
#
# Everything above this point is deliberately *asymmetric*: `core.yaml`'s
# current pick for an axis is frozen as "main" and never compared against
# its own siblings -- only non-core variants get tested, and only for
# possible promotion. That's the right guard against post-hoc test-set
# selection on the *paper's* headline claims, but it also means the module
# cannot answer "which setting should this axis actually use" independent
# of whatever core.yaml happens to say today.
#
# The functions below answer that different, narrower question: for each
# pre-registered axis, rank every candidate setting (core default included)
# against every other candidate on the *same* fully-paired task set, using
# the same tau/flip thresholds already frozen above. This never edits
# `decide_retention`/`evaluate_all`'s promotion-only behavior -- it is a
# separate, explicitly-labeled "suggestion" pass, not a second scoring path
# for the same tiers.
# ---------------------------------------------------------------------------


def axis_candidate_pool(recipes: Iterable[str], spec: AxisSpec) -> List[str]:
    """Every candidate setting for `spec`'s axis that has data, default included."""
    present = set(recipes)
    out = set(variants_for_axis(recipes, spec))
    out.update(r for r in _POLE_AXIS_VARIANTS.get(spec.axis, ()) if r in present)
    if spec.default in present:
        out.add(spec.default)
    return sorted(out)


def symmetric_axis_ranking(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    candidates: Sequence[str],
    metric: str,
    model: Optional[str] = None,
    eligible_tasks: Optional[Iterable[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Rank `candidates` against each other, none privileged as the reference.

    Restricts to the task set where *every* candidate reports `metric` (a
    "full house") so the comparison is apples-to-apples -- a candidate with
    wider task coverage doesn't win just by having more data. Returns the
    per-candidate mean on that shared set, the winner (highest mean), and,
    for every other candidate, the paired per-task delta (winner - other)
    restricted to the same shared set. ``eligible_tasks``, when given,
    further restricts the shared set to those (model, task) pairs (e.g. only
    tasks with >2 classes, for the pole-count question).
    """
    if not candidates:
        return {"n_paired": 0, "tasks": [], "means": {}, "winner": None, "deltas_vs_winner": {}}
    task_sets = [
        {
            (m, t)
            for (m, t, rec, met) in idx
            if rec == c and met == metric and (model is None or m == model)
        }
        for c in candidates
    ]
    tasks = sorted(set.intersection(*task_sets)) if task_sets else []
    if eligible_tasks is not None:
        eligible = set(eligible_tasks)
        tasks = [t for t in tasks if t in eligible]
    per_task: Dict[str, Dict[Tuple[str, str], float]] = {c: {} for c in candidates}
    means: Dict[str, float] = {}
    for c in candidates:
        vals = []
        for key in tasks:
            v = idx.get((key[0], key[1], c, metric))
            if v is not None:
                vals.append(v)
                per_task[c][key] = v
        if vals:
            means[c] = sum(vals) / len(vals)
    winner: Optional[str] = None
    best = float("-inf")
    for c in candidates:
        v = means.get(c)
        if v is not None and v > best + 1e-12:
            best = v
            winner = c
    deltas_vs_winner: Dict[str, List[float]] = {}
    if winner is not None:
        for c in candidates:
            if c == winner:
                continue
            deltas_vs_winner[c] = [
                per_task[winner][key] - per_task[c][key]
                for key in tasks
                if key in per_task[winner] and key in per_task[c]
            ]
    return {
        "n_paired": len(tasks),
        "tasks": tasks,
        "means": means,
        "winner": winner,
        "deltas_vs_winner": deltas_vs_winner,
    }


def suggest_core_setting(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    spec: AxisSpec,
    recipes: Iterable[str],
    model: Optional[str] = None,
    eligible_tasks: Optional[Iterable[Tuple[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Symmetric recommendation for one axis, independent of core.yaml's current pick."""
    candidates = axis_candidate_pool(recipes, spec)
    if len(candidates) < 2:
        return None

    e_rank = symmetric_axis_ranking(
        idx, candidates=candidates, metric="encoding_E", model=model, eligible_tasks=eligible_tasks
    )
    c_rank = symmetric_axis_ranking(
        idx,
        candidates=candidates,
        metric="causal_signed_effect",
        model=model,
        eligible_tasks=eligible_tasks,
    )
    s_rank = symmetric_axis_ranking(
        idx, candidates=candidates, metric="suitability", model=model, eligible_tasks=eligible_tasks
    )

    winner = e_rank["winner"]
    n_paired = e_rank["n_paired"]
    default_present = spec.default in candidates

    flips: List[Dict[str, Any]] = []
    if winner and winner != spec.default and spec.family in HEADLINE_FAMILY_RECIPES:
        flips = ranking_flips(idx, variant=winner, family=spec.family, metric="encoding_E", model=model)
        if eligible_tasks is not None:
            eligible = set(eligible_tasks)
            flips = [f for f in flips if (f["model"], f["task"]) in eligible]

    delta_vs_default: Optional[float] = None
    wilcoxon_p: Optional[float] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    if winner and winner != spec.default:
        d = e_rank["deltas_vs_winner"].get(spec.default)
        if d:
            delta_vs_default = sum(d) / len(d)
            wilcoxon_p = wilcoxon_signed_rank_p(d)
            ci_lo, ci_hi = bootstrap_ci(d)

    if winner is None or n_paired < MIN_PAIRED:
        verdict = "insufficient_data"
        reason = (
            f"Fewer than {MIN_PAIRED} tasks report every candidate in this axis on the "
            "same metric -- too little paired data to recommend a setting."
        )
    elif not default_present:
        verdict = "insufficient_data"
        reason = (
            f"Current core default {spec.default!r} has no data in this dump; cannot "
            "compare it to its siblings."
        )
    elif winner == spec.default:
        verdict = "keep_core"
        reason = "Current core default already wins the symmetric comparison on encoding_E."
    else:
        large = delta_vs_default is not None and abs(delta_vs_default) >= LARGE_DELTA
        flip_enough = len(flips) >= MIN_FLIP_TASKS
        if large or flip_enough:
            verdict = "suggest_change"
            reason = (
                f"{winner} beats current core default {spec.default} by "
                f"{delta_vs_default:+.3f} mean encoding_E over {n_paired} paired tasks"
                + (f", flips the 4-family winner on {len(flips)} tasks" if flips else "")
                + " -- worth a deliberate core.yaml change, not automatic."
            )
        else:
            verdict = "no_clear_winner"
            reason = (
                f"{winner} nominally ranks highest but the margin over {spec.default} "
                f"({delta_vs_default:+.3f} mean encoding_E, n={n_paired}) is below the "
                f"pre-specified thresholds (tau={DELTA_E_TAU}, large={LARGE_DELTA}, "
                f"min_flips={MIN_FLIP_TASKS}) -- not a strong enough signal to change core.yaml."
            )

    if spec.axis == "caa_scope":
        reason = reason + _CAA_SCOPE_CAVEAT

    return {
        "axis": spec.axis,
        "hypothesis": spec.hypothesis,
        "family": spec.family,
        "current_core_default": spec.default,
        "candidates": ",".join(candidates),
        "n_paired_E": n_paired,
        "means_E": ",".join(f"{c}={means:.3f}" for c, means in e_rank["means"].items()),
        "n_paired_causal": c_rank["n_paired"],
        "means_causal": ",".join(f"{c}={means:.3f}" for c, means in c_rank["means"].items()),
        "n_paired_S": s_rank["n_paired"],
        "means_suitability": ",".join(f"{c}={means:.3f}" for c, means in s_rank["means"].items()),
        "suggested": winner,
        "delta_vs_current_default": delta_vs_default,
        "wilcoxon_p": wilcoxon_p,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_ranking_flips": len(flips),
        "verdict": verdict,
        "reason": reason,
    }


def suggest_core_settings(
    cells: Sequence[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    idx = _index_cells(cells)
    recipes = sorted({rec for (_m, _t, rec, _met) in idx if model is None or _m == model})
    multi_class = multi_class_tasks(cells, model=model)
    out: List[Dict[str, Any]] = []
    for spec in AXES:
        row = suggest_core_setting(idx, spec=spec, recipes=recipes, model=model)
        if row is not None:
            out.append(row)
    for spec in POLE_AXES:
        row = suggest_core_setting(
            idx, spec=spec, recipes=recipes, model=model, eligible_tasks=multi_class
        )
        if row is not None:
            row["reason"] = f"[n_classes>2 tasks only] {row['reason']}"
            out.append(row)
    return out


def format_core_suggestions_text(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Symmetric core-setting suggestions (independent of core.yaml's current pick)",
        f"tau_E={DELTA_E_TAU}  large_delta={LARGE_DELTA}  min_flips={MIN_FLIP_TASKS}  min_paired={MIN_PAIRED}",
        "verdict: keep_core | suggest_change | no_clear_winner | insufficient_data",
        "Ranking metric: encoding_E (full coverage). causal/suitability means shown as context only.",
        "This does NOT auto-edit core.yaml -- suggest_change rows need a human look before promoting.",
        "",
        "-- What each axis actually varies (mechanism, not just a label) --",
        "",
    ]
    seen_axes = set()
    for row in rows:
        axis = row["axis"]
        if axis in seen_axes:
            continue
        seen_axes.add(axis)
        lines.append(f"[{axis}] default={row['current_core_default']}")
        lines.append(f"  {row['hypothesis']}")
        lines.append("")
    lines.append("-- Per-axis verdict --")
    lines.append("")
    headers = (
        f"{'axis':<14} {'current_default':<40} {'suggested':<40} {'verdict':<16} "
        f"{'nE':>3} {'dE':>7}  reason"
    )
    lines.append(headers)
    lines.append("-" * len(headers) + "-" * 40)

    def _fmt(v: Any, width: int = 7) -> str:
        if v is None:
            return f"{'-':>{width}}"
        return f"{float(v):>{width}.3f}"

    for row in rows:
        lines.append(
            f"{row['axis']:<14} {row['current_core_default']:<40} "
            f"{str(row['suggested']):<40} {row['verdict']:<16} "
            f"{row['n_paired_E']:>3} {_fmt(row['delta_vs_current_default'])}  {row['reason']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Best-single-layer-vs-all-layers: diagnostic only, never a suggestion.
#
# "Which single layer wins" can only be answered by looking at each task's
# own scores and picking the best one after the fact -- that is test-set
# selection by construction (the same reason classify_recipe marks every
# per-layer recipe an oracle and decide_retention refuses to promote them).
# This reports the gap so a human can judge how much the all_k1/all_act_*
# aggregation is leaving on the table relative to a hypothetical oracle
# single-layer pick, without ever recommending "pin layer N in core.yaml".
# ---------------------------------------------------------------------------

_SAE_LAYER = re.compile(r"^L(\d+)$")
_CAA_LAYER_PREDICTION = re.compile(r"^L(\d+)_act_prediction$")

# sae/sae_pre's per-layer oracle recipe comes in two shapes: bare `L{n}` (full
# metric set, including encoding_E -- what we need here) and `L{n}_k1` (a
# causal-only k=1-at-that-layer probe, no encoding_E). Use the bare form so
# the comparison actually has an encoding_E number per layer to rank.
_LAYER_DIAGNOSTIC_SPECS: Tuple[Tuple[str, str, "re.Pattern[str]"], ...] = (
    ("sae", "sae:all_k1", _SAE_LAYER),
    ("sae_pre", "sae_pre:all_k1", _SAE_LAYER),
    ("caa", "caa:all_act_prediction", _CAA_LAYER_PREDICTION),
)


def best_single_layer_vs_all(
    idx: Mapping[Tuple[str, str, str, str], float],
    *,
    family: str,
    all_recipe: str,
    layer_pattern: "re.Pattern[str]",
    metric: str = "encoding_E",
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Per task: does the best-performing single layer beat the all-layers recipe?

    Restricted to same-k-rule layer recipes (`L{n}_k1` for sae/sae_pre,
    `L{n}_act_prediction` for caa) so the comparison isolates aggregation
    (all layers vs one) from any k/pooling difference.
    """
    by_task: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (m, t, rec, met) in idx:
        if met != metric or (model is not None and m != model):
            continue
        fam, _, rest = rec.partition(":")
        if fam != family:
            continue
        if rec == all_recipe:
            by_task.setdefault((m, t), {})["__all__"] = idx[(m, t, rec, met)]
            continue
        mobj = layer_pattern.match(rest)
        if mobj:
            by_task.setdefault((m, t), {})[mobj.group(1)] = idx[(m, t, rec, met)]

    out: List[Dict[str, Any]] = []
    for (m, t), layer_vals in sorted(by_task.items()):
        all_val = layer_vals.pop("__all__", None)
        if all_val is None or not layer_vals:
            continue
        best_layer, best_val = max(layer_vals.items(), key=lambda kv: kv[1])
        out.append(
            {
                "model": m,
                "task": t,
                "family": family,
                "all_recipe": all_recipe,
                "all_value": all_val,
                "n_layers_seen": len(layer_vals),
                "best_layer": f"L{best_layer}",
                "best_layer_value": best_val,
                "delta_best_layer_minus_all": best_val - all_val,
            }
        )
    return out


def format_layer_diagnostic_text(rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    lines = [
        "Best-single-layer vs all-layers (DIAGNOSTIC ONLY -- not a core suggestion)",
        "",
        "This table is a HYPOTHETICAL bound, computed here for inspection only: it",
        "picks, after the fact, whichever layer happened to score highest on the",
        "same data used to report the metric. That is test-set selection, and if",
        "core.yaml ever pinned a layer chosen this way, the resulting number would",
        "be optimistic. Nothing in the real pipeline does this: no core.yaml",
        "recipe picks a layer, k, or checkpoint by scanning the eval/test split for",
        "the best score. Concretely: sae:kstar picks k by max(0.5*(auc_n+auc_o)) on",
        "the VALIDATION split ONLY (sae_eval.py, study/sae_engine.py), then reports that fixed k's score on test -- k is never re-chosen",
        "from test data. GRADIEND/ACTIEND checkpoint selection is the same shape:",
        "convergent_metric is computed from validation data (gradiend/trainer/trainer.py:"
        "1227-1256, resolve_split_for_role -> 'validation'). This table exists only to",
        "show how much all_k1/all_act_prediction's aggregation costs vs. that",
        "hypothetical oracle pick -- never to recommend pinning a layer in core.yaml.",
        "",
    ]
    header = f"{'family':<10} {'task':<24} {'all_value':>10} {'best_layer':>11} {'best_value':>11} {'delta':>8}"
    for family, rows in rows_by_family.items():
        if not rows:
            continue
        lines.append(f"-- {family} --")
        lines.append(header)
        for row in sorted(rows, key=lambda r: -abs(r["delta_best_layer_minus_all"])):
            lines.append(
                f"{row['family']:<10} {row['task']:<24} {row['all_value']:>10.3f} "
                f"{row['best_layer']:>11} {row['best_layer_value']:>11.3f} "
                f"{row['delta_best_layer_minus_all']:>+8.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def format_retention_text(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Ablation retention (pre-registered tests; not a winner table)",
        f"tau_E={DELTA_E_TAU}  tau_causal={DELTA_CAUSAL_TAU}  large_delta={LARGE_DELTA}  min_flips={MIN_FLIP_TASKS}",
        "tier: main | review_promote | appendix_small | oracle",
        "pE = Wilcoxon signed-rank p on paired dE (diagnostic only, n<8 -> '-'; does not gate tier)",
        "",
    ]
    headers = (
        f"{'axis':<14} {'variant':<42} {'tier':<16} "
        f"{'nE':>3} {'dE':>7} {'pE':>6} {'dCaus':>7} {'flips':>5}  reason"
    )
    lines.append(headers)
    lines.append("-" * len(headers) + "-" * 40)

    def _fmt(v: Any, width: int = 7) -> str:
        if v is None:
            return f"{'-':>{width}}"
        return f"{float(v):>{width}.3f}"

    for row in rows:
        lines.append(
            f"{row['axis']:<14} {row['variant']:<42} {row['tier']:<16} "
            f"{row['n_paired_E']:>3} {_fmt(row['mean_dE'])} {_fmt(row.get('dE_wilcoxon_p'), 6)} "
            f"{_fmt(row['mean_dCausal'])} "
            f"{row['n_ranking_flips']:>5}  {row['reason']}"
        )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=Path,
        default=ROOT / "analysis" / "tables" / "recipe" / "recipe_cells_long.csv",
        help="Locked-recipe cells (from summarize_family_overview -- recipe tables)",
    )
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument(
        "--study-tasks",
        action="store_true",
        help="Restrict to visible TASKS=all study tasks; excludes hidden diagnostic tasks such as ioi.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "analysis" / "tables" / "ablation_retention",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    cells = load_recipe_cells(args.cells)
    if args.model:
        cells = [c for c in cells if c["model"] == args.model]
    if args.study_tasks:
        cells = restrict_to_study_tasks(cells)
    rows = evaluate_all(cells, model=args.model or None)
    oracles = oracle_inventory(cells, model=args.model or None)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = [
        "axis",
        "variant",
        "default",
        "family",
        "tier",
        "reason",
        "oracle",
        "core",
        "n_tasks_variant",
        "n_paired_E",
        "mean_dE",
        "median_dE",
        "frac_E_beat_tau",
        "dE_ci_lo",
        "dE_ci_hi",
        "dE_wilcoxon_p",
        "n_paired_causal",
        "mean_dCausal",
        "frac_causal_beat_tau",
        "dCausal_ci_lo",
        "dCausal_ci_hi",
        "dCausal_wilcoxon_p",
        "n_paired_S",
        "mean_dS",
        "n_ranking_flips",
        "flip_tasks",
        "hypothesis",
    ]
    _write_csv(args.out / "retention.csv", rows, fields)
    text = format_retention_text(rows)

    suggestion_rows = suggest_core_settings(cells, model=args.model or None)
    suggestion_fields = [
        "axis",
        "family",
        "current_core_default",
        "suggested",
        "verdict",
        "reason",
        "n_paired_E",
        "delta_vs_current_default",
        "wilcoxon_p",
        "ci_lo",
        "ci_hi",
        "n_ranking_flips",
        "candidates",
        "means_E",
        "n_paired_causal",
        "means_causal",
        "n_paired_S",
        "means_suitability",
        "hypothesis",
    ]
    _write_csv(args.out / "core_suggestions.csv", suggestion_rows, suggestion_fields)
    suggestion_text = format_core_suggestions_text(suggestion_rows)
    (args.out / "core_suggestions.txt").write_text(suggestion_text + "\n", encoding="utf-8")
    print()
    print(suggestion_text)
    print(f"Wrote {args.out / 'core_suggestions.csv'}")

    idx_for_layers = _index_cells(cells)
    layer_fields = [
        "family",
        "model",
        "task",
        "all_recipe",
        "all_value",
        "n_layers_seen",
        "best_layer",
        "best_layer_value",
        "delta_best_layer_minus_all",
    ]
    layer_rows_by_family: Dict[str, List[Dict[str, Any]]] = {}
    layer_rows_flat: List[Dict[str, Any]] = []
    for family, all_recipe, pattern in _LAYER_DIAGNOSTIC_SPECS:
        rows_f = best_single_layer_vs_all(
            idx_for_layers,
            family=family,
            all_recipe=all_recipe,
            layer_pattern=pattern,
            model=args.model or None,
        )
        layer_rows_by_family[family] = rows_f
        layer_rows_flat.extend(rows_f)
    _write_csv(args.out / "layer_vs_all_diagnostic.csv", layer_rows_flat, layer_fields)
    layer_text = format_layer_diagnostic_text(layer_rows_by_family)
    (args.out / "layer_vs_all_diagnostic.txt").write_text(layer_text + "\n", encoding="utf-8")
    print()
    print(layer_text)
    print(f"Wrote {args.out / 'layer_vs_all_diagnostic.csv'}")

    extra = [
        "",
        f"Oracle recipes (not scored as promotion candidates): {oracles or '{}'}",
        f"Source cells: {args.cells}",
        f"Model: {args.model or 'all'}",
        "Headline 4-way: GRADIEND pole / ACTIEND pole / SAE k=1 / CAA act_prediction.",
        "Diagnostic stats (2026-08-18): dE_ci_lo/hi, dE_wilcoxon_p, dCausal_ci_lo/hi, "
        "dCausal_wilcoxon_p added to retention.csv. Bootstrap CI + Wilcoxon signed-rank, "
        "pure stdlib. These do NOT gate tiers -- the frozen tau/flip thresholds are",
        "unchanged. They exist to check, before acting on a review_promote row, whether",
        "the paired mean delta is distinguishable from noise at n~10-15 tasks.",
    ]
    (args.out / "retention.txt").write_text(text + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(text)
    print("\n".join(extra))
    print(f"Wrote {args.out / 'retention.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
