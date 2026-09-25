"""
Shared causal evaluation for GRADIEND / ACTIEND / SAE (gender EN).

Protocol
--------
- **GRADIEND / ACTIEND**: headline causal metrics are derived from the same
  decoder-evaluation grid/summary that backs the LR / P(class) plots
  (``trainer.evaluate_decoder`` → ``decoder_grid_cache.json``). Selection:
  LMS-gated maximize **P(target) on the other class's dataset**.
- **SAE**: same scoring/LMS helpers under durable residual steering hooks
  (``compute_probability_shift_score_clm`` / package ``compute_lms``).

Interventions (not interchangeable)
-----------------------------------
- **GRADIEND**: weight rewrite via ``modify_model(learning_rate=…)``. No token
  selector — there is no residual-site gating.
- **ACTIEND**: activation hooks via ``modify_model``. Package default is the
  **combined** form ``token_selector="encoder_direction"`` (= steer all tokens
  where enc·direction > τ). Explicit equivalent (preferred in this study, matches
  the experimental sweep): ``token_selector="all", activation_gate="encoder_direction"``.
  Ablations: ungated ``all``, ungated ``prediction``, and (optional) gated prediction.
- **SAE**: ``h ← h + α · W_dec[f]`` at the SAE residual module. Default token
  scope is ``all`` (standard SAE feature steering). ``prediction`` is a valid
  alternative ablation for activation steering, not a GRADIEND analogue.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax

# Match GRADIEND default decoder LR spacing, extended well past prior ceilings
# (≤1000 then ≤10000 still hit SAE L11 / tok_prediction ceiling flags — now
# ≤100000) and below the old 1e-3 floor (circuit one-pole / continuation_match
# often lands on floor+null).
DEFAULT_CAUSAL_STRENGTHS: Tuple[float, ...] = tuple(
    m * 10 ** e
    for e in range(5, -6, -1)
    for m in (5, 2, 1)
    if m * 10 ** e <= 100000
)
# The study selects directly from the shared 1/2/5 coarse grid. This keeps the
# causal protocol identical across backends and avoids extra inference passes.
CAUSAL_STRENGTH_REFINE_POINTS = 0
# Activation-clamp target grid (Templeton et al. 2024 ablation, see
# run_sae_clamp_causal_sweep): multiples of the feature's own observed max
# activation on the target class. 0 tests suppression; the rest amplification.
# No ±sign sweep, unlike DEFAULT_CAUSAL_STRENGTHS — SAE (ReLU) feature
# activations are already >= 0, so a target grid alone covers both directions.
DEFAULT_CLAMP_TARGET_MULTIPLIERS: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
LMS_RATIO = 0.99
# Headroom-normalized effect (AxBench-style "effectiveness": delta_target /
# (1 - base_p)) is nulled out below this remaining headroom rather than
# reported as a huge/undefined ratio -- a task already at base_p > 0.98 has
# almost no room left to move P(target) further, so a tiny absolute delta
# would otherwise blow up into a misleadingly large normalized effect.
CAUSAL_EFFECTIVENESS_MIN_HEADROOM = 0.02
# Paper defaults: 1000 texts/group for causal scoring; LMS and decoder eval share N.
CAUSAL_N_PER_GROUP = 1000
CAUSAL_LMS_MAX_TEXTS = 1000
# Package ``evaluate_decoder(max_size=…)`` sets training-like *and* LMS/neutral caps.
DECODER_EVAL_MAX_SIZE = CAUSAL_LMS_MAX_TEXTS



@dataclass
class CausalSample:
    sample_id: str
    group: str  # "M" | "F" | "neutral" | …
    text: str
    p_he_base: float
    p_she_base: float
    margin_base: float
    p_he_mod: float
    p_she_mod: float
    margin_mod: float

    @property
    def delta_p_he(self) -> float:
        return self.p_he_mod - self.p_he_base

    @property
    def delta_p_she(self) -> float:
        return self.p_she_mod - self.p_she_base

    @property
    def delta_margin(self) -> float:
        return self.margin_mod - self.margin_base

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["delta_p_he"] = self.delta_p_he
        d["delta_p_she"] = self.delta_p_she
        d["delta_margin"] = self.delta_margin
        return d


@dataclass
class StrengthResult:
    strength: float
    lms: Optional[float]
    base_lms: Optional[float]
    lms_ok: bool
    samples: List[CausalSample] = field(default_factory=list)
    summary_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Toward-target effect on the *other* class's dataset (ΔP(target|other)).
    signed_effect: float = 0.0
    is_random_control: bool = False
    notes: str = ""
    # Absolute P(target|other) used for LMS-gated selection (decoder strengthen).
    selection_metric: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strength": self.strength,
            "lms": self.lms,
            "base_lms": self.base_lms,
            "lms_ok": self.lms_ok,
            "signed_effect": self.signed_effect,
            "selection_metric": self.selection_metric,
            "is_random_control": self.is_random_control,
            "notes": self.notes,
            "summary_by_group": self.summary_by_group,
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass
class CausalMethodResult:
    method: str  # e.g. gradiend:M
    backend: str
    target_class: str
    strengths: List[StrengthResult]
    selected_strength: Optional[float]
    selected: Optional[StrengthResult]
    modified_model_path: Optional[str] = None
    random_control: Optional[StrengthResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # Weakening is a distinct intervention, not a statistic that can be
    # recovered from the strengthen-polarity curve.  ``weaken_selected`` is
    # selected on validation and re-evaluated on the report split exactly like
    # ``selected``.  Keeping it first-class prevents pairwise IEND from looking
    # for a decrease while still steering toward the class being weakened.
    weaken_strengths: List[StrengthResult] = field(default_factory=list)
    weaken_selected_strength: Optional[float] = None
    weaken_selected: Optional[StrengthResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
            "target_class": self.target_class,
            "selected_strength": self.selected_strength,
            "modified_model_path": self.modified_model_path,
            "meta": self.meta,
            "selected": None if self.selected is None else {
                k: v for k, v in self.selected.to_dict().items() if k != "samples"
            },
            "strengths": [
                {k: v for k, v in s.to_dict().items() if k != "samples"}
                for s in self.strengths
            ],
            "random_control": None
            if self.random_control is None
            else {k: v for k, v in self.random_control.to_dict().items() if k != "samples"},
            "weaken_selected_strength": self.weaken_selected_strength,
            "weaken_selected": None
            if self.weaken_selected is None
            else {
                k: v
                for k, v in self.weaken_selected.to_dict().items()
                if k != "samples"
            },
            "weaken_strengths": [
                {k: v for k, v in s.to_dict().items() if k != "samples"}
                for s in self.weaken_strengths
            ],
        }


# ---------------------------------------------------------------------------
# Token / LMS helpers
# ---------------------------------------------------------------------------

def _token_id(tokenizer, surface: str) -> int:
    """Fallback single id: last piece of a space-prefixed encode.

    Study causal scoring does not use this. Prefer :func:`_surface_token_ids`
    (SUM over space/case vocab matches).
    """
    ids = tokenizer.encode(f" {surface}", add_special_tokens=False) or tokenizer.encode(
        surface, add_special_tokens=False
    )
    if not ids:
        raise ValueError(f"Could not encode token {surface!r}")
    return int(ids[-1])


def _surface_token_ids(tokenizer, surface: str) -> List[int]:
    """Ids whose vocab surface matches ``surface``, ignoring leading space and case.

    Aggregation is SUM over this list (package decoder eval), not MAX.
    """
    from study.clm_score_patch import surface_variant_token_ids

    ids = surface_variant_token_ids(tokenizer, surface)
    if ids:
        return ids
    return [_token_id(tokenizer, surface)]


def token_pair_probs(
    logits: torch.Tensor,
    tokenizer,
    *,
    token_a: str,
    token_b: str,
) -> Tuple[float, float, float]:
    """SUM-over-variants P(token_a) vs P(token_b) on a 1D logit vector.

    Unit-test helper for decoder id aggregation. Headline causal scoring uses
    :func:`score_class_probs_by_dataset`, which shares the same id resolver.
    """
    a_ids = _surface_token_ids(tokenizer, token_a)
    b_ids = _surface_token_ids(tokenizer, token_b)
    probs = softmax(logits.float(), dim=-1)
    p_a = float(probs[a_ids].sum().item())
    p_b = float(probs[b_ids].sum().item())
    return p_a, p_b, p_a - p_b


def _lms_scalar(lms_result: Any) -> Optional[float]:
    if lms_result is None:
        return None
    if isinstance(lms_result, (int, float)):
        return float(lms_result)
    if isinstance(lms_result, dict):
        if "lms" in lms_result and isinstance(lms_result["lms"], (int, float)):
            return float(lms_result["lms"])
        if "perplexity" in lms_result and isinstance(lms_result["perplexity"], (int, float)):
            ppl = float(lms_result["perplexity"])
            if not math.isfinite(ppl) or ppl <= 0:
                return 0.0
            nll = math.log(ppl)
            return 1.0 / (1.0 + nll) if nll < 700 else 0.0
    return None


def compute_lms_safe(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_texts: int = CAUSAL_LMS_MAX_TEXTS,
    ignore: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """Package LMS (``compute_lms``). Requires durable intervention on ``model`` if steered."""
    try:
        from gradiend.trainer.text.common.lm_eval import compute_lms
    except ImportError:
        return None
    if not texts:
        return None
    try:
        result = compute_lms(
            model,
            tokenizer,
            list(texts),
            ignore=list(ignore) if ignore else None,
            max_texts=max_texts,
            batch_size=8,
        )
        return _lms_scalar(result)
    except Exception:
        return None


def select_lms_gated(
    candidates: Sequence[StrengthResult],
    *,
    ratio: float = LMS_RATIO,
    prefer_signed_effect: bool = True,
) -> Optional[StrengthResult]:
    """LMSThresholdPolicy among lms >= ratio * base_lms.

    Default objective: maximize ``signed_effect`` (ΔP toward target on other),
    with tie-break to **smaller** |strength|. Used for SAE/CAA sweeps that have
    no decoder-plot star. GRADIEND/ACTIEND headlines must **not** go through
    this: they use ``select_decoder_headline`` (package ``learning_rate``).

    ``prefer_signed_effect=False`` maximizes absolute P(target) on the other
    class's dataset — the same objective as package ``LMSThresholdPolicy``.

    If no candidate passes the LMS gate, prefer best score among those with
    finite LMS; only then fall back to smallest |strength| (and set note).
    """
    if not candidates:
        return None

    def _key(c: StrengthResult) -> Tuple[float, float]:
        # Lexicographic max: primary score, then smaller |strength| via -abs.
        if prefer_signed_effect:
            primary = float(c.signed_effect)
        elif c.selection_metric is not None:
            primary = float(c.selection_metric)
        else:
            primary = float(c.signed_effect)
        return (primary, -abs(float(c.strength)))

    base = next((c.base_lms for c in candidates if c.base_lms is not None), None)
    if base is None:
        return max(candidates, key=_key)
    cutoff = ratio * float(base)
    passing = [c for c in candidates if c.lms is not None and c.lms >= cutoff]
    if passing:
        return max(passing, key=_key)
    finite = [c for c in candidates if c.lms is not None and math.isfinite(float(c.lms))]
    if finite:
        chosen = max(finite, key=_key)
        note = "lms_gate_empty_best_finite"
        if note not in (chosen.notes or ""):
            chosen.notes = (chosen.notes + " " if chosen.notes else "") + note
        return chosen
    non_zero = [c for c in candidates if c.strength != 0]
    if not non_zero:
        return candidates[0]
    chosen = min(non_zero, key=lambda c: abs(c.strength))
    note = "lms_gate_empty_min_strength"
    if note not in (chosen.notes or ""):
        chosen.notes = (chosen.notes + " " if chosen.notes else "") + note
    return chosen


def _parse_sweep_sign(notes: Optional[str]) -> Optional[float]:
    """Recover the ``sign=`` tag from a StrengthResult's notes, if present.

    Returns ``None`` when no ``sign=`` tag exists in ``notes`` -- callers must
    decide what that means for them: some sweeps (SAE clamp) never mix signs
    at all, so "no distinction needed" is a safe read of ``None``; others
    (the additive SAE sweeps) genuinely sweep both polarities and tag every
    result, so ``None`` there means a real bug (a result that should have
    been tagged and wasn't). This function deliberately does NOT default to
    +1.0 on a missing/malformed tag: for a sign-mixing sweep, silently
    treating an untagged result as positive would point a selected/reported
    steering direction the wrong way with no error -- see
    ``refine_and_select_lms_gated`` for the loud-failure guard that replaced
    that silent default.
    """
    if notes and "sign=" in notes:
        try:
            return float(notes.split("sign=")[1].split()[0])
        except Exception:
            return None
    return None


def refine_and_select_lms_gated(
    results: List[StrengthResult],
    run_one: Callable[[float, float], StrengthResult],
    *,
    n_refine: int = CAUSAL_STRENGTH_REFINE_POINTS,
    prefer_signed_effect: bool = True,
    ratio: float = LMS_RATIO,
) -> Tuple[List[StrengthResult], Optional[StrengthResult]]:
    """Bisect the first same-sign LMS pass→fail boundary, then re-select."""
    coarse = select_lms_gated(results, ratio=ratio, prefer_signed_effect=prefer_signed_effect)
    if coarse is None or n_refine <= 0:
        return results, coarse
    sign0 = _parse_sweep_sign(coarse.notes)
    all_signs = {_parse_sweep_sign(r.notes) for r in results}
    if sign0 is None:
        if all_signs - {None}:
            # Some results in this sweep carry a real sign= tag but the
            # coarse-selected one doesn't -- this sweep genuinely mixes
            # signs and forgot to tag this result. Silently treating it as
            # "no distinction needed" (old behavior: default sign=1.0 for
            # everyone) could refine/report the wrong polarity with no
            # error. Fail loudly instead -- see _parse_sweep_sign's
            # docstring for why a silent default was removed.
            raise ValueError(
                "refine_and_select_lms_gated: results carry a sign= tag but "
                "the coarse-selected result's notes do not "
                f"(notes={coarse.notes!r}); cannot determine which sign "
                "group to refine. Every StrengthResult from a sign-mixing "
                "sweep must include sign= in its notes."
            )
        # No result in this sweep carries a sign= tag at all -- this sweep
        # never distinguishes polarity (e.g. SAE clamp, target >= 0 only),
        # so every result is one group.
        same_sign = sorted(results, key=lambda r: r.strength)
    else:
        same_sign = sorted(
            (r for r in results if _parse_sweep_sign(r.notes) == sign0),
            key=lambda r: r.strength,
        )
    if len(same_sign) < 2:
        return results, coarse

    base_lms = next((r.base_lms for r in same_sign if r.base_lms is not None), None)
    if base_lms is None:
        return results, coarse
    cutoff = ratio * base_lms
    lo = hi = None
    for a, b in zip(same_sign[:-1], same_sign[1:]):
        if a.lms is None or b.lms is None:
            continue
        if a.lms >= cutoff > b.lms:
            lo, hi = a.strength, b.strength
            break
    if lo is None or hi is None:
        return results, coarse

    all_results = list(results)
    for _ in range(n_refine):
        mid = math.sqrt(lo * hi) if lo > 0 and hi > 0 else (lo + hi) / 2.0
        if any(abs(r.strength - mid) < 1e-12 for r in all_results):
            break
        new_r = run_one(sign0, mid)
        all_results.append(new_r)
        if new_r.lms is None:
            break
        if new_r.lms >= cutoff:
            lo = mid
        else:
            hi = mid

    selected = select_lms_gated(all_results, ratio=ratio, prefer_signed_effect=prefer_signed_effect)
    return all_results, selected




def strength_curve_all(
    candidates: Sequence[StrengthResult],
) -> List[StrengthResult]:
    """All non-control strength points, sorted by signed strength (for LMS plots)."""
    rows = [c for c in candidates if not c.is_random_control]
    return sorted(rows, key=lambda c: float(c.strength))




def _lms_passes(sr: StrengthResult, *, ratio: float) -> bool:
    if sr.base_lms is None or sr.lms is None:
        return True
    return float(sr.lms) >= float(ratio) * float(sr.base_lms)


def selection_lms_metadata(selected: Optional[StrengthResult]) -> Dict[str, Any]:
    """Validation-selection LMS, kept separate from the frozen test headline."""
    if selected is None:
        return {
            "selection_strength": None,
            "selection_signed_effect": None,
            "selection_lms": None,
            "selection_base_lms": None,
            "selection_lms_ratio_to_base": None,
            "selection_lms_ok": None,
        }
    ratio_to_base = (
        None
        if selected.lms is None or selected.base_lms in (None, 0)
        else float(selected.lms) / float(selected.base_lms)
    )
    return {
        "selection_strength": selected.strength,
        "selection_signed_effect": selected.signed_effect,
        "selection_lms": selected.lms,
        "selection_base_lms": selected.base_lms,
        "selection_lms_ratio_to_base": ratio_to_base,
        "selection_lms_ok": _lms_passes(selected, ratio=LMS_RATIO),
    }


def other_class(target_class: str, classes: Sequence[str]) -> Optional[str]:
    """Rival panel for ``target_class``, or ``None`` when one-pole (same-panel).

    Raises ``KeyError`` if ``classes`` is empty or ``target_class`` is absent.
    Never invents a default class list (e.g. M/F).
    """
    target = str(target_class)
    named = [str(c) for c in classes]
    if not named:
        raise KeyError("classes")
    if target not in named:
        raise KeyError(target)
    others = [c for c in named if c != target]
    return others[0] if others else None


def class_names_from_probs(*maps: Mapping[str, Any]) -> Tuple[str, ...]:
    """Class ids from ``probs_by_dataset`` / ``summary_by_group`` keys (skip neutral).

    Raises ``KeyError`` when no class keys are present — never defaults to M/F.
    """
    names: List[str] = []
    seen = set()
    for probs in maps:
        if not isinstance(probs, dict):
            raise TypeError(
                f"probs map must be a dict, got {type(probs).__name__}"
            )
        for g in probs:
            gs = str(g)
            if gs.lower() == "neutral":
                continue
            if gs not in seen:
                seen.add(gs)
                names.append(gs)
    if not names:
        raise KeyError("class names from probs_by_dataset/summary_by_group")
    return tuple(names)


def _target_panel_has_claim(
    probs_by_dataset: Mapping[str, Any],
    target_class: str,
) -> bool:
    """True when the claim class panel carries P(claim) under the claim key."""
    tc = str(target_class)
    panel = probs_by_dataset.get(tc)
    return isinstance(panel, dict) and _resolve_target_prob_key(panel, tc) is not None


def strengthen_panel(target_class: str, classes: Sequence[str]) -> str:
    """Dataset panel for the decoder strengthen metric.

    Multi-class: rival panel (P(target) on other). One-pole: same-panel (P(target)
    on target). Raises ``KeyError`` if ``target_class`` ∉ ``classes``.
    """
    other = other_class(target_class, classes)
    return other if other is not None else str(target_class)


def strengthen_panel_from_summary(
    target_class: str, summary: Mapping[str, Any]
) -> str:
    """Strengthen panel from ``summary_by_group`` keys.

    Prefer the bipolar rival when ``target_class`` is a panel. If the claim
    panel was omitted (P(target) only scored on rival datasets), use the first
    non-neutral panel — do not raise ``KeyError(target)``.
    """
    names = class_names_from_probs(summary)
    target = str(target_class)
    if target in names:
        return strengthen_panel(target, names)
    return names[0]


def strengthen_panel_from_probs(
    probs_by_dataset: Mapping[str, Any],
    target_class: str,
) -> str:
    """Pick strengthen panel from ``probs_by_dataset`` shape (not outer panel count).

    Circuit one-pole tasks may expose both MATCH and DISTRACTOR *outer* panels
    (expanded CF rows for CAA) while the decoder strengthen contract remains
    same-panel: P(claim) on the claim factual dataset. When a rival outer panel
    also exposes P(claim), keep the cross-panel decoder contract (bipolar /
    prob_on_other_class).

    Multi-class MIB/ravel tasks may score P(target) only on *rival* outer panels
    (no ``target`` outer key). Use the first rival panel that carries P(target).
    """
    tc = str(target_class)

    def _rival_panels_with_target() -> List[str]:
        return [
            str(g)
            for g in probs_by_dataset
            if str(g).lower() != "neutral"
            and str(g) != tc
            and isinstance(probs_by_dataset.get(g), dict)
            and _resolve_target_prob_key(probs_by_dataset[g], tc) is not None
        ]

    if not _target_panel_has_claim(probs_by_dataset, tc):
        rivals = _rival_panels_with_target()
        if rivals:
            return rivals[0]
        names = class_names_from_probs(probs_by_dataset)
        if tc in names:
            return strengthen_panel(tc, names)
        if names:
            return names[0]
        raise KeyError(f"no probs_by_dataset panels containing {tc!r}")
    rival_panels = _rival_panels_with_target()
    if rival_panels:
        return rival_panels[0]
    return tc


def signed_effect_from_group_summary(
    summary: Mapping[str, Any],
    *,
    target_class: str,
) -> float:
    """ΔP(target) on the strengthen panel from a stored group summary.

    Rival panel when present; same-panel when one-pole. Requires
    ``mean_delta_p_target`` on that panel — no silent defaults.
    """
    if not isinstance(summary, dict):
        raise TypeError(f"summary_by_group must be a dict, got {type(summary).__name__}")
    if not summary:
        raise KeyError("summary_by_group")
    tc = str(target_class)
    panel = strengthen_panel_from_summary(tc, summary)
    g = summary[panel]
    return float(g["mean_delta_p_target"])


def lookup_strength_result(
    results: Sequence[StrengthResult],
    by_lr: Mapping[float, StrengthResult],
    lr: float,
) -> Optional[StrengthResult]:
    hit = by_lr.get(float(lr))
    if hit is not None:
        return hit
    return next((r for r in results if abs(float(r.strength) - float(lr)) < 1e-12), None)


def select_decoder_headline(
    results: Sequence[StrengthResult],
    by_lr: Mapping[float, StrengthResult],
    *,
    package_lr: Optional[float],
) -> StrengthResult:
    """Headline LR is the decoder-plot star (package summary), not a second sweep.

    When ``package_lr`` is set it **must** exist on the grid (``KeyError`` otherwise).
    Only if the package summary has no LR: LMS×0.99 maximize absolute P(target)
    on the strengthen panel — same objective as ``LMSThresholdPolicy``.
    """
    if package_lr is not None:
        selected = lookup_strength_result(results, by_lr, float(package_lr))
        if selected is None:
            raise KeyError(f"package_learning_rate={package_lr!r} not in decoder grid")
        return selected
    chosen = select_lms_gated(results, prefer_signed_effect=False)
    if chosen is None:
        raise KeyError("decoder grid has no LMS-selectable strength")
    return chosen


# ---------------------------------------------------------------------------
# Data sampling
# ---------------------------------------------------------------------------

def stratified_causal_texts(
    gender_df,
    neutral_df,
    *,
    target_classes: Sequence[str] = ("M", "F"),
    n_per_group: int = CAUSAL_N_PER_GROUP,
    split: str = "test",
    text_col: str = "masked",
    label_col: str = "label_class",
    neutral_text_col: str = "text",
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Return list of {sample_id, group, text} with n_per_group per class + neutrals."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, str]] = []
    split_df = gender_df[gender_df["split"] == split] if "split" in gender_df.columns else gender_df
    if split_df.empty:
        split_df = gender_df

    extra_cols = (
        "label",
        "alternative",
        "label_class",
        "alternative_class",
    )
    for cls in target_classes:
        cls = str(cls)
        sub = split_df[split_df[label_col].astype(str) == cls]
        if sub.empty:
            continue
        idx = rng.choice(len(sub), size=min(n_per_group, len(sub)), replace=False)
        for i, j in enumerate(idx):
            row = sub.iloc[int(j)]
            entry: Dict[str, str] = {
                "sample_id": f"{cls}:{i}",
                "group": cls,
                "text": str(row[text_col]),
            }
            for col in extra_cols:
                if col in sub.columns and pd.notna(row.get(col)):
                    entry[col] = str(row[col])
            rows.append(entry)

    if neutral_df is not None and len(neutral_df):
        neu = neutral_df
        if "split" in neu.columns:
            want = str(split)
            mask = neu["split"].astype(str).isin(["validation", "val"]) if want == "validation" else neu["split"].astype(str) == want
            sub = neu.loc[mask]
            if not sub.empty:
                neu = sub
        ntexts = neu[neutral_text_col].astype(str).tolist()
        idx = rng.choice(len(ntexts), size=min(n_per_group, len(ntexts)), replace=False)
        for i, j in enumerate(idx):
            rows.append({"sample_id": f"neutral:{i}", "group": "neutral", "text": ntexts[int(j)]})
    return rows


def meta_rows_to_frame(meta_rows: Sequence[Dict[str, str]]) -> pd.DataFrame:
    """Build a decoder-scoring frame from causal meta rows.

    Circuit / continuation_match tasks carry ``label`` + ``alternative`` on each
    row; map those to the package row-wise column names (``factual`` /
    ``factual_id``, …) so :func:`score_class_probs_by_dataset` can use exact
    row-wise scoring. Non-circuit tasks (no ``alternative``) still carry
    ``label`` here, which :func:`score_class_probs_by_dataset` uses to derive
    this task's own per-class word targets instead of any hardcoded set.
    """
    frame = pd.DataFrame(
        {
            "masked": [r["text"] for r in meta_rows],
            "label_class": [r.get("label_class", r["group"]) for r in meta_rows],
            "sample_id": [r["sample_id"] for r in meta_rows],
        }
    )
    if meta_rows and "label" in meta_rows[0]:
        frame["label"] = [str(r.get("label", "")) for r in meta_rows]
        frame["factual"] = frame["label"]
    if meta_rows and "alternative" in meta_rows[0]:
        frame["alternative"] = [str(r.get("alternative", "")) for r in meta_rows]
    if meta_rows and "alternative_class" in meta_rows[0]:
        frame["alternative_class"] = [
            str(r.get("alternative_class", "")) for r in meta_rows
        ]
        frame["alternative_id"] = frame["alternative_class"]
    frame["factual_id"] = frame["label_class"]
    return frame


def _frame_supports_row_wise_scoring(df: pd.DataFrame) -> bool:
    """True when ``df`` has per-row factual/alternative tokens + class ids."""
    if not {"label", "alternative"}.issubset(df.columns):
        if not {"factual", "alternative"}.issubset(df.columns):
            return False
    if {"label_class", "alternative_class"}.issubset(df.columns):
        return True
    return {"factual_id", "alternative_id"}.issubset(df.columns)


def _class_word_targets_from_frame(
    df: pd.DataFrame, *, dataset_class_col: str = "label_class"
) -> Dict[str, List[str]]:
    """Derive ``{class: [observed answer words]}`` from the frame's own rows.

    Task-agnostic replacement for a hardcoded word-probe dict (there used to be
    one hardcoded to gender he/she — see 2026-08-21 CLAUDE.md entry): reads
    whatever answer words *this task's own data* actually uses per class
    (``label``/``factual`` column) instead of assuming any particular task.
    Excludes the ``neutral`` group (not a scored class). Returns ``{}`` when
    the frame doesn't carry the columns needed to derive anything — callers
    must treat that as "cannot score", not silently substitute another task's
    words.
    """
    label_col = "label" if "label" in df.columns else ("factual" if "factual" in df.columns else None)
    ds_col = (
        dataset_class_col
        if dataset_class_col in df.columns
        else ("label_class" if "label_class" in df.columns else "factual_id")
    )
    if label_col is None or ds_col not in df.columns:
        return {}
    out: Dict[str, List[str]] = {}
    for cls, group in df.groupby(ds_col):
        cls = str(cls)
        if cls.lower() == "neutral":
            continue
        words = sorted(
            {str(w).strip() for w in group[label_col].dropna().astype(str) if str(w).strip()}
        )
        if words:
            out[cls] = words
    return out


# ---------------------------------------------------------------------------
# Package decoder-protocol scoring (mask-slot / prefix-before-[MASK])
# ---------------------------------------------------------------------------

def score_class_probs_by_dataset(
    model,
    tokenizer,
    df: pd.DataFrame,
    *,
    targets: Optional[Dict[str, List[str]]] = None,
    key_text: str = "masked",
    dataset_class_col: str = "label_class",
    batch_size: int = 8,
) -> Dict[str, Dict[str, float]]:
    """Mean P(class) per factual dataset group — package decoder scoring."""
    from study.clm_score_patch import apply_clm_surface_variant_sum_patch
    from gradiend.trainer.text.prediction.decoder_eval_utils import (
        compute_probability_shift_score_clm,
        compute_probability_shift_score_row_wise,
    )

    apply_clm_surface_variant_sum_patch()

    if targets is None and _frame_supports_row_wise_scoring(df):
        factual_col = "label" if "label" in df.columns else "factual"
        alt_col = "alternative"
        ds_col = (
            "label_class"
            if "label_class" in df.columns
            else ("factual_id" if "factual_id" in df.columns else dataset_class_col)
        )
        other_col = (
            "alternative_class"
            if "alternative_class" in df.columns
            else "alternative_id"
        )
        class_df = df[df[ds_col].astype(str).str.lower() != "neutral"].copy()
        if class_df.empty:
            raise ValueError("row-wise causal scoring: no non-neutral class rows")
        return compute_probability_shift_score_row_wise(
            model,
            tokenizer,
            class_df,
            key_text=key_text,
            factual_col=factual_col,
            alternative_col=alt_col,
            dataset_class_col=ds_col,
            other_class_col=other_col,
            batch_size=batch_size,
        )

    if targets is None:
        targets = _class_word_targets_from_frame(df, dataset_class_col=dataset_class_col)
    if not targets:
        raise ValueError(
            "score_class_probs_by_dataset: cannot score — no explicit "
            "'targets', no row-wise factual/alternative columns, and no "
            f"per-row label values to derive class word targets from "
            f"(dataset_class_col={dataset_class_col!r}, columns={list(df.columns)}). "
            "This used to silently fall back to a hardcoded gender {'M': "
            "['he'], 'F': ['she']} probe for any task, which is wrong for "
            "every non-gender task and was fixed 2026-08-21 (see CLAUDE.md)."
        )
    return compute_probability_shift_score_clm(
        model,
        tokenizer,
        df,
        targets,
        key_text=key_text,
        batch_size=batch_size,
        dataset_class_col=dataset_class_col,
    )


def p_target_on_group(
    probs_by_dataset: Dict[str, Dict[str, float]],
    *,
    target_class: str,
    group: str,
) -> float:
    """P(target_class) on ``group``'s dataset. Missing keys raise ``KeyError``."""
    block = probs_by_dataset[str(group)]
    key = _resolve_target_prob_key(block, str(target_class))
    if key is None:
        raise KeyError(str(target_class))
    return float(block[key])


def _resolve_target_prob_key(
    block: Mapping[str, Any],
    target_class: str,
) -> Optional[str]:
    """Return key in ``block`` representing ``target_class`` (exact/case/*_factual)."""
    tc = str(target_class)
    if tc in block:
        return tc
    factual = f"{tc}_factual"
    if factual in block:
        return factual
    tc_l = tc.lower()
    for k in block.keys():
        ks = str(k)
        if ks.lower() == tc_l:
            return ks
        if ks.lower().endswith("_factual") and ks[: -len("_factual")].lower() == tc_l:
            return ks
    return None


def strengthen_metric_from_probs(
    probs_by_dataset: Dict[str, Dict[str, float]],
    *,
    target_class: str,
    class_names: Optional[Sequence[str]] = None,
) -> float:
    """Decoder strengthen scalar: P(target) on the strengthen panel."""
    if class_names is not None:
        panel = strengthen_panel(str(target_class), tuple(str(c) for c in class_names))
    else:
        panel = strengthen_panel_from_probs(probs_by_dataset, str(target_class))
    return p_target_on_group(
        probs_by_dataset, target_class=str(target_class), group=panel
    )


def summary_by_group_from_probs(
    base_probs: Dict[str, Dict[str, float]],
    mod_probs: Dict[str, Dict[str, float]],
    *,
    target_class: str,
) -> Dict[str, Dict[str, float]]:
    """Per-group ΔP(target) (+ absolute base/mod) for headline dtgt/doth/dneu.

    Only groups present in **both** maps **and** containing ``target_class``.
    Panels that omit the claim (e.g. some neutral blocks) are skipped — not
    zero-filled. Raises ``KeyError`` if no scored panel remains.
    """
    target = str(target_class)
    groups = sorted(set(map(str, base_probs)) & set(map(str, mod_probs)))
    if not groups:
        raise KeyError("probs_by_dataset groups (base ∩ mod)")
    summary: Dict[str, Dict[str, float]] = {}
    for g in groups:
        b_block = base_probs[str(g)]
        m_block = mod_probs[str(g)]
        b_key = _resolve_target_prob_key(b_block, target)
        m_key = _resolve_target_prob_key(m_block, target)
        if b_key is None or m_key is None:
            continue
        b_v = float(b_block[b_key])
        m_v = float(m_block[m_key])
        summary[str(g)] = {
            "n": float("nan"),  # aggregate means; n unknown from dataset means
            "mean_delta_p_target": m_v - b_v,
            "mean_p_target_base": b_v,
            "mean_p_target_mod": m_v,
            # Alias used by headline_metrics for dtgt/doth/dneu columns:
            "mean_delta_margin": m_v - b_v,
        }
    if not summary:
        raise KeyError(f"no probs_by_dataset panels containing {target!r}")
    class_panels = [g for g in summary if str(g).lower() != "neutral"]
    if not class_panels:
        raise KeyError(f"no non-neutral probs_by_dataset panels containing {target!r}")
    return summary


# ---------------------------------------------------------------------------
# SAE intervention + persistence (durable package hooks)
# ---------------------------------------------------------------------------

def sae_decoder_direction(sae, feature_index: int) -> torch.Tensor:
    if hasattr(sae, "W_dec"):
        w = sae.W_dec
        if w.ndim == 2 and w.shape[0] > w.shape[1]:
            return w[feature_index].detach().float().cpu()
        if w.ndim == 2:
            return w[:, feature_index].detach().float().cpu()
    raise AttributeError("Could not read SAE decoder direction from W_dec")


def sae_decoder_bag_direction(
    sae,
    feature_indices: Sequence[int],
    *,
    reduce: str = "sum",
) -> torch.Tensor:
    """Combine several decoder columns (e.g. k* bag) into one steering vector."""
    idxs = [int(i) for i in feature_indices]
    if not idxs:
        raise ValueError("sae_decoder_bag_direction requires feature_indices")
    vecs = [sae_decoder_direction(sae, i) for i in idxs]
    stacked = torch.stack(vecs, dim=0)
    if reduce == "mean":
        return stacked.mean(dim=0)
    return stacked.sum(dim=0)


def make_sae_activation_intervention(
    module_path: str,
    direction: torch.Tensor,
    strength: float,
    *,
    token_selector: str = "all",
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Build package-compatible activation steering intervention (all-token by default)."""
    vec = (float(strength) * direction.flatten()).detach().float().cpu()
    interventions = [
        {
            "module": module_path,
            "tensor_key": "steering_0",
            "application": {"axis": "last_dim", "token_selector": token_selector},
        }
    ]
    tensors = {"steering_0": vec}
    return interventions, tensors


@contextmanager
def sae_steering_context(
    model,
    module_path: str,
    direction: torch.Tensor,
    strength: float,
    *,
    token_selector: str = "all",
) -> Iterator[Any]:
    """Attach durable SAE steering hooks for the block; package LMS sees the intervention."""
    from gradiend.model.modified import (
        apply_activation_steering,
        remove_hook_handles,
    )

    interventions, tensors = make_sae_activation_intervention(
        module_path, direction, strength, token_selector=token_selector
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


def sae_encoder_row(sae, feature_index: int) -> torch.Tensor:
    """Read a single feature's encoder weight vector off ``sae`` (length d_model).

    Mirrors ``sae_decoder_direction``'s shape-detection rule (index along
    whichever axis is the larger, i.e. feature-count, axis) reapplied to
    ``W_enc``'s opposite orientation (``(d_model, n_features)`` rather than
    ``W_dec``'s ``(n_features, d_model)``).
    """
    if not hasattr(sae, "W_enc"):
        raise AttributeError("Could not read SAE encoder weight from W_enc")
    w = sae.W_enc
    if w.ndim == 2 and w.shape[0] > w.shape[1]:
        weight = w[feature_index]
    elif w.ndim == 2:
        weight = w[:, feature_index]
    else:
        raise AttributeError(f"Unexpected SAE W_enc shape {tuple(w.shape)}")
    return weight.detach().float().cpu()


def sae_encoder_bias(sae, feature_index: int) -> torch.Tensor:
    """Read a single feature's encoder bias off ``sae`` (0.0 if the SAE has none)."""
    bias = getattr(sae, "b_enc", None)
    if bias is None:
        return torch.zeros(1)
    return bias[feature_index].detach().float().cpu().reshape(1)


def make_sae_clamp_intervention(
    module_path: str,
    sae,
    feature_index: int,
    target_activation: float,
    *,
    token_selector: str = "all",
    direction: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Build a package-compatible activation-clamping intervention (Templeton et al. 2024).

    Unlike ``make_sae_activation_intervention`` (plain addition), this forces
    the SAE feature's own activation to ``target_activation`` at each hooked
    position: ``h' = h + (target - relu(h·w + b))·d``. The steering vector
    is the raw (unscaled) decoder direction by default — the package hook
    computes the per-position correction itself from the encoder
    weight/bias. Pass ``direction`` to override ``d`` (e.g. a matched-norm
    random direction for a control run) while keeping the encoder
    weight/bias — and hence the feature-tracking correction magnitude — real.
    """
    direction = sae_decoder_direction(sae, feature_index) if direction is None else direction
    weight = sae_encoder_row(sae, feature_index).reshape(1, -1)
    bias = sae_encoder_bias(sae, feature_index)
    interventions = [
        {
            "module": module_path,
            "tensor_key": "steering_0",
            "application": {
                "axis": "last_dim",
                "token_selector": token_selector,
                "mode": "clamp",
                "target_activation": float(target_activation),
                "clamp_encoder_weight_key": "clamp_enc_w",
                "clamp_encoder_bias_key": "clamp_enc_b",
            },
        }
    ]
    tensors = {
        "steering_0": direction,
        "clamp_enc_w": weight,
        "clamp_enc_b": bias,
    }
    return interventions, tensors


@contextmanager
def sae_clamp_steering_context(
    model,
    module_path: str,
    sae,
    feature_index: int,
    target_activation: float,
    *,
    token_selector: str = "all",
    direction: Optional[torch.Tensor] = None,
) -> Iterator[Any]:
    """Attach a durable SAE activation-clamp hook for the block (parallel to ``sae_steering_context``)."""
    from gradiend.model.modified import (
        apply_activation_steering,
        remove_hook_handles,
    )

    interventions, tensors = make_sae_clamp_intervention(
        module_path,
        sae,
        feature_index,
        target_activation,
        token_selector=token_selector,
        direction=direction,
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


def save_sae_modified_model(
    output_dir: Path | str,
    *,
    base_model,
    tokenizer,
    module_path: str,
    feature_index: int,
    strength: float,
    direction: torch.Tensor,
    meta: Optional[Dict[str, Any]] = None,
    token_selector: str = "all",
) -> str:
    """Persist base HF weights + SAE intervention sidecar for downstream eval."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vec = direction.flatten().detach().float().cpu()
    # Store unmodified weights; intervention is in sidecar files.
    base_model.save_pretrained(str(out))
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(out))

    payload = {
        "format": "sae_modified_model",
        "version": 2,
        "module_path": module_path,
        "feature_index": int(feature_index),
        "strength": float(strength),
        "token_selector": token_selector,
        "meta": meta or {},
    }
    (out / "sae_modified_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({"direction": vec, "strength": float(strength)}, out / "sae_modified_tensors.pt")

    # Also write package-compatible gradiend_modified_* so load_modified_model works.
    try:
        from gradiend.model.modified import (
            MODIFIED_CONFIG_NAME,
            MODIFIED_TENSORS_NAME,
            activation_steering_config,
        )

        interventions, tensors = make_sae_activation_intervention(
            module_path, direction, strength, token_selector=token_selector
        )
        cfg = activation_steering_config(interventions)
        (out / MODIFIED_CONFIG_NAME).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        torch.save({k: v.detach().cpu() for k, v in tensors.items()}, out / MODIFIED_TENSORS_NAME)
    except Exception:
        pass
    return str(out)




# ---------------------------------------------------------------------------
# Decoder-cache → causal metrics (GRADIEND / ACTIEND)
# ---------------------------------------------------------------------------

def _alias_nested_class_keys(block: Dict[str, Any]) -> Dict[str, Any]:
    """Copy ``<class>_factual`` onto ``<class>`` inside a probs/group dict."""
    out = dict(block)
    for key, val in list(block.items()):
        if not str(key).endswith("_factual"):
            continue
        base = str(key)[: -len("_factual")]
        if not base or base in out:
            continue
        out[base] = val
    return out


def _alias_legacy_factual_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy ``<class>_factual`` scalars onto ``<class>`` for strengthen selection.

    Older decoder grids exposed top-level ``christian_factual`` etc. Package extractors
    now prefer ``probs`` / ``probs_factual``; alias so cached grids still work.
    Also alias nested ``probs_by_dataset`` inner keys (same suffix).
    """
    out = dict(entry)
    # Prefer explicit probs[class] when present.
    probs = _alias_nested_class_keys(dict(out.get("probs") or {}))
    probs_factual = dict(out.get("probs_factual") or {})
    for key, val in list(out.items()):
        if not isinstance(val, (int, float)) or not str(key).endswith("_factual"):
            continue
        base = str(key)[: -len("_factual")]
        if not base or base in probs:
            continue
        probs.setdefault(base, float(val))
        probs_factual.setdefault(base, float(val))
    if probs:
        out["probs"] = probs
    if probs_factual:
        out["probs_factual"] = probs_factual
    pbd = out.get("probs_by_dataset")
    if isinstance(pbd, dict):
        out["probs_by_dataset"] = {
            str(g): _alias_nested_class_keys(v) if isinstance(v, dict) else v
            for g, v in pbd.items()
        }
    return out


def _normalize_probs_by_dataset_panels(
    pbd: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for g, v in pbd.items():
        gs = str(g)
        if gs.endswith("_factual"):
            gs = gs[: -len("_factual")]
        if not isinstance(v, dict):
            continue
        normalized[gs] = {
            str(k): float(val)
            for k, val in _alias_nested_class_keys(v).items()
            if isinstance(val, (int, float))
        }
    return normalized


def _claim_scalar_from_entry(entry: Mapping[str, Any], target_class: str) -> Optional[float]:
    tc = str(target_class)
    for key in ("probs", "probs_factual"):
        block = entry.get(key) or {}
        if isinstance(block, dict) and tc in block:
            return float(block[tc])
    scalar = entry.get(f"{tc}_factual")
    if isinstance(scalar, (int, float)):
        return float(scalar)
    return None


def _coerce_probs_by_dataset_for_target(
    entry: Dict[str, Any],
    target_class: str,
) -> Dict[str, Dict[str, float]]:
    """Return ``probs_by_dataset`` with inner keys that include ``target_class``.

    Decoder grids normally expose full panels. When only same-panel ``probs`` /
    ``probs_factual`` scalars are present (one-pole refresh edge cases), synthesize
    the minimal ``{panel: {target: p}}`` shape causal headlines require.
    """
    entry = _alias_legacy_factual_metrics(dict(entry))
    tc = str(target_class)
    claim_scalar = _claim_scalar_from_entry(entry, tc)
    pbd = entry.get("probs_by_dataset")
    if isinstance(pbd, dict):
        normalized = _normalize_probs_by_dataset_panels(pbd)
        if normalized:
            for block in normalized.values():
                if tc not in block:
                    continue
                panel = normalized.get(tc)
                if isinstance(panel, dict) and tc not in panel:
                    if "factual" in panel:
                        normalized[tc] = {**panel, tc: float(panel["factual"])}
                    elif claim_scalar is not None:
                        normalized[tc] = {**panel, tc: claim_scalar}
                return normalized
            panel = normalized.get(tc)
            if isinstance(panel, dict):
                if tc not in panel and "factual" in panel:
                    normalized[tc] = {**panel, tc: float(panel["factual"])}
                elif tc not in panel and claim_scalar is not None:
                    normalized[tc] = {**panel, tc: claim_scalar}
                if _target_panel_has_claim(normalized, tc):
                    return normalized
    if claim_scalar is not None:
        return {tc: {tc: claim_scalar}}
    raise KeyError(f"no probs_by_dataset panels containing {tc!r}")


def invalidate_decoder_grid_cache(trainer, extra_paths: Optional[Sequence[Any]] = None) -> List[str]:
    """Delete decoder grid caches after a failed eval so later ``use_cache=True`` cannot replay them.

    Study causal currently passes ``use_cache=False``, but the package still writes
    ``decoder_grid_cache.json``. A failed run must not leave that file as a future hit.
    """
    paths: List[Path] = []
    args = getattr(trainer, "args", None) or getattr(trainer, "training_args", None)
    exp = getattr(args, "experiment_dir", None) if args is not None else None
    if exp:
        paths.append(Path(exp) / "decoder_grid_cache.json")
    for raw in extra_paths or ():
        if raw:
            paths.append(Path(raw))
    removed: List[str] = []
    seen: set = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                continue
    if removed:
        print(f"  invalidated decoder cache: {removed}", flush=True)
    return removed


def _trainer_claim_classes(trainer) -> List[str]:
    """Primary trained classes (one-pole claim only — not expanded CF names).

    ``get_target_feature_classes()`` expands one-pole to target+CFs for token
    vocab. Decoder *strengthen* grids only score the claim class, so causal
    must not request CF names like ``DISTRACTOR`` / ``OTHER`` / rival continents.
    """
    # One-pole positive is the authoritative claim when config is present.
    one_pole = getattr(trainer, "_one_pole_positive_class", None)
    if callable(one_pole):
        try:
            pos = one_pole()
        except Exception:
            pos = None
        if pos is not None and str(pos).strip():
            return [str(pos)]

    cfg = getattr(trainer, "config", None)
    tc = getattr(cfg, "target_classes", None) if cfg is not None else None
    if tc is None:
        tc = getattr(trainer, "target_classes", None)
    if not tc:
        # Last resort: bipolar pair only (never expanded get_target_feature_classes).
        pair = getattr(trainer, "pair", None)
        if pair is not None and len(pair) >= 2:
            return [str(pair[0]), str(pair[1])]
        return []
    return [str(c) for c in tc if str(c).strip()]


def evaluate_decoder_for_classes(trainer, classes: Sequence[str], **eval_kw):
    """Run ``evaluate_decoder`` for this trainer's poles only (not full task vocab).

    ``{cls}`` is strengthen; ``{cls}_weaken`` is a different metric and is never
    used as a substitute. One-pole trainers omit CF pole names
    (``DISTRACTOR`` / rival continents): we pin ``target_class`` +
    ``summary_metrics`` to claim classes (from ``_trainer_claim_classes``, the
    single source of truth for what a trainer actually claims) so the package
    cannot expand metrics via ``get_target_feature_classes()``. On failure the
    decoder grid cache is deleted so a later cached run cannot reuse the
    broken grid.

    Deliberately does NOT retry with a narrowed class list on a "requested
    metrics not present" error. An earlier version caught that error, parsed
    which metrics the grid actually scored, and silently retried with those
    instead. Removed: if ``_trainer_claim_classes`` is correct, that retry
    path was provably dead (its own tests showed the pre-filter already
    narrows ``wanted`` before the first call, so the "caller asked for CF
    names" case the retry existed for was already handled upfront). If
    ``_trainer_claim_classes`` is ever wrong for some trainer shape, silently
    recovering downstream hides that bug instead of surfacing it -- exactly
    the same class of problem as the ``probs_by_dataset[X][Y] is absent``
    decoder.py bug this investigation started from. Let a real mismatch
    raise loudly here too.
    """
    wanted = [str(c) for c in classes if str(c).strip()]
    claim = _trainer_claim_classes(trainer)
    if claim:
        claim_set = set(claim)
        filtered = [c for c in wanted if c in claim_set]
        wanted = filtered if filtered else list(claim)
    elif not wanted:
        # Never fall through to expanded get_target_feature_classes().
        raise ValueError(
            "evaluate_decoder_for_classes: empty classes and no claim "
            "target_classes on trainer; refuse expanded CF metric set"
        )

    kw = dict(eval_kw)
    extra_cache = [kw.get("output_path")] if kw.get("output_path") else []
    if "split" not in kw:
        raise ValueError(
            "evaluate_decoder_for_classes requires an explicit split; "
            "use split='validation' for selection and split='test' only for "
            "a frozen validation-selected candidate set"
        )
    if wanted:
        # Pin both: target_class alone is not enough — summary defaults can still
        # pull CF names from get_target_feature_classes on some package paths.
        kw["target_class"] = list(wanted)
        kw["summary_metrics"] = (
            list(wanted)
            if kw.get("increase_target_probabilities", True)
            else [f"{c}_weaken" for c in wanted]
        )
    try:
        return trainer.evaluate_decoder(**kw)
    except Exception:
        invalidate_decoder_grid_cache(trainer, extra_paths=extra_cache)
        raise


def _extract_decoder_summary_and_grid(
    decoder_results: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[Any, Dict[str, Any]]]:
    """Normalize evaluate_decoder / decoder_grid_cache payloads to (summary, grid)."""
    if not decoder_results:
        return {}, {}
    reserved = {
        "grid",
        "plot_path",
        "plot_paths",
        "summary",
        "selection_grid",
        "selection_split",
        "evaluation_split",
        "intervention_kwargs",
        "raw_output_path",
        "results",
        "part",
        "split",
        "max_size_training_like",
        "max_size_neutral",
        "feature_factors",
        "lrs",
    }
    # Raw cache file shape: {results: [entries...], ...} without top-level grid.
    if "grid" not in decoder_results and isinstance(decoder_results.get("results"), list):
        from gradiend.evaluator.decoder_eval_utils import convert_results_to_dict

        grid = convert_results_to_dict(decoder_results["results"])
        summary = (
            decoder_results.get("summary")
            if isinstance(decoder_results.get("summary"), dict)
            else {}
        )
        grid = {
            k: _alias_legacy_factual_metrics(v) if isinstance(v, dict) else v
            for k, v in (grid or {}).items()
        }
        return summary, grid

    summary = (
        decoder_results.get("summary")
        if isinstance(decoder_results.get("summary"), dict)
        else {k: v for k, v in decoder_results.items() if k not in reserved}
    )
    grid = decoder_results.get("grid") or {}
    if isinstance(grid, list):
        from gradiend.evaluator.decoder_eval_utils import convert_results_to_dict

        grid = convert_results_to_dict(grid)
    if isinstance(grid, dict):
        grid = {
            k: _alias_legacy_factual_metrics(v) if isinstance(v, dict) else v
            for k, v in grid.items()
        }
    return summary, grid


def bind_decoder_selection_to_evaluation_grid(
    selection_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine validation selection with a separately evaluated frozen grid.

    ``trainer.evaluate_decoder`` ordinarily returns both a summary-selected LR
    and the grid on which it was selected.  Confirmatory theory experiments
    need validation to choose feature polarity/LR and test only to estimate the
    resulting effect.  This adapter preserves the validation summary while
    replacing only the grid by the test-split grid; downstream
    :func:`run_encoder_causal_for_class` therefore cannot select on test.
    """
    selection_summary, selection_grid = _extract_decoder_summary_and_grid(selection_results)
    _, evaluation_grid = _extract_decoder_summary_and_grid(evaluation_results)
    if not selection_summary:
        raise ValueError("Decoder selection results contain no summary")
    if not evaluation_grid:
        raise ValueError("Decoder evaluation results contain no grid")
    return {
        "summary": selection_summary,
        "grid": evaluation_grid,
        # Kept for split-honest validation curves and selection metadata.
        # Downstream headline scoring still reads only ``grid`` (test).
        "selection_grid": selection_grid,
        "selection_split": "validation",
        "evaluation_split": "test",
    }


def _grid_base_entry(grid: Dict[Any, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if "base" in grid:
        return grid["base"]
    for k, v in grid.items():
        if k == "base" or (isinstance(v, dict) and v.get("id") == "base"):
            return v
    return None


def _candidate_ff_lr(key: Any, entry: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    from gradiend.evaluator.decoder_eval_utils import parse_grid_candidate_id

    return parse_grid_candidate_id(key, entry)




def decoder_selected_learning_rates(
    decoder_results: Dict[str, Any],
    *,
    classes: Optional[Sequence[str]] = None,
) -> List[float]:
    """Return only the LR(s) selected by the validation decoder summary.

    Test is a confirmatory report split.  It scores the selected candidate for
    each requested direction, rather than replaying the whole validation grid.
    A multi-class call can legitimately have one selected LR per class.

    Some package payloads use a synthetic summary key (for example
    ``combined_score``).  If no requested class key is present, use its summary
    LR as a compatibility fallback rather than silently evaluating no test
    candidate.
    """
    summary, _ = _extract_decoder_summary_and_grid(decoder_results)
    requested = {str(cls) for cls in (classes or ())}
    selected: List[float] = []
    fallback: List[float] = []
    for name, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        lr = entry.get("learning_rate")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            continue
        value = float(lr)
        fallback.append(value)
        if (
            not requested
            or str(name) in requested
            or str(name).removesuffix("_weaken") in requested
        ):
            selected.append(value)
    return sorted(set(selected or fallback))


def strength_result_from_decoder_cell(
    *,
    target_class: str,
    strength: float,
    base_entry: Dict[str, Any],
    cell: Dict[str, Any],
    notes: str = "",
    is_random_control: bool = False,
) -> StrengthResult:
    base_pbd = _coerce_probs_by_dataset_for_target(base_entry, target_class)
    cell_pbd = _coerce_probs_by_dataset_for_target(cell, target_class)
    base_lms = _lms_scalar(base_entry.get("lms"))
    lms = _lms_scalar(cell.get("lms"))
    summary = summary_by_group_from_probs(base_pbd, cell_pbd, target_class=target_class)
    sel_abs = strengthen_metric_from_probs(cell_pbd, target_class=target_class)
    base_abs = strengthen_metric_from_probs(base_pbd, target_class=target_class)
    delta = float(sel_abs) - float(base_abs)
    lms_ok = True
    if base_lms is not None and lms is not None:
        lms_ok = lms >= LMS_RATIO * base_lms
    return StrengthResult(
        strength=float(strength),
        lms=lms,
        base_lms=base_lms,
        lms_ok=lms_ok,
        samples=[],
        summary_by_group=summary,
        signed_effect=float(delta),
        selection_metric=float(sel_abs),
        is_random_control=is_random_control,
        notes=notes,
    )


def weaken_headline_from_decoder_results(
    decoder_results: Mapping[str, Any],
    *,
    target_class: str,
) -> Tuple[List[StrengthResult], StrengthResult, StrengthResult, Any]:
    """Read a package ``<class>_weaken`` selection and frozen test result.

    ``bind_decoder_selection_to_evaluation_grid`` keeps the validation summary
    and validation grid alongside the separately evaluated test grid.  This
    helper therefore returns ``(validation_curve, test_selected,
    validation_selected, feature_factor)`` without making any selection on
    test.  The dedicated weaken summary is mandatory: silently falling back to
    the strengthen summary is the protocol bug this function replaces.
    """
    target = str(target_class)
    metric = f"{target}_weaken"
    summary, test_grid = _extract_decoder_summary_and_grid(dict(decoder_results))
    entry = summary.get(metric)
    if not isinstance(entry, Mapping):
        raise KeyError(
            f"decoder weaken summary missing {metric!r}; available={sorted(summary)}"
        )
    if entry.get("feature_factor") is None or entry.get("learning_rate") is None:
        raise KeyError(f"decoder weaken summary {metric!r} lacks feature_factor/LR")

    feature_factor: Any = entry["feature_factor"]
    ff_match = (
        float(feature_factor[0])
        if isinstance(feature_factor, (list, tuple))
        else float(feature_factor)
    )
    package_lr = float(entry["learning_rate"])

    def _curve(grid: Mapping[Any, Any], *, note: str) -> Tuple[List[StrengthResult], Dict[float, StrengthResult]]:
        base_entry = _grid_base_entry(dict(grid))
        if base_entry is None:
            raise ValueError(f"{note} decoder weaken grid missing base entry")
        rows: List[StrengthResult] = []
        by_lr: Dict[float, StrengthResult] = {}
        for key, cell in grid.items():
            if key == "base" or not isinstance(cell, dict):
                continue
            parsed = _candidate_ff_lr(key, cell)
            if parsed is None:
                continue
            ff, lr = parsed
            if abs(float(ff) - ff_match) > 1e-12:
                continue
            raw = strength_result_from_decoder_cell(
                target_class=target,
                strength=float(lr),
                base_entry=base_entry,
                cell=cell,
                notes=f"{note} ff={ff}",
            )
            weakened = as_weaken_strength(raw, target_class=target)
            if weakened is None:
                raise KeyError(f"decoder weaken grid has no factual panel for {target!r}")
            rows.append(weakened)
            by_lr[float(lr)] = weakened
        rows.sort(key=lambda row: -abs(float(row.strength)))
        return rows, by_lr

    test_rows, test_by_lr = _curve(test_grid, note="decoder_weaken_test")
    test_selected = select_decoder_headline(
        test_rows, test_by_lr, package_lr=package_lr
    )

    selection_grid = decoder_results.get("selection_grid")
    if isinstance(selection_grid, Mapping):
        validation_rows, validation_by_lr = _curve(
            selection_grid, note="decoder_weaken_validation"
        )
        validation_selected = select_decoder_headline(
            validation_rows, validation_by_lr, package_lr=package_lr
        )
    else:
        validation_rows = list(test_rows)
        validation_selected = test_selected

    return validation_rows, test_selected, validation_selected, feature_factor


def evaluate_decoder_for_classes_refined(
    trainer,
    classes: Sequence[str],
    *,
    refine_points: int = CAUSAL_STRENGTH_REFINE_POINTS,
    **eval_kw,
) -> Dict[str, Any]:
    """Evaluate the shared causal grid; refinement is disabled by default."""
    return evaluate_decoder_for_classes(
        trainer, classes, refine_points=refine_points, **eval_kw
    )


def run_encoder_causal_for_class(
    trainer,
    meta_rows: Sequence[Dict[str, str]],
    *,
    target_class: str,
    backend: str,
    decoder_results: Optional[Dict[str, Any]] = None,
    strengths: Optional[Sequence[float]] = None,
    lms_texts: Optional[Sequence[str]] = None,
    save_dir: Optional[Path | str] = None,
    include_random_control: bool = True,
    random_seed: int = 0,
    use_cache: bool = False,
    max_size: int = DECODER_EVAL_MAX_SIZE,
    method_id: Optional[str] = None,
    activation_modules: Optional[Any] = None,
    token_selector: Optional[Any] = None,
    activation_gate: Optional[Any] = None,
    lrs: Optional[Sequence[float]] = None,
    training_like_df: Optional[pd.DataFrame] = None,
    neutral_df: Optional[pd.DataFrame] = None,
    decoder_split: Optional[str] = None,
    reuse_strengthen: Optional[CausalMethodResult | Mapping[str, Any]] = None,
    reuse_inverted_bidirectional: Optional[CausalMethodResult | Mapping[str, Any]] = None,
) -> CausalMethodResult:
    """Causal headline from decoder eval grid/summary (same numbers as LR / P(class) plots).

    Does **not** re-score he/she with a last-token probe. Saves the package-selected
    modified model via ``modify_model`` (GRADIEND rewrite / ACTIEND durable hooks).
    ``meta_rows`` / ``lms_texts`` / ``strengths`` are retained for API
    compatibility; strengths come from the decoder grid when present.

    ``training_like_df`` / ``neutral_df`` / ``decoder_split``: the exact frames and
    split the caller's already-provided ``decoder_results`` were computed on. When
    supplied, the norm-matched random-vector control is scored on this *same*
    population. An opposite class pole is a meaningful causal direction and is
    never called a random control.

    For per-layer ACTIEND, pass ``activation_modules`` (e.g. ``transformer.h.11``)
    and ``method_id`` like ``actiend:M:L11_tok_all_gate_encoder_direction``.

    ACTIEND application (two axes — do not conflate):
    - ``token_selector``: where hooks *may* fire (``all`` / ``prediction`` / …).
    - ``activation_gate``: optional encoder gate on top (``encoder_direction`` / …).
    Package default ``token_selector="encoder_direction"`` (omit gate) is the
    **combined** form of ``token_selector="all", activation_gate="encoder_direction"``.
    This study always passes the explicit split so ids match the intervention sweep.

    ``use_cache`` is accepted for API compat but ignored: ``evaluate_decoder`` is always
    called with ``use_cache=False`` (none-split ACTIEND ablations share one
    ``decoder_grid_cache.json``; see PLANNING.md §8.6). Prefer passing ``decoder_results``.
    """
    del strengths  # selection/grid from decoder; strengths unused
    del use_cache  # always False at evaluate_decoder (shared-path overwrite hazard)
    mid = method_id or f"{backend}:{target_class}"
    eval_kw: Dict[str, Any] = {
        "max_size": max_size,
        "plot": False,
        "use_cache": False,
        "split": "validation",
    }
    if activation_modules is not None:
        eval_kw["activation_modules"] = activation_modules
    if token_selector is not None:
        eval_kw["token_selector"] = token_selector
    if activation_gate is not None:
        eval_kw["activation_gate"] = activation_gate
    if lrs is not None:
        eval_kw["lrs"] = [float(x) for x in lrs]

    reused = (
        causal_method_result_from_dict(reuse_strengthen)
        if isinstance(reuse_strengthen, Mapping)
        else reuse_strengthen
    )
    polarity_repair = None
    if reuse_inverted_bidirectional is not None:
        if reused is not None:
            raise ValueError(
                "reuse_strengthen and reuse_inverted_bidirectional are mutually exclusive"
            )
        polarity_repair = reselect_inverted_bidirectional_curves(
            reuse_inverted_bidirectional,
            target_class=str(target_class),
        )
        (
            repaired_strengthen_curve,
            repaired_strengthen_selected,
            repaired_weaken_curve,
            repaired_weaken_selected,
        ) = polarity_repair
        # The complete +/- validation grids are already persisted. Evaluate
        # only each reselected frozen candidate on the requested report split.
        report_kw = dict(eval_kw)
        report_kw["split"] = decoder_split or "test"
        report_kw["lrs"] = [float(repaired_strengthen_selected.strength)]
        if training_like_df is not None:
            report_kw["training_like_df"] = training_like_df
        if neutral_df is not None:
            report_kw["neutral_df"] = neutral_df
        decoder_results = evaluate_decoder_for_classes_refined(
            trainer,
            [str(target_class)],
            refine_points=0,
            **report_kw,
        )
        weaken_kw = dict(report_kw)
        weaken_kw["lrs"] = [float(repaired_weaken_selected.strength)]
        weaken_kw["increase_target_probabilities"] = False
        weaken_report = evaluate_decoder_for_classes_refined(
            trainer,
            [str(target_class)],
            refine_points=0,
            **weaken_kw,
        )
        decoder_results = dict(decoder_results or {})
        decoder_results["weaken_results"] = weaken_report

    # Caller must supply a decoder grid built under the same token_selector /
    # activation_gate (or leave decoder_results=None to evaluate here).
    if decoder_results is None and reused is None:
        decoder_results = evaluate_decoder_for_classes_refined(
            trainer, [str(target_class)], **eval_kw
        )

    weaken_decoder_results = (decoder_results or {}).get("weaken_results")
    if not isinstance(weaken_decoder_results, Mapping):
        maybe_summary, _ = _extract_decoder_summary_and_grid(decoder_results or {})
        if f"{target_class}_weaken" in maybe_summary:
            weaken_decoder_results = decoder_results
        else:
            weaken_kw = dict(eval_kw)
            weaken_kw["increase_target_probabilities"] = False
            weaken_decoder_results = evaluate_decoder_for_classes_refined(
                trainer, [str(target_class)], **weaken_kw
            )

    summary, grid = _extract_decoder_summary_and_grid(decoder_results or {})
    entry = summary.get(str(target_class))
    if not entry or not isinstance(entry, dict) or "feature_factor" not in entry:
        for cand in (str(target_class), "combined_score", *list(summary.keys())):
            e = summary.get(cand)
            if isinstance(e, dict) and "feature_factor" in e:
                entry = e
                break
    if reused is None and (not entry or not grid) and decoder_results is not None:
        # Retry evaluate_decoder once (cache may have been raw results-only).
        # Always pin target_class — bare evaluate_decoder expands one-pole CFs.
        try:
            decoder_results = evaluate_decoder_for_classes_refined(
                trainer, [str(target_class)], **eval_kw
            )
            summary, grid = _extract_decoder_summary_and_grid(decoder_results)
            entry = summary.get(str(target_class))
            if not entry or not isinstance(entry, dict) or "feature_factor" not in entry:
                for cand in (str(target_class), "combined_score", *list(summary.keys())):
                    e = summary.get(cand)
                    if isinstance(e, dict) and "feature_factor" in e:
                        entry = e
                        break
        except Exception as exc:
            if not grid:
                return CausalMethodResult(
                    method=mid,
                    backend=backend,
                    target_class=str(target_class),
                    strengths=[],
                    selected_strength=None,
                    selected=None,
                    meta={"error": f"no decoder summary/grid for {target_class}: {exc}"},
                )

    selection_curve_results: Optional[List[StrengthResult]] = None
    validation_selected: Optional[StrengthResult] = None

    if reused is not None:
        if reused.selected is None or not reused.strengths:
            raise ValueError(
                f"cannot reuse incomplete strengthen result for {mid}: "
                "selected/strengths missing"
            )
        # Component-level causal repair: these are the already-selected v5
        # validation curve and its frozen test report.  They are copied as-is;
        # no strengthen decoder evaluation or model scoring is performed.
        results = list(reused.strengths)
        selection_curve_results = list(reused.strengths)
        selected = reused.selected
        validation_selected = None
        feature_factor = reused.meta.get("feature_factor")
        if feature_factor is None:
            feature_factor = (
                (reused.meta.get("package_decoder_entry") or {}).get("feature_factor")
            )
        if feature_factor is None:
            raise ValueError(f"reused strengthen result for {mid} lacks feature_factor")
        # The genuine random control needs the unmodified test baseline.  It is
        # already present in the frozen strengthen headline's group summary, so
        # reconstruct only the package cell shape; do not evaluate base/strengthen.
        base_pbd = {
            str(group): {str(target_class): float(values["mean_p_target_base"])}
            for group, values in selected.summary_by_group.items()
            if isinstance(values, Mapping) and values.get("mean_p_target_base") is not None
        }
        if not base_pbd:
            raise ValueError(f"reused strengthen result for {mid} lacks test baseline probabilities")
        base_entry = {"probs_by_dataset": base_pbd, "lms": selected.base_lms}
        entry = dict(reused.meta.get("package_decoder_entry") or {})
        entry.setdefault("feature_factor", feature_factor)
        entry.setdefault("learning_rate", selected.strength)
        _synth = False

    # Grid-only path: synthesize LMS-gated selection when summary lacks class entry.
    elif grid and (not entry or not isinstance(entry, dict) or "feature_factor" not in entry):
        base_entry = _grid_base_entry(grid)
        if base_entry is None:
            return CausalMethodResult(
                method=mid,
                backend=backend,
                target_class=str(target_class),
                strengths=[],
                selected_strength=None,
                selected=None,
                meta={"error": "decoder grid missing base entry"},
            )
        candidates: List[StrengthResult] = []
        by_lr_ff: Dict[Tuple[float, float], StrengthResult] = {}
        for key, cell in grid.items():
            if key == "base":
                continue
            parsed = _candidate_ff_lr(key, cell)
            if parsed is None:
                continue
            ff, lr = parsed
            sr = strength_result_from_decoder_cell(
                target_class=str(target_class),
                strength=float(lr),
                base_entry=base_entry,
                cell=cell,
                notes=f"decoder_grid_synth ff={ff}",
            )
            candidates.append(sr)
            by_lr_ff[(float(ff), float(lr))] = sr
        selected = select_lms_gated(candidates, prefer_signed_effect=False)
        if selected is None:
            return CausalMethodResult(
                method=mid,
                backend=backend,
                target_class=str(target_class),
                strengths=candidates,
                selected_strength=None,
                selected=None,
                meta={"error": f"no LMS-gated candidate for {target_class}", "source": "grid_synth"},
            )
        # Recover feature_factor from matching cell notes / by searching.
        # Preserve the grid's actual ff value (never snap to +-1.0): a
        # non-unit ff (e.g. an encoder-mean-calibrated sweep) has a real,
        # distinct value here, and canonicalizing it would silently break
        # the `ff_match` filter below the moment any caller ever builds a
        # grid with a non-unit feature_factor (found while designing
        # scripts/ablation_actiend_calibrated_ff.py; that ablation's design
        # sidesteps this path entirely, but the bug is real regardless --
        # see CLAUDE.md's 2026-08-27 entry).
        feature_factor = 1.0
        for (ff, lr), sr in by_lr_ff.items():
            if abs(sr.strength - selected.strength) < 1e-12 and abs(
                sr.signed_effect - selected.signed_effect
            ) < 1e-12:
                feature_factor = float(ff)
                selected = sr
                break
        ff_match = float(feature_factor)
        results = [
            sr for (ff, _lr), sr in by_lr_ff.items() if abs(ff - ff_match) < 1e-12
        ]
        results.sort(key=lambda r: -abs(r.strength))
        by_lr = {float(r.strength): r for r in results}
        selected = by_lr.get(float(selected.strength)) or selected
        entry = {
            "feature_factor": feature_factor,
            "learning_rate": selected.strength,
            "lms": selected.lms,
            "base_lms": selected.base_lms,
        }
        selection_curve_results = list(results)
        validation_selected = selected
        _synth = True
    else:
        _synth = False

    if reused is None and not _synth:
        if not entry or not grid:
            return CausalMethodResult(
                method=mid,
                backend=backend,
                target_class=str(target_class),
                strengths=[],
                selected_strength=None,
                selected=None,
                meta={"error": f"no decoder summary/grid for {target_class}"},
            )

        # Preserve the summary's actual ff value(s) -- see the matching
        # comment in the grid-synth branch above for why snapping to +-1.0
        # is a real bug, not a style choice: it silently breaks the
        # `ff_match` filter just below for any non-unit feature_factor.
        feature_factor = entry["feature_factor"]
        if isinstance(feature_factor, (list, tuple)):
            feature_factor = [float(x) for x in feature_factor]
            ff_match = float(feature_factor[0]) if feature_factor else 1.0
        else:
            feature_factor = float(feature_factor)
            ff_match = feature_factor

        base_entry = _grid_base_entry(grid)
        if base_entry is None:
            return CausalMethodResult(
                method=mid,
                backend=backend,
                target_class=str(target_class),
                strengths=[],
                selected_strength=None,
                selected=None,
                meta={"error": "decoder grid missing base entry"},
            )

        results = []
        by_lr = {}
        for key, cell in grid.items():
            if key == "base":
                continue
            parsed = _candidate_ff_lr(key, cell)
            if parsed is None:
                continue
            ff, lr = parsed
            if abs(float(ff) - ff_match) > 1e-12:
                continue
            sr = strength_result_from_decoder_cell(
                target_class=str(target_class),
                strength=float(lr),
                base_entry=base_entry,
                cell=cell,
                notes=f"decoder_cache ff={ff}",
            )
            results.append(sr)
            by_lr[float(lr)] = sr

        results.sort(key=lambda r: -abs(r.strength))

        # Headline = decoder-plot star (package summary LR). Do not re-select.
        pkg_lr = entry["learning_rate"]
        selected = select_decoder_headline(results, by_lr, package_lr=pkg_lr)

        selection_grid = (decoder_results or {}).get("selection_grid")
        if isinstance(selection_grid, dict):
            selection_base_entry = _grid_base_entry(selection_grid)
            if selection_base_entry is None:
                raise ValueError("validation selection grid missing base entry")
            selection_curve_results = []
            selection_by_lr: Dict[float, StrengthResult] = {}
            for key, cell in selection_grid.items():
                if key == "base" or not isinstance(cell, dict):
                    continue
                parsed = _candidate_ff_lr(key, cell)
                if parsed is None:
                    continue
                ff, lr = parsed
                if abs(float(ff) - ff_match) > 1e-12:
                    continue
                sr = strength_result_from_decoder_cell(
                    target_class=str(target_class),
                    strength=float(lr),
                    base_entry=selection_base_entry,
                    cell=cell,
                    notes=f"decoder_validation_cache ff={ff}",
                )
                selection_curve_results.append(sr)
                selection_by_lr[float(lr)] = sr
            selection_curve_results.sort(key=lambda r: -abs(r.strength))
            validation_selected = select_decoder_headline(
                selection_curve_results, selection_by_lr, package_lr=pkg_lr
            )
        else:
            selection_curve_results = list(results)
            validation_selected = selected
        entry = dict(entry)
        entry["package_learning_rate"] = pkg_lr
        entry["study_learning_rate"] = selected.strength

    # Coarse-to-fine strength refinement already happened inside
    # ``evaluate_decoder_for_classes_refined`` above (or was supplied
    # pre-refined via the ``decoder_results`` argument) — ``entry``/``results``/
    # ``by_lr``/``selected`` here already reflect the refined grid, one call.

    (
        weaken_curve_results,
        weaken_selected,
        weaken_validation_selected,
        weaken_feature_factor,
    ) = weaken_headline_from_decoder_results(
        weaken_decoder_results,
        target_class=str(target_class),
    )
    if polarity_repair is not None:
        selection_curve_results = repaired_strengthen_curve
        validation_selected = repaired_strengthen_selected
        weaken_curve_results = repaired_weaken_curve
        weaken_validation_selected = repaired_weaken_selected

    # Persist selected modified model only. Never re-score / overwrite the
    # decoder-grid signed_effect (that previously zeroed one-pole headlines via
    # he/she probe + empty summary_by_group).
    path = None
    save_error = None
    actiend_mod_kw: Dict[str, Any] = {}
    if activation_modules is not None:
        actiend_mod_kw["activation_modules"] = activation_modules
    if token_selector is not None:
        actiend_mod_kw["token_selector"] = token_selector
    if activation_gate is not None:
        actiend_mod_kw["activation_gate"] = activation_gate
    if selected is not None and save_dir is not None:
        slug = mid.replace(":", "_")
        out = Path(save_dir) / f"{slug}_lr{selected.strength:g}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            model_with = trainer.get_model()
            tokenizer = trainer.tokenizer
            modifier = getattr(model_with, "modify_model", None) or model_with.rewrite_base_model
            modified = modifier(
                learning_rate=float(selected.strength),
                feature_factor=feature_factor,
                **actiend_mod_kw,
            )
            save_modified = getattr(modified, "save_pretrained_modified", None)
            if save_modified is not None:
                save_modified(str(out))
            else:
                modified.save_pretrained(str(out))
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(str(out))
            path = str(out)
            del modified
        except Exception as exc:
            save_error = str(exc)

    random_ctrl = None
    if include_random_control and selected is not None:
        try:
            model_with = trainer.get_model()
            tokenizer = trainer.tokenizer
            real_update = model_with.intervention_update_vector(
                value=float(selected.strength),
                feature_factor=feature_factor,
            )
            generator = torch.Generator(device=real_update.device)
            generator.manual_seed(int(random_seed))
            random_update = torch.randn(
                real_update.shape,
                dtype=real_update.dtype,
                device=real_update.device,
                generator=generator,
            )
            random_update = random_update * (
                real_update.norm() / (random_update.norm() + 1e-12)
            )
            control_df = (
                training_like_df
                if training_like_df is not None
                else meta_rows_to_frame(meta_rows)
            )
            with model_with.intervene(
                value=1.0,
                feature_factor=feature_factor,
                update_vector=random_update,
                **actiend_mod_kw,
            ):
                control_cell = trainer.evaluate_base_model(
                    model_with.base_model,
                    tokenizer,
                    use_cache=False,
                    training_like_df=control_df,
                    neutral_df=neutral_df,
                    max_size_training_like=max_size,
                    max_size_neutral=max_size,
                )
            random_ctrl = strength_result_from_decoder_cell(
                target_class=str(target_class),
                strength=float(selected.strength),
                base_entry=base_entry,
                cell=control_cell,
                notes=(
                    "genuine random update matched to selected intervention L2; "
                    f"seed={random_seed} split={decoder_split or 'eval'}"
                ),
                is_random_control=True,
            )
        except Exception as exc:
            raise RuntimeError(f"matched-random causal control failed for {mid}") from exc

    return CausalMethodResult(
        method=mid,
        backend=backend,
        target_class=str(target_class),
        strengths=(
            selection_curve_results
            if selection_curve_results is not None
            else results
        ),
        selected_strength=None if selected is None else selected.strength,
        selected=selected,
        modified_model_path=path,
        random_control=random_ctrl,
        weaken_strengths=weaken_curve_results,
        weaken_selected_strength=weaken_selected.strength,
        weaken_selected=weaken_selected,
        meta={
            **(dict(reused.meta) if reused is not None else {}),
            "feature_factor": feature_factor,
            "package_learning_rate": (entry or {}).get("package_learning_rate", (entry or {}).get("learning_rate")),
            "study_learning_rate": None if selected is None else selected.strength,
            "source": "decoder_grid_cache",
            "metric": "P(target)_strengthen_panel",
            "weaken_metric": "base_P(target)-modified_P(target)_own_panel",
            "weaken_feature_factor": weaken_feature_factor,
            "save_error": save_error,
            "activation_modules": activation_modules,
            "token_selector": token_selector,
            "activation_gate": activation_gate,
            "method_id": mid,
            "random_control_kind": (
                "matched_l2_random_vector" if include_random_control else None
            ),
            "random_control_seed": int(random_seed) if include_random_control else None,
            "selection_split": (
                "validation"
                if polarity_repair is not None
                else reused.meta.get("selection_split", "validation")
                if reused is not None
                else (decoder_results or {}).get("selection_split", "eval")
            ),
            "report_split": (
                (decoder_split or "test")
                if polarity_repair is not None
                else reused.meta.get("report_split", "test")
                if reused is not None
                else (decoder_results or {}).get("evaluation_split", "eval")
            ),
            "strengthen_reused": reused is not None,
            "polarity_reselected_from_bidirectional_grid": polarity_repair is not None,
            **(
                {}
                if reused is not None
                else selection_lms_metadata(validation_selected)
            ),
            **{
                f"weaken_{k}": v
                for k, v in selection_lms_metadata(
                    weaken_validation_selected
                ).items()
            },
            "package_decoder_entry": {
                k: (entry or {}).get(k)
                for k in (
                    "feature_factor",
                    "learning_rate",
                    "lms",
                    "base_lms",
                    "value",
                    "id",
                    "strengthen",
                )
                if entry and k in entry
            },
        },
    )


# ---------------------------------------------------------------------------
# SAE causal sweep (package scoring + durable hooks + package LMS)
# ---------------------------------------------------------------------------

def run_strength_on_model(
    model,
    tokenizer,
    meta_rows: Sequence[Dict[str, str]],
    *,
    target_class: str,
    strength: float,
    base_probs: Dict[str, Dict[str, float]],
    base_lms: Optional[float],
    lms_texts: Sequence[str],
    is_random_control: bool = False,
    notes: str = "",
) -> StrengthResult:
    """Score steered model with package mask-slot P(class) + package LMS."""
    frame = meta_rows_to_frame(meta_rows)
    mod_probs = score_class_probs_by_dataset(model, tokenizer, frame)
    summary = summary_by_group_from_probs(base_probs, mod_probs, target_class=target_class)
    signed = signed_effect_from_group_summary(summary, target_class=str(target_class))
    sel_abs = strengthen_metric_from_probs(mod_probs, target_class=str(target_class))
    lms = compute_lms_safe(model, tokenizer, lms_texts)
    lms_ok = True
    if base_lms is not None and lms is not None:
        lms_ok = lms >= LMS_RATIO * base_lms
    return StrengthResult(
        strength=float(strength),
        lms=lms,
        base_lms=base_lms,
        lms_ok=lms_ok,
        summary_by_group=summary,
        signed_effect=float(signed),
        selection_metric=float(sel_abs),
        is_random_control=is_random_control,
        notes=notes,
    )


def run_sae_causal_sweep(
    model,
    tokenizer,
    meta_rows: Sequence[Dict[str, str]],
    *,
    sae,
    module_path: str,
    feature_index: Optional[int] = None,
    feature_indices: Optional[Sequence[int]] = None,
    target_class: str,
    strengths: Sequence[float] = DEFAULT_CAUSAL_STRENGTHS,
    lms_texts: Optional[Sequence[str]] = None,
    include_random_control: bool = True,
    random_seed: int = 0,
    save_dir: Optional[Path | str] = None,
    method_id: Optional[str] = None,
    token_selector: str = "all",
    bag_reduce: str = "sum",
    report_rows: Optional[Sequence[Dict[str, str]]] = None,
    base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    base_lms: Optional[Any] = None,
    r_base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    r_base_lms: Optional[Any] = None,
    reuse_strengthen: Optional[CausalMethodResult | Mapping[str, Any]] = None,
) -> CausalMethodResult:
    """LMS-gated SAE strength sweep (single feature or k* bag).

    Pass ``feature_index`` or ``feature_indices``. Both polarities of the
    (bag) direction are swept; LMS-gate maximizes toward-target on other data.
    Default ``token_selector='all'`` (standard SAE residual steering).

    ``base_probs``/``base_lms`` (and ``r_base_probs``/``r_base_lms`` for the
    ``report_rows`` split) are the *unsteered* model's scores on ``meta_rows``/
    ``report_rows`` — identical for every feature/layer/class swept against the
    same rows, so callers sweeping many methods over one fixed ``meta_rows``/
    ``report_rows`` pair should compute them once and pass them in here instead
    of paying a fresh unsteered forward pass per call.
    """
    if feature_indices is not None and len(list(feature_indices)) > 0:
        idxs = [int(i) for i in feature_indices]
    elif feature_index is not None:
        idxs = [int(feature_index)]
    else:
        raise ValueError("run_sae_causal_sweep requires feature_index or feature_indices")

    texts = [r["text"] for r in meta_rows]
    lms_texts = list(
        lms_texts or [r["text"] for r in meta_rows if r["group"] == "neutral"] or texts
    )
    if base_probs is None or base_lms is None:
        frame = meta_rows_to_frame(meta_rows)
        base_probs = score_class_probs_by_dataset(model, tokenizer, frame)
        base_lms = compute_lms_safe(model, tokenizer, lms_texts)
    if len(idxs) == 1:
        direction = sae_decoder_direction(sae, idxs[0])
    else:
        direction = sae_decoder_bag_direction(sae, idxs, reduce=bag_reduce)

    reused = (
        causal_method_result_from_dict(reuse_strengthen)
        if isinstance(reuse_strengthen, Mapping)
        else reuse_strengthen
    )
    results: List[StrengthResult] = []
    feat_tag = ",".join(str(i) for i in idxs[:8]) + ("..." if len(idxs) > 8 else "")
    if reused is None:
        for sign in (1.0, -1.0):
            steered = direction * float(sign)
            for s in strengths:
                with sae_steering_context(
                    model, module_path, steered, float(s), token_selector=token_selector
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
                                f"sae_feats=[{feat_tag}] n={len(idxs)} sign={sign:g} "
                                f"token_selector={token_selector}"
                            ),
                        )
                    )
        selected = None
        validation_selected = None
    else:
        if reused.selected is None or not reused.strengths:
            raise ValueError("cannot reuse incomplete SAE strengthen result")
        results = list(reused.strengths)
        selected = reused.selected
        validation_selected = None

    def _run_one(sign: float, s: float) -> StrengthResult:
        steered = direction * float(sign)
        with sae_steering_context(
            model, module_path, steered, float(s), token_selector=token_selector
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
                    f"sae_feats=[{feat_tag}] n={len(idxs)} sign={sign:g} "
                    f"token_selector={token_selector} refine=1"
                ),
            )

    if reused is None:
        results, selected = refine_and_select_lms_gated(results, _run_one)
        validation_selected = selected

    weaken_results, weaken_validation_selected = refine_and_select_weaken_lms_gated(
        results, _run_one, target_class=target_class
    )
    if selected is None:
        # Nothing passed the LMS gate. steered_dir is never actually used
        # below (its only real consumer is gated on `selected is not None`)
        # -- give it a safe, unscaled placeholder rather than guessing a
        # sign. meta["sign"] correctly reports None (not a fake 1.0) so a
        # "nothing selected" run doesn't look like it selected +1.
        selected_sign = None
        steered_dir = direction
    else:
        selected_sign = _parse_sweep_sign(selected.notes)
        if selected_sign is None:
            raise ValueError(
                f"selected StrengthResult is missing a sign= tag in notes "
                f"(notes={selected.notes!r}); cannot determine steering "
                "direction. Every result in this sweep must be tagged."
            )
        steered_dir = direction * float(selected_sign)
    random_ctrl = None
    if include_random_control and selected is not None and not report_rows:
        rng = np.random.default_rng(random_seed + int(idxs[0]))
        rand = torch.tensor(rng.normal(size=direction.numel()), dtype=torch.float32)
        rand = rand * (direction.norm() / (rand.norm() + 1e-8))
        with sae_steering_context(
            model, module_path, rand, float(selected.strength), token_selector=token_selector
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
                notes="matched-L2 random direction",
            )

    if report_rows and (selected is not None or weaken_validation_selected is not None):
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        if r_base_probs is None or r_base_lms is None:
            r_frame = meta_rows_to_frame(report_rows)
            r_base_probs = score_class_probs_by_dataset(model, tokenizer, r_frame)
            r_base_lms = compute_lms_safe(model, tokenizer, r_lms)
    if report_rows and selected is not None and reused is None:
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        with sae_steering_context(
            model, module_path, steered_dir, float(selected.strength), token_selector=token_selector
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
            rng = np.random.default_rng(random_seed + int(idxs[0]))
            rand = torch.tensor(rng.normal(size=direction.numel()), dtype=torch.float32)
            rand = rand * (direction.norm() / (rand.norm() + 1e-8))
            with sae_steering_context(
                model, module_path, rand, float(selected.strength), token_selector=token_selector
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
                    notes="matched-L2 random direction report=test",
                )
    elif report_rows and selected is not None and include_random_control:
        rng = np.random.default_rng(random_seed + int(idxs[0]))
        rand = torch.tensor(rng.normal(size=direction.numel()), dtype=torch.float32)
        rand = rand * (direction.norm() / (rand.norm() + 1e-8))
        with sae_steering_context(
            model, module_path, rand, float(selected.strength), token_selector=token_selector
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
                notes="matched-L2 random direction report=test repair=1",
            )

    weaken_selected = weaken_validation_selected
    if report_rows and weaken_validation_selected is not None:
        weaken_sign = _parse_sweep_sign(weaken_validation_selected.notes)
        if weaken_sign is None:
            raise ValueError("weaken selection is missing its sign tag")
        with sae_steering_context(
            model,
            module_path,
            direction * float(weaken_sign),
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

    path = None
    if save_dir is not None and selected is not None:
        mid_slug = (method_id or f"sae_{target_class}").replace(":", "_")
        path = save_sae_modified_model(
            Path(save_dir) / f"{mid_slug}_str{selected.strength:g}",
            base_model=model,
            tokenizer=tokenizer,
            module_path=module_path,
            feature_index=idxs[0],
            strength=selected.strength,
            direction=steered_dir,
            meta={
                "target_class": target_class,
                "lms_ok": selected.lms_ok,
                "sign": selected_sign,
                "feature_indices": idxs,
            },
            token_selector=token_selector,
        )

    mid = method_id or f"sae:{target_class}"
    return CausalMethodResult(
        method=mid,
        backend="sae",
        target_class=str(target_class),
        strengths=results,
        selected_strength=None if selected is None else selected.strength,
        selected=selected,
        modified_model_path=path,
        random_control=random_ctrl,
        weaken_strengths=weaken_results,
        weaken_selected_strength=(
            None if weaken_selected is None else weaken_selected.strength
        ),
        weaken_selected=weaken_selected,
        meta={
            **(dict(reused.meta) if reused is not None else {}),
            "feature_index": idxs[0],
            "feature_indices": idxs,
            "n_features": len(idxs),
            "module_path": module_path,
            "token_selector": token_selector,
            "sign": selected_sign,
            "bag_reduce": bag_reduce if len(idxs) > 1 else None,
            "source": "package_decoder_protocol",
            "metric": "delta_P(target)_on_other_dataset",
            "selection_split": "validation" if report_rows else "eval",
            "report_split": "test" if report_rows else "eval",
            "strengthen_reused": reused is not None,
            **({} if reused is not None else selection_lms_metadata(validation_selected)),
            **{
                f"weaken_{k}": v
                for k, v in selection_lms_metadata(
                    weaken_validation_selected
                ).items()
            },
        },
    )


def sae_feature_max_activation(
    model,
    tokenizer,
    sae,
    module_path: str,
    feature_index: int,
    texts: Sequence[str],
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> float:
    """Max observed activation of one SAE feature over ``texts`` at ``module_path``.

    Used only to set the clamp-target grid's scale for
    ``run_sae_clamp_causal_sweep`` (a normalization heuristic, not a scored
    metric) — mirrors ``axbench_ext.sae._StudySAEBase._token_latents``'s
    residual-capture-then-encode pattern as a standalone helper so this
    module doesn't need to import the AxBench adapter.
    """
    modules = dict(model.named_modules())
    if module_path not in modules:
        raise KeyError(f"SAE clamp module {module_path!r} not found in model")
    device = next(model.parameters()).device
    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["value"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = modules[module_path].register_forward_hook(_hook)
    max_act = 0.0
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                if not batch:
                    continue
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model(**encoded)
                residual = captured.get("value")
                if residual is None:
                    continue
                attention_mask = encoded.get("attention_mask")
                latents = sae.encode(residual.to(next(sae.parameters()).device))[..., feature_index]
                if attention_mask is not None:
                    keep = attention_mask.to(device=latents.device, dtype=torch.bool)
                    latents = latents[keep]
                if latents.numel():
                    max_act = max(max_act, float(latents.max().item()))
    finally:
        handle.remove()
    return max_act


def run_sae_clamp_causal_sweep(
    model,
    tokenizer,
    meta_rows: Sequence[Dict[str, str]],
    *,
    sae,
    module_path: str,
    feature_index: int,
    target_class: str,
    target_multipliers: Sequence[float] = DEFAULT_CLAMP_TARGET_MULTIPLIERS,
    lms_texts: Optional[Sequence[str]] = None,
    include_random_control: bool = True,
    random_seed: int = 0,
    save_dir: Optional[Path | str] = None,
    method_id: Optional[str] = None,
    token_selector: str = "all",
    report_rows: Optional[Sequence[Dict[str, str]]] = None,
    base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    base_lms: Optional[Any] = None,
    r_base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    r_base_lms: Optional[Any] = None,
    reuse_strengthen: Optional[CausalMethodResult | Mapping[str, Any]] = None,
) -> CausalMethodResult:
    """LMS-gated SAE activation-clamp sweep (single feature; Templeton et al. 2024 ablation).

    Parallel to ``run_sae_causal_sweep`` but forces the feature's own
    activation to a target value instead of adding a fixed vector at every
    position (see ``make_sae_clamp_intervention``). K1-only by design — a
    bag/``kstar`` clamp target would need a per-feature scheme, out of scope
    for this ablation. ``base_probs``/``base_lms``/``r_base_probs``/
    ``r_base_lms`` semantics match ``run_sae_causal_sweep``.
    """
    feature_index = int(feature_index)
    texts = [r["text"] for r in meta_rows]
    lms_texts = list(
        lms_texts or [r["text"] for r in meta_rows if r["group"] == "neutral"] or texts
    )
    if base_probs is None or base_lms is None:
        frame = meta_rows_to_frame(meta_rows)
        base_probs = score_class_probs_by_dataset(model, tokenizer, frame)
        base_lms = compute_lms_safe(model, tokenizer, lms_texts)

    reused = (
        causal_method_result_from_dict(reuse_strengthen)
        if isinstance(reuse_strengthen, Mapping)
        else reuse_strengthen
    )
    target_texts = [r["text"] for r in meta_rows if r["group"] == str(target_class)] or texts
    reference = (
        float(reused.meta["clamp_reference_activation"])
        if reused is not None and reused.meta.get("clamp_reference_activation") is not None
        else sae_feature_max_activation(
            model, tokenizer, sae, module_path, feature_index, target_texts
        )
    )

    strengths_grid = sorted({float(m) * reference for m in target_multipliers})

    def _score_at(target: float, *, refine: bool) -> StrengthResult:
        with sae_clamp_steering_context(
            model, module_path, sae, feature_index, target, token_selector=token_selector
        ):
            return run_strength_on_model(
                model,
                tokenizer,
                meta_rows,
                target_class=target_class,
                strength=float(target),
                base_probs=base_probs,
                base_lms=base_lms,
                lms_texts=lms_texts,
                notes=(
                    f"sae_clamp_feat={feature_index} target={target:g} "
                    f"ref={reference:g} token_selector={token_selector}"
                    + (" refine=1" if refine else "")
                ),
            )

    if reused is None:
        results = [_score_at(target, refine=False) for target in strengths_grid]
        results, selected = refine_and_select_lms_gated(
            results, lambda _sign, target: _score_at(target, refine=True)
        )
        validation_selected = selected
    else:
        if reused.selected is None or not reused.strengths:
            raise ValueError("cannot reuse incomplete SAE clamp strengthen result")
        results = list(reused.strengths)
        selected = reused.selected
        validation_selected = None
    weaken_results, weaken_validation_selected = refine_and_select_weaken_lms_gated(
        results,
        lambda _sign, target: _score_at(target, refine=True),
        target_class=target_class,
    )

    random_ctrl = None
    if include_random_control and selected is not None and not report_rows:
        rng = np.random.default_rng(random_seed + feature_index)
        real_direction = sae_decoder_direction(sae, feature_index)
        rand = torch.tensor(rng.normal(size=real_direction.numel()), dtype=torch.float32)
        rand = rand * (real_direction.norm() / (rand.norm() + 1e-8))
        with sae_clamp_steering_context(
            model,
            module_path,
            sae,
            feature_index,
            float(selected.strength),
            token_selector=token_selector,
            direction=rand,
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
                notes="matched-L2 random direction (clamp)",
            )

    if report_rows and (selected is not None or weaken_validation_selected is not None):
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        if r_base_probs is None or r_base_lms is None:
            r_frame = meta_rows_to_frame(report_rows)
            r_base_probs = score_class_probs_by_dataset(model, tokenizer, r_frame)
            r_base_lms = compute_lms_safe(model, tokenizer, r_lms)
    if report_rows and selected is not None and reused is None:
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        with sae_clamp_steering_context(
            model, module_path, sae, feature_index, float(selected.strength), token_selector=token_selector
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
            rng = np.random.default_rng(random_seed + feature_index)
            real_direction = sae_decoder_direction(sae, feature_index)
            rand = torch.tensor(rng.normal(size=real_direction.numel()), dtype=torch.float32)
            rand = rand * (real_direction.norm() / (rand.norm() + 1e-8))
            with sae_clamp_steering_context(
                model,
                module_path,
                sae,
                feature_index,
                float(selected.strength),
                token_selector=token_selector,
                direction=rand,
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
                    notes="matched-L2 random direction (clamp) report=test",
                )
    elif report_rows and selected is not None and include_random_control:
        rng = np.random.default_rng(random_seed + feature_index)
        real_direction = sae_decoder_direction(sae, feature_index)
        rand = torch.tensor(rng.normal(size=real_direction.numel()), dtype=torch.float32)
        rand = rand * (real_direction.norm() / (rand.norm() + 1e-8))
        with sae_clamp_steering_context(
            model,
            module_path,
            sae,
            feature_index,
            float(selected.strength),
            token_selector=token_selector,
            direction=rand,
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
                notes="matched-L2 random direction (clamp) report=test repair=1",
            )

    weaken_selected = weaken_validation_selected
    if report_rows and weaken_validation_selected is not None:
        with sae_clamp_steering_context(
            model,
            module_path,
            sae,
            feature_index,
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

    path = None
    if save_dir is not None and selected is not None:
        mid_slug = (method_id or f"sae_clamp_{target_class}").replace(":", "_")
        path = save_sae_modified_model(
            Path(save_dir) / f"{mid_slug}_tgt{selected.strength:g}",
            base_model=model,
            tokenizer=tokenizer,
            module_path=module_path,
            feature_index=feature_index,
            strength=selected.strength,
            direction=sae_decoder_direction(sae, feature_index),
            meta={
                "target_class": target_class,
                "lms_ok": selected.lms_ok,
                "mode": "clamp",
                "feature_index": feature_index,
            },
            token_selector=token_selector,
        )

    mid = method_id or f"sae:{target_class}:clamp"
    return CausalMethodResult(
        method=mid,
        backend="sae",
        target_class=str(target_class),
        strengths=results,
        selected_strength=None if selected is None else selected.strength,
        selected=selected,
        modified_model_path=path,
        random_control=random_ctrl,
        weaken_strengths=weaken_results,
        weaken_selected_strength=(
            None if weaken_selected is None else weaken_selected.strength
        ),
        weaken_selected=weaken_selected,
        meta={
            **(dict(reused.meta) if reused is not None else {}),
            "feature_index": feature_index,
            "module_path": module_path,
            "token_selector": token_selector,
            "mode": "clamp",
            "clamp_reference_activation": reference,
            "source": "package_decoder_protocol",
            "metric": "delta_P(target)_on_other_dataset",
            "selection_split": "validation" if report_rows else "eval",
            "report_split": "test" if report_rows else "eval",
            "strengthen_reused": reused is not None,
            **({} if reused is not None else selection_lms_metadata(validation_selected)),
            **{
                f"weaken_{k}": v
                for k, v in selection_lms_metadata(
                    weaken_validation_selected
                ).items()
            },
        },
    )


@contextmanager
def multi_site_steering_context(
    model,
    specs: Sequence[Tuple[str, torch.Tensor]],
    strength: float,
    *,
    token_selector: str = "all",
) -> Iterator[Any]:
    """Add ``strength * direction`` at each ``(module_path, direction)`` site."""
    from gradiend.model.modified import apply_activation_steering, remove_hook_handles

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


def run_sae_multi_layer_causal_sweep(
    model,
    tokenizer,
    meta_rows: Sequence[Dict[str, str]],
    *,
    layer_specs: Sequence[Dict[str, Any]],
    target_class: str,
    strengths: Sequence[float] = DEFAULT_CAUSAL_STRENGTHS,
    lms_texts: Optional[Sequence[str]] = None,
    include_random_control: bool = True,
    random_seed: int = 0,
    method_id: Optional[str] = None,
    token_selector: str = "all",
    bag_reduce: str = "sum",
    report_rows: Optional[Sequence[Dict[str, str]]] = None,
    base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    base_lms: Optional[Any] = None,
    r_base_probs: Optional[Dict[str, Dict[str, float]]] = None,
    r_base_lms: Optional[Any] = None,
    reuse_strengthen: Optional[CausalMethodResult | Mapping[str, Any]] = None,
) -> CausalMethodResult:
    """LMS-gated multi-layer SAE steering (top-k decoder dirs at every layer).

    Each ``layer_specs`` entry is either
    ``{module_path, sae, feature_indices, layer?}`` or
    ``{module_path, direction, feature_indices, layer?}``.  The latter lets
    callers extract the small decoder direction while loading SAEs one at a
    time, rather than retaining every full dictionary on the GPU.
    At strength α, adds α·bag(W_dec[feats]) simultaneously at every listed residual site.

    See :func:`run_sae_causal_sweep` for ``base_probs``/``base_lms``/
    ``r_base_probs``/``r_base_lms`` — pass them in to skip the redundant
    unsteered forward pass when sweeping many layer/feature combos over the
    same fixed ``meta_rows``/``report_rows``.
    """
    dirs: List[Tuple[str, torch.Tensor]] = []
    feat_meta: List[Dict[str, Any]] = []
    for spec in layer_specs:
        idxs = [int(i) for i in (spec.get("feature_indices") or [])]
        if not idxs:
            continue
        module_path = str(spec["module_path"])
        if spec.get("direction") is not None:
            direction = spec["direction"]
        else:
            sae = spec["sae"]
            if len(idxs) == 1:
                direction = sae_decoder_direction(sae, idxs[0])
            else:
                direction = sae_decoder_bag_direction(sae, idxs, reduce=bag_reduce)
        dirs.append((module_path, direction.detach().float().cpu().flatten()))
        feat_meta.append(
            {
                "layer": spec.get("layer"),
                "module_path": module_path,
                "feature_indices": idxs,
            }
        )
    if not dirs:
        mid = method_id or f"sae:{target_class}:all_k"
        return CausalMethodResult(
            method=mid,
            backend="sae",
            target_class=str(target_class),
            strengths=[],
            selected_strength=None,
            selected=None,
            meta={"error": "no layer features", "token_selector": token_selector},
        )

    texts = [r["text"] for r in meta_rows]
    lms_texts = list(
        lms_texts or [r["text"] for r in meta_rows if r["group"] == "neutral"] or texts
    )
    if base_probs is None or base_lms is None:
        frame = meta_rows_to_frame(meta_rows)
        base_probs = score_class_probs_by_dataset(model, tokenizer, frame)
        base_lms = compute_lms_safe(model, tokenizer, lms_texts)

    reused = (
        causal_method_result_from_dict(reuse_strengthen)
        if isinstance(reuse_strengthen, Mapping)
        else reuse_strengthen
    )
    results: List[StrengthResult] = []
    if reused is None:
        for sign in (1.0, -1.0):
            signed = [(m, d * float(sign)) for m, d in dirs]
            for s in strengths:
                with multi_site_steering_context(
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
                                f"sae_all_layers n_sites={len(dirs)} sign={sign:g} "
                                f"token_selector={token_selector}"
                            ),
                        )
                    )

    def _run_one(sign: float, s: float) -> StrengthResult:
        signed_s = [(m, d * float(sign)) for m, d in dirs]
        with multi_site_steering_context(
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
                    f"sae_all_layers n_sites={len(dirs)} sign={sign:g} "
                    f"token_selector={token_selector} refine=1"
                ),
            )

    if reused is None:
        results, selected = refine_and_select_lms_gated(results, _run_one)
        validation_selected = selected
    else:
        if reused.selected is None or not reused.strengths:
            raise ValueError("cannot reuse incomplete multi-layer SAE strengthen result")
        results = list(reused.strengths)
        selected = reused.selected
        validation_selected = None
    weaken_results, weaken_validation_selected = refine_and_select_weaken_lms_gated(
        results, _run_one, target_class=target_class
    )
    selected_sign = None if selected is None else _parse_sweep_sign(selected.notes)
    if selected is not None and selected_sign is None:
        raise ValueError(
            f"selected StrengthResult is missing a sign= tag in notes "
            f"(notes={selected.notes!r}); cannot determine steering "
            "direction. Every result in this sweep must be tagged."
        )

    random_ctrl = None
    if include_random_control and selected is not None and not report_rows:
        rng = np.random.default_rng(random_seed)
        rand_specs = []
        for m, d in dirs:
            r = torch.tensor(rng.normal(size=d.numel()), dtype=torch.float32)
            r = r * (d.norm() / (r.norm() + 1e-8))
            rand_specs.append((m, r * float(selected_sign)))
        with multi_site_steering_context(
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
                notes="matched-L2 random multi-layer SAE directions",
            )

    if report_rows and (selected is not None or weaken_validation_selected is not None):
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        if r_base_probs is None or r_base_lms is None:
            r_frame = meta_rows_to_frame(report_rows)
            r_base_probs = score_class_probs_by_dataset(model, tokenizer, r_frame)
            r_base_lms = compute_lms_safe(model, tokenizer, r_lms)
    if report_rows and selected is not None and dirs and reused is None:
        r_texts = [r["text"] for r in report_rows]
        r_lms = [r["text"] for r in report_rows if r["group"] == "neutral"] or r_texts
        signed_sel = [(m, d * float(selected_sign)) for m, d in dirs]
        with multi_site_steering_context(
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
            with multi_site_steering_context(
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
                    notes="matched-L2 random multi-layer SAE directions report=test",
                )
    elif report_rows and selected is not None and dirs and include_random_control:
        rng = np.random.default_rng(random_seed)
        rand_specs = []
        for m, d in dirs:
            r = torch.tensor(rng.normal(size=d.numel()), dtype=torch.float32)
            r = r * (d.norm() / (r.norm() + 1e-8))
            rand_specs.append((m, r * float(selected_sign)))
        with multi_site_steering_context(
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
                notes="matched-L2 random multi-layer SAE directions report=test repair=1",
            )

    weaken_selected = weaken_validation_selected
    if report_rows and weaken_validation_selected is not None and dirs:
        weaken_sign = _parse_sweep_sign(weaken_validation_selected.notes)
        if weaken_sign is None:
            raise ValueError("weaken selection is missing its sign tag")
        weaken_specs = [(m, d * float(weaken_sign)) for m, d in dirs]
        with multi_site_steering_context(
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

    mid = method_id or f"sae:{target_class}:all_k"
    return CausalMethodResult(
        method=mid,
        backend="sae",
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
            "token_selector": token_selector,
            "sign": selected_sign,
            "n_sites": len(dirs),
            "layer_specs": feat_meta,
            "bag_reduce": bag_reduce,
            "source": "sae_all_layers_decoder",
            "metric": "delta_P(target)_on_other_dataset",
            "selection_split": "validation" if report_rows else "eval",
            "report_split": "test" if report_rows else "eval",
            "strengthen_reused": reused is not None,
            **({} if reused is not None else selection_lms_metadata(validation_selected)),
            **{
                f"weaken_{k}": v
                for k, v in selection_lms_metadata(
                    weaken_validation_selected
                ).items()
            },
        },
    )


def _round_float(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, ndigits)
    return value


def _sample_row_for_persist(
    sample: CausalSample,
    *,
    include_text: bool,
    float_digits: int = 6,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = sample.to_dict()
    if not include_text:
        row.pop("text", None)
    for key, val in list(row.items()):
        row[key] = _round_float(val, float_digits)
    if extra:
        for key, val in extra.items():
            row[key] = _round_float(val, float_digits)
    return row


def strength_result_from_dict(payload: Mapping[str, Any]) -> "StrengthResult":
    """Inverse of :meth:`StrengthResult.to_dict`, minus ``samples``.

    ``to_dict`` drops per-sample text, which nothing in the headline metrics
    reads, so a persisted bundle round-trips exactly for scoring purposes.
    """
    return StrengthResult(
        strength=float(payload["strength"]),
        lms=payload.get("lms"),
        base_lms=payload.get("base_lms"),
        lms_ok=bool(payload.get("lms_ok")),
        samples=[],
        summary_by_group=dict(payload.get("summary_by_group") or {}),
        signed_effect=float(payload.get("signed_effect") or 0.0),
        is_random_control=bool(payload.get("is_random_control")),
        notes=str(payload.get("notes") or ""),
        selection_metric=payload.get("selection_metric"),
    )


def causal_method_result_from_dict(
    payload: Mapping[str, Any],
) -> "CausalMethodResult":
    """Inverse of :meth:`CausalMethodResult.to_dict`.

    Lets a persisted ``summary.json`` be re-scored through
    :func:`headline_metrics` without re-running the intervention, which is what
    recovering an E4 run whose rows were dropped at aggregation time requires.
    """
    def _opt(key: str) -> Optional["StrengthResult"]:
        raw = payload.get(key)
        return strength_result_from_dict(raw) if isinstance(raw, Mapping) else None

    return CausalMethodResult(
        method=str(payload["method"]),
        backend=str(payload.get("backend") or ""),
        target_class=str(payload.get("target_class") or ""),
        strengths=[
            strength_result_from_dict(s)
            for s in (payload.get("strengths") or [])
            if isinstance(s, Mapping)
        ],
        selected_strength=payload.get("selected_strength"),
        selected=_opt("selected"),
        modified_model_path=payload.get("modified_model_path"),
        random_control=_opt("random_control"),
        meta=dict(payload.get("meta") or {}),
        weaken_strengths=[
            strength_result_from_dict(s)
            for s in (payload.get("weaken_strengths") or [])
            if isinstance(s, Mapping)
        ],
        weaken_selected_strength=payload.get("weaken_selected_strength"),
        weaken_selected=_opt("weaken_selected"),
    )


def persist_causal_bundle(
    output_dir: Path | str,
    results: Sequence[CausalMethodResult],
    *,
    write_curve_samples: bool = False,
    write_sample_text: bool = False,
    max_samples_per_method: Optional[int] = None,
    float_digits: int = 6,
) -> str:
    """Write summaries + optional per-sample JSONL under ``output_dir/causal/``.

    Defaults are deliberately compact: curve dumps are off (they repeat every
    sample × every strength with full prompts and dominated disk on gender_en),
    and selected-strength JSONL omits ``text`` unless requested.
    """
    root = Path(output_dir) / "causal"
    root.mkdir(parents=True, exist_ok=True)
    summary = [r.to_dict() for r in results]
    persist_causal_summary(output_dir, summary)
    for r in results:
        safe = r.method.replace(":", "_")
        sample_path = root / f"samples_{safe}.jsonl"
        selected = r.selected
        with sample_path.open("w", encoding="utf-8") as fh:
            if selected is not None:
                samples = list(selected.samples)
                if max_samples_per_method is not None:
                    samples = samples[: max(0, int(max_samples_per_method))]
                for s in samples:
                    fh.write(
                        json.dumps(
                            _sample_row_for_persist(
                                s,
                                include_text=write_sample_text,
                                float_digits=float_digits,
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        if not write_curve_samples:
            # Remove stale full dumps from earlier runs when re-persisting.
            stale = root / f"curve_samples_{safe}.jsonl"
            if stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass
            continue
        curve_path = root / f"curve_samples_{safe}.jsonl"
        with curve_path.open("w", encoding="utf-8") as fh:
            for sr in r.strengths:
                samples = list(sr.samples)
                if max_samples_per_method is not None:
                    samples = samples[: max(0, int(max_samples_per_method))]
                for s in samples:
                    row = _sample_row_for_persist(
                        s,
                        include_text=write_sample_text,
                        float_digits=float_digits,
                        extra={
                            "strength": sr.strength,
                            "lms": sr.lms,
                            "lms_ok": sr.lms_ok,
                            "signed_effect": sr.signed_effect,
                        },
                    )
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return str(root)


def persist_causal_summary(
    output_dir: Path | str,
    entries: Sequence[Mapping[str, Any]],
) -> str:
    """Atomically persist the complete causal method summary.

    ``persist_causal_bundle`` initially writes the methods evaluated in the
    current invocation.  A skip-existing causal refresh may later merge those
    rows with completed sweeps from a prior ``results.json``; callers must then
    invoke this helper with that merged inventory so ``causal/summary.json``
    does not shrink to only the retried failures (or an empty list).
    """
    root = Path(output_dir) / "causal"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.json"
    tmp = root / "summary.json.tmp"
    payload = [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return str(path)


def _strength_result_from_stored(
    payload: Mapping[str, Any], *, target_class: str
) -> StrengthResult:
    summary = payload["summary_by_group"]
    if not isinstance(summary, dict) or not summary:
        raise KeyError("summary_by_group")
    return StrengthResult(
        strength=float(payload["strength"]),
        lms=payload["lms"],
        base_lms=payload["base_lms"],
        lms_ok=bool(payload["lms_ok"]),
        summary_by_group=dict(summary),
        signed_effect=float(
            signed_effect_from_group_summary(summary, target_class=str(target_class))
        ),
        selection_metric=(
            None
            if "selection_metric" not in payload
            else (
                None
                if payload["selection_metric"] is None
                else float(payload["selection_metric"])
            )
        ),
        is_random_control=bool(payload["is_random_control"])
        if "is_random_control" in payload
        else False,
        notes=str(payload["notes"]) if "notes" in payload else "",
    )


def _rebind_row_decoder_headline(row: Dict[str, Any]) -> None:
    """Repair a legacy method row from its stored decoder grid, when necessary."""
    extras = row.get("extras") or {}
    causal = extras.get("causal")
    if not isinstance(causal, dict):
        return
    meta = causal.get("meta") or {}
    # Since causal protocol v4, ``strengths`` is the *validation* decoder
    # curve, while ``selected`` and the row metrics are the selected LR's
    # separately evaluated, frozen *test* result.  The legacy rebind below
    # predates that split and must never turn a current test headline back
    # into its validation-selection score merely because both use the same
    # package-selected LR.
    #
    # Keep this provenance check deliberately independent of the numerical
    # values: test and validation can legitimately agree, and a zero test
    # effect is still a valid reported outcome.
    selected_payload = causal.get("selected")
    if (
        meta.get("selection_split") == "validation"
        and meta.get("report_split") == "test"
        and isinstance(selected_payload, Mapping)
        and selected_payload.get("signed_effect") is not None
    ):
        return
    strengths_raw = [s for s in (causal.get("strengths") or []) if isinstance(s, dict)]
    if str(meta.get("source") or "") != "decoder_grid_cache":
        metrics = row.get("metrics") or {}
        if metrics.get("causal_signed_effect") is not None or not strengths_raw:
            return
    # This compatibility path only serves pre-validation/test-split payloads.
    # Do not let a legacy result get silently rewritten during report rebuilds:
    # operators need an explicit trace of every row whose headline came from a
    # cached decoder curve rather than a frozen report result.
    print(
        "WARNING: applying legacy decoder-headline rebind for "
        f"{row.get('method')!r}; no frozen validation-selected test result "
        "was found.",
        flush=True,
    )
    pkg = meta.get("package_learning_rate")
    if pkg is None:
        pkg = (meta.get("package_decoder_entry") or {}).get("learning_rate")
    if pkg is None and causal.get("selected_strength") is not None:
        pkg = causal.get("selected_strength")
    if pkg is None:
        sel = causal.get("selected") or {}
        pkg = sel.get("strength")
    if pkg is None:
        raise KeyError(
            f"package_learning_rate missing for decoder_grid_cache method "
            f"{row.get('method')!r}"
        )
    pkg_f = float(pkg)
    if not strengths_raw:
        raise KeyError(f"strengths missing/empty for {row.get('method')!r}")
    hit = next(
        (
            s
            for s in strengths_raw
            if "strength" in s and abs(float(s["strength"]) - pkg_f) < 1e-12
        ),
        None,
    )
    if hit is None:
        raise KeyError(
            f"package_learning_rate={pkg_f!r} not in strengths for "
            f"{row.get('method')!r}"
        )
    if "target_class" in causal and causal["target_class"] is not None:
        target = str(causal["target_class"])
    elif "target_class" in row and row["target_class"] is not None:
        target = str(row["target_class"])
    else:
        raise KeyError(f"target_class missing for {row.get('method')!r}")
    selected = _strength_result_from_stored(hit, target_class=target)
    strengths = [
        _strength_result_from_stored(s, target_class=target) for s in strengths_raw
    ]
    rnd_raw = causal.get("random_control")
    random_ctrl = None
    meta = dict(meta)
    if isinstance(rnd_raw, dict) and "strength" in rnd_raw:
        if abs(float(rnd_raw["strength"]) - pkg_f) < 1e-12:
            rnd_summary = rnd_raw.get("summary_by_group")
            if isinstance(rnd_summary, dict) and rnd_summary:
                random_ctrl = _strength_result_from_stored(
                    rnd_raw, target_class=target
                )
            else:
                meta["random_control_omitted"] = "legacy empty summary_by_group"
    weaken_strengths = [
        strength_result_from_dict(s)
        for s in (causal.get("weaken_strengths") or [])
        if isinstance(s, Mapping)
    ]
    weaken_selected_raw = causal.get("weaken_selected")
    weaken_selected = (
        strength_result_from_dict(weaken_selected_raw)
        if isinstance(weaken_selected_raw, Mapping)
        else None
    )
    bound = CausalMethodResult(
        method=str(causal["method"] if "method" in causal else row["method"]),
        backend=str(causal["backend"]) if "backend" in causal else "",
        target_class=target,
        strengths=strengths,
        selected_strength=pkg_f,
        selected=selected,
        modified_model_path=causal.get("modified_model_path"),
        random_control=random_ctrl,
        weaken_strengths=weaken_strengths,
        weaken_selected_strength=(
            None if weaken_selected is None else weaken_selected.strength
        ),
        weaken_selected=weaken_selected,
        meta={
            **meta,
            "package_learning_rate": pkg_f,
            "study_learning_rate": pkg_f,
            "metric": "P(target)_strengthen_panel",
        },
    )
    hm = headline_metrics(bound)
    metrics = row.setdefault("metrics", {})
    if isinstance(metrics, dict):
        metrics.update(hm)
    causal["selected_strength"] = pkg_f
    causal["selected"] = {
        k: v for k, v in selected.to_dict().items() if k != "samples"
    }
    if random_ctrl is None:
        causal["random_control"] = None
    else:
        causal["random_control"] = {
            k: v for k, v in random_ctrl.to_dict().items() if k != "samples"
        }
    causal["meta"] = bound.meta
    extras["causal"] = causal
    row["extras"] = extras


def apply_package_decoder_headlines(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility no-op retained for existing report/plot call sites.

    Consumers must only read the persisted headline metrics.  Repairing a
    legacy decoder cache is an explicit migration/causal-refresh operation,
    never something a report, plot, or aggregate is allowed to inspect or do.
    """
    return results


def _factual_p_from_summary(
    summary: Mapping[str, Any], target_class: str
) -> Tuple[Optional[float], Optional[float]]:
    g = summary.get(str(target_class)) if isinstance(summary, Mapping) else None
    if not isinstance(g, Mapping):
        return None, None
    base = g.get("mean_p_target_base")
    mod = g.get("mean_p_target_mod")
    b = float(base) if isinstance(base, (int, float)) else None
    m = float(mod) if isinstance(mod, (int, float)) else None
    return b, m


def weaken_signed_effect(sr: StrengthResult, target_class: str) -> Optional[float]:
    """ΔP⁻: drop in P(target) on the factual/same-class panel (LMS-gated max)."""
    base, mod = _factual_p_from_summary(sr.summary_by_group or {}, target_class)
    if base is not None and mod is not None:
        return float(base) - float(mod)
    g = (sr.summary_by_group or {}).get(str(target_class)) if sr.summary_by_group else None
    if isinstance(g, Mapping) and g.get("mean_delta_p_target") is not None:
        return -float(g["mean_delta_p_target"])
    return None


def as_weaken_strength(
    sr: StrengthResult,
    *,
    target_class: str,
) -> Optional[StrengthResult]:
    """Copy one intervention result with ``signed_effect`` set to factual drop.

    Strengthen and weaken use the same probability panels but opposite causal
    objectives.  The copy is deliberate: callers often retain the original
    result in a strengthen curve, where overwriting ``signed_effect`` would
    corrupt ``dP+``.
    """
    value = weaken_signed_effect(sr, target_class)
    if value is None:
        return None
    return replace(sr, signed_effect=float(value), selection_metric=None)


def as_strengthen_strength(
    sr: StrengthResult,
    *,
    target_class: str,
) -> StrengthResult:
    """Copy one stored intervention result with the strengthen-panel objective.

    This is used when repairing an inverted bidirectional decoder bundle: the
    old weaken curve has the now-correct strengthening feature factor, but its
    cached ``signed_effect`` was deliberately expressed as factual-class drop.
    The per-group probabilities retain everything needed to reconstruct the
    strengthening objective without another validation forward pass.
    """
    summary = sr.summary_by_group or {}
    panel = strengthen_panel_from_summary(str(target_class), summary)
    group = summary.get(panel)
    if not isinstance(group, Mapping):
        raise KeyError(f"stored curve has no strengthen panel {panel!r}")
    delta = group.get("mean_delta_p_target")
    modified = group.get("mean_p_target_mod")
    if delta is None or modified is None:
        raise KeyError(
            f"stored strengthen panel {panel!r} lacks delta/modified probability"
        )
    return replace(
        sr,
        signed_effect=float(delta),
        selection_metric=float(modified),
    )


def reselect_inverted_bidirectional_curves(
    result: CausalMethodResult | Mapping[str, Any],
    *,
    target_class: str,
) -> Tuple[
    List[StrengthResult],
    StrengthResult,
    List[StrengthResult],
    StrengthResult,
]:
    """Swap/reselect persisted +/- validation grids after a polarity fix.

    No model evaluation occurs here. The former weaken grid becomes the
    corrected strengthen grid, and vice versa; both objectives and LMS gates
    are recomputed from their persisted per-group summaries.
    """
    stored = (
        causal_method_result_from_dict(result)
        if isinstance(result, Mapping)
        else result
    )
    strengthen_curve = [
        as_strengthen_strength(sr, target_class=target_class)
        for sr in stored.weaken_strengths
        if not sr.is_random_control
    ]
    weaken_curve = [
        converted
        for sr in stored.strengths
        if not sr.is_random_control
        for converted in [as_weaken_strength(sr, target_class=target_class)]
        if converted is not None
    ]
    strengthen_selected = select_lms_gated(
        strengthen_curve, prefer_signed_effect=False
    )
    weaken_selected = select_lms_gated(weaken_curve, prefer_signed_effect=True)
    if strengthen_selected is None or weaken_selected is None:
        raise ValueError(
            f"cannot reselect incomplete inverted causal grids for {target_class!r}"
        )
    return (
        strengthen_curve,
        strengthen_selected,
        weaken_curve,
        weaken_selected,
    )


def reselect_direct_bidirectional_curves(
    result: CausalMethodResult | Mapping[str, Any],
    *,
    target_class: str,
) -> Tuple[
    List[StrengthResult],
    StrengthResult,
    List[StrengthResult],
    StrengthResult,
]:
    """Re-score/reselect a persisted direct +/- sweep on the correct panels.

    Older CAA-style sweeps computed ``signed_effect`` from the target's own
    dataset whenever that panel existed.  Their intended strengthen objective
    is P(target) on the rival dataset, while weakening is the drop on the
    target dataset.  Every stored point retains both group summaries, so both
    validation objectives can be reconstructed without another model forward.
    """
    stored = (
        causal_method_result_from_dict(result)
        if isinstance(result, Mapping)
        else result
    )
    strengthen_curve = [
        as_strengthen_strength(sr, target_class=target_class)
        for sr in stored.strengths
        if not sr.is_random_control
    ]
    weaken_curve = [
        converted
        for sr in stored.strengths
        if not sr.is_random_control
        for converted in [as_weaken_strength(sr, target_class=target_class)]
        if converted is not None
    ]
    strengthen_selected = select_lms_gated(
        strengthen_curve, prefer_signed_effect=True
    )
    weaken_selected = select_lms_gated(
        weaken_curve, prefer_signed_effect=True
    )
    if strengthen_selected is None or weaken_selected is None:
        raise ValueError(
            f"cannot reselect incomplete direct causal grids for {target_class!r}"
        )
    return (
        strengthen_curve,
        strengthen_selected,
        weaken_curve,
        weaken_selected,
    )




def refine_and_select_weaken_lms_gated(
    results: Sequence[StrengthResult],
    run_one: Callable[[float, float], StrengthResult],
    *,
    target_class: str,
    n_refine: int = CAUSAL_STRENGTH_REFINE_POINTS,
    ratio: float = LMS_RATIO,
) -> Tuple[List[StrengthResult], Optional[StrengthResult]]:
    """Refine/select a genuine weaken curve independently of strengthen.

    ``run_one`` returns the ordinary per-intervention probability result.  This
    adapter converts each point to ``base P(target) - modified P(target)`` on
    the target's own panel before selection, so the chosen sign/strength is the
    intervention that actually weakens the target.
    """
    weakened = [
        converted
        for sr in results
        if not sr.is_random_control
        for converted in [as_weaken_strength(sr, target_class=target_class)]
        if converted is not None
    ]

    def _run_weaken(sign: float, strength: float) -> StrengthResult:
        raw = run_one(sign, strength)
        converted = as_weaken_strength(raw, target_class=target_class)
        if converted is None:
            raise KeyError(
                f"weaken evaluation has no factual panel for {target_class!r}"
            )
        return converted

    return refine_and_select_lms_gated(
        weakened,
        _run_weaken,
        n_refine=n_refine,
        prefer_signed_effect=True,
        ratio=ratio,
    )


def specificity_vs_random_control(
    effect: Optional[float],
    random_effect: Optional[float],
    *,
    epsilon: float = 1e-6,
) -> Tuple[Optional[float], Optional[bool]]:
    """Turn (real effect, matched random-control effect) into a reportable pair.

    ``causal_random_signed_effect`` used to sit next to ``causal_signed_effect``
    with nothing computed from the pair anywhere downstream — added 2026-08-21
    after review flagged that as compute with no paper-facing payoff. Returns
    ``(specificity_ratio, beats_random_control)``:

    - ``beats_random_control``: ``|effect| > |random_effect|`` — always
      well-defined when both inputs are present, this is the number to
      aggregate across methods (see :func:`specificity_control_summary`) for
      a paper-citable "N% of causal methods exceed their matched random
      control" sanity-check statistic.
    - ``specificity_ratio``: ``|effect| / |random_effect|`` — how many times
      larger the real effect is. ``None`` when the random control itself is
      ~0 (``epsilon``), since the ratio is undefined/misleadingly huge there;
      ``beats_random_control`` is still meaningful in that case.

    Returns ``(None, None)`` when either input is missing — this is a single
    random-direction draw, not a null distribution; treat both outputs as a
    coarse sanity check, not a substitute for a real significance test.
    """
    if effect is None or random_effect is None:
        return None, None
    eff = float(effect)
    rnd = float(random_effect)
    beats = abs(eff) > abs(rnd)
    ratio = None if abs(rnd) < epsilon else abs(eff) / abs(rnd)
    return ratio, beats


def specificity_control_summary(
    method_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate ``causal_beats_random_control`` across many headline-metric rows.

    Pass a sequence of dicts shaped like :func:`headline_metrics`'s output
    (e.g. the ``extras``/method rows already stored in a task's
    ``results.json``). Produces the single paper-citable sanity-check
    statistic this random control was computed for — "N of M causal methods
    show a larger effect than their matched random-direction control" —
    instead of leaving two point estimates uncompared. Rows missing either
    value (no random control was computed, or the method itself had no
    selected result) are excluded from ``n_total``, not counted as failures.
    """
    considered = 0
    exceeds = 0
    ratios: List[float] = []
    for row in method_rows:
        if not isinstance(row, Mapping):
            continue
        beats = row.get("causal_beats_random_control")
        if beats is None:
            continue
        considered += 1
        if beats:
            exceeds += 1
        ratio = row.get("causal_specificity_ratio")
        if isinstance(ratio, (int, float)) and ratio == ratio:
            ratios.append(float(ratio))
    ratios.sort()
    median_ratio = None
    if ratios:
        mid = len(ratios) // 2
        median_ratio = (
            ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2.0
        )
    return {
        "n_total": considered,
        "n_exceeds_random_control": exceeds,
        "fraction_exceeds_random_control": (
            exceeds / considered if considered else None
        ),
        "median_specificity_ratio": median_ratio,
    }


def headline_metrics(result: CausalMethodResult) -> Dict[str, Any]:
    """Compact metrics for results schema / comparison table.

    LMS gate = ``LMS_RATIO`` (0.99). Includes strength→LMS/effect curve for plots.
    Headline strength is ``result.selected`` (decoder-plot LR for GRADIEND/ACTIEND).
    """
    strength_vals = [
        float(s.strength) for s in (result.strengths or []) if s.strength is not None
    ]
    grid_min = min(strength_vals) if strength_vals else None
    grid_max = max(strength_vals) if strength_vals else None
    curve = strength_curve_all(result.strengths or [])
    sel = result.selected
    weaken_sel = result.weaken_selected
    weaken_curve = strength_curve_all(result.weaken_strengths or [])

    def _flags(selected: Optional[StrengthResult]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "causal_grid_ceiling": False,
            "causal_grid_floor": False,
            "causal_null_effect": False,
        }
        if selected is None:
            return out
        if grid_max is not None and abs(selected.strength - grid_max) < 1e-12:
            out["causal_grid_ceiling"] = True
        if grid_min is not None and abs(selected.strength - grid_min) < 1e-12:
            out["causal_grid_floor"] = True
        eff = selected.signed_effect
        if isinstance(eff, (int, float)) and abs(float(eff)) < 1e-6:
            out["causal_null_effect"] = True
        return out

    def _pack_sel(selected: Optional[StrengthResult], *, ratio: float) -> Dict[str, Any]:
        if selected is None:
            return {
                "strength": None,
                "signed_effect": None,
                "lms": None,
                "lms_ok": None,
                "lms_ratio": ratio,
                "gate_empty": True,
            }
        return {
            "strength": selected.strength,
            "signed_effect": selected.signed_effect,
            "lms": selected.lms,
            "base_lms": selected.base_lms,
            "lms_ok": _lms_passes(selected, ratio=ratio),
            "lms_ratio": ratio,
            "gate_empty": "lms_gate_empty" in (selected.notes or ""),
            "notes": selected.notes,
        }

    curve_rows = [
        {
            "strength": float(sr.strength),
            "lms": sr.lms,
            "base_lms": sr.base_lms,
            "signed_effect": sr.signed_effect,
            "lms_ratio_to_base": (
                None
                if sr.lms is None or sr.base_lms in (None, 0)
                else float(sr.lms) / float(sr.base_lms)
            ),
            "lms_ok": _lms_passes(sr, ratio=LMS_RATIO),
        }
        for sr in curve
    ]

    extras = {
        "causal_lms_curve": curve_rows,
        "causal_selected": _pack_sel(sel, ratio=LMS_RATIO),
        "causal_weaken_lms_curve": [
            {
                "strength": float(sr.strength),
                "lms": sr.lms,
                "base_lms": sr.base_lms,
                "signed_effect": sr.signed_effect,
                "lms_ratio_to_base": (
                    None
                    if sr.lms is None or sr.base_lms in (None, 0)
                    else float(sr.lms) / float(sr.base_lms)
                ),
                "lms_ok": _lms_passes(sr, ratio=LMS_RATIO),
            }
            for sr in weaken_curve
        ],
        "causal_weaken_selected": _pack_sel(weaken_sel, ratio=LMS_RATIO),
    }
    result_meta = result.meta or {}
    selection_metrics = {
        "causal_selection_strength": result_meta.get(
            "selection_strength", result.selected_strength
        ),
        "causal_selection_signed_effect": result_meta.get("selection_signed_effect"),
        "causal_selection_lms": result_meta.get("selection_lms"),
        "causal_selection_base_lms": result_meta.get("selection_base_lms"),
        "causal_selection_lms_ratio_to_base": result_meta.get(
            "selection_lms_ratio_to_base"
        ),
        "causal_selection_lms_ok": result_meta.get("selection_lms_ok"),
        "causal_selection_split": result_meta.get("selection_split"),
        "causal_report_split": result_meta.get("report_split"),
        "causal_weaken_selection_strength": result_meta.get(
            "weaken_selection_strength", result.weaken_selected_strength
        ),
        "causal_weaken_selection_signed_effect": result_meta.get(
            "weaken_selection_signed_effect"
        ),
        "causal_weaken_selection_lms": result_meta.get("weaken_selection_lms"),
        "causal_weaken_selection_base_lms": result_meta.get(
            "weaken_selection_base_lms"
        ),
        "causal_weaken_selection_lms_ratio_to_base": result_meta.get(
            "weaken_selection_lms_ratio_to_base"
        ),
        "causal_weaken_selection_lms_ok": result_meta.get(
            "weaken_selection_lms_ok"
        ),
    }

    if sel is None:
        flags = _flags(None)
        err = (result.meta or {}).get("error")
        return {
            "causal_selected_strength": None,
            "causal_signed_effect": None,
            "causal_signed_effect_weaken": None,
            "causal_weaken_selected_strength": None,
            "causal_weaken_lms": None,
            "causal_weaken_base_lms": None,
            "causal_weaken_lms_ok": None,
            "causal_base_p": None,
            "causal_lms": None,
            "causal_lms_ok": None,
            "causal_delta_mean": None,
            "causal_delta_target": None,
            "causal_delta_other": None,
            "causal_delta_neutral": None,
            "causal_effectiveness": None,
            "causal_random_signed_effect": None,
            "causal_specificity_ratio": None,
            "causal_beats_random_control": None,
            "causal_n_per_group": None,
            "causal_error": err,
            **selection_metrics,
            **flags,
            **extras,
        }
    groups = sel.summary_by_group
    if not isinstance(groups, dict) or not groups:
        raise KeyError("selected.summary_by_group")
    target = str(result.target_class)
    names = class_names_from_probs(groups)
    panel = strengthen_panel_from_summary(target, groups)
    rnd = result.random_control

    def _delta(group_key: str) -> Optional[float]:
        g = groups.get(group_key)
        if not isinstance(g, dict):
            return None
        v = g.get("mean_delta_p_target")
        if v is None:
            return None
        return float(v)

    effect = float(sel.signed_effect)
    # Reconcile: headline signed_effect must match strengthen-panel ΔP.
    panel_delta = _delta(panel)
    if panel_delta is not None and abs(effect - panel_delta) > 1e-9:
        effect = panel_delta
        sel.signed_effect = panel_delta
    flags = _flags(sel)
    doth = None
    if target in names:
        other = other_class(target, names)
    else:
        other = names[0] if names else None
    if other is not None:
        doth = _delta(other)
    base_p, _mod_p = _factual_p_from_summary(groups, target)
    dtgt = _delta(target)
    # Headroom-normalize the *headline* effect (`effect`, reported everywhere
    # as causal_signed_effect / dP+), not causal_delta_target -- for one-pole
    # rows with a rival panel scored, `effect` is measured on `panel` (often
    # the rival class, per the decoder strengthen contract), which is *not*
    # the same panel `causal_base_p` describes (always the target's own).
    # Normalizing by the wrong panel's base rate can even flip the sign of
    # the "effectiveness" number relative to the effect it's meant to scale.
    base_p_panel, _ = _factual_p_from_summary(groups, panel)
    effectiveness = None
    if base_p_panel is not None:
        headroom = 1.0 - float(base_p_panel)
        if headroom > CAUSAL_EFFECTIVENESS_MIN_HEADROOM:
            effectiveness = float(effect) / headroom
    rnd_effect = None if rnd is None else rnd.signed_effect
    specificity_ratio, beats_random = specificity_vs_random_control(effect, rnd_effect)
    return {
        "causal_selected_strength": sel.strength,
        "causal_signed_effect": effect,
        "causal_signed_effect_weaken": (
            None if weaken_sel is None else float(weaken_sel.signed_effect)
        ),
        "causal_weaken_selected_strength": (
            None if weaken_sel is None else float(weaken_sel.strength)
        ),
        "causal_weaken_lms": None if weaken_sel is None else weaken_sel.lms,
        "causal_weaken_base_lms": (
            None if weaken_sel is None else weaken_sel.base_lms
        ),
        "causal_weaken_lms_ok": (
            None if weaken_sel is None else _lms_passes(weaken_sel, ratio=LMS_RATIO)
        ),
        "causal_base_p": base_p,
        "causal_lms": sel.lms,
        "causal_base_lms": sel.base_lms,
        "causal_lms_ok": _lms_passes(sel, ratio=LMS_RATIO),
        "causal_delta_mean": effect,
        "causal_delta_target": dtgt,
        "causal_delta_other": doth if doth is not None else effect,
        "causal_delta_neutral": (
            _delta("neutral") if "neutral" in groups else None
        ),
        "causal_effectiveness": effectiveness,
        "causal_random_signed_effect": rnd_effect,
        "causal_specificity_ratio": specificity_ratio,
        "causal_beats_random_control": beats_random,
        "causal_n_per_group": (
            int(n)
            if isinstance((n := (groups.get(target) or {}).get("n")), (int, float))
            and n == n
            else None
        ),
        "causal_modified_model_path": result.modified_model_path,
        "causal_gate_empty": "lms_gate_empty" in (sel.notes or ""),
        **selection_metrics,
        **flags,
        **extras,
    }
