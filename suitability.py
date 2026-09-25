"""Feature suitability score: S = E * G_causal.

Encoding bottleneck E:
  - full (rival texts): min(auc_n, auc_o, excl)
  - one-pole / vs-neutral only: min(auc_n, spec)

Causal gate G (after LMS×0.99 selection):
  - 1 if intended-direction eff >= tau_c and not floor+null / gate_empty
  - 0 when LMS explicitly fails or the effect does not beat its random control
  - 0 otherwise
  - soft_causal: G = clip(eff/tau_c, 0, 1)

Global view (``refresh_global_suitability``):
  model × task matrix of mean primary S, with mean row/column margins.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_TAU_C = 0.05
DEFAULT_E_OK = 0.8  # reason "ok" vs "E" when G passes
MEAN_LABEL = "mean"
ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = ROOT / "runs"
DEFAULT_GLOBAL_OUT = ROOT / "analysis" / "tables"


# Legacy compatibility constant. Current SAE site selection is locked by
# validation detection; encoding_E is not a selection metric anymore.


def _f(metrics: Mapping[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def encoding_e(metrics: Mapping[str, Any]) -> Optional[float]:
    """Fair encoding bottleneck E (same rules as suitability, without causal gate).

    Always scores the metrics it is handed. Site/layer selection happens on the
    validation readout (``val_readout``) via ``detection_score``, so there is no
    split-switching flag here to reach for the wrong split by accident.
    """
    auc_n = _f(metrics, "roc_auc_neutral", "roc_auc")
    if auc_n is None:
        return None
    auc_o = _f(metrics, "roc_auc_other", "min_pairwise_auroc")
    excl = _f(metrics, "class_exclusivity")
    spec = _f(metrics, "neutral_specificity", "specificity")
    if auc_o is not None and excl is not None:
        return float(min(float(auc_n), float(auc_o), float(excl)))
    if spec is not None:
        return float(min(float(auc_n), float(spec)))
    return float(auc_n)


def detection_score(metrics: Mapping[str, Any]) -> Optional[float]:
    """Paper-facing worst-case feature Detection score.

    Unlike the historical ``encoding_E`` selector, Detection always includes
    neutral specificity. Pairwise/rival evaluations additionally include
    rival AUROC and class exclusivity. Keeping this separate avoids silently
    changing the checkpoint/site selection rule of completed experiments.
    """
    auc_n = _f(metrics, "roc_auc_neutral", "roc_auc")
    spec = _f(metrics, "neutral_specificity", "specificity")
    if auc_n is None or spec is None:
        return None
    auc_o = _f(metrics, "roc_auc_other", "min_pairwise_auroc")
    excl = _f(metrics, "class_exclusivity")
    if auc_o is not None or excl is not None:
        if auc_o is None or excl is None:
            return None
        return float(min(auc_n, spec, auc_o, excl))
    return float(min(auc_n, spec))


def intervention_score(metrics: Mapping[str, Any]) -> Optional[float]:
    """Mean bidirectional intervention effect used by the paper tables.

    Both constituents are required.  Falling back to the available direction
    would make a one-sided causal evaluation look like a complete intervention
    score and would make cross-model means incomparable.
    """
    strengthen = _f(metrics, "causal_signed_effect")
    weaken = _f(metrics, "causal_signed_effect_weaken")
    if strengthen is None or weaken is None:
        return None
    return float((strengthen + weaken) / 2.0)


def _has_causal_payload(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics.get("causal_signed_effect") is not None
        or metrics.get("causal_delta_mean") is not None
        or metrics.get("causal_selected_strength") is not None
        or metrics.get("causal_error") is not None
    )


def _recover_legacy_weaken(entry: Mapping[str, Any]) -> Optional[float]:
    """Read an explicitly selected, held-out weaken result when present.

    Old ``strengths`` arrays are strengthen-polarity validation curves for IEND
    and cannot safely reconstruct weakening. Returning ``None`` for those files
    is intentional: an invalid derived headline must not survive aggregation.
    """
    selected = entry.get("weaken_selected")
    if not isinstance(selected, Mapping):
        return None
    value = selected.get("signed_effect")
    return float(value) if isinstance(value, (int, float)) else None


def resolve_causal_metrics(
    results: Mapping[str, Any], method_id: str
) -> Tuple[Dict[str, Any], str]:
    """Find causal metrics for an encoder id (``mid`` or ``mid:causal``).

    ACTIEND stores interventions on policy suffixes (``:tok_all_gate_encoder_direction``,
    ``:tok_all``, …) while encode lives on ``actiend:{cls}``. Prefer the package
    default gate policy, then ``tok_all``, then other inherit children.

    Pair-qualified encoder ids (``gradiend:negative-positive:negative``) also
    fall back to PoC short class ids (``gradiend:negative`` / ``actiend:negative:tok_*``).
    """
    from results_schema import (
        display_method_id,
        encoder_inherit_source,
        short_class_method_aliases,
    )

    cache_key = "_analysis_causal_method_index_v1"
    cached = results.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        by, row_by = cached
    else:
        by = {}
        row_by = {}
        for row in results.get("methods") or []:
            if not row.get("method"):
                continue
            metrics = dict(row.get("metrics") or {})
            # Some SAE/CAA refreshes persist the selected causal result only in
            # ``extras.causal.selected`` while leaving the flat headline fields
            # empty.  Promote that canonical selected result for aggregation.
            causal = (row.get("extras") or {}).get("causal") or {}
            selected = causal.get("selected") if isinstance(causal, Mapping) else None
            if isinstance(selected, Mapping):
                if metrics.get("causal_signed_effect") is None and isinstance(
                    selected.get("signed_effect"), (int, float)
                ):
                    metrics["causal_signed_effect"] = selected["signed_effect"]
                if metrics.get("causal_selected_strength") is None and isinstance(
                    selected.get("strength"), (int, float)
                ):
                    metrics["causal_selected_strength"] = selected["strength"]
                # ``selected`` is the canonical held-out, LMS-gated point.
                # Promoting its effect without its LMS split a single causal
                # observation across two incompatible headline columns.
                if metrics.get("causal_lms") is None and isinstance(
                    selected.get("lms"), (int, float)
                ):
                    metrics["causal_lms"] = selected["lms"]
                if metrics.get("causal_lms_ok") is None and isinstance(
                    selected.get("lms_ok"), bool
                ):
                    metrics["causal_lms_ok"] = selected["lms_ok"]
            weaken = causal.get("weaken_selected") if isinstance(causal, Mapping) else None
            if (
                metrics.get("causal_signed_effect_weaken") is None
                and isinstance(weaken, Mapping)
                and causal.get("weaken_selected_strength") is not None
                and isinstance(weaken.get("signed_effect"), (int, float))
            ):
                metrics["causal_signed_effect_weaken"] = weaken["signed_effect"]
            # The held-out weakening effect and its LMS are one observation.
            # Do not promote ΔP− while dropping the corresponding LMS−.
            if (
                metrics.get("causal_weaken_lms") is None
                and isinstance(weaken, Mapping)
                and causal.get("weaken_selected_strength") is not None
                and isinstance(weaken.get("lms"), (int, float))
            ):
                metrics["causal_weaken_lms"] = weaken["lms"]
            if (
                metrics.get("causal_weaken_lms_ok") is None
                and isinstance(weaken, Mapping)
                and causal.get("weaken_selected_strength") is not None
                and isinstance(weaken.get("lms_ok"), bool)
            ):
                metrics["causal_weaken_lms_ok"] = weaken["lms_ok"]
            raw_id = str(row.get("method")).split("|", 1)[0]
            # Results can contain an encoder row and a causal companion with the
            # same method id.  Merge them; last-row-wins silently erased nested
            # CGA/CAGA causal payloads from the table resolver.
            previous = by.get(raw_id)
            if isinstance(previous, Mapping):
                merged = dict(previous)
                merged.update({k: v for k, v in metrics.items() if v is not None})
                metrics = merged
            by[raw_id] = metrics
            row_by[raw_id] = row
            disp = str(display_method_id(row)).split("|", 1)[0]
            previous = by.get(disp)
            if isinstance(previous, Mapping):
                merged = dict(previous)
                merged.update({k: v for k, v in metrics.items() if v is not None})
                metrics = merged
            by[disp] = metrics
            row_by[disp] = row
        if isinstance(results, MutableMapping):
            results[cache_key] = (by, row_by)

    mid = str(method_id).split("|", 1)[0]
    bases: List[str] = []
    for cand in (mid, *short_class_method_aliases(mid)):
        if cand.endswith(":causal"):
            cand = cand[: -len(":causal")]
        if cand not in bases:
            bases.append(cand)

    # Ungated all-token (``tok_all``) is the ACTIEND causal headline: on gpt2-small
    # matched cells it steers better than the gated default (ACTIEND-CAA -0.035 gated
    # vs +0.006 ungated), so it is the honest ACTIEND-at-its-best. The gated policy
    # stays as the next fallback for suites that only computed it. (2026-09-03)
    preferred_tails = (
        "tok_all",
        "tok_all_gate_encoder_direction",
        "tok_prediction",
    )
    raw = (results.get("raw") or {}).get("causal") or {}
    by_method = raw.get("by_method") or {}

    def _candidate_metrics(metrics: Mapping[str, Any], candidate: str) -> Dict[str, Any]:
        out = dict(metrics)
        entry = by_method.get(candidate)
        # Some persisted SAE rows have flat ΔP fields but retain the selected
        # LMS values only in raw.by_method.  Complete those fields from the
        # same held-out weakening observation, never from a curve point.
        raw_weaken = entry.get("weaken_selected") if isinstance(entry, Mapping) else None
        if (
            out.get("causal_weaken_lms") is None
            and isinstance(raw_weaken, Mapping)
            and entry.get("weaken_selected_strength") is not None
            and isinstance(raw_weaken.get("lms"), (int, float))
        ):
            out["causal_weaken_lms"] = raw_weaken["lms"]
        if (
            out.get("causal_weaken_lms_ok") is None
            and isinstance(raw_weaken, Mapping)
            and entry.get("weaken_selected_strength") is not None
            and isinstance(raw_weaken.get("lms_ok"), bool)
        ):
            out["causal_weaken_lms_ok"] = raw_weaken["lms_ok"]
        has_held_out_weaken = (
            isinstance(entry, Mapping)
            and isinstance(entry.get("weaken_selected"), Mapping)
            and entry.get("weaken_selected_strength") is not None
        )
        if not has_held_out_weaken:
            row_entry = row_by.get(candidate)
            row_causal = (row_entry.get("extras") or {}).get("causal") if isinstance(row_entry, Mapping) else None
            has_held_out_weaken = (
                isinstance(row_causal, Mapping)
                and isinstance(row_causal.get("weaken_selected"), Mapping)
                and row_causal.get("weaken_selected_strength") is not None
            )
        # Pre-v6 rows sometimes persisted a value under the weaken field even
        # though it came from the strengthen-polarity curve. It is reliably
        # stale when the raw entry has no separately selected held-out weaken
        # result, so never expose it to tables/headline aggregation.
        if out.get("causal_signed_effect_weaken") is not None and not has_held_out_weaken:
            out.pop("causal_signed_effect_weaken", None)
            out["causal_weaken_stale"] = True
        if out.get("causal_signed_effect_weaken") is None:
            if isinstance(entry, Mapping):
                recovered = _recover_legacy_weaken(entry)
                if recovered is not None:
                    out["causal_signed_effect_weaken"] = recovered
        return out

    for base in bases:
        for cand in (base, f"{base}:causal"):
            m = by.get(cand) or {}
            if m.get("causal_signed_effect") is not None:
                return _candidate_metrics(m, cand), cand
            if _has_causal_payload(m):
                return _candidate_metrics(m, cand), cand

        # Encoder id → intervention children (actiend:M ← actiend:M:tok_*).
        children: List[Tuple[str, Dict[str, Any]]] = []
        for cand, m in by.items():
            if not _has_causal_payload(m):
                continue
            if encoder_inherit_source(cand) == base:
                children.append((cand, m))
        for tail in preferred_tails:
            for cand, m in children:
                if cand.split(":")[-1] == tail:
                    if m.get("causal_signed_effect") is not None:
                        return _candidate_metrics(m, cand), cand
        for tail in preferred_tails:
            for cand, m in children:
                if cand.split(":")[-1] == tail:
                    return _candidate_metrics(m, cand), cand
        if children:
            def _eff_abs(item: Tuple[str, Dict[str, Any]]) -> float:
                v = item[1].get("causal_signed_effect")
                return abs(float(v)) if isinstance(v, (int, float)) else -1.0

            with_eff = [item for item in children if item[1].get("causal_signed_effect") is not None]
            pool = with_eff or children
            cand, m = max(pool, key=_eff_abs)
            return _candidate_metrics(m, cand), cand

        raw_cands = [base, f"{base}:causal"]
        for tail in preferred_tails:
            raw_cands.append(f"{base}:{tail}")
        for cand in raw_cands:
            entry = by_method.get(cand)
            if not isinstance(entry, dict):
                continue
            hm = entry.get("headline_metrics")
            if isinstance(hm, dict) and hm.get("causal_signed_effect") is not None:
                return _candidate_metrics(hm, cand), cand
            if isinstance(hm, dict) and hm:
                return _candidate_metrics(hm, cand), cand
            sel = entry.get("selected") or {}
            if sel or entry.get("selected_strength") is not None:
                return _candidate_metrics({
                    "causal_selected_strength": entry.get("selected_strength")
                    or sel.get("strength"),
                    "causal_signed_effect": sel.get("signed_effect"),
                    "causal_lms": sel.get("lms"),
                    "causal_lms_ok": sel.get("lms_ok"),
                    "causal_delta_other": sel.get("signed_effect"),
                    "causal_gate_empty": "lms_gate_empty" in str(sel.get("notes") or ""),
                    "causal_grid_floor": bool(entry.get("grid_floor") or sel.get("grid_floor")),
                    "causal_null_effect": bool(entry.get("null_effect") or sel.get("null_effect")),
                }, cand), cand

        # Some dumps store causal only in raw.by_method children (no method row).
        inherited_raw: List[Tuple[str, Dict[str, Any]]] = []
        for cand, entry in by_method.items():
            if not isinstance(entry, dict):
                continue
            if encoder_inherit_source(str(cand)) == base:
                inherited_raw.append((str(cand), entry))
        for tail in preferred_tails:
            for cand, entry in inherited_raw:
                last = cand.split(":")[-1]
                if not (last == tail or last.endswith(f"_{tail}")):
                    continue
                hm = entry.get("headline_metrics")
                if isinstance(hm, dict) and hm.get("causal_signed_effect") is not None:
                    return _candidate_metrics(hm, cand), cand
                sel = entry.get("selected") or {}
                if sel.get("signed_effect") is not None:
                    return _candidate_metrics({
                        "causal_selected_strength": entry.get("selected_strength")
                        or sel.get("strength"),
                        "causal_signed_effect": sel.get("signed_effect"),
                        "causal_lms": sel.get("lms"),
                        "causal_lms_ok": sel.get("lms_ok"),
                        "causal_delta_other": sel.get("signed_effect"),
                        "causal_gate_empty": "lms_gate_empty" in str(sel.get("notes") or ""),
                        "causal_grid_floor": bool(entry.get("grid_floor") or sel.get("grid_floor")),
                        "causal_null_effect": bool(entry.get("null_effect") or sel.get("null_effect")),
                    }, cand), cand

        row = row_by.get(base)
        if isinstance(row, dict) and isinstance((row.get("extras") or {}).get("causal"), dict):
            from causal_eval import _rebind_row_decoder_headline

            scratch = dict(row)
            scratch["metrics"] = dict(scratch.get("metrics") or {})
            scratch["extras"] = dict(scratch.get("extras") or {})
            try:
                _rebind_row_decoder_headline(scratch)
            except Exception:
                pass
            else:
                metrics = dict(scratch.get("metrics") or {})
                causal = (row.get("extras") or {}).get("causal") or {}
                weaken = causal.get("weaken_selected")
                if (
                    metrics.get("causal_signed_effect_weaken") is None
                    and isinstance(weaken, Mapping)
                    and causal.get("weaken_selected_strength") is not None
                    and weaken.get("signed_effect") is not None
                ):
                    # Some ACTIEND rows keep the held-out weaken result only
                    # under extras.causal; promote it exactly as the raw cache
                    # path does. This is a persistence-format difference, not
                    # a missing causal evaluation.
                    metrics["causal_signed_effect_weaken"] = weaken["signed_effect"]
                if metrics.get("causal_signed_effect") is not None:
                    return metrics, base
    return {}, mid


def compute_feature_suitability(
    enc_metrics: Mapping[str, Any],
    causal_metrics: Optional[Mapping[str, Any]] = None,
    *,
    tau_c: float = DEFAULT_TAU_C,
    soft_causal: bool = False,
    e_ok: float = DEFAULT_E_OK,
) -> Dict[str, Any]:
    """Return suitability fields for one method.

    Keys: suitability (S), suitability_E, suitability_G, suitability_scope,
    suitability_reason, suitability_tau_c.
    """
    auc_n = _f(enc_metrics, "roc_auc_neutral", "roc_auc")
    auc_o = _f(enc_metrics, "roc_auc_other", "min_pairwise_auroc")
    excl = _f(enc_metrics, "class_exclusivity")

    out: Dict[str, Any] = {
        "suitability": None,
        "suitability_E": None,
        "suitability_G": None,
        "suitability_scope": None,
        "suitability_reason": "no_encoding",
        "suitability_gate_reason": None,
        "suitability_tau_c": float(tau_c),
    }
    if auc_n is None:
        return out

    e = encoding_e(enc_metrics)
    if e is None:
        return out
    if auc_o is not None and excl is not None:
        scope = "full"
    elif _f(enc_metrics, "neutral_specificity", "specificity") is not None:
        scope = "vs_neutral_only"
    else:
        scope = "vs_neutral_auc_only"

    out["suitability_E"] = e
    out["suitability_scope"] = scope

    cau = dict(causal_metrics or {})
    if not _has_causal_payload(cau) and _has_causal_payload(enc_metrics):
        cau = dict(enc_metrics)
    if not _has_causal_payload(cau):
        out["suitability_reason"] = "no_causal"
        return out

    eff = _f(cau, "causal_signed_effect", "causal_delta_mean", "causal_delta_other")
    gate_empty = bool(cau.get("causal_gate_empty"))
    floor = bool(cau.get("causal_grid_floor"))
    null = bool(cau.get("causal_null_effect"))
    lms_ok = cau.get("causal_lms_ok")
    beats_random = cau.get("causal_beats_random_control")
    if null is False and eff is not None and abs(eff) < 1e-6:
        null = True
    floor_null = floor and null

    gate_reason: Optional[str] = None
    if gate_empty:
        gate_reason = "gate_empty"
    elif floor_null:
        gate_reason = "floor_null"
    elif eff is None:
        gate_reason = "no_effect"
    elif lms_ok is False:
        gate_reason = "lms_failed"
    elif beats_random is False:
        gate_reason = "random_control_failed"
    elif float(eff) < 0.0:
        gate_reason = "wrong_direction"
    elif not soft_causal and float(eff) < float(tau_c):
        gate_reason = "below_threshold"

    if gate_reason is not None:
        g = 0.0
    elif soft_causal:
        g = max(0.0, min(1.0, float(eff) / max(float(tau_c), 1e-12)))
    elif float(eff) >= float(tau_c):
        g = 1.0
    else:
        g = 0.0

    out["suitability_G"] = g
    out["suitability_gate_reason"] = gate_reason
    s = float(e) * float(g)
    out["suitability"] = s
    if g <= 0.0:
        out["suitability_reason"] = "G"
    elif e < float(e_ok):
        out["suitability_reason"] = "E"
    else:
        out["suitability_reason"] = "ok"
    return out


def _default_method_ids(
    results: Mapping[str, Any], *, primary_only: bool = False
) -> Sequence[str]:
    """Prefer schema ablation lists; fall back to encoder-metric rows (legacy dumps)."""
    from results_schema import (
        _has_encoder_metrics,
        encoder_ablation_methods,
        primary_fair_methods,
    )

    try:
        if primary_only:
            return primary_fair_methods(results)  # type: ignore[arg-type]
        return encoder_ablation_methods(results)  # type: ignore[arg-type]
    except (ValueError, TypeError, KeyError):
        ids: list[str] = []
        for r in results.get("methods") or []:
            mid = r.get("method")
            if not mid:
                continue
            if _has_encoder_metrics(r.get("metrics") or {}):
                ids.append(str(mid))
        return tuple(ids)


def attach_feature_suitability(
    results: Dict[str, Any],
    *,
    tau_c: float = DEFAULT_TAU_C,
    soft_causal: bool = False,
    e_ok: float = DEFAULT_E_OK,
    method_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Write suitability_* onto method metric dicts (in-place); return results."""
    from results_schema import resolve_encoder_metrics

    if method_ids is None:
        method_ids = _default_method_ids(results, primary_only=False)
    by_row = {
        str(r.get("method")): r
        for r in (results.get("methods") or [])
        if r.get("method")
    }
    summary_rows = []
    for mid in method_ids:
        enc, src, _inh = resolve_encoder_metrics(results, mid)
        cau, cau_src = resolve_causal_metrics(results, mid)
        payload = compute_feature_suitability(
            enc,
            cau,
            tau_c=tau_c,
            soft_causal=soft_causal,
            e_ok=e_ok,
        )
        # Prefer writing onto the encoder-owning row when present.
        target_id = src if src in by_row else mid
        row = by_row.get(target_id)
        if row is not None:
            metrics = dict(row.get("metrics") or {})
            metrics.update(payload)
            metrics["suitability_causal_source"] = cau_src if cau else None
            row["metrics"] = metrics
        summary_rows.append({"method": mid, **payload})

    results.setdefault("raw", {})["suitability"] = {
        "tau_c": float(tau_c),
        "soft_causal": bool(soft_causal),
        "e_ok": float(e_ok),
        "methods": summary_rows,
    }
    return results


def suitability_table(results: Dict[str, Any], *, primary_only: bool = True) -> str:
    """Console / TABLES block for S = E × G."""
    from results_schema import resolve_encoder_metrics, _fmt_metric

    methods = _default_method_ids(results, primary_only=primary_only)
    raw = (results.get("raw") or {}).get("suitability") or {}
    tau = raw.get("tau_c", DEFAULT_TAU_C)
    lines = [
        "",
        "Feature suitability  S = E * G_causal",
        "  E = min(auc_n, auc_o, excl)           # full (rival texts present)",
        "    | min(auc_n, spec)                  # one-pole / vs-neutral only",
        f"  G = 1 if intended eff>=tau_c={tau:g}, LMS passes, and random control is beaten",
        "      (missing legacy LMS/random flags are non-blocking); soft optional",
        "  reason: ok | E (encoding weak) | G (causal gate closed) | no_causal | no_encoding",
        f"{'method':<48} {'S':>7} {'E':>7} {'G':>5} {'scope':<18} {'reason':<12} {'gate detail'}",
        "-" * 122,
    ]
    by = {
        str(r.get("method")): (r.get("metrics") or {})
        for r in (results.get("methods") or [])
        if r.get("method")
    }
    for mid in methods:
        enc, src, _ = resolve_encoder_metrics(results, mid)
        m = by.get(src) or by.get(mid) or enc
        # Recompute if not attached yet
        if m.get("suitability_E") is None and m.get("suitability_reason") != "no_encoding":
            cau, _ = resolve_causal_metrics(results, mid)
            m = {**m, **compute_feature_suitability(enc, cau, tau_c=float(tau))}
        lines.append(
            f"{mid:<48} "
            f"{_fmt_metric(m.get('suitability'), width=7)} "
            f"{_fmt_metric(m.get('suitability_E'), width=7)} "
            f"{_fmt_metric(m.get('suitability_G'), width=5)} "
            f"{str(m.get('suitability_scope') or '—'):<18} "
            f"{str(m.get('suitability_reason') or '—'):<12} "
            f"{str(m.get('suitability_gate_reason') or '—')}"
        )
    return "\n".join(lines)


def _method_backend(method: str) -> str:
    return str(method).split(":", 1)[0]


def _nanmean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _iter_results_json(runs_root: Path) -> Iterable[Path]:
    runs_root = Path(runs_root)
    seen = set()
    for pattern in ("*/*/results.json", "*/results.json"):
        for path in sorted(runs_root.glob(pattern)):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


def _load_results(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    from causal_eval import apply_package_decoder_headlines

    return apply_package_decoder_headlines(payload)


def _suitability_params(results: Mapping[str, Any]) -> Tuple[float, bool, float]:
    raw = (results.get("raw") or {}).get("suitability") or {}
    cfg = (results.get("config") or {}).get("suitability") or {}
    tau = raw.get("tau_c", cfg.get("causal_effect_threshold", DEFAULT_TAU_C))
    soft = raw.get("soft_causal", cfg.get("soft_causal", False))
    e_ok = raw.get("e_ok", cfg.get("e_ok", DEFAULT_E_OK))
    return float(tau), bool(soft), float(e_ok)


def collect_suitability_method_rows(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    *,
    primary_only: bool = True,
) -> List[Dict[str, Any]]:
    """One row per (model, task, method) with S/E/G from each ``results.json``."""
    from results_schema import resolve_encoder_metrics

    rows: List[Dict[str, Any]] = []
    for path in _iter_results_json(Path(runs_root)):
        results = _load_results(path)
        if not results:
            continue
        parts = path.relative_to(Path(runs_root)).parts
        model = str(results.get("model") or (parts[0] if parts else "unknown"))
        if len(parts) >= 2:
            default_task = parts[1] if parts[-1] == "results.json" else parts[0]
        else:
            default_task = parts[0] if parts else "unknown"
        task = str(results.get("task") or default_task)
        tau, soft, e_ok = _suitability_params(results)
        try:
            method_ids = list(
                _default_method_ids(results, primary_only=primary_only)
            )
        except Exception:
            method_ids = list(_default_method_ids(results, primary_only=False))
        if not method_ids:
            continue
        by = {
            str(r.get("method")): (r.get("metrics") or {})
            for r in (results.get("methods") or [])
            if r.get("method")
        }
        for mid in method_ids:
            enc, src, _ = resolve_encoder_metrics(results, mid)
            m = dict(by.get(src) or by.get(mid) or enc or {})
            if m.get("suitability_E") is None and m.get("suitability_reason") != "no_encoding":
                cau, _ = resolve_causal_metrics(results, mid)
                m.update(
                    compute_feature_suitability(
                        enc or m,
                        cau,
                        tau_c=tau,
                        soft_causal=soft,
                        e_ok=e_ok,
                    )
                )
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "method": str(mid),
                    "backend": _method_backend(str(mid)),
                    "suitability": m.get("suitability"),
                    "suitability_E": m.get("suitability_E"),
                    "suitability_G": m.get("suitability_G"),
                    "suitability_scope": m.get("suitability_scope"),
                    "suitability_reason": m.get("suitability_reason"),
                    "suitability_gate_reason": m.get("suitability_gate_reason"),
                    "results_path": str(path),
                }
            )
    return rows


def model_task_cell_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "suitability",
    backend: Optional[str] = None,
) -> Dict[Tuple[str, str], Optional[float]]:
    """Mean ``value_key`` per (model, task), optionally restricted to one backend."""
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        if backend is not None and str(row.get("backend")) != str(backend):
            continue
        val = row.get(value_key)
        if not isinstance(val, (int, float)):
            continue
        key = (str(row["model"]), str(row["task"]))
        buckets.setdefault(key, []).append(float(val))
    return {k: (sum(vs) / len(vs) if vs else None) for k, vs in buckets.items()}


def matrix_with_mean_margins(
    cells: Mapping[Tuple[str, str], Optional[float]],
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Optional[float]]]:
    """Return (models, tasks, cells_with_mean_row_and_col).

    Adds a ``mean`` column (row-wise task mean) and ``mean`` row (column-wise
    model mean). Corner ``(mean, mean)`` is the grand mean over finite cells.
    """
    models = sorted({m for m, _ in cells})
    tasks = sorted({t for _, t in cells})
    out: Dict[Tuple[str, str], Optional[float]] = dict(cells)
    for model in models:
        out[(model, MEAN_LABEL)] = _nanmean(cells.get((model, t)) for t in tasks)
    for task in tasks:
        out[(MEAN_LABEL, task)] = _nanmean(cells.get((m, task)) for m in models)
    out[(MEAN_LABEL, MEAN_LABEL)] = _nanmean(
        cells.get((m, t)) for m in models for t in tasks
    )
    return models + [MEAN_LABEL], tasks + [MEAN_LABEL], out


def format_model_task_matrix(
    cells: Mapping[Tuple[str, str], Optional[float]],
    *,
    title: str = "Feature suitability S (mean over primary methods)",
    models: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
) -> str:
    """Plain-text model × task matrix with mean margins."""
    from results_schema import _fmt_metric

    if models is None or tasks is None:
        models, tasks, cells = matrix_with_mean_margins(cells)
    models = list(models)
    tasks = list(tasks)
    model_w = max(12, max((len(m) for m in models), default=5))
    col_w = 8
    header = f"{'model':<{model_w}}" + "".join(f"{t:>{col_w}}" for t in tasks)
    lines = ["", title, header, "-" * len(header)]
    for model in models:
        line = f"{model:<{model_w}}"
        for task in tasks:
            line += _fmt_metric(cells.get((model, task)), width=col_w)
        lines.append(line)
    return "\n".join(lines)


def _write_matrix_csv(
    path: Path,
    cells: Mapping[Tuple[str, str], Optional[float]],
    models: Sequence[str],
    tasks: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", *tasks])
        for model in models:
            row = [model]
            for task in tasks:
                val = cells.get((model, task))
                row.append("" if val is None else f"{float(val):.6f}")
            writer.writerow(row)


def _write_long_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "task",
        "method",
        "backend",
        "suitability",
        "suitability_E",
        "suitability_G",
        "suitability_scope",
        "suitability_reason",
        "suitability_gate_reason",
        "results_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def refresh_global_suitability(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    out_dir: Path = DEFAULT_GLOBAL_OUT,
    *,
    primary_only: bool = True,
) -> Dict[str, Any]:
    """Scan ``runs/``, write global S tables, return paths + printable text.

    Cheap (JSON parse only). Safe to call at the end of every study run.
    """
    runs_root = Path(runs_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_suitability_method_rows(runs_root, primary_only=primary_only)
    long_path = out_dir / "suitability_long.csv"
    _write_long_csv(long_path, rows)

    cells = model_task_cell_means(rows)
    models, tasks, margined = matrix_with_mean_margins(cells)
    matrix_csv = out_dir / "suitability_model_task.csv"
    matrix_txt = out_dir / "suitability_model_task.txt"
    _write_matrix_csv(matrix_csv, margined, models, tasks)
    overall_text = format_model_task_matrix(
        margined,
        title=(
            "Global feature suitability  S = E * G_causal  "
            f"(cell = mean over {'primary' if primary_only else 'all'} methods; "
            f"margins = {MEAN_LABEL})"
        ),
        models=models,
        tasks=tasks,
    )
    matrix_txt.write_text(overall_text + "\n", encoding="utf-8")

    backend_texts: List[str] = []
    backend_paths: Dict[str, str] = {}
    backends = sorted({str(r.get("backend")) for r in rows if r.get("backend")})
    for backend in backends:
        b_cells = model_task_cell_means(rows, backend=backend)
        if not b_cells:
            continue
        b_models, b_tasks, b_margined = matrix_with_mean_margins(b_cells)
        b_csv = out_dir / f"suitability_model_task_{backend}.csv"
        b_txt = out_dir / f"suitability_model_task_{backend}.txt"
        _write_matrix_csv(b_csv, b_margined, b_models, b_tasks)
        b_text = format_model_task_matrix(
            b_margined,
            title=f"Global suitability S — backend={backend} (mean over methods)",
            models=b_models,
            tasks=b_tasks,
        )
        b_txt.write_text(b_text + "\n", encoding="utf-8")
        backend_texts.append(b_text)
        backend_paths[backend] = str(b_csv)

    text = "\n".join([overall_text, *backend_texts])
    return {
        "text": text,
        "overall_text": overall_text,
        "n_method_rows": len(rows),
        "n_cells": len(cells),
        "paths": {
            "long_csv": str(long_path),
            "model_task_csv": str(matrix_csv),
            "model_task_txt": str(matrix_txt),
            "backends": backend_paths,
        },
        "cells": {f"{m}|{t}": v for (m, t), v in margined.items()},
    }
