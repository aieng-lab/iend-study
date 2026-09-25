"""Causal evaluation stage — canonical ``causal_study.run_study_causal``."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set

from causal_eval import DECODER_EVAL_MAX_SIZE, headline_metrics, persist_causal_summary
from results_schema import _has_encoder_metrics, method_result
from study.config import StudyConfig
from study.results_merge import causal_raw_entry_ok
from study.tasks import TaskBundle, claim_classes_for_study, encode_target_classes, labeled_df_for_eval


# Bump whenever causal results cannot safely be reused.  In particular, v2
# supplies a task-level decoder frame for one-pole features, so old rows may
# contain silently degraded / zero causal effects after missing rival panels.
# v3 (2026-08-20): introduced optional coarse-to-fine strength refinement.
# v4 (2026-09-02): every backend now selects and refines on validation, then
# evaluates the exact frozen validation candidate set once on test with
# refinement disabled. This replaces test-selected CAA parts and ACTIEND
# tensor/layer/ablation paths, adds missing frozen test reports for the default
# encoder paths, and reduces boundary refinement from ten points to five.
# v5: CAA now follows the same pair + one-pole regimes as IEND/CGA. Its method
# ids and directions changed, so legacy one-vs-rest-only CAA causal rows cannot
# be reused as the pair headline.
# v6: weakening is a separately selected package metric/direction, frozen on
# validation and reported on test; AGIEND uses the package activation API; and
# IEND random controls are genuine seeded, L2-matched random vectors rather
# than the opposite semantic pole. v5 rows are upgraded component-wise: their
# valid strengthen component is retained while weaken/random are repaired.
CAUSAL_PROTOCOL_VERSION = "decoder-bidirectional-val-select-test-v6-random"
STRENGTHEN_COMPATIBLE_CAUSAL_PROTOCOLS: Set[str] = {
    "decoder-val-select-test-report-v5-caa-poles"
}
FULLY_COMPATIBLE_CAUSAL_PROTOCOLS: Set[str] = set()

# Ridge-only method-id protocol. Version 2 retains all reloaded pole trainers
# and their pair metadata. Version 1 refreshes could write a pair checkpoint's
# result under a one-pole ``actiend_ridge:{class}:...`` id. This deliberately
# does not bump CAUSAL_PROTOCOL_VERSION: unrelated causal families stay cached.
ACTIEND_RIDGE_ID_PROTOCOL_VERSION = 2

# Version 2 fixes the selected causal direction/objective for activation-gradient
# AGIEND, CGA's fixed factual/diff estimator orientation, and CAGA's direct
# sweep (whose stored signed effect was read from the factual instead of rival
# panel). Keep this family-scoped so valid SAE, CAA, ACTIEND, and GRADIEND
# sweeps remain reusable.
DIRECTION_POLARITY_PROTOCOL_VERSION = 2
DIRECTION_POLARITY_METHOD_PREFIXES = (
    "agiend:",
    "caga:",
    "cga:",
    "cga_tensor_norm:",
)

# Version 2 invalidates CGA causal grids that survived replacement of the
# fitted CGA artifact during the one-pole data-protocol migration. Those grids
# are not eligible for the polarity shortcut: validation curves from one
# fitted direction cannot select a test candidate for another direction. Keep
# this CGA-scoped because only CGA has demonstrated a numerically bad selection.
CGA_CAUSAL_GRID_PROTOCOL_VERSION = 2
CGA_CAUSAL_GRID_METHOD_PREFIXES = ("cga:", "cga_tensor_norm:")

# AGIEND decoder grids are fitted-artifact specific for the same reason as CGA.
# Several Pythia one-pole artifacts were retrained on 2026-09-13/14 while their
# older decoder grids survived, so polarity-only reselection is insufficient.
AGIEND_CAUSAL_GRID_PROTOCOL_VERSION = 2
AGIEND_CAUSAL_GRID_METHOD_PREFIXES = ("agiend:",)


def _uses_direction_polarity_protocol(method_id: str) -> bool:
    return str(method_id).startswith(DIRECTION_POLARITY_METHOD_PREFIXES)


def invalid_direction_polarity_method_ids(
    method_ids: Set[str],
    previous_version: Any,
    previous_by_method: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    """Completed causal ids that must be recomputed after the sign fix."""
    # A task-level marker is insufficient after method-isolated merges: one
    # family can update it while stale rows from another family survive.
    if (
        previous_by_method is None
        and int(previous_version or 1) >= DIRECTION_POLARITY_PROTOCOL_VERSION
    ):
        return set()
    invalid: Set[str] = set()
    for mid in method_ids:
        mid = str(mid)
        if not _uses_direction_polarity_protocol(mid):
            continue
        entry = (previous_by_method or {}).get(mid) or {}
        meta = entry.get("meta") or {} if isinstance(entry, dict) else {}
        entry_valid = bool(
            int(meta.get("direction_polarity_protocol_version") or 1)
            >= DIRECTION_POLARITY_PROTOCOL_VERSION
            or meta.get("polarity_reselected_from_bidirectional_grid")
        )
        if not entry_valid:
            invalid.add(mid)
    return invalid


def invalid_cga_causal_grid_method_ids(
    method_ids: Set[str], previous_version: Any
) -> Set[str]:
    """Completed CGA ids whose validation grid may predate their train artifact."""
    if int(previous_version or 1) >= CGA_CAUSAL_GRID_PROTOCOL_VERSION:
        return set()
    return {
        str(mid)
        for mid in method_ids
        if str(mid).startswith(CGA_CAUSAL_GRID_METHOD_PREFIXES)
    }


def invalid_agiend_causal_grid_method_ids(
    method_ids: Set[str], previous_version: Any
) -> Set[str]:
    """Completed AGIEND ids whose grid may predate their fitted artifact."""
    if int(previous_version or 1) >= AGIEND_CAUSAL_GRID_PROTOCOL_VERSION:
        return set()
    return {
        str(mid)
        for mid in method_ids
        if str(mid).startswith(AGIEND_CAUSAL_GRID_METHOD_PREFIXES)
    }


def reusable_strengthen_entries(
    previous_causal: Dict[str, Any], *, skip_existing: bool
) -> Dict[str, Dict[str, Any]]:
    """Return only frozen strengthen components safe to carry from v5 to v6."""
    if not skip_existing:
        return {}
    if str(previous_causal.get("protocol_version") or "") not in (
        STRENGTHEN_COMPATIBLE_CAUSAL_PROTOCOLS
    ):
        return {}
    ridge_ids_valid = (
        int(previous_causal.get("actiend_ridge_id_protocol_version") or 1)
        >= ACTIEND_RIDGE_ID_PROTOCOL_VERSION
    )
    reusable: Dict[str, Dict[str, Any]] = {}
    for mid, entry in (previous_causal.get("by_method") or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("selected") is None or not entry.get("strengths"):
            continue
        if (entry.get("meta") or {}).get("error"):
            continue
        mid_s = str(mid)
        if mid_s.startswith("actiend_ridge:") and not ridge_ids_valid:
            continue
        reusable[mid_s] = entry
    return reusable


def stamp_progress_entry(entry: Dict[str, Any], progress: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a ``progress-*.json`` entry and stamp it with its file's protocol versions.

    Progress checkpoints are written *before* the post-run provenance stamping,
    so their per-entry ``meta`` lacks the ``*_protocol_version`` markers even
    though the file itself records them. Without the stamp a valid recovered
    entry later looks pre-fix and is queued for repair instead of being kept.
    """
    out = dict(entry)
    meta = dict(out.get("meta") or {})
    for key in (
        "direction_polarity_protocol_version",
        "cga_causal_grid_protocol_version",
        "agiend_causal_grid_protocol_version",
    ):
        if key not in meta and progress.get(key) is not None:
            meta[key] = progress[key]
    out["meta"] = meta
    return out


def orphaned_skip_ids(
    skip_ids: Set[str],
    *,
    progress_ids: Set[str],
    previous_ids: Set[str],
    invalidated_ids: Set[str],
) -> Set[str]:
    """Skipped ids that would have no entry to restore afterwards.

    A method id in the skip set is neither recomputed nor, unless an entry for
    it is restored, written back -- it silently vanishes from ``results.json``
    (AGIEND/CAGA cells lost this way on Llama-3.1-8B). An id may only be skipped
    when a valid stored entry exists to restore: ids invalidated for
    repair/recompute are never restored, and an id present only in a progress
    file that was not recovered has nothing to restore either.
    """
    restorable = (set(progress_ids) | set(previous_ids)) - set(invalidated_ids)
    return set(skip_ids) - restorable


def previous_causal_entries(previous: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Recover completed causal bundles from raw storage and method-row extras.

    Method-isolated merges can leave ``raw.causal`` containing only the most
    recently run family, while every merged method row still carries its full
    causal bundle. Polarity repair needs those persisted +/- validation grids.
    """
    payload = previous or {}
    raw = ((payload.get("raw") or {}).get("causal") or {}).get("by_method") or {}
    out = {
        str(mid): dict(entry)
        for mid, entry in raw.items()
        if isinstance(entry, dict) and causal_raw_entry_ok(entry)
    }
    for row in payload.get("methods") or []:
        if not isinstance(row, dict):
            continue
        entry = (row.get("extras") or {}).get("causal")
        # Proxy encode rows may carry the base method's causal bundle in their
        # extras. Key that bundle by its own causal method id, not by the proxy
        # alias, or refresh would invent duplicate causal sweeps.
        mid = (entry or {}).get("method") or row.get("method")
        if mid and causal_raw_entry_ok(entry):
            out.setdefault(str(mid), dict(entry))
    return out


def _merged_causal_summary_entries(
    serial_by: Dict[str, Any],
    *,
    current: Optional[Sequence[Dict[str, Any]]] = None,
    previous: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build one complete, deterministic summary row per merged method id."""
    summaries: Dict[str, Dict[str, Any]] = {}
    for collection in (previous or [], current or []):
        for source in collection:
            if not isinstance(source, dict):
                continue
            mid = source.get("method")
            if mid:
                summaries[str(mid)] = dict(source)

    merged: List[Dict[str, Any]] = []
    for mid in sorted(str(key) for key in serial_by):
        raw = serial_by.get(mid)
        entry = dict(summaries.get(mid) or (raw if isinstance(raw, dict) else {}))
        entry["method"] = mid
        merged.append(entry)
    return merged


def _skip_causal_shell(mid: str, err: Any, *, claim_classes: Sequence[str]) -> bool:
    """Drop SAE shells for CF poles and empty-feature skips (not real errors)."""
    if err in ("no feature", "no layer features"):
        return True
    parts = str(mid).split(":")
    if len(parts) >= 2 and parts[0] in {"sae", "sae_pre"}:
        if parts[1] == "joint":
            pole = parts[2] if len(parts) >= 3 else ""
        else:
            pole = parts[1]
        if pole and pole not in {str(c) for c in claim_classes}:
            return True
    return False


def _is_keyerror_message(err: Any) -> bool:
    """True for ``str(KeyError('english'))`` → ``'english'`` and ``KeyError(...)``."""
    s = str(err).strip()
    if not s:
        return False
    if s.startswith("KeyError"):
        return True
    return (
        len(s) >= 3
        and s[0] in {"'", '"'}
        and s[-1] == s[0]
        and " " not in s[1:-1]
    )


def clear_stale_causal_error_on_encode_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Drop leftover causal ``KeyError(class)`` flags from reused encode rows.

    REFRESH_CAUSAL copies prior SAE rows including ``status=error`` /
    ``error='english'`` even when encoder AUCs are present. Causal merge used
    to copy error one-way and never clear success.
    """
    out = dict(row)
    metrics = dict(out.get("metrics") or {})
    err = out.get("error") or metrics.get("causal_error")
    if not _has_encoder_metrics(metrics):
        return out
    if not err and out.get("status") != "error":
        return out
    if err and not _is_keyerror_message(err):
        return out
    out["status"] = "ok"
    out.pop("error", None)
    metrics.pop("causal_error", None)
    out["metrics"] = metrics
    return out


def apply_causal_crow_to_row(row: Dict[str, Any], crow: Dict[str, Any]) -> None:
    """Merge a causal method row onto an existing encode row (in-place).

    Successful causal clears a stale encode-row error. Causal failure with
    encoder AUCs is ``partial`` (not ``error``), so the summary table still
    shows separability metrics.
    """
    row.setdefault("metrics", {}).update(crow.get("metrics") or {})
    extras = row.setdefault("extras", {})
    for k, v in (crow.get("extras") or {}).items():
        extras[k] = v
    arts = row.setdefault("artifacts", {})
    for k, v in (crow.get("artifacts") or {}).items():
        arts[k] = v
    err = crow.get("error")
    if err:
        row["error"] = err
        row.setdefault("metrics", {})["causal_error"] = err
        if _has_encoder_metrics(row.get("metrics") or {}):
            row["status"] = "partial"
        else:
            row["status"] = crow.get("status") or "error"
    else:
        row["status"] = crow.get("status") or "ok"
        row.pop("error", None)
        (row.get("metrics") or {}).pop("causal_error", None)


def _attach_causal_to_rows(
    method_rows: List[Dict[str, Any]],
    causal_raw: Dict[str, Any],
    *,
    model_key: str,
    task_id: str,
    claim_classes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Merge LMS-gated causal headlines onto shared method ids."""
    by_method = causal_raw.get("by_method") or {}
    summaries = {
        str(s.get("method")): s
        for s in (causal_raw.get("summaries") or [])
        if isinstance(s, dict) and s.get("method")
    }
    existing = {str(r.get("method")): r for r in method_rows if r.get("method")}

    for mid, cres in by_method.items():
        mid = str(mid)
        metrics: Dict[str, Any] = {}
        err = None
        target = None
        mod_path = None
        cdict: Dict[str, Any] = {}
        try:
            if hasattr(cres, "meta"):
                metrics = headline_metrics(cres)
                err = (cres.meta or {}).get("error")
                target = cres.target_class
                mod_path = getattr(cres, "modified_model_path", None)
                cdict = cres.to_dict() if hasattr(cres, "to_dict") else {}
            else:
                summ = summaries.get(mid) or (cres if isinstance(cres, dict) else {})
                metrics = dict(summ.get("headline_metrics") or {})
                err = (summ.get("meta") or {}).get("error") if isinstance(summ, dict) else None
                target = summ.get("target_class") if isinstance(summ, dict) else None
                mod_path = summ.get("modified_model_path") if isinstance(summ, dict) else None
                cdict = summ if isinstance(summ, dict) else {}
        except Exception as exc:
            err = str(exc)
            if hasattr(cres, "to_dict"):
                cdict = cres.to_dict()
            elif isinstance(cres, dict):
                cdict = cres
            target = getattr(cres, "target_class", None) or (
                cdict.get("target_class") if isinstance(cdict, dict) else None
            )
            mod_path = getattr(cres, "modified_model_path", None) or (
                cdict.get("modified_model_path") if isinstance(cdict, dict) else None
            )

        if claim_classes and _skip_causal_shell(mid, err, claim_classes=claim_classes):
            continue

        causal_meta = dict(cdict.get("meta") or {}) if isinstance(cdict, dict) else {}
        provenance_metrics = {
            key: causal_meta.get(key)
            for key in ("ablation", "pair", "vector_key")
            if causal_meta.get(key) is not None
        }

        row = existing.get(mid)
        if row is None:
            row = method_result(
                method=mid,
                model=model_key,
                task=task_id,
                status="error" if err else "ok",
                error=err,
                metrics={
                    **metrics,
                    **provenance_metrics,
                    "target_class": target,
                    "readout_kind": "causal_only",
                    **({"causal_error": err} if err else {}),
                },
                extras={"causal_ablation": True},
            )
            method_rows.append(row)
            existing[mid] = row
        else:
            apply_causal_crow_to_row(
                row,
                {
                    "status": "error" if err else "ok",
                    "error": err,
                    "metrics": {
                        **metrics,
                        **provenance_metrics,
                        "target_class": target,
                        **({"causal_error": err} if err else {}),
                    },
                    "extras": {},
                    "artifacts": {},
                },
            )

        if mod_path:
            row.setdefault("artifacts", {})["modified_model_path"] = mod_path
        slim_strengths = []
        for sr in cdict.get("strengths") or []:
            if isinstance(sr, dict):
                slim_strengths.append({k: v for k, v in sr.items() if k != "samples"})
        row.setdefault("extras", {})["causal"] = {
            k: v for k, v in cdict.items() if k != "strengths"
        }
        row["extras"]["causal"]["strengths"] = slim_strengths
        row["extras"]["causal_lms_curve"] = metrics.get("causal_lms_curve") or []
    return method_rows




def build_causal_study_kwargs(
    cfg: StudyConfig,
    *,
    target_classes: Sequence[str],
    enabled: Set[str],
    skip_causal_methods: Optional[Set[str]] = None,
    reuse_strengthen_by_method: Optional[Dict[str, Dict[str, Any]]] = None,
    polarity_repair_by_method: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Map study config → ``causal_study.run_study_causal`` keyword args."""
    from caa_eval import CAA_ACT_POLICIES
    from study.causal_policies import normalize_actiend_causal_policies
    from study.stages.sae import sae_suite_params

    causal_cfg = cfg.raw.get("causal") or {}
    # ``--causal-only``: execution filter (not hashed) naming the only ids to compute.
    from study.causal_policies import causal_only_patterns

    only_ids = causal_only_patterns(
        (cfg.raw.get("cli") or {}).get("causal_only"),
        results_path=cfg.output_dir / "results.json",
    )
    actiend_policies = normalize_actiend_causal_policies(
        causal_cfg.get("actiend_policies")
        if "actiend_policies" in causal_cfg
        else None
    )
    n_per = int(causal_cfg.get("n_per_group") or 100)
    # Validation-only cap for the strength sweep (selection); test report stays at
    # n_per_group. None => no cap (legacy behavior). See causal_study.run_study_causal.
    val_n_per = causal_cfg.get("val_n_per_group")
    val_n_per = int(val_n_per) if val_n_per else None
    max_size = int(causal_cfg.get("decoder_max_size") or DECODER_EVAL_MAX_SIZE)
    # Validation-only decoder-frame cap for GRADIEND/ACTIEND LR selection (test
    # report stays at decoder_max_size). None => no cap.
    val_max_size = causal_cfg.get("val_decoder_max_size")
    val_max_size = int(val_max_size) if val_max_size else None
    strengths = causal_cfg.get("strengths")
    # Deliberately NOT derived from n_per_group/decoder_max_size (those are tuned
    # per task for the strengthen/LMS grid sweep, not for how much data a one-shot
    # closed-form ridge fit needs) -- see causal_study.py's ACTIEND_RIDGE_FIT_*.
    actiend_ridge_max_size_per_group = int(
        causal_cfg.get("actiend_ridge_max_size_per_group") or 1000
    )
    actiend_ridge_max_samples = int(causal_cfg.get("actiend_ridge_max_samples") or 3000)

    if cfg.raw.get("_smoke"):
        n_per = min(n_per, 4)
        max_size = min(max_size, 32)
        strengths = strengths or [0.1, 1.0, 10.0, 100.0]
        actiend_ridge_max_size_per_group = min(actiend_ridge_max_size_per_group, 32)
        actiend_ridge_max_samples = min(actiend_ridge_max_samples, 64)

    suite_methods = cfg.suite.get("methods") or {}
    cga_methods = suite_methods.get("cga") or []
    caga_methods = suite_methods.get("caga") or []
    if isinstance(cga_methods, str):
        cga_methods = [cga_methods]
    if isinstance(caga_methods, str):
        caga_methods = [caga_methods]
    # ``layers`` always materializes the candidate readouts. In core,
    # ``primary_only`` means validation selects one aggregate-or-layer causal
    # representation; full suites retain every candidate as an ablation.
    layerwise_cga = "layers" in {str(x).strip().lower() for x in cga_methods}
    layerwise_caga = "layers" in {str(x).strip().lower() for x in caga_methods}
    suite_caa = suite_methods.get("caa") or []
    policies = list(CAA_ACT_POLICIES)
    if suite_caa:
        suite_pols = []
        for p in suite_caa:
            s = str(p)
            suite_pols.append(s[4:] if s.startswith("act_") else s)
        policies = [p for p in policies if p in suite_pols] or list(CAA_ACT_POLICIES)

    sae_params = sae_suite_params(suite_methods.get("sae") or [])
    if sae_params["tags"]:
        print(
            f"causal: sae suite tags={sorted(sae_params['tags'])} "
            f"fixed_ks={list(sae_params['fixed_ks'])} "
            f"opp_fire={sae_params['opp_fire']} arad={sae_params['arad']} "
            f"jh_f1={sae_params['jh_f1']} clamp={sae_params['clamp']}",
            flush=True,
        )

    model_cfg = cfg.model or {}
    sae_layers = tuple(
        int(x) for x in (model_cfg.get("sae_layers") or model_cfg.get("layers") or ())
    )
    if cfg.raw.get("_smoke"):
        from study.training_profiles import resolve_study_layers

        sae_layers = tuple(resolve_study_layers(model_cfg, smoke=True))
    n_layers = model_cfg.get("n_layers")
    if n_layers is None and sae_layers:
        n_layers = int(max(sae_layers)) + 1

    cli = cfg.raw.get("cli") or {}
    max_samples = causal_cfg.get("persist_max_samples")

    kw: Dict[str, Any] = {
        "target_classes": target_classes,
        "output_dir": cfg.output_dir,
        "study_model_key": cfg.model_key,
        "enabled": enabled,
        "sae_release": model_cfg.get("sae_release"),
        "sae_layers": sae_layers,
        "sae_fixed_ks": sae_params["fixed_ks"],
        "sae_opp_fire_enabled": sae_params["opp_fire"],
        "sae_arad_enabled": sae_params["arad"],
        "sae_jh_f1_enabled": sae_params["jh_f1"],
        "sae_clamp_enabled": sae_params["clamp"],
        # kstar is a suite tag like k1; an untagged suite keeps the legacy "emit it".
        "sae_kstar_enabled": (not sae_params["tags"]) or "kstar" in sae_params["tags"],
        "n_per_group": n_per,
        "val_n_per_group": val_n_per,
        "decoder_max_size": max_size,
        "val_decoder_max_size": val_max_size,
        "actiend_ridge_max_size_per_group": actiend_ridge_max_size_per_group,
        "actiend_ridge_max_samples": actiend_ridge_max_samples,
        "save_modified_models": bool(cli.get("save_modified_models")),
        "n_model_layers": n_layers,
        "persist_curve_samples": bool(causal_cfg.get("persist_curve_samples", False)),
        "persist_sample_text": bool(causal_cfg.get("persist_sample_text", False)),
        "persist_max_samples": (
            None if max_samples is None else int(max_samples)
        ),
        "caa_causal_policies": policies,
        "actiend_causal_policies": actiend_policies,
        # Core is the headline protocol: retain only validation-selected causal
        # rows. Full suites deliberately keep the layerwise appendix sweeps.
        "primary_only": bool(causal_cfg.get("primary_only", False)) and not only_ids,
        "only_method_id_patterns": only_ids or None,
        "skip_causal_methods": skip_causal_methods,
        "reuse_strengthen_by_method": reuse_strengthen_by_method,
        "polarity_repair_by_method": polarity_repair_by_method,
        "causal_protocol_version": CAUSAL_PROTOCOL_VERSION,
        "direction_polarity_protocol_version": DIRECTION_POLARITY_PROTOCOL_VERSION,
        "cga_causal_grid_protocol_version": CGA_CAUSAL_GRID_PROTOCOL_VERSION,
        "agiend_causal_grid_protocol_version": AGIEND_CAUSAL_GRID_PROTOCOL_VERSION,
        "layerwise_cga": layerwise_cga,
        "layerwise_caga": layerwise_caga,
    }
    if strengths is not None:
        kw["causal_strengths"] = list(strengths)
        kw["sae_causal_strengths"] = tuple(strengths)
        kw["caa_causal_strengths"] = tuple(strengths)
        kw["actiend_decoder_lrs"] = list(strengths)
    return kw


def run_causal_stage(
    cfg: StudyConfig,
    bundle: TaskBundle,
    train_raw_none: Dict[str, Dict[str, Any]],
    *,
    train_raw_tensors: Optional[Dict[str, Dict[str, Any]]] = None,
    train_raw_none_by_class: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    train_raw_tensors_by_class: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    sae_raw: Dict[str, Any] | None = None,
    caa_raw: Dict[str, Any] | None = None,
    enabled: Optional[Set[str]] = None,
    previous: Optional[Dict[str, Any]] = None,
    skip_existing: bool = False,
    force_causal: bool = False,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Full causal: GRADIEND/ACTIEND tok+split+L*, SAE k*/all_k/sel_*, CAA steer."""
    from causal_study import run_study_causal

    from study.training_profiles import expand_cga_backends

    enabled_set = expand_cga_backends(
        enabled if enabled is not None
        else {"gradiend", "actiend", "sae", "caa", "cga", "caga", "causal"}
    )
    # Pass every reloaded trainer through. run_study_causal filters intervention
    # loops by --methods; SAE/CAA still need a live HF model when METHODS=sae.
    train_none = dict(train_raw_none or {})
    train_tensors = dict(train_raw_tensors or {})
    none_by_class = None
    if train_raw_none_by_class:
        none_by_class = {
            str(b): dict(m)
            for b, m in train_raw_none_by_class.items()
            if m
        } or None
    tensors_by_class = None
    if train_raw_tensors_by_class:
        tensors_by_class = {
            str(b): dict(m)
            for b, m in train_raw_tensors_by_class.items()
            if m
        } or None
    causal_backends = sorted(k for k in train_none if str(k) in enabled_set)
    causal_tensors = sorted(k for k in train_tensors if str(k) in enabled_set)
    sae_payload = (sae_raw or {}) if "sae" in enabled_set else {}
    caa_payload = (caa_raw or {}) if "caa" in enabled_set else {}

    # Expand one-pole CF rows when alternative_* is present; target_classes =
    # labels actually on the eval frame (do not invent missing CF poles).
    eval_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    classes = encode_target_classes(bundle, eval_df)
    if not classes:
        classes = [str(c) for c in bundle.classes]
    abl = dict(cfg.raw.get("ablations") or {})
    one_pole = bool(abl.get("one_pole")) and abl.get("pair") is False
    opc = cfg.training.get("one_pole_classes")
    claim_classes = claim_classes_for_study(
        bundle,
        one_pole=one_pole,
        one_pole_classes=opc if isinstance(opc, (list, tuple)) else None,
    )
    if classes != [str(c) for c in bundle.classes]:
        print(
            f"causal: target_classes {list(bundle.classes)} → {classes} "
            f"(classes present after one-pole expand)",
            flush=True,
        )

    errors: List[str] = []
    # Force mode is deliberately narrower than SKIP_EXISTING=0: retain and
    # reload fitted artifacts, but discard every selected prior causal sweep.
    # This is the operator escape hatch when a protocol marker did not change
    # yet the stored intervention grid is known to be stale.
    prev_causal = {} if force_causal else (((previous or {}).get("raw") or {}).get("causal") or {})
    prev_by = {} if force_causal else previous_causal_entries(previous)
    progress_by: Dict[str, Any] = {}
    progress_dir = cfg.output_dir / "causal"
    progress_paths = list(progress_dir.glob("progress*.json")) if skip_existing and not force_causal else []
    if force_causal:
        print(
            "causal: force-causal discarding completed selected sweeps; "
            "re-evaluating validation grids and frozen tests without retraining",
            flush=True,
        )
    for progress_path in progress_paths:
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("protocol_version") == CAUSAL_PROTOCOL_VERSION:
                progress_polarity_valid = (
                    int(progress.get("direction_polarity_protocol_version") or 1)
                    >= DIRECTION_POLARITY_PROTOCOL_VERSION
                )
                progress_cga_grid_valid = (
                    int(progress.get("cga_causal_grid_protocol_version") or 1)
                    >= CGA_CAUSAL_GRID_PROTOCOL_VERSION
                )
                progress_agiend_grid_valid = (
                    int(progress.get("agiend_causal_grid_protocol_version") or 1)
                    >= AGIEND_CAUSAL_GRID_PROTOCOL_VERSION
                )
                progress_by.update(
                    {
                        str(mid): stamp_progress_entry(entry, progress)
                        for mid, entry in (progress.get("by_method") or {}).items()
                        if causal_raw_entry_ok(entry)
                        and (
                            progress_polarity_valid
                            or not _uses_direction_polarity_protocol(str(mid))
                        )
                        and (
                            progress_cga_grid_valid
                            or not str(mid).startswith(CGA_CAUSAL_GRID_METHOD_PREFIXES)
                        )
                        and (
                            progress_agiend_grid_valid
                            or not str(mid).startswith(AGIEND_CAUSAL_GRID_METHOD_PREFIXES)
                        )
                    }
                )
        except Exception as exc:
            print(
                f"causal: ignoring unreadable progress checkpoint "
                f"{progress_path.name}: {exc}",
                flush=True,
            )
    if progress_by:
        print(
            f"causal: recovered {len(progress_by)} completed method sweeps "
            f"from {len(progress_paths)} timeout checkpoint file(s)",
            flush=True,
        )
    # ``--causal-only`` adds a few ids and must never touch a stored one: no
    # protocol migration, invalidation or repair of anything already computed.
    causal_only = bool((cfg.raw.get("cli") or {}).get("causal_only"))
    reuse_completed_causal = skip_existing and (
        causal_only
        or str(prev_causal.get("protocol_version") or "")
        in {CAUSAL_PROTOCOL_VERSION, *FULLY_COMPATIBLE_CAUSAL_PROTOCOLS}
    )
    skip_causal_methods = (
        set(prev_by) if skip_existing and prev_by else set()
    )
    reuse_strengthen_by_method = (
        {}
        if causal_only
        else reusable_strengthen_entries(prev_causal, skip_existing=skip_existing)
    )
    invalidated_ridge_ids: Set[str] = set()
    invalidated_direction_ids: Set[str] = set()
    invalidated_cga_grid_ids: Set[str] = set()
    invalidated_agiend_grid_ids: Set[str] = set()
    polarity_repair_by_method: Dict[str, Dict[str, Any]] = {}
    if not reuse_completed_causal and (
        skip_causal_methods or reuse_strengthen_by_method
    ):
        if reuse_strengthen_by_method:
            print(
                "causal: protocol changed; repairing weaken/random components "
                f"while reusing {len(reuse_strengthen_by_method)} frozen v5 "
                "strengthen results",
                flush=True,
            )
        else:
            print(
                "causal: protocol changed; re-running all causal sweeps "
                "without retraining",
                flush=True,
            )
        skip_causal_methods = set()
    elif (
        not causal_only
        and skip_causal_methods
        and "actiend_ridge" in enabled_set
        and int(prev_causal.get("actiend_ridge_id_protocol_version") or 1)
        < ACTIEND_RIDGE_ID_PROTOCOL_VERSION
    ):
        invalidated_ridge_ids = {
            str(mid)
            for mid in skip_causal_methods
            if str(mid).startswith("actiend_ridge:")
        }
        skip_causal_methods -= invalidated_ridge_ids
        print(
            "causal: ridge method-id protocol changed; re-running only "
            f"{len(invalidated_ridge_ids)} completed ACTIEND-ridge sweeps",
            flush=True,
        )
    if skip_existing and not causal_only:
        invalidated_direction_ids = invalid_direction_polarity_method_ids(
            skip_causal_methods,
            prev_causal.get("direction_polarity_protocol_version"),
            prev_by,
        )
        skip_causal_methods -= invalidated_direction_ids
        polarity_repair_by_method = {
            mid: prev_by[mid]
            for mid in invalidated_direction_ids
            if mid in prev_by
        }
        if invalidated_direction_ids:
            print(
                "causal: direction/objective protocol changed; repairing only "
                f"{len(invalidated_direction_ids)} completed AGIEND/CGA/CAGA sweeps",
                flush=True,
            )
    if (
        skip_existing
        and not causal_only
        and int(prev_causal.get("agiend_causal_grid_protocol_version") or 1)
        < AGIEND_CAUSAL_GRID_PROTOCOL_VERSION
    ):
        invalidated_agiend_grid_ids = invalid_agiend_causal_grid_method_ids(
            set(prev_by),
            prev_causal.get("agiend_causal_grid_protocol_version"),
        )
        skip_causal_methods -= invalidated_agiend_grid_ids
        for mid in invalidated_agiend_grid_ids:
            polarity_repair_by_method.pop(mid, None)
            reuse_strengthen_by_method.pop(mid, None)
        if invalidated_agiend_grid_ids:
            print(
                "causal: AGIEND train/grid provenance changed; re-running fresh "
                f"validation + frozen-test grids for {len(invalidated_agiend_grid_ids)} "
                "completed AGIEND sweeps (no retraining)",
                flush=True,
            )
    if (
        skip_existing
        and not causal_only
        and int(prev_causal.get("cga_causal_grid_protocol_version") or 1)
        < CGA_CAUSAL_GRID_PROTOCOL_VERSION
    ):
        invalidated_cga_grid_ids = invalid_cga_causal_grid_method_ids(
            set(prev_by),
            prev_causal.get("cga_causal_grid_protocol_version"),
        )
        skip_causal_methods -= invalidated_cga_grid_ids
        # A direction-protocol migration may already have queued these bundles
        # for its cheap curve-swap repair. These methods instead need a fresh
        # validation grid against the currently loaded train artifact.
        for mid in invalidated_cga_grid_ids:
            polarity_repair_by_method.pop(mid, None)
            reuse_strengthen_by_method.pop(mid, None)
        if invalidated_cga_grid_ids:
            print(
                "causal: CGA train/grid provenance changed; re-running fresh "
                f"validation + frozen-test grids for {len(invalidated_cga_grid_ids)} "
                "completed CGA sweeps (no retraining)",
                flush=True,
            )
    # Current-protocol progress is independently safe even when results.json
    # still contains a v5 bundle whose strengthen component is being repaired.
    skip_causal_methods.update(progress_by)
    # Never skip an id that will not be restored (see orphaned_skip_ids): the
    # progress update above can re-add ids that were just invalidated, and the
    # restore loop below deliberately skips those, so they would be dropped.
    _orphans = orphaned_skip_ids(
        skip_causal_methods,
        progress_ids=set(progress_by),
        previous_ids=set(prev_by) if reuse_completed_causal else set(),
        invalidated_ids=(
            invalidated_ridge_ids
            | invalidated_direction_ids
            | invalidated_cga_grid_ids
            | invalidated_agiend_grid_ids
        ),
    )
    if _orphans:
        print(
            f"causal: {len(_orphans)} skipped sweep id(s) have no restorable entry "
            f"(invalidated or unrecovered); recomputing instead: {sorted(_orphans)[:8]}",
            flush=True,
        )
        skip_causal_methods -= _orphans
    if skip_causal_methods:
        print(
            f"causal: skip-existing will keep {len(skip_causal_methods)} completed "
            f"method sweeps (re-run only failures)",
            flush=True,
        )

    causal_kwargs = build_causal_study_kwargs(
        cfg,
        target_classes=classes,
        enabled=enabled_set,
        skip_causal_methods=skip_causal_methods or None,
        reuse_strengthen_by_method=reuse_strengthen_by_method or None,
        polarity_repair_by_method=polarity_repair_by_method or None,
    )

    try:
        print(
            "causal: invoking causal_study.run_study_causal "
            f"(backends={causal_backends}; tensors={causal_tensors}; "
            f"backbone={sorted(train_none)}; "
            f"none_by_class="
            f"{ {b: sorted(m) for b, m in (none_by_class or {}).items()} }; "
            f"tensors_by_class="
            f"{ {b: sorted(m) for b, m in (tensors_by_class or {}).items()} }; "
            f"sae={'yes' if sae_payload and not sae_payload.get('error') else 'no'}; "
            f"caa={'yes' if caa_payload.get('vectors') else 'no'}; "
            f"classes={list(classes)})",
            flush=True,
        )
        causal_raw = run_study_causal(
            train_none,
            sae_payload,
            eval_df,
            bundle.neutrals,
            train_raw_tensors=train_tensors,
            train_raw_none_by_class=none_by_class,
            train_raw_tensors_by_class=tensors_by_class,
            caa_raw=caa_payload,
            fail_fast=fail_fast,
            **causal_kwargs,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        if fail_fast:
            raise
        errors.append(f"causal: {exc}")
        return {
            "methods": [
                method_result(
                    method="causal",
                    model=cfg.model_key,
                    task=cfg.task_id,
                    status="error",
                    error=str(exc),
                )
            ],
            "raw": {"error": str(exc)},
            "errors": errors,
        }

    by_method = causal_raw.get("by_method") or {}
    serial_by = {}
    for mid, cres in by_method.items():
        if hasattr(cres, "to_dict"):
            serial_by[str(mid)] = cres.to_dict()
        else:
            serial_by[str(mid)] = cres
    # Persist provenance on each newly evaluated bundle, not only on the
    # task-level causal block. Method-isolated result merges may retain rows
    # from other families, so analysis must not trust a global marker.
    for mid, entry in serial_by.items():
        if not _uses_direction_polarity_protocol(str(mid)):
            continue
        if not isinstance(entry, dict):
            continue
        entry_meta = dict(entry.get("meta") or {})
        entry_meta["direction_polarity_protocol_version"] = (
            DIRECTION_POLARITY_PROTOCOL_VERSION
        )
        entry["meta"] = entry_meta
    for mid, entry in serial_by.items():
        if not str(mid).startswith(CGA_CAUSAL_GRID_METHOD_PREFIXES):
            continue
        if not isinstance(entry, dict):
            continue
        entry_meta = dict(entry.get("meta") or {})
        entry_meta["cga_causal_grid_protocol_version"] = (
            CGA_CAUSAL_GRID_PROTOCOL_VERSION
        )
        entry["meta"] = entry_meta
    for mid, entry in serial_by.items():
        if not str(mid).startswith(AGIEND_CAUSAL_GRID_METHOD_PREFIXES):
            continue
        if not isinstance(entry, dict):
            continue
        entry_meta = dict(entry.get("meta") or {})
        entry_meta["agiend_causal_grid_protocol_version"] = (
            AGIEND_CAUSAL_GRID_PROTOCOL_VERSION
        )
        entry["meta"] = entry_meta
    actually_reused_strengthen: List[str] = []
    for mid, previous_entry in reuse_strengthen_by_method.items():
        repaired = serial_by.get(mid)
        if not isinstance(repaired, dict):
            continue
        repaired_meta = dict(repaired.get("meta") or {})
        if not repaired_meta.get("strengthen_reused"):
            # A weaken/random failure must not throw away the still-valid v5
            # strengthen component. Preserve it alongside the repair error so
            # the next refresh can retry only the failed component again.
            if not repaired_meta.get("error"):
                continue
            for key in ("strengths", "selected_strength", "selected"):
                repaired[key] = previous_entry.get(key)
            repaired_meta["strengthen_reused"] = True
            repaired["meta"] = repaired_meta
        for key in ("strengths", "selected_strength", "selected"):
            if repaired.get(key) != previous_entry.get(key):
                raise RuntimeError(
                    f"causal component repair changed frozen strengthen field "
                    f"{mid}.{key}; refusing to persist"
                )
        actually_reused_strengthen.append(mid)
    restore_by = dict(progress_by)
    if reuse_completed_causal:
        restore_by = {**prev_by, **restore_by}
    if restore_by:
        kept = 0
        for mid, prev_c in restore_by.items():
            if (
                str(mid) in invalidated_ridge_ids
                or str(mid) in invalidated_direction_ids
                or str(mid) in invalidated_cga_grid_ids
                or str(mid) in invalidated_agiend_grid_ids
            ):
                continue
            if not causal_raw_entry_ok(prev_c):
                continue
            new_c = serial_by.get(str(mid))
            if new_c is None or not causal_raw_entry_ok(new_c):
                serial_by[str(mid)] = prev_c
                by_method[str(mid)] = prev_c
                kept += 1
        if kept:
            print(
                f"causal: restored {kept} prior completed sweeps after partial re-run",
                flush=True,
            )
    out_raw = {
        **{k: v for k, v in causal_raw.items() if k != "by_method"},
        "by_method": serial_by,
        "protocol_version": CAUSAL_PROTOCOL_VERSION,
        "actiend_ridge_id_protocol_version": ACTIEND_RIDGE_ID_PROTOCOL_VERSION,
        "direction_polarity_protocol_version": DIRECTION_POLARITY_PROTOCOL_VERSION,
        "cga_causal_grid_protocol_version": CGA_CAUSAL_GRID_PROTOCOL_VERSION,
        "agiend_causal_grid_protocol_version": AGIEND_CAUSAL_GRID_PROTOCOL_VERSION,
        "strengthen_reuse_source_protocol": (
            str(prev_causal.get("protocol_version"))
            if actually_reused_strengthen
            else None
        ),
        "strengthen_reused_method_ids": sorted(actually_reused_strengthen),
    }
    merged_summaries = _merged_causal_summary_entries(
        serial_by,
        current=causal_raw.get("summaries") or [],
        previous=prev_causal.get("summaries") or [],
    )
    out_raw["summaries"] = merged_summaries
    # run_study_causal persisted only the methods executed in this invocation.
    # Rewrite after skip-existing restoration so the standalone summary mirrors
    # the complete merged inventory kept in results.json.raw.causal.by_method.
    persist_causal_summary(cfg.output_dir, merged_summaries)

    method_rows: List[Dict[str, Any]] = []
    _attach_causal_to_rows(
        method_rows,
        {**out_raw, "by_method": serial_by},
        model_key=cfg.model_key,
        task_id=cfg.task_id,
        claim_classes=claim_classes,
    )
    for row in method_rows:
        row["model"] = cfg.model_key
        row["task"] = cfg.task_id

    n = len(serial_by)
    print(f"causal: finished with {n} method results", flush=True)
    return {"methods": method_rows, "raw": out_raw, "errors": errors}
