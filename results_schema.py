"""
Unified result schema for study experiments.

Every method (gradiend, actiend, sae/<mode or class>) writes the same core fields
so runs are comparable without ad-hoc JSON digging.

Class-specific rows use ``{backend}:{class_id}`` (e.g. ``gradiend:M``, ``sae:F``)
and share the class-vs-neutral metric protocol.
"""

from __future__ import annotations

import re
from typing import Any, AbstractSet, Dict, List, Mapping, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = "1.13"
# Encode/causal tables from schema < 1.12 mixed SAE last-token MASK hooks with
# ACTIEND/CAA filled prediction — not comparable. See PROTOCOL_INVALIDATION.md.
# activation_protocol_version >= 3: neutrals scored per token, not document-mean.
# 1.13 / protocol v4: val/test neutrals, excluded tokens, Youden/causal select on val.

# Feature-learning families that claim / fair tables may invent shells for.
# Pipeline stages (causal, localization) are not included.
TABLE_METHOD_FAMILIES: Tuple[str, ...] = (
    "gradiend",
    "actiend",
    "actiend_ridge",
    "actiend_pre",
    "sae",
    "sae_pre",
    "caa",
)

# Metrics used for none vs by_tensor (:tensors) encoding ablation.
# Each entry: (column_key, metric_field_or_tuple, higher_is_better).
NONE_VS_TENSOR_METRICS: tuple = (
    ("auc_n", ("roc_auc_neutral", "roc_auc"), True),
    ("auc_o", ("roc_auc_other", "min_pairwise_auroc"), True),
    ("bal", ("balanced_accuracy",), True),
    ("d", ("cohens_d",), True),
    ("spec", ("neutral_specificity", "specificity"), True),
    ("excl", ("class_exclusivity",), True),
    ("J", ("youden_j",), True),
    ("smag", ("neutral_specificity_mag",), True),
    ("emag", ("class_exclusivity_mag",), True),
)


def method_result(
    *,
    method: str,
    model: str,
    task: str,
    status: str = "ok",
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """One row in results['methods']."""
    row: Dict[str, Any] = {
        "method": method,
        "model": model,
        "task": task,
        "status": status,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
    }
    if error:
        row["error"] = error
    if extras:
        row["extras"] = extras
    return row




def comparison_methods_for(
    target_classes: Sequence[str],
    *,
    backends: Sequence[str] = ("gradiend", "actiend", "sae"),
) -> tuple:
    """Legacy: backend×class only (sae:{cls} = k*). Prefer :func:`fair_comparison_methods`."""
    return tuple(f"{b}:{c}" for b in backends for c in target_classes)


def fair_comparison_methods(
    target_classes: Sequence[str],
    *,
    gradiend_actiend: Sequence[str] = ("gradiend", "actiend"),
    include_sae: bool = True,
    include_joint_sae: bool = True,
    include_caa: bool = True,
    caa_act_policies: Sequence[str] = ("prediction", "mean", "last"),
) -> tuple:
    """Primary claim subset (sparse SAE vs GRADIEND/ACTIEND/CAA). Not the full ablation set.

    Full encoder analysis must use :func:`encoder_ablation_methods` — never drop
    k-fixed / sel_opp_fire / layer / :all / joint:{cls} / CAA L* rows from TABLES/REPORT
    when those families were enabled. Pass ``include_sae=False`` / ``include_caa=False``
    (or use :func:`primary_fair_methods`) when the task disables those families
    (e.g. ``race_one_pole``).
    """
    if not target_classes:
        raise ValueError("fair_comparison_methods requires a non-empty target_classes sequence")
    out: List[str] = []
    for cls in target_classes:
        c = str(cls)
        for b in gradiend_actiend:
            out.append(f"{b}:{c}")
        if include_sae:
            out.append(f"sae:{c}:k1")
            out.append(f"sae:{c}:kstar")
            out.append(f"sae:{c}:all_k1")
            out.append(f"sae_pre:{c}:k1")
            out.append(f"sae_pre:{c}:kstar")
            out.append(f"sae_pre:{c}:all_k1")
        if include_caa:
            for pol in caa_act_policies:
                out.append(f"caa:{c}:act_{pol}")
    if include_sae and include_joint_sae:
        out.append("sae:joint")
        out.append("sae_pre:joint")
    return tuple(out)


def enabled_method_families(results: Mapping[str, Any]) -> Set[str]:
    """Method families that should appear in claim / fair / none-vs-tensor tables.

    Prefer ``config.enabled_methods``, then ``raw.enabled``, then infer from method
    id prefixes. Falls back to all :data:`TABLE_METHOD_FAMILIES` only when nothing
    is recorded (legacy dumps).
    """
    cfg = results.get("config")
    raw_list: Any = None
    if isinstance(cfg, Mapping):
        raw_list = cfg.get("enabled_methods")
    if raw_list is None:
        raw_list = (results.get("raw") or {}).get("enabled")
    if isinstance(raw_list, (list, tuple, set, frozenset)):
        found = {str(x) for x in raw_list if str(x) in TABLE_METHOD_FAMILIES}
        if found:
            return found
    inferred: Set[str] = set()
    for row in results.get("methods") or []:
        mid = str(row.get("method") or "")
        fam = mid.split(":", 1)[0]
        if fam in TABLE_METHOD_FAMILIES:
            inferred.add(fam)
    return inferred if inferred else set(TABLE_METHOD_FAMILIES)


def primary_fair_methods(results: Mapping[str, Any]) -> tuple:
    """Primary claim subset for ``results``, omitting disabled method families."""
    def _active_caa_policies() -> tuple[str, ...]:
        # Only show CAA policies that were actually produced with encoder metrics
        # in this run (e.g. core => prediction; full => prediction/mean/last).
        policies: List[str] = []
        for row in (results.get("methods") or []):
            did = str(display_method_id(row) or "")
            parts = did.split(":")
            if len(parts) < 3 or parts[0] != "caa":
                continue
            tail = parts[-1]
            if not tail.startswith("act_"):
                continue
            pol = tail[4:]
            metrics = row.get("metrics") or {}
            if _has_encoder_metrics(metrics) and pol not in policies:
                policies.append(pol)
        return tuple(policies) if policies else ("prediction",)

    classes = require_claim_classes(results)  # type: ignore[arg-type]
    enabled = enabled_method_families(results)
    backends: List[str] = []
    if "gradiend" in enabled:
        backends.append("gradiend")
    if "actiend" in enabled:
        backends.append("actiend")
    if "actiend_ridge" in enabled:
        backends.append("actiend_ridge")
    if "actiend_pre" in enabled:
        backends.append("actiend_pre")
    invented = fair_comparison_methods(
        classes,
        gradiend_actiend=tuple(backends),
        include_sae="sae" in enabled,
        include_joint_sae="sae" in enabled,
        include_caa="caa" in enabled,
        caa_act_policies=_active_caa_policies(),
    )
    pair_ids: List[str] = []
    claim = set(classes)
    for did, row in method_rows_by_display_id(results).items():
        parts = str(did).split(":")
        if len(parts) != 3:
            continue
        if parts[0] not in backends:
            continue
        if "-" in parts[1] and parts[2] in claim:
            pair_ids.append(did)
    # Pair-qualified ids first (when present), then the usual one-pole / short ids.
    ordered = list(dict.fromkeys([*pair_ids, *invented]))
    return tuple(ordered)


def require_target_classes(results: Dict[str, Any]) -> List[str]:
    """Return ``config.target_classes`` or raise — never invent task labels.

    TABLES / REPORT / plot helpers must call this instead of defaulting to gender
    ``(\"M\", \"F\")``. Full task vocabulary (includes CF poles on one-pole tasks).
    """
    cfg = results.get("config")
    if not isinstance(cfg, Mapping):
        raise ValueError(
            "results['config'] is required (with config['target_classes']). "
            "Refusing silent defaults (e.g. gender M/F)."
        )
    classes = cfg.get("target_classes")
    if not isinstance(classes, (list, tuple)) or not [c for c in classes if str(c).strip()]:
        raise ValueError(
            "results['config']['target_classes'] must be a non-empty list of class ids; "
            f"got {classes!r}. Refusing silent defaults (e.g. gender M/F)."
        )
    return [str(c) for c in classes]


def require_claim_classes(results: Dict[str, Any]) -> List[str]:
    """Classes for primary / fair / none-vs-tensor claim tables.

    Prefer ``config.claim_classes`` (one-pole factual poles). Falls back to
    ``target_classes`` for bipolar / legacy dumps.
    """
    cfg = results.get("config")
    if isinstance(cfg, Mapping):
        claim = cfg.get("claim_classes")
        if isinstance(claim, (list, tuple)):
            cleaned = [str(c) for c in claim if str(c).strip()]
            if cleaned:
                return cleaned
        opc = cfg.get("one_pole_classes")
        abl = cfg.get("ablations") if isinstance(cfg.get("ablations"), Mapping) else {}
        if (
            isinstance(opc, (list, tuple))
            and [c for c in opc if str(c).strip()]
            and (abl.get("one_pole") is True or abl.get("pair") is False)
        ):
            return [str(c) for c in opc if str(c).strip()]
    return require_target_classes(results)


def method_pole_class(method_id: str) -> Optional[str]:
    """Task pole encoded in ``{family}:{pole}:…`` (e.g. ``sae:MATCH:k1`` → MATCH)."""
    parts = str(method_id).split(":")
    if len(parts) < 2 or parts[0] not in TABLE_METHOD_FAMILIES:
        return None
    if parts[0] == "sae" and parts[1] == "joint":
        return parts[2] if len(parts) >= 3 else None
    return parts[1]


def sae_method_on_non_claim_class(results: Dict[str, Any], method_id: str) -> bool:
    """True when an SAE id targets a CF pole that SAE encode never scored."""
    parts = str(method_id).split(":")
    if not parts or parts[0] != "sae":
        return False
    pole = method_pole_class(method_id)
    if pole is None:
        return False
    return pole not in set(require_claim_classes(results))


def table_visible_method(results: Dict[str, Any], method_id: str) -> bool:
    """Exclude SAE ghost rows for non-claim poles from TABLES / REPORT."""
    return not sae_method_on_non_claim_class(results, method_id)


def display_method_id(row: Mapping[str, Any]) -> str:
    """Qualify pair trainer rows so they do not collide with one-pole ``:{cls}`` ids.

    Legacy dumps stored pair success rows as ``gradiend:white`` with
    ``metrics.ablation=pair`` / ``metrics.pair``. Tables must show
    ``gradiend:asian-white:white`` instead of last-wins collapsing onto one-pole.
    """
    mid = str(row.get("method") or "")
    m = row.get("metrics") or {}
    if str(m.get("ablation") or "") != "pair":
        return mid
    pair = m.get("pair")
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return mid
    parts = mid.split(":")
    if len(parts) >= 3 and "-" in parts[1]:
        return mid
    a, b = sorted(str(x) for x in pair)
    key = f"{a}-{b}"
    backend = parts[0] if parts else mid
    if len(parts) <= 1:
        return f"{backend}:{key}"
    cls = parts[1]
    rest = parts[2:]
    if rest and rest[0] in {"tensors", "all"}:
        return ":".join([backend, key, rest[0], cls, *rest[1:]])
    return ":".join([backend, key, cls, *rest])


def short_class_method_aliases(method_id: str) -> Tuple[str, ...]:
    """Inverse of pair qualification used by PoC causal ids.

    Pair encoders are ``gradiend:negative-positive:negative``; causal historically
    wrote short ``gradiend:negative``. Analysis must join those spellings.
    ``actiend:F-M:F:tok_all`` → ``actiend:F:tok_all``.
    ``gradiend:F-M:tensors:F`` → ``gradiend:F:tensors``.
    """
    mid = str(method_id).split("|", 1)[0]
    parts = mid.split(":")
    if len(parts) < 3 or parts[0] not in TABLE_METHOD_FAMILIES:
        return ()
    backend, maybe_pair, *rest = parts
    if "-" not in maybe_pair:
        return ()
    left, right = maybe_pair.split("-", 1)
    if not left or not right or left.startswith("L") or "tok_" in maybe_pair:
        return ()
    if rest and rest[0] in {"tensors", "all"}:
        if len(rest) < 2:
            return ()
        cls = rest[1]
        suffix = rest[2:]
        return (":".join([backend, cls, rest[0], *suffix]),)
    cls = rest[0]
    suffix = rest[1:]
    if suffix:
        return (":".join([backend, cls, *suffix]),)
    return (f"{backend}:{cls}",)


def _pair_keys_for_class(
    results: Mapping[str, Any], backend: str, cls: str
) -> List[str]:
    """Pair keys (``negative-positive``) that have an encoder row for ``cls``."""
    keys: List[str] = []
    seen: Set[str] = set()
    cls_s = str(cls)
    for row in results.get("methods") or []:
        if not isinstance(row, Mapping):
            continue
        metrics = row.get("metrics") or {}
        if not _has_encoder_metrics(metrics):
            continue
        disp = display_method_id(row)
        parts = disp.split(":")
        if not parts or parts[0] != backend:
            continue
        key = None
        if len(parts) >= 3 and "-" in parts[1]:
            left, right = parts[1].split("-", 1)
            if left and right and not left.startswith("L") and cls_s in parts[2:]:
                key = parts[1]
        if key is None:
            pair = metrics.get("pair")
            if (
                str(metrics.get("ablation") or "") == "pair"
                and isinstance(pair, (list, tuple))
                and len(pair) == 2
                and (
                    cls_s == str(metrics.get("target_class") or "")
                    or cls_s in {str(x) for x in pair}
                )
            ):
                a, b = sorted(str(x) for x in pair)
                key = f"{a}-{b}"
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _has_onepole_encoder(results: Mapping[str, Any], backend: str, cls: str) -> bool:
    """True when a real one-pole encoder row exists (not a causal-only shell)."""
    cls_s = str(cls)
    for row in results.get("methods") or []:
        if not isinstance(row, Mapping):
            continue
        metrics = row.get("metrics") or {}
        if not _has_encoder_metrics(metrics):
            continue
        if str(metrics.get("ablation") or "") == "pair":
            continue
        mid = str(row.get("method") or "").split("|", 1)[0]
        parts = mid.split(":")
        if len(parts) >= 2 and parts[0] == backend and parts[1] == cls_s:
            if len(parts) == 2 or (len(parts) == 3 and parts[2] in {"tensors", "all"}):
                return True
        if str(metrics.get("ablation") or "") == "one_pole" and str(
            metrics.get("target_class") or ""
        ) == cls_s:
            return True
    return False


def pair_qualify_method_id(results: Mapping[str, Any], method_id: str) -> str:
    """Map legacy short causal ids onto the pair encoder id when unambiguous.

    Old PoC causal wrote ``gradiend:negative`` even for a pair trainer whose
    encoder is ``gradiend:negative-positive:negative``. If that pair encoder
    exists and there is no one-pole encoder for the class, the short id is the
    pair feature and tables must show the pair spelling.

    When both pair and one-pole encoders exist, leave the short id (one-pole).
    """
    mid = str(method_id).split("|", 1)[0]
    parts = mid.split(":")
    if len(parts) < 2 or parts[0] not in {
        "gradiend",
        "actiend",
        "actiend_ridge",
        "actiend_pre",
        # CGA/CAGA/AGIEND share the GRADIEND id shape, so they need the same short-id repair.
        "cga",
        "cga_tensor_norm",
        "caga",
        "agiend",
    }:
        return mid
    if "-" in parts[1]:
        return mid
    backend, cls = parts[0], parts[1]
    rest = parts[2:]
    keys = _pair_keys_for_class(results, backend, cls)
    if len(keys) != 1 or _has_onepole_encoder(results, backend, cls):
        return mid
    key = keys[0]
    if rest and rest[0] in {"tensors", "all"}:
        return ":".join([backend, key, rest[0], cls, *rest[1:]])
    return ":".join([backend, key, cls, *rest])


def _causal_metrics_for_id(
    by_method: Mapping[str, Mapping[str, Any]], method_id: str
) -> Dict[str, Any]:
    """Prefer the payload that actually has causal fields (pair id or short alias)."""
    cands = [str(method_id), *short_class_method_aliases(method_id)]
    fallback: Dict[str, Any] = {}
    for cand in cands:
        m = dict(by_method.get(cand) or {})
        if not m:
            continue
        if (
            m.get("causal_signed_effect") is not None
            or m.get("causal_selected_strength") is not None
            or m.get("causal_error") is not None
            or m.get("causal_lms") is not None
        ):
            return m
        if not fallback:
            fallback = m
    return fallback


def method_rows_by_display_id(results: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index method rows by :func:`display_method_id` (pair + one-pole both kept)."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in results.get("methods") or []:
        if not isinstance(row, Mapping) or not row.get("method"):
            continue
        out[display_method_id(row)] = row  # type: ignore[assignment]
    return out


def require_class_ids(classes: Optional[Sequence[Any]], *, where: str) -> List[str]:
    """Require an explicit non-empty class list — never invent labels."""
    if not isinstance(classes, (list, tuple)) or not [c for c in classes if str(c).strip()]:
        raise ValueError(
            f"{where}: target_classes must be a non-empty sequence of class ids; "
            f"got {classes!r}. Refusing silent defaults (e.g. gender M/F)."
        )
    return [str(c) for c in classes]


# Provenance stamp for stored encoder/detection metrics. Version 1 (unstamped)
# fit the Spec_n / Excl decision rules on the split being reported; version 2 fits
# them on validation and freezes them for test (``sae_eval.frozen_decision_rules``).
# It is deliberately NOT folded into the ``*_ENCODER_EVAL_VERSION`` constants: the
# causal reload requires that exact version to read the saved validation readouts,
# and those readouts are unaffected by this fix, so bumping it would wrongly mark
# them unavailable. Stage-level reuse gates check this stamp instead.
ENCODER_EVAL_RULES_VERSION = 2


def encoder_eval_rules_current(stamp: Any) -> bool:
    """Whether a stored ``encoder_eval_rules_version`` stamp is current (missing = v1)."""
    try:
        return int(stamp) >= ENCODER_EVAL_RULES_VERSION
    except (TypeError, ValueError):
        return False


def unfrozen_rule_readouts(readouts: Any) -> List[str]:
    """Keys of class-vs-neutral readouts whose Spec_n/Excl rules were fit on the split
    they report (i.e. NOT frozen on validation).

    ``readouts`` maps a key (class / component / method id) to a
    ``class_vs_neutral_metrics`` output; entries that are errors or not such a
    readout (no ``neutral_youden_threshold_source``) are ignored. Rival rules count
    only when the readout actually scored rivals.
    """
    if not isinstance(readouts, Mapping):
        return []
    bad: List[str] = []
    for key, r in readouts.items():
        if not isinstance(r, Mapping) or r.get("error"):
            continue
        if "neutral_youden_threshold_source" not in r:
            continue
        rival_src = r.get("rival_youden_threshold_source")
        if r.get("neutral_youden_threshold_source") != "provided" or rival_src == "eval":
            bad.append(str(key))
    return bad


def fair_metric_fields(readout: Mapping[str, Any]) -> Dict[str, Any]:
    """Core fair-comparison fields shared by GRADIEND / ACTIEND / SAE / CAA."""
    return {
        "roc_auc": readout.get("roc_auc"),
        "roc_auc_neutral": readout.get("roc_auc_neutral", readout.get("roc_auc")),
        "roc_auc_other": readout.get("roc_auc_other", readout.get("min_pairwise_auroc")),
        "roc_auc_boot_std": readout.get("roc_auc_boot_std"),
        "balanced_accuracy": readout.get("balanced_accuracy"),
        "cohens_d": readout.get("cohens_d"),
        "specificity": readout.get("specificity"),
        "neutral_specificity": readout.get("neutral_specificity", readout.get("specificity")),
        "neutral_tnr": readout.get("neutral_tnr"),
        "class_exclusivity": readout.get("class_exclusivity"),
        "min_pairwise_auroc": readout.get("min_pairwise_auroc", readout.get("roc_auc_other")),
        "youden_threshold": readout.get("youden_threshold"),
        "youden_j": readout.get("youden_j"),
        "youden_negatives": readout.get("youden_negatives"),
        "target_tpr": readout.get("target_tpr"),
        "neutral_specificity_mag": readout.get("neutral_specificity_mag"),
        "class_exclusivity_mag": readout.get("class_exclusivity_mag"),
        "specificity_mid": readout.get("specificity_mid"),
        "neutral_spread": readout.get("neutral_spread"),
        "neutral_gap": readout.get("neutral_gap"),
        "neutral_abs_mean": readout.get("neutral_abs_mean"),
        "encoder_correlation": readout.get("encoder_correlation"),
        "class_separation": readout.get("class_separation"),
        # Site/layer/model-selection metric.  This is deliberately distinct
        # from the test-set fields above: any downstream selector comparing
        # multiple candidate readouts must use this frozen validation value.
        # The full validation readout, not a single derived scalar: selection
        # must rank on the same quantity the tables report (Det), and persisting
        # only one derived number is what made that impossible without a rerun.
        "val_readout": readout.get("val_readout"),
        "eval_split": "test",
    }


# Gender-EN convenience only — do NOT use as a TABLES/REPORT fallback.
COMPARISON_METHODS_GENDER_EN = fair_comparison_methods(("M", "F"))
# Back-compat alias for gender scripts; TABLES must use require_target_classes().


def _has_encoder_metrics(metrics: Dict[str, Any]) -> bool:
    if not metrics:
        return False
    return any(
        metrics.get(k) is not None
        for k in (
            "roc_auc",
            "roc_auc_neutral",
            "balanced_accuracy",
            "cohens_d",
            "youden_j",
            "neutral_specificity",
            "specificity",
            "class_exclusivity",
            "neutral_specificity_mag",
            "class_exclusivity_mag",
            "encoder_correlation",
            "class_separation",
            "min_pairwise_auroc",
            "roc_auc_other",
        )
    )


def encoder_inherit_source(method_id: str) -> Optional[str]:
    """Map causal-only suffixes to the encoder id that owns the same readout.

    Token-selector / prediction-site are causal policies; encoding is shared with
    the base feature id (e.g. ``actiend:M:tok_all`` → ``actiend:M``,
    ``sae:M:k1_tok_prediction`` → ``sae:M:k1``,
    ``sae_pre:M:k1_tok_prediction`` → ``sae_pre:M:k1``,
    legacy ``sae:M:k1_pre_tok_prediction`` → ``sae_pre:M:k1_tok_prediction``,
    ``sae:M:L0_k1`` → ``sae:M:L0``,
    ``actiend:asian-white:asian:tok_all`` → ``actiend:asian-white:asian``,
    ``actiend:M:L3_tok_all_gate_encoder_direction`` → ``actiend:M:L3``,
    ``actiend:M:tensors_tok_all_gate_encoder_direction`` → ``actiend:M:tensors``;
    legacy ``all_tok_*`` → ``:{cls}:all``).
    """
    from study.method_ids import normalize_sae_method_id

    mid = normalize_sae_method_id(str(method_id))
    parts = mid.split(":")
    if len(parts) < 3:
        return None
    backend = parts[0]
    if backend not in {"actiend", "actiend_ridge", "actiend_pre", "sae", "sae_pre", "gradiend", "caa"}:
        return None
    if backend == "caa":
        return None
    tail = parts[-1]
    prefix = parts[:-1]
    if tail.endswith("_tok_prediction") and (
        tail.startswith("k") or tail.startswith("all_k")
    ):
        return mid[: -len("_tok_prediction")]
    if tail.startswith("tensors_tok_"):
        return ":".join([*prefix, "tensors"])
    if tail.startswith("all_tok_"):
        return ":".join([*prefix, "all"])
    if re.fullmatch(r"L\d+_k\d+", tail):
        layer = tail.split("_", 1)[0]
        return ":".join([*prefix, layer])
    if tail.startswith("L") and "_tok_" in tail:
        layer = tail.split("_", 1)[0]
        if layer[1:].isdigit():
            return ":".join([*prefix, layer])
    if tail.startswith("tok_"):
        return ":".join(prefix)
    return None


_SAE_LAYER_ID_RE = re.compile(r"^sae:([^:]+):L(\d+)(?:_k\d+)?$")


def _sae_layer_readout_metrics(
    results: Mapping[str, Any], method_id: str
) -> Optional[tuple]:
    """Pull per-layer SAE encoder metrics from ``raw.sae.by_layer_readouts``."""
    match = _SAE_LAYER_ID_RE.fullmatch(str(method_id))
    if not match:
        return None
    cls, layer = match.group(1), match.group(2)
    raw = (results.get("raw") or {}).get("sae") or {}
    by_layer = raw.get("by_layer_readouts") or {}
    layer_map = by_layer.get(layer)
    if layer_map is None:
        try:
            layer_map = by_layer.get(int(layer))
        except (TypeError, ValueError):
            layer_map = None
    rd = (layer_map or {}).get(cls) or {}
    if not _has_encoder_metrics(rd):
        return None
    src = f"sae:{cls}:L{layer}"
    return rd, src, True


def tensors_all_alias(method_id: str) -> Optional[str]:
    """Map canonical ``:tensors`` ↔ legacy ``:all`` encoder aggregate ids.

    Does not touch CAA ``all_act_*`` or SAE ``all_k*`` tails.
    """
    mid = str(method_id)
    parts = mid.split(":")
    if len(parts) != 3:
        return None
    backend, cls, tail = parts
    if backend not in {"actiend", "actiend_ridge", "gradiend", "sae"}:
        return None
    if tail == "tensors":
        return f"{backend}:{cls}:all"
    if tail == "all":
        return f"{backend}:{cls}:tensors"
    return None


METHOD_ID_LEGEND: str = """
Method-id legend (encoder + causal share the same id when the ablation applies to both):
  gradiend:{cls}              encoder readout + causal weight rewrite (none split)
  gradiend:{pair}:{cls}       pair trainer (e.g. asian-white) evaluated on {cls}
  gradiend:{cls}:tensors      by_tensor aggregate encoder + causal weight rewrite
  actiend:{cls}               encoder readout only (NOT an intervention; none split)
  actiend:{pair}:{cls}        pair trainer encoder readout for {cls}
  actiend:{cls}:tensors       by_tensor aggregate encoder readout
  actiend:{cls}:tok_all_gate_encoder_direction
                              ACTIEND causal default none-split
                              (= package token_selector='encoder_direction')
  actiend:{cls}:tok_all / :tok_prediction
                              ACTIEND causal ungated scope ablations (none split)
  actiend:{cls}:tensors_tok_all_gate_encoder_direction
                              ACTIEND causal default on by_tensor aggregate
  actiend:{cls}:tensors_tok_all / :tensors_tok_prediction
                              ACTIEND tok ablations on by_tensor aggregate
  actiend:{cls}:L{n}_tok_all_gate_encoder_direction
                              per-layer ACTIEND causal (same default policy)
  actiend_pre:{cls}           mixed-site ACTIEND (source=pre_prediction, target=prediction)
  actiend_pre:{pair}:{cls} / :tensors / :tok_* / :L*
                              same id shapes as actiend, separate family (like sae_pre)
  sae:{cls}:k1                encoder k=1 + causal tok=all (filled prediction site)
  sae:{cls}:kstar             encoder k* bag + causal tok=all (shared id)
  sae:{cls}:k{n}              encoder fixed-k + causal tok=all (shared id; n in FIXED_KS)
  sae:{cls}:all_k{n}          top-n features @ every layer (sum encode; multi-site steer)
  sae_pre:{cls}:k1            last-context / classical SAE site (separate family)
  sae_pre:{cls}:kstar / :all_k1
                              pre_prediction activation selection
  sae:{cls}:k1_tok_prediction SAE causal-only: top-1, prediction-site
  sae_pre:{cls}:k1_tok_prediction
                              SAE causal-only tok ablation on pre site
  sae:{cls}:sel_opp_fire      encoder + causal; opp_fire ranking AND layer pick
                              (same score: mean_diff - lambda_neu|mu_neu| - lambda_opp P(z>0|rival))
  sae:{cls}:k1_clamp          SAE causal-only: top-1, activation clamp (Templeton et al. 2024,
                              full-suite ablation vs default additive steering; AxBench App. F)
  sae:joint / sae:joint:{cls} bipolar feature (encoder joint; causal per class ±)
  sae_pre:joint / sae_pre:joint:{cls}
                              joint feature at pre_prediction site
  caa:{cls}:act_{policy}      CAA mean-diff concat cosine (policy=prediction|mean|last)
  caa:{cls}:all_act_{policy}  mean of per-layer cosines + multi-layer residual steer
  caa:{cls}:L{n}_act_{policy} per-layer CAA encode + residual add (tok=all)
""".strip()


def method_id_legend(enabled: Optional[AbstractSet[str]] = None) -> str:
    """Full method-id legend, optionally omitting disabled families (sae/caa/…)."""
    if enabled is None or set(TABLE_METHOD_FAMILIES).issubset(set(enabled)):
        return METHOD_ID_LEGEND
    drop = {f for f in TABLE_METHOD_FAMILIES if f not in enabled}
    if not drop:
        return METHOD_ID_LEGEND
    out: List[str] = []
    skipping = False
    for line in METHOD_ID_LEGEND.splitlines():
        stripped = line.lstrip()
        lead = len(line) - len(stripped)
        if any(stripped.startswith(f"{f}:") for f in TABLE_METHOD_FAMILIES):
            fam = stripped.split(":", 1)[0]
            skipping = fam in drop
            if not skipping:
                out.append(line)
            continue
        if skipping and lead >= 4 and stripped:
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


# Pseudo-formulas for console / REPORT (ASCII-safe).
METRIC_FORMULAS: str = """
Separability metrics (class-vs-neutral readout score s; predict target if s > τ):
  auc_n   = AUROC(s | target vs neutral)
  auc_o   = min_c AUROC(s | target vs rival c)          # ranking vs other class(es)
  bal     = 0.5*(TPR+TNR) at midpoint τ_mid=(μ_tgt+μ_neu)/2   # balanced accuracy
  d       = |μ_tgt-μ_neu| / s_pooled                    # Cohen's d (tgt vs neu)
  τ*, J   = argmax_τ [TPR(τ)+TNR_neg(τ)-1]              # Youden; neg=rivals∪neutrals
  spec    = TNR_neu(τ*) = P(s≤τ*|neutral)               # primary specificity (Youden)
  excl    = min_c TNR_c(τ*) = P(s≤τ*|rival c)           # class exclusivity at same τ*
  tpr     = P(s>τ*|target)
  corr    = Pearson(s, ±1 labels)                       # joint bipolar diagnostic
  sep     = |μ_a-μ_b|                                   # raw mean gap (scale-dependent)
Magnitude soft-scores (ALWAYS shown; SAE often wins here vs GRADIEND/ACTIEND):
  half    = 0.5*|μ_tgt-μ_neu|
  smag    = half / (half + mean_|s_neu|)                # neutral_specificity_mag
  emag    = min_c half / (half + mean_max(s_c,0))       # class_exclusivity_mag
SAE feature + layer selection (same metric; top-k always within one layer):
  per_class score = (μ_c − μ_rest) − λ_neu |μ_neu|
  opp_fire  score = per_class − λ_opp P(z>0 | rival)
  layer* = argmax_L score(top-1 feature @ L)   # → sae:{cls}:k* / :sel_opp_fire
k* bag size (within selected layer): k* = argmax_k 0.5*(auc_n+auc_o); tie → smaller k
  → method id sae:{cls}:kstar (never bare sae:{cls})
Feature suitability (suitable feature learnt?):
  E = min(auc_n, auc_o, excl)                         # full (rival texts)
    | min(auc_n, spec)                                # one-pole / vs-neutral only
  G = 1[|eff|≥τ_c under LMS×0.99]  (else 0; soft: clip(|eff|/τ_c,0,1))
  S = E × G                                           # → suitability / reason ok|E|G|…
""".strip()




def _fmt_metric(value: Any, *, width: int = 8, precision: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:>{width}.{precision}f}"
    return f"{'-':>{width}}"


def _resolve_comparison_methods(results: Dict[str, Any]) -> tuple:
    """Primary-claim subset only. Prefer :func:`encoder_ablation_methods` for tables."""
    return primary_fair_methods(results)


def _encoder_sort_key(method: str, primary: Sequence[str]) -> tuple:
    mid = str(method)
    if mid in primary:
        return (0, list(primary).index(mid), mid)
    parts = mid.split(":")
    cls_order = {"M": 0, "F": 1}
    tail = parts[-1] if parts else ""
    backend = parts[0] if parts else ""
    cls = parts[1] if len(parts) > 1 else ""
    # Prefer shared encoder+causal ablations before layer dumps / selection modes.
    if len(parts) >= 3 and (
        tail in {"k1", "kstar", "sel_opp_fire", "all", "full"}
        or (tail.startswith("k") and tail[1:].isdigit())
        or tail.startswith("all_k")
        or tail.startswith("joint")
        or tail.startswith("act_")
        or tail.startswith("all_act_")
    ):
        k_ord = 0
        if tail.startswith("all_k") and tail[5:].isdigit():
            k_ord = 5_000 + int(tail[5:])
        elif tail.startswith("k") and tail[1:].isdigit():
            k_ord = int(tail[1:])
        elif tail == "kstar":
            k_ord = 10_000
        elif tail == "sel_opp_fire":
            k_ord = 20_000
        elif tail.startswith("act_"):
            k_ord = {"prediction": 1, "mean": 2, "last": 3}.get(tail[4:], 9)
        elif tail.startswith("all_act_"):
            k_ord = 100 + {"prediction": 1, "mean": 2, "last": 3}.get(tail[8:], 9)
        return (1, backend, cls_order.get(cls, 50), k_ord, mid)
    if len(parts) == 3 and (
        (tail.startswith("L") and tail[1:].isdigit())
        or (tail.startswith("L") and "_act_" in mid)
    ):
        layer_tok = tail.split("_", 1)[0]
        if layer_tok.startswith("L") and layer_tok[1:].isdigit():
            return (2, backend, cls_order.get(cls, 50), int(layer_tok[1:]), mid)
    if mid.startswith("sae:") and ":" not in mid[4:]:
        return (3, mid)  # selection-mode labels
    return (4, mid)


def encoder_ablation_methods(results: Dict[str, Any]) -> tuple:
    """All encoder ablations of interest — aligned with causal, nothing dropped.

    Includes:
    - every method row with encoder separability metrics
    - every causal method id (inherits encoder metrics from
      :func:`encoder_inherit_source` when the id is causal-policy-only)

    Primary fair-claim ids are sorted first when present as real method rows;
    empty primary shells are never invented for disabled families.
    """
    primary = primary_fair_methods(results)
    by_rows = method_rows_by_display_id(results)
    by_metrics = {
        did: (row.get("metrics") or {})
        for did, row in by_rows.items()
    }
    found: List[str] = []
    for mid, m in by_metrics.items():
        if not table_visible_method(results, mid):
            continue
        if _has_encoder_metrics(m) or mid in primary:
            found.append(mid)
    # Pull in causal ids so encoder and causal tables share the same ablation set
    # when those ids actually have encoder metrics (own or inherited).
    for mid in _causal_methods(results):
        if not table_visible_method(results, mid):
            continue
        enc, _src, _inh = resolve_encoder_metrics(results, str(mid))
        if not _has_encoder_metrics(enc) and str(mid) not in primary:
            continue
        found.append(str(mid))
        src = encoder_inherit_source(str(mid))
        candidates = [src] if src else []
        if src:
            alt = tensors_all_alias(src)
            if alt:
                candidates.append(alt)
        for cand in candidates:
            if cand in by_metrics and _has_encoder_metrics(by_metrics[cand]):
                found.append(cand)
                break
    uniq = sorted(set(found), key=lambda m: _encoder_sort_key(m, primary))
    return tuple(uniq)


def resolve_encoder_metrics(
    results: Dict[str, Any], method_id: str
) -> tuple:
    """Return ``(metrics, source_id, inherited)`` for an encoder/causal-aligned id."""
    by_rows = method_rows_by_display_id(results)
    by = {did: (row.get("metrics") or {}) for did, row in by_rows.items()}
    for row in results.get("methods") or []:
        mid0 = str(row.get("method") or "")
        if mid0 and mid0 not in by:
            by[mid0] = row.get("metrics") or {}
    mid = str(method_id)
    own = by.get(mid) or {}
    if _has_encoder_metrics(own):
        return own, mid, False

    def _try(src: Optional[str]) -> Optional[tuple]:
        if not src:
            return None
        sm = by.get(src) or {}
        if _has_encoder_metrics(sm):
            return sm, src, True
        alt = tensors_all_alias(src)
        if alt:
            am = by.get(alt) or {}
            if _has_encoder_metrics(am):
                return am, alt, True
        return None

    hit = _try(encoder_inherit_source(mid))
    if hit is not None:
        return hit
    hit = _try(tensors_all_alias(mid))
    if hit is not None:
        return hit
    layer_hit = _sae_layer_readout_metrics(results, mid)
    if layer_hit is not None:
        return layer_hit
    src = encoder_inherit_source(mid)
    if src:
        layer_hit = _sae_layer_readout_metrics(results, src)
        if layer_hit is not None:
            return layer_hit
    return own, mid, False


def _summary_note(row: Dict[str, Any], *, verbose_feats: bool = False) -> str:
    m = row.get("metrics") or {}
    note = row.get("error") or ""
    kind = m.get("readout_kind") or ""
    if note:
        return str(note)
    if kind == "class_vs_neutral":
        note = "vs neutral"
        if m.get("sae_layer") is not None:
            note += f" L{m['sae_layer']}"
        if m.get("readout_k") is not None:
            note += f" k={m['readout_k']}"
        if verbose_feats:
            used = m.get("readout_features")
            if used is None and m.get("sae_top_features") is not None:
                used = m["sae_top_features"]
            if used is not None:
                feats = used if isinstance(used, list) else [used]
                k = m.get("readout_k")
                show = feats[: int(k)] if isinstance(k, int) and k > 0 else feats[:3]
                note += f" feats={show}"
        std = m.get("roc_auc_boot_std")
        if isinstance(std, (int, float)):
            note += f" ±{std:.3f}"
        auc_o = m.get("roc_auc_other", m.get("min_pairwise_auroc"))
        if isinstance(auc_o, (int, float)):
            note += f" auc_o={auc_o:.3f}"
        # Avoid duplicating columns already in the wide summary table.
        return note
    if kind == "joint_bipolar":
        return "joint bipolar"
    if m.get("readout_k") is not None:
        note = f"k={m['readout_k']}"
        std = m.get("roc_auc_boot_std")
        if isinstance(std, (int, float)):
            note += f" ±{std:.3f}"
        return note
    return ""


def summary_table(
    results: Dict[str, Any],
    *,
    headline_only: bool = False,
    verbose_feats: bool = False,
) -> str:
    """Plain-text table for console print (full separability suite + formulas).

    Default: **all** encoder ablations (aligned with causal). Pass
    ``headline_only=True`` only for the short primary-claim subset.
    """
    lines = [
        f"experiment: {results.get('experiment_id')}  task={results.get('task')}  model={results.get('model')}",
        METRIC_FORMULAS,
        "",
        (
            "Encoder ablations (primary fair subset only):"
            if headline_only
            else "Encoder ablations (FULL set — same ids as causal where shared; nothing dropped):"
        ),
        f"{'method':<48} {'status':<8} {'auc_n':>7} {'auc_o':>7} {'bal':>7} {'d':>8} "
        f"{'spec':>7} {'excl':>7} {'J':>7} {'smag':>7} {'emag':>7} {'notes'}",
        "-" * 156,
    ]
    if headline_only:
        methods = list(_resolve_comparison_methods(results))
    else:
        methods = list(encoder_ablation_methods(results))
    by_row = method_rows_by_display_id(results)
    for mid in methods:
        row = by_row.get(mid) or {"method": mid, "status": "ok", "metrics": {}}
        m, src, inherited = resolve_encoder_metrics(results, mid)
        note = _summary_note({**row, "metrics": m}, verbose_feats=verbose_feats)
        if inherited:
            note = (note + f" enc<-{src}").strip()
        elif not _has_encoder_metrics(m):
            note = (note + " (no encoder metrics)").strip()
        lines.append(
            f"{mid or '?':<48} {row.get('status', '?'):<8} "
            f"{_fmt_metric(m.get('roc_auc_neutral', m.get('roc_auc')), width=7)} "
            f"{_fmt_metric(m.get('roc_auc_other', m.get('min_pairwise_auroc')), width=7)} "
            f"{_fmt_metric(m.get('balanced_accuracy'), width=7)} "
            f"{_fmt_metric(m.get('cohens_d'))} "
            f"{_fmt_metric(m.get('neutral_specificity', m.get('specificity')), width=7)} "
            f"{_fmt_metric(m.get('class_exclusivity'), width=7)} "
            f"{_fmt_metric(m.get('youden_j'), width=7)} "
            f"{_fmt_metric(m.get('neutral_specificity_mag'), width=7)} "
            f"{_fmt_metric(m.get('class_exclusivity_mag'), width=7)} "
            f"{note}"
        )
    return "\n".join(lines)


def comparison_table(results: Dict[str, Any], *, primary_only: bool = False) -> str:
    """Encoder class-vs-neutral for **all** ablations (or primary subset if requested)."""
    primary = _resolve_comparison_methods(results)
    methods = primary if primary_only else encoder_ablation_methods(results)
    enabled = enabled_method_families(results)
    blurb = [
        "  gradiend/actiend:{cls} = encoder readout (NOT causal intervention)",
    ]
    if "sae" in enabled:
        blurb.extend(
            [
                "  sae:{cls}:k1 / :kstar / :k{n} = filled-span selection (fair vs ACTIEND/CAA encode)",
                "  sae_pre:{cls}:k1 / :kstar / :all_k1 = last-context / classical SAE (separate family)",
                "  sae:{cls}:sel_opp_fire = opp_fire rank + layer pick (same score; may differ in L)",
                "  actiend:{cls}:tok_* / sae:{cls}:k1_tok_prediction = causal policy; encoder inherited",
                "  actiend:{cls}:L* / sae:{cls}:L* = layer encoder profiles",
            ]
        )
    else:
        blurb.append("  actiend:{cls}:tok_* / :L* = causal policy / layer encoder profiles")
    if "caa" in enabled:
        blurb.append("  caa:{cls}:act_* = CAA mean-diff concat cosine")
    lines = [
        "",
        (
            "Primary fair subset (encoder):"
            if primary_only
            else "Encoder comparison (ALL ablations; same method ids as causal where shared):"
        ),
        *blurb,
        METRIC_FORMULAS,
        "",
        method_id_legend(enabled),
        "",
        f"{'method':<48} {'k':>4} {'L':>3} {'auc_n':>7} {'±':>6} {'auc_o':>7} {'bal':>7} "
        f"{'d':>8} {'spec':>7} {'excl':>7} {'J':>7} {'tpr':>7} {'smag':>7} {'emag':>7}",
        "-" * 146,
    ]
    for method in methods:
        m, src, inherited = resolve_encoder_metrics(results, method)
        k = m.get("readout_k")
        layer = m.get("sae_layer")
        k_s = f"{int(k)}" if isinstance(k, (int, float)) else "—"
        l_s = f"{int(layer)}" if isinstance(layer, (int, float)) else "—"
        empty = not _has_encoder_metrics(m)
        tag = f"  # enc<-{src}" if inherited else ""
        if empty:
            lines.append(
                f"{method:<48} {k_s:>4} {l_s:>3} "
                + " ".join(f"{'—':>7}" for _ in range(11))
                + ("  # no encoder metrics" if method not in primary else "")
            )
            continue
        lines.append(
            f"{method:<48} "
            f"{k_s:>4} "
            f"{l_s:>3} "
            f"{_fmt_metric(m.get('roc_auc_neutral', m.get('roc_auc')), width=7)} "
            f"{_fmt_metric(m.get('roc_auc_boot_std'), width=6)} "
            f"{_fmt_metric(m.get('roc_auc_other', m.get('min_pairwise_auroc')), width=7)} "
            f"{_fmt_metric(m.get('balanced_accuracy'), width=7)} "
            f"{_fmt_metric(m.get('cohens_d'))} "
            f"{_fmt_metric(m.get('neutral_specificity', m.get('specificity')), width=7)} "
            f"{_fmt_metric(m.get('class_exclusivity'), width=7)} "
            f"{_fmt_metric(m.get('youden_j'), width=7)} "
            f"{_fmt_metric(m.get('target_tpr'), width=7)} "
            f"{_fmt_metric(m.get('neutral_specificity_mag'), width=7)} "
            f"{_fmt_metric(m.get('class_exclusivity_mag'), width=7)}"
            f"{tag}"
        )
    if not primary_only:
        lines.append("")
        lines.append(
            f"(Primary fair subset still available via comparison_table(..., primary_only=True): "
            f"{', '.join(primary)})"
        )
    return "\n".join(lines)


def _causal_sort_key(method: str, headlines: Sequence[str]) -> tuple:
    mid = str(method)
    if mid in headlines:
        return (0, list(headlines).index(mid), mid)
    parts = mid.split(":")
    cls_order = {"M": 0, "F": 1}
    tail = parts[-1] if parts else ""
    # actiend:M:L11_tok_all_gate_encoder_direction
    if len(parts) >= 3 and tail.startswith("L") and "_tok_" in tail:
        layer_s = tail.split("_", 1)[0]
        if layer_s[1:].isdigit():
            return (
                1,
                parts[0],
                cls_order.get(parts[1], 50),
                int(layer_s[1:]),
                mid,
            )
    if len(parts) >= 3 and tail.startswith("L") and tail[1:].isdigit():
        return (
            1,
            parts[0],
            cls_order.get(parts[1], 50),
            int(tail[1:]),
            mid,
        )
    if len(parts) >= 3 and tail.startswith("k") and tail[1:].isdigit():
        return (2, parts[0], cls_order.get(parts[1], 50), int(tail[1:]), mid)
    return (3, mid)


def _causal_methods(results: Dict[str, Any]) -> tuple:
    """All methods with causal metrics: headlines first, then ACTIEND/SAE ``:L*``, then rest.

    Intentionally more than the 6 backend×class headlines when per-layer ACTIEND
    (or other part) causal rows are present.
    """
    classes = require_claim_classes(results)
    headlines = comparison_methods_for(classes)
    found: List[str] = []
    for row in results.get("methods") or []:
        mid = row.get("method")
        if not mid:
            continue
        if not table_visible_method(results, str(mid)):
            continue
        m = row.get("metrics") or {}
        extras = row.get("extras") or {}
        if (
            m.get("causal_signed_effect") is not None
            or m.get("causal_selected_strength") is not None
            or m.get("causal_error") is not None
            or m.get("causal_lms_curve")
            or extras.get("causal") is not None
            or extras.get("causal_lms_curve")
        ):
            found.append(pair_qualify_method_id(results, display_method_id(row)))
    raw = (results.get("raw") or {}).get("causal") or {}
    for s in raw.get("summaries") or []:
        if isinstance(s, dict) and s.get("method"):
            mid = str(s["method"])
            if table_visible_method(results, mid):
                found.append(pair_qualify_method_id(results, mid))
    for mid in raw.get("by_method") or {}:
        mid_s = str(mid)
        if table_visible_method(results, mid_s):
            found.append(pair_qualify_method_id(results, mid_s))
    # Sort only real causal hits — do not invent empty headline shells.
    uniq = sorted(set(found), key=lambda m: _causal_sort_key(m, headlines))
    return tuple(uniq)


def causal_table(results: Dict[str, Any]) -> str:
    """LMS×0.99-gated causal effects by method (includes per-layer ACTIEND when present)."""
    methods = _causal_methods(results)
    by_row = method_rows_by_display_id(results)
    for row in results.get("methods") or []:
        mid0 = str(row.get("method") or "")
        if mid0 and mid0 not in by_row:
            by_row[mid0] = row  # type: ignore[assignment]
    by_method = {
        mid: (row.get("metrics") or {})
        for mid, row in by_row.items()
    }
    raw = (results.get("raw") or {}).get("causal") or {}
    for s in raw.get("summaries") or []:
        if not isinstance(s, dict):
            continue
        mid = s.get("method")
        if not mid:
            continue
        existing = by_method.get(mid) or {}
        if existing.get("causal_signed_effect") is not None or existing.get(
            "causal_selected_strength"
        ) is not None:
            by_method[mid] = existing
            continue
        hm = s.get("headline_metrics")
        if isinstance(hm, dict) and hm:
            by_method[mid] = {**existing, **hm}
            continue
        sel = s.get("selected") or {}
        by_method[mid] = {
            **existing,
            "causal_selected_strength": s.get("selected_strength") or sel.get("strength"),
            "causal_signed_effect": sel.get("signed_effect"),
            "causal_lms": sel.get("lms"),
            "causal_lms_ok": sel.get("lms_ok"),
            "causal_delta_other": sel.get("signed_effect"),
        }
    col_w = max(40, max((len(str(m)) for m in methods), default=40))
    lines = [
        "",
        "Causal (LMS×0.99; GRADIEND/ACTIEND str = decoder-plot LR):",
        "  str = package decoder star when present; else LMS×0.99 SAE/CAA pick.",
        "  eff/lms/ok = values at that str; dtgt/doth/dneu/rnd = group deltas there.",
        "  effn = eff / (1 - causal_base_p): headroom-normalized effectiveness",
        "         (AxBench-style); blank when remaining headroom < 0.02.",
        method_id_legend(enabled_method_families(results)),
        "  flag meanings:",
        "    ceiling = selected strength == GRID MAX → try higher LR before concluding",
        "              the method is ineffective; effect may still grow.",
        "    floor   = selected strength == GRID MIN (tiny).",
        "    null    = |signed_effect|~0 at the selected point.",
        "    floor+null = tiny strength AND ~0 effect → no useful causal effect on this grid.",
        "    gate_empty = no LMS-passing strength (fallback used).",
        "    spec_fail  = |eff| does not exceed the matched random-direction control",
        "                 (causal_beats_random_control=False) — effect isn't clearly",
        "                 distinguishable from perturbing the model by that much in any",
        "                 direction. See causal_specificity_ratio for the |eff|/|rnd| ratio.",
        f"{'method':<{col_w}} {'str':>8} {'eff':>8} {'effn':>8} {'lms':>7} {'ok':>4} "
        f"{'dtgt':>8} {'doth':>8} {'dneu':>8} {'rnd':>8} {'flag'}",
        "-" * (col_w + 88),
    ]
    for method in methods:
        m = _causal_metrics_for_id(by_method, method)
        flag_parts = []
        if m.get("causal_grid_ceiling"):
            flag_parts.append("ceiling")
        elif m.get("causal_grid_floor"):
            flag_parts.append("floor")
        if m.get("causal_null_effect"):
            flag_parts.append("null")
        if m.get("causal_gate_empty"):
            flag_parts.append("gate_empty")
        if m.get("causal_beats_random_control") is False:
            flag_parts.append("spec_fail")
        flag = "+".join(flag_parts) if flag_parts else "—"

        def _ok(v: Any) -> str:
            if v is None:
                return "—"
            return "yes" if v else "no"

        missing = m.get("causal_signed_effect") is None and m.get("causal_delta_mean") is None
        if missing:
            err = (
                m.get("causal_error")
                or (by_row.get(method) or {}).get("error")
                or ""
            )
            if err:
                flag = str(err).replace("\n", " ")[:90]
            elif flag == "—":
                flag = "no causal"
            lines.append(
                f"{method:<{col_w}} "
                + " ".join(f"{'—':>8}" for _ in range(4))
                + f" {'—':>4} "
                + " ".join(f"{'—':>8}" for _ in range(4))
                + f" {flag}"
            )
            continue
        lines.append(
            f"{method:<{col_w}} "
            f"{_fmt_metric(m.get('causal_selected_strength'))} "
            f"{_fmt_metric(m.get('causal_signed_effect', m.get('causal_delta_mean')))} "
            f"{_fmt_metric(m.get('causal_effectiveness'))} "
            f"{_fmt_metric(m.get('causal_lms'), width=7)} "
            f"{_ok(m.get('causal_lms_ok')):>4} "
            f"{_fmt_metric(m.get('causal_delta_target'))} "
            f"{_fmt_metric(m.get('causal_delta_other', m.get('causal_signed_effect')))} "
            f"{_fmt_metric(m.get('causal_delta_neutral'))} "
            f"{_fmt_metric(m.get('causal_random_signed_effect'))} "
            f"{flag}"
        )
    from causal_eval import specificity_control_summary

    spec = specificity_control_summary(by_method.values())
    if spec["n_total"]:
        frac = spec["fraction_exceeds_random_control"]
        med = spec["median_specificity_ratio"]
        lines.append(
            f"  specificity: {spec['n_exceeds_random_control']}/{spec['n_total']} "
            f"({frac:.0%}) causal methods exceed their matched random-direction "
            f"control"
            + (f"; median |eff|/|rnd| = {med:.2f}" if med is not None else "")
        )
    return "\n".join(lines)






def _metric_lookup(metrics: Dict[str, Any], fields: Sequence[str]) -> Optional[float]:
    for f in fields:
        v = metrics.get(f)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def none_vs_tensor_pairs(
    results: Dict[str, Any],
    *,
    backends: Optional[Sequence[str]] = None,
    target_classes: Optional[Sequence[str]] = None,
    abs_eps: float = 1e-4,
    rel_eps: float = 1e-3,
    include_caa: Optional[bool] = None,
    caa_act_policies: Sequence[str] = ("prediction", "mean", "last"),
) -> List[Dict[str, Any]]:
    """Paired encoding rows: ``backend:cls`` (none) vs ``backend:cls:tensors`` (by_tensor).

    Each pair includes per-metric ``none`` / ``tensors`` / ``delta`` (= tensors − none)
    and a coarse ``winner`` (``tensors`` / ``none`` / ``tie`` / ``incomplete``).
    Metric dicts also keep legacy key ``all`` (= tensors) for older report code.

    CAA pairs: ``caa:{cls}:act_{policy}`` vs ``caa:{cls}:all_act_{policy}``
    (``all_act_*`` = multi-layer CAA, not GradiendSplit).
    Disabled families (task ``methods.sae/caa: false``) are omitted by default.
    """
    enabled = enabled_method_families(results)
    if backends is None:
        backends_list: List[str] = []
        if "gradiend" in enabled:
            backends_list.append("gradiend")
        if "actiend" in enabled:
            backends_list.append("actiend")
        if "actiend_pre" in enabled:
            backends_list.append("actiend_pre")
        backends = tuple(backends_list)
    if include_caa is None:
        include_caa = "caa" in enabled
    classes = (
        [str(c) for c in target_classes]
        if target_classes
        else require_claim_classes(results)
    )
    by = {
        str(r.get("method")): (r.get("metrics") or {})
        for r in (results.get("methods") or [])
        if r.get("method")
    }
    id_pairs: List[Tuple[str, str, str, str]] = []
    for backend in backends:
        for cls in classes:
            id_pairs.append(
                (str(backend), str(cls), f"{backend}:{cls}", f"{backend}:{cls}:tensors")
            )
    if include_caa:
        for pol in caa_act_policies:
            for cls in classes:
                id_pairs.append(
                    (
                        f"caa:{pol}",
                        str(cls),
                        f"caa:{cls}:act_{pol}",
                        f"caa:{cls}:all_act_{pol}",
                    )
                )

    def _metrics_for(*candidates: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Prefer canonical id; accept legacy ``:all`` gender-PoC ids."""
        for mid in candidates:
            if mid in by:
                return by[mid], mid
        return {}, None

    pairs: List[Dict[str, Any]] = []
    for backend, cls, none_id, tensors_id in id_pairs:
        none_m, resolved_none = _metrics_for(none_id)
        tensors_candidates = [tensors_id]
        if tensors_id.endswith(":tensors"):
            tensors_candidates.append(f"{tensors_id[: -len(':tensors')]}:all")
        all_m, resolved_all = _metrics_for(*tensors_candidates)
        metrics_out: Dict[str, Any] = {}
        votes_tensors = votes_none = votes_tie = 0
        n_comparable = 0
        for col, fields, higher_better in NONE_VS_TENSOR_METRICS:
            nv = _metric_lookup(none_m, fields)
            av = _metric_lookup(all_m, fields)
            delta = None if nv is None or av is None else av - nv
            winner = "incomplete"
            if delta is not None:
                n_comparable += 1
                scale = max(abs(nv or 0.0), abs(av or 0.0), 1.0)
                if abs(delta) <= max(abs_eps, rel_eps * scale):
                    winner = "tie"
                    votes_tie += 1
                elif (delta > 0) == higher_better:
                    winner = "tensors"
                    votes_tensors += 1
                else:
                    winner = "none"
                    votes_none += 1
            metrics_out[col] = {
                "none": nv,
                "tensors": av,
                "all": av,  # legacy alias
                "delta": delta,
                "winner": winner,
            }
        if n_comparable == 0:
            pair_winner = "incomplete"
        elif votes_tensors == 0 and votes_none == 0:
            pair_winner = "tie"
        elif votes_none == 0 and votes_tensors > 0:
            pair_winner = "tensors"
        elif votes_tensors == 0 and votes_none > 0:
            pair_winner = "none"
        elif votes_tensors > votes_none:
            pair_winner = "tensors"
        elif votes_none > votes_tensors:
            pair_winner = "none"
        else:
            pair_winner = "mixed"
        pairs.append(
            {
                "backend": backend,
                "class": cls,
                "none_id": resolved_none or none_id,
                "all_id": resolved_all or tensors_id,
                "tensors_id": resolved_all or tensors_id,
                "present_none": resolved_none is not None,
                "present_all": resolved_all is not None,
                "present_tensors": resolved_all is not None,
                "metrics": metrics_out,
                "votes": {
                    "tensors": votes_tensors,
                    "all": votes_tensors,  # legacy alias
                    "none": votes_none,
                    "tie": votes_tie,
                    "n": n_comparable,
                },
                "winner": pair_winner,
            }
        )
    return pairs


def none_vs_tensor_table(results: Dict[str, Any]) -> str:
    """Plain-text paired none vs by_tensor encoding table with Δ columns."""
    pairs = none_vs_tensor_pairs(results)
    lines = [
        "",
        "Encoding split ablation: none (GradiendSplit.none) vs by_tensor aggregate (:tensors).",
        "  delta = :tensors - none (positive => by_tensor better for that metric).",
        "  Scope: encoder class-vs-neutral only — not a matched causal none-vs-tensor test.",
        "",
        f"{'pair':<14} {'split':<6} {'auc_n':>7} {'auc_o':>7} {'bal':>7} {'d':>8} "
        f"{'spec':>7} {'excl':>7} {'smag':>7} {'emag':>7} {'verdict'}",
        "-" * 112,
    ]
    if not any(p.get("present_none") or p.get("present_all") for p in pairs):
        lines.append("(no none / :tensors encoding pairs in results — re-run with both split modes)")
        return "\n".join(lines)

    for p in pairs:
        label = f"{p['backend']}:{p['class']}"
        for split_key, split_lab in (("none", "none"), ("tensors", ":tensors")):
            vals = []
            for col, _, _ in NONE_VS_TENSOR_METRICS:
                if col in {"J"}:
                    continue
                cell = (p["metrics"].get(col) or {}).get(split_key)
                vals.append(_fmt_metric(cell, width=7 if col != "d" else 8))
            # drop J from display (already skipped); we have auc_n..emag = 8 cols
            while len(vals) < 8:
                vals.append(_fmt_metric(None, width=7))
            tag = p["winner"] if split_key == "tensors" else ""
            lines.append(
                f"{label:<14} {split_lab:<6} "
                f"{vals[0]} {vals[1]} {vals[2]} {vals[3]} "
                f"{vals[4]} {vals[5]} {vals[6]} {vals[7]} {tag}"
            )
        # delta row
        dvals = []
        for col, _, _ in NONE_VS_TENSOR_METRICS:
            if col == "J":
                continue
            cell = (p["metrics"].get(col) or {}).get("delta")
            dvals.append(_fmt_metric(cell, width=7 if col != "d" else 8))
        while len(dvals) < 8:
            dvals.append(_fmt_metric(None, width=7))
        lines.append(
            f"{'':<14} {'delta':<6} "
            f"{dvals[0]} {dvals[1]} {dvals[2]} {dvals[3]} "
            f"{dvals[4]} {dvals[5]} {dvals[6]} {dvals[7]}"
        )
    lines.append("")
    lines.append(none_vs_tensor_verdict(results).strip())
    return "\n".join(lines)


def none_vs_tensor_verdict(results: Dict[str, Any]) -> str:
    """Human-readable conclusion for the none vs by_tensor encoding ablation."""
    pairs = none_vs_tensor_pairs(results)
    if not any(p.get("present_none") and p.get("present_all") for p in pairs):
        return (
            "Verdict: incomplete — need both `:{cls}` (none) and `:{cls}:tensors` "
            "(by_tensor) encoder rows to conclude. Causal none-vs-tensor is out of scope "
            "(aggregate causal uses none only)."
        )

    by_backend: Dict[str, List[str]] = {}
    for p in pairs:
        if not (p.get("present_none") and p.get("present_all")):
            by_backend.setdefault(p["backend"], []).append(f"{p['class']}=incomplete")
            continue
        by_backend.setdefault(p["backend"], []).append(f"{p['class']}={p['winner']}")

    # Discriminating metrics when AUCs saturate near 1.
    disc_cols = ("bal", "d", "spec", "excl", "smag", "emag")
    bullets: List[str] = [
        "Verdict (encoding only; delta = :tensors - none):",
    ]
    for backend, outcomes in by_backend.items():
        counts = {"tensors": 0, "all": 0, "none": 0, "tie": 0, "mixed": 0, "incomplete": 0}
        for o in outcomes:
            w = o.split("=", 1)[-1]
            counts[w] = counts.get(w, 0) + 1
        if counts["incomplete"] and counts["tensors"] + counts["all"] + counts["none"] + counts["tie"] + counts["mixed"] == 0:
            bullets.append(f"  - {backend}: incomplete pairs")
            continue
        if counts["tie"] and not (counts["tensors"] or counts["all"] or counts["none"] or counts["mixed"]):
            bullets.append(
                f"  - {backend}: tie — by_tensor :tensors matches none on encoding "
                f"({', '.join(outcomes)}). Partitioning does not change the joint readout."
            )
        elif (counts["tensors"] or counts["all"]) and not counts["none"]:
            bullets.append(
                f"  - {backend}: :tensors edges none on some secondary metrics "
                f"({', '.join(outcomes)}). Ranking metrics (auc_n/auc_o) are often "
                f"already saturated for both — do not treat this as a strong preference."
            )
        elif counts["none"] and not (counts["tensors"] or counts["all"]):
            bullets.append(
                f"  - {backend}: none edges :tensors on some secondary metrics "
                f"({', '.join(outcomes)}). Again auc often saturated; small bal/d deltas only."
            )
        else:
            bullets.append(
                f"  - {backend}: mixed / inconclusive ({', '.join(outcomes)}). "
                f"auc_n/auc_o usually ceilinged; bal/d/smag deltas are secondary only."
            )

        # Highlight discriminating deltas
        highlights = []
        for p in pairs:
            if p["backend"] != backend or not (p["present_none"] and p["present_all"]):
                continue
            for col in disc_cols:
                cell = p["metrics"].get(col) or {}
                d = cell.get("delta")
                if isinstance(d, float) and abs(d) > 1e-3:
                    highlights.append(
                        f"{p['class']}.{col} delta={d:+.4f} ({cell.get('winner')})"
                    )
        if highlights:
            bullets.append(f"    discriminating: {'; '.join(highlights[:8])}")

    bullets.append(
        "  - Causal none vs :tensors: run as matched ablations when causal stage executes "
        "(gradiend:{cls}:tensors; actiend:{cls}:tensors_tok_*; plus per-layer :L*_tok_*). "
        "Re-run causal if results predate this."
    )
    return "\n".join(bullets)
