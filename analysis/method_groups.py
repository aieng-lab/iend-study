"""Roll up per-class / ablation method rows into headline comparison groups.

Raw ``results.json`` dumps expose many method ids (layers, tok policies, pair vs
one-pole variants).  For cross-task overviews we collapse those into a small set
of rows such as ``gradiend:two_pole``, ``gradiend:one_pole``, ``sae:kstar``, and
``caa:two_pole`` and ``caa:one_pole``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from analysis.task_specs import (
    GRADIEND_LIKE_BACKENDS,
    GAP_DISPLAY,
    MISSING_DISPLAY,
    group_is_applicable,
    resolve_spec,
    spec_for_task,
)
from results_schema import fair_metric_fields, _has_encoder_metrics, require_claim_classes, require_target_classes
from study.method_ids import is_pair_key, normalize_sae_method_id
from study.validation_selection import (
    lock_by_validation_detection,
)
from suitability import (
    compute_feature_suitability,
    detection_score,
    encoding_e,
    intervention_score,
    resolve_causal_metrics,
)

ROOT = Path(__file__).resolve().parents[1]

def _encoding_e(m: Mapping[str, Any]) -> Optional[float]:
    """Fair encoding bottleneck E (same rules as suitability, without causal gate)."""
    return encoding_e(m)


# Numeric fields averaged when building a group row.
HEADLINE_METRICS: Tuple[str, ...] = (
    "encoding_E",
    "detection_score",
    "roc_auc",
    "roc_auc_neutral",
    "roc_auc_other",
    "balanced_accuracy",
    "cohens_d",
    "neutral_specificity",
    "specificity",
    "class_exclusivity",
    "min_pairwise_auroc",
    "encoder_correlation",
    "class_separation",
    "suitability",
    "suitability_E",
    "suitability_G",
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "intervention_score",
    "causal_base_p",
    "causal_lms",
    "causal_weaken_lms",
    "causal_effectiveness",
)

DETECTION_MEMBER_AGGREGATION = "mean"

# Appendix-only representation views.  ``best_overall`` is the paper default:
# validation selects from the aggregate and every physical layer.  The other
# two views expose the predeclared comparison without touching held-out data.
REPRESENTATION_VIEWS = frozenset({"best_overall", "all_layer", "best_single_layer"})

POLE_LABELS = {"pair": "two_pole", "one_pole": "one_pole"}

# Ridge is a causal steering baseline, not an encoder.  Keep its rows in the
# causal pivots, but never render a meaningless all-NaN ridge row in encoder
# metric tables.
CAUSAL_ONLY_GROUP_PREFIXES: Tuple[str, ...] = ("actiend_ridge:",)
CAUSAL_METRICS: Tuple[str, ...] = (
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "intervention_score",
    "causal_base_p",
    "causal_lms",
    "causal_weaken_lms",
    "causal_effectiveness",
)

# Stable row order for method_group pivots (not MEAN-sorted).
METHOD_GROUP_ORDER: Tuple[str, ...] = (
    "gradiend:two_pole",
    "gradiend:one_pole",
    "actiend:two_pole",
    "actiend:one_pole",
    "actiend_ridge:two_pole",
    "actiend_ridge:one_pole",
    "actiend_pre:two_pole",
    "actiend_pre:one_pole",
    "cga:two_pole",
    "cga:one_pole",
    "cga_tensor_norm:two_pole",
    "cga_tensor_norm:one_pole",
    "caga:two_pole",
    "caga:one_pole",
    "agiend:two_pole",
    "agiend:one_pole",
    "sae:kstar",
    "sae:k1",
    "sae_pre:kstar",
    "sae_pre:k1",
    "sae:joint",
    "sae_pre:joint",
    "sae",
    "sae_pre",
    "caa:two_pole",
    "caa:one_pole",
)

# SAE readout tails that map to a headline group (backend prefix added below).


def _iter_results_json(runs_root: Path) -> Iterable[Path]:
    runs_root = Path(runs_root)
    seen: set[Path] = set()
    for pattern in ("*/*/results.json", "*/results.json"):
        for path in sorted(runs_root.glob(pattern)):
            # Ignore archived trees (e.g. runs/gpt2-small_old/*) by default.
            parts = path.parts
            model_dir = parts[-3] if len(parts) >= 3 else ""
            if str(model_dir).endswith("_old"):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


def _load_results(path: Path) -> Optional[Dict[str, Any]]:
    # A sync that overwrote results.json without merging its pre-pull snapshot
    # leaves a file missing local-only methods.  Never let that reach a table.
    from study.snapshot_guard import assert_resolved

    assert_resolved(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    from causal_eval import apply_package_decoder_headlines

    return apply_package_decoder_headlines(payload)


def _method_parts(method_id: str) -> List[str]:
    return [p for p in str(method_id).split(":") if p]


def _pole_from_row(row: Mapping[str, Any]) -> Optional[str]:
    metrics = row.get("metrics") or {}
    abl = metrics.get("ablation")
    if abl in POLE_LABELS:
        return str(abl)
    # Ridge decoder rows are causal-only, so they do not inherit the training
    # row's ``ablation`` metric. Their ids still encode pair vs one-pole.
    parts = _method_parts(str(row.get("method") or ""))
    if parts and parts[0] == "caa":
        if len(parts) >= 4 and is_pair_key(parts[1]):
            return "pair"
        if len(parts) >= 3:
            # Legacy CAA was one-vs-rest but did not record its pole regime.
            return "one_pole"
    if parts and parts[0] == "actiend_ridge":
        if len(parts) >= 3 and "-" in parts[1]:
            return "pair"
        if len(parts) >= 2:
            return "one_pole"
    # Activation-gradient rows encode the pole in the id: {caga,agiend}:A-B:C (pair)
    # vs {caga,agiend}:C (one-pole) -- fallback when no encoding row supplies ``ablation``.
    if parts and parts[0] in {"caga", "agiend"}:
        if len(parts) >= 3 and is_pair_key(parts[1]):
            return "pair"
        if len(parts) >= 2:
            return "one_pole"
    return None


def _best_encoder_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        metrics = dict(row.get("metrics") or {})
        if not _has_encoder_metrics(metrics):
            continue
        candidates.append({"method": row.get("method"), "metrics": metrics, "status": row.get("status")})
    if not candidates:
        return None

    def _score(item: Mapping[str, Any]) -> Tuple[int, float]:
        status = str(item.get("status") or "")
        rank = 1 if status in {"ok", "partial"} else 0
        m = item.get("metrics") or {}
        v = m.get("roc_auc_neutral", m.get("roc_auc"))
        auc = float(v) if isinstance(v, (int, float)) else -1.0
        return (rank, auc)

    return max(candidates, key=_score)


def _dedupe_method_rows(method_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, List[Mapping[str, Any]]] = {}
    for row in method_rows:
        # Causal refreshes often retain the encoder/readout payload under
        # ``extras`` while replacing the flat metrics with causal-only fields.
        # Normalize that representation before grouping so SAE/CGA/CAGA use
        # the same table input.
        row = dict(row)
        metrics = dict(row.get("metrics") or {})
        extras = row.get("extras") or {}
        for field_name in ("encoder_metrics", "readout_metrics"):
            embedded = extras.get(field_name)
            if isinstance(embedded, Mapping):
                for field, value in embedded.items():
                    if metrics.get(field) is None and value is not None:
                        metrics[field] = value
        row["metrics"] = metrics
        mid = normalize_sae_method_id(str(row.get("method") or ""))
        if not mid:
            continue
        abl = (row.get("metrics") or {}).get("ablation")
        key = f"{mid}|{abl}" if abl else mid
        by_key.setdefault(key, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for key, rows in by_key.items():
        best = _best_encoder_row(rows)
        if best is not None:
            # Encode and causal refreshes can emit the same method id as
            # separate rows.  Keep the best encoder row for selection, but
            # do not discard causal fields from its companion row.
            merged = dict(best)
            merged_metrics = dict(best.get("metrics") or {})
            merged_extras = dict(best.get("extras") or {})
            for row in rows:
                for field, value in (row.get("metrics") or {}).items():
                    if merged_metrics.get(field) is None and value is not None:
                        merged_metrics[field] = value
                for field, value in (row.get("extras") or {}).items():
                    if field not in merged_extras or merged_extras[field] is None:
                        merged_extras[field] = value
            merged["metrics"] = merged_metrics
            if merged_extras:
                merged["extras"] = merged_extras
            out[key] = merged
            continue
        # ACTIEND-ridge is intentionally causal-only. Retain its headline row
        # even though it has no encoder metric to rank it by.
        ridge_rows = [
            row
            for row in rows
            if str(row.get("method") or "").startswith("actiend_ridge:")
            and any(
                (row.get("metrics") or {}).get(field) is not None
                for field in ("causal_signed_effect", "causal_signed_effect_weaken")
            )
        ]
        if ridge_rows:
            out[key] = dict(ridge_rows[-1])
            continue
        # Some pair encoder rows are intentionally correlation-only in the
        # flat schema; their fair class readouts live in
        # ``extras.per_class_readouts`` and are promoted by _collect_buckets.
        # Keep the base row long enough for that promotion/join to happen.
        split_readout_rows = [
            row
            for row in rows
            if str(row.get("method") or "").split(":", 1)[0] in GRADIEND_LIKE_BACKENDS
            and str(row.get("method") or "").split(":", 1)[0] != "actiend_ridge"
            and isinstance((row.get("extras") or {}).get("per_class_readouts"), Mapping)
            and bool((row.get("extras") or {}).get("per_class_readouts"))
        ]
        if split_readout_rows:
            out[key] = dict(split_readout_rows[-1])
    return out


def _caa_encode_rows_from_raw(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Restore CAA encoder rows when an old causal refresh overwrote their ids.

    CAA encode and causal results share method ids.  The detailed encode rows
    remain in ``raw.caa.method_metrics`` even in affected legacy dumps.
    """
    metrics_by_id = ((results.get("raw") or {}).get("caa") or {}).get("method_metrics") or {}
    rows: List[Dict[str, Any]] = []
    for method, readout in metrics_by_id.items():
        if not isinstance(readout, Mapping) or not _has_encoder_metrics(dict(readout)):
            continue
        rows.append(
            {
                "method": str(method),
                "status": "error" if readout.get("error") else "ok",
                "metrics": {
                    **fair_metric_fields(readout),
                    # Legacy CAA dumps predate the full ``val_readout`` blob
                    # but do carry the scalar validation bottleneck used at
                    # the time.  Keep it so _validation_detection can perform
                    # its documented compatibility fallback.
                    "val_encoding_E": readout.get("val_encoding_E"),
                    "backend": "caa",
                    "readout_kind": "class_vs_neutral",
                    "target_class": readout.get("target_class"),
                    "act_policy": readout.get("act_policy"),
                    "component_part": readout.get("component_part"),
                    "score": readout.get("score") or "cosine",
                    "ablation": readout.get("ablation") or "one_pole",
                    "pair": readout.get("pair"),
                    "vector_key": readout.get("vector_key"),
                },
            }
        )
    return rows


def _gradient_encode_rows_from_artifacts(
    results: Mapping[str, Any], results_path: str
) -> List[Dict[str, Any]]:
    """Restore CGA/CAGA/AGIEND encoder rows from train ``done.json`` files.

    Method-isolated causal writers historically replaced shared method rows
    with causal-only shells.  The expensive encoder evaluation is still
    persisted under ``artifacts/*/done.json:extras.encoder_eval``.  Reports
    must consume that canonical payload instead of rendering ``NaN`` or
    requiring a GPU rerun.  Aggregate and per-layer rows are both restored so
    the validation-locked layer choice remains identical to a clean run.
    """
    if not results_path:
        return []
    artifacts = Path(results_path).parent / "artifacts"
    if not artifacts.is_dir():
        return []
    backends = {"cga", "cga_tensor_norm", "caga", "agiend"}
    rows: List[Dict[str, Any]] = []

    def _append_row(
        *,
        backend: str,
        pole: str,
        identity: str,
        cls: str,
        readout: Mapping[str, Any],
        done_path: Path,
        encoder_eval: Mapping[str, Any],
        component: Optional[str] = None,
    ) -> None:
        metrics = {
            **fair_metric_fields(readout),
            "val_encoding_E": readout.get("val_encoding_E"),
            "backend": backend,
            "target_class": str(cls),
            "readout_kind": "class_vs_neutral",
            "component_part": component,
            "ablation": "pair" if pole == "pair" else "one_pole",
            "pair": identity.split("-") if pole == "pair" else None,
        }
        if not _has_encoder_metrics(metrics):
            return
        method_parts = [backend]
        if pole == "pair":
            method_parts.extend((identity, str(cls)))
        else:
            method_parts.append(str(cls))
        if component:
            method_parts.append(str(component))
        status = "collapsed" if readout.get("collapsed_encoder") else (
            "error" if readout.get("error") else "ok"
        )
        rows.append(
            {
                "method": ":".join(method_parts),
                "status": status,
                "metrics": metrics,
                "artifacts": {"experiment_dir": str(done_path.parent)},
                "extras": {
                    "encoder_metrics": dict(encoder_eval.get("encoder_metrics") or {}),
                    "readout_metrics": dict(readout),
                },
            }
        )

    for done_path in sorted(artifacts.glob("*__*__*/done.json")):
        name_parts = done_path.parent.name.split("__", 2)
        if len(name_parts) != 3:
            continue
        backend, pole, identity = name_parts
        if backend not in backends or pole not in {"pair", "onepole"}:
            continue
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        encoder_eval = ((done.get("extras") or {}).get("encoder_eval") or {})
        if not isinstance(encoder_eval, Mapping):
            continue
        per_class = encoder_eval.get("per_class_readouts") or {}
        if not isinstance(per_class, Mapping):
            continue
        pole_name = "pair" if pole == "pair" else "onepole"
        # Encoder evaluation may include readouts for every target class even
        # when this artifact was trained/evaluated as a single one-pole
        # direction.  Only the artifact identity has a matching causal sweep;
        # restoring distractor readouts here makes group-level causal
        # completeness fail and incorrectly renders the valid pole as NaN.
        allowed_classes = (
            {str(value) for value in identity.split("-")}
            if pole == "pair"
            else {str(identity)}
        )
        for cls, readout in per_class.items():
            if (
                isinstance(readout, Mapping)
                and (allowed_classes is None or str(cls) in allowed_classes)
            ):
                _append_row(
                    backend=backend,
                    pole=pole_name,
                    identity=identity,
                    cls=str(cls),
                    readout=readout,
                    done_path=done_path,
                    encoder_eval=encoder_eval,
                )
        per_component = encoder_eval.get("per_component_readouts") or {}
        if isinstance(per_component, Mapping):
            for component, class_map in per_component.items():
                if not isinstance(class_map, Mapping):
                    continue
                for cls, readout in class_map.items():
                    if (
                        isinstance(readout, Mapping)
                        and (allowed_classes is None or str(cls) in allowed_classes)
                    ):
                        _append_row(
                            backend=backend,
                            pole=pole_name,
                            identity=identity,
                            cls=str(cls),
                            readout=readout,
                            done_path=done_path,
                            encoder_eval=encoder_eval,
                            component=str(component),
                        )
    return rows


def _sae_encode_rows_from_artifacts(
    results: Mapping[str, Any], results_path: str
) -> List[Dict[str, Any]]:
    """Restore SAE encoder rows that the encode stage wrote only to its artifact.

    ``results.json``'s SAE recipe rows (``k1``/``all_k1``/``kstar``) carry causal
    but no encoder metrics, so they are skipped when the site pool is built: the
    ``k1`` pool shrinks to the bare per-layer rows (its own ``k1``/``all_k1``
    recipes drop out of their own selection) and ``kstar`` gets no pool at all,
    which drops the group entirely.

    ``artifacts/sae/encode_method_rows.json`` holds the full encode output --
    every Detection input -- for all of them, and is
    synced by ``scripts/rsync_analysis.sh``. Mirrors ``_caa_encode_rows_from_raw``.

    Only ids whose ``results.json`` row is missing or lacks encoder metrics are
    restored, so a live row always wins over the artifact copy.
    """
    if not results_path:
        return []
    path = Path(results_path).parent / "artifacts" / "sae" / "encode_method_rows.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    if not isinstance(payload, Mapping):
        return []
    # The artifact is addressed by a caller-supplied path, so confirm it belongs
    # to this run before merging: a mismatched path must not inject another
    # model/task's rows into these results.
    for field in ("model", "task"):
        want = results.get(field)
        got = payload.get(field)
        if want and got and str(want) != str(got):
            return []
    entries = payload.get("methods")
    if not isinstance(entries, list):
        return []

    from_artifact: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        mid = str(entry.get("method") or "")
        metrics = dict(entry.get("metrics") or {})
        if not mid or not _has_encoder_metrics(metrics):
            continue
        from_artifact[mid] = {
            "method": mid,
            "status": entry.get("status") or "ok",
            "metrics": metrics,
        }
    if not from_artifact:
        return []

    # A live row can carry encoder metrics yet still lack the validation readout
    # the site lock ranks on, which would leave its pool unselectable. Merge per
    # field -- a non-null live value always wins, the artifact only fills gaps --
    # so the live result is never replaced, only completed.
    # Older SAE artifacts contain complete layerwise rows, but the canonical
    # ``...:k1`` row was written without its validation readout.  Reconstruct
    # only that small selection record from the already-computed layer rows;
    # never recompute features or causal interventions.
    for mid, entry in list(from_artifact.items()):
        parts = _method_parts(mid)
        if len(parts) != 3 or parts[0] not in {"sae", "sae_pre"} or parts[2] != "k1":
            continue
        if isinstance((entry.get("metrics") or {}).get("val_readout"), Mapping):
            continue
        prefix = f"{parts[0]}:{parts[1]}:"
        layer_candidates = []
        for candidate_id, candidate in from_artifact.items():
            if not candidate_id.startswith(prefix):
                continue
            tail = candidate_id[len(prefix):]
            if not (re.fullmatch(r"L\d+", tail) or re.fullmatch(r"L\d+_k1", tail)):
                continue
            val_readout = (candidate.get("metrics") or {}).get("val_readout")
            score = detection_score(val_readout) if isinstance(val_readout, Mapping) else None
            if isinstance(score, (int, float)):
                layer_candidates.append((float(score), candidate_id, candidate))
        if not layer_candidates:
            continue
        _, selected_id, selected = max(layer_candidates, key=lambda item: item[0])
        causal_fields = {
            "causal_signed_effect",
            "causal_signed_effect_weaken",
            "causal_base_p",
            "causal_lms",
            "causal_weaken_lms",
            "causal_weaken_lms_ok",
            "causal_lms_ok",
            "causal_selected_strength",
        }
        original = entry["metrics"]
        repaired = dict(selected.get("metrics") or {})
        repaired.update({k: v for k, v in original.items() if k in causal_fields and v is not None})
        repaired["layer_selection"] = "validation_detection"
        repaired["selected_layer"] = selected_id.rsplit(":", 1)[-1]
        entry["metrics"] = repaired
        entry["_repaired_from_layer"] = True

    rows: List[Dict[str, Any]] = []
    for row in results.get("methods") or []:
        if not isinstance(row, Mapping):
            continue
        mid = str(row.get("method") or "")
        base = from_artifact.pop(mid, None)
        if base is None:
            continue
        merged = dict(base["metrics"])
        live_metrics = row.get("metrics") or {}
        if base.get("_repaired_from_layer"):
            # The live canonical k1 row is the incomplete legacy copy. Keep
            # only its causal fields; the artifact-derived encoder payload is
            # the repaired, validation-selected one.
            merged.update({
                k: v for k, v in live_metrics.items()
                if k.startswith("causal_") and v is not None
            })
        else:
            merged.update({k: v for k, v in live_metrics.items() if v is not None})
        rows.append({**dict(row), "metrics": merged})
    rows.extend(from_artifact.values())
    return rows


def _class_headline_gradiend_actiend(
    method_id: str,
    *,
    pole: str,
    claim_classes: Sequence[str],
    target_classes: Sequence[str],
) -> bool:
    parts = _method_parts(method_id)
    if not parts or parts[0] not in GRADIEND_LIKE_BACKENDS:
        return False
    if parts[0] == "actiend_ridge":
        if len(parts) >= 3 and "-" in parts[1]:
            return pole == "pair" and parts[2] in claim_classes
        return len(parts) >= 2 and pole == "one_pole" and parts[1] in claim_classes
    if len(parts) == 1:
        return pole == "pair"
    if len(parts) == 3 and "-" in parts[1]:
        cls = parts[2]
        if pole == "one_pole":
            return False
        return cls in claim_classes or cls in target_classes
    if len(parts) != 2:
        return False
    cls = parts[1]
    if pole == "one_pole":
        return cls in claim_classes
    # AGIEND/CAGA pair aggregates use ids such as ``agiend:asian-black``;
    # the pair key itself is the headline candidate, not a single class.
    return is_pair_key(cls) or cls in target_classes




def _sae_k1_site_tail(tail: str) -> bool:
    """Canonical k=1 headline candidates.

    ``L*_k1`` rows are layerwise diagnostics.  They must not compete with
    the canonical ``k1``/``all_k1`` recipes in the headline aggregation:
    ``k1`` already represents the selected best single-layer result.
    """
    return tail in {"k1", "all_k1"}


def _sae_kstar_site_tail(tail: str) -> bool:
    """k* site candidates: selected-layer k* bag vs all-layers bags with k>1.

    ``all_k1`` belongs to the k=1 recipe, not k*.
    """
    if tail == "kstar":
        return True
    m = re.fullmatch(r"all_k(\d+)", tail)
    return bool(m) and int(m.group(1)) != 1


def _caa_site_tail(tail: str) -> bool:
    """CAA paper column candidates: concat and per-layer.

    The concat cosine is CAA's single all-layer candidate, matching the
    aggregate cosine for CAGA/CGA. Mean-of-layer cosine is an appendix
    diagnostic, not a second headline all-layer candidate.
    """
    if tail == "act_prediction":
        return True
    return bool(re.fullmatch(r"L\d+_act_prediction", tail))


_LAYERWISE_MEAN_BACKENDS = frozenset({"cga", "cga_tensor_norm", "caga"})


def _layerwise_mean_candidate(
    parts: Sequence[str], *, pole: str
) -> Optional[Tuple[str, str]]:
    """Return ``(class, contrast_key)`` for aggregate/per-layer CGA/CAGA ids."""
    if not parts or parts[0] not in _LAYERWISE_MEAN_BACKENDS:
        return None
    if pole == "pair":
        if len(parts) not in {3, 4} or not is_pair_key(parts[1]):
            return None
        if len(parts) == 4 and not re.fullmatch(r"L\d+", parts[3]):
            return None
        return str(parts[2]), str(parts[1])
    if pole == "one_pole":
        if len(parts) not in {2, 3}:
            return None
        if len(parts) == 3 and not re.fullmatch(r"L\d+", parts[2]):
            return None
        return str(parts[1]), str(parts[1])
    return None


def _is_physical_layer_candidate(method_id: str) -> bool:
    """Whether a CGA/CAGA candidate id denotes one physical transformer layer."""
    parts = _method_parts(method_id)
    return bool(parts and re.fullmatch(r"L\d+", parts[-1]))




def _lock_row_by_validation_detection(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Lock one site by validation Detection, never by the reported test Det.

    Selection must rank on the same quantity the tables report, scored on a
    split we do not report, otherwise the winner is picked on the split it is
    then published on.

    A sole candidate needs no selection. With several, *every* candidate must
    carry a validation Detection score: ranking an incomplete pool would silently
    crown the best of whatever happened to be scored, which is how a partial pool
    previously produced a confident-looking but under-selected number.
    """
    return lock_by_validation_detection(rows)


def _lock_sae_k1_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Use canonical SAE k1 when legacy layers lack validation readouts."""
    locked = _lock_row_by_validation_detection(rows)
    if locked is not None:
        return locked
    canonical = [
        row for row in rows
        if str(row.get("method") or "").split(":")[-1] == "k1"
    ]
    if len(canonical) != 1:
        return None
    out = dict(canonical[0])
    metrics = dict(out.get("metrics") or {})
    metrics["selection_incomplete"] = True
    out["metrics"] = metrics
    return out


def _lock_caa_site(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick one CAA readout site per (pole, class) by validation Detection.

    CAA's candidate set is concat / per-layer. There is no
    fallback: substituting a fixed site when selection is impossible reports an
    unselected readout as though it had been chosen, which is exactly the kind of
    quietly-wrong number the lock exists to prevent. No validation score means no
    selection, which means NaN.
    """
    return _lock_row_by_validation_detection(rows)


def _apply_causal_table_policy(
    results: Mapping[str, Any],
    method_id: str,
    metrics: Dict[str, Any],
    causal_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prefer valid test effects, optionally falling back to validation.

    Fitted AGIEND/CGA rows without train/grid provenance and unrepaired CAGA
    direction rows are unsafe. For every method family, the opt-in
    fallback uses the split-honest validation-selected effect only when a test
    headline is absent; it never overwrites a valid test score.
    """
    out = dict(metrics)
    raw_causal = ((results.get("raw") or {}).get("causal") or {})
    raw_by_method = raw_causal.get("by_method") or {}
    source_mid = str(out.get("causal_source_method") or method_id)
    entry = raw_by_method.get(source_mid) or {}
    if not entry:
        for row in results.get("methods") or []:
            if str(row.get("method") or "") != source_mid:
                continue
            candidate = (row.get("extras") or {}).get("causal") or {}
            if isinstance(candidate, Mapping) and candidate:
                entry = candidate
                break
    entry_meta = entry.get("meta") or {} if isinstance(entry, Mapping) else {}
    is_cga = source_mid.startswith(("cga:", "cga_tensor_norm:"))
    is_agiend = source_mid.startswith("agiend:")
    is_caga = source_mid.startswith("caga:")
    cga_grid_valid = int(entry_meta.get("cga_causal_grid_protocol_version") or 1) >= 2
    agiend_grid_valid = (
        int(entry_meta.get("agiend_causal_grid_protocol_version") or 1) >= 2
    )
    direction_valid = bool(
        int(entry_meta.get("direction_polarity_protocol_version") or 1) >= 2
        or entry_meta.get("polarity_reselected_from_bidirectional_grid")
    )
    cga_reselection_invalid = is_cga and not cga_grid_valid and bool(
        entry_meta.get("polarity_reselected_from_bidirectional_grid")
        or entry_meta.get("validation_grid_reused")
    )
    # Protocol v2 is the positive proof that a CGA grid and its frozen-test
    # score were produced against the same fitted artifact.  Some older rows
    # predate the intermediate reselection markers entirely, so absence of
    # those markers must not make an unversioned CGA test look trustworthy.
    known_invalid = (
        (is_cga and not cga_grid_valid)
        or (is_agiend and not agiend_grid_valid)
        or (is_caga and not direction_valid)
    )

    if known_invalid:
        if (
            bool(results.get("_causal_validation_fallback"))
            and cga_reselection_invalid
        ):
            try:
                # The bad CGA repair swapped an already-correct historical
                # strengthen grid into ``weaken_strengths``. Recover and score
                # that persisted validation curve on its intended panel. This
                # is an explicitly requested intermediate table value; the
                # fresh causal rerun will replace it with a frozen-test score.
                from causal_eval import reselect_inverted_bidirectional_curves

                target_class = str(
                    entry.get("target_class")
                    or out.get("target_class")
                    or source_mid.split(":")[1]
                )
                _, selected, _, weaken_selected = (
                    reselect_inverted_bidirectional_curves(
                        entry, target_class=target_class
                    )
                )
                out["causal_signed_effect"] = float(selected.signed_effect)
                out["causal_delta_mean"] = float(selected.signed_effect)
                out["causal_lms"] = selected.lms
                out["causal_signed_effect_weaken"] = float(
                    weaken_selected.signed_effect
                )
                out["causal_score_source"] = "validation_fallback_invalid_test"
                return out
            except (KeyError, TypeError, ValueError):
                pass
        if bool(results.get("_causal_validation_fallback")):
            selection_split = out.get(
                "causal_selection_split",
                causal_metrics.get(
                    "causal_selection_split", entry_meta.get("selection_split")
                ),
            )
            validation_effect = out.get(
                "causal_selection_signed_effect",
                causal_metrics.get(
                    "causal_selection_signed_effect",
                    entry_meta.get("selection_signed_effect"),
                ),
            )
            if (
                str(selection_split or "").lower() == "validation"
                and isinstance(validation_effect, (int, float))
                and not isinstance(validation_effect, bool)
            ):
                out["causal_signed_effect"] = float(validation_effect)
                out["causal_delta_mean"] = float(validation_effect)
                out["causal_score_source"] = "validation_fallback_invalid_test"
                return out
        # Do not silently erase completed historical causal results simply
        # because a later protocol added provenance fields.  The source is
        # retained so downstream reports can flag it, while strict audit
        # builds can opt back into suppression explicitly.
        if os.environ.get("STRICT_CAUSAL_PROVENANCE", "0") == "1":
            for key in (
                "causal_signed_effect",
                "causal_signed_effect_weaken",
                "causal_delta_mean",
                "causal_delta_target",
                "causal_delta_other",
                "causal_delta_neutral",
                "causal_effectiveness",
                "intervention_score",
            ):
                out[key] = None
            out["causal_score_source"] = "invalid_artifact_grid_provenance"
        else:
            out["causal_score_source"] = "legacy_unvalidated_artifact_grid_provenance"
        return out

    test_effect = out.get("causal_signed_effect")
    if isinstance(test_effect, (int, float)) and not isinstance(test_effect, bool):
        out["causal_score_source"] = "test"
        return out

    if not bool(results.get("_causal_validation_fallback")):
        out["causal_score_source"] = "missing"
        return out

    selection_split = out.get(
        "causal_selection_split",
        causal_metrics.get("causal_selection_split", entry_meta.get("selection_split")),
    )
    validation_effect = out.get(
        "causal_selection_signed_effect",
        causal_metrics.get(
            "causal_selection_signed_effect", entry_meta.get("selection_signed_effect")
        ),
    )
    if (
        str(selection_split or "").lower() == "validation"
        and isinstance(validation_effect, (int, float))
        and not isinstance(validation_effect, bool)
    ):
        out["causal_signed_effect"] = float(validation_effect)
        out["causal_delta_mean"] = float(validation_effect)
        validation_weaken = out.get(
            "causal_weaken_selection_signed_effect",
            causal_metrics.get("causal_weaken_selection_signed_effect"),
        )
        if isinstance(validation_weaken, (int, float)) and not isinstance(
            validation_weaken, bool
        ):
            out["causal_signed_effect_weaken"] = float(validation_weaken)
        validation_lms = out.get(
            "causal_selection_lms", causal_metrics.get("causal_selection_lms")
        )
        if isinstance(validation_lms, (int, float)) and not isinstance(
            validation_lms, bool
        ):
            out["causal_lms"] = float(validation_lms)
        out["causal_score_source"] = "validation_fallback"
    else:
        out["causal_score_source"] = "missing"
    return out


def _attach_suitability(results: Mapping[str, Any], method_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Attach S=E×G using causal linked from intervention ids when needed.

    Always re-resolve causal for ACTIEND/SAE policy children even if a prior
    pipeline pass left ``suitability_E`` with G=0 (encoder-only id).
    """
    cau, cau_src = resolve_causal_metrics(results, method_id)
    raw = (results.get("raw") or {}).get("suitability") or {}
    cfg = (results.get("config") or {}).get("suitability") or {}
    tau = float(raw.get("tau_c", cfg.get("causal_effect_threshold", 0.05)))
    soft = bool(raw.get("soft_causal", cfg.get("soft_causal", False)))
    e_ok = float(raw.get("e_ok", cfg.get("e_ok", 0.8)))
    out = dict(metrics)
    # Fold causal headline fields onto the encoder row for group averages.
    for key in (
        "causal_signed_effect",
        "causal_signed_effect_weaken",
        "causal_base_p",
        "causal_lms",
        "causal_weaken_lms",
        "causal_weaken_base_lms",
        "causal_weaken_lms_ok",
        "causal_lms_ok",
        "causal_selected_strength",
        "causal_delta_mean",
        "causal_delta_target",
        "causal_delta_other",
        "causal_delta_neutral",
        "causal_effectiveness",
        "causal_gate_empty",
        "causal_grid_floor",
        "causal_null_effect",
        "causal_selection_split",
        "causal_selection_signed_effect",
        "causal_selection_lms",
        "causal_weaken_selection_signed_effect",
    ):
        if out.get(key) is None and cau.get(key) is not None:
            out[key] = cau.get(key)
    if cau_src and cau_src != method_id:
        out["causal_source_method"] = cau_src
    out = _apply_causal_table_policy(results, method_id, out, cau)
    policy_causal = dict(cau)
    for key in (
        "causal_signed_effect",
        "causal_signed_effect_weaken",
        "causal_delta_mean",
        "causal_delta_target",
        "causal_delta_other",
        "causal_delta_neutral",
    ):
        policy_causal[key] = out.get(key)
    out.update(
        compute_feature_suitability(
            out, policy_causal, tau_c=tau, soft_causal=soft, e_ok=e_ok
        )
    )
    return out


def _mean_metric(values: Sequence[Any]) -> Optional[float]:
    import math

    nums = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(float(v))
    ]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _aggregate_metric_dict(
    rows: Sequence[Mapping[str, Any]], *, pair: bool = True
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    metric_rows = [r.get("metrics") or {} for r in rows]
    e_vals = [_encoding_e(metrics) for metrics in metric_rows]
    out["encoding_E"] = _mean_metric([v for v in e_vals if v is not None])
    detection_vals = [detection_score(metrics) for metrics in metric_rows]
    # Detection is a worst-case score at the task/method level: first compute
    # D=min(...) for each atomic member, then take the minimum across all
    # contributing members.  Averaging these member scores would hide the
    # weakest class/feature and contradict the paper's Detection definition.
    detection_vals = [value for value in detection_vals if value is not None]
    if DETECTION_MEMBER_AGGREGATION == "min":
        out["detection_score"] = min(detection_vals) if detection_vals else None
    elif DETECTION_MEMBER_AGGREGATION == "mean":
        out["detection_score"] = _mean_metric(detection_vals)
    else:
        raise ValueError(
            "DETECTION_MEMBER_AGGREGATION must be 'min' or 'mean', got "
            f"{DETECTION_MEMBER_AGGREGATION!r}"
        )
    intervention_vals = [intervention_score(metrics) for metrics in metric_rows]
    out["intervention_score"] = _mean_metric(
        [value for value in intervention_vals if value is not None]
    )
    out["detection_complete"] = bool(metric_rows) and all(
        all(
            isinstance(metrics.get(field), (int, float))
            and not isinstance(metrics.get(field), bool)
            for field in (
                ("roc_auc_neutral", "neutral_specificity", "roc_auc_other", "class_exclusivity")
                if pair
                else ("roc_auc_neutral", "neutral_specificity")
            )
        )
        for metrics in metric_rows
    )
    out["intervention_complete"] = bool(metric_rows) and all(
        intervention_score(metrics) is not None for metrics in metric_rows
    )
    # A headline causal number is valid only when every class member has a
    # causal headline.  Never average the completed subset of a partial run.
    causal_complete = bool(metric_rows) and all(
        metrics.get("causal_signed_effect") is not None
        or metrics.get("causal_delta_mean") is not None
        or metrics.get("causal_selected_strength") is not None
        for metrics in metric_rows
    )
    for key in HEADLINE_METRICS:
        if key in {"encoding_E", "detection_score", "intervention_score"}:
            continue
        if key.startswith("causal_") and not causal_complete:
            out[key] = None
        else:
            out[key] = _mean_metric([(r.get("metrics") or {}).get(key) for r in rows])
    if not causal_complete:
        out["intervention_score"] = None
    causal_sources = sorted(
        {
            str((r.get("metrics") or {}).get("causal_score_source"))
            for r in rows
            if (r.get("metrics") or {}).get("causal_score_source")
        }
    )
    if len(causal_sources) == 1:
        out["causal_score_source"] = causal_sources[0]
    elif causal_sources:
        out["causal_score_source"] = "mixed:" + ",".join(causal_sources)
    reasons = sorted({str((r.get("metrics") or {}).get("suitability_reason")) for r in rows if (r.get("metrics") or {}).get("suitability_reason")})
    scopes = sorted({str((r.get("metrics") or {}).get("suitability_scope")) for r in rows if (r.get("metrics") or {}).get("suitability_scope")})
    if len(reasons) == 1:
        out["suitability_reason"] = reasons[0]
    elif reasons:
        out["suitability_reason"] = "mixed"
    if len(scopes) == 1:
        out["suitability_scope"] = scopes[0]
    elif scopes:
        out["suitability_scope"] = "mixed"
    return out


def _member_target_class(member: Mapping[str, Any]) -> str:
    """Recover the single target class a headline-group member speaks for.

    Members inside a bucket are already one row per class (the CAA/SAE
    E-lock picks one site per class; gradiend/actiend rows are per-class by
    id).  Prefer an explicit ``target_class`` metric, else parse the id.  The
    bare pair aggregate (``gradiend``/``actiend`` with no class in the id) has
    no single class and returns ``""``.
    """
    metrics = member.get("metrics") or {}
    tc = metrics.get("target_class")
    if isinstance(tc, str) and tc:
        return tc
    parts = _method_parts(normalize_sae_method_id(str(member.get("method") or "")))
    if not parts:
        return ""
    backend = parts[0]
    if backend in {"sae", "sae_pre"}:
        return parts[1] if len(parts) >= 3 else ""
    if backend == "caa":
        if len(parts) >= 4 and is_pair_key(parts[1]):
            return parts[2]
        if len(parts) >= 3:
            return parts[1]
        return ""
    if backend == "actiend_ridge":
        if len(parts) >= 3 and "-" in parts[1]:
            return parts[2]
        return parts[1] if len(parts) >= 2 else ""
    # gradiend-like: ``gradiend`` (pooled pair), ``gradiend:F-M:F``, ``gradiend:F``.
    if len(parts) == 3 and "-" in parts[1]:
        return parts[2]
    if len(parts) == 2:
        return parts[1]
    return ""


def _collect_buckets(
    results: Mapping[str, Any],
    results_path: str,
    *,
    causal_validation_fallback: bool = False,
    representation_view: str = "best_overall",
) -> Optional[Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]]:
    """Shared bucketing for the group- and class-level collectors.

    Returns ``(meta, buckets)`` where ``buckets`` maps a headline
    ``method_group`` to its per-class member rows (post E-lock).  ``None`` when
    the payload has no usable claim/target classes or no method rows.  Keeping
    this one code path means the class-level export and the paper's group
    tables can never disagree about which readout site is canonical.
    """
    if representation_view not in REPRESENTATION_VIEWS:
        raise ValueError(
            f"unknown representation_view={representation_view!r}; "
            f"expected one of {sorted(REPRESENTATION_VIEWS)}"
        )
    if causal_validation_fallback:
        results = {**dict(results), "_causal_validation_fallback": True}
    parts = Path(results_path).parts if results_path else ()
    model = str(results.get("model") or (parts[-3] if len(parts) >= 3 else "unknown"))
    task = str(results.get("task") or (parts[-2] if len(parts) >= 2 else "unknown"))
    suite = results.get("suite")
    run_status = results.get("status", "unknown")

    try:
        claim_classes = require_claim_classes(dict(results))
        target_classes = require_target_classes(dict(results))
    except ValueError:
        return None

    deduped = _dedupe_method_rows(
        [
            *(results.get("methods") or []),
            *_caa_encode_rows_from_raw(results),
            *_sae_encode_rows_from_artifacts(results, results_path),
            *_gradient_encode_rows_from_artifacts(results, results_path),
        ]
    )
    if not deduped:
        return None

    # Join CAGA/CGA pair encoder rows (backend:A-B) with their per-class
    # causal rows (backend:A-B:A / :B).  Refreshes commonly persist these as
    # separate rows; without this join the headline drops the family.
    collected_rows: List[Dict[str, Any]] = list(deduped.values())
    raw_by_method = {
        str(row.get("method") or ""): row
        for row in (results.get("methods") or [])
        if row.get("method")
    }
    complete_join_pair_ids: set[str] = set()
    for base in list(deduped.values()):
        mid = str(base.get("method") or "")
        if "|proxy=" in mid:
            continue
        parts_base = _method_parts(mid)
        if (
            len(parts_base) != 2
            or parts_base[0] not in GRADIEND_LIKE_BACKENDS
            or parts_base[0] == "actiend_ridge"
            or not is_pair_key(parts_base[1])
        ):
            continue
        # CGA/CAGA/AGIEND commonly persist encoder readouts in
        # ``extras.per_class_readouts`` and leave the flat base-row metrics as
        # correlation-only. Accept either representation.
        per_class_readouts = (base.get("extras") or {}).get("per_class_readouts") or {}
        if not _has_encoder_metrics(base.get("metrics") or {}) and not per_class_readouts:
            continue
        joined_classes = 0
        for cls in parts_base[1].split("-"):
            cid = f"{parts_base[0]}:{parts_base[1]}:{cls}"
            causal_row = raw_by_method.get(cid)
            if causal_row is None:
                continue
            causal_metrics = _attach_suitability(
                results, cid, dict(causal_row.get("metrics") or {})
            )
            merged = dict(base)
            metrics = dict(base.get("metrics") or {})
            readout = per_class_readouts.get(cls)
            if isinstance(readout, Mapping):
                # Promote the class-specific held-out readout to the canonical
                # row shape.  Do not overwrite non-null flat values from a
                # newer producer, but fill the legacy split representation.
                for key, value in readout.items():
                    if metrics.get(key) is None and value is not None:
                        metrics[key] = value
            metrics["target_class"] = cls
            for field in CAUSAL_METRICS:
                if causal_metrics.get(field) is not None:
                    metrics[field] = causal_metrics[field]
            merged["method"] = cid
            merged["metrics"] = metrics
            collected_rows.append(merged)
            joined_classes += 1
        if joined_classes == len(parts_base[1].split("-")):
            # The base pair row is an aggregate encoder row. Once every class
            # has a class-specific causal companion, retaining that base row
            # in the headline member set would make causal completeness false
            # (the base row has no causal effect of its own).
            # Include proxy-suffixed encoder rows (e.g. ``|proxy=her_his``):
            # they are alternate base readouts, not extra causal members.
            complete_join_pair_ids.add(
                f"{parts_base[0]}:{parts_base[1].split('|', 1)[0]}"
            )

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    sae_k1_by_cls: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    sae_kstar_by_cls: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    caa_by_cls: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    layerwise_mean_by_cls: Dict[
        Tuple[str, str, str, str], List[Dict[str, Any]]
    ] = {}

    for row in collected_rows:
        row_mid = str(row.get("method") or "")
        if "|proxy=" in row_mid:
            continue
        row_base_mid = row_mid.split("|", 1)[0]
        if row_base_mid in complete_join_pair_ids and len(_method_parts(row_base_mid)) == 2:
            continue
        mid = normalize_sae_method_id(str(row.get("method") or ""))
        pole = _pole_from_row(row)
        backend = _method_parts(mid)[0] if _method_parts(mid) else ""

        if backend in _LAYERWISE_MEAN_BACKENDS and pole in POLE_LABELS:
            parts_layer = _method_parts(mid)
            identity = _layerwise_mean_candidate(parts_layer, pole=pole)
            if identity is None:
                continue
            cls, contrast_key = identity
            if cls not in claim_classes and cls not in target_classes:
                continue
            if not _has_encoder_metrics(row.get("metrics") or {}):
                continue
            enriched = dict(row)
            metrics = dict(row.get("metrics") or {})
            metrics["encoding_E"] = _encoding_e(metrics)
            enriched["metrics"] = _attach_suitability(results, mid, metrics)
            layerwise_mean_by_cls.setdefault(
                (backend, POLE_LABELS[pole], cls, contrast_key), []
            ).append(enriched)
            # Base + Lk candidates are locked together below, per class.
            continue

        if pole in POLE_LABELS and backend in GRADIEND_LIKE_BACKENDS:
            if not _class_headline_gradiend_actiend(
                mid, pole=pole, claim_classes=claim_classes, target_classes=target_classes
            ):
                continue
            group = f"{backend}:{POLE_LABELS[pole]}"
            enriched = dict(row)
            metrics = dict(row.get("metrics") or {})
            metrics["encoding_E"] = _encoding_e(metrics)
            enriched["metrics"] = _attach_suitability(results, mid, metrics)
            buckets.setdefault(group, []).append(enriched)
            continue

        parts_mid = _method_parts(mid)
        if backend in {"sae", "sae_pre"} and len(parts_mid) == 3:
            cls = parts_mid[1]
            if cls not in claim_classes:
                continue
            tail = parts_mid[2]
            if "tok_" in tail:
                continue
            if not _has_encoder_metrics(row.get("metrics") or {}):
                continue
            enriched = dict(row)
            enriched["metrics"] = _attach_suitability(results, mid, dict(row.get("metrics") or {}))
            if _sae_k1_site_tail(tail):
                sae_k1_by_cls.setdefault((backend, cls), []).append(enriched)
            if _sae_kstar_site_tail(tail):
                sae_kstar_by_cls.setdefault((backend, cls), []).append(enriched)
            continue

        if backend == "caa" and pole in POLE_LABELS:
            metrics = dict(row.get("metrics") or {})
            inferred_cls = ""
            if len(parts_mid) >= 4 and is_pair_key(parts_mid[1]):
                inferred_cls = parts_mid[2]
            elif len(parts_mid) >= 3:
                inferred_cls = parts_mid[1]
            cls = str(metrics.get("target_class") or inferred_cls)
            tail = parts_mid[-1] if parts_mid else ""
            # CAA headline selection compares the aggregate recipes and the
            # per-layer ablations, locked by validation detection.  The test
            # score is read only from the validation-selected row.
            if cls not in claim_classes or not _caa_site_tail(tail):
                continue
            if not _has_encoder_metrics(row.get("metrics") or {}):
                continue
            enriched = dict(row)
            metrics["encoding_E"] = _encoding_e(metrics)
            enriched["metrics"] = _attach_suitability(results, mid, metrics)
            contrast_key = parts_mid[1] if pole == "pair" and len(parts_mid) >= 4 else cls
            caa_by_cls.setdefault((POLE_LABELS[pole], cls, contrast_key), []).append(enriched)

    for (backend, _cls), members in sae_k1_by_cls.items():
        if representation_view == "all_layer":
            members = [
                m for m in members
                if _method_parts(str(m.get("method") or ""))[-1] == "all_k1"
            ]
        elif representation_view == "best_single_layer":
            members = [
                m for m in members
                if _method_parts(str(m.get("method") or ""))[-1] == "k1"
            ]
        locked = _lock_sae_k1_row(members)
        if locked is not None:
            buckets.setdefault(f"{backend}:k1", []).append(locked)
    for (backend, _cls), members in sae_kstar_by_cls.items():
        locked = _lock_row_by_validation_detection(members)
        if locked is not None:
            buckets.setdefault(f"{backend}:kstar", []).append(locked)
    for (pole_label, _cls, _contrast_key), members in caa_by_cls.items():
        if representation_view == "all_layer":
            members = [
                m for m in members
                if str(m.get("method") or "").endswith(":act_prediction")
            ]
        elif representation_view == "best_single_layer":
            members = [
                m for m in members
                if re.search(r":L\d+_act_prediction$", str(m.get("method") or ""))
            ]
        locked = _lock_caa_site(members)
        if locked is not None:
            buckets.setdefault(f"caa:{pole_label}", []).append(locked)
    for (backend, pole_label, _cls, _contrast_key), members in layerwise_mean_by_cls.items():
        # Core's headline is a validation lock over one aggregate candidate and
        # every physical layer.  An old aggregate-only result is a useful raw
        # all-layer measurement, but it is *not* a valid selected-headline
        # pool.  Suppress it until the one-time layerwise detection backfill
        # arrives; never silently relabel it as the best representation.
        has_aggregate = any(
            not _is_physical_layer_candidate(str(member.get("method") or ""))
            for member in members
        )
        has_layer = any(
            _is_physical_layer_candidate(str(member.get("method") or ""))
            for member in members
        )
        if not (has_aggregate and has_layer):
            continue
        if representation_view == "all_layer":
            members = [
                m for m in members
                if not _is_physical_layer_candidate(str(m.get("method") or ""))
            ]
        elif representation_view == "best_single_layer":
            members = [
                m for m in members
                if _is_physical_layer_candidate(str(m.get("method") or ""))
            ]
        locked = _lock_row_by_validation_detection(members)
        if locked is not None:
            buckets.setdefault(f"{backend}:{pole_label}", []).append(locked)
    meta = {
        "model": model,
        "task": task,
        "suite": suite,
        "run_status": run_status,
        "claim_classes": list(claim_classes),
        "target_classes": list(target_classes),
        "pair": bool(
            spec_for_task(
                task,
                suite=str(suite) if suite else None,
                model=model,
                results=results,
                results_path=results_path,
            ).get("pair", True)
        ),
    }
    return meta, buckets


def collect_group_rows_for_results(
    results: Mapping[str, Any],
    *,
    results_path: str = "",
    causal_validation_fallback: bool = False,
    representation_view: str = "best_overall",
) -> List[Dict[str, Any]]:
    """Build headline group rows for one ``results.json`` payload.

    One row per ``(model, task, method_group)`` with every headline metric
    averaged over the group's per-class members.
    """
    collected = _collect_buckets(
        results,
        results_path,
        causal_validation_fallback=causal_validation_fallback,
        representation_view=representation_view,
    )
    if collected is None:
        return []
    meta, buckets = collected

    out: List[Dict[str, Any]] = []
    for group, members in sorted(buckets.items()):
        metrics = _aggregate_metric_dict(members, pair=bool(meta["pair"]))
        source_methods = sorted({str(m.get("method")) for m in members})
        out.append(
            {
                "model": meta["model"],
                "task": meta["task"],
                "suite": meta["suite"],
                "method_group": group,
                "backend": group.split(":", 1)[0],
                "pole": group.split(":", 1)[1] if ":" in group else None,
                "run_status": meta["run_status"],
                "n_sources": len(source_methods),
                "source_methods": ",".join(source_methods),
                "results_path": results_path,
                **metrics,
            }
        )
    return out


def collect_class_rows_for_results(
    results: Mapping[str, Any],
    *,
    results_path: str = "",
    causal_validation_fallback: bool = False,
    representation_view: str = "best_overall",
) -> List[Dict[str, Any]]:
    """Per-class headline rows for one ``results.json`` payload.

    One row per ``(model, task, method_group, target_class)`` — the same
    canonical members ``collect_group_rows_for_results`` averages, emitted
    unaggregated so downstream tables can pool over class on the fly.  The
    bare pair aggregate (no class in its id) is labelled ``__pooled__`` so it
    is distinguishable from a real class and easy to drop when averaging.
    """
    collected = _collect_buckets(
        results,
        results_path,
        causal_validation_fallback=causal_validation_fallback,
        representation_view=representation_view,
    )
    if collected is None:
        return []
    meta, buckets = collected

    out: List[Dict[str, Any]] = []
    for group, members in sorted(buckets.items()):
        backend = group.split(":", 1)[0]
        pole = group.split(":", 1)[1] if ":" in group else None
        for member in members:
            metrics = member.get("metrics") or {}
            row: Dict[str, Any] = {
                "model": meta["model"],
                "task": meta["task"],
                "suite": meta["suite"],
                "method_group": group,
                "backend": backend,
                "construction": pole,
                "target_class": _member_target_class(member) or "__pooled__",
                "method": member.get("method"),
                "run_status": meta["run_status"],
                "row_status": member.get("status"),
                "results_path": results_path,
            }
            e_val = _encoding_e(metrics)
            row["encoding_E"] = e_val
            row["detection_score"] = detection_score(metrics)
            row["intervention_score"] = intervention_score(metrics)
            detection_fields = (
                ("roc_auc_neutral", "neutral_specificity", "roc_auc_other", "class_exclusivity")
                if meta["pair"]
                else ("roc_auc_neutral", "neutral_specificity")
            )
            row["detection_complete"] = all(
                isinstance(metrics.get(field), (int, float))
                and not isinstance(metrics.get(field), bool)
                for field in detection_fields
            )
            row["intervention_complete"] = row["intervention_score"] is not None
            for key in HEADLINE_METRICS:
                if key in {"encoding_E", "detection_score", "intervention_score"}:
                    continue
                v = metrics.get(key)
                row[key] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            out.append(row)
    return out


def collect_group_rows(runs_root: Path = ROOT / "runs") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in _iter_results_json(Path(runs_root)):
        payload = _load_results(path)
        if not payload:
            continue
        rows.extend(collect_group_rows_for_results(payload, results_path=str(path)))
    return pd.DataFrame(rows)


def _method_group_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (METHOD_GROUP_ORDER.index(str(name)), str(name))
    except ValueError:
        return (len(METHOD_GROUP_ORDER), str(name))


def _wanted_groups(
    pivot: pd.DataFrame,
    *,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]],
    model: Optional[str],
) -> List[str]:
    present = [str(i) for i in pivot.index]
    extra = [g for g in present if g not in METHOD_GROUP_ORDER]
    wanted: List[str] = []
    for group in METHOD_GROUP_ORDER:
        if group == "sae":
            if group in present:
                wanted.append(group)
            continue
        if group in present:
            wanted.append(group)
            continue
        if not specs:
            continue
        applicable = any(
            group_is_applicable(group, spec)
            for (mod, _task), spec in specs.items()
            if model is None or mod == model
        )
        if applicable:
            wanted.append(group)
    return wanted + extra


def pivot_group_task(
    df: pd.DataFrame,
    *,
    metric: str = "roc_auc_neutral",
    model: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    sub = df.copy()
    if model:
        sub = sub[sub["model"] == model]
    if metric not in sub.columns or sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index="method_group", columns="task", values=metric, aggfunc="mean")
    if specs:
        spec_tasks = sorted(
            {task for (mod, task) in specs if model is None or mod == model}
        )
        for task in spec_tasks:
            if task not in pivot.columns:
                pivot[task] = float("nan")
        ordered_tasks = [c for c in spec_tasks if c in pivot.columns]
        leftover = [c for c in pivot.columns if c not in ordered_tasks]
        pivot = pivot.reindex(columns=ordered_tasks + leftover)
    wanted = _wanted_groups(pivot, specs=specs, model=model)
    if metric not in CAUSAL_METRICS:
        wanted = [
            group
            for group in wanted
            if not group.startswith(CAUSAL_ONLY_GROUP_PREFIXES)
        ]
    if wanted:
        pivot = pivot.reindex(wanted)
    pivot["MEAN"] = pivot.mean(axis=1, skipna=True)
    ordered = sorted(pivot.index.astype(str), key=_method_group_sort_key)
    return pivot.reindex(ordered)


def format_pivot_cell(value: Any, *, expected: bool = True, width: int = 0) -> str:
    """``-`` = not applicable; ``NaN`` = applicable but missing."""
    import math

    try:
        missing = value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)
    except (TypeError, ValueError):
        missing = True
    if missing:
        body = GAP_DISPLAY if not expected else MISSING_DISPLAY
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        body = f"{float(value):.4f}"
    else:
        body = str(value)
    return body if not width else f"{body:>{width}}"


def format_pivot_display(
    pivot: pd.DataFrame,
    *,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """String table: numeric values, ``-`` for gaps, ``NaN`` for missing."""
    display = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=object)
    for idx in pivot.index:
        for col in pivot.columns:
            expected = True
            if str(col) != "MEAN":
                spec = resolve_spec(specs, model=str(model or ""), task=str(col))
                expected = group_is_applicable(str(idx), spec)
            display.loc[idx, col] = format_pivot_cell(pivot.loc[idx, col], expected=expected)
    return display


def format_overview_text(
    df: pd.DataFrame,
    *,
    metric: str = "roc_auc_neutral",
    model: Optional[str] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> str:
    pivot = pivot_group_task(df, metric=metric, model=model, specs=specs)
    if pivot.empty:
        return f"(no headline groups for metric={metric})"
    col_w = max(8, max((len(str(c)) for c in pivot.columns), default=4) + 1)
    idx_w = max(14, max((len(str(i)) for i in pivot.index), default=6))
    header = f"{'method_group':<{idx_w}}" + "".join(f"{str(c):>{col_w}}" for c in pivot.columns)
    lines = [
        f"Headline method groups x task ({metric})"
        + (f" [{model}]" if model else ""),
        f"{GAP_DISPLAY} = expected gap (not applicable); {MISSING_DISPLAY} = missing value",
        header,
        "-" * len(header),
    ]
    for idx, row in pivot.iterrows():
        line = f"{str(idx):<{idx_w}}"
        for col in pivot.columns:
            expected = True
            if str(col) != "MEAN":
                spec = resolve_spec(specs, model=str(model or ""), task=str(col))
                expected = group_is_applicable(str(idx), spec)
            line += format_pivot_cell(row[col], expected=expected, width=col_w)
        lines.append(line)
    return "\n".join(lines)
