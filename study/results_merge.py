"""Smart-merge of study ``results.json`` across partial ``--methods`` runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# Method families (CLI ``--methods``). Causal/localization are stages, not methods,
# but still appear in the internal ``enabled`` set when those stages run.
METHOD_FAMILIES = (
    "gradiend",
    "actiend",
    "actiend_ridge",
    "sae",
    "caa",
    "cga",
    "caga",
    "agiend",
)

PIPELINE_STAGES = (
    "causal",
    "localization",
)


def causal_raw_entry_ok(cres: Any) -> bool:
    """True when a ``raw.causal.by_method`` entry has a completed LMS-gated sweep."""
    if not isinstance(cres, dict):
        return False
    meta = cres.get("meta") or {}
    if meta.get("error"):
        return False
    strengths = cres.get("strengths") or []
    if not strengths:
        return False
    if cres.get("selected_strength") is None:
        return False
    # Bidirectional causal protocol: a strengthen-only row is incomplete and
    # must never be reused as if it contained a held-out weaken headline.
    if not (cres.get("weaken_strengths") or []):
        return False
    if cres.get("weaken_selected_strength") is None:
        return False
    if not isinstance(cres.get("weaken_selected"), dict):
        return False
    return True


def method_eval_row_ok(row: Any) -> bool:
    """True when a method row has usable encode or causal metrics (no hard error)."""
    if not isinstance(row, dict):
        return False
    status = str(row.get("status") or "")
    if status == "error":
        return False
    if row.get("error"):
        return False
    metrics = row.get("metrics") or {}
    if metrics.get("causal_error"):
        return False
    if metrics.get("causal_signed_effect") is not None:
        return True
    try:
        from results_schema import _has_encoder_metrics

        if _has_encoder_metrics(metrics):
            return status in {"ok", "partial", "collapsed"}
    except Exception:
        pass
    return status in {"ok", "partial", "collapsed"} and bool(metrics)




def _strip_causal_suffix(method_id: str) -> str:
    mid = str(method_id or "")
    if mid.endswith(":causal"):
        return mid[: -len(":causal")]
    return mid


def method_family(method_id: str) -> Optional[str]:
    """Map a method id to its method family (gradiend/actiend/sae/caa/…).

    Causal rows like ``sae:VALUE:k1:causal`` attribute to ``sae``, not a fake
    ``causal`` method family. Localization keeps its own stage id.
    ``actiend_pre`` / ``sae_pre`` are analysis families that piggyback on
    CLI ``actiend`` / ``sae``.
    """
    mid = str(method_id or "")
    if not mid:
        return None
    if mid == "localization" or mid.startswith("localization"):
        return "localization"
    base = _strip_causal_suffix(mid)
    if base == "sae" or (base.startswith("sae:") and not base.startswith("sae_pre:")) or base.startswith("sae|"):
        return "sae"
    if base == "sae_pre" or base.startswith("sae_pre:") or base.startswith("sae_pre|"):
        return "sae_pre"
    if base.startswith("caa:"):
        return "caa"
    # CGA variants keep separate analysis families (so the tensor-norm ablation
    # is never averaged into headline CGA), but both map back to the single
    # ``cga`` CLI family below via ``_cli_family_for``.
    if base == "cga_tensor_norm" or base.startswith("cga_tensor_norm:"):
        return "cga_tensor_norm"
    if base == "cga" or base.startswith("cga:"):
        return "cga"
    if base == "caga" or base.startswith("caga:"):
        return "caga"
    if base == "agiend" or base.startswith("agiend:"):
        return "agiend"
    if base.startswith("gradiend:"):
        return "gradiend"
    if base == "actiend_pre" or base.startswith("actiend_pre:") or base.startswith("actiend_pre|"):
        return "actiend_pre"
    if base == "actiend_ridge" or base.startswith("actiend_ridge:") or base.startswith("actiend_ridge|"):
        return "actiend_ridge"
    if base.startswith("actiend:"):
        return "actiend"
    return None


def _cli_family_for(fam: Optional[str]) -> Optional[str]:
    """Map analysis family → CLI ``--methods`` family."""
    if fam == "sae_pre":
        return "sae"
    if fam == "cga_tensor_norm":
        return "cga"
    # actiend_pre is opt-in (training.actiend_pre), not an automatic CLI piggyback.
    if fam == "actiend_pre":
        return "actiend_pre"
    return fam


def _is_causal_row(method_id: str) -> bool:
    mid = str(method_id or "")
    return mid.endswith(":causal") or ":causal:" in mid


def _has_encoder_payload(metrics: Any) -> bool:
    """True when ``metrics`` contains a real encoder evaluation payload."""
    if not isinstance(metrics, dict):
        return False
    try:
        from results_schema import _has_encoder_metrics

        return bool(_has_encoder_metrics(metrics))
    except Exception:
        return any(
            metrics.get(key) is not None
            for key in ("roc_auc", "roc_auc_neutral", "balanced_accuracy")
        )


def _has_causal_payload(metrics: Any) -> bool:
    """True when ``metrics`` contains a causal evaluation or causal failure."""
    if not isinstance(metrics, dict):
        return False
    return any(
        key.startswith("causal_") and value is not None
        for key, value in metrics.items()
    )


def _row_has_causal_payload(row: Any) -> bool:
    """True for a causal row even when no headline metrics were flattened.

    Some causal-only rows retain the complete bundle under ``extras.causal``
    but have only routing metadata (``target_class`` / ``readout_kind``) in
    ``metrics``.  Treating those rows as payload-free made them replace the
    matching encoder row instead of complementing it.
    """
    if not isinstance(row, dict):
        return False
    metrics = row.get("metrics")
    if _has_causal_payload(metrics):
        return True
    if isinstance(metrics, dict) and metrics.get("readout_kind") == "causal_only":
        return True
    extras = row.get("extras")
    return isinstance(extras, dict) and isinstance(extras.get("causal"), dict)


def _merge_complementary_method_rows(
    previous: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, Any]:
    """Preserve encode+causal halves when a shared method id is refreshed.

    Shared ids such as ``sae:IO:k1`` and ``gradiend:IO`` intentionally carry
    both encoder and causal metrics.  A causal-only refresh can emit a new row
    with the same id but only causal fields; replacing the prior row would
    silently erase its encoder evaluation.  Merge only when the payloads are
    complementary.  If the new row contains encoder metrics, it remains the
    authority for a genuine family retrain/rerun.
    """
    prev_metrics = dict(previous.get("metrics") or {})
    new_metrics = dict(new.get("metrics") or {})
    prev_has_encoder = _has_encoder_payload(prev_metrics)
    new_has_encoder = _has_encoder_payload(new_metrics)
    prev_has_causal = _row_has_causal_payload(previous)
    new_has_causal = _row_has_causal_payload(new)

    preserve_previous_encoder = prev_has_encoder and new_has_causal and not new_has_encoder
    preserve_previous_causal = prev_has_causal and new_has_encoder and not new_has_causal
    if not (preserve_previous_encoder or preserve_previous_causal):
        return new

    merged = {**previous, **new}
    merged_metrics = {**prev_metrics, **new_metrics}
    # ``causal_only`` describes the incoming shell, not the merged row.  Keep
    # the real encoder readout kind once its metrics have been preserved.
    if (
        preserve_previous_encoder
        and new_metrics.get("readout_kind") == "causal_only"
        and prev_metrics.get("readout_kind") is not None
    ):
        merged_metrics["readout_kind"] = prev_metrics["readout_kind"]
    merged["metrics"] = merged_metrics
    for key in ("artifacts", "extras"):
        prev_part = previous.get(key)
        new_part = new.get(key)
        if isinstance(prev_part, dict) or isinstance(new_part, dict):
            merged[key] = {
                **(prev_part if isinstance(prev_part, dict) else {}),
                **(new_part if isinstance(new_part, dict) else {}),
            }
    return merged


def _drop_stale_causal_for_retrained(
    mid: str,
    *,
    enabled: Set[str],
) -> bool:
    """If we re-run a method family but skip causal, drop that family's old causal rows."""
    if "causal" in enabled:
        return False
    if not _is_causal_row(mid):
        return False
    fam = method_family(mid)
    cli_fam = _cli_family_for(fam)
    if fam == "actiend_pre":
        # Only drop when this run actually retrains actiend_pre.
        return "actiend_pre" in enabled
    return cli_fam in enabled and cli_fam in METHOD_FAMILIES


def should_replace_method(method_id: str, *, enabled: Set[str]) -> bool:
    mid = str(method_id or "")
    fam = method_family(mid)
    if fam is None:
        return False
    if fam == "actiend_pre":
        # Opt-in family: only replace when explicitly in enabled (or wipe via new rows).
        if _is_causal_row(mid):
            return "actiend_pre" in enabled and "causal" in enabled
        return "actiend_pre" in enabled
    cli_fam = _cli_family_for(fam)
    if _is_causal_row(mid):
        if cli_fam in enabled and "causal" in enabled:
            return True
        return _drop_stale_causal_for_retrained(mid, enabled=enabled)
    if fam == "localization":
        return "localization" in enabled
    return cli_fam in enabled


# Families whose pair / one-pole trainers can be split across concurrent jobs
# (``--train-ablations``).  CAA/SAE compute both regimes in one pass and are not sliced.
_SLICE_FAMILIES = frozenset({"gradiend", "actiend", "actiend_pre", "agiend", "cga", "caga"})


def row_regime(method_id: str) -> Optional[str]:
    """``"pair"`` / ``"one_pole"`` for ``family:A-B[...]`` / ``family:C[...]`` ids, else ``None``.

    ``None`` (a bare family stub, ``family:``-less ids) is treated as belonging to every slice.
    """
    parts = str(method_id or "").split(":")
    if len(parts) < 2 or not parts[1]:
        return None
    from study.method_ids import is_pair_key

    return "pair" if is_pair_key(parts[1].split("|", 1)[0]) else "one_pole"


def _normalize_slice(value: Any) -> Optional[frozenset]:
    """``config.train_ablations`` (already normalized by the writer) -> frozenset or ``None``."""
    if not value:
        return None
    items = [value] if isinstance(value, str) else list(value)
    out = {str(v) for v in items if str(v) in {"pair", "one_pole"}}
    return frozenset(out) or None


def merge_method_rows(
    previous: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    *,
    enabled: Set[str],
    replace_train_splits: Optional[Set[str]] = None,
    regimes: Optional[frozenset] = None,
) -> List[Dict[str, Any]]:
    """Keep prior rows not covered by this run; replace overlapping families with ``new_rows``.

    A family-level stub failure (e.g. lone ``sae`` with status=error and no
    class/k rows) does **not** wipe prior successful rows for that family.

    ``regimes`` (``{"pair"}`` / ``{"one_pole"}``): this writer is a ``--train-ablations`` slice.
    It is authoritative only for rows of its own regime in the sliceable families; the other
    slice runs concurrently, so its rows on disk are fresher than the copy this process loaded
    at startup and must neither be wiped by the family-level overwrite nor overwritten by that
    stale copy.
    """
    def _other_slice(mid: str) -> bool:
        if not regimes or method_family(mid) not in _SLICE_FAMILIES:
            return False
        regime = row_regime(mid)
        return regime is not None and regime not in regimes

    own_rows = [r for r in new_rows if not _other_slice(str(r.get("method") or ""))]
    wipe_families = {
        fam
        for fam in METHOD_FAMILIES
        if fam in enabled and _family_has_substantive_encoder_rows(own_rows, fam)
    }
    if "actiend_pre" in enabled and _family_has_substantive_encoder_rows(new_rows, "actiend_pre"):
        wipe_families.add("actiend_pre")
    if "sae" in enabled and _family_has_substantive_encoder_rows(new_rows, "sae_pre"):
        wipe_families.add("sae_pre")
    targeted_splits = {
        str(split).strip().lower()
        for split in (replace_train_splits or set())
        if str(split).strip()
    }

    def _unrequested_train_split(mid: str) -> bool:
        fam = method_family(mid)
        if fam not in {"gradiend", "actiend"} or fam not in enabled:
            return False
        split = "tensors" if ":tensors" in str(mid) else "none"
        return bool(targeted_splits) and split not in targeted_splits

    kept = []
    for row in previous:
        mid = str(row.get("method") or "")
        if not mid:
            continue
        # A targeted split writer is authoritative only for that split. Keep
        # the freshly re-read on-disk counterpart, not the preserved copy that
        # this process captured before a concurrent writer may have updated it.
        if _unrequested_train_split(mid) or _other_slice(mid):
            kept.append(row)
            continue
        if not should_replace_method(mid, enabled=enabled):
            kept.append(row)
            continue
        fam = method_family(mid)
        if fam in wipe_families:
            continue
        # Keep prior rows when this run only produced a family stub error.
        kept.append(row)
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in kept:
        mid = str(row.get("method") or "")
        if not mid:
            continue
        by_id[mid] = row
        order.append(mid)
    piggyback = ("actiend_pre", "sae_pre")
    for row in new_rows:
        mid = str(row.get("method") or "")
        if not mid:
            continue
        if (_unrequested_train_split(mid) or _other_slice(mid)) and mid in by_id:
            # ``new_rows`` includes unrequested rows preserved at process
            # startup. Under concurrency those may now be stale; the row in
            # ``by_id`` came from the fresh read performed under the lock.
            continue
        fam = method_family(mid)
        # Drop bare family error stubs when prior class rows were preserved.
        if (
            fam in (*METHOD_FAMILIES, *piggyback)
            and mid == fam
            and row.get("status") == "error"
            and any(method_family(str(r.get("method") or "")) == fam for r in kept)
        ):
            continue
        prev_row = by_id.get(mid)
        if prev_row is not None and method_eval_row_ok(prev_row) and not method_eval_row_ok(row):
            continue
        if prev_row is not None and method_eval_row_ok(prev_row) and method_eval_row_ok(row):
            row = _merge_complementary_method_rows(prev_row, row)
        if mid not in by_id:
            order.append(mid)
        by_id[mid] = row
    return [by_id[mid] for mid in order if mid in by_id]




def _family_has_substantive_encoder_rows(
    rows: Sequence[Dict[str, Any]], family: str
) -> bool:
    """True only when this writer carries authoritative encoder results.

    A causal-only refresh may emit perfectly substantive rows for a method
    family, but it is not authoritative for that family's encoder inventory.
    Wiping the family in that case loses the saved AUC/specificity metrics and
    creates table ``NaN`` values even though ``done.json`` still has them.
    """
    return any(
        method_family(str(row.get("method") or "")) == family
        and row.get("status") != "error"
        and _has_encoder_payload(row.get("metrics"))
        for row in rows
    )


def _is_family_stub_error_payload(value: Any) -> bool:
    """True for ``{"error": "..."}``-only family raw blocks (failed stage stub)."""
    if not isinstance(value, dict) or not value.get("error"):
        return False
    keys = {str(k) for k in value.keys() if not str(k).startswith("_")}
    return keys <= {"error"}


def _enabled_method_backends(enabled: Set[str]) -> List[str]:
    return [b for b in METHOD_FAMILIES if b in enabled]


def _merge_train_raw(previous: Any, new: Any) -> Dict[str, Any]:
    """Merge train-stage backend inventories instead of replacing all of them."""
    prev = dict(previous or {}) if isinstance(previous, dict) else {}
    cur = dict(new or {}) if isinstance(new, dict) else {}
    out = {**prev, **cur}

    for key in ("backends", "tensor_backends"):
        values: List[Any] = []
        for value in [*(prev.get(key) or []), *(cur.get(key) or [])]:
            if value not in values:
                values.append(value)
        out[key] = values

    for key in ("none_by_class", "tensors_by_class"):
        out[key] = {
            **(prev.get(key) or {}),
            **(cur.get(key) or {}),
        }
    return out


def _mid_in_backends(mid: str, backends: Sequence[str]) -> bool:
    fam = _cli_family_for(method_family(mid))
    return fam in backends


def _merge_causal_raw(
    previous: Dict[str, Any],
    new: Dict[str, Any],
    *,
    enabled: Set[str],
) -> Dict[str, Any]:
    """Merge causal raw per method id — never replace a good row with a failure."""
    backends = _enabled_method_backends(enabled)
    if not backends:
        return dict(previous or {})

    out = dict(previous or {})
    by_method = dict(out.get("by_method") or {})
    panels = dict(out.get("poc_decoder_panels") or {})

    if "causal" in enabled:
        replace_splits = {
            str(split).strip().lower()
            for split in (new.get("replace_train_splits") or [])
            if str(split).strip()
        }
        if "tensors" in replace_splits:
            stale_tensor_ids = {
                str(mid)
                for mid in set(by_method) | set(panels)
                if ":tensors" in str(mid)
                and _mid_in_backends(str(mid), backends)
            }
            for mid in stale_tensor_ids:
                by_method.pop(mid, None)
                panels.pop(mid, None)
            out["summaries"] = [
                summary
                for summary in (out.get("summaries") or [])
                if not (
                    isinstance(summary, dict)
                    and ":tensors" in str(summary.get("method") or "")
                    and _mid_in_backends(str(summary.get("method") or ""), backends)
                )
            ]
        for mid, val in (new.get("by_method") or {}).items():
            mid = str(mid)
            if not _mid_in_backends(mid, backends):
                continue
            prev_val = by_method.get(mid)
            if (
                prev_val is not None
                and causal_raw_entry_ok(prev_val)
                and not causal_raw_entry_ok(val)
            ):
                continue
            by_method[mid] = val
        for mid, val in (new.get("poc_decoder_panels") or {}).items():
            mid = str(mid)
            if not _mid_in_backends(mid, backends):
                continue
            prev_panel = panels.get(mid)
            if prev_panel is not None and not val:
                continue
            panels[mid] = val
        for key, val in new.items():
            if key in ("by_method", "poc_decoder_panels", "summaries"):
                continue
            out[key] = val

        prev_summaries = {
            str(s.get("method")): s
            for s in (out.get("summaries") or [])
            if isinstance(s, dict) and s.get("method")
        }
        for s in new.get("summaries") or []:
            if not isinstance(s, dict):
                continue
            mid = str(s.get("method") or "")
            if not mid or not _mid_in_backends(mid, backends):
                continue
            prev_s = prev_summaries.get(mid)
            prev_ok = isinstance(prev_s, dict) and not (prev_s.get("meta") or {}).get("error")
            new_ok = not (s.get("meta") or {}).get("error")
            if prev_ok and not new_ok:
                continue
            prev_summaries[mid] = s
        out["summaries"] = list(prev_summaries.values())

    out["by_method"] = by_method
    out["poc_decoder_panels"] = panels
    return out


def merge_raw_sections(
    previous_raw: Dict[str, Any],
    new_raw: Dict[str, Any],
    *,
    enabled: Set[str],
) -> Dict[str, Any]:
    """Merge top-level ``raw`` blocks; enabled families overwrite (causal is per-backend)."""
    out = dict(previous_raw or {})
    for key, value in (new_raw or {}).items():
        if key in ("enabled", "cost", "cost_summary"):
            out[key] = value
            continue
        if key == "cost_ledger":
            # A stage checkpoint may be written while other method-array
            # cells have already committed their own timers.  Retain one
            # complete timer batch per method family rather than letting a
            # later checkpoint erase completed timing/RAM evidence.
            from cost_timer import merge_cost_ledger

            out[key] = merge_cost_ledger(out.get(key), value)
            continue
        if key == "causal":
            out["causal"] = _merge_causal_raw(
                dict(out.get("causal") or {}),
                value if isinstance(value, dict) else {},
                enabled=enabled,
            )
            continue
        if key == "train" and (
            "gradiend" in enabled or "actiend" in enabled
        ):
            out[key] = _merge_train_raw(out.get(key), value)
            continue
        if key == "localization" and "localization" in enabled:
            out[key] = value
            continue
        # Method-family named sections (sae / caa / …)
        if key in enabled and key in METHOD_FAMILIES:
            # Keep prior encode payload when this run only left a stub ``{error: ...}``.
            if _is_family_stub_error_payload(value) and isinstance(out.get(key), dict):
                continue
            out[key] = value
            continue
        if key not in out:
            out[key] = value
    out["enabled"] = sorted(enabled)
    out["merged_from_previous"] = True
    return out


_AUTO_COMPACT_RESULTS_BYTES = 1_000_000_000


def load_previous_results(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(output_dir) / "results.json"
    if not path.is_file():
        return None
    # Historical concurrent checkpoint merges could duplicate the same errors
    # geometrically. Repair pathological files *before* json.loads expands a
    # multi-gigabyte JSON document past the worker's host-RAM limit. The lock
    # ensures array-by-method cells never compact the same file concurrently.
    if path.stat().st_size > _AUTO_COMPACT_RESULTS_BYTES:
        try:
            from filelock import FileLock
            from scripts.compact_results_errors import compact

            lock = FileLock(str(path) + ".compact.lock", timeout=1800)
            with lock:
                if path.stat().st_size > _AUTO_COMPACT_RESULTS_BYTES:
                    before_size = path.stat().st_size
                    before_n, after_n = compact(path)
                    print(
                        f"Auto-compacted pathological results.json before load: "
                        f"errors {before_n} -> {after_n}; bytes "
                        f"{before_size} -> {path.stat().st_size}",
                        flush=True,
                    )
        except Exception as exc:
            # Preserve the ordinary loader behavior/error handling below. A
            # failed repair must not silently replace or delete results.json.
            print(f"Automatic results.json compaction failed: {exc}", flush=True)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def merge_study_payload(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
    *,
    enabled: Iterable[str],
) -> Dict[str, Any]:
    """Smart-merge ``current`` run into ``previous`` results for partial method runs."""
    enabled_set = {str(x) for x in enabled}
    if not previous:
        current = dict(current)
        current.setdefault("raw", {})["merged_from_previous"] = False
        return current

    prev_methods = list(previous.get("methods") or [])
    new_methods = list(current.get("methods") or [])
    current_train_raw = (current.get("raw") or {}).get("train") or {}
    replace_train_splits = {
        str(split).strip().lower()
        for split in (current_train_raw.get("requested_splits") or [])
        if str(split).strip()
    }
    merged_methods = merge_method_rows(
        prev_methods,
        new_methods,
        enabled=enabled_set,
        replace_train_splits=replace_train_splits,
        regimes=_normalize_slice((current.get("config") or {}).get("train_ablations")),
    )
    merged_raw = merge_raw_sections(
        dict(previous.get("raw") or {}),
        dict(current.get("raw") or {}),
        enabled=enabled_set,
    )
    errors = []
    # Keep prior errors for families not re-run; drop errors mentioning replaced families.
    for err in previous.get("errors") or []:
        text = str(err).lower()
        drop = False
        for fam in enabled_set:
            if fam in METHOD_FAMILIES and fam in text:
                drop = True
                break
        if not drop:
            errors.append(err)
    errors.extend(current.get("errors") or [])
    # Stage checkpoints and concurrent method cells repeatedly merge the same
    # historical errors. Without stable deduplication this list grows
    # geometrically; one gender_en file reached 13.7 GB and expanded past the
    # worker's 64 GB host-RAM limit while parsing.
    errors = list(dict.fromkeys(str(err) for err in errors if str(err)))

    out = dict(previous)
    prev_cfg = dict(previous.get("config") or {})
    cur_cfg = dict(current.get("config") or {})
    merged_cfg = {**prev_cfg, **cur_cfg}
    # A slice marker describes one writer, not the file: keeping the last writer's value
    # would mislabel the merged results.
    merged_cfg.pop("train_ablations", None)
    prev_en = {
        str(x)
        for x in (prev_cfg.get("enabled_methods") or [])
        if str(x) in METHOD_FAMILIES
    }
    cur_en = {
        str(x)
        for x in (cur_cfg.get("enabled_methods") or [])
        if str(x) in METHOD_FAMILIES
    }
    # Also fold this-run enabled set (partial --methods) into the union.
    run_en = {str(x) for x in enabled_set if str(x) in METHOD_FAMILIES}
    union_en = prev_en | cur_en | run_en
    if union_en:
        merged_cfg["enabled_methods"] = sorted(union_en)
    out.update(
        {
            "schema_version": current.get("schema_version") or previous.get("schema_version"),
            "experiment_id": current.get("experiment_id") or previous.get("experiment_id"),
            "model": current.get("model") or previous.get("model"),
            "hf_model": current.get("hf_model") or previous.get("hf_model"),
            "task": current.get("task") or previous.get("task"),
            "suite": current.get("suite") or previous.get("suite"),
            "config_hash": current.get("config_hash") or previous.get("config_hash"),
            "config": merged_cfg,
            "methods": merged_methods,
            "raw": merged_raw,
            "errors": errors,
            "created_at": previous.get("created_at") or current.get("created_at"),
            "started_at": previous.get("started_at") or current.get("started_at"),
            "finished_at": current.get("finished_at"),
            "compute": current.get("compute") or previous.get("compute"),
        }
    )
    # status from merged methods
    has_ok = any(r.get("status") == "ok" for r in merged_methods)
    has_err = bool(errors) or any(r.get("status") == "error" for r in merged_methods)
    if has_ok and not has_err:
        out["status"] = "ok"
    elif has_ok:
        out["status"] = "partial"
    else:
        out["status"] = current.get("status") or "error"
    if out.get("status") in ("ok", "partial"):
        out.pop("error", None)
        out.pop("traceback", None)
        out.pop("run_failed", None)
    out["merge"] = {
        "enabled_this_run": sorted(enabled_set),
        "n_previous_methods": len(prev_methods),
        "n_new_methods": len(new_methods),
        "n_merged_methods": len(merged_methods),
    }
    return out
