"""Full study deep pipeline: train → SAE → CAA → causal → localization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from study.config import StudyConfig, parse_methods_arg
from results_schema import ENCODER_EVAL_RULES_VERSION, encoder_eval_rules_current
from study.results_merge import load_previous_results, merge_study_payload
from study.stages.caa import run_caa_stage
from study.stages.causal import (
    apply_causal_crow_to_row,
    clear_stale_causal_error_on_encode_row,
    run_causal_stage,
)
from study.stages.localization import run_localization_stage
from study.stages.sae import run_sae_stage
from study.stages.train import (
    rehydrate_train_method_rows,
    reload_train_raw_from_artifacts,
    run_train_stage,
)
from study.tasks import TaskBundle
from sae_eval import release_sae_raw_heavy_memory, strip_sae_raw_for_json

from error_tracker import track_error


from study.json_util import json_ready as _json_ready


def _normalized_train_split_filter(value: Any) -> Set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = list(value)
    return {
        (
            "tensors"
            if str(item).strip().lower() in {"tensor", "by_tensor"}
            else str(item).strip().lower()
        )
        for item in values
        if str(item).strip()
    }


def _unrequested_train_rows(
    previous: Optional[Dict[str, Any]],
    *,
    backends: Set[str],
    requested_splits: Set[str],
) -> List[Dict[str, Any]]:
    """Carry forward rows outside a targeted split refresh.

    Result merging normally replaces an entire method family. A tensor-only
    refresh must instead retain that family's scalar/one-pole rows.
    """
    if not previous or not requested_splits:
        return []
    kept: List[Dict[str, Any]] = []
    for row in previous.get("methods") or []:
        mid = str(row.get("method") or "")
        family = mid.split(":", 1)[0]
        if family not in backends:
            continue
        split = str((row.get("metrics") or {}).get("gradiend_split") or "").lower()
        if not split and ":tensors" in mid:
            split = "tensors"
        if split not in requested_splits:
            kept.append(dict(row))
    return kept


def _stage_flags(cfg: StudyConfig, cli: Dict[str, Any]) -> Dict[str, bool]:
    """Causal / localization are stages (not methods): on by default, ``--skip-*`` off."""
    methods_cfg = dict((cfg.suite.get("methods") or {}))
    causal_on = (cfg.suite.get("causal") or {}).get("enabled", True) is not False
    localization_on = True
    if "localization" in methods_cfg and not methods_cfg.get("localization"):
        localization_on = False
    if cli.get("skip_causal"):
        causal_on = False
    if cli.get("skip_localization"):
        localization_on = False
    return {"causal": causal_on, "localization": localization_on}


def _default_enabled(cfg: StudyConfig, cli: Dict[str, Any]) -> Set[str]:
    """Methods from ``--methods`` / suite; causal+localization stages on unless skipped.

    ``--methods`` selects among gradiend / actiend / actiend_ridge / sae / caa / cga
    only. Causal and
    localization always attach to that selection unless ``--skip-causal`` /
    ``--skip-localization`` (or suite disables them).
    """
    methods_cfg = dict((cfg.suite.get("methods") or {}))
    stages = _stage_flags(cfg, cli)
    requested = parse_methods_arg(cli.get("methods"))

    if requested is not None:
        enabled: Set[str] = set(requested)
    else:
        # Omitted family → on; explicit false / empty → off.
        enabled = {"gradiend", "actiend", "sae", "caa"}
        # CGA is opt-in: a suite must list ``methods.cga`` to get it (unlike the
        # four headline families, which are on unless a suite disables them).
        if methods_cfg.get("cga"):
            enabled.add("cga")
        # CAGA (activation-gradient mean-diff, activation-steering causal) is opt-in
        # the same way -- a suite must list ``methods.caga``.
        if methods_cfg.get("caga"):
            enabled.add("caga")
        # AGIEND (learned activation-gradient encoder-decoder; trained like ACTIEND
        # but on the dL/dh signal, activation-steering causal) -- opt-in.
        if methods_cfg.get("agiend"):
            enabled.add("agiend")
        for name in ("gradiend", "actiend", "sae", "caa"):
            if name in methods_cfg and not methods_cfg.get(name):
                enabled.discard(name)
        if cli.get("skip_sae"):
            enabled.discard("sae")
        if cli.get("skip_caa"):
            enabled.discard("caa")

    # No SAE release → drop sae quietly
    if not (cfg.model or {}).get("sae_release"):
        enabled.discard("sae")

    if stages["causal"]:
        enabled.add("causal")
    if stages["localization"]:
        enabled.add("localization")
    # Opt-in mixed-site ACTIEND-PRE (training.actiend_pre); not default with actiend.
    if "actiend" in enabled and bool(
        (cfg.training or {}).get("actiend_pre")
        or (cfg.training or {}).get("train_actiend_pre")
    ):
        enabled.add("actiend_pre")
    # ACTIEND ridge decoder audit/variant piggybacks ACTIEND checkpoints. It is
    # opt-in per suite (``methods.actiend_ridge: true``); an explicit
    # ``--methods actiend_ridge`` is already in ``enabled`` from ``requested``.
    if "actiend" in enabled and methods_cfg.get("actiend_ridge"):
        enabled.add("actiend_ridge")
    return enabled


def _trainers_needed_after_train(enabled: Set[str]) -> bool:
    """Whether a stage after training consumes live trainer objects."""
    return bool({"sae", "caa", "causal", "localization"} & set(enabled))


def _artifact_backends_to_reload(enabled: Set[str]) -> tuple[str, ...]:
    """Trainer artifacts needed by a method-isolated resumed run.

    SAE and CAA can load the configured HF backbone directly. Reloading every
    IEND artifact for those cells wastes VRAM and defeats per-method Slurm
    sizing, especially for large models.
    """
    out = [b for b in ("gradiend", "actiend") if b in enabled]
    # Ridge is a causal-only decoder fitted over an ACTIEND checkpoint. Allow
    # ``--methods actiend_ridge`` to reload that source checkpoint without also
    # enabling/recomputing the learned ACTIEND causal policies.
    if "actiend_ridge" in enabled and "actiend" not in out:
        out.append("actiend")
    if "cga" in enabled:
        from study.training_profiles import CGA_BACKENDS

        out.extend(CGA_BACKENDS)
    if "caga" in enabled:
        from study.training_profiles import CAGA_BACKENDS

        out.extend(CAGA_BACKENDS)
    if "agiend" in enabled:
        from study.training_profiles import AGIEND_BACKENDS

        out.extend(AGIEND_BACKENDS)
    return tuple(out)


def _train_families_to_rehydrate(enabled: Set[str]) -> Set[str]:
    """Train-result families owned by this method-isolated pipeline cell."""
    recoverable = {
        backend for backend in ("gradiend", "actiend") if backend in enabled
    }
    if "actiend_ridge" in enabled:
        recoverable.add("actiend")
    if "cga" in enabled:
        from study.training_profiles import CGA_BACKENDS

        recoverable.update(CGA_BACKENDS)
    if "caga" in enabled:
        from study.training_profiles import CAGA_BACKENDS

        recoverable.update(CAGA_BACKENDS)
    if "agiend" in enabled:
        from study.training_profiles import AGIEND_BACKENDS

        recoverable.update(AGIEND_BACKENDS)
    return recoverable


def _missing_train_families_to_rehydrate(
    enabled: Set[str], *method_sets: Sequence[Mapping[str, Any]]
) -> Set[str]:
    """Requested train families absent from both old and newly produced rows."""
    present = {
        str(row.get("method") or "").split(":", 1)[0]
        for rows in method_sets
        for row in rows
        if row.get("method")
    }
    return _train_families_to_rehydrate(enabled) - present


def _poc_sae_report_config() -> Dict[str, Any]:
    """Snapshot gender-PoC SAE globals for REPORT header metadata."""
    try:
        import study.sae_engine as poc
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for key, attr in (
        ("sae_top_k", "SAE_TOP_K"),
        ("k_selection", "K_SELECTION"),
        ("layer_selection", "LAYER_SELECTION_PER_CLASS"),
        ("layer_selection_opp_fire", "LAYER_SELECTION_OPP_FIRE"),
        ("layer_selection_arad_out", "LAYER_SELECTION_ARAD_OUT"),
        ("layer_selection_jh_f1", "LAYER_SELECTION_JH_F1"),
    ):
        if hasattr(poc, attr):
            out[key] = getattr(poc, attr)
    for key, attr in (
        ("sae_readout_ks", "SAE_READOUT_KS"),
        ("sae_fixed_ks", "SAE_FIXED_KS"),
    ):
        if hasattr(poc, attr):
            val = getattr(poc, attr)
            out[key] = list(val) if val is not None else None
    return out


def _results_config_block(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    enabled: Set[str],
) -> Dict[str, Any]:
    from study.tasks import claim_classes_for_study

    abl = dict(cfg.raw.get("ablations") or {})
    from study.stages.train import train_ablations_filter

    _slice = train_ablations_filter(cfg)
    # Circuit one-pole: pair=false + one_pole=true (see induction / ioi YAMLs).
    one_pole = bool(abl.get("one_pole")) and abl.get("pair") is False
    opc = cfg.training.get("one_pole_classes")
    claim = claim_classes_for_study(
        bundle,
        one_pole=one_pole,
        one_pole_classes=opc if isinstance(opc, (list, tuple)) else None,
    )
    block: Dict[str, Any] = {
        "target_classes": [str(c) for c in bundle.classes],
        "claim_classes": claim,
        "model_key": cfg.model_key,
        "task_id": cfg.task_id,
        "suite_id": cfg.suite_id,
        "hf_model": cfg.hf_model,
        "ablations": abl,
        # Slice marker for concurrent ``--train-ablations`` writers (see results_merge).
        "train_ablations": sorted(_slice) if _slice else None,
        # Claim/fair tables omit families not in this list (e.g. race_one_pole).
        "enabled_methods": sorted(
            x
            for x in enabled
            if x in ("gradiend", "actiend", "actiend_ridge", "sae", "caa", "cga", "caga", "agiend")
        ),
    }
    if isinstance(opc, (list, tuple)) and opc:
        block["one_pole_classes"] = [str(c) for c in opc]
    block.update(_poc_sae_report_config())
    # The PoC module is GPT-2-specific and its model globals must never leak
    # into another model's results metadata (or back into GPT-2 after a
    # Pythia process mutated those globals). Scientific SAE identity comes
    # from the resolved model config used by this run.
    model_cfg = cfg.model or {}
    if model_cfg.get("sae_release") is not None:
        block["sae_release"] = model_cfg.get("sae_release")
    n_layers = model_cfg.get("n_layers")
    if n_layers is not None:
        n_layers = int(n_layers)
        block["sae_layer"] = n_layers - 1
        block["sae_layers"] = list(range(n_layers))
    # Suite/model defaults when PoC globals unavailable.
    if block.get("sae_top_k") is None and (cfg.model or {}).get("sae_top_k") is not None:
        block["sae_top_k"] = cfg.model.get("sae_top_k")
    return block


def _build_results_payload(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    enabled: Set[str],
    method_rows: List[Dict[str, Any]],
    errors: List[str],
    raw: Dict[str, Any],
    started: str,
    status: str,
    finished: Optional[str] = None,
    pipeline_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the results.json blob (checkpoint or final)."""
    config_block = _results_config_block(cfg, bundle, enabled=enabled)
    for key, val in (raw.get("sae_report") or {}).items():
        if val is not None and config_block.get(key) is None:
            config_block[key] = val
    for key, val in ((raw.get("sae") or {})).items():
        if key in {
            "k_selection",
            "layer_selection",
            "layer_selection_opp_fire",
            "sae_readout_ks",
            "sae_top_k",
            "selected_layer_global",
            "selected_layer_by_class",
        } and val is not None and config_block.get(key) is None:
            config_block[key] = val
    payload: Dict[str, Any] = {
        "schema_version": __import__(
            "results_schema", fromlist=["SCHEMA_VERSION"]
        ).SCHEMA_VERSION,
        "activation_protocol_version": __import__(
            "activation_protocol", fromlist=["ACTIVATION_PROTOCOL_VERSION"]
        ).ACTIVATION_PROTOCOL_VERSION,
        "experiment_id": f"{cfg.model_key}/{cfg.task_id}",
        "model": cfg.model_key,
        "hf_model": cfg.hf_model,
        "task": cfg.task_id,
        "suite": cfg.suite_id,
        "config_hash": cfg.config_hash(),
        "config": config_block,
        "status": status,
        "methods": method_rows,
        "raw": raw,
        "errors": errors,
        "created_at": started,
        "started_at": started,
        "finished_at": finished,
    }
    if pipeline_stage:
        payload["pipeline_stage"] = pipeline_stage
    return payload


def _atomic_write_results(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def _results_lock(output_dir: Path):
    """File lock guarding read-merge-write of ``output_dir/results.json``.

    ``tmp.replace(path)`` in ``_atomic_write_results`` already makes any
    single write atomic (readers never see a torn/partial file), but that
    alone does not make concurrent *writers* safe: nothing previously
    serialized the read-merge-write sequence, so two overlapping processes
    (e.g. separate ``--methods gradiend`` / ``--methods actiend`` Slurm jobs
    against the same output dir) could each merge against their own
    already-stale copy and the later write would silently discard whatever
    method rows the other process had already committed. This lock, combined
    with re-reading the file fresh from disk *after* acquiring it (see
    ``merge_and_write_results_locked``), closes that race.
    """
    from filelock import FileLock

    output_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(output_dir / "results.json.lock"), timeout=600.0)


def merge_and_write_results_locked(
    output_dir: Path,
    payload: Dict[str, Any],
    *,
    enabled: Set[str],
    after_merge: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Merge ``payload`` onto the *current* on-disk results.json and write it back.

    Re-reads ``results.json`` fresh from disk under a file lock immediately
    before merging, instead of trusting a ``previous`` snapshot captured
    earlier in the run (at process start, potentially long before this
    write). That earlier snapshot can be arbitrarily stale by the time a
    long-running train/SAE/causal stage finishes — if a concurrent process
    writing to the same output dir committed rows in the meantime, merging
    against the stale snapshot would silently drop them.

    ``after_merge``, if given, is applied to the merged payload (e.g. to force
    ``status``/``pipeline_stage`` back to "partial" for a checkpoint) *before*
    the write, inside the same lock — not as a separate second write, which
    would reopen the same race it's meant to close. Returns the payload
    actually written so the caller can log/derive status from it.
    """
    with _results_lock(output_dir):
        fresh_previous = load_previous_results(output_dir)
        if fresh_previous is not None:
            payload = merge_study_payload(fresh_previous, payload, enabled=enabled)
        if after_merge is not None:
            payload = after_merge(payload)
        _atomic_write_results(output_dir / "results.json", payload)
    return payload


def write_error_results_locked(output_dir: Path, error_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record a failed method-array cell without erasing concurrent results.

    Unlike normal pipeline payloads, a top-level exception may occur before an
    enabled-method set is available. Preserve every method row currently on
    disk and overlay only the failure metadata under the same results lock.
    """
    with _results_lock(output_dir):
        previous = load_previous_results(output_dir)
        payload = dict(error_payload)
        if previous is not None:
            payload = dict(previous)
            payload.update(
                {
                    key: value
                    for key, value in error_payload.items()
                    if key not in {"methods"}
                }
            )
            payload["methods"] = list(previous.get("methods") or [])
            payload["run_failed"] = True
        _atomic_write_results(output_dir / "results.json", payload)
    return payload


def write_stage_checkpoint(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    enabled: Set[str],
    method_rows: List[Dict[str, Any]],
    errors: List[str],
    raw: Dict[str, Any],
    started: str,
    stage: str,
    previous: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``results.json`` after a stage. Never ``status=ok`` (skip-existing must re-enter).

    ``previous`` is the results.json payload that existed on disk *before this
    run started* (``None`` if there wasn't one). When given, the checkpoint is
    merged against it via the same ``merge_study_payload`` used for the final
    write, so a stage that hasn't run yet this pass (or a run that crashes
    before finishing) keeps whatever complete rows were already there instead
    of the file regressing to only this run's progress-so-far. Families whose
    rows *have* landed in ``method_rows`` by this checkpoint still replace the
    prior versions, same as the final merge.
    """
    # Preserve cost evidence even if a later stage crashes.  Previously the
    # durable ledger was written only at final completion; a SAE import error
    # could therefore discard hours of completed GRADIEND/ACTIEND/CGA work.
    # ``merge_raw_sections`` merges this batch family-wise under the results
    # lock, which remains safe for method-isolated array cells.
    try:
        from cost_timer import get_cost_records, summarize_cost_records

        records = get_cost_records()
        summary = summarize_cost_records(records)
        raw = dict(raw)
        raw["cost"] = records
        raw["cost_ledger"] = records
        raw["cost_summary"] = {k: v for k, v in summary.items() if k != "stages"}
    except Exception:
        # Checkpoints must not turn a completed scientific stage into a
        # failure merely because optional accounting failed.
        pass

    if errors and not method_rows:
        status = "error"
    else:
        status = "partial"
    payload = _build_results_payload(
        cfg,
        bundle,
        enabled=enabled,
        method_rows=method_rows,
        errors=errors,
        raw=raw,
        started=started,
        status=status,
        finished=datetime.now(timezone.utc).isoformat(),
        pipeline_stage=stage,
    )
    # Always merge against whatever is truly on disk *right now*, under a
    # lock — not against the ``previous`` argument (a snapshot captured once,
    # possibly much earlier, at process start in run_deep_pipeline, and
    # possibly None even when a concurrent process has since created the
    # file). A concurrent process writing to this same output dir (e.g. a
    # separate --methods actiend job while this one runs --methods gradiend
    # — including the case where BOTH start with no results.json at all, so
    # both capture previous=None) may have committed rows in the meantime
    # that ``previous`` doesn't reflect; deciding whether to merge from
    # ``previous is None`` (as an earlier version of this function did) means
    # exactly that concurrent-first-write case skips merging entirely and
    # silently clobbers the other writer. Always go through
    # merge_and_write_results_locked, which re-reads fresh under the lock and
    # merges only if something is actually there now — a genuine first-ever
    # write still degrades correctly to a plain write (fresh_previous is
    # None inside the lock too), just via the same code path.
    path = cfg.output_dir / "results.json"

    def _force_checkpoint_status(merged: Dict[str, Any]) -> Dict[str, Any]:
        # merge_study_payload derives status from the merged rows (can read
        # "ok" once any prior family already succeeded) — checkpoints must
        # stay "partial"/"error" so --skip-existing always re-enters an
        # unfinished job.
        merged["status"] = status
        merged["pipeline_stage"] = stage
        return merged

    payload = merge_and_write_results_locked(
        cfg.output_dir, payload, enabled=enabled, after_merge=_force_checkpoint_status
    )
    n_methods = len(payload.get("methods") or method_rows)
    print(
        f"checkpoint {stage}: wrote {path} status={status} "
        f"methods={n_methods}"
        + (f" ({len(method_rows)} from this run)" if previous is not None else ""),
        flush=True,
    )
    try:
        from study.progress import write_pipeline_progress

        write_pipeline_progress(
            cfg.output_dir,
            stage=f"checkpoint_{stage}",
            status=status,
            n_methods=n_methods,
        )
    except Exception:
        pass
    return path


def _caa_row_is_pair(method_id: Any) -> bool:
    """True for a two-pole CAA method id (``caa:F-M:C:...``), false for one-pole."""
    from study.method_ids import is_pair_key

    parts = str(method_id or "").split(":")
    return len(parts) >= 3 and is_pair_key(parts[1])


_CAUSAL_ONLY_FAMILIES = frozenset({"sae", "caa", "cga", "caga", "causal"})


def assert_causal_only_is_reuse_only(
    enabled: Set[str], previous: Optional[Mapping[str, Any]]
) -> None:
    """``--causal-only`` must never train or encode: it needs stored SAE/CAA blocks.

    Raises instead of silently recomputing an encoding (hours per task on a 2B+
    model) when a stored block is missing, or when another family is enabled.
    """
    extra = sorted(set(enabled) - _CAUSAL_ONLY_FAMILIES)
    if extra:
        raise RuntimeError(
            f"--causal-only only supports SAE/CAA/CGA/CAGA causal from stored encodings "
            f"and checkpoints; also enabled: {extra}. Pass --methods sae caa cga caga."
        )
    if not previous:
        raise RuntimeError(
            "--causal-only needs a stored results.json in the output dir "
            "(SAE/CAA encodings are reused, never recomputed)."
        )
    from study.results_merge import method_family

    for family, raw_key in (("sae", "sae"), ("caa", "caa")):
        if family not in enabled:
            continue
        rows = [
            r
            for r in (previous.get("methods") or [])
            if method_family(str(r.get("method") or "")) in (
                {"sae", "sae_pre"} if family == "sae" else {"caa"}
            )
        ]
        block = (previous.get("raw") or {}).get(raw_key) or {}
        if not rows or block.get("error") or (family == "caa" and not block.get("vectors")):
            raise RuntimeError(
                f"--causal-only: stored {family.upper()} encoding missing/erroneous in "
                f"results.json (rows={len(rows)}, error={bool(block.get('error'))}); "
                "refusing to re-encode."
            )


def should_reuse_caa_encode(
    prev_caa_rows: Sequence[Mapping[str, Any]],
    prev_caa_raw: Mapping[str, Any],
    *,
    want_pair: bool,
    want_one_pole: bool,
) -> bool:
    """Whether ``--skip-existing`` may reuse prior CAA rows instead of recomputing.

    Reuse only when prior CAA output exists, is entirely error-free, AND already
    covers every pole regime the current config requests. The last clause is the
    fix for the coarse all-or-nothing reuse that would otherwise short-circuit the
    whole CAA stage on the mere presence of any error-free row — so a newly-wired
    regime (two-pole ``caa:F-M:C:*`` after CAA gained pair support) would never
    compute, since a brand-new id cannot be "already done". Same spirit as
    ``_should_skip_existing`` re-entering when a suite adds ablations; the causal
    stage already does the equivalent per-id.
    """
    if not prev_caa_rows or (prev_caa_raw or {}).get("error"):
        return False
    if not all(
        str(r.get("status") or "") != "error" and not r.get("error")
        for r in prev_caa_rows
    ):
        return False
    have_pair = any(_caa_row_is_pair(r.get("method")) for r in prev_caa_rows)
    have_one_pole = any(not _caa_row_is_pair(r.get("method")) for r in prev_caa_rows)
    if want_pair and not have_pair:
        return False
    if want_one_pole and not have_one_pole:
        return False
    return True


def run_deep_pipeline(
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    smoke: bool = False,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full analysis stack for one task."""
    cli = dict(cli_overrides or {})
    if smoke:
        cfg.raw["_smoke"] = True
        from study.training_profiles import apply_smoke_model_layers

        apply_smoke_model_layers(cfg)
    cfg.raw["cli"] = cli
    enabled = _default_enabled(cfg, cli)
    fail_fast = bool(cli.get("fail_fast"))
    previous = load_previous_results(cfg.output_dir)
    causal_only = bool(cli.get("causal_only"))
    if causal_only:
        assert_causal_only_is_reuse_only(enabled, previous)
    skip_existing = bool(cli.get("skip_existing"))
    refresh_causal = bool(cli.get("refresh_causal"))
    refresh_encoder_eval = bool(cli.get("refresh_encoder_eval"))
    force_causal = bool(cli.get("force_causal"))
    partial_methods = parse_methods_arg(cli.get("methods")) is not None
    # Merge whenever a prior results.json exists — not just for --skip-existing /
    # --refresh-causal / --methods runs. A plain full rerun used to start
    # method_rows at [] and only reconcile with `previous` at the very end
    # (guarded by the flags above), so every intermediate checkpoint — and any
    # results.json left behind by a crash/timeout before the final write —
    # reflected only this run's progress-so-far, silently dropping prior
    # complete rows for families/stages not yet reached this pass. Cache
    # invalidation (force a truly clean slate) is the operator's call: delete
    # results.json / the relevant artifacts/*/done.json before rerunning, per
    # PROTOCOL_INVALIDATION.md.
    should_merge = previous is not None
    if should_merge:
        print(
            f"Merging into existing {cfg.output_dir / 'results.json'} "
            f"(skip_existing={skip_existing} refresh_causal={refresh_causal} "
            f"methods={sorted(enabled) if partial_methods else 'all'}) "
            f"({len(previous.get('methods') or [])} prior methods)",
            flush=True,
        )
    else:
        print(
            f"no prior {cfg.output_dir / 'results.json'}; "
            f"will checkpoint after each stage (status=partial until the job finishes)",
            flush=True,
        )

    started = datetime.now(timezone.utc).isoformat()
    method_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    raw: Dict[str, Any] = {"enabled": sorted(enabled)}

    def record_stage_errors(stage: str, stage_errors: Any) -> None:
        """Keep normal runs restartable; make early-fail mode genuinely strict."""
        messages = [str(error) for error in (stage_errors or []) if str(error)]
        errors.extend(messages)
        if fail_fast and messages:
            raise RuntimeError(f"{stage} reported an error: {messages[0]}")

    def checkpoint(stage: str) -> None:
        write_stage_checkpoint(
            cfg,
            bundle,
            enabled=enabled,
            previous=previous,
            method_rows=method_rows,
            errors=errors,
            raw=raw,
            started=started,
            stage=stage,
        )

    train_raw_none: Dict[str, Dict[str, Any]] = {}
    train_raw_tensors: Dict[str, Dict[str, Any]] = {}
    train_raw_none_by_class: Dict[str, Dict[str, Dict[str, Any]]] = {}
    train_raw_tensors_by_class: Dict[str, Dict[str, Dict[str, Any]]] = {}
    train_backends = [b for b in ("gradiend", "actiend", "cga", "caga", "agiend") if b in enabled]
    requested_train_splits = _normalized_train_split_filter(cli.get("train_splits"))
    if requested_train_splits:
        preserved = _unrequested_train_rows(
            previous,
            backends=set(train_backends),
            requested_splits=requested_train_splits,
        )
        method_rows.extend(preserved)
        print(
            f"targeted train splits={sorted(requested_train_splits)}; "
            f"preserving {len(preserved)} prior rows from other train splits",
            flush=True,
        )
    needs_trainers = bool(train_backends or ({"sae", "caa", "causal", "localization"} & enabled))
    # actiend_pre is opt-in via training.actiend_pre (added to enabled above).
    if train_backends:
        # Do not keep one full HF model per pair/one-pole ablation alive while
        # training the next artifact. SAE/CAA have a direct HF-backbone path;
        # causal is rehydrated explicitly below from the completed checkpoints.
        retain_trainers = False
        train_out = run_train_stage(
            cfg,
            bundle,
            enabled_backends=train_backends,
            train_splits=cli.get("train_splits"),
            smoke=smoke,
            skip_existing=bool(cli.get("skip_existing")),
            rerun_collapsed=bool(cli.get("rerun_collapsed")),
            rerun_orphaned_artifacts=bool(cli.get("rerun_orphaned_artifacts")),
            fail_fast=fail_fast,
            retain_trainers=retain_trainers,
            previous_methods=(previous or {}).get("methods") if previous else None,
            refresh_encoder_eval=refresh_encoder_eval,
        )
        method_rows.extend(train_out.get("methods") or [])
        record_stage_errors("train", train_out.get("errors"))
        train_raw_none = train_out.get("train_raw_none") or {}
        train_raw_tensors = train_out.get("train_raw_tensors") or {}
        train_raw_none_by_class = train_out.get("train_raw_none_by_class") or {}
        train_raw_tensors_by_class = train_out.get("train_raw_tensors_by_class") or {}
        raw["train"] = {
            "n_jobs": len(train_out.get("all_trains") or []),
            "backends": list(train_raw_none.keys()),
            "tensor_backends": list(train_raw_tensors.keys()),
            "requested_splits": sorted(requested_train_splits),
            "none_by_class": {
                b: sorted(m.keys()) for b, m in train_raw_none_by_class.items()
            },
            "tensors_by_class": {
                b: sorted(m.keys()) for b, m in train_raw_tensors_by_class.items()
            },
        }
        checkpoint("train")
    elif needs_trainers:
        reload_backends = _artifact_backends_to_reload(enabled)
        if reload_backends:
            print(
                "No train backends in --methods; reloading requested trainers "
                "from artifacts/",
                flush=True,
            )
            reloaded = reload_train_raw_from_artifacts(
                cfg,
                bundle,
                backends=reload_backends,
            )
        else:
            print(
                "No train backends in --methods; SAE/CAA will load the HF "
                "backbone directly",
                flush=True,
            )
            reloaded = {}
        # Backward compat: old reload returned {backend: raw} directly.
        if reloaded and "train_raw_none" in reloaded:
            train_raw_none = reloaded.get("train_raw_none") or {}
            train_raw_none_by_class = reloaded.get("train_raw_none_by_class") or {}
        else:
            train_raw_none = reloaded or {}
            train_raw_none_by_class = {}
        train_raw_tensors = {}
        train_raw_tensors_by_class = {}
        raw["train"] = {
            "n_jobs": 0,
            "backends": list(train_raw_none.keys()),
            "none_by_class": {
                b: sorted(m.keys()) for b, m in train_raw_none_by_class.items()
            },
            "reloaded": bool(reload_backends),
        }
        if not train_raw_none and not train_raw_none_by_class:
            # SAE/CAA can load the HF backbone themselves; only warn.
            need_ckpt = bool({"gradiend", "actiend"} & enabled)
            msg = (
                "no trainer checkpoints under artifacts/ "
                f"(looked for gradiend/actiend). "
                f"{'Required for gradiend/actiend.' if need_ckpt else 'SAE/CAA will load HF backbone directly.'}"
            )
            print(f"reload: {msg}", flush=True)
            if need_ckpt:
                if fail_fast:
                    raise RuntimeError(msg)
                errors.append(msg)

    sae_raw: Dict[str, Any] = {}
    if "sae" in enabled:
        reuse_sae = False
        if skip_existing and previous:
            from study.results_merge import method_family
            from study.stages.sae import _sae_rows_are_substantive

            prev_sae_rows = [
                r
                for r in (previous.get("methods") or [])
                if method_family(str(r.get("method") or "")) in {"sae", "sae_pre"}
            ]
            prev_sae_raw = (previous.get("raw") or {}).get("sae") or {}
            # --refresh-encoder-eval must recompute SAE encoding too: without this
            # gate the flag was a silent no-op for SAE (Llama ioi_mib/key_value/
            # function_composition kept their NaN roc_auc_other after a "refresh").
            reuse_sae = (
                _sae_rows_are_substantive(prev_sae_rows)
                and not prev_sae_raw.get("error")
                # Rows from before the validation-frozen Spec_n/Excl fix carry
                # test-fitted oracle thresholds; never reuse them silently. This
                # stamp check replaces the bare ``not refresh_encoder_eval`` gate:
                # --refresh-encoder-eval still forces SAE re-encoding of stale rows
                # (they are unstamped) but no longer redoes already-migrated ones.
                and encoder_eval_rules_current(
                    prev_sae_raw.get("encoder_eval_rules_version")
                )
            )
        if causal_only:
            # Guarded above: stored rows exist. Reuse regardless of the rules stamp.
            reuse_sae = True
        if reuse_sae:
            print(
                "skip-existing: reusing prior SAE encode rows (no encode error)",
                flush=True,
            )
            method_rows.extend(
                clear_stale_causal_error_on_encode_row(dict(r))
                for r in prev_sae_rows
            )
            sae_raw = dict(prev_sae_raw)
            raw["sae"] = strip_sae_raw_for_json(sae_raw)
        else:
            sae_out = run_sae_stage(
                cfg, bundle, train_raw_none, fail_fast=fail_fast
            )
            method_rows.extend(sae_out.get("methods") or [])
            record_stage_errors("sae", sae_out.get("errors"))
            sae_raw = sae_out.get("raw") or {}
            raw["sae"] = strip_sae_raw_for_json(sae_raw)
            release_sae_raw_heavy_memory(sae_raw)
        # Stamp after both branches: a reused block is current by the gate above,
        # a recomputed one was just produced by the current encode path, and the
        # JSON-safe strip may drop unknown keys.
        if causal_only:
            # Reused rows keep whatever stamp they had; never relabel them current.
            _prev_stamp = (previous.get("raw") or {}).get("sae", {}).get("encoder_eval_rules_version")
            if _prev_stamp is not None:
                raw["sae"]["encoder_eval_rules_version"] = _prev_stamp
        else:
            raw["sae"]["encoder_eval_rules_version"] = ENCODER_EVAL_RULES_VERSION
        # Persist selection metadata into config for REPORT header even if PoC
        # globals were never snapshotted at payload-build time.
        for key in (
            "k_selection",
            "layer_selection",
            "layer_selection_opp_fire",
            "sae_readout_ks",
            "sae_top_k",
            "selected_layer_global",
            "selected_layer_by_class",
        ):
            if key in sae_raw and sae_raw[key] is not None:
                raw.setdefault("sae_report", {})[key] = sae_raw[key]
        checkpoint("sae")

    caa_raw: Dict[str, Any] = {}
    if "caa" in enabled:
        reuse_caa = False
        prev_caa_rows: List[Dict[str, Any]] = []
        prev_caa_raw: Dict[str, Any] = {}
        if skip_existing and previous:
            from study.results_merge import method_family

            prev_caa_rows = [
                r
                for r in (previous.get("methods") or [])
                if method_family(str(r.get("method") or "")) == "caa"
            ]
            prev_caa_raw = (previous.get("raw") or {}).get("caa") or {}
            ablations_cfg = cfg.raw.get("ablations") or {}
            want_pair = bool(ablations_cfg.get("pair", True))
            want_one_pole = bool(ablations_cfg.get("one_pole", True))
            reuse_caa = should_reuse_caa_encode(
                prev_caa_rows,
                prev_caa_raw,
                want_pair=want_pair,
                want_one_pole=want_one_pole,
            )
            # --refresh-encoder-eval used to be a silent no-op for CAA (this gate
            # ignored it). Staleness is now decided by the rules stamp: rows from
            # before the validation-frozen Spec_n/Excl fix are recomputed (test-only:
            # stored vectors + validation readouts are reused), current ones never
            # are, so a resumed refresh does not redo finished work.
            if reuse_caa and not encoder_eval_rules_current(
                prev_caa_raw.get("encoder_eval_rules_version")
            ):
                reuse_caa = False
                print(
                    "skip-existing: prior CAA rows predate validation-frozen "
                    "Spec_n/Excl rules — re-encoding test split only",
                    flush=True,
                )
            if not reuse_caa and prev_caa_rows:
                have_pair = any(_caa_row_is_pair(r.get("method")) for r in prev_caa_rows)
                have_one_pole = any(
                    not _caa_row_is_pair(r.get("method")) for r in prev_caa_rows
                )
                if (want_pair and not have_pair) or (want_one_pole and not have_one_pole):
                    print(
                        "skip-existing: prior CAA rows lack a requested pole regime "
                        f"(want_pair={want_pair} have_pair={have_pair}; "
                        f"want_one_pole={want_one_pole} have_one_pole={have_one_pole})"
                        " — recomputing CAA encode",
                        flush=True,
                    )
        if causal_only:
            reuse_caa = True  # guarded above: stored vectors + rows exist
        if reuse_caa:
            print(
                "skip-existing: reusing prior CAA encode rows (no encode error)",
                flush=True,
            )
            method_rows.extend(dict(r) for r in prev_caa_rows)
            caa_raw = dict(prev_caa_raw)
            raw["caa"] = {k: v for k, v in caa_raw.items() if not str(k).startswith("_")}
        else:
            caa_out = run_caa_stage(
                cfg,
                bundle,
                train_raw_none,
                fail_fast=fail_fast,
                previous_caa=(
                    {"raw": prev_caa_raw, "rows": prev_caa_rows}
                    if prev_caa_rows
                    else None
                ),
            )
            method_rows.extend(caa_out.get("methods") or [])
            record_stage_errors("caa", caa_out.get("errors"))
            caa_raw = caa_out.get("raw") or {}
            raw["caa"] = {k: v for k, v in caa_raw.items() if not str(k).startswith("_")}
        if causal_only:
            _prev_stamp = (previous.get("raw") or {}).get("caa", {}).get("encoder_eval_rules_version")
            if _prev_stamp is not None:
                raw["caa"]["encoder_eval_rules_version"] = _prev_stamp
        else:
            raw["caa"]["encoder_eval_rules_version"] = ENCODER_EVAL_RULES_VERSION
        checkpoint("caa")

    if "causal" in enabled:
        # Training releases completed trainers eagerly to keep peak VRAM bounded
        # across pair + one-pole ablations. Causal evaluation needs trainer
        # wrappers (unlike SAE/CAA), so rehydrate them only after training and
        # encode stages have finished. This is intentionally one lifecycle
        # boundary rather than retaining every trainer during training.
        if train_backends and not any(
            (raw_map or {}).get("trainer")
            for raw_map in train_raw_none.values()
        ):
            reload_backends = _artifact_backends_to_reload(enabled)
            if reload_backends:
                print(
                    "causal: reloading completed trainer artifacts after "
                    "eager training cleanup",
                    flush=True,
                )
                reloaded = reload_train_raw_from_artifacts(
                    cfg,
                    bundle,
                    backends=reload_backends,
                    split_modes=(
                        ("none",)
                        if "actiend_ridge" in enabled and "actiend" not in enabled
                        else ("none", "tensors")
                    ),
                )
                train_raw_none = reloaded.get("train_raw_none") or {}
                train_raw_tensors = reloaded.get("train_raw_tensors") or {}
                train_raw_none_by_class = reloaded.get(
                    "train_raw_none_by_class"
                ) or {}
                train_raw_tensors_by_class = reloaded.get(
                    "train_raw_tensors_by_class"
                ) or {}
        causal_out = run_causal_stage(
            cfg,
            bundle,
            train_raw_none,
            train_raw_tensors=train_raw_tensors,
            train_raw_none_by_class=train_raw_none_by_class,
            train_raw_tensors_by_class=train_raw_tensors_by_class,
            sae_raw=sae_raw,
            caa_raw=caa_raw,
            enabled=enabled,
            previous=previous if skip_existing or refresh_causal else None,
            skip_existing=skip_existing or refresh_causal,
            force_causal=force_causal,
            fail_fast=fail_fast,
        )
        record_stage_errors("causal", causal_out.get("errors"))
        raw["causal"] = causal_out.get("raw") or {}
        if requested_train_splits:
            # Old causal payloads for freshly retrained splits are stale even
            # when the replacement evaluation reports an error.
            raw["causal"]["replace_train_splits"] = sorted(requested_train_splits)
        # Merge onto shared PoC method ids; append causal-only ablations.
        by_existing = {str(r.get("method")): r for r in method_rows if r.get("method")}
        for crow in causal_out.get("methods") or []:
            mid = str(crow.get("method") or "")
            if not mid:
                continue
            if mid in by_existing:
                apply_causal_crow_to_row(by_existing[mid], crow)
            else:
                method_rows.append(crow)
                by_existing[mid] = crow
        checkpoint("causal")

    if "localization" in enabled:
        reuse_loc = False
        tensor_only_refresh = requested_train_splits == {"tensors"}
        if (skip_existing or tensor_only_refresh) and previous:
            prev_loc_rows = [
                r
                for r in (previous.get("methods") or [])
                if str(r.get("method") or "") == "localization"
            ]
            prev_loc_raw = (previous.get("raw") or {}).get("localization") or {}
            reuse_loc = bool(prev_loc_rows) and not prev_loc_raw.get("error")
            if reuse_loc:
                reuse_loc = all(
                    str(r.get("status") or "") != "error" and not r.get("error")
                    for r in prev_loc_rows
                )
        if reuse_loc:
            print(
                (
                    "tensor-only refresh: preserving prior localization "
                    "(localization does not consume tensor trainers)"
                    if tensor_only_refresh
                    else "skip-existing: reusing prior localization (no error)"
                ),
                flush=True,
            )
            method_rows.extend(dict(r) for r in prev_loc_rows)
            raw["localization"] = dict(prev_loc_raw)
        else:
            loc_out = run_localization_stage(
                cfg, bundle, train_raw_none, sae_raw, fail_fast=fail_fast
            )
            method_rows.extend(loc_out.get("methods") or [])
            record_stage_errors("localization", loc_out.get("errors"))
            raw["localization"] = loc_out.get("raw") or {}
        checkpoint("localization")

    # Gender-only proxy transfer scaffolding
    if str(cfg.task_id).startswith("gender_en") and bundle.proxies:
        from study.proxy_metrics import bank_stats, transfer_summary

        raw["proxy_banks"] = {name: bank_stats(df) for name, df in bundle.proxies.items()}
        for row in list(method_rows):
            if not str(row.get("method", "")).startswith(
                (
                    "gradiend:",
                    "actiend:",
                    "actiend_ridge:",
                    "actiend_pre:",
                    "sae:",
                    "sae_pre:",
                    "caa:",
                    "cga:",
                    "cga_tensor_norm:",
                    "caga:",
                    "agiend:",
                )
            ):
                continue
            transfer = dict(row)
            transfer["proxy"] = "her_his"
            transfer["method"] = f"{row['method']}|proxy=her_his"
            transfer["status"] = "partial"
            base_metrics = dict(row.get("metrics") or {})
            transfer["metrics"] = {
                **base_metrics,
                "transfer": True,
                **transfer_summary(in_domain=base_metrics, transfer={}),
            }
            method_rows.append(transfer)

    status = "ok" if method_rows and not errors else ("partial" if method_rows else "error")
    finished = datetime.now(timezone.utc).isoformat()
    payload = _build_results_payload(
        cfg,
        bundle,
        enabled=enabled,
        method_rows=method_rows,
        errors=errors,
        raw=raw,
        started=started,
        status=status,
        finished=finished,
        pipeline_stage="done",
    )

    # Merge with prior results when resuming / refreshing (never full replace).
    if should_merge:
        prev_methods = list(previous.get("methods") or [])
        # Recover only train families owned by this method-isolated cell.  The
        # previous unconditional GRADIEND+ACTIEND set could load a 26 GiB
        # unrelated trainer at final merge (and OOM after successful causal
        # evaluation) merely because a concurrent writer had not committed its
        # row yet.
        missing_train = _missing_train_families_to_rehydrate(
            enabled, prev_methods, method_rows
        )
        if missing_train:
            rehydrated = rehydrate_train_method_rows(
                cfg, bundle, backends=tuple(sorted(missing_train)), smoke=smoke
            )
            if rehydrated:
                previous = dict(previous)
                previous["methods"] = list(prev_methods) + rehydrated
                prev_methods = list(previous.get("methods") or [])
        # Recover SAE encode rows from artifacts/sae snapshot when results only
        # has a family stub (or no SAE) after an older wipe.
        from study.results_merge import method_family
        from study.stages.sae import _load_sae_encode_snapshot, _sae_rows_are_substantive

        prev_sae = [
            r
            for r in prev_methods
            if method_family(str(r.get("method") or "")) in {"sae", "sae_pre"}
        ]
        if not _sae_rows_are_substantive(prev_sae):
            sae_snap = _load_sae_encode_snapshot(cfg)
            if sae_snap:
                print(
                    f"rehydrate: restoring {len(sae_snap)} SAE encode rows from artifacts/sae/",
                    flush=True,
                )
                previous = dict(previous)
                kept = [
                    r
                    for r in (previous.get("methods") or [])
                    if method_family(str(r.get("method") or "")) in {"sae", "sae_pre"}
                ]
                previous["methods"] = kept + sae_snap
        payload = merge_study_payload(previous, payload, enabled=enabled)
        status = str(payload.get("status") or status)
        print(f"Merged results: {payload.get('merge')}", flush=True)

    try:
        from cost_timer import get_cost_records, merge_cost_ledger, summarize_cost_records

        records = get_cost_records()
        summary = summarize_cost_records(records)
        summary_slim = {k: v for k, v in summary.items() if k != "stages"}
        raw = payload.setdefault("raw", {})
        raw["cost_ledger"] = merge_cost_ledger(raw.get("cost_ledger"), records)
        raw["cost"] = records
        raw["cost_summary"] = summary_slim
        payload["compute"] = summary_slim
    except Exception:
        if fail_fast:
            raise

    try:
        from suitability import attach_feature_suitability

        suit = cfg.suitability
        attach_feature_suitability(
            payload,
            tau_c=float(suit.get("causal_effect_threshold", 0.05)),
            soft_causal=bool(suit.get("soft_causal", False)),
            e_ok=float(suit.get("e_ok", 0.8)),
        )
    except Exception as exc:
        track_error(exc, context="suitability attach")
        if fail_fast:
            raise

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    # Locked + re-merged against the current on-disk state, not just the
    # `previous` snapshot captured at process start above — see
    # merge_and_write_results_locked's docstring. Safe even though `payload`
    # was already merged against (possibly stale) `previous` earlier in
    # `should_merge`: for this run's own `enabled` families, `payload`'s rows
    # still win here; this only additionally picks up *other* families a
    # concurrent process committed in the meantime that this run doesn't
    # touch.
    payload = merge_and_write_results_locked(out_dir, payload, enabled=enabled)
    status = str(payload.get("status") or status)
    print(
        f"Wrote {results_path} status={status} methods={len(payload.get('methods') or [])}",
        flush=True,
    )
    if errors:
        print("Pipeline errors:", flush=True)
        for err in errors:
            print(f"  - {err}", flush=True)

    # Opt-in (training.cleanup_checkpoints): free this cell's weight files once
    # the cell is fully done. Runs after the final locked results.json write,
    # because the final merge above may still rehydrate rows from checkpoints.
    from study.artifact_cleanup import (
        cleanup_finished_cell_checkpoints,
        should_cleanup_checkpoints,
    )

    if should_cleanup_checkpoints(
        cfg.training, enabled=enabled, errors=errors, method_rows=method_rows
    ):
        cleanup_finished_cell_checkpoints(cfg.output_dir, train_backends)

    if cli.get("skip_reports"):
        print(
            "Skipping plots/reports in study worker; run "
            "slurm/regenerate_reports.sh on CPU after results complete.",
            flush=True,
        )
        return payload

    # Same end-of-run bundle as gender: plots + REPORT.md + metrics.csv + TABLES.txt
    # Refresh global model×task suitability first so TABLES.txt can embed it.
    global_suit: Dict[str, Any] = {}
    try:
        from suitability import refresh_global_suitability

        global_suit = refresh_global_suitability()
        print(
            f"Global suitability: {global_suit['paths'].get('model_task_txt')} "
            f"({global_suit.get('n_cells', 0)} model×task cells)",
            flush=True,
        )
    except Exception as exc:
        track_error(exc, context="global suitability refresh")
        if fail_fast:
            raise

    plot_paths: Dict[str, str] = {}
    try:
        from plot_study import write_all_study_plots

        plot_paths = write_all_study_plots(payload, out_dir)
    except Exception as exc:
        track_error(exc, context="plot step")
        if fail_fast:
            raise

    try:
        from report_study import build_console_tables_text, write_study_report

        report_paths = write_study_report(payload, out_dir, plot_paths=plot_paths)
        payload["artifacts"] = {
            "plots": plot_paths,
            **report_paths,
            "global_suitability": (global_suit or {}).get("paths"),
        }
        payload = merge_and_write_results_locked(out_dir, payload, enabled=enabled)
        print("Report:", report_paths.get("report"), flush=True)
        print("Metrics CSV:", report_paths.get("metrics_csv"), flush=True)
        print("Tables:", report_paths.get("tables"), flush=True)
        tables_text = build_console_tables_text(payload)
        print("\n" + tables_text, flush=True)
        panels = (payload.get("raw") or {}).get("causal", {}).get("poc_decoder_panels") or {}
        for mid, panel in panels.items():
            print(f"\n=== PoC decoder panel ({mid}) ===", flush=True)
            for row in panel.get("table") or []:
                print(
                    f"  {row.get('metric')}: before={row.get('before')} "
                    f"after={row.get('after')} delta={row.get('delta')}",
                    flush=True,
                )
    except Exception as exc:
        track_error(exc, context="report step")
        if fail_fast:
            raise
        payload["artifacts"] = {
            "plots": plot_paths,
            "global_suitability": (global_suit or {}).get("paths"),
        }
        payload = merge_and_write_results_locked(out_dir, payload, enabled=enabled)

    return payload
