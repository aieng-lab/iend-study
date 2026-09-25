"""Unified study runner: one (model, task) job via the deep pipeline."""

from __future__ import annotations

import copy
import json
import os
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from study.clm_score_patch import apply_clm_surface_variant_sum_patch
from study.config import CONFIGS, _load_yaml, load_study_config
from study.deep_pipeline import run_deep_pipeline, write_error_results_locked
from study.registry import build_task

apply_clm_surface_variant_sum_patch()


def _deep_merge_dicts(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


from study.json_util import json_ready as _json_ready


def _is_pair_method_id(method_id: str) -> bool:
    parts = str(method_id).split(":")
    return len(parts) >= 2 and "-" in parts[1]


def _payload_has_pair_caa(payload: Mapping[str, Any]) -> bool:
    for row in payload.get("methods") or []:
        mid = str(row.get("method") or "")
        if mid.startswith("caa:") and _is_pair_method_id(mid):
            return True
    raw_caa = ((payload.get("raw") or {}).get("caa") or {}).get("method_metrics") or {}
    return any(k.startswith("caa:") and _is_pair_method_id(k) for k in raw_caa)


def _payload_has_requested_method_families(
    payload: Mapping[str, Any], requested_methods: set[str]
) -> bool:
    """Whether every explicitly requested family has at least one successful row.

    ``results.json`` is shared by separate method-array cells.  A task can be
    ``status=ok`` because GRADIEND finished while a later CGA/AGIEND cell has
    never been run.  Treating that task-level status as a complete cache hit
    makes the later cell a silent no-op.  This deliberately tests *families*,
    not every ablation id: per-artifact and per-causal-id resume logic owns
    the finer-grained decision after the runner re-enters.
    """
    successful: set[str] = set()
    for row in payload.get("methods") or []:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            continue
        family = str(row.get("method") or "").split(":", 1)[0]
        if family:
            successful.add(family)
    if not requested_methods.issubset(successful):
        return False

    # A causal-only row is not a complete encoder-family cache hit.  This is
    # especially important for CAGA/CGA, where a causal refresh can leave
    # ``family:class`` rows present while the encoder readout was never
    # persisted (or was overwritten).  Re-enter only the incomplete method
    # array cell; per-stage resume then keeps completed pairwise rows.
    cfg = payload.get("config") or {}
    claim = cfg.get("claim_classes") or cfg.get("target_classes") or []
    ablations = cfg.get("ablations") or {}
    rows = [
        row for row in (payload.get("methods") or [])
        if isinstance(row, Mapping) and row.get("status") == "ok"
    ]

    def has_encoder(row: Mapping[str, Any]) -> bool:
        metrics = row.get("metrics") or {}
        if any(
            metrics.get(field) is not None
            for field in (
                "roc_auc", "roc_auc_neutral", "balanced_accuracy",
                "neutral_specificity", "specificity", "class_exclusivity",
                "encoder_correlation", "class_separation", "roc_auc_other",
            )
        ):
            return True
        extras = row.get("extras") or {}
        return bool(
            isinstance(extras.get("per_class_readouts"), Mapping)
            and extras.get("per_class_readouts")
        )

    for family in requested_methods & {"cga", "caga", "agiend", "gradiend", "actiend"}:
        family_rows = [
            row for row in rows
            if str(row.get("method") or "").split(":", 1)[0] == family
        ]
        if not family_rows:
            return False
        if ablations.get("one_pole"):
            for cls in claim:
                prefix = f"{family}:{cls}"
                if not any(
                    str(row.get("method") or "").split("|", 1)[0] == prefix
                    and has_encoder(row)
                    for row in family_rows
                ):
                    return False
        if ablations.get("pair"):
            if not any(
                len(str(row.get("method") or "").split("|", 1)[0].split(":")) == 2
                and "-" in str(row.get("method") or "").split("|", 1)[0].split(":", 2)[1]
                and has_encoder(row)
                for row in family_rows
            ):
                return False
    return True


def _should_skip_existing(
    output_dir: Path,
    *,
    skip_existing: bool,
    refresh_causal: bool = False,
    refresh_encoder_eval: bool = False,
    config_hash: Optional[str] = None,
    expects_caa_pairs: bool = False,
    requested_methods: Optional[set[str]] = None,
    train_slice: bool = False,
) -> tuple[bool, bool]:
    """Whether a completed task is current for the requested configuration.

    Refresh modes still want ``--skip-existing`` for train artifact reload, but
    must re-enter ``status=ok`` results.json so their requested stage can run.

    A task-level ``status=ok`` is only reusable for the same resolved config.
    In particular, switching suites can add method/split ablations; silently
    skipping based on the old status would leave those rows absent forever.

    ``expects_caa_pairs`` closes the same gap for a *code*-level change that is
    invisible to ``config_hash``: the two-pole CAA regime was added by fixing the
    CAA-encode reuse bug, not by changing any config field, so an ``ok`` task
    trained before it would be skipped forever. When this run will compute CAA and
    the config requests pairs (``ablations.pair``) but the stored results have no
    ``caa:A-B:*`` rows, re-enter to fill them (the CAA-encode reuse fix then
    recomputes them; every other stage is still reused per-stage). Self-limiting:
    once the pairs exist the task skips again, and the guard only fires for runs
    that actually run CAA, so it cannot loop on a ``--methods gradiend`` run.

    Likewise, an explicit method-array cell must re-enter if its method family
    is absent (or has no successful row).  The normal default, where no method
    filter was supplied, retains the historical task-level skip behavior.

    ``train_slice`` (``--train-ablations``): a pair-only and a one-pole-only job
    share one ``results.json``. Whichever slice finishes first would mark the task
    ``ok`` and make a still-queued sibling skip itself, so a slice run never takes
    the whole-task skip; per-artifact skip-existing keeps re-entry cheap.
    """
    if not skip_existing or refresh_causal or refresh_encoder_eval:
        return False, False
    path = output_dir / "results.json"
    if not path.is_file():
        return False, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, False
    config_changed = bool(
        config_hash and str(payload.get("config_hash") or "") != str(config_hash)
    )
    if payload.get("status") != "ok":
        return False, config_changed
    if config_changed:
        return False, True
    if train_slice:
        return False, False
    if expects_caa_pairs and not _payload_has_pair_caa(payload):
        # Re-enter to fill the missing two-pole CAA; per-stage reuse keeps
        # train/SAE/ok-causal-ids, so only CAA encode + new pair causal recompute.
        return False, False
    if requested_methods and not _payload_has_requested_method_families(
        payload, requested_methods
    ):
        return False, False
    return True, False


def paired_pass_overrides(
    cfg: Any, cli: Dict[str, Any], payload: Dict[str, Any], *, smoke: bool = False
) -> Optional[Dict[str, Any]]:
    """CLI for the reuse-only second causal pass of a core task, or None.

    ``primary_only`` (core) keeps causal for the ONE validation-selected
    representation per method.  The layer-selection strip/paired plots compare the
    all-layer estimate with the best single layer, so a core run must also cover the
    other one.  This pass computes exactly those missing sweeps, from stored encodings
    and checkpoints (``--causal-only paired``): nothing is trained or encoded, and
    every stored sweep is kept.  Per-layer curves stay a full-suite feature.
    """
    causal_cfg = (cfg.raw.get("causal") or {})
    if not causal_cfg.get("paired_layer_pass") or smoke:
        return None
    if cli.get("causal_only") or cli.get("skip_causal") or cli.get("_paired_pass"):
        return None
    if str(payload.get("status")) not in ("ok", "partial"):
        return None
    from study.causal_policies import PAIRED_FAMILIES
    from study.config import parse_methods_arg

    suite_methods = (cfg.suite.get("methods") or {})
    requested = parse_methods_arg(cli.get("methods"))
    families = [
        f
        for f in PAIRED_FAMILIES
        if suite_methods.get(f) and (requested is None or f in requested)
    ]
    if not families:
        return None
    second = dict(cli)
    second.update(
        {
            "methods": families,
            "causal_only": ["paired"],
            "skip_existing": True,
            "refresh_causal": True,
            "_paired_pass": True,
        }
    )
    return second


def run_study(
    *,
    model: str,
    task: str,
    suite: Optional[str] = None,
    smoke: bool = False,
    scale: Optional[str] = None,
    skip_existing: bool = False,
    refresh_causal: bool = False,
    force_causal: bool = False,
    lr: Optional[float] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if task == "ioi":
        raise RuntimeError(
            "Task 'ioi' is retired; use 'ioi_mib'. Refusing before config/model loading "
            "so queued legacy jobs cannot consume GPU time."
        )
    cli = dict(cli_overrides or {})
    # Runner-only; do not merge into StudyConfig (would change config_hash).
    force_causal = bool(force_causal or cli.pop("force_causal", False))
    refresh_causal = bool(refresh_causal or cli.pop("refresh_causal", False) or force_causal)
    refresh_encoder_eval = bool(cli.pop("refresh_encoder_eval", False))
    if refresh_causal:
        # Causal refresh implies train reload without whole-run skip.
        cli["skip_existing"] = True
        cli["refresh_causal"] = True
        if force_causal:
            cli["force_causal"] = True
        skip_existing = True
    if refresh_encoder_eval:
        # Detection refresh reloads fitted artifacts but must not task-level skip.
        cli["skip_existing"] = True
        cli["refresh_encoder_eval"] = True
        skip_existing = True
    if lr is not None:
        cli.setdefault("training", {})["learning_rate"] = float(lr)
    if scale is not None:
        profiles = _load_yaml(CONFIGS / "defaults.yaml").get("scale_profiles", {})
        profile = profiles.get(scale)
        if profile is None:
            raise ValueError(f"Unknown --scale {scale!r}; available: {list(profiles)}")
        cli = _deep_merge_dicts(cli, profile)

    cfg = load_study_config(model=model, task=task, suite=suite, cli_overrides=cli)
    if lr is not None:
        cfg.raw.setdefault("training", {})["learning_rate"] = cfg.learning_rate(lr)

    from study_models import set_active_model
    set_active_model(cfg.model_key)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # A run computes two-pole CAA when the suite runs CAA, the current
    # ``--methods`` filter (if any) includes it, and the config requests pairs.
    from study.config import parse_methods_arg

    _requested_methods = parse_methods_arg(cli.get("methods"))
    _caa_will_run = bool((cfg.suite.get("methods") or {}).get("caa")) and (
        _requested_methods is None or "caa" in _requested_methods
    )
    expects_caa_pairs = bool((cfg.raw.get("ablations") or {}).get("pair")) and _caa_will_run

    should_skip, config_changed = _should_skip_existing(
        out_dir,
        skip_existing=skip_existing,
        refresh_causal=refresh_causal,
        refresh_encoder_eval=refresh_encoder_eval,
        config_hash=cfg.config_hash(),
        expects_caa_pairs=expects_caa_pairs,
        requested_methods=set(_requested_methods) if _requested_methods is not None else None,
        train_slice=bool(cli.get("train_ablations")),
    )
    if should_skip:
        print(f"Skipping existing ok run: {out_dir / 'results.json'}")
        return json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    if config_changed:
        # Keep completed artifacts, but do not trust an orphaned artifact to
        # stand in for a newly enabled ablation after a suite/config change.
        if cli.get("no_rerun_orphaned_artifacts"):
            print(
                "skip-existing: config changed, but --no-rerun-orphans is set: artifacts "
                "without a result row are reloaded, not retrained",
                flush=True,
            )
        else:
            cli["rerun_orphaned_artifacts"] = True
        print(
            "skip-existing: config changed "
            f"(requested {cfg.config_hash()}); re-entering to fill newly "
            "enabled work",
            flush=True,
        )
    if force_causal:
        print(
            f"force-causal: re-entering {out_dir} (reload train, re-run every selected causal sweep)",
            flush=True,
        )
    elif refresh_causal:
        print(
            f"refresh-causal: re-entering {out_dir} (reload train, re-run causal)",
            flush=True,
        )
    elif refresh_encoder_eval:
        print(
            f"refresh-encoder-eval: re-entering {out_dir} (reload train, recompute detection)",
            flush=True,
        )

    started = datetime.now(timezone.utc).isoformat()
    from cost_timer import reset_cost_records, set_run_compute_meta, snapshot_device

    reset_cost_records()
    training_cfg = cfg.training
    set_run_compute_meta(
        **snapshot_device(),
        hf_model=cfg.hf_model,
        model_key=cfg.model_key,
        task=cfg.task_id,
        suite=cfg.suite_id,
        max_steps=training_cfg.get("max_steps"),
        train_batch_size=training_cfg.get("train_batch_size"),
        learning_rate=cfg.learning_rate(),
        learning_rate_gradiend=cfg.learning_rate(backend="gradiend"),
        learning_rate_actiend=cfg.learning_rate(backend="actiend"),
        # Large-model CGA uses a pruned, CPU-resident accumulator; retain
        # these facts with every timer so appendix comparisons never mistake
        # an implementation-path difference for a method-memory difference.
        torch_dtype=training_cfg.get("torch_dtype"),
        cga_pre_prune=training_cfg.get("cga_pre_prune"),
        cga_accumulate_device=training_cfg.get("cga_accumulate_device"),
        run_started_at=started,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    try:
        bundle = build_task(task, cfg.raw, smoke=smoke)
        payload = run_deep_pipeline(cfg, bundle, smoke=smoke, cli_overrides=cli)
        _second = paired_pass_overrides(cfg, cli, payload, smoke=smoke)
        if _second is not None:
            print(
                "paired layer pass: computing the all-layer vs best-single-layer causal "
                f"sweeps still missing (families={_second['methods']})",
                flush=True,
            )
            payload = run_deep_pipeline(cfg, bundle, smoke=smoke, cli_overrides=_second)
        if "started_at" not in payload:
            payload["started_at"] = started
        return payload
    except Exception as exc:
        error_payload = {
            "schema_version": "1.10",
            "model": cfg.model_key,
            "task": task,
            "suite": cfg.suite_id,
            "config_hash": cfg.config_hash(),
            "status": "error",
            "methods": [],
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        # Method-isolated array cells share one results.json. Re-read and merge
        # under the same lock as normal stage checkpoints so a failing cell
        # cannot erase rows committed by a concurrent successful cell.
        write_error_results_locked(out_dir, _json_ready(error_payload))
        raise
