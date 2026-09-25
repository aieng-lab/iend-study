"""SAE stage — full PoC encode ablations via gender ``run_sae`` / ``_sae_method_rows``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from results_schema import method_result
from study.config import StudyConfig
from study.method_ids import label_tokens_from_config
from study.tasks import TaskBundle, encode_target_classes, labeled_df_for_eval


from error_tracker import track_error
from study.method_ids import normalize_sae_method_id


# Suite YAML tags → PoC behavior. Core: [k1, kstar, all_k1];
# full adds opp_fire, joint, arad_out, jh_f1.
_SUITE_MODE_TAGS = frozenset({"joint", "pairwise", "pair_aware", "dense_probe"})
_K_TAG_RE = re.compile(r"^k(\d+)$")
_ALL_K_TAG_RE = re.compile(r"^all_k(\d+)$")

# Durable encode snapshot (survives a failed SAE rerun that would otherwise
# leave only a family-level error stub in results.json).
_SAE_SNAPSHOT_STAGE = "sae"
_SAE_SNAPSHOT_NAME = "encode_method_rows.json"


def _normalize_sae_suite_tags(suite_sae: Sequence[Any]) -> Set[str]:
    tags: Set[str] = set()
    for raw in suite_sae:
        s = str(raw).strip().lower()
        if not s:
            continue
        if s.startswith("sel_"):
            s = s[4:]
        if s in {"arad", "arad_out", "sel_arad_out"}:
            tags.add("arad_out")
        elif s in {"jh", "jh_f1", "sel_jh_f1"}:
            tags.add("jh_f1")
        elif s in {"opp_fire", "sel_opp_fire"}:
            tags.add("opp_fire")
        else:
            tags.add(s)
    return tags


def sae_suite_params(suite_sae: Optional[Sequence[Any]]) -> Dict[str, Any]:
    """Parse suite SAE tags into encode/causal knobs (no PoC globals)."""
    if not suite_sae:
        return {
            "tags": set(),
            "selection_modes": ("per_class",),
            "fixed_ks": (1,),
            "readout_ks": (1,),
            "opp_fire": False,
            "arad": False,
            "jh_f1": False,
            "clamp": False,
        }
    tags = _normalize_sae_suite_tags(suite_sae)

    modes = ["per_class"]  # required for k1 / kstar / all_k*
    for m in ("joint", "pairwise", "pair_aware", "dense_probe"):
        if m in tags:
            modes.append(m)

    ks: Set[int] = set()
    for t in tags:
        m = _K_TAG_RE.match(t)
        if m:
            ks.add(int(m.group(1)))
        m = _ALL_K_TAG_RE.match(t)
        if m:
            ks.add(int(m.group(1)))
    if not ks:
        ks.add(1)
    fixed = tuple(sorted(ks))
    if "kstar" in tags:
        readout_ks = tuple(sorted(set(fixed) | {1, 2, 4, 8, 16, 32, 64, 128}))
    else:
        readout_ks = fixed

    return {
        "tags": tags,
        "selection_modes": tuple(modes),
        "fixed_ks": fixed,
        "readout_ks": readout_ks,
        "opp_fire": "opp_fire" in tags,
        "arad": "arad_out" in tags,
        "jh_f1": "jh_f1" in tags,
        "clamp": "clamp" in tags,
    }


def apply_sae_suite(poc, suite_sae: Optional[Sequence[Any]]) -> Set[str]:
    """Configure PoC SAE globals from suite method tags. Returns normalized tags."""
    params = sae_suite_params(suite_sae)
    tags = params["tags"]
    if not tags:
        return set()

    poc.SAE_SELECTION_MODES = params["selection_modes"]
    poc.SAE_FIXED_KS = params["fixed_ks"]
    poc.SAE_READOUT_KS = params["readout_ks"]
    poc.SAE_OPP_FIRE_ENABLED = params["opp_fire"]
    poc.SAE_ARAD_ENABLED = params["arad"]
    poc.SAE_JH_F1_ENABLED = params["jh_f1"]

    print(
        f"sae: suite tags={sorted(tags)} modes={list(poc.SAE_SELECTION_MODES)} "
        f"fixed_ks={list(poc.SAE_FIXED_KS)} "
        f"opp_fire={poc.SAE_OPP_FIRE_ENABLED} arad={poc.SAE_ARAD_ENABLED} "
        f"jh_f1={poc.SAE_JH_F1_ENABLED}",
        flush=True,
    )
    return tags


def filter_sae_method_rows(
    rows: List[Dict[str, Any]], tags: Set[str]
) -> List[Dict[str, Any]]:
    """Keep only method rows allowed by suite tags (safety net after PoC emit)."""
    if not tags:
        return rows
    keep: List[Dict[str, Any]] = []
    for row in rows:
        mid = str(row.get("method") or "")
        if _sae_method_allowed(mid, tags):
            keep.append(row)
    dropped = len(rows) - len(keep)
    if dropped:
        print(f"sae: suite filtered {dropped} method rows (kept {len(keep)})", flush=True)
    return keep


def _sae_method_allowed(method_id: str, tags: Set[str]) -> bool:
    mid = normalize_sae_method_id(str(method_id))
    if not mid.startswith("sae"):
        return True
    backend = mid.split(":", 1)[0]
    # sae:joint / sae:joint:M / sae_pre:joint / sae_pre:joint:M
    if (
        mid == f"{backend}:joint"
        or mid.startswith(f"{backend}:joint:")
    ):
        return "joint" in tags
    parts = mid.split(":")
    suffix = parts[-1] if len(parts) >= 2 else ""
    for mode in ("pairwise", "pair_aware", "dense_probe", "per_class_diff"):
        if suffix == mode or mid == f"{backend}:{mode}":
            return mode in tags
    # sae:M:kstar / sae_pre:M:k1 / sae:M:all_k1 / sae:M:sel_opp_fire / sae:M:L11
    if len(parts) < 3:
        return False
    if suffix == "kstar":
        return "kstar" in tags
    if _K_TAG_RE.match(suffix):
        return suffix in tags
    if _ALL_K_TAG_RE.match(suffix):
        return suffix in tags
    if suffix in {"sel_opp_fire", "opp_fire"}:
        return "opp_fire" in tags
    if suffix in {"sel_arad_out", "arad_out"}:
        return "arad_out" in tags
    if suffix in {"sel_jh_f1", "jh_f1"}:
        return "jh_f1" in tags
    # Per-layer diagnostics always keep encoder L* rows (causal L*_k1 inherits).
    if re.fullmatch(r"L\d+", suffix):
        return True
    if re.fullmatch(r"L\d+_k\d+", suffix):
        return "layers" in tags
    if "tok_" in suffix:
        return False
    return False


def _sae_rows_are_substantive(rows: Sequence[Dict[str, Any]]) -> bool:
    """True if rows include real class/ablation SAE encode results (not a stub)."""
    from results_schema import _has_encoder_metrics

    for row in rows:
        mid = str(row.get("method") or "")
        if not mid.startswith("sae"):
            continue
        if mid == "sae" and row.get("status") == "error":
            continue
        if _has_encoder_metrics(row.get("metrics") or {}):
            return True
    return False


def _repair_canonical_k1_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Populate canonical ``k1`` rows from already-computed layerwise rows.

    The SAE encoder computes ``L*`` rows and the canonical k=1 recipe from the
    same features.  Older writers emitted the former's validation readout but
    not the latter's, leaving downstream aggregation unable to lock the
    canonical row.  This is a metadata-only repair: no feature extraction or
    causal evaluation is performed here.
    """
    from suitability import detection_score

    by_id = {str(row.get("method") or ""): row for row in rows}
    for mid, row in by_id.items():
        parts = mid.split(":")
        if len(parts) != 3 or parts[0] not in {"sae", "sae_pre"} or parts[2] != "k1":
            continue
        metrics = row.setdefault("metrics", {})
        if isinstance(metrics.get("val_readout"), dict):
            continue
        prefix = f"{parts[0]}:{parts[1]}:"
        candidates = []
        for candidate_id, candidate in by_id.items():
            if not candidate_id.startswith(prefix):
                continue
            tail = candidate_id[len(prefix):]
            if not (re.fullmatch(r"L\d+", tail) or re.fullmatch(r"L\d+_k1", tail)):
                continue
            candidate_metrics = candidate.get("metrics") or {}
            val = candidate_metrics.get("val_readout")
            score = detection_score(val) if isinstance(val, dict) else None
            if isinstance(score, (int, float)):
                candidates.append((float(score), candidate_id, candidate_metrics))
        if not candidates:
            continue
        _, selected_id, selected_metrics = max(candidates, key=lambda item: item[0])
        preserved = {
            key: value
            for key, value in metrics.items()
            if key.startswith("causal_") and value is not None
        }
        metrics.clear()
        metrics.update(selected_metrics)
        metrics.update(preserved)
        metrics["layer_selection"] = "validation_detection"
        metrics["selected_layer"] = selected_id.rsplit(":", 1)[-1]
    return rows


def _save_sae_encode_snapshot(
    cfg: StudyConfig,
    rows: Sequence[Dict[str, Any]],
    *,
    raw_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Persist encode method rows under artifacts/sae/ for failure recovery."""
    if not _sae_rows_are_substantive(rows):
        return None
    try:
        from experiment_cache import ExperimentCache

        # Drop non-JSON blobs; keep metrics/status/method for later restore.
        slim_rows: List[Dict[str, Any]] = []
        for row in rows:
            slim = {
                "method": row.get("method"),
                "model": row.get("model") or cfg.model_key,
                "task": row.get("task") or cfg.task_id,
                "status": row.get("status"),
                "metrics": dict(row.get("metrics") or {}),
                "artifacts": dict(row.get("artifacts") or {}),
            }
            if row.get("error") is not None:
                slim["error"] = row.get("error")
            slim_rows.append(slim)
        cache = ExperimentCache(cfg.output_dir)
        path = cache.write_json(
            _SAE_SNAPSHOT_STAGE,
            _SAE_SNAPSHOT_NAME,
            {
                "config_hash": cfg.config_hash(),
                "model": cfg.model_key,
                "task": cfg.task_id,
                "n_methods": len(slim_rows),
                "methods": slim_rows,
                "meta": {
                    k: raw_meta.get(k)
                    for k in (
                        "k_selection",
                        "layer_selection",
                        "selected_layer_global",
                        "selected_layer_by_class",
                        "sae_readout_ks",
                        "sae_top_k",
                    )
                    if isinstance(raw_meta, dict) and raw_meta.get(k) is not None
                },
            },
        )
        cache.mark_done(
            _SAE_SNAPSHOT_STAGE,
            config_hash=cfg.config_hash(),
            paths={"encode_method_rows": str(path)},
            extras={"n_methods": len(slim_rows)},
        )
        print(f"sae: wrote encode snapshot ({len(slim_rows)} methods) → {path}", flush=True)
        return path
    except Exception as exc:
        track_error(exc, context="sae encode snapshot save")
        print(f"sae: warning: could not write encode snapshot: {exc}", flush=True)
        return None


def _load_sae_encode_snapshot(cfg: StudyConfig) -> List[Dict[str, Any]]:
    """Load last successful SAE encode rows from artifacts/sae/."""
    try:
        from experiment_cache import ExperimentCache

        cache = ExperimentCache(cfg.output_dir)
        payload = cache.read_json(_SAE_SNAPSHOT_STAGE, _SAE_SNAPSHOT_NAME)
    except Exception as exc:
        track_error(exc, context="sae encode snapshot load")
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("methods") or []
    if not isinstance(rows, list):
        return []
    out = [r for r in rows if isinstance(r, dict) and r.get("method")]
    if _sae_rows_are_substantive(out):
        return out
    return []


def _error_stub(cfg: StudyConfig, message: str) -> Dict[str, Any]:
    return method_result(
        method="sae",
        model=cfg.model_key,
        task=cfg.task_id,
        status="error",
        error=message,
    )


def _configure_poc_globals(
    poc,
    cfg: StudyConfig,
    bundle: TaskBundle,
    *,
    classes: Optional[Sequence[str]] = None,
) -> None:
    """Point the gender PoC module globals at this study run."""
    class_list = [str(c) for c in (classes if classes is not None else bundle.classes)]
    try:
        poc.apply_study_model(cfg.model_key)
    except Exception as exc:
        track_error(exc, context="sae apply_study_model")
    poc.TARGET_CLASSES = class_list
    poc.OUTPUT_DIR = cfg.output_dir
    poc.STUDY_NEUTRAL_DF = bundle.neutrals
    poc.STUDY_EXCLUDED_WORDS = list(bundle.excluded_words or [])
    t = cfg.training or {}
    # SAE latent/selection cache only (evaluate_* paths stay use_cache=False).
    # See CLAUDE.md's SAE-cache note: this writes sae_cache/L{n}_*/*.npz per
    # layer for skip-on-recompute -- real disk cost with zero benefit on a
    # cluster where each SAE task runs once and is never resumed with the
    # same fingerprint. training.sae_disk_cache (default true) lets a
    # storage-constrained deployment turn writing/reading it off entirely.
    poc.USE_CACHE = bool(t.get("sae_disk_cache", True))
    if t.get("activation_site") is not None:
        from activation_protocol import normalize_activation_site

        poc.ACTIVATION_SITE = normalize_activation_site(t.get("activation_site"))
    if t.get("sae_select_sites") is not None:
        from sae_eval import normalize_sae_select_sites

        poc.SAE_SELECT_SITES = normalize_sae_select_sites(t.get("sae_select_sites"))
    if t.get("encoder_eval_max_size") is not None:
        poc.ENCODER_EVAL_MAX = int(t["encoder_eval_max_size"])
    if t.get("encoder_bootstrap_auc") is not None:
        poc.SAE_BOOTSTRAP_AUC = int(t["encoder_bootstrap_auc"])
    if t.get("train_batch_size") is not None:
        poc.TRAIN_BATCH_SIZE = int(t["train_batch_size"])
    if cfg.raw.get("_smoke"):
        try:
            poc.apply_smoke()
            poc.TARGET_CLASSES = class_list
            poc.OUTPUT_DIR = cfg.output_dir
            poc.STUDY_NEUTRAL_DF = bundle.neutrals
            poc.STUDY_EXCLUDED_WORDS = list(bundle.excluded_words or [])
            poc.USE_CACHE = bool(t.get("sae_disk_cache", True))
        except Exception as exc:
            track_error(exc, context="sae apply_smoke")


def _retag_rows(
    rows: List[Dict[str, Any]], *, model_key: str, task_id: str
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        r["model"] = model_key
        r["task"] = task_id
        out.append(r)
    return out


# ``pre_prediction`` is one token before the filled prediction span (see
# caa_eval.py). On a one-pole-only task, the rival class comes from
# ``expand_one_pole_cf_rows`` duplicating the factual row and relabeling it —
# same left context up to that point, so under causal attention the
# ``pre_prediction`` activation is identical for the factual and rival label.
# Any roc_auc_other / class_exclusivity computed there is tautological
# (empirically: auc_o=0.5, excl=0.0 — chance, not signal). The ``prediction``
# site is unaffected: rival scoring fills the actual rival token first, so the
# activations genuinely differ there.
_TAUTOLOGICAL_SAE_PRE_RIVAL_FIELDS: Tuple[str, ...] = (
    "roc_auc_other",
    "min_pairwise_auroc",
    "class_exclusivity",
    "class_exclusivity_mag",
)


def _strip_tautological_pre_prediction_rival_metrics(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Null rival-class metrics on ``sae_pre:*`` rows (see comment above)."""
    out = []
    for row in rows:
        mid = str(row.get("method") or "")
        if mid == "sae_pre" or mid.startswith("sae_pre:"):
            row = dict(row)
            metrics = dict(row.get("metrics") or {})
            for key in _TAUTOLOGICAL_SAE_PRE_RIVAL_FIELDS:
                if metrics.get(key) is not None:
                    metrics[key] = None
            row["metrics"] = metrics
        out.append(row)
    return out


def _hf_trainer_stub(cfg: StudyConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from study.config import resolve_torch_dtype

    name = cfg.hf_model
    print(f"sae: loading HF backbone {name!r} (no trainer checkpoint)", flush=True)
    model_kwargs: Dict[str, Any] = {}
    dtype_name = (cfg.training or {}).get("torch_dtype")
    if dtype_name is not None:
        model_kwargs["torch_dtype"] = resolve_torch_dtype(dtype_name)
    device_map = (cfg.training or {}).get("base_model_device_map")
    if device_map not in (None, False, "false"):
        model_kwargs["device_map"] = device_map
    else:
        # Without a device_map, ``from_pretrained`` otherwise holds the full
        # state dict AND the freshly-initialized module in CPU RAM at once
        # (~2x the weight size -- ~32GB transient for an 8B bf16 model), which
        # is a large chunk of the host-RAM budget before a single activation is
        # captured. Streaming the shards into a meta-initialized model is
        # numerically identical and halves that peak.
        model_kwargs.setdefault("low_cpu_mem_usage", True)
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    # ``from_pretrained`` defaults to CPU when no device_map is supplied. The
    # direct-backbone SAE/CAA path must mirror the trainer path and explicitly
    # place the model on the available accelerator; otherwise every activation
    # batch becomes a full CPU forward (catastrophic for 8B+ models).
    if device_map in (None, False, "false") and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        print(f"sae: set tokenizer.pad_token = eos_token ({tok.eos_token!r})", flush=True)

    class _M:
        base_model = model

    class _T:
        # Use a different RHS name: ``tokenizer = tokenizer`` is a NameError in
        # class bodies (assignment makes ``tokenizer`` local to the class scope).
        tokenizer = tok

        def get_model(self):
            return _M()

    return _T()


def run_sae_stage(
    cfg: StudyConfig,
    bundle: TaskBundle,
    train_raw_none: Dict[str, Dict[str, Any]],
    *,
    layers: Optional[Sequence[int]] = None,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Full SAE encode set matching ``study.sae_engine.run_sae``."""
    errors: List[str] = []

    any_raw = next(iter(train_raw_none.values()), None) if train_raw_none else None
    trainer = (any_raw or {}).get("trainer") if any_raw else None
    if trainer is None:
        try:
            trainer = _hf_trainer_stub(cfg)
        except Exception as exc:
            if fail_fast:
                raise
            return {
                "methods": [_error_stub(cfg, f"backbone load failed: {exc}")],
                "raw": {"error": f"backbone load failed: {exc}"},
                "errors": [f"sae: backbone load failed: {exc}"],
            }

    try:
        # The engine is configured through module globals (see its docstring).
        import study.sae_engine as poc

        # Feature selection on factual ``label_class`` rows only. Rival scoring
        # (auc_o / excl) uses CF-expanded rows with filled prediction activations
        # (same geometry as ACTIEND/CAA).
        eval_df = labeled_df_for_eval(bundle, expand_one_pole=False)
        classes = encode_target_classes(bundle, eval_df)
        claim_set = set(classes)
        expanded = labeled_df_for_eval(bundle, expand_one_pole=True)
        one_pole_cf_rival = "one_pole_cf_row" in expanded.columns
        if one_pole_cf_rival:
            rival_df = expanded[expanded["one_pole_cf_row"].astype(bool)].copy()
        else:
            rival_df = expanded[
                ~expanded["label_class"].astype(str).isin(claim_set)
            ].copy()
        if rival_df.empty:
            rival_df = None
        else:
            print(
                f"sae: rival readout rows for {sorted(rival_df['label_class'].astype(str).unique())} "
                f"(n={len(rival_df)}; filled prediction site)",
                flush=True,
            )
        if classes != [str(c) for c in bundle.classes]:
            print(
                f"sae: encode classes {classes} (claim poles on label_class; "
                f"rivals scored separately on CF fills)",
                flush=True,
            )
        _configure_poc_globals(poc, cfg, bundle, classes=classes)
        suite_sae = (cfg.suite.get("methods") or {}).get("sae") or []
        sae_tags = apply_sae_suite(poc, suite_sae)
        label_tokens = label_tokens_from_config(
            bundle.classes, cfg.raw.get("causal") or {}
        )
        if layers is not None:
            poc.SAE_LAYERS = tuple(int(x) for x in layers)
            poc.SAE_LAYER = int(poc.SAE_LAYERS[-1])
        elif cfg.raw.get("_smoke"):
            listed = (cfg.model or {}).get("sae_layers") or (cfg.model or {}).get(
                "layers"
            )
            if listed:
                poc.SAE_LAYERS = tuple(int(x) for x in listed)
                poc.SAE_LAYER = int(poc.SAE_LAYERS[-1])

        print(
            f"sae: PoC run_sae layers={list(poc.SAE_LAYERS)} "
            f"release={poc.SAE_RELEASE} classes={list(poc.TARGET_CLASSES)}",
            flush=True,
        )
        raw = poc.run_sae(
            trainer,
            eval_df,
            rival_eval_df=rival_df,
            rival_skip_sites={"pre_prediction"} if one_pole_cf_rival else frozenset(),
            label_tokens=label_tokens,
        )
        rows = _retag_rows(
            poc._sae_method_rows(raw),
            model_key=cfg.model_key,
            task_id=cfg.task_id,
        )
        if one_pole_cf_rival:
            # Defense in depth: rival_skip_sites already prevents the wasted
            # extraction above; this guards any other path that might still
            # populate these fields (e.g. a non-per-site fallback).
            rows = _strip_tautological_pre_prediction_rival_metrics(rows)
        rows = filter_sae_method_rows(rows, sae_tags)
        rows = _repair_canonical_k1_rows(rows)
        if raw.get("_model") is None:
            try:
                raw["_model"] = trainer.get_model().base_model
                raw["_tokenizer"] = trainer.tokenizer
            except Exception:
                pass
        _save_sae_encode_snapshot(cfg, rows, raw_meta=raw)
        return {"methods": rows, "raw": raw, "errors": errors}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        if fail_fast:
            raise
        track_error(
            exc,
            context="sae stage",
            model=cfg.model_key,
            task=cfg.task_id,
            print_msg=False,
        )
        errors.append(f"sae: {exc}")
        restored = _load_sae_encode_snapshot(cfg)
        if restored:
            print(
                f"sae: stage failed ({exc}); restoring {len(restored)} methods "
                f"from artifacts/sae/{_SAE_SNAPSHOT_NAME}",
                flush=True,
            )
            errors.append(
                f"sae: restored encode snapshot after failure ({len(restored)} methods)"
            )
            return {
                "methods": restored,
                "raw": {
                    "error": str(exc),
                    "restored_from_snapshot": True,
                    "n_restored": len(restored),
                },
                "errors": errors,
            }
        return {
            "methods": [_error_stub(cfg, str(exc))],
            "raw": {"error": str(exc)},
            "errors": errors,
        }
