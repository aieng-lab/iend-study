"""CAA mean-diff baseline stage — full PoC encode set (act / all_act / L*_act)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from caa_eval import CAA_ACT_POLICIES, run_caa_study
from results_schema import fair_metric_fields, method_result
from study.config import StudyConfig
from study.method_ids import label_tokens_from_config
from study.tasks import TaskBundle, encode_target_classes, labeled_df_for_eval
from study.training_profiles import SMOKE_LAYER_CAP, resolve_study_layers


def resolve_caa_layers(
    cfg: StudyConfig, layers: Optional[Sequence[int]] = None
) -> List[int]:
    """CAA layer list. Smoke matches SAE: first ``SMOKE_LAYER_CAP`` layers, not all 12."""
    smoke = bool(cfg.raw.get("_smoke"))
    model_cfg = cfg.model or {}
    if layers is not None:
        out = [int(x) for x in layers]
    else:
        listed = model_cfg.get("sae_layers") or model_cfg.get("layers")
        if listed is not None:
            out = [int(x) for x in listed]
        elif smoke:
            out = resolve_study_layers(model_cfg, smoke=True)
        else:
            try:
                from study_models import set_active_model

                out = [int(x) for x in set_active_model(cfg.model_key).layers]
            except Exception:
                n = int(model_cfg.get("n_layers") or 12)
                out = [max(0, n - 1)]
    if smoke:
        out = out[:SMOKE_LAYER_CAP] or list(range(SMOKE_LAYER_CAP))
    return out


def prior_caa_val_readouts(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """``{method_id: val_readout}`` from earlier CAA method rows (error rows skipped).

    The validation readout is deterministic given the stored CAA vectors, so it can
    stand in for rescoring the validation split (see ``encode_caa_scores``).
    """
    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("status") or "") == "error" or row.get("error"):
            continue
        metrics = row.get("metrics") or {}
        vr = metrics.get("val_readout")
        if not isinstance(vr, Mapping):
            vr = ((row.get("extras") or {}).get("readout_metrics") or {}).get("val_readout")
        if isinstance(vr, Mapping) and row.get("method"):
            out[str(row["method"])] = vr
    return out


def run_caa_stage(
    cfg: StudyConfig,
    bundle: TaskBundle,
    train_raw_none: Dict[str, Dict[str, Any]],
    *,
    layers: Optional[Sequence[int]] = None,
    policies: Optional[Sequence[str]] = None,
    fail_fast: bool = False,
    previous_caa: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    any_raw = next(iter(train_raw_none.values()), None)
    trainer = (any_raw or {}).get("trainer") if any_raw else None
    if trainer is None:
        # Fall back to HF backbone (same as SAE) so CAA encode works without train.
        try:
            from study.stages.sae import _hf_trainer_stub

            trainer = _hf_trainer_stub(cfg)
        except Exception as exc:
            if fail_fast:
                raise
            return {
                "methods": [],
                "raw": {"error": f"no trainer for CAA: {exc}"},
                "errors": [f"caa: no trainer: {exc}"],
            }

    model = trainer.get_model().base_model
    tokenizer = trainer.tokenizer
    # Expand one-pole CF rows so prediction-fill sees both tokens.
    eval_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    classes = encode_target_classes(bundle, eval_df)
    if len(classes) < 2:
        return {
            "methods": [],
            "raw": {"error": f"CAA needs ≥2 label_class values; got {classes}"},
            "errors": [f"caa: need ≥2 classes after one-pole expand; got {classes}"],
        }
    if classes != [str(c) for c in bundle.classes]:
        print(
            f"caa: TARGET_CLASSES {list(bundle.classes)} → {classes} "
            f"(after one-pole CF expand)",
            flush=True,
        )
    layers = resolve_caa_layers(cfg, layers)
    model_cfg = cfg.model or {}
    template = model_cfg.get("hf_resid_template") or "transformer.h.{layer}"
    suite_caa = (cfg.suite.get("methods") or {}).get("caa") or []
    if policies is None:
        policies = list(CAA_ACT_POLICIES)
        if suite_caa:
            suite_pols = []
            for p in suite_caa:
                s = str(p)
                suite_pols.append(s[4:] if s.startswith("act_") else s)
            policies = [p for p in policies if p in suite_pols]
            if not policies:
                policies = list(CAA_ACT_POLICIES)

    label_tokens = label_tokens_from_config(classes, cfg.raw.get("causal") or {})
    ablations = cfg.raw.get("ablations") or {}
    include_pair = bool(ablations.get("pair", True))
    include_one_pole = bool(ablations.get("one_pole", True))
    # Large models stream the CAA mean-diff into O(d) accumulators instead of
    # holding the full {layer: (N, d)} host matrix -- the llama-3.1-8b CAA
    # OUT_OF_MEMORY. The result is mathematically identical (CAA is a mean);
    # small models keep the proven materialized path byte-for-byte. Override
    # with training.caa_stream_fit.
    n_layers = int(model_cfg.get("n_layers") or len(layers) or 12)
    stream_cfg = (cfg.training or {}).get("caa_stream_fit")
    stream_fit = bool(stream_cfg) if stream_cfg is not None else n_layers >= 24
    print(
        f"caa: layers={list(layers)} policies={list(policies)}"
        f" pair={include_pair} one_pole={include_one_pole} stream_fit={stream_fit}"
        + (" (smoke cap)" if cfg.raw.get("_smoke") else ""),
        flush=True,
    )
    # Test-only migration: reuse the stored vectors and validation readouts of an
    # earlier, error-free run that already covers every requested pole regime.
    # ``previous_caa`` is only supplied under --skip-existing; run_caa_study
    # additionally falls back to a full fit if the layers/policies differ.
    prior_vectors = None
    prior_val = None
    eval_cap = int((cfg.training or {}).get("encoder_eval_max_size") or 200)
    if previous_caa:
        prev_raw = previous_caa.get("raw") or {}
        want_regimes = {
            name
            for name, on in (("pair", include_pair), ("one_pole", include_one_pole))
            if on
        }
        if (
            prev_raw.get("vectors")
            and not prev_raw.get("error")
            and want_regimes <= set(prev_raw.get("pole_regimes") or [])
        ):
            prior_vectors = prev_raw["vectors"]
            prior_val = prior_caa_val_readouts(previous_caa.get("rows") or []) or None
    method_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        # ``run_caa_study`` records backend-attributed fit and encode timers.
        # Do not wrap it in an additional feature-select timer: that would
        # double-count the CAA sweep in the method ledger.
        raw = run_caa_study(
            model,
            tokenizer,
            eval_df,
            bundle.neutrals,
            target_classes=classes,
            layers=list(layers),
            policies=list(policies),
            hf_resid_template=str(template),
            max_eval=eval_cap,
            max_neutral=eval_cap,
            batch_size=int((cfg.training or {}).get("train_batch_size") or 8),
            n_bootstrap=int((cfg.training or {}).get("encoder_bootstrap_auc") or 100),
            label_tokens=label_tokens,
            excluded_words=list(bundle.excluded_words or []),
            output_dir=cfg.output_dir,
            include_pair=include_pair,
            include_one_pole=include_one_pole,
            stream_fit=stream_fit,
            prior_vectors_payload=prior_vectors,
            prior_val_readouts=prior_val,
        )
        raw["_model"] = model
        raw["_tokenizer"] = tokenizer
        for mid, rd in (raw.get("method_metrics") or {}).items():
            method_rows.append(
                method_result(
                    method=str(mid),
                    model=cfg.model_key,
                    task=cfg.task_id,
                    status="error" if rd.get("error") else "ok",
                    error=rd.get("error"),
                    metrics={
                        **fair_metric_fields(rd),
                        "backend": "caa",
                        "readout_kind": "class_vs_neutral",
                        "target_class": rd.get("target_class"),
                        "act_policy": rd.get("act_policy"),
                        "component_part": rd.get("component_part"),
                        "score": rd.get("score") or "cosine",
                        "ablation": rd.get("ablation"),
                        "pair": rd.get("pair"),
                        "vector_key": rd.get("vector_key"),
                        "mean_target": rd.get("mean_target"),
                        "mean_neutral": rd.get("mean_neutral"),
                        "mean_other": rd.get("mean_other"),
                        "n_neutral": rd.get("n_neutral"),
                        "n_neutral_texts": rd.get("n_neutral_texts"),
                        "neutral_unit": rd.get("neutral_unit"),
                    },
                    extras={"readout_metrics": rd},
                )
            )
    except Exception as exc:
        if fail_fast:
            raise
        errors.append(f"caa: {exc}")
        raw = {"error": str(exc)}
        method_rows.append(
            method_result(
                method="caa",
                model=cfg.model_key,
                task=cfg.task_id,
                status="error",
                error=str(exc),
            )
        )
    return {"methods": method_rows, "raw": raw, "errors": errors}
