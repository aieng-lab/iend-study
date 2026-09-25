"""End-of-run analysis bundle: REPORT.md + metrics.csv + TABLES.txt."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from results_schema import (
    NONE_VS_TENSOR_METRICS,
    causal_table,
    comparison_table,
    enabled_method_families,
    encoder_ablation_methods,
    none_vs_tensor_pairs,
    none_vs_tensor_table,
    resolve_encoder_metrics,
    summary_table,
)
from suitability import suitability_table

CSV_COLUMNS = (
    "method",
    "status",
    "target_class",
    "readout_k",
    "sae_layer",
    "k_selection",
    "roc_auc",
    "roc_auc_neutral",
    "roc_auc_other",
    "roc_auc_boot_std",
    "balanced_accuracy",
    "cohens_d",
    "specificity",
    "neutral_specificity",
    "class_exclusivity",
    "youden_threshold",
    "youden_j",
    "target_tpr",
    "neutral_specificity_mag",
    "class_exclusivity_mag",
    "encoder_correlation",
    "class_separation",
    "gradiend_split",
    "component_part",
    "causal_selected_strength",
    "causal_signed_effect",
    "causal_delta_target",
    "causal_delta_other",
    "causal_delta_neutral",
    "causal_effectiveness",
    "causal_lms",
    "causal_lms_ok",
    "causal_gate_empty",
    "causal_grid_ceiling",
    "causal_grid_floor",
    "causal_null_effect",
    "causal_random_signed_effect",
    "causal_specificity_ratio",
    "causal_beats_random_control",
    "suitability",
    "suitability_E",
    "suitability_G",
    "suitability_scope",
    "suitability_reason",
    "suitability_gate_reason",
    "suitability_tau_c",
)


def _ensure_suitability(results: Dict[str, Any]) -> None:
    """Attach S = E×G if missing / refresh from stored knobs."""
    from suitability import attach_feature_suitability, DEFAULT_TAU_C, DEFAULT_E_OK

    raw = (results.get("raw") or {}).get("suitability") or {}
    attach_feature_suitability(
        results,
        tau_c=float(raw.get("tau_c", DEFAULT_TAU_C)),
        soft_causal=bool(raw.get("soft_causal", False)),
        e_ok=float(raw.get("e_ok", DEFAULT_E_OK)),
    )


def _suitability_md_rows(results: Dict[str, Any]) -> List[List[Any]]:
    from suitability import (
        _default_method_ids,
        resolve_causal_metrics,
        compute_feature_suitability,
    )

    methods = _default_method_ids(results, primary_only=True)
    by = {
        str(r.get("method")): (r.get("metrics") or {})
        for r in (results.get("methods") or [])
        if r.get("method")
    }
    rows = []
    for mid in methods:
        m, src, _ = resolve_encoder_metrics(results, mid)
        met = by.get(src) or by.get(mid) or m
        if met.get("suitability_E") is None and met.get("suitability_reason") != "no_encoding":
            cau, _ = resolve_causal_metrics(results, mid)
            met = {**met, **compute_feature_suitability(m, cau)}
        rows.append(
            [
                mid,
                met.get("suitability"),
                met.get("suitability_E"),
                met.get("suitability_G"),
                met.get("suitability_scope"),
                met.get("suitability_reason"),
                met.get("suitability_gate_reason"),
            ]
        )
    return rows


def _fmt(v: Any, *, precision: int = 4) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    if v is None:
        return "—"
    return str(v)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(lines)


def write_metrics_csv(results: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results.get("methods") or []:
            m = dict(row.get("metrics") or {})
            m["method"] = row.get("method")
            m["status"] = row.get("status")
            writer.writerow({k: m.get(k) for k in CSV_COLUMNS})
    return path


def _fair_md_rows(results: Dict[str, Any]) -> List[List[Any]]:
    """Markdown rows for the FULL encoder ablation set (not the primary claim subset)."""
    methods = encoder_ablation_methods(results)
    rows = []
    for mid in methods:
        m, src, inherited = resolve_encoder_metrics(results, mid)
        note = f"<-{src}" if inherited else ""
        rows.append(
            [
                mid,
                m.get("readout_k"),
                m.get("sae_layer"),
                m.get("roc_auc_neutral", m.get("roc_auc")),
                m.get("roc_auc_other", m.get("min_pairwise_auroc")),
                m.get("balanced_accuracy"),
                m.get("cohens_d"),
                m.get("neutral_specificity", m.get("specificity")),
                m.get("class_exclusivity"),
                m.get("youden_j"),
                m.get("target_tpr"),
                m.get("neutral_specificity_mag"),
                m.get("class_exclusivity_mag"),
                note,
            ]
        )
    return rows


def _none_vs_tensor_md_rows(results: Dict[str, Any]) -> List[List[Any]]:
    """Markdown rows: none, :tensors, and Δ for each backend×class pair."""
    rows: List[List[Any]] = []
    metric_cols = [c for c, _, _ in NONE_VS_TENSOR_METRICS if c != "J"]
    for p in none_vs_tensor_pairs(results):
        label = f"{p['backend']}:{p['class']}"
        for split_key, split_lab in (("none", "none"), ("tensors", ":tensors"), ("delta", "delta")):
            cells: List[Any] = [label, split_lab]
            for col in metric_cols:
                cells.append((p["metrics"].get(col) or {}).get(split_key))
            cells.append(p["winner"] if split_key == "tensors" else "")
            rows.append(cells)
    return rows


def _known_failures(results: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for row in results.get("methods") or []:
        mid = row.get("method") or "?"
        m = row.get("metrics") or {}
        if row.get("status") == "error" and row.get("error"):
            notes.append(f"- `{mid}`: {row['error']}")
        if m.get("causal_grid_ceiling"):
            notes.append(
                f"- `{mid}`: causal strength at **grid ceiling** "
                f"({m.get('causal_selected_strength')}) — "
                f"may need higher LR; do not conclude the method is ineffective"
            )
        if m.get("causal_grid_floor"):
            notes.append(
                f"- `{mid}`: causal strength at **grid floor** "
                f"({m.get('causal_selected_strength')})"
            )
        if m.get("causal_null_effect") and m.get("causal_grid_floor"):
            notes.append(
                f"- `{mid}`: **floor+null** — tiny strength and ~0 effect "
                f"(no useful causal effect on this grid)"
            )
        elif m.get("causal_null_effect"):
            notes.append(f"- `{mid}`: causal effect ~0 (null) at selected strength")
        if m.get("causal_gate_empty"):
            notes.append(
                f"- `{mid}`: **no strength** passes LMS×0.99 "
                f"(fallback str={m.get('causal_selected_strength')}, "
                f"lms_ok={m.get('causal_lms_ok')})"
            )
        if mid.startswith("gradiend:") and ":" in mid[len("gradiend:") :]:
            # gradiend:M style
            if m.get("causal_signed_effect") is None and m.get("causal_delta_mean") is None:
                if mid.count(":") == 1:
                    notes.append(f"- `{mid}`: missing causal metrics")
    loc = ((results.get("raw") or {}).get("localization") or {})
    if loc.get("error"):
        notes.append(f"- localization: {loc['error']}")
    return notes


def build_report_markdown(
    results: Dict[str, Any],
    *,
    plot_paths: Optional[Dict[str, str]] = None,
    metrics_csv: Optional[str] = None,
) -> str:
    _ensure_suitability(results)
    cfg = results.get("config") or {}
    enabled = enabled_method_families(results)
    exp_id = (
        results.get("experiment_id")
        or cfg.get("experiment_id")
        or (
            f"{results.get('model') or cfg.get('model_key')}/"
            f"{results.get('task') or cfg.get('task_id')}"
        )
    )
    hf_model = results.get("hf_model") or cfg.get("hf_model") or "—"
    created = (
        results.get("created_at")
        or results.get("started_at")
        or results.get("finished_at")
        or "—"
    )
    raw_sae = (results.get("raw") or {}).get("sae") or {}
    k_sel = (
        cfg.get("k_selection")
        or cfg.get("sae_k_selection")
        or raw_sae.get("k_selection")
        or "—"
    )
    layer_sel = cfg.get("layer_selection") or raw_sae.get("layer_selection") or "—"
    layer_opp = (
        cfg.get("layer_selection_opp_fire")
        or raw_sae.get("layer_selection_opp_fire")
        or "—"
    )
    sae_ks = cfg.get("sae_readout_ks")
    if sae_ks is None:
        sae_ks = raw_sae.get("sae_readout_ks")
    sae_top = cfg.get("sae_top_k")
    if sae_top is None:
        sae_top = raw_sae.get("sae_top_k")
    enc_id_bits = ["`actiend:{cls}:L*` / `tok_*`"]
    if "sae" in enabled:
        enc_id_bits = [
            "`sae:{cls}:k*` / `sel_opp_fire` / `all_k*`",
            "`actiend:{cls}:L*`",
        ] + (["`caa:{cls}:act_*` / `:all_act_*` / `:L*_act_*`"] if "caa" in enabled else [])
    elif "caa" in enabled:
        enc_id_bits.append("`caa:{cls}:act_*` / `:all_act_*` / `:L*_act_*`")
    enabled_note = (
        f"Enabled method families for this run: `{sorted(enabled)}` "
        "(disabled families are omitted from claim / fair / none-vs-tensor tables)."
    )
    sections: List[str] = [
        f"# Study report: {exp_id}",
        "",
        f"- **task**: {results.get('task') or cfg.get('task_id')}  ",
        f"- **model**: {results.get('model') or cfg.get('model_key')} (`{hf_model}`)  ",
        f"- **created**: {created}  ",
        f"- **enabled methods**: `{sorted(enabled)}`  ",
        (
            f"- **claim classes**: `{list(cfg.get('claim_classes') or cfg.get('one_pole_classes') or [])}` "
            f"(full vocabulary `{list(cfg.get('target_classes') or [])}`; "
            f"CF-only poles omitted from primary fair / none-vs-tensor)  "
            if (cfg.get("claim_classes") or cfg.get("one_pole_classes"))
            and list(cfg.get("claim_classes") or cfg.get("one_pole_classes") or [])
            != list(cfg.get("target_classes") or [])
            else ""
        ),
        f"- **k selection**: `{k_sel}`  ",
        f"- **layer selection**: `{layer_sel}` "
        f"(opp_fire: `{layer_opp}`)  ",
        f"- **SAE readout ks**: `{sae_ks}`  TOP_K=`{sae_top}`  ",
        "",
        "## Encoding (all ablations)",
        "",
        "Full encoder class-vs-neutral table for **every** ablation of interest — "
        "same method ids as causal where shared ("
        + ", ".join(enc_id_bits)
        + ", …). "
        "Causal-only policies (`tok_*`"
        + (", `k1_tok_prediction`" if "sae" in enabled else "")
        + ") "
        "inherit encoder metrics from the base feature id (`enc<-source`). "
        + (
            "CAA scores are cosine with mean-diff steering vectors "
            "(act policies: prediction / mean / last; neutrals scored per "
            "non-pad token for prediction/pre_prediction). "
            if "caa" in enabled
            else ""
        )
        + "The primary fair subset is only a claim summary, not a filter. "
        + enabled_note,
        "",
        _md_table(
            [
                "method",
                "k",
                "L",
                "auc_n",
                "auc_o",
                "bal",
                "d",
                "spec",
                "excl",
                "J",
                "tpr",
                "spec_mag",
                "excl_mag",
                "note",
            ],
            _fair_md_rows(results),
        ),
        "",
        "```",
        comparison_table(results).strip(),
        "```",
        "",
        "### Primary fair subset (claim only)",
        "",
        "```",
        comparison_table(results, primary_only=True).strip(),
        "```",
        "",
        "## Encoding: none vs by_tensor",
        "",
        "Paired ablation of `GradiendSplit.none()` (`:{cls}`) vs `by_tensor()` aggregate "
        "(`:{cls}:tensors`). **delta = :tensors - none** (positive => by_tensor better). "
        "When `auc_n`/`auc_o` saturate near 1, use **bal / d / spec / smag / emag** only as "
        "secondary signals — do not overclaim. "
        "Matched **causal** for the same split lives under `gradiend:{cls}:tensors` and "
        "`actiend:{cls}:tensors_tok_*` (see Causal table).",
        "",
        _md_table(
            [
                "pair",
                "split",
                "auc_n",
                "auc_o",
                "bal",
                "d",
                "spec",
                "excl",
                "smag",
                "emag",
                "verdict",
            ],
            _none_vs_tensor_md_rows(results),
        ),
        "",
        "```",
        none_vs_tensor_table(results).strip(),
        "```",
        "",
        "## Feature suitability",
        "",
        "Single score for \"suitable feature learnt?\": **S = E * G**. "
        "Encoding bottleneck **E** is `min(auc_n, auc_o, excl)` when rivals exist, "
        "else `min(auc_n, spec)` (`scope=vs_neutral_only`). "
        "Causal gate **G** is 1 iff intended-direction LMS-gated `eff >= tau_c` "
        "(default 0.05), the LMS check passes, the effect beats its matched random "
        "control when available, and the row is not floor+null / gate_empty; else 0. "
        "`reason`: `ok` | `E` (weak encoding) | `G` (causal fail) | `no_causal` | `no_encoding`.",
        "",
        "Cross-run matrix (models × tasks, mean primary S, with mean margins): "
        "`analysis/tables/suitability_model_task.txt` (refreshed at end of each run).",
        "",
        _md_table(
            ["method", "S", "E", "G", "scope", "reason", "gate detail"],
            _suitability_md_rows(results),
        ),
        "",
        "```",
        suitability_table(results, primary_only=True).strip(),
        "```",
        "",
        "## Causal",
        "",
        "Causal: LMS×0.99 gate only. Ids shared for encoder+causal ablations when "
        "applicable. **flag=ceiling** = at grid max → try higher LR. "
        "**floor+null** ≈ no useful effect. "
        "**smag/emag** are first-class (SAE often beats GRADIEND/ACTIEND on smag). "
        "See `lms_vs_strength` and `decoder_artifacts.json`.",
        "",
        "```",
        causal_table(results).strip(),
        "```",
        "",
    ]
    dec_art = ((results.get("raw") or {}).get("causal") or {}).get("decoder_artifacts") or []
    if dec_art:
        sections.extend(
            [
                "### Decoder artifacts (GRADIEND API)",
                "",
                "Paths to package decoder plots / grid JSON / row-wise CSV:",
                "",
            ]
        )
        for ent in dec_art:
            label = ent.get("method") or "?"
            plots = ", ".join(f"`{p}`" for p in (ent.get("plot_paths") or [])[:4]) or "—"
            sections.append(
                f"- **{label}**: plots={plots}; "
                f"grid=`{ent.get('output_path') or '—'}`; "
                f"csv=`{ent.get('raw_output_path') or '—'}`"
            )
        sections.append("")
        cat = ((results.get("raw") or {}).get("causal") or {}).get("decoder_artifacts_path")
        if cat:
            sections.append(f"Full catalog: `{cat}`\n")
    fails = _known_failures(results)
    compute = results.get("compute") or ((results.get("raw") or {}).get("cost_summary") or {})
    if compute:
        from cost_timer import format_cost_summary_md

        sections.append(format_cost_summary_md(compute))
    sections.extend(
        [
            "## Known issues / flags",
            "",
            *(fails if fails else ["- (none flagged)"]),
            "",
            "## Encoder methods (full)",
            "",
            "```",
            summary_table(results, headline_only=False).strip(),
            "```",
            "",
        ]
    )
    if plot_paths:
        sections.append("## Plots")
        sections.append("")
        for name, path in sorted(plot_paths.items()):
            sections.append(f"- **{name}**: `{path}`")
        sections.append("")
    if metrics_csv:
        sections.append(f"## Metrics CSV\n\n`{metrics_csv}`\n")
    sections.append(
        "## Console tables\n\n"
        "Plain-text copies of the end-of-run tables (summary / fair / causal / localization) "
        "are also written to `TABLES.txt` in the run directory.\n"
    )
    sections.extend(
        [
            "## Config",
            "",
            "```",
            "\n".join(f"{k}: {v}" for k, v in sorted(cfg.items())),
            "```",
            "",
        ]
    )
    return "\n".join(sections)


def normalize_legacy_causal_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Map pre-fixup causal rows (selected_strength/lms/…) onto causal_* keys.

    Safe to call on already-normalized payloads (no-op when causal_* present).
    Also re-binds decoder-backed headlines to the package plot LR.
    """
    from causal_eval import apply_package_decoder_headlines

    apply_package_decoder_headlines(results)
    raw_causal = (results.get("raw") or {}).get("causal")
    if isinstance(raw_causal, dict):
        by = raw_causal.get("by_method")
        if isinstance(by, dict):
            raw_causal["by_method"] = {
                k: v
                for k, v in by.items()
                if not (isinstance(v, str) and v.startswith("<class "))
            }
        summaries = raw_causal.get("summaries")
        if isinstance(summaries, list):
            raw_causal["summaries"] = [s for s in summaries if isinstance(s, dict)]

    for row in results.get("methods") or []:
        mid = str(row.get("method") or "")
        m = row.setdefault("metrics", {})
        extras = row.get("extras") or {}
        is_causal = mid.endswith(":causal") or extras.get("causal") is not None
        if not is_causal:
            continue
        if m.get("causal_selected_strength") is None and (
            m.get("selected_strength") is not None or m.get("strength") is not None
        ):
            m["causal_selected_strength"] = m.get("selected_strength", m.get("strength"))
        if m.get("causal_signed_effect") is None and m.get("signed_effect") is not None:
            m["causal_signed_effect"] = m.get("signed_effect")
        if m.get("causal_lms") is None and m.get("lms") is not None:
            m["causal_lms"] = m.get("lms")
        if m.get("causal_base_lms") is None and m.get("base_lms") is not None:
            m["causal_base_lms"] = m.get("base_lms")
        if m.get("causal_lms_ok") is None and m.get("lms_ok") is not None:
            m["causal_lms_ok"] = m.get("lms_ok")
        groups = m.get("summary_by_group") or {}
        tgt = m.get("target_class")
        if tgt and isinstance(groups.get(tgt), dict):
            g = groups[tgt]
            if m.get("causal_delta_target") is None:
                m["causal_delta_target"] = g.get("mean_delta_p_target")
            if m.get("causal_delta_mean") is None:
                m["causal_delta_mean"] = g.get("mean_delta_p_target")
        if m.get("causal_null_effect") is None:
            eff = m.get("causal_signed_effect")
            if isinstance(eff, (int, float)) and abs(float(eff)) < 1e-6:
                m["causal_null_effect"] = True
    return results


def _poc_decoder_panels_table(results: Dict[str, Any]) -> str:
    """Plain-text PoC before/after P(target)/P(partner) panels when present."""
    raw = (results.get("raw") or {}).get("causal") or {}
    panels = dict(raw.get("poc_decoder_panels") or {})
    # Also accept panels stored on method extras (after sync / older dumps).
    for row in results.get("methods") or []:
        mid = str(row.get("method") or "").removesuffix(":causal")
        panel = (row.get("extras") or {}).get("poc_decoder_panel")
        if isinstance(panel, dict) and mid not in panels:
            panels[mid] = panel
        metrics = row.get("metrics") or {}
        if mid not in panels and metrics.get("poc_before") and metrics.get("poc_after"):
            panels[mid] = {
                "before": metrics["poc_before"],
                "after": metrics["poc_after"],
                "target_class": metrics.get("target_class"),
            }
    if not panels:
        return ""
    lines = [
        "",
        "PoC causal decoder panel (before → after rewrite on factual prompts):",
        "  Same table as gradiend/examples/train_source_both_ioi_one_pole.py",
    ]
    for mid, panel in panels.items():
        lines.append(f"  [{mid}]")
        table = panel.get("table")
        if table:
            lines.append(f"    {'metric':<16} {'before':>12} {'after':>12} {'delta':>12}")
            lines.append("    " + "-" * 56)
            for row in table:
                b, a, d = row.get("before"), row.get("after"), row.get("delta")

                def _f(v: Any) -> str:
                    if v is None:
                        return "—"
                    return f"{float(v):.6f}"

                lines.append(
                    f"    {str(row.get('metric')):<16} {_f(b):>12} {_f(a):>12} {_f(d):>12}"
                )
            continue
        before = panel.get("before") or {}
        after = panel.get("after") or {}
        keys = list(dict.fromkeys([*before.keys(), *after.keys()]))
        lines.append(f"    {'metric':<16} {'before':>12} {'after':>12} {'delta':>12}")
        lines.append("    " + "-" * 56)
        for key in keys:
            b, a = before.get(key), after.get(key)
            d = None if b is None or a is None else float(a) - float(b)

            def _f(v: Any) -> str:
                if v is None:
                    return "—"
                return f"{float(v):.6f}"

            lines.append(f"    {f'P({key})':<16} {_f(b):>12} {_f(a):>12} {_f(d):>12}")
    return "\n".join(lines)


def build_console_tables_text(results: Dict[str, Any]) -> str:
    """Same plain-text tables printed at end of run (for TABLES.txt)."""
    from localization_eval import localization_table as _loc_table
    from suitability import DEFAULT_GLOBAL_OUT

    normalize_legacy_causal_metrics(results)
    _ensure_suitability(results)
    loc = (results.get("raw") or {}).get("localization") or {}
    parts = [
        summary_table(results, headline_only=False),
        comparison_table(results, primary_only=True),
        none_vs_tensor_table(results),
        suitability_table(results, primary_only=True),
        causal_table(results),
        _poc_decoder_panels_table(results),
        _loc_table(loc),
        "",
        "(Layer + fixed-k + tok-policy encoder rows are in the FULL encoder table above; "
        "see also REPORT.md / metrics.csv / results.json)",
        "",
    ]
    global_txt = Path(DEFAULT_GLOBAL_OUT) / "suitability_model_task.txt"
    if global_txt.is_file():
        try:
            overall = global_txt.read_text(encoding="utf-8").strip()
            if overall:
                parts.extend(["", overall, "", f"(Global matrix: {global_txt})", ""])
        except Exception as exc:
            parts.extend(["", f"(Global suitability matrix unreadable: {exc})", ""])
    return "\n".join(parts)


def write_study_tables(results: Dict[str, Any], output_dir: Path) -> str:
    """Write console-style tables to ``TABLES.txt``; return path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "TABLES.txt"
    path.write_text(build_console_tables_text(results), encoding="utf-8")
    return str(path)


def write_study_report(
    results: Dict[str, Any],
    output_dir: Path,
    *,
    plot_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Write REPORT.md + metrics.csv + TABLES.txt; return artifact path map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_legacy_causal_metrics(results)
    _ensure_suitability(results)
    csv_path = write_metrics_csv(results, output_dir / "metrics.csv")
    md = build_report_markdown(
        results, plot_paths=plot_paths, metrics_csv=str(csv_path)
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    tables_path = write_study_tables(results, output_dir)
    return {
        "report": str(report_path),
        "metrics_csv": str(csv_path),
        "tables": tables_path,
    }


def _iter_results_json(
    runs_root: Path,
    *,
    model: Optional[str] = None,
    task: Optional[str] = None,
) -> Iterator[Path]:
    runs_root = Path(runs_root)
    seen: set[Path] = set()
    for pattern in ("*/*/results.json", "*/results.json"):
        for path in sorted(runs_root.glob(pattern)):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            parts = path.relative_to(runs_root).parts
            model_key = parts[0] if parts else None
            task_id = parts[1] if len(parts) >= 3 else None
            if model and model_key != model:
                continue
            if task and task_id != task:
                continue
            yield path


def _plot_paths_from_results(
    results: Dict[str, Any], run_dir: Path
) -> Dict[str, str]:
    arts = results.get("artifacts") or {}
    plots = arts.get("plots")
    if isinstance(plots, dict) and plots:
        return {str(k): str(v) for k, v in plots.items()}
    plots_dir = Path(run_dir) / "plots"
    if plots_dir.is_dir():
        return {p.stem: str(p) for p in sorted(plots_dir.glob("*.png"))}
    return {}


def regenerate_run_report(run_dir: Path) -> Dict[str, str]:
    """Rewrite REPORT.md / TABLES.txt / metrics.csv from ``results.json``.

    No GPU / causal re-run. Decoder-backed ``str`` is rebound to the plot LR.
    """
    run_dir = Path(run_dir)
    path = run_dir / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a results dict: {path}")
    return write_study_report(
        payload,
        run_dir,
        plot_paths=_plot_paths_from_results(payload, run_dir),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate REPORT.md, TABLES.txt, and metrics.csv from results.json "
            "(no causal re-run)."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="*",
        type=Path,
        help="Run directory containing results.json (e.g. runs/gpt2-small/emotion)",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Root of runs/ to scan when regenerating several tasks",
    )
    parser.add_argument("--model", default=None, help="Filter to one model key")
    parser.add_argument("--task", default=None, help="Filter to one task id")
    args = parser.parse_args(argv)

    dirs = [Path(p) for p in args.run_dir]
    if args.runs is not None or args.model or args.task:
        root = args.runs or Path("runs")
        dirs.extend(p.parent for p in _iter_results_json(root, model=args.model, task=args.task))
    uniq: List[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        key = d.resolve()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    if not uniq:
        parser.error("pass a run dir, or --runs / --model / --task")

    for d in uniq:
        paths = regenerate_run_report(d)
        print(f"{d}: {paths.get('report')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
