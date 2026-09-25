"""Tests for analysis/convergence_overview.py."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.convergence_overview import (
    classify_status,
    collect_convergence_rows,
    format_convergence_counts_latex,
    format_convergence_problems_latex,
    format_convergence_report,
    load_convergence_row,
    parse_stage,
    summarize_by_backend,
)


def _write_done_json(path: Path, *, stage: str, extras: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "config_hash": "deadbeef", "extras": extras}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parse_stage_splits_backend_ablation_class_tensors():
    assert parse_stage("gradiend__pair__F-M") == {
        "backend": "gradiend",
        "ablation": "pair",
        "class_key": "F-M",
        "tensors": False,
    }
    assert parse_stage("actiend__onepole__M__tensors") == {
        "backend": "actiend",
        "ablation": "onepole",
        "class_key": "M",
        "tensors": True,
    }


def test_classify_status_collapsed_overrides_everything():
    row = {"collapsed_encoder": True, "has_convergence_info": True, "converged": True}
    assert classify_status(row) == "collapsed"


def test_classify_status_converged_and_not_converged():
    assert classify_status({"collapsed_encoder": False, "has_convergence_info": True, "converged": True}) == "converged"
    assert classify_status({"collapsed_encoder": False, "has_convergence_info": True, "converged": False}) == "not_converged"


def test_classify_status_legacy_run_without_convergence_info():
    good = {
        "collapsed_encoder": False,
        "has_convergence_info": False,
        "roc_auc_neutral_all": 0.99,
        "roc_auc_other_all": 0.95,
        "correlation_all": 0.9,
    }
    assert classify_status(good) == "unknown_no_convergence_info"
    bad = {
        "collapsed_encoder": False,
        "has_convergence_info": False,
        "roc_auc_neutral_all": 0.99,
        "roc_auc_other_all": 0.3,
        "correlation_all": 0.9,
    }
    assert classify_status(bad) == "unknown_low_quality"


def test_load_convergence_row_extracts_convergence_info_and_best_checkpoint(tmp_path):
    done = tmp_path / "runs" / "gpt2-small" / "induction" / "artifacts" / "gradiend__onepole__MATCH" / "done.json"
    _write_done_json(
        done,
        stage="gradiend__onepole__MATCH",
        extras={
            "learning_rate": 0.0001,
            "convergent_metric": "min_auc_n_o",
            "collapsed_encoder": False,
            "training_max_steps": 500,
            "best_score_checkpoint": {
                "min_auc_n_o": 0.778,
                "global_step": 500,
                "selection_metric": "min_auc_n_o",
            },
            "convergence_info": {
                "converged": False,
                "convergent_count": 0,
                "min_convergent_seeds": 1,
                "convergence_metric": "min_auc_n_o",
                "threshold": 0.7,
                "auc_positive_target_mean_required": True,
                "convergent_positive_target_class_mean": 0.0,
                "convergent_positive_target_class_mean_ok": False,
            },
            "encoder_eval": {
                "encoder_metrics": {"all_data": {"correlation": 0.5, "roc_auc_neutral": 0.6, "roc_auc_other": 0.6}}
            },
        },
    )
    row = load_convergence_row(done, model="gpt2-small", task="induction")
    assert row["backend"] == "gradiend"
    assert row["ablation"] == "onepole"
    assert row["class_key"] == "MATCH"
    assert row["has_convergence_info"] is True
    assert row["converged"] is False
    assert row["convergence_threshold"] == 0.7
    assert row["best_step"] == 500
    assert row["best_metric_value"] == 0.778
    assert row["auc_positive_target_mean_required"] is True
    assert row["positive_target_class_mean"] == 0.0
    assert row["positive_target_class_mean_ok"] is False
    assert row["status"] == "not_converged"
    assert row["seeds_summary"] == "0/1"


def test_legacy_auc_same_sign_summary_uses_target_positive_contract(tmp_path):
    done = tmp_path / "runs" / "gpt2-small" / "gender_en" / "artifacts" / "actiend__onepole__M" / "done.json"
    _write_done_json(
        done,
        stage="actiend__onepole__M",
        extras={
            "convergent_metric": "min_auc_n_o",
            "best_score_checkpoint": {
                "min_auc_n_o": 0.99,
                "global_step": 200,
                "selection_metric": "min_auc_n_o",
            },
            "convergence_info": {
                "converged": False,
                "convergent_count": 0,
                "min_convergent_seeds": 1,
                "convergence_metric": "min_auc_n_o",
                "threshold": 0.9,
            },
            "encoder_eval": {
                "encoder_metrics": {
                    "mean_by_class": {"-1.0": 0.95, "0.0": 0.0, "1.0": 0.98},
                    "all_data": {"roc_auc_neutral": 0.99, "roc_auc_other": 1.0},
                }
            },
        },
    )

    row = load_convergence_row(done, model="gpt2-small", task="gender_en")

    assert row["stored_converged"] is False
    assert row["converged"] is True
    assert row["convergence_reinterpreted"] is True
    assert row["positive_target_class_mean"] == 0.98
    assert row["positive_target_class_mean_ok"] is True
    assert row["status"] == "converged"


def test_legacy_auc_negative_target_mean_is_canonicalized(tmp_path):
    done = tmp_path / "runs" / "gpt2-small" / "gender_en" / "artifacts" / "actiend__onepole__M" / "done.json"
    _write_done_json(
        done,
        stage="actiend__onepole__M",
        extras={
            "convergent_metric": "min_auc_n_o",
            "best_score_checkpoint": {
                "min_auc_n_o": 0.99,
                "global_step": 200,
                "selection_metric": "min_auc_n_o",
            },
            "convergence_info": {
                "converged": False,
                "min_convergent_seeds": 1,
                "convergence_metric": "min_auc_n_o",
                "threshold": 0.9,
            },
            "encoder_eval": {
                "encoder_metrics": {
                    "mean_by_class": {"-1.0": -0.95, "0.0": 0.0, "1.0": -0.98},
                    "all_data": {"roc_auc_neutral": 0.99, "roc_auc_other": 1.0},
                }
            },
        },
    )

    row = load_convergence_row(done, model="gpt2-small", task="gender_en")

    assert row["converged"] is True
    assert row["positive_target_class_mean"] == 0.98
    assert row["positive_target_class_mean_ok"] is True
    assert row["orientation_flip_required"] is True
    assert row["status"] == "converged"


def test_load_convergence_row_missing_convergence_info_falls_back_to_encoder_quality(tmp_path):
    done = tmp_path / "runs" / "gpt2-small" / "gender_en" / "artifacts" / "gradiend__pair__F-M" / "done.json"
    _write_done_json(
        done,
        stage="gradiend__pair__F-M",
        extras={
            "learning_rate": 0.0001,
            "convergent_metric": "correlation",
            "collapsed_encoder": False,
            "encoder_eval": {
                "encoder_metrics": {"all_data": {"correlation": 0.98, "roc_auc_neutral": 1.0, "roc_auc_other": 1.0}}
            },
        },
    )
    row = load_convergence_row(done, model="gpt2-small", task="gender_en")
    assert row["has_convergence_info"] is False
    assert row["converged"] is None
    assert row["status"] == "unknown_no_convergence_info"


def test_collect_convergence_rows_skips_sae_and_non_backend_artifacts(tmp_path):
    runs_root = tmp_path / "runs"
    _write_done_json(
        runs_root / "gpt2-small" / "gender_en" / "artifacts" / "gradiend__pair__F-M" / "done.json",
        stage="gradiend__pair__F-M",
        extras={"collapsed_encoder": False},
    )
    _write_done_json(
        runs_root / "gpt2-small" / "gender_en" / "artifacts" / "sae" / "done.json",
        stage="sae",
        extras={},
    )
    rows = collect_convergence_rows(runs_root, "gpt2-small")
    assert len(rows) == 1
    assert rows[0]["backend"] == "gradiend"


def test_collect_convergence_rows_honors_subdir(tmp_path):
    runs_root = tmp_path / "runs"
    _write_done_json(
        runs_root / "gpt2-small" / "suite_full" / "gender_en" / "artifacts" / "actiend__onepole__F" / "done.json",
        stage="actiend__onepole__F",
        extras={"collapsed_encoder": True},
    )
    assert collect_convergence_rows(runs_root, "gpt2-small") == []
    rows = collect_convergence_rows(runs_root, "gpt2-small", subdir="suite_full")
    assert len(rows) == 1
    assert rows[0]["task"] == "gender_en"
    assert rows[0]["status"] == "collapsed"


def test_summarize_by_backend_counts_statuses():
    rows = [
        {"backend": "gradiend", "status": "converged"},
        {"backend": "gradiend", "status": "collapsed"},
        {"backend": "actiend", "status": "converged"},
    ]
    summary = summarize_by_backend(rows)
    by_backend = {r["backend"]: r for r in summary}
    assert by_backend["gradiend"]["total"] == 2
    assert by_backend["gradiend"]["converged"] == 1
    assert by_backend["gradiend"]["collapsed"] == 1
    assert by_backend["actiend"]["total"] == 1


def test_format_report_only_problems_filters_converged_rows():
    rows = [
        {
            "task": "gender_en",
            "backend": "gradiend",
            "ablation": "pair",
            "class_key": "F-M",
            "tensors": False,
            "status": "converged",
            "convergence_metric": "correlation",
            "convergence_threshold": 0.5,
            "best_metric_value": 0.9,
            "best_step": 200,
            "training_max_steps": 500,
        },
        {
            "task": "induction",
            "backend": "gradiend",
            "ablation": "onepole",
            "class_key": "MATCH",
            "tensors": False,
            "status": "not_converged",
            "convergence_metric": "min_auc_n_o",
            "convergence_threshold": 0.7,
            "best_metric_value": 0.6,
            "best_step": 500,
            "training_max_steps": 500,
        },
    ]
    report = format_convergence_report(rows, model="gpt2-small", only_problems=True)
    assert "induction" in report
    assert "gender_en" not in report
    assert "Problems (1 of 2 total)" in report
    assert "selection" in report


def test_format_convergence_counts_latex_renders_table():
    rows = [
        {"backend": "gradiend", "status": "converged"},
        {"backend": "gradiend", "status": "collapsed"},
        {"backend": "actiend", "status": "not_converged"},
    ]
    tex = format_convergence_counts_latex(rows, model="gpt2-small")
    assert r"\begin{table}" in tex
    assert "convergence status" in tex
    assert "GRADIEND" in tex
    assert "ACTIEND" in tex


def test_format_convergence_problems_latex_is_longtable_and_escapes_underscores():
    rows = [
        {
            "task": "function_composition",
            "backend": "gradiend",
            "ablation": "onepole",
            "class_key": "RESULT",
            "tensors": False,
            "status": "not_converged",
            "convergence_metric": "min_auc_n_o",
            "convergence_threshold": 0.7,
            "best_metric_value": 0.64,
            "best_step": 400,
        },
        {
            "task": "gender_en",
            "backend": "gradiend",
            "ablation": "pair",
            "class_key": "F-M",
            "tensors": False,
            "status": "converged",
            "convergence_metric": "correlation",
            "convergence_threshold": 0.5,
            "best_metric_value": 0.98,
            "best_step": 200,
        },
    ]
    tex = format_convergence_problems_latex(rows, model="gpt2-small")
    assert r"\begin{longtable}" in tex
    # "converged" row is not a problem status and must not appear.
    assert "gender_en" not in tex
    # each underscore is escaped and gets an explicit break opportunity so a
    # long identifier can wrap within a narrow p{} column instead of
    # overflowing into the next one.
    assert r"function\_\allowbreak{}composition" in tex
    assert "Selection score" in tex
    assert "Selection" in tex


def test_seed_summary_labels_configured_maximum_not_seeds_tried(tmp_path):
    done = (
        tmp_path
        / "runs"
        / "gpt2-small"
        / "induction"
        / "artifacts"
        / "gradiend__onepole__MATCH"
        / "done.json"
    )
    _write_done_json(
        done,
        stage="gradiend__onepole__MATCH",
        extras={
            "training_max_seeds": 3,
            "best_score_checkpoint": {"min_auc_n_o": 0.95},
            "convergence_info": {
                "converged": True,
                "convergent_count": 1,
                "min_convergent_seeds": 1,
                "convergence_metric": "min_auc_n_o",
                "threshold": 0.9,
            },
        },
    )
    row = load_convergence_row(done, model="gpt2-small", task="induction")
    assert row["seeds_summary"] == "1/1 (max 3)"


def test_component_coverage_is_exposed_in_row_and_report(tmp_path):
    done = (
        tmp_path
        / "runs"
        / "gpt2-small"
        / "ravel_country"
        / "artifacts"
        / "actiend__pair__china-united_states__tensors"
        / "done.json"
    )
    _write_done_json(
        done,
        stage="actiend__pair__china-united_states__tensors",
        extras={
            "best_score_checkpoint": {"correlation": 0.98},
            "convergence_info": {
                "converged": False,
                "convergent_count": 0,
                "min_convergent_seeds": 1,
                "convergence_metric": "correlation",
                "threshold": 0.5,
                "component_seed_summary": {
                    "n_components": 12,
                    "n_satisfied_components": 11,
                    "missing_component_ids": ["activation:transformer.h.11"],
                },
            },
        },
    )
    row = load_convergence_row(done, model="gpt2-small", task="ravel_country")
    assert row["component_summary"] == "11/12"
    assert row["missing_component_ids"] == ["activation:transformer.h.11"]
    report = format_convergence_report([row], only_problems=True)
    assert "components" in report
    assert "11/12" in report


def test_format_convergence_problems_latex_empty_when_no_problems():
    rows = [{"backend": "gradiend", "status": "converged", "task": "gender_en"}]
    assert format_convergence_problems_latex(rows, model="gpt2-small") == ""
