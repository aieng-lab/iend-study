from __future__ import annotations

import pandas as pd
import pytest

from analysis.method_statistics import (
    CORE_PANEL_METHODS,
    SAE_REFERENCE_METHODS,
    build_bayesian_outcomes,
    build_descriptive_leaderboard,
    build_panel_data,
    parse_sources,
    posterior_probability_matrices,
    run_autorank_panel,
    select_intermediate_panel,
)


def test_primary_autorank_panel_includes_directly_comparable_sae_variants():
    assert CORE_PANEL_METHODS["two_pole"] == (
        "gradiend:two_pole",
        "actiend:two_pole",
        "caga:two_pole",
        "agiend:two_pole",
        "cga:two_pole",
        "caa:two_pole",
        "sae:k1",
    )
    assert CORE_PANEL_METHODS["one_pole"] == (
        "gradiend:one_pole",
        "actiend:one_pole",
        "caga:one_pole",
        "agiend:one_pole",
        "cga:one_pole",
        "caa:one_pole",
        "sae:k1",
    )
    assert SAE_REFERENCE_METHODS == ("sae:k1",)


def _frame(methods, tasks=("emotion", "gender_en"), *, incomplete=()):
    rows = []
    for task in tasks:
        for index, method in enumerate(methods):
            rows.append(
                {
                    "task": task,
                    "method_group": method,
                    "run_status": "ok",
                    "detection_score": 0.7 + index / 100,
                    "intervention_score": 0.1 + index / 100,
                    "detection_complete": (task, method) not in incomplete,
                    "intervention_complete": (task, method) not in incomplete,
                }
            )
    return pd.DataFrame(rows)


def test_parse_sources_preserves_mixed_output_subdirs():
    assert parse_sources(["gpt2-small=suite_full2", "llama-3.1-8b"]) == (
        ("gpt2-small", "suite_full2"),
        ("llama-3.1-8b", ""),
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_sources(["gpt2-small=a", "gpt2-small=b"])


def test_panel_matrix_keeps_only_listwise_complete_blocks():
    methods = ("gradiend:one_pole", "actiend:one_pole")
    frames = {
        "m1": _frame(methods, incomplete=(("gender_en", "actiend:one_pole"),)),
        "m2": _frame(methods),
    }
    panel = build_panel_data(
        frames,
        sources=(("m1", ""), ("m2", "")),
        metric="detection_score",
        panel="one_pole",
        tasks=("emotion", "gender_en"),
        methods=methods,
    )
    assert list(panel.matrix.index) == [
        ("m1", "emotion"),
        ("m2", "emotion"),
        ("m2", "gender_en"),
    ]
    assert panel.n_expected == 4
    assert panel.n_complete == 3
    dropped = panel.coverage[
        (panel.coverage["model"] == "m1") & (panel.coverage["task"] == "gender_en")
    ].iloc[0]
    assert dropped["raw_incomplete_methods"] == "actiend:one_pole"


def test_detection_structural_components_exclude_sae_pre_one_pole_task():
    methods = ("gradiend:one_pole", "sae_pre:k1")
    panel = build_panel_data(
        {"m": _frame(methods, tasks=("induction",))},
        sources=(("m", ""),),
        metric="detection_score",
        panel="one_pole",
        tasks=("induction",),
        methods=methods,
    )
    assert panel.n_expected == 0
    assert panel.n_complete == 0
    assert panel.coverage.iloc[0]["structural_methods"] == "sae_pre:k1"


def test_strict_panel_writes_audit_but_refuses_incomplete_inference(tmp_path):
    methods = ("gradiend:one_pole", "actiend:one_pole")
    panel = build_panel_data(
        {"m": _frame(methods, incomplete=(("gender_en", "actiend:one_pole"),))},
        sources=(("m", ""),),
        metric="intervention_score",
        panel="one_pole",
        tasks=("emotion", "gender_en"),
        methods=methods,
    )
    result = run_autorank_panel(panel, tmp_path, strict=True)
    assert result["status"] == "incomplete-strict"
    assert (tmp_path / "intervention_one_pole_matrix.csv").is_file()
    assert (tmp_path / "intervention_one_pole_coverage.csv").is_file()
    assert not (tmp_path / "intervention_one_pole_report.txt").exists()


def test_bayesian_outcomes_distinguish_equivalence_from_inconclusive():
    labels = ["A", "B", "C"]
    posterior = pd.DataFrame(index=labels, columns=labels, dtype=object)
    posterior.loc["A", "B"] = (0.97, 0.02, 0.01)
    posterior.loc["A", "C"] = (0.02, 0.96, 0.02)
    posterior.loc["B", "C"] = (0.50, 0.10, 0.40)
    pairs, decisions = build_bayesian_outcomes(
        posterior,
        label_to_method={"A": "a", "B": "b", "C": "c"},
        alpha=0.05,
    )
    assert decisions.loc["A", "B"] == "better"
    assert decisions.loc["B", "A"] == "worse"
    assert decisions.loc["A", "C"] == "practically equivalent"
    assert decisions.loc["B", "C"] == "inconclusive"
    assert len(pairs) == 3


def test_posterior_probability_matrices_preserve_a_vs_b_direction():
    labels = ["A", "B"]
    posterior = pd.DataFrame(index=labels, columns=labels, dtype=object)
    posterior.loc["A", "B"] = (0.91, 0.05, 0.04)
    matrices = posterior_probability_matrices(posterior)
    assert matrices["a_better"].loc["A", "B"] == 0.91
    assert matrices["b_better"].loc["A", "B"] == 0.04
    assert matrices["a_better"].loc["B", "A"] == 0.04
    assert matrices["equivalent"].loc["B", "A"] == 0.05


def test_descriptive_leaderboard_exists_below_autorank_minimum():
    methods = ("gradiend:one_pole", "actiend:one_pole")
    panel = build_panel_data(
        {"m": _frame(methods, tasks=("emotion", "gender_en"))},
        sources=(("m", ""),),
        metric="intervention_score",
        panel="one_pole",
        tasks=("emotion", "gender_en"),
        methods=methods,
    )
    leaderboard = build_descriptive_leaderboard(panel)
    assert leaderboard.iloc[0]["method_group"] == "actiend:one_pole"
    assert leaderboard.iloc[0]["position"] == 1


def test_intermediate_selection_ranks_methods_not_blocks():
    methods = (
        "gradiend:one_pole",
        "actiend:one_pole",
        "caa:one_pole",
    )
    tasks = tuple(f"task_{index}" for index in range(6))
    frame = _frame(methods, tasks=tasks)
    frame.loc[
        (frame["task"] == "task_0") & (frame["method_group"] == "caa:one_pole"),
        "intervention_complete",
    ] = False
    panel, excluded = select_intermediate_panel(
        {"m": frame},
        sources=(("m", ""),),
        metric="intervention_score",
        panel="one_pole",
        tasks=tasks,
        methods=methods,
    )
    assert excluded == ()
    assert panel.methods == methods
    assert panel.n_complete == 5
    assert list(panel.matrix.columns) == list(methods)
