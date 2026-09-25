from __future__ import annotations

import re

from analysis.ablation_retention import (
    DELTA_E_TAU,
    LARGE_DELTA,
    MIN_FLIP_TASKS,
    MIN_PAIRED,
    axis_candidate_pool,
    best_single_layer_vs_all,
    bootstrap_ci,
    classify_recipe,
    decide_retention,
    evaluate_all,
    multi_class_tasks,
    ranking_flips,
    restrict_to_study_tasks,
    suggest_core_setting,
    suggest_core_settings,
    symmetric_axis_ranking,
    variants_for_axis,
    wilcoxon_signed_rank_p,
    AXES,
    POLE_AXES,
    _index_cells,
)
from analysis.family_overview import _metric_value


def test_study_task_filter_excludes_hidden_ioi_and_keeps_visible_ioi_mib():
    rows = restrict_to_study_tasks(
        [
            {"task": "ioi", "recipe": "sae:k1"},
            {"task": "ioi_mib", "recipe": "sae:k1"},
            {"task": "gender_en_pre", "recipe": "sae:k1"},
            {"task": "gender_en", "recipe": "sae:k1"},
        ]
    )
    assert [row["task"] for row in rows] == ["ioi_mib", "gender_en"]


def test_recipe_metric_adapter_uses_paper_detection_not_legacy_encoding_e():
    metrics = {
        "roc_auc_neutral": 0.95,
        "neutral_specificity": 0.80,
        "roc_auc_other": 0.90,
        "class_exclusivity": 0.85,
        "encoding_E": 0.90,
    }
    assert _metric_value(metrics, "detection_score") == 0.80


def test_classify_core_vs_oracle_vs_select():
    assert classify_recipe("sae:k1")["core"] is True
    assert classify_recipe("sae:k1")["oracle"] is False
    assert classify_recipe("sae:kstar")["core"] is True
    assert classify_recipe("sae:sel_opp_fire")["axis"] == "sae_select"
    assert classify_recipe("sae:sel_opp_fire")["core"] is False
    assert classify_recipe("sae:L7")["oracle"] is True
    assert classify_recipe("caa:L3_act_prediction")["oracle"] is True
    assert classify_recipe("caa:act_mean")["axis"] == "caa_pool"
    assert classify_recipe("caa:act_mean")["oracle"] is False
    assert classify_recipe("actiend:tok_all")["axis"] == "actiend_tok"


def test_variants_for_axis_skips_oracles():
    recipes = [
        "sae:k1",
        "sae:kstar",
        "sae:all_k1",
        "sae:L3",
        "sae:sel_opp_fire",
        "caa:act_prediction",
        "caa:act_mean",
        "caa:L0_act_mean",
    ]
    bag = next(a for a in AXES if a.axis == "sae_bag")
    sel = next(a for a in AXES if a.axis == "sae_select")
    pool = next(a for a in AXES if a.axis == "caa_pool")
    assert variants_for_axis(recipes, bag) == ["sae:all_k1", "sae:kstar"]
    assert variants_for_axis(recipes, sel) == ["sae:sel_opp_fire"]
    assert variants_for_axis(recipes, pool) == ["caa:act_mean"]


def test_ranking_flip_detects_family_winner_change():
    cells = [
        {"model": "gpt2-small", "task": "t1", "recipe": "gradiend:two_pole", "metric": "encoding_E", "mean": 0.70},
        {"model": "gpt2-small", "task": "t1", "recipe": "actiend:two_pole", "metric": "encoding_E", "mean": 0.60},
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:k1", "metric": "encoding_E", "mean": 0.50},
        {"model": "gpt2-small", "task": "t1", "recipe": "caa:act_prediction", "metric": "encoding_E", "mean": 0.55},
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:sel_opp_fire", "metric": "encoding_E", "mean": 0.90},
        {"model": "gpt2-small", "task": "t2", "recipe": "gradiend:two_pole", "metric": "encoding_E", "mean": 0.95},
        {"model": "gpt2-small", "task": "t2", "recipe": "sae:k1", "metric": "encoding_E", "mean": 0.40},
        {"model": "gpt2-small", "task": "t2", "recipe": "sae:sel_opp_fire", "metric": "encoding_E", "mean": 0.41},
        {"model": "gpt2-small", "task": "t2", "recipe": "actiend:two_pole", "metric": "encoding_E", "mean": 0.50},
        {"model": "gpt2-small", "task": "t2", "recipe": "caa:act_prediction", "metric": "encoding_E", "mean": 0.50},
    ]
    idx = _index_cells(cells)
    flips = ranking_flips(idx, variant="sae:sel_opp_fire", family="sae", metric="encoding_E")
    assert len(flips) == 1
    assert flips[0]["task"] == "t1"
    assert flips[0]["base_winner"] == "gradiend"
    assert flips[0]["alt_winner"] == "sae"


def test_decide_does_not_promote_oracles_or_core_defaults():
    tier, _ = decide_retention(
        recipe="sae:k1",
        axis="sae_bag",
        core=True,
        oracle=False,
        n_paired_e=10,
        n_paired_c=10,
        mean_dE=0.2,
        mean_dC=0.2,
        n_flips=9,
        n_flip_tasks_scored=10,
    )
    assert tier == "main"

    tier, reason = decide_retention(
        recipe="sae:L7",
        axis="sae_layer",
        core=False,
        oracle=True,
        n_paired_e=10,
        n_paired_c=10,
        mean_dE=0.3,
        mean_dC=0.3,
        n_flips=9,
        n_flip_tasks_scored=10,
    )
    assert tier == "oracle"
    assert "test-set" in reason

    tier, _ = decide_retention(
        recipe="caa:act_mean",
        axis="caa_pool",
        core=False,
        oracle=False,
        n_paired_e=8,
        n_paired_c=8,
        mean_dE=0.001,
        mean_dC=0.001,
        n_flips=0,
        n_flip_tasks_scored=8,
    )
    assert tier == "appendix_small"

    tier, _ = decide_retention(
        recipe="caa:act_last",
        axis="caa_pool",
        core=False,
        oracle=False,
        n_paired_e=1,
        n_paired_c=1,
        mean_dE=0.12,
        mean_dC=0.0,
        n_flips=1,
        n_flip_tasks_scored=1,
    )
    assert tier == "appendix_small"

    tier, _ = decide_retention(
        recipe="sae:sel_opp_fire",
        axis="sae_select",
        core=False,
        oracle=False,
        n_paired_e=8,
        n_paired_c=8,
        mean_dE=0.01,
        mean_dC=0.0,
        n_flips=MIN_FLIP_TASKS,
        n_flip_tasks_scored=8,
    )
    assert tier == "review_promote"


def test_evaluate_all_on_tiny_cells():
    cells = []
    for task, k1, kstar, opp in (
        ("a", 0.50, 0.51, 0.90),
        ("b", 0.50, 0.50, 0.91),
        ("c", 0.50, 0.52, 0.92),
    ):
        for rec, val in (
            ("gradiend:two_pole", 0.70),
            ("actiend:two_pole", 0.60),
            ("caa:act_prediction", 0.55),
            ("sae:k1", k1),
            ("sae:kstar", kstar),
            ("sae:sel_opp_fire", opp),
        ):
            cells.append(
                {
                    "model": "gpt2-small",
                    "task": task,
                    "recipe": rec,
                    "metric": "encoding_E",
                    "mean": val,
                }
            )
    rows = {r["variant"]: r for r in evaluate_all(cells, model="gpt2-small")}
    assert rows["sae:kstar"]["tier"] == "main"
    assert rows["sae:sel_opp_fire"]["tier"] == "review_promote"
    assert rows["sae:sel_opp_fire"]["n_ranking_flips"] >= MIN_FLIP_TASKS
    assert abs(rows["sae:kstar"]["mean_dE"] - 0.01) < 1e-9
    assert DELTA_E_TAU == 0.02
    # diagnostic stats are present but never override the frozen tier logic
    assert "dE_ci_lo" in rows["sae:kstar"] and "dE_ci_hi" in rows["sae:kstar"]
    assert rows["sae:kstar"]["dE_wilcoxon_p"] is None  # n=3 < _MIN_N_WILCOXON


def test_wilcoxon_p_small_n_returns_none():
    assert wilcoxon_signed_rank_p([0.1, 0.2, -0.1]) is None
    assert wilcoxon_signed_rank_p([]) is None


def test_wilcoxon_p_distinguishes_signal_from_noise():
    consistent = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05]
    noisy = [0.05, -0.04, 0.03, -0.06, 0.02, -0.05, 0.04, -0.03, 0.01, -0.02]
    p_signal = wilcoxon_signed_rank_p(consistent)
    p_noise = wilcoxon_signed_rank_p(noisy)
    assert p_signal is not None and p_noise is not None
    assert p_signal < 0.01
    assert p_noise > 0.2


def test_bootstrap_ci_matches_sign_of_effect():
    lo, hi = bootstrap_ci([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05])
    assert lo is not None and hi is not None
    assert lo > 0  # consistently positive effect -> CI excludes 0

    lo2, hi2 = bootstrap_ci([0.05, -0.04, 0.03, -0.06, 0.02, -0.05, 0.04, -0.03, 0.01, -0.02])
    assert lo2 is not None and hi2 is not None
    assert lo2 < 0 < hi2  # noisy around zero -> CI should straddle 0


def test_bootstrap_ci_too_few_points():
    assert bootstrap_ci([0.1, 0.2]) == (None, None)


def test_axis_candidate_pool_includes_default_and_present_variants():
    recipes = ["sae:k1", "sae:kstar", "sae:all_k1", "sae:sel_opp_fire", "caa:act_prediction"]
    bag = next(a for a in AXES if a.axis == "sae_bag")
    assert axis_candidate_pool(recipes, bag) == ["sae:all_k1", "sae:k1", "sae:kstar"]


def test_symmetric_axis_ranking_restricts_to_full_house():
    cells = [
        {"model": "gpt2-small", "task": "a", "recipe": "sae:k1", "metric": "encoding_E", "mean": 0.5},
        {"model": "gpt2-small", "task": "a", "recipe": "sae:kstar", "metric": "encoding_E", "mean": 0.6},
        {"model": "gpt2-small", "task": "b", "recipe": "sae:k1", "metric": "encoding_E", "mean": 0.5},
        {"model": "gpt2-small", "task": "b", "recipe": "sae:kstar", "metric": "encoding_E", "mean": 0.3},
        # task c: kstar has no row here -> must be excluded from the paired comparison
        {"model": "gpt2-small", "task": "c", "recipe": "sae:k1", "metric": "encoding_E", "mean": 0.9},
    ]
    idx = _index_cells(cells)
    rank = symmetric_axis_ranking(idx, candidates=["sae:k1", "sae:kstar"], metric="encoding_E")
    assert rank["n_paired"] == 2
    assert abs(rank["means"]["sae:k1"] - 0.5) < 1e-9
    assert abs(rank["means"]["sae:kstar"] - 0.45) < 1e-9
    assert rank["winner"] == "sae:k1"
    assert rank["deltas_vs_winner"]["sae:kstar"] == [0.5 - 0.6, 0.5 - 0.3]


def test_suggest_core_setting_keeps_core_when_default_wins():
    cells = []
    for task, k1, kstar, allk1 in (
        ("a", 0.90, 0.50, 0.30),
        ("b", 0.80, 0.40, 0.35),
        ("c", 0.85, 0.45, 0.30),
    ):
        for rec, val in (("sae:k1", k1), ("sae:kstar", kstar), ("sae:all_k1", allk1)):
            cells.append(
                {"model": "gpt2-small", "task": task, "recipe": rec, "metric": "encoding_E", "mean": val}
            )
    idx = _index_cells(cells)
    recipes = sorted({c["recipe"] for c in cells})
    bag = next(a for a in AXES if a.axis == "sae_bag")
    row = suggest_core_setting(idx, spec=bag, recipes=recipes, model="gpt2-small")
    assert row["suggested"] == "sae:k1"
    assert row["verdict"] == "keep_core"
    assert row["delta_vs_current_default"] is None


def test_suggest_core_setting_flags_change_when_variant_dominates():
    # Same shape as the existing ranking-flip test: sae:sel_opp_fire dominates
    # sae:k1 badly enough (large mean gap, flips the 4-family winner every task)
    # that the symmetric pass should recommend it over the current core default.
    cells = []
    for task in ("a", "b", "c"):
        for rec, val in (
            ("gradiend:two_pole", 0.70),
            ("actiend:two_pole", 0.60),
            ("caa:act_prediction", 0.55),
            ("sae:k1", 0.50),
            ("sae:sel_opp_fire", 0.90),
        ):
            cells.append(
                {"model": "gpt2-small", "task": task, "recipe": rec, "metric": "encoding_E", "mean": val}
            )
    idx = _index_cells(cells)
    recipes = sorted({c["recipe"] for c in cells})
    sel = next(a for a in AXES if a.axis == "sae_select")
    row = suggest_core_setting(idx, spec=sel, recipes=recipes, model="gpt2-small")
    assert row["suggested"] == "sae:sel_opp_fire"
    assert row["verdict"] == "suggest_change"
    assert row["n_ranking_flips"] >= MIN_FLIP_TASKS
    assert row["delta_vs_current_default"] is not None and row["delta_vs_current_default"] >= LARGE_DELTA


def test_suggest_core_setting_no_clear_winner_below_threshold():
    cells = []
    for task, base, other in (("a", 0.50, 0.505), ("b", 0.51, 0.515), ("c", 0.49, 0.495)):
        cells.append(
            {"model": "gpt2-small", "task": task, "recipe": "caa:act_prediction", "metric": "encoding_E", "mean": base}
        )
        cells.append(
            {"model": "gpt2-small", "task": task, "recipe": "caa:act_mean", "metric": "encoding_E", "mean": other}
        )
    idx = _index_cells(cells)
    recipes = sorted({c["recipe"] for c in cells})
    pool = next(a for a in AXES if a.axis == "caa_pool")
    row = suggest_core_setting(idx, spec=pool, recipes=recipes, model="gpt2-small")
    assert row["suggested"] == "caa:act_mean"
    assert row["verdict"] == "no_clear_winner"
    assert row["n_ranking_flips"] == 0
    assert abs(row["delta_vs_current_default"]) < LARGE_DELTA


def test_suggest_core_setting_insufficient_data_below_min_paired():
    cells = [
        {"model": "gpt2-small", "task": "a", "recipe": "caa:act_prediction", "metric": "encoding_E", "mean": 0.5},
        {"model": "gpt2-small", "task": "a", "recipe": "caa:act_mean", "metric": "encoding_E", "mean": 0.9},
    ]
    idx = _index_cells(cells)
    recipes = sorted({c["recipe"] for c in cells})
    pool = next(a for a in AXES if a.axis == "caa_pool")
    row = suggest_core_setting(idx, spec=pool, recipes=recipes, model="gpt2-small")
    assert row["n_paired_E"] < MIN_PAIRED
    assert row["verdict"] == "insufficient_data"


def test_suggest_core_setting_returns_none_with_only_one_candidate():
    cells = [
        {"model": "gpt2-small", "task": "a", "recipe": "caa:act_prediction", "metric": "encoding_E", "mean": 0.5},
    ]
    idx = _index_cells(cells)
    recipes = sorted({c["recipe"] for c in cells})
    pool = next(a for a in AXES if a.axis == "caa_pool")
    assert suggest_core_setting(idx, spec=pool, recipes=recipes, model="gpt2-small") is None


def test_suggest_core_settings_skips_axes_without_multiple_candidates():
    cells = []
    for task in ("a", "b", "c"):
        for rec, val in (("sae:k1", 0.5), ("sae:kstar", 0.9)):
            cells.append(
                {"model": "gpt2-small", "task": task, "recipe": rec, "metric": "encoding_E", "mean": val}
            )
    rows = suggest_core_settings(cells, model="gpt2-small")
    axes = {r["axis"] for r in rows}
    assert axes == {"sae_bag"}
    assert rows[0]["suggested"] == "sae:kstar"
    assert rows[0]["verdict"] == "suggest_change"


def test_multi_class_tasks_filters_by_n_classes():
    cells = [
        {"model": "gpt2-small", "task": "binary_task", "recipe": "x", "metric": "m", "mean": 0.1, "n_classes": 2},
        {"model": "gpt2-small", "task": "triple_task", "recipe": "x", "metric": "m", "mean": 0.1, "n_classes": 3},
        {"model": "pythia-70m-deduped", "task": "quad_task", "recipe": "x", "metric": "m", "mean": 0.1, "n_classes": 4},
    ]
    assert multi_class_tasks(cells) == {
        ("gpt2-small", "triple_task"),
        ("pythia-70m-deduped", "quad_task"),
    }
    assert multi_class_tasks(cells, model="gpt2-small") == {("gpt2-small", "triple_task")}


def test_pole_axes_registered_and_scoped_to_multi_class_tasks():
    axes = {a.axis for a in POLE_AXES}
    assert axes == {"gradiend_pole", "actiend_pole"}

    cells = []
    # binary_task: one_pole dominates, but it's a 2-class task -> should not count.
    for rec, val in (("gradiend:two_pole", 0.10), ("gradiend:one_pole", 0.90)):
        cells.append(
            {
                "model": "gpt2-small", "task": "binary_task", "recipe": rec,
                "metric": "encoding_E", "mean": val, "n_classes": 2,
            }
        )
    # three >2-class tasks: two_pole modestly ahead each time -> keep_core expected.
    for task in ("multi_a", "multi_b", "multi_c"):
        for rec, val in (("gradiend:two_pole", 0.60), ("gradiend:one_pole", 0.55)):
            cells.append(
                {
                    "model": "gpt2-small", "task": task, "recipe": rec,
                    "metric": "encoding_E", "mean": val, "n_classes": 3,
                }
            )
    rows = {r["axis"]: r for r in suggest_core_settings(cells, model="gpt2-small")}
    row = rows["gradiend_pole"]
    assert row["n_paired_E"] == 3  # binary_task excluded
    assert row["suggested"] == "gradiend:two_pole"
    assert row["verdict"] == "keep_core"
    assert row["reason"].startswith("[n_classes>2 tasks only]")


def test_best_single_layer_vs_all_picks_max_layer_and_reports_delta():
    cells = [
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:all_k1", "metric": "encoding_E", "mean": 0.50},
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:L0", "metric": "encoding_E", "mean": 0.40},
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:L1", "metric": "encoding_E", "mean": 0.65},
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:L2", "metric": "encoding_E", "mean": 0.30},
        # sae:L0_k1 is a *different* (causal-only) recipe -- must not be picked up
        # by the bare-L{n} pattern used for the encoding_E comparison.
        {"model": "gpt2-small", "task": "t1", "recipe": "sae:L0_k1", "metric": "encoding_E", "mean": 0.99},
        # t2 has no all_k1 row -> must be excluded entirely.
        {"model": "gpt2-small", "task": "t2", "recipe": "sae:L0", "metric": "encoding_E", "mean": 0.90},
    ]
    idx = _index_cells(cells)
    rows = best_single_layer_vs_all(
        idx, family="sae", all_recipe="sae:all_k1", layer_pattern=re.compile(r"^L(\d+)$")
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "t1"
    assert row["best_layer"] == "L1"
    assert abs(row["best_layer_value"] - 0.65) < 1e-9
    assert abs(row["all_value"] - 0.50) < 1e-9
    assert abs(row["delta_best_layer_minus_all"] - 0.15) < 1e-9
    assert row["n_layers_seen"] == 3
