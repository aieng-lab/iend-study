import pandas as pd

from analysis.across_model_detail import (
    TASK_REGISTRY,
    _configure_latex_task_labels,
    automatic_rank_table,
    detail_matrix,
    headline_automatic_rank_table,
    headline_pooled_detail_matrix,
)
from analysis.layer_selection_appendix import _study_model_order


def test_pgf_task_labels_use_portable_serif_font_and_fa7():
    class FakePyplot:
        rcParams = {}

    pyplot = FakePyplot()
    _configure_latex_task_labels(pyplot)

    assert pyplot.rcParams["pgf.texsystem"] == "xelatex"
    assert pyplot.rcParams["font.family"] == "serif"
    assert r"\setmainfont{times.ttf}" in pyplot.rcParams["pgf.preamble"]
    assert r"\usepackage{fontawesome7}" in pyplot.rcParams["pgf.preamble"]


def test_detail_matrix_uses_percentage_det_int_cells():
    frame = pd.DataFrame(
        [
            {"model": "m1", "task": "gender_en", "method_group": "sae:k1", "detection": 0.901, "intervention": 0.456},
            {"model": "m1", "task": "emotion", "method_group": "sae:k1", "detection": 0.8, "intervention": float("nan")},
        ]
    )

    matrix = detail_matrix(frame, "m1")

    assert set(matrix.columns) == {"emotion", "gender_en"}
    assert matrix.loc["sae:k1", "emotion"] == "80.0/--"
    assert matrix.loc["sae:k1", "gender_en"] == "90.1/45.6"


def test_automatic_ranks_compete_within_pole_stratum_with_sae_in_both():
    frame = pd.DataFrame(
        [
            {"model": "m1", "task": "t1", "method_group": "one_pole_method", "pole": "one_pole", "backend": "caga", "detection": 0.9, "intervention": 0.1},
            {"model": "m1", "task": "t1", "method_group": "two_pole_method", "pole": "two_pole", "backend": "agiend", "detection": 0.8, "intervention": 0.2},
            {"model": "m1", "task": "t1", "method_group": "sae:k1", "pole": "two_pole", "backend": "sae", "detection": 0.85, "intervention": 0.15},
        ]
    )

    ranks = automatic_rank_table(frame)

    def rank(metric, comparison, method):
        row = ranks[ranks["metric"].eq(metric) & ranks["comparison"].eq(comparison) & ranks["method_group"].eq(method)]
        return float(row["mean_rank"].iloc[0])

    assert rank("Det.", "One-sided", "one_pole_method") == 1.0
    assert rank("Det.", "Pairwise", "two_pole_method") == 2.0
    assert rank("Int.", "Pairwise", "two_pole_method") == 1.0
    assert rank("Int.", "One-sided", "one_pole_method") == 2.0
    # A one-sided method never competes with a pairwise one.
    assert ranks[ranks["comparison"].eq("Pairwise")]["method_group"].tolist().count("one_pole_method") == 0


def test_headline_pooled_matrix_averages_model_scores_per_method_task():
    frame = pd.DataFrame(
        [
            {"model": "m1", "task": "gender_en", "method_group": "sae:k1", "detection": 0.8, "intervention": 0.1},
            {"model": "m2", "task": "gender_en", "method_group": "sae:k1", "detection": 1.0, "intervention": 0.3},
            # Not a headline method; it must not enter the compact table.
            {"model": "m1", "task": "gender_en", "method_group": "sae:kstar", "detection": 0.9, "intervention": 0.2},
        ]
    )
    matrix = headline_pooled_detail_matrix(frame)
    assert list(matrix.index) == ["sae:k1"]
    assert matrix.loc["sae:k1", "gender_en"] == "90.0/20.0"


def test_headline_automatic_rank_excludes_appendix_methods():
    frame = pd.DataFrame(
        [
            {"model": "m1", "task": "gender_en", "method_group": "sae:k1", "pole": "two_pole", "backend": "sae", "detection": 0.8, "intervention": 0.1},
            {"model": "m1", "task": "gender_en", "method_group": "sae:kstar", "pole": "two_pole", "backend": "sae", "detection": 0.9, "intervention": 0.2},
        ]
    )
    assert set(headline_automatic_rank_table(frame)["method_group"]) == {"sae:k1"}


def test_one_sided_tasks_have_one_feature_class():
    for task in ("ioi_mib", "key_value", "induction", "repetition", "function_composition"):
        assert TASK_REGISTRY[task][4] == 1


def test_layer_selection_uses_stable_study_model_order():
    assert _study_model_order(
        ["llama-3.1-8b", "gpt2-small", "pythia-70m-deduped"]
    ) == ["pythia-70m-deduped", "gpt2-small", "llama-3.1-8b"]
