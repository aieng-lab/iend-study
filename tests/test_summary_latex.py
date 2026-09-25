"""Tests for analysis/summary_latex.py merge and LaTeX helpers."""



from __future__ import annotations



import pandas as pd



from analysis.summary_latex import (

    HIDDEN_MATRIX_TASKS,

    MEAN_LABEL,
    PAPER_BINARY_TASKS,

    PAPER_METHOD_GROUPS,

    PAPER_MULTICLASS_TASKS,

    PAPER_ONE_POLE_TASKS,

    PAPER_TASK_ORDER,

    SUMMARY_TASKS,

    build_method_metric_rank_summary,

    build_method_metric_summary,

    build_task_method_matrix,

    column_winner_methods,

    compute_task_metric_ranks,

    fallback_footnote,

    format_metric_matrix_latex,

    format_concatenated_metric_matrices_latex,

    format_cross_model_headline_latex,

    format_method_summary_latex,

    merge_with_fallback,

    methods_with_data,

    methods_with_encoder_summary_data,

    resolve_matrix_tasks,

    result_view_methods,

    result_view_tasks,

    resolve_summary_tasks,

    row_winner_methods,

    table_snippet_for_pdf,

    write_headline_tables_book,

    write_diagnostics_book,

    write_tables_book,

    write_concatenated_tables_book,

)





def _row(model, task, group, causal=None, enc=0.5, source="current"):

    return {

        "model": model,

        "task": task,

        "method_group": group,

        "causal_signed_effect": causal,

        "causal_lms": 0.09 if causal is not None else None,

        "encoding_E": enc,

        "roc_auc_neutral": enc,

        "roc_auc_other": enc,

        "neutral_specificity": enc,

        "class_exclusivity": enc,

        "data_source": source,

    }





def test_merge_prefers_current_causal():

    cur = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "sae:k1", causal=0.2),

        ]

    )

    fb = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "sae:k1", causal=0.05),

        ]

    )

    merged, prov = merge_with_fallback(cur, fb, model="gpt2-small")

    row = merged.iloc[0]

    assert float(row["causal_signed_effect"]) == 0.2

    assert prov == []





def test_merge_patches_missing_sae_causal_from_fallback():

    cur = pd.DataFrame(

        [

            _row("gpt2-small", "emotion", "sae:k1", causal=None, enc=0.9),

        ]

    )

    fb = pd.DataFrame(

        [

            _row("gpt2-small", "emotion", "sae:k1", causal=0.0007, enc=0.4),

        ]

    )

    merged, prov = merge_with_fallback(cur, fb, model="gpt2-small")

    row = merged.iloc[0]

    assert float(row["causal_signed_effect"]) == 0.0007

    assert float(row["encoding_E"]) == 0.9

    assert len(prov) == 1





def test_merge_injects_missing_sae_row():

    cur = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.2),

        ]

    )

    fb = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "sae:k1", causal=0.16),

        ]

    )

    merged, prov = merge_with_fallback(cur, fb, model="gpt2-small")

    sae = merged[merged["method_group"] == "sae:k1"]

    assert len(sae) == 1

    assert float(sae.iloc[0]["causal_signed_effect"]) == 0.16

    assert prov[0]["kind"] == "row"





def test_default_matrix_tasks_use_all_available():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.1),

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.1),

            _row("gpt2-small", "ioi_mib", "gradiend:two_pole", causal=0.1),

        ]

    )

    tasks = resolve_matrix_tasks(df, model="gpt2-small", tasks_arg="default")

    assert tasks == ["gender_en", "emotion", "ioi_mib"]





def test_matrix_tasks_drop_hidden_and_follow_multiclass_then_one_pole():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en_pre", "gradiend:two_pole", causal=0.1),

            _row("gpt2-small", "ioi_mib", "gradiend:one_pole", causal=0.1),

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.1),

            _row("gpt2-small", "race_one_pole", "actiend:one_pole", causal=0.1),

        ]

    )

    tasks = resolve_matrix_tasks(df, model="gpt2-small", tasks_arg="default")

    assert "gender_en_pre" not in tasks

    # race_one_pole is study_hidden (configs/tasks/race_one_pole.yaml), so
    # resolve_matrix_tasks drops it from the paper tables like any hidden task --
    # this is the "drop_hidden" behavior the test name promises.
    assert "race_one_pole" not in tasks

    assert tasks == ["emotion", "ioi_mib"]

    assert set(PAPER_TASK_ORDER) == (
        set(PAPER_MULTICLASS_TASKS)
        | set(PAPER_BINARY_TASKS)
        | set(PAPER_ONE_POLE_TASKS)
    )

    assert "gender_en_pre" in HIDDEN_MATRIX_TASKS
    assert "ioi" in HIDDEN_MATRIX_TASKS





def test_metric_matrix_rotates_established_task_headers():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.2),

            _row("gpt2-small", "ioi_mib", "sae:k1", causal=0.1),

        ]

    )

    matrix = build_task_method_matrix(

        df,

        model="gpt2-small",

        tasks=["emotion", "ioi_mib"],

        methods=["gradiend:two_pole", "sae:k1"],

        metric="causal_signed_effect",

        specs={},

    )

    tex = format_metric_matrix_latex(

        matrix,

        metric="causal_signed_effect",

        model="gpt2-small",

        tasks=["emotion", "ioi_mib"],

        methods=["gradiend:two_pole", "sae:k1"],

        specs={},

    )

    assert r"Method & \rotatebox{45}{\taskEmotion}" in tex
    assert r"& \rotatebox{45}{\taskMIBIOI}" in tex
    assert r"\begin{tabular}{l*{2}{c}}" in tex





def test_summary_tasks_are_reliable_subset_only():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.1),

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.1),

            _row("gpt2-small", "ioi", "gradiend:two_pole", causal=0.1),

        ]

    )

    tasks = resolve_summary_tasks(df, model="gpt2-small")

    assert tasks == ["gender_en", "emotion"]





def test_method_summary_averages_summary_tasks_only():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.2, enc=0.8),

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.4, enc=1.0),

            _row("gpt2-small", "ioi", "gradiend:two_pole", causal=0.0, enc=0.0),

        ]

    )

    summary = build_method_metric_summary(

        df,

        model="gpt2-small",

        tasks=SUMMARY_TASKS[:2],

        methods=["gradiend:two_pole"],

        metrics=["encoding_E", "causal_signed_effect"],

    )

    assert abs(float(summary.loc["gradiend:two_pole", "encoding_E"]) - 0.9) < 1e-9

    assert abs(float(summary.loc["gradiend:two_pole", "causal_signed_effect"]) - 0.3) < 1e-9


def test_method_summary_supports_median_aggregation():
    df = pd.DataFrame(
        [
            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.1),
            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.2),
            _row("gpt2-small", "race", "gradiend:two_pole", causal=9.0),
        ]
    )
    summary = build_method_metric_summary(
        df,
        model="gpt2-small",
        tasks=["emotion", "gender_en", "race"],
        methods=["gradiend:two_pole"],
        metrics=["causal_signed_effect"],
        aggregation="median",
    )
    assert float(summary.loc["gradiend:two_pole", "causal_signed_effect"]) == 0.2


def test_sae_pre_detection_summary_excludes_one_pole_only_tasks():
    df = pd.DataFrame(
        [
            _row("gpt2-small", "emotion", "sae_pre:k1", enc=0.8),
            _row("gpt2-small", "induction", "sae_pre:k1", enc=0.1),
        ]
    )
    df["detection_score"] = [0.8, 0.1]
    summary = build_method_metric_summary(
        df,
        model="gpt2-small",
        tasks=["emotion", "induction"],
        methods=["sae_pre:k1"],
        metrics=["detection_score"],
        specs={},
    )
    assert float(summary.loc["sae_pre:k1", "detection_score"]) == 0.8





def test_mixed_encoder_summary_excludes_causal_only_ridge():

    summary = pd.DataFrame(

        {

            "encoding_E": [0.7, None],

            "causal_signed_effect": [0.2, 0.03],

        },

        index=["actiend:two_pole", "actiend_ridge:two_pole"],

    )

    active = methods_with_encoder_summary_data(summary, list(summary.index))

    assert active == ["actiend:two_pole"]



def test_summary_table_bolds_column_winners():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.2, enc=0.9),

            _row("gpt2-small", "emotion", "sae:k1", causal=0.1, enc=0.5),

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.4, enc=1.0),

            _row("gpt2-small", "gender_en", "sae:k1", causal=0.3, enc=0.8),

        ]

    )

    methods = ["gradiend:two_pole", "sae:k1"]

    summary = build_method_metric_summary(

        df,

        model="gpt2-small",

        tasks=["emotion", "gender_en"],

        methods=methods,

        metrics=["encoding_E", "causal_signed_effect"],

    )

    tex = format_method_summary_latex(

        summary,

        model="gpt2-small",

        methods=methods,

        metrics=["encoding_E", "causal_signed_effect"],

    )

    assert r"\textbf{0.950}" in tex

    assert r"\textbf{0.300}" in tex

    assert r"\textbf{0.500}" not in tex





def test_metric_matrix_latex_methods_as_rows():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.218),

            _row("gpt2-small", "gender_en", "sae:k1", causal=0.110),

        ]

    )

    matrix = build_task_method_matrix(

        df,

        model="gpt2-small",

        tasks=["gender_en"],

        methods=["gradiend:two_pole", "sae:k1"],

        metric="causal_signed_effect",

        specs={},

    )

    tex = format_metric_matrix_latex(

        matrix,

        metric="causal_signed_effect",

        model="gpt2-small",

        tasks=["gender_en"],

        methods=["gradiend:two_pole", "sae:k1"],

        specs={},

    )

    assert "Gender" in tex

    assert "GRADIEND" in tex

    assert "0.218" in tex

    assert "0.110" in tex

    assert r"\caption{$\Delta P^{+}$}" in tex

    assert r"\begin{table*}[t]" in tex
    assert r"\scriptsize" in tex
    assert r"\setlength{\tabcolsep}{3pt}" in tex
    assert r"\renewcommand{\arraystretch}{1.05}" in tex
    assert r"Method & \rotatebox{45}{\taskGender}" in tex
    assert MEAN_LABEL not in tex





def test_row_winners_bold_in_latex():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.218, enc=0.9),

            _row("gpt2-small", "gender_en", "sae:k1", causal=0.110, enc=1.0),

        ]

    )

    methods = ["gradiend:two_pole", "sae:k1"]

    matrix = build_task_method_matrix(

        df,

        model="gpt2-small",

        tasks=["gender_en"],

        methods=methods,

        metric="encoding_E",

        specs={},

    )

    assert row_winner_methods(matrix, "gender_en", methods) == {"sae:k1"}

    tex = format_metric_matrix_latex(

        matrix,

        metric="encoding_E",

        model="gpt2-small",

        tasks=["gender_en"],

        methods=methods,

        specs={},

    )

    assert r"\textbf{1.000}" in tex

    assert r"\textbf{0.218}" not in tex





def test_matrix_includes_mean_row_and_column():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.2, enc=1.0),

            _row("gpt2-small", "emotion", "gradiend:two_pole", causal=0.0, enc=0.8),

        ]

    )

    matrix = build_task_method_matrix(

        df,

        model="gpt2-small",

        tasks=["gender_en", "emotion"],

        methods=["gradiend:two_pole"],

        metric="causal_signed_effect",

        specs={},

    )

    assert abs(float(matrix.loc["gender_en", MEAN_LABEL]) - 0.2) < 1e-9

    assert abs(float(matrix.loc["emotion", MEAN_LABEL]) - 0.0) < 1e-9

    assert abs(float(matrix.loc[MEAN_LABEL, "gradiend:two_pole"]) - 0.1) < 1e-9

    assert abs(float(matrix.loc[MEAN_LABEL, MEAN_LABEL]) - 0.1) < 1e-9





def test_fallback_footnote():

    prov = [{"task": "emotion", "method_group": "sae:k1", "fields": ["causal_signed_effect"]}]

    note = fallback_footnote(prov)

    assert "gpt2-small_old" in note





def test_paper_method_groups_include_sae_and_exclude_actiend_pre():

    assert "sae:k1" in PAPER_METHOD_GROUPS

    assert "sae:kstar" in PAPER_METHOD_GROUPS

    assert "sae_pre:k1" in PAPER_METHOD_GROUPS

    assert "sae_pre:kstar" in PAPER_METHOD_GROUPS

    # actiend_pre is off by default in training config, so its row was always
    # structurally empty -- it is not a paper column.
    assert "actiend_pre:two_pole" not in PAPER_METHOD_GROUPS

    assert "actiend_pre:one_pole" not in PAPER_METHOD_GROUPS





def test_methods_with_data_drops_empty_columns():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.2),

            _row("gpt2-small", "gender_en", "actiend_pre:two_pole", causal=None),

        ]

    )

    active = methods_with_data(

        df,

        model="gpt2-small",

        tasks=["gender_en"],

        methods=["gradiend:two_pole", "actiend_pre:two_pole", "sae:k1"],

        metric="causal_signed_effect",

    )

    assert active == ["gradiend:two_pole"]





def test_matrix_omits_empty_method_columns_from_latex():

    df = pd.DataFrame(

        [

            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.218),

        ]

    )

    methods = ["gradiend:two_pole", "actiend_pre:two_pole"]

    active = methods_with_data(

        df,

        model="gpt2-small",

        tasks=["gender_en"],

        methods=methods,

        metric="causal_signed_effect",

    )

    matrix = build_task_method_matrix(

        df,

        model="gpt2-small",

        tasks=["gender_en"],

        methods=active,

        metric="causal_signed_effect",

        specs={},

    )

    tex = format_metric_matrix_latex(

        matrix,

        metric="causal_signed_effect",

        model="gpt2-small",

        tasks=["gender_en"],

        methods=active,

        specs={},

    )

    assert "ACTIEND" not in tex

    assert "GRADIEND" in tex





def test_merge_injects_missing_sae_pre_row():

    cur = pd.DataFrame([_row("gpt2-small", "repetition", "gradiend:one_pole", causal=0.2)])

    fb = pd.DataFrame([_row("gpt2-small", "repetition", "sae_pre:kstar", causal=0.01)])

    merged, prov = merge_with_fallback(cur, fb, model="gpt2-small")

    pre = merged[merged["method_group"] == "sae_pre:kstar"]

    assert len(pre) == 1

    assert prov[0]["kind"] == "row"





def test_concatenated_metric_matrix_groups_metrics_and_excludes_lms():

    methods = ["gradiend:two_pole", "sae:k1"]
    matrices = {
        "roc_auc_neutral": pd.DataFrame(
            {"gradiend:two_pole": [0.8], "sae:k1": [0.9]}, index=["gender_en"]
        ),
        "detection_score": pd.DataFrame(
            {"gradiend:two_pole": [0.7], "sae:k1": [0.6]}, index=["gender_en"]
        ),
    }
    tex = format_concatenated_metric_matrices_latex(
        matrices,
        metrics=["roc_auc_neutral", "detection_score"],
        model="gpt2-small",
        tasks=["gender_en"],
        methods=methods,
        specs={},
        label="tab:concatenated",
        caption="Detection raw results",
    )

    assert r"\tiny" in tex
    assert r"\setlength{\tabcolsep}{1pt}" in tex
    assert r"\renewcommand{\arraystretch}{0.8}" in tex
    assert r"\begin{table*}[!t]" in tex
    assert r"\multicolumn{2}{c}{AUC$_n$}" in tex
    assert r"\multicolumn{2}{c}{$\mathrm{Det.}$}" in tex
    assert r"\textbf{0.900}" in tex
    assert r"\textbf{0.700}" in tex
    assert "LMS" not in tex


def test_concatenated_book_uses_the_matching_snippet(tmp_path):

    snippet = r"\begin{table}[!t]\begin{tabular}{lr}A & 1 \\ \end{tabular}\end{table}"
    (tmp_path / "summary_concatenated_detection_gpt2-small.tex").write_text(
        snippet, encoding="utf-8"
    )
    book = write_concatenated_tables_book(
        tmp_path,
        ["gpt2-small"],
        category="detection",
        filename="summary_tables_detection.tex",
    )
    assert r"\begin{table}[p]" in book.read_text(encoding="utf-8")


def test_tables_book_inlines_snippets_only():

    import tempfile

    from pathlib import Path

    snippet = (
        r"\begin{table}[!t]" "\n"
        r"  \centering" "\n"
        r"  \caption{$E$}" "\n"
        r"  \begin{tabular}{lr}" "\n"
        r"    A & 1 \\" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        figures = out / "figures"
        figures.mkdir()
        figure_name = "headline_scatter_gpt2-small_non_one_side_tasks.pdf"
        (figures / figure_name).write_bytes(b"placeholder")
        (out / "summary_methods_gpt2-small.tex").write_text(snippet, encoding="utf-8")
        (out / "summary_matrix_encoding_e_gpt2-small.tex").write_text(snippet, encoding="utf-8")
        book = write_tables_book(out, ["gpt2-small"], figure_dir=figures)
        text = book.read_text(encoding="utf-8")
    assert r"\documentclass" in text
    assert r"\usepackage{fontawesome7}" in text
    assert r"\begin{table}[p]" in text
    pdf = table_snippet_for_pdf(snippet)
    assert r"\begin{table}[p]" in pdf
    assert r"\resizebox{\textwidth}{!}{%" not in pdf
    assert r"\clearpage" in pdf
    assert rf"\detokenize{{figures/{figure_name}}}" in text
    assert figures.resolve().as_posix() not in text

    wide = snippet.replace("{lr}", "{lrrrrrrrrrrrr}")
    assert r"\resizebox{\textwidth}{!}{%" in table_snippet_for_pdf(wide)

    starred = snippet.replace(r"\begin{table}[!t]", r"\begin{table*}[t]").replace(
        r"\end{table}", r"\end{table*}"
    )
    assert r"\begin{table*}[p]" in table_snippet_for_pdf(starred)


def test_headline_book_includes_tables_and_available_scatter_figures(tmp_path):
    snippet = (
        r"\begin{table}[!t]" "\n"
        r"\begin{tabular}{lr} Method & Mean \\ \end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    (tmp_path / "summary_across_models_submetrics.tex").write_text(snippet, encoding="utf-8")
    (tmp_path / "summary_across_models_detection.tex").write_text(snippet, encoding="utf-8")
    (tmp_path / "summary_across_models_intervention.tex").write_text(snippet, encoding="utf-8")
    figures = tmp_path / "figures"
    figures.mkdir()
    figure_name = "headline_scatter_non_one_side_tasks_model_balanced.pdf"
    signal_figure_name = "headline_scatter_signal_mean.pdf"
    (figures / figure_name).write_bytes(b"placeholder")
    (figures / signal_figure_name).write_bytes(b"placeholder")
    book = write_headline_tables_book(tmp_path, figure_dir=figures)
    text = book.read_text(encoding="utf-8")
    assert r"\usepackage{fontawesome7}" in text
    assert text.count(r"\begin{table}[p]") == 3
    assert figure_name in text
    assert signal_figure_name in text
    assert text.index(signal_figure_name) < text.index(figure_name)
    assert "headline_scatter_non_one_side_tasks_by_model.pdf" not in text
    assert rf"\detokenize{{figures/{figure_name}}}" in text
    assert figures.resolve().as_posix() not in text


def test_diagnostics_are_excluded_from_results_book_and_written_separately(tmp_path):
    snippet = (
        r"\begin{table}[!t]" "\n"
        r"\begin{tabular}{lr} Status & Count \\ \end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    result_snippet = snippet.replace("Status & Count", "Method & Score")
    result = tmp_path / "summary_methods_gpt2-small.tex"
    result.write_text(result_snippet, encoding="utf-8")
    counts = tmp_path / "summary_convergence_counts_gpt2-small.tex"
    counts.write_text(snippet, encoding="utf-8")
    diagnostics = write_diagnostics_book(tmp_path, ["gpt2-small"])
    assert "Status & Count" in diagnostics.read_text(encoding="utf-8")
    book = write_tables_book(tmp_path, ["gpt2-small"])
    assert "Status & Count" not in book.read_text(encoding="utf-8")


def test_median_book_selects_median_summary_and_omits_mean_scatter(tmp_path):
    snippet = (
        r"\begin{table}[!t]" "\n"
        r"\begin{tabular}{lr} Method & Median \\ \end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    (tmp_path / "summary_methods_gpt2-small_median.tex").write_text(
        snippet, encoding="utf-8"
    )
    (tmp_path / "summary_matrix_detection_gpt2-small.tex").write_text(
        snippet.replace("Median", "Raw"), encoding="utf-8"
    )
    book = write_tables_book(
        tmp_path,
        ["gpt2-small"],
        filename="summary_tables_median.tex",
        aggregation="median",
    )
    text = book.read_text(encoding="utf-8")
    assert "Method & Median" in text
    assert "Method & Raw" in text
    assert "includegraphics" not in text


def test_tables_book_accepts_reusable_snippet_older_than_source():
    """A valid independently generated snippet remains usable across runs."""
    import os
    import tempfile

    from pathlib import Path

    snippet = (
        r"\begin{table}[!t]" "\n"
        r"  \centering" "\n"
        r"  \caption{$E$}" "\n"
        r"  \begin{tabular}{lr}" "\n"
        r"    A & 1 \\" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        stale = out / "summary_methods_gpt2-small.tex"
        stale.write_text(snippet, encoding="utf-8")
        # Simulate a matrix/artifact unchanged by the current regeneration.
        os.utime(stale, (1, 1))
        book = write_tables_book(out, ["gpt2-small"])
        assert book.is_file()


def test_failed_disk_dumps_still_enter_matrix_task_list():
    df = pd.DataFrame([_row("gpt2-small", "gender_en", "gradiend:two_pole", causal=0.1)])
    tasks = resolve_matrix_tasks(
        df,
        model="gpt2-small",
        tasks_arg="default",
        disk_tasks=["gender_en", "race", "religion", "race_one_pole"],
    )
    # Non-hidden disk tasks enter the list even though only gender_en is in df.
    assert "race" in tasks
    assert "religion" in tasks
    # ...but a study_hidden disk task (race_one_pole) is still dropped, so the
    # hidden-filter applies to disk-recovered tasks too, not just df rows.
    assert "race_one_pole" not in tasks


def test_proportional_rank_basic_ordering():
    df = pd.DataFrame(
        [
            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=None, enc=0.9),
            _row("gpt2-small", "gender_en", "sae:k1", causal=None, enc=0.5),
            _row("gpt2-small", "gender_en", "actiend:two_pole", causal=None, enc=0.1),
        ]
    )
    methods = ["gradiend:two_pole", "sae:k1", "actiend:two_pole"]
    ranks = compute_task_metric_ranks(
        df, model="gpt2-small", tasks=["gender_en"], methods=methods, metrics=["encoding_E"]
    )
    cell = ranks[("gender_en", "encoding_E")]
    assert cell["gradiend:two_pole"] == 1.0
    assert cell["sae:k1"] == 0.5
    assert cell["actiend:two_pole"] == 0.0


def test_proportional_rank_ties_share_average():
    df = pd.DataFrame(
        [
            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=None, enc=0.5),
            _row("gpt2-small", "gender_en", "sae:k1", causal=None, enc=0.5),
        ]
    )
    methods = ["gradiend:two_pole", "sae:k1"]
    ranks = compute_task_metric_ranks(
        df, model="gpt2-small", tasks=["gender_en"], methods=methods, metrics=["encoding_E"]
    )
    cell = ranks[("gender_en", "encoding_E")]
    assert cell["gradiend:two_pole"] == 0.5
    assert cell["sae:k1"] == 0.5


def test_proportional_rank_normalizes_across_varying_field_size():
    """A method that is 2nd-of-2 on one task and 2nd-of-4 on another should
    not average the same as if both were plain raw rank 2 -- 2nd-of-4 is a
    much better relative placement, and the mean-rank table exists precisely
    so one-pole/two-pole methods with different competing fields per task
    are comparable (see CLAUDE.md / user request 2026-08-20)."""
    df = pd.DataFrame(
        [
            # task1: only 2 methods have data; "sae:k1" comes in 2nd (raw rank 2 of 2).
            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=None, enc=0.9),
            _row("gpt2-small", "gender_en", "sae:k1", causal=None, enc=0.5),
            # task2: 4 methods have data; "sae:k1" again comes in 2nd (raw rank 2 of 4).
            _row("gpt2-small", "emotion", "actiend:two_pole", causal=None, enc=0.9),
            _row("gpt2-small", "emotion", "sae:k1", causal=None, enc=0.8),
            _row("gpt2-small", "emotion", "actiend_pre:two_pole", causal=None, enc=0.5),
            _row("gpt2-small", "emotion", "gradiend:one_pole", causal=None, enc=0.1),
        ]
    )
    methods = [
        "gradiend:two_pole",
        "sae:k1",
        "actiend:two_pole",
        "actiend_pre:two_pole",
        "gradiend:one_pole",
    ]
    summary = build_method_metric_rank_summary(
        df, model="gpt2-small", tasks=["gender_en", "emotion"], methods=methods, metrics=["encoding_E"]
    )
    # 2nd-of-2 -> 0.0; 2nd-of-4 -> 2/3. A raw-rank average would give (2+2)/2
    # unnormalized ambiguity; proportional distinguishes them.
    assert abs(float(summary.loc["sae:k1", "encoding_E"]) - (0.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_rank_summary_table_bolds_highest_as_best():
    df = pd.DataFrame(
        [
            _row("gpt2-small", "gender_en", "gradiend:two_pole", causal=None, enc=0.9),
            _row("gpt2-small", "gender_en", "sae:k1", causal=None, enc=0.5),
        ]
    )
    methods = ["gradiend:two_pole", "sae:k1"]
    summary = build_method_metric_rank_summary(
        df, model="gpt2-small", tasks=["gender_en"], methods=methods, metrics=["encoding_E"]
    )
    winners = column_winner_methods(summary, "encoding_E", methods)
    assert winners == {"gradiend:two_pole"}
    tex = format_method_summary_latex(
        summary,
        model="gpt2-small",
        methods=methods,
        metrics=["encoding_E"],
        caption="Summary (mean proportional placement; 1 = best)",
        decimals=2,
    )
    assert r"\textbf{1.00}" in tex
    assert r"\textbf{0.00}" not in tex
    assert r"\caption{Summary (mean proportional placement; 1 = best)}" in tex


def test_race_religion_labels_and_weaken_metric():
    from analysis.summary_latex import MATRIX_METRICS, METRIC_LATEX, TASK_LATEX

    assert TASK_LATEX["race"] == r"\taskRace"
    assert TASK_LATEX["race_one_pole"] == r"\taskRace"
    assert TASK_LATEX["religion"] == r"\taskReligion"
    assert TASK_LATEX["religion_one_pole"] == r"\taskReligion"
    assert "race" in PAPER_MULTICLASS_TASKS
    assert "religion" in PAPER_MULTICLASS_TASKS
    assert "race_one_pole" in PAPER_ONE_POLE_TASKS
    assert "causal_signed_effect_weaken" in MATRIX_METRICS
    assert "causal_weaken_lms" in MATRIX_METRICS
    assert "detection_score" in MATRIX_METRICS
    assert "intervention_score" in MATRIX_METRICS
    assert METRIC_LATEX["causal_signed_effect"] == r"$\Delta P^{+}$"
    assert METRIC_LATEX["causal_signed_effect_weaken"] == r"$\Delta P^{-}$"
    assert METRIC_LATEX["causal_lms"] == r"LMS$^{+}$"
    assert METRIC_LATEX["causal_weaken_lms"] == r"LMS$^{-}$"
    assert METRIC_LATEX["detection_score"] == r"$\mathrm{Det.}$"
    assert METRIC_LATEX["intervention_score"] == r"$\mathrm{Int.}$"


def test_detailed_summary_groups_raw_values_before_headlines():
    from analysis.summary_latex import MATRIX_METRICS

    summary = pd.DataFrame(
        [{metric: 0.5 for metric in MATRIX_METRICS}],
        index=["gradiend:two_pole"],
    )
    tex = format_method_summary_latex(
        summary,
        model="gpt2-small",
        methods=["gradiend:two_pole"],
        metrics=MATRIX_METRICS,
    )
    assert r"\multicolumn{5}{c}{Detection}" in tex
    assert r"\multicolumn{5}{c}{Intervention}" in tex
    header = next(line for line in tex.splitlines() if line.strip().startswith("Method &"))
    assert header.index(r"AUC$_n$") < header.index(r"$\mathrm{Det.}$")
    assert header.index(r"$\Delta P^{-}$") < header.index(r"$\mathrm{Int.}$")
    assert header.index(r"LMS$^{+}$") < header.index(r"LMS$^{-}$")


def test_result_views_include_one_sided_sae_as_pairwise_task_reference():
    methods = [
        "caa:two_pole",
        "caa:one_pole",
        "sae:k1",
        "gradiend:two_pole",
        "gradiend:one_pole",
    ]
    pairwise = result_view_methods(methods, "pairwise")
    one_pole = result_view_methods(methods, "one_pole")
    assert pairwise == ["caa:two_pole", "sae:k1", "gradiend:two_pole"]
    assert one_pole == ["caa:one_pole", "sae:k1", "gradiend:one_pole"]

    tasks = ["gender_en", "ioi", "repetition"]
    assert result_view_tasks(tasks, "pairwise") == ["gender_en"]
    assert result_view_tasks(tasks, "one_pole") == tasks


def test_paper_method_order_follows_signal_then_contrastive_before_iend():
    positions = {method: index for index, method in enumerate(PAPER_METHOD_GROUPS)}
    assert positions["cga:two_pole"] < positions["gradiend:two_pole"]
    assert positions["gradiend:two_pole"] < positions["caga:two_pole"]
    assert positions["caga:two_pole"] < positions["agiend:two_pole"]
    assert positions["agiend:two_pole"] < positions["caa:two_pole"]
    assert positions["caa:two_pole"] < positions["actiend:two_pole"]
    assert positions["actiend:two_pole"] < positions["sae:k1"]


def test_method_summary_uses_overridable_signal_and_method_rules():
    methods = ["cga:two_pole", "gradiend:two_pole", "caga:two_pole"]
    summary = pd.DataFrame({"encoding_E": [0.4, 0.5, 0.6]}, index=methods)
    tex = format_method_summary_latex(
        summary,
        model="gpt2-small",
        methods=methods,
        metrics=["encoding_E"],
    )
    assert r"\SummaryMethodRule{2}" in tex
    assert r"\SummarySignalRule{2}" in tex


def test_method_variants_do_not_get_inter_method_rules():
    methods = ["cga:two_pole", "cga_tensor_norm:two_pole", "gradiend:two_pole"]
    summary = pd.DataFrame({"encoding_E": [0.4, 0.5, 0.6]}, index=methods)
    tex = format_method_summary_latex(
        summary, model="gpt2-small", methods=methods, metrics=["encoding_E"]
    )
    assert tex.count(r"\SummaryMethodRule{2}") == 1

    sae_methods = ["sae:k1", "sae_pre:k1"]
    sae_summary = pd.DataFrame({"encoding_E": [0.4, 0.5]}, index=sae_methods)
    sae_tex = format_method_summary_latex(
        sae_summary, model="gpt2-small", methods=sae_methods, metrics=["encoding_E"]
    )
    assert r"\SummaryMethodRule" not in sae_tex
    assert r"\MethodSAE" in sae_tex
    assert r"\MethodSAEkOne" not in sae_tex
    assert r"\MethodSAEPrekOne" in sae_tex

    appendix_tex = format_method_summary_latex(
        sae_summary,
        model="gpt2-small",
        methods=sae_methods,
        metrics=["encoding_E"],
        method_set="full",
    )
    assert r"\MethodSAEkOne" in appendix_tex


def test_cross_model_table_marks_incomplete_means_with_star():
    summary = pd.DataFrame(
        {"gpt2-small": [0.8], "qwen3.5-9b-base": [0.9], "Mean": [0.85]},
        index=["gradiend:two_pole"],
    )
    coverage = [
        {
            "metric": "detection_score",
            "method_group": "gradiend:two_pole",
            "model": "gpt2-small",
            "applicable": True,
            "complete": True,
        },
        {
            "metric": "detection_score",
            "method_group": "gradiend:two_pole",
            "model": "qwen3.5-9b-base",
            "applicable": True,
            "complete": False,
        },
        {
            "metric": "detection_score",
            "method_group": "gradiend:two_pole",
            "model": "Mean",
            "applicable": True,
            "complete": False,
        },
    ]
    tex = format_cross_model_headline_latex(
        summary,
        coverage,
        metric="detection_score",
        sources=(("gpt2-small", "suite_full2"), ("qwen3.5-9b-base", "")),
        min_model_completion=0.9,
    )
    assert "GPT-2" in tex and "Qwen-3.5-9B" in tex
    assert "at least 90\\% overall headline coverage" in tex
    assert r"0.900\textsuperscript{*}" in tex
    assert r"0.850\textsuperscript{*}" in tex


