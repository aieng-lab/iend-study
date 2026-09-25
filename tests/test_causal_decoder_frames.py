import pandas as pd
from pathlib import Path

from study.decoder_frames import decoder_eval_frames_for_study


def test_one_pole_decoder_frames_keep_rival_factual_panel():
    labeled = pd.DataFrame(
        {
            "masked": [f"F {i}" for i in range(4)] + [f"M {i}" for i in range(4)],
            "label": ["she"] * 4 + ["he"] * 4,
            "label_class": ["F"] * 4 + ["M"] * 4,
            "split": ["validation"] * 8,
        }
    )
    neutral = pd.DataFrame(
        {"text": [f"neutral {i}" for i in range(4)], "split": ["validation"] * 4}
    )

    frame, neutral_frame = decoder_eval_frames_for_study(
        labeled,
        neutral,
        target_classes=["F", "M"],
        max_size=2,
        split="validation",
    )

    # The one-pole F feature still needs the M factual panel to measure P(F)
    # on the rival dataset; both are passed explicitly to evaluate_decoder.
    assert frame["label_class"].value_counts().to_dict() == {"F": 2, "M": 2}
    assert set(frame["label_class"]) == {"F", "M"}
    assert len(neutral_frame) == 2


def test_decoder_frames_alias_label_columns_to_package_factual_columns():
    """Package row-wise decoder scoring hard-requires 'factual'/'factual_id'/
    'alternative_id' (compute_probability_shift_score_row_wise); this study's
    frames carry the same data under 'label'/'label_class'/'alternative_class'
    (row-wise tasks like ravel_country) and must alias them, or evaluate_decoder
    fails with "Row-wise decoder eval requires column 'factual'"."""
    labeled = pd.DataFrame(
        {
            "masked": [f"F {i}" for i in range(4)] + [f"M {i}" for i in range(4)],
            "label": ["she"] * 4 + ["he"] * 4,
            "label_class": ["F"] * 4 + ["M"] * 4,
            "alternative": ["he"] * 4 + ["she"] * 4,
            "alternative_class": ["M"] * 4 + ["F"] * 4,
            "split": ["validation"] * 8,
        }
    )
    neutral = pd.DataFrame(
        {"text": [f"neutral {i}" for i in range(4)], "split": ["validation"] * 4}
    )

    frame, _ = decoder_eval_frames_for_study(
        labeled,
        neutral,
        target_classes=["F", "M"],
        max_size=2,
        split="validation",
    )

    assert (frame["factual"] == frame["label"]).all()
    assert (frame["factual_id"] == frame["label_class"]).all()
    assert (frame["alternative_id"] == frame["alternative_class"]).all()


def test_causal_study_passes_the_task_frames_to_every_decoder_grid():
    source = (Path(__file__).resolve().parents[1] / "causal_study.py").read_text(
        encoding="utf-8"
    )
    assert "training_like_df, decoder_neutral_df = _decoder_eval_frames(split)" in source
    assert "eval_kw.setdefault(\"training_like_df\", training_like_df)" in source
    assert "eval_kw.setdefault(\"neutral_df\", decoder_neutral_df)" in source
    assert "def _evaluate_decoder_split_clean" in source
    assert 'split="validation"' in source
    assert 'split="test"' in source
    assert "decoder_selected_learning_rates(" in source
    assert 'test_kw["lrs"] = frozen_lrs' in source
    assert 'test_kw["refine_points"] = 0' in source
    # Every production decoder-grid call goes through the split-clean helper;
    # the raw helper is used only by that helper's validation/test calls.
    assert source.count("_evaluate_decoder_split_clean(") >= 7
    assert source.count("_evaluate_decoder(") == 3


def test_study_decoder_wrappers_have_no_implicit_split_default():
    source = Path("causal_eval.py").read_text(encoding="utf-8")
    assert "evaluate_decoder_for_classes requires an explicit split" in source
    study_source = Path("causal_study.py").read_text(encoding="utf-8")
    assert "decoder evaluation requires an explicit split" in study_source


def test_selection_plot_is_validation_curve_plus_one_frozen_test_point():
    source = Path("plot_study.py").read_text(encoding="utf-8")
    assert '_safe_float(m.get("causal_selection_strength"))' in source
    assert 'label="validation-selected"' in source
    assert 'label="frozen test report"' in source
    assert "[s99, s99]" in source


def test_fail_fast_persists_partial_causal_results_before_abort():
    source = Path("causal_study.py").read_text(encoding="utf-8")
    assert "def _persist_partial_before_abort" in source
    assert source.count("_persist_partial_before_abort()") >= 15


def test_caa_parts_select_on_validation_and_report_once_on_test():
    source = (Path(__file__).resolve().parents[1] / "causal_study.py").read_text(
        encoding="utf-8"
    )
    caa_section = source[source.index("# --- CAA residual steering"):]
    assert "run_caa_causal_sweep(\n                                    model,\n                                    tokenizer,\n                                    val_rows," in caa_section
    assert "report_rows=meta_rows" in caa_section
    assert "base_probs=_caa_val_base_probs" in caa_section
    assert "r_base_probs=_caa_base_probs" in caa_section


def test_causal_protocol_repairs_v5_without_recomputing_strengthen():
    source = (
        Path(__file__).resolve().parents[1] / "study" / "stages" / "causal.py"
    ).read_text(encoding="utf-8")
    assert 'CAUSAL_PROTOCOL_VERSION = "decoder-bidirectional-val-select-test-v6-random"' in source
    assert "STRENGTHEN_COMPATIBLE_CAUSAL_PROTOCOLS" in source
    assert "reuse_strengthen_by_method" in source
    assert "strengthen_reused_method_ids" in source
    assert "ACTIEND_RIDGE_ID_PROTOCOL_VERSION = 2" in source
    assert "re-running only " in source
    assert 'startswith("actiend_ridge:")' in source
    assert "reuse_completed_causal" in source
    assert "protocol changed; re-running all causal sweeps" in source
