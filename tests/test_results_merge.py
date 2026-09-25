from __future__ import annotations

from study.config import parse_methods_arg
from study.results_merge import merge_method_rows, merge_raw_sections, merge_study_payload


def test_merge_deduplicates_repeated_errors():
    previous = {"methods": [], "raw": {}, "errors": ["old", "old"]}
    current = {"methods": [], "raw": {}, "errors": ["old", "new", "new"]}
    merged = merge_study_payload(previous, current, enabled={"causal"})
    assert merged["errors"] == ["old", "new"]


def test_load_previous_results_auto_compacts_pathological_file(tmp_path, monkeypatch):
    import study.results_merge as results_merge

    path = tmp_path / "results.json"
    path.write_text(
        '{\n  "methods": [],\n  "raw": {},\n  "errors": [\n'
        '    "same",\n    "same",\n    "new"\n  ],\n  "status": "partial"\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(results_merge, "_AUTO_COMPACT_RESULTS_BYTES", 1)
    loaded = results_merge.load_previous_results(tmp_path)
    assert loaded is not None
    assert loaded["errors"] == ["same", "new"]


def test_targeted_tensor_refresh_drops_stale_tensor_causal_only():
    from study.results_merge import _merge_causal_raw

    previous = {
        "by_method": {
            "actiend:F-M:F:tok_all": {"selected_strength": 1.0, "meta": {}},
            "actiend:F-M:tensors:F:tok_all": {
                "selected_strength": 2.0,
                "meta": {},
            },
        },
        "poc_decoder_panels": {
            "actiend:F-M:F:tok_all": {"ok": True},
            "actiend:F-M:tensors:F:tok_all": {"ok": True},
        },
        "summaries": [
            {"method": "actiend:F-M:F:tok_all", "meta": {}},
            {"method": "actiend:F-M:tensors:F:tok_all", "meta": {}},
        ],
    }
    new = {"replace_train_splits": ["tensors"], "by_method": {}}
    merged = _merge_causal_raw(previous, new, enabled={"actiend", "causal"})

    assert "actiend:F-M:F:tok_all" in merged["by_method"]
    assert "actiend:F-M:tensors:F:tok_all" not in merged["by_method"]
    assert "actiend:F-M:tensors:F:tok_all" not in merged["poc_decoder_panels"]
    assert [row["method"] for row in merged["summaries"]] == [
        "actiend:F-M:F:tok_all"
    ]


def test_concurrent_tensor_writer_keeps_fresh_none_row_from_locked_read():
    """A late tensor writer must not restore its stale startup copy of none."""
    fresh_previous = {
        "methods": [
            {"method": "actiend:F-M:F", "status": "ok", "metrics": {"roc_auc": 0.97}},
            {"method": "actiend:F-M:tensors:F", "status": "ok", "metrics": {"roc_auc": 0.40}},
        ],
        "raw": {},
        "config": {"enabled_methods": ["actiend"]},
        "errors": [],
    }
    stale_tensor_process_payload = {
        "methods": [
            # Preserved when this process started, before the none writer landed.
            {"method": "actiend:F-M:F", "status": "ok", "metrics": {"roc_auc": 0.55}},
            {"method": "actiend:F-M:tensors:F", "status": "ok", "metrics": {"roc_auc": 0.99}},
        ],
        "raw": {"train": {"requested_splits": ["tensors"]}},
        "config": {"enabled_methods": ["actiend"]},
        "errors": [],
    }

    merged = merge_study_payload(
        fresh_previous,
        stale_tensor_process_payload,
        enabled={"actiend", "causal"},
    )
    by_id = {row["method"]: row for row in merged["methods"]}
    assert by_id["actiend:F-M:F"]["metrics"]["roc_auc"] == 0.97
    assert by_id["actiend:F-M:tensors:F"]["metrics"]["roc_auc"] == 0.99


def test_parse_methods_arg_flattens_argparse_append():
    assert parse_methods_arg([["gradiend"], ["actiend"]]) == ["gradiend", "actiend"]
    assert parse_methods_arg(["gradiend,actiend"]) == ["gradiend", "actiend"]
    assert parse_methods_arg(["gradiend", "actiend"]) == ["gradiend", "actiend"]


def test_failed_sae_stub_does_not_wipe_prior_class_rows():
    previous = [
        {"method": "sae:M:kstar", "status": "ok", "metrics": {"roc_auc_neutral": 0.9}},
        {"method": "sae:F:k1", "status": "ok", "metrics": {"roc_auc_neutral": 0.8}},
        {"method": "gradiend:M", "status": "ok", "metrics": {"roc_auc_neutral": 0.7}},
    ]
    new_rows = [
        {
            "method": "sae",
            "status": "error",
            "error": "name 'label_tokens_from_config' is not defined",
            "metrics": {},
        }
    ]
    merged = merge_method_rows(previous, new_rows, enabled={"sae", "causal"})
    ids = {r["method"] for r in merged}
    assert "sae:M:kstar" in ids
    assert "sae:F:k1" in ids
    assert "gradiend:M" in ids
    assert "sae" not in ids  # stub error dropped when priors kept


def test_successful_sae_rerun_replaces_prior_family():
    previous = [
        {"method": "sae:M:kstar", "status": "ok", "metrics": {"roc_auc_neutral": 0.5}},
    ]
    new_rows = [
        {"method": "sae:M:kstar", "status": "ok", "metrics": {"roc_auc_neutral": 0.99}},
        {"method": "sae:F:kstar", "status": "ok", "metrics": {"roc_auc_neutral": 0.98}},
    ]
    merged = merge_method_rows(previous, new_rows, enabled={"sae"})
    by = {r["method"]: r for r in merged}
    assert abs(by["sae:M:kstar"]["metrics"]["roc_auc_neutral"] - 0.99) < 1e-9
    assert "sae:F:kstar" in by


def test_failed_causal_does_not_wipe_prior_ok_entry():
    from study.results_merge import _merge_causal_raw

    previous = {
        "by_method": {
                "sae:IO:k1": {
                    "strengths": [{"strength": 1.0}],
                    "selected_strength": 1.0,
                    "weaken_strengths": [{"strength": 2.0}],
                    "weaken_selected_strength": 2.0,
                    "weaken_selected": {"strength": 2.0},
                    "meta": {},
            },
            "gradiend:IO": {
                "strengths": [],
                "selected_strength": None,
                "meta": {"error": "IO"},
            },
        }
    }
    new = {
        "by_method": {
            "sae:IO:k1": {
                "strengths": [],
                "selected_strength": None,
                "meta": {"error": "IO"},
            },
                "gradiend:IO": {
                    "strengths": [{"strength": 10.0}],
                    "selected_strength": 10.0,
                    "weaken_strengths": [{"strength": 20.0}],
                    "weaken_selected_strength": 20.0,
                    "weaken_selected": {"strength": 20.0},
                    "meta": {},
            },
        }
    }
    merged = _merge_causal_raw(previous, new, enabled={"sae", "gradiend", "causal"})
    assert merged["by_method"]["sae:IO:k1"]["selected_strength"] == 1.0
    assert merged["by_method"]["gradiend:IO"]["selected_strength"] == 10.0


def test_ok_method_row_not_replaced_by_error():
    previous = [
        {"method": "sae:IO:k1", "status": "ok", "metrics": {"causal_signed_effect": 0.01}},
    ]
    new_rows = [
        {"method": "sae:IO:k1", "status": "error", "error": "IO", "metrics": {}},
    ]
    merged = merge_method_rows(previous, new_rows, enabled={"sae", "causal"})
    assert merged[0]["status"] == "ok"
    assert merged[0]["metrics"]["causal_signed_effect"] == 0.01


def test_causal_refresh_merges_shared_id_without_erasing_encoder_metrics():
    previous = [
        {
            "method": "sae:IO:k1",
            "status": "ok",
            "metrics": {
                "roc_auc_neutral": 0.99,
                "balanced_accuracy": 0.95,
            },
            "artifacts": {"sae_layer": 11},
        }
    ]
    new_rows = [
        {
            "method": "sae:IO:k1",
            "status": "ok",
            "metrics": {
                "causal_signed_effect": 0.12,
                "causal_lms_ok": True,
            },
            "artifacts": {"modified_model": "model.pt"},
        }
    ]

    merged = merge_method_rows(
        previous,
        new_rows,
        enabled={"actiend", "causal", "localization"},
    )

    assert len(merged) == 1
    metrics = merged[0]["metrics"]
    assert metrics["roc_auc_neutral"] == 0.99
    assert metrics["balanced_accuracy"] == 0.95
    assert metrics["causal_signed_effect"] == 0.12
    assert metrics["causal_lms_ok"] is True
    assert merged[0]["artifacts"] == {
        "sae_layer": 11,
        "modified_model": "model.pt",
    }


def test_caga_causal_shell_preserves_encoder_metrics_and_family_inventory():
    previous = [
        {
            "method": "caga:asian",
            "status": "ok",
            "metrics": {
                "target_class": "asian",
                "readout_kind": "one_pole",
                "roc_auc_neutral": 1.0,
                "class_exclusivity": 0.99,
            },
            "extras": {"experiment_dir": "artifacts/caga__onepole__asian"},
        },
        {
            "method": "caga:black",
            "status": "ok",
            "metrics": {
                "target_class": "black",
                "readout_kind": "one_pole",
                "roc_auc_neutral": 0.98,
            },
        },
    ]
    new_rows = [
        {
            "method": "caga:asian",
            "status": "ok",
            "metrics": {
                "target_class": "asian",
                "readout_kind": "causal_only",
            },
            "extras": {
                "causal": {
                    "method": "caga:asian",
                    "selected_strength": 10.0,
                    "selected": {"signed_effect": 0.28},
                }
            },
        }
    ]

    merged = merge_method_rows(
        previous,
        new_rows,
        enabled={"caga", "causal"},
    )
    by_id = {row["method"]: row for row in merged}

    # The untouched class remains because a causal-only writer is not an
    # authoritative replacement for the encoder family inventory.
    assert by_id["caga:black"]["metrics"]["roc_auc_neutral"] == 0.98
    # The matching causal shell complements rather than replaces its encoder.
    asian = by_id["caga:asian"]
    assert asian["metrics"]["roc_auc_neutral"] == 1.0
    assert asian["metrics"]["class_exclusivity"] == 0.99
    assert asian["metrics"]["readout_kind"] == "one_pole"
    assert asian["extras"]["causal"]["selected"]["signed_effect"] == 0.28


def test_new_encoder_rerun_remains_authoritative_for_same_method_id():
    previous = [
        {
            "method": "sae:IO:k1",
            "status": "ok",
            "metrics": {
                "roc_auc_neutral": 0.60,
                "causal_signed_effect": 0.20,
            },
        }
    ]
    new_rows = [
        {
            "method": "sae:IO:k1",
            "status": "ok",
            "metrics": {"roc_auc_neutral": 0.98},
        }
    ]

    merged = merge_method_rows(previous, new_rows, enabled={"sae"})

    assert merged[0]["metrics"] == {"roc_auc_neutral": 0.98}


def test_partial_train_refresh_preserves_other_backend_inventory():
    previous = {
        "train": {
            "n_jobs": 2,
            "backends": ["gradiend"],
            "tensor_backends": ["gradiend"],
            "none_by_class": {"gradiend": ["IO"]},
            "tensors_by_class": {"gradiend": ["IO"]},
        }
    }
    current = {
        "train": {
            "n_jobs": 1,
            "backends": ["actiend"],
            "tensor_backends": [],
            "none_by_class": {"actiend": ["IO"]},
            "tensors_by_class": {},
        }
    }

    merged = merge_raw_sections(previous, current, enabled={"actiend"})

    assert merged["train"]["backends"] == ["gradiend", "actiend"]
    assert merged["train"]["tensor_backends"] == ["gradiend"]
    assert merged["train"]["none_by_class"] == {
        "gradiend": ["IO"],
        "actiend": ["IO"],
    }
    assert merged["train"]["tensors_by_class"] == {"gradiend": ["IO"]}


def test_method_group_pivot_stable_order():
    import pandas as pd

    from analysis.method_groups import pivot_group_task

    df = pd.DataFrame(
        [
            {"model": "gpt2-small", "task": "gender_en", "method_group": "sae:kstar", "roc_auc_neutral": 0.99},
            {"model": "gpt2-small", "task": "gender_en", "method_group": "caa:act_prediction", "roc_auc_neutral": 0.5},
            {"model": "gpt2-small", "task": "gender_en", "method_group": "gradiend:two_pole", "roc_auc_neutral": 0.8},
            {"model": "gpt2-small", "task": "gender_en", "method_group": "actiend:one_pole", "roc_auc_neutral": 0.7},
            {"model": "gpt2-small", "task": "race", "method_group": "sae:k1", "roc_auc_neutral": 0.95},
            {"model": "gpt2-small", "task": "race", "method_group": "gradiend:one_pole", "roc_auc_neutral": 0.6},
        ]
    )
    pivot = pivot_group_task(df, metric="roc_auc_neutral", model="gpt2-small")
    assert list(pivot.index) == [
        "gradiend:two_pole",
        "gradiend:one_pole",
        "actiend:one_pole",
        "sae:kstar",
        "sae:k1",
        "caa:act_prediction",
    ]


def test_family_overview_row_order():
    from analysis.family_overview import FAMILIES, format_overview_text

    cells = [
        {"model": "gpt2-small", "task": "gender_en", "family": "caa", "metric": "encoding_E", "mean": 0.9, "min": 0.9, "max": 0.9, "n_classes": 1},
        {"model": "gpt2-small", "task": "gender_en", "family": "sae", "metric": "encoding_E", "mean": 0.8, "min": 0.8, "max": 0.8, "n_classes": 1},
        {"model": "gpt2-small", "task": "gender_en", "family": "gradiend", "metric": "encoding_E", "mean": 0.7, "min": 0.7, "max": 0.7, "n_classes": 1},
        {"model": "gpt2-small", "task": "gender_en", "family": "actiend", "metric": "encoding_E", "mean": 0.6, "min": 0.6, "max": 0.6, "n_classes": 1},
    ]
    text = format_overview_text(cells, metric="encoding_E", model="gpt2-small")
    rows = [ln.split()[0] for ln in text.splitlines() if ln.split() and ln.split()[0] in FAMILIES]
    assert rows == ["gradiend", "actiend", "actiend_ridge", "actiend_pre", "sae", "sae_pre", "caa"]
