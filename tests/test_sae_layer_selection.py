from __future__ import annotations

import study.sae_engine as poc


def test_select_sae_layers_uses_target_classes_not_bare_classes_name():
    poc.TARGET_CLASSES = ("MATCH",)
    by_layer = {
        0: {
            "selections": {
                "per_class": {
                    "features_by_class": {"MATCH": [1]},
                    "scores_by_class": {"MATCH": {1: 0.5}},
                }
            }
        }
    }
    selected, global_layer, scores = poc._select_sae_layers(
        by_layer, selection_mode="per_class", classes=["MATCH"]
    )
    assert selected == {"MATCH": 0}
    assert global_layer == 0
    assert scores["MATCH"] == 0.5

    readouts, curves, fbc, sel = poc._assemble_class_headline(
        by_layer,
        selected,
        readout_key="per_class_by_class",
        selection_mode="per_class",
        layer_selection="test",
        classes=["MATCH"],
    )
    assert "MATCH" in readouts
    assert fbc["MATCH"] == [1]


def test_select_sae_layers_locks_by_validation_detection_not_rank():
    poc.TARGET_CLASSES = ("MATCH",)
    by_layer = {
        0: {
            "selections": {
                "per_class": {
                    "features_by_class": {"MATCH": [1]},
                    "scores_by_class": {"MATCH": {1: 9.0}},
                }
            },
            "readouts": {
                "per_class_by_class": {
                    "MATCH": {
                        "roc_auc_neutral": 0.55,
                        "neutral_specificity": 0.55,
                        "val_readout": {
                            "roc_auc_neutral": 0.55,
                            "neutral_specificity": 0.55,
                        },
                    }
                }
            },
        },
        1: {
            "selections": {
                "per_class": {
                    "features_by_class": {"MATCH": [2]},
                    "scores_by_class": {"MATCH": {2: 0.1}},
                }
            },
            "readouts": {
                "per_class_by_class": {
                    "MATCH": {
                        "roc_auc_neutral": 0.95,
                        "neutral_specificity": 0.9,
                        "val_readout": {
                            "roc_auc_neutral": 0.95,
                            "neutral_specificity": 0.9,
                        },
                    }
                }
            },
        },
    }
    selected, global_layer, scores = poc._select_sae_layers(
        by_layer, selection_mode="per_class", classes=["MATCH"]
    )
    assert selected == {"MATCH": 1}
    assert global_layer == 1
    assert scores["MATCH"] == 0.9


def test_select_sae_layers_can_pick_all():
    poc.TARGET_CLASSES = ("MATCH",)
    by_layer = {
        0: {
            "selections": {
                "per_class": {
                    "features_by_class": {"MATCH": [1]},
                    "scores_by_class": {"MATCH": {1: 1.0}},
                }
            },
            "readouts": {
                "per_class_by_class": {
                    "MATCH": {
                        "roc_auc_neutral": 0.5,
                        "neutral_specificity": 0.5,
                        "val_readout": {
                            "roc_auc_neutral": 0.5,
                            "neutral_specificity": 0.5,
                        },
                    }
                }
            },
        },
        "all": {
            "selections": {
                "per_class": {
                    "features_by_class": {"MATCH": ["all"]},
                    "scores_by_class": {},
                }
            },
            "readouts": {
                "per_class_by_class": {
                    "MATCH": {
                        "roc_auc_neutral": 0.99,
                        "neutral_specificity": 0.99,
                        "val_readout": {
                            "roc_auc_neutral": 0.99,
                            "neutral_specificity": 0.99,
                        },
                    }
                }
            },
        },
    }
    selected, global_layer, scores = poc._select_sae_layers(
        by_layer, selection_mode="per_class", classes=["MATCH"]
    )
    assert selected == {"MATCH": "all"}
    assert global_layer == "all"
    assert abs(scores["MATCH"] - 0.99) < 1e-12

    readouts, _curves, fbc, _sel = poc._assemble_class_headline(
        by_layer,
        selected,
        readout_key="per_class_by_class",
        selection_mode="per_class",
        layer_selection="encoding_E",
        classes=["MATCH"],
    )
    assert readouts["MATCH"]["sae_layer"] == "all"
    assert fbc["MATCH"] == ["all"]
    assert poc._top1_rank_score(by_layer["all"]["selections"]["per_class"], "MATCH") is None


def test_localization_skips_all_layers_site():
    import tempfile
    from pathlib import Path

    from localization_eval import _optional_int_layer, run_localization_analysis

    assert _optional_int_layer("all") is None
    assert _optional_int_layer(7) == 7
    sae_raw = {
        "selections": {"per_class": {"features_by_class": {"M": [3]}}},
        "selected_layer_by_class": {"M": "all"},
        "selected_layer_global": "all",
        "readouts": {"per_class_by_class": {"M": {"readout_k": 1}}},
        "_sae_by_layer": {},
    }
    with tempfile.TemporaryDirectory() as td:
        out = run_localization_analysis(
            {}, sae_raw, output_dir=Path(td), target_classes=["M"]
        )
        assert not out.get("error")
        dumped = (Path(td) / "localization" / "sae_resid_channels.json").read_text(
            encoding="utf-8"
        )
    assert "all_layers_site" in dumped


def test_sae_method_rows_skip_non_int_layer_keys():
    poc.TARGET_CLASSES = ("M",)
    rows = poc._sae_method_rows_for_site(
        {
            "selections": {"per_class": {"features_by_class": {"M": [1]}}},
            "readouts": {"per_class_by_class": {"M": {"readout_k": 1, "sae_layer": "all"}}},
            "selected_layer_by_class": {"M": "all"},
            "by_layer_readouts": {
                "all": {"M": {"roc_auc": 0.9}},
                7: {"M": {"roc_auc": 0.8}},
            },
        },
        site="prediction",
    )
    ids = [r["method"] for r in rows]
    assert "sae:M:L7" in ids
    assert not any(":Lall" in str(i) for i in ids)
