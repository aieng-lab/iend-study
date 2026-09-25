"""SAE all_k compact slices keep RAM bounded across layers."""

from __future__ import annotations

import numpy as np

from sae_eval import compact_all_k_layer_slices, sae_all_layers_class_vs_neutral_metrics


def test_compact_all_k_layer_slices_shape():
    n_test, n_val, n_neu = 20, 30, 40
    d = 32
    feats = {"A": [1, 5, 9], "B": [2, 7]}
    test = np.random.randn(n_test, d).astype(np.float32)
    val = np.random.randn(n_val, d).astype(np.float32)
    neu_t = np.random.randn(n_neu, d).astype(np.float32)
    neu_v = np.random.randn(n_neu // 2, d).astype(np.float32)
    compact = compact_all_k_layer_slices(
        test_latents=test,
        val_latents=val,
        neutral_test=neu_t,
        neutral_val=neu_v,
        test_labels=["A"] * 10 + ["B"] * 10,
        val_labels=["A"] * 15 + ["B"] * 15,
        features_by_class=feats,
        classes=["A", "B"],
        max_k=2,
    )
    a = compact["by_class"]["A"]
    assert a["test_cols"].shape == (n_test, 2)
    assert a["val_cols"].shape == (n_val, 2)
    assert a["neutral_test_cols"].shape == (n_neu, 2)
    assert a["feature_indices"] == [1, 5]


def test_all_k_metrics_from_compact_matches_full_latents():
    rng = np.random.default_rng(0)
    labels = np.array(["A", "A", "B", "B"])
    feats = [0, 2]
    lat = rng.normal(size=(4, 8)).astype(np.float32)
    neu = rng.normal(size=(6, 8)).astype(np.float32)
    full = sae_all_layers_class_vs_neutral_metrics(
        [
            {
                "layer": 0,
                "latents": lat,
                "neutral_latents": neu,
                "labels": labels,
                "features": feats,
            }
        ],
        target_class="A",
        k=2,
        target_classes=["A", "B"],
        n_bootstrap=0,
    )
    compact = sae_all_layers_class_vs_neutral_metrics(
        [
            {
                "layer": 0,
                "latent_cols": lat[:, feats],
                "neutral_cols": neu[:, feats],
                "labels": labels,
                "feature_indices": feats,
            }
        ],
        target_class="A",
        k=2,
        target_classes=["A", "B"],
        n_bootstrap=0,
    )
    assert "error" not in full and "error" not in compact
    assert full["roc_auc"] == compact["roc_auc"]
    assert full["balanced_accuracy"] == compact["balanced_accuracy"]


def test_all_k_metrics_keep_counterfactual_rivals_for_one_pole_tasks():
    """One-pole factual frames still report AUC_o/Excl from CF rival columns."""
    target = np.array([[4.0], [3.0]], dtype=np.float32)
    neutral = np.array([[0.0], [0.5]], dtype=np.float32)
    rival = np.array([[1.0], [1.5]], dtype=np.float32)
    result = sae_all_layers_class_vs_neutral_metrics(
        [
            {
                "layer": 0,
                "latent_cols": target,
                "neutral_cols": neutral,
                "labels": ["A", "A"],
                "feature_indices": [0],
                "rival_cols_by_class": {"B": rival},
            }
        ],
        target_class="A",
        k=1,
        target_classes=["A"],
        n_bootstrap=0,
    )
    assert result["roc_auc_other"] == 1.0
    assert result["class_exclusivity"] == 1.0


def test_all_k_metrics_mismatched_neutral_rows_across_layers_errors_cleanly():
    """A layer whose neutral row count drifts must not raise a raw np.stack ValueError.

    Regression for the multi-class SAE stage crash
    ("all input arrays must have the same shape"): only ``labels`` (target-class
    rows) was ever checked for cross-layer consistency, not the separately
    extracted neutral rows.
    """
    rng = np.random.default_rng(0)
    labels = np.array(["A", "A", "B", "B"])
    feats = [0, 2]
    lat = rng.normal(size=(4, 8)).astype(np.float32)

    result = sae_all_layers_class_vs_neutral_metrics(
        [
            {
                "layer": 0,
                "latents": lat,
                "neutral_latents": rng.normal(size=(6, 8)).astype(np.float32),
                "labels": labels,
                "features": feats,
            },
            {
                "layer": 1,
                "latents": lat,
                "neutral_latents": rng.normal(size=(5, 8)).astype(np.float32),
                "labels": labels,
                "features": feats,
            },
        ],
        target_class="A",
        k=2,
        target_classes=["A", "B"],
        n_bootstrap=0,
    )
    assert "error" in result
    assert "neutral" in result["error"]
