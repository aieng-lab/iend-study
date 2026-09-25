"""CAA/SAE neutrals: per-token scores, not document-mean."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from caa_eval import (
    cosine_score,
    caa_encoder_id,
    fit_caa_vectors,
    last_token_selector,
    nonpad_token_selector,
    resolve_act_selector,
    resolve_neutral_act_selector,
)
from sae_eval import (
    MAX_NEUTRAL_TOKEN_ROWS,
    NEUTRAL_TOKEN_SUBSAMPLE_SEED,
    SAE_CACHE_VERSION,
    class_vs_neutral_metrics,
    collect_study_neutral_activations,
    subsample_row_indices,
    subsample_rows,
)


def test_neutral_selector_is_per_token_not_mean():
    assert resolve_neutral_act_selector("prediction") is nonpad_token_selector
    assert resolve_neutral_act_selector("pre_prediction") is nonpad_token_selector
    assert resolve_neutral_act_selector("unfilled_prediction") is nonpad_token_selector
    assert resolve_neutral_act_selector("mean") == "all"
    assert resolve_neutral_act_selector("last") is last_token_selector
    assert resolve_act_selector("prediction") == "prediction"


def test_nonpad_token_selector_flattens_mask():
    act = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    out = nonpad_token_selector(act, {"attention_mask": mask})
    assert out.shape == (5, 3)
    assert torch.equal(out[0], act[0, 0])
    assert torch.equal(out[3], act[1, 0])
    assert torch.equal(out[4], act[1, 1])


def test_mean_pool_inflates_class_vs_neutral_auc():
    """Document-mean cosine hides a token that already points along v."""
    v = np.array([1.0, 0.0])
    class_scores = np.array([0.95, 0.90, 0.85])
    token_acts = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    per_token = cosine_score(token_acts, v)
    pooled = cosine_score(token_acts.mean(axis=0, keepdims=True), v)
    pooled = np.concatenate([pooled, pooled], axis=0)
    auc_pool = class_vs_neutral_metrics(
        class_scores, pooled, target_class="M"
    )["roc_auc"]
    auc_tok = class_vs_neutral_metrics(
        class_scores, per_token, target_class="M"
    )["roc_auc"]
    assert auc_pool is not None and auc_tok is not None
    assert float(auc_pool) > float(auc_tok)
    assert abs(float(pooled[0])) < 0.05
    assert float(per_token[0]) > 0.99


def test_sae_cache_dir_is_per_layer_and_site():
    from pathlib import Path

    from sae_eval import sae_cache_dir

    root = Path("runs") / "x"
    assert sae_cache_dir(root, layer=0) == root / "sae_cache" / "L0"
    assert sae_cache_dir(root, layer=0, act_policy="prediction") == (
        root / "sae_cache" / "L0_prediction"
    )
    assert sae_cache_dir(root, layer=0, act_policy="pre_prediction") == (
        root / "sae_cache" / "L0_pre_prediction"
    )
    assert sae_cache_dir(root, layer=0, act_policy="prediction") != sae_cache_dir(
        root, layer=0, act_policy="pre_prediction"
    )


def test_sae_cache_version_per_token_neutrals():
    assert SAE_CACHE_VERSION >= 6


def test_subsample_rows_is_deterministic():
    x = np.arange(20).reshape(10, 2)
    a = subsample_rows(x, 4, seed=3)
    b = subsample_rows(x, 4, seed=3)
    assert a.shape == (4, 2)
    assert np.array_equal(a, b)
    assert np.array_equal(subsample_rows(x, 50, seed=0), x)


def test_sae_neutral_subsample_seed_aligned_across_layers():
    """all_k sums per-token scores; layer-dependent seeds would mix different tokens."""
    n = 50
    cap = 8
    a = subsample_row_indices(n, cap, seed=NEUTRAL_TOKEN_SUBSAMPLE_SEED)
    b = subsample_row_indices(n, cap, seed=NEUTRAL_TOKEN_SUBSAMPLE_SEED)
    assert a is not None and np.array_equal(a, b)
    assert not np.array_equal(a, subsample_row_indices(n, cap, seed=11))


def test_sae_collect_neutral_uses_caa_per_token_extractor(monkeypatch):
    import caa_eval as ce

    seen = {}

    def fake_extract(model, tokenizer, texts, **kw):
        seen["policy"] = kw.get("policy")
        seen["n_texts"] = len(list(texts))
        return np.zeros((9, 4))

    monkeypatch.setattr(ce, "_extract_neutral_layer_acts", fake_extract)
    out = collect_study_neutral_activations(
        object(), object(), ["a", "b", "c"], layer=3, act_policy="prediction"
    )
    assert seen["policy"] == "prediction"
    assert seen["n_texts"] == 3
    assert out.shape == (9, 4)


def test_caa_neutral_token_cap_matches_sae():
    assert MAX_NEUTRAL_TOKEN_ROWS == 16384


def test_caa_fits_all_pair_endpoints_and_one_poles(monkeypatch):
    import caa_eval as ce

    frame = pd.DataFrame(
        {
            "masked": ["a", "b", "c"],
            "label": ["a", "b", "c"],
            "label_class": ["A", "B", "C"],
            "split": ["train", "train", "train"],
        }
    )
    acts = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    monkeypatch.setattr(ce, "_make_extractor", lambda *args, **kwargs: object())
    monkeypatch.setattr(ce, "_gender_fill_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        ce,
        "extract_activations_batched",
        lambda *args, **kwargs: acts.copy(),
    )

    fitted = fit_caa_vectors(
        object(),
        object(),
        frame,
        target_classes=["A", "B", "C"],
        layers=[0],
        policies=["prediction"],
        hf_resid_template="layer.{layer}",
    )["prediction"]

    assert len(fitted.contrast_meta) == 9  # 3 pairs x 2 endpoints + 3 one-poles
    assert fitted.contrast_meta["pair:A-B:A"]["rival_classes"] == ["B"]
    assert fitted.contrast_meta["one_pole:A"]["rival_classes"] == ["B", "C"]
    assert np.allclose(
        fitted.layer_vec("pair:A-B:A", 0),
        np.asarray([1.0, -1.0]) / np.sqrt(2.0),
    )
    assert caa_encoder_id("A", policy="prediction", pair=["A", "B"]) == (
        "caa:A-B:A:act_prediction"
    )
