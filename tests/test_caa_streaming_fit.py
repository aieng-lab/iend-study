"""Streaming CAA fit must match the materialized (N, d) path exactly.

``stream_caa_contrast_vectors`` folds CAA's mean-diff into O(d) accumulators so
large-model CAA never holds the full ``{layer: (N, d)}`` host matrix (the source
of the llama-3.1-8b OUT_OF_MEMORY). CAA's direction is a mean, so the streaming
result must equal the reference ``extract_activations_by_layer_batched`` +
``_paired_mean_diff`` build up to floating-point reduction tolerance -- including
the per-pair first-``pos``/first-``neg`` row selection for the paired path.
"""

from types import SimpleNamespace

import numpy as np
import torch

from caa_eval import (
    _paired_mean_diff,
    extract_activations_by_layer_batched,
    normalize_vec,
    stream_caa_contrast_vectors,
)


class _FakeExtractor:
    """Deterministic per-layer activations keyed by row index.

    Shared by both paths so any difference is the aggregation, not the capture.
    """

    def __init__(self, acts_by_layer):
        self._acts = {int(L): np.asarray(a, dtype=np.float32) for L, a in acts_by_layer.items()}
        self._names = {int(L): f"m{int(L)}" for L in self._acts}
        self._module_items = [(self._names[L], None) for L in sorted(self._acts)]
        self.signal = SimpleNamespace(options={"token_selector": "prediction"})

    def _capture(self, sub):
        idx = sub["_idx"]
        idx = idx.tolist() if torch.is_tensor(idx) else list(idx)
        captured = {name: name for name in self._names.values()}
        return captured, idx

    def _flatten_site(self, captured_val, prepared, selector=None):
        L = int(str(captured_val)[1:])
        return torch.from_numpy(self._acts[L][np.asarray(prepared, dtype=int)])


def _batch(n):
    return {"input_ids": torch.arange(n).reshape(n, 1), "_idx": torch.arange(n)}


def _reference(acts_by_layer, labels, pair_ids, contrast_defs, layers):
    extractor = _FakeExtractor(acts_by_layer)
    by_layer = extract_activations_by_layer_batched(
        extractor, _batch(len(labels)), layers=layers, batch_size=3
    )
    layer_map, concat = {}, {}
    for vk, cls, rivals, _abl, _pair in contrast_defs:
        other = rivals[0] if len(rivals) == 1 else rivals
        parts = []
        lm = {}
        for L in layers:
            v = _paired_mean_diff(by_layer[L], labels, pair_ids, class_pos=cls, class_neg=other)
            lm[L] = v
            parts.append(v)
        layer_map[vk] = lm
        concat[vk] = normalize_vec(np.concatenate(parts, axis=0))
    return layer_map, concat


def _assert_matches(acts_by_layer, labels, pair_ids, contrast_defs, layers):
    ref_layer, ref_concat = _reference(acts_by_layer, labels, pair_ids, contrast_defs, layers)
    stream_layer, stream_concat = stream_caa_contrast_vectors(
        _FakeExtractor(acts_by_layer),
        _batch(len(labels)),
        layers=layers,
        labels=labels,
        pair_ids=pair_ids,
        contrast_defs=contrast_defs,
        batch_size=3,
    )
    for vk in ref_concat:
        np.testing.assert_allclose(stream_concat[vk], ref_concat[vk], rtol=1e-5, atol=1e-7)
        for L in layers:
            np.testing.assert_allclose(stream_layer[vk][L], ref_layer[vk][L], rtol=1e-5, atol=1e-7)


def test_streaming_matches_paired_and_one_pole_contrasts():
    rng = np.random.default_rng(0)
    n, d = 12, 5
    layers = [0, 1]
    acts = {L: rng.standard_normal((n, d)) for L in layers}
    # 6 contiguous pairs, alternating F/M -- the study's own pair layout.
    labels = ["F", "M"] * 6
    pair_ids = [i // 2 for i in range(n)]
    contrast_defs = [
        ("pair:F-M:F", "F", ["M"], "pair", ["F", "M"]),
        ("pair:F-M:M", "M", ["F"], "pair", ["F", "M"]),
        ("one_pole:F", "F", ["M"], "one_pole", None),
    ]
    _assert_matches(acts, labels, pair_ids, contrast_defs, layers)


def test_streaming_honors_first_row_selection_in_duplicate_role_pairs():
    """A pair with two ``F`` rows and one ``M`` row: the paired path uses only
    the FIRST ``F`` per pair, so per-class mean would differ -- streaming must
    reproduce the first-row selection, not fall back to a class mean."""
    rng = np.random.default_rng(1)
    d = 4
    layers = [0]
    # pair 0: F(row0), F(row1), M(row2)  -> uses row0 - row2
    # pair 1: F(row3), M(row4)           -> uses row3 - row4
    acts = {0: rng.standard_normal((5, d))}
    labels = ["F", "F", "M", "F", "M"]
    pair_ids = [0, 0, 0, 1, 1]
    contrast_defs = [("pair:F-M:F", "F", ["M"], "pair", ["F", "M"])]
    _assert_matches(acts, labels, pair_ids, contrast_defs, layers)
    # And confirm this is a non-trivial case: the class-mean fallback would differ.
    ref_layer, _ = _reference(acts, labels, pair_ids, contrast_defs, layers)
    class_mean = normalize_vec(acts[0][[0, 1, 3]].mean(0) - acts[0][[2, 4]].mean(0))
    assert not np.allclose(ref_layer["pair:F-M:F"][0], class_mean, rtol=1e-3)


def test_streaming_handles_multiclass_one_pole_pooled_negatives():
    rng = np.random.default_rng(2)
    n, d = 9, 4
    layers = [0, 1]
    acts = {L: rng.standard_normal((n, d)) for L in layers}
    labels = ["asian", "black", "white"] * 3
    pair_ids = None  # multiclass one-pole has no pairing -> pooled class means
    contrast_defs = [
        ("one_pole:asian", "asian", ["black", "white"], "one_pole", None),
        ("one_pole:black", "black", ["asian", "white"], "one_pole", None),
    ]
    _assert_matches(acts, labels, pair_ids, contrast_defs, layers)
