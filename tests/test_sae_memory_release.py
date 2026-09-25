from sae_eval import release_sae_layer_device_memory


def test_sae_device_release_keeps_cross_layer_selection_payload():
    sae = object()
    compact = {"by_class": {"x": {"test_cols": [1.0]}}}
    raw = {
        "_sae": sae,
        "_all_k_compact": compact,
        "_test_latents": [1.0, 2.0],
        "readouts": {"per_class_by_class": {}},
    }

    release_sae_layer_device_memory(raw)

    assert "_sae" not in raw
    assert raw["_all_k_compact"] is compact
    assert raw["_test_latents"] == [1.0, 2.0]
    assert "readouts" in raw
