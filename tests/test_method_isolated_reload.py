from study.deep_pipeline import (
    _artifact_backends_to_reload,
    _missing_train_families_to_rehydrate,
    _train_families_to_rehydrate,
)
from study.training_profiles import CGA_BACKENDS


def test_sae_and_caa_do_not_reload_unrelated_iend_artifacts():
    assert _artifact_backends_to_reload({"sae", "causal", "localization"}) == ()
    assert _artifact_backends_to_reload({"caa", "causal", "localization"}) == ()


def test_reload_scope_only_contains_requested_train_family():
    assert _artifact_backends_to_reload({"gradiend", "causal"}) == ("gradiend",)
    assert _artifact_backends_to_reload({"actiend", "causal"}) == ("actiend",)
    # Ridge-only causal needs ACTIEND as its source checkpoint, without
    # enabling ACTIEND's learned causal policies.
    assert _artifact_backends_to_reload({"actiend_ridge", "causal"}) == ("actiend",)
    assert _artifact_backends_to_reload({"cga", "causal"}) == CGA_BACKENDS


def test_final_merge_rehydrates_only_the_current_method_family():
    assert _train_families_to_rehydrate({"actiend", "actiend_ridge", "causal"}) == {
        "actiend"
    }
    assert _train_families_to_rehydrate({"sae", "causal"}) == set()
    assert _train_families_to_rehydrate({"caa", "causal"}) == set()
    assert _train_families_to_rehydrate({"gradiend", "causal"}) == {"gradiend"}


def test_final_merge_does_not_reload_a_family_produced_by_current_cell():
    prior = [{"method": "gradiend:x"}]
    current = [{"method": "actiend:x"}]
    assert _missing_train_families_to_rehydrate(
        {"actiend", "actiend_ridge", "causal"}, prior, current
    ) == set()
    assert _missing_train_families_to_rehydrate(
        {"actiend", "causal"}, prior, []
    ) == {"actiend"}
