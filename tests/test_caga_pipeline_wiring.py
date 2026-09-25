"""CAGA is a first-class opt-in method in the main pipeline (like CGA).

Fast, no-GPU checks that the enabled-methods resolution, backend expansion, and
artifact-reload wiring recognize ``caga`` exactly the way they recognize ``cga``,
so CAGA rides the normal pipeline (train checkpoint -> claim-filtered causal ->
merge) instead of a parallel runner. The activation-steering causal block itself
needs a real trainer (GPU) and is exercised by the standalone-runner e2e.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from study.deep_pipeline import _artifact_backends_to_reload, _default_enabled
from study.training_profiles import CAGA_BACKENDS, expand_cga_backends


def _cfg(methods):
    return SimpleNamespace(
        suite={"methods": methods, "causal": {"enabled": True}},
        model={"sae_release": "some-release"},
        training={},
    )


def test_caga_enabled_only_when_suite_lists_it():
    assert "caga" in _default_enabled(_cfg({"caga": ["plain"]}), {})
    assert "caga" not in _default_enabled(_cfg({}), {})
    # Falsy value = off, same rule as cga.
    assert "caga" not in _default_enabled(_cfg({"caga": []}), {})


def test_caga_opt_in_does_not_disturb_headline_families():
    enabled = _default_enabled(_cfg({"caga": ["plain"]}), {})
    assert {"gradiend", "actiend", "sae", "caa"} <= enabled


def test_expand_cga_backends_adds_caga_backend_id():
    assert set(CAGA_BACKENDS) <= expand_cga_backends({"caga"})
    # Not added when caga is absent.
    assert not (set(CAGA_BACKENDS) & expand_cga_backends({"gradiend"}))


def test_artifact_reload_includes_caga_trainer_when_enabled():
    out = _artifact_backends_to_reload({"gradiend", "caga"})
    assert "gradiend" in out
    assert set(CAGA_BACKENDS) <= set(out)
    # Absent when caga not enabled.
    assert not (set(CAGA_BACKENDS) & set(_artifact_backends_to_reload({"gradiend"})))


# --- analysis wiring: caga buckets into two_pole / one_pole like cga ----------
def test_caga_is_a_gradiend_like_backend_for_grouping():
    from analysis.task_specs import GRADIEND_LIKE_BACKENDS

    assert "caga" in GRADIEND_LIKE_BACKENDS


def test_pole_from_row_buckets_caga_ids_by_shape():
    # Causal-only caga rows carry the pole in the id (no encoding row to inherit
    # ``ablation`` from): caga:A-B:C -> pair, caga:C -> one_pole.
    from analysis.method_groups import _pole_from_row

    assert _pole_from_row({"method": "caga:F-M:F", "metrics": {}}) == "pair"
    assert _pole_from_row({"method": "caga:F", "metrics": {}}) == "one_pole"
    # An attached row inherits ablation directly (pair/one_pole strings).
    assert _pole_from_row({"method": "caga:F", "metrics": {"ablation": "pair"}}) == "pair"


def test_caga_groups_registered_in_order_and_latex():
    from analysis.method_groups import METHOD_GROUP_ORDER
    from analysis.summary_latex import METHOD_LATEX, PAPER_METHOD_GROUPS

    assert "caga:two_pole" in METHOD_GROUP_ORDER
    assert "caga:one_pole" in METHOD_GROUP_ORDER
    assert "caga:two_pole" in PAPER_METHOD_GROUPS
    assert METHOD_LATEX["caga:two_pole"] == r"CAGA$_{\mathrm{pw}}$"
    assert "CAGA" in METHOD_LATEX["caga:one_pole"]


def test_parse_methods_arg_accepts_caga():
    # The --methods CLI validator must know caga (the slurm array builder calls it
    # first; an unknown method aborts the whole launch before anything runs).
    from study.config import parse_methods_arg

    assert parse_methods_arg(["caga"]) == ["caga"]
    assert parse_methods_arg(["caga,gradiend"]) == ["caga", "gradiend"]


def test_results_merge_family_of_maps_caga_ids():
    from study.results_merge import method_family

    assert method_family("caga:F-M:F") == "caga"
    assert method_family("caga:F") == "caga"
    assert method_family("caga") == "caga"


def test_group_is_applicable_gates_caga_on_enabled():
    from analysis.task_specs import group_is_applicable

    spec_on = {"enabled_methods": ["caga", "gradiend"], "pair": True, "one_pole": True}
    spec_off = {"enabled_methods": ["gradiend"], "pair": True, "one_pole": True}
    assert group_is_applicable("caga:two_pole", spec_on) is True
    assert group_is_applicable("caga:one_pole", spec_on) is True
    assert group_is_applicable("caga:two_pole", spec_off) is False


# --- AGIEND (learned act-grad) wired the same way as caga --------------------
def test_agiend_enabled_only_when_suite_lists_it():
    assert "agiend" in _default_enabled(_cfg({"agiend": ["plain"]}), {})
    assert "agiend" not in _default_enabled(_cfg({}), {})


def test_agiend_backend_routing():
    from study.training_profiles import (
        AGIEND_BACKENDS, expand_cga_backends, is_activation_gradient_backend,
        is_agiend_backend, is_caga_backend,
    )

    assert is_agiend_backend("agiend") and is_activation_gradient_backend("agiend")
    assert not is_caga_backend("agiend")  # trained, not the closed-form mean
    assert set(AGIEND_BACKENDS) <= expand_cga_backends({"agiend"})


def test_agiend_causal_uses_package_decoder_not_caa_bypass():
    source = (
        Path(__file__).resolve().parents[1] / "causal_study.py"
    ).read_text(encoding="utf-8")
    assert "if is_caga_backend(backend):" in source
    assert "else (\"all\" if is_agiend_backend(backend) else None)" in source
    assert 'if "caga" in enabled_set:' in source
    assert 'if "caga" in enabled_set or "agiend" in enabled_set:' not in source


def test_agiend_full_whitelist_membership():
    from study.config import parse_methods_arg
    from study.results_merge import method_family
    from analysis.task_specs import GRADIEND_LIKE_BACKENDS
    from analysis.method_groups import METHOD_GROUP_ORDER
    from analysis.summary_latex import METHOD_LATEX, PAPER_METHOD_GROUPS

    assert parse_methods_arg(["agiend"]) == ["agiend"]
    assert method_family("agiend:F-M:F") == "agiend"
    assert "agiend" in GRADIEND_LIKE_BACKENDS
    assert "agiend:two_pole" in METHOD_GROUP_ORDER and "agiend:one_pole" in METHOD_GROUP_ORDER
    assert "agiend:two_pole" in PAPER_METHOD_GROUPS
    assert METHOD_LATEX["agiend:two_pole"] == r"AGIEND$_{\mathrm{pw}}$"


def test_agiend_build_args_use_activation_gradient_signal():
    from study.training_profiles import build_training_arguments

    a = build_training_arguments(
        "agiend", experiment_dir="/tmp/x",
        shared={"learning_rate": 1e-5, "eval_steps": 50, "max_steps": 500}, one_pole=True,
    )
    assert a.signal.kind == "activation_gradient"
    assert a.signal_scope is not None
    # activation-space: no weight-space pruning (like actiend)
    assert a.pre_prune_config is None and a.post_prune_config is None


def test_pole_from_row_buckets_agiend_ids():
    from analysis.method_groups import _pole_from_row

    assert _pole_from_row({"method": "agiend:F-M:F", "metrics": {}}) == "pair"
    assert _pole_from_row({"method": "agiend:F", "metrics": {}}) == "one_pole"
