"""``--causal-only``: a whitelist of causal ids layered on the per-id skip set."""

from __future__ import annotations

from study.causal_policies import SkipUnlessWanted, causal_only_patterns
from study.config import StudyConfig


def _skip(stored=()):
    return SkipUnlessWanted(stored, causal_only_patterns("layerwise"))


def test_layerwise_preset_names_layers_and_aggregates_in_full_plus_id_format():
    skip = _skip()
    wanted = [
        "sae:F:L5_k1", "sae:F:all_k1",
        "caa:F:L3_act_prediction", "caa:F-M:F:L3_act_prediction",
        "caa:F:act_prediction", "caa:F-M:F:all_act_prediction",
    ]
    for mid in wanted:
        assert mid not in skip, mid


def test_everything_else_counts_as_done():
    skip = _skip()
    for mid in [
        "gradiend:F", "actiend:F:tok_all", "sae:F:kstar", "sae:F:all_k5",
        "sae:F:sel_opp_fire", "sae_pre:F:L2_k1", "caa:F:L3_act_mean",
        "caa:F:act_last", "cga:F:L20", "agiend:F",
    ]:
        assert mid in skip, mid


def test_stored_ok_ids_stay_skipped_and_set_is_truthy_when_empty():
    skip = _skip({"sae:F:L5_k1"})
    assert "sae:F:L5_k1" in skip
    assert bool(_skip())  # ``if not skip`` short-circuits must not fire


def test_group_complete_helper_only_completes_unwanted_groups():
    from causal_study import causal_group_already_complete

    skip = _skip()
    assert causal_group_already_complete(["gradiend:F", "actiend:F:tok_all"], skip)
    assert not causal_group_already_complete(["caa:F:L3_act_prediction"], skip)


def test_raw_regex_and_hash_exclusion():
    assert causal_only_patterns(["^foo$"]) == ["^foo$"]
    assert causal_only_patterns(None) == []
    base = StudyConfig.__new__(StudyConfig)
    base.raw = {"model": "m", "cli": {"causal_only": ["layerwise"]}}
    other = StudyConfig.__new__(StudyConfig)
    other.raw = {"model": "m"}
    assert base.config_hash() == other.config_hash()


def test_causal_only_refuses_to_train_or_encode():
    import pytest

    from study.deep_pipeline import assert_causal_only_is_reuse_only

    prev = {
        "methods": [{"method": "sae:F:k1"}, {"method": "caa:F:act_prediction"}],
        "raw": {"sae": {"layers": [0]}, "caa": {"vectors": {"prediction": {}}}},
    }
    assert_causal_only_is_reuse_only({"sae", "caa", "causal"}, prev)  # fine
    with pytest.raises(RuntimeError, match="also enabled"):
        assert_causal_only_is_reuse_only({"sae", "gradiend", "causal"}, prev)
    with pytest.raises(RuntimeError, match="stored results.json"):
        assert_causal_only_is_reuse_only({"sae", "caa", "causal"}, None)
    no_vectors = {"methods": prev["methods"], "raw": {"sae": {}, "caa": {}}}
    with pytest.raises(RuntimeError, match="refusing to re-encode"):
        assert_causal_only_is_reuse_only({"sae", "caa", "causal"}, no_vectors)
    errored = {"methods": prev["methods"], "raw": {"sae": {"error": "x"}, "caa": prev["raw"]["caa"]}}
    with pytest.raises(RuntimeError, match="refusing to re-encode"):
        assert_causal_only_is_reuse_only({"sae", "caa", "causal"}, errored)


def test_layer_slices_partition_the_layerwise_ids_without_overlap():
    slices = ["layerwise@0-2", "layerwise@3-5", "layerwise@6-8"]
    skips = [SkipUnlessWanted(set(), causal_only_patterns(s)) for s in slices]
    ids = (
        [f"sae:F:L{n}_k1" for n in range(9)]
        + [f"caa:A-B:F:L{n}_act_prediction" for n in range(9)]
        + ["sae:F:all_k1", "sae:F:k1", "caa:F:act_prediction", "caa:F:all_act_prediction"]
    )
    for mid in ids:
        owners = [i for i, sk in enumerate(skips) if mid not in sk]
        assert len(owners) == 1, (mid, owners)
    # aggregates go with the slice holding layer 0; anything else stays skipped
    assert "sae:F:all_k1" not in skips[0] and "sae:F:all_k1" in skips[1]
    assert "caa:F:L3_act_mean" in skips[1] and "sae:F:kstar" in skips[0]


def _cfg(paired=True, methods=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        raw={"causal": {"paired_layer_pass": paired}},
        suite={"methods": methods or {"sae": ["k1"], "caa": ["act_prediction"], "cga": ["plain"],
                                       "caga": ["plain"], "gradiend": ["none"]}},
    )


def test_core_task_gets_a_reuse_only_paired_second_pass():
    from study.runner import paired_pass_overrides

    second = paired_pass_overrides(_cfg(), {"fail_fast": True}, {"status": "ok"})
    assert second["causal_only"] == ["paired"]
    assert second["methods"] == ["sae", "caa", "cga", "caga"]  # never gradiend/actiend
    assert second["refresh_causal"] and second["skip_existing"] and second["_paired_pass"]
    assert second["fail_fast"] is True  # operator switches carry over


def test_second_pass_is_skipped_when_it_must_be():
    from study.runner import paired_pass_overrides

    ok = {"status": "ok"}
    assert paired_pass_overrides(_cfg(paired=False), {}, ok) is None          # full suites
    assert paired_pass_overrides(_cfg(), {"causal_only": ["layerwise"]}, ok) is None
    assert paired_pass_overrides(_cfg(), {"_paired_pass": True}, ok) is None  # no recursion
    assert paired_pass_overrides(_cfg(), {"skip_causal": True}, ok) is None
    assert paired_pass_overrides(_cfg(), {}, {"status": "error"}) is None
    assert paired_pass_overrides(_cfg(), {}, ok, smoke=True) is None
    # --methods narrows the families; nothing left => no pass
    assert paired_pass_overrides(_cfg(), {"methods": ["gradiend"]}, ok) is None
    only = paired_pass_overrides(_cfg(), {"methods": ["sae"]}, ok)
    assert only["methods"] == ["sae"]


def test_paired_preset_reads_both_views_for_the_four_families(monkeypatch, tmp_path):
    import analysis.method_groups as mg

    f = tmp_path / "results.json"
    f.write_text("{}")
    monkeypatch.setattr(mg, "_load_results", lambda p: {"x": 1})

    def fake_rows(payload, results_path, representation_view):
        if representation_view == "all_layer":
            return [{"backend": "sae", "source_methods": "sae:F:all_k1,sae:M:all_k1"},
                    {"backend": "caa", "source_methods": "caa:F:act_prediction"}]
        return [{"backend": "caa", "source_methods": "caa:F:L7_act_prediction"},
                {"backend": "cga", "source_methods": "cga:F:L20"},
                {"backend": "gradiend", "source_methods": "gradiend:F"}]  # not a paired family

    monkeypatch.setattr(mg, "collect_group_rows_for_results", fake_rows)
    skip = SkipUnlessWanted(set(), causal_only_patterns("paired", results_path=f))
    for mid in ["sae:F:all_k1", "sae:M:all_k1", "caa:F:act_prediction",
                "caa:F:L7_act_prediction", "cga:F:L20"]:
        assert mid not in skip, mid
    for mid in ["gradiend:F", "caa:F:L8_act_prediction", "sae:F:L3_k1", "cga:F:L21"]:
        assert mid in skip, mid


def test_paired_preset_with_nothing_missing_matches_nothing(tmp_path):
    skip = SkipUnlessWanted(set(), causal_only_patterns("paired", results_path=tmp_path / "none.json"))
    assert "sae:F:all_k1" in skip and "cga:F:L20" in skip


def test_hash_ignores_the_second_pass_family_filter():
    a = StudyConfig.__new__(StudyConfig)
    a.raw = {"model": "m", "cli": {"methods": ["sae", "caa"], "causal_only": ["paired"], "_paired_pass": True}}
    b = StudyConfig.__new__(StudyConfig)
    b.raw = {"model": "m"}
    assert a.config_hash() == b.config_hash()
    c = StudyConfig.__new__(StudyConfig)
    c.raw = {"model": "m", "cli": {"methods": ["sae"]}}  # a plain --methods run still hashes differently
    assert c.config_hash() != b.config_hash()


def test_paired_slices_partition_the_ids(monkeypatch):
    import study.causal_policies as cp

    ids = [f"caa:C{i}:act_prediction" for i in range(9)] + ["sae:F:all_k1"]
    monkeypatch.setattr(cp, "_paired_view_ids", lambda path: list(ids))
    skips = [SkipUnlessWanted(set(), causal_only_patterns(f"paired@{i}/4", results_path="x")) for i in range(4)]
    for mid in ids:
        assert sum(mid not in sk for sk in skips) == 1, mid
    import pytest

    with pytest.raises(ValueError):
        causal_only_patterns("paired@4/4", results_path="x")
