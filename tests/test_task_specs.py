from analysis.task_specs import family_is_applicable, group_is_applicable, spec_for_task


def test_two_pole_gap_on_pair_false_tasks():
    induction = spec_for_task("induction")
    gender = spec_for_task("gender_en")
    assert induction["pair"] is False
    assert gender["pair"] is True
    assert group_is_applicable("gradiend:two_pole", induction) is False
    assert group_is_applicable("gradiend:one_pole", induction) is True
    assert group_is_applicable("gradiend:two_pole", gender) is True
    assert group_is_applicable("gradiend:one_pole", gender) is True


def test_sae_pre_one_pole_detection_is_undefined_with_other_class_components():
    induction = spec_for_task("induction")
    assert group_is_applicable("sae_pre:k1", induction, metric="roc_auc_neutral")
    assert not group_is_applicable("sae_pre:k1", induction, metric="roc_auc_other")
    assert not group_is_applicable("sae_pre:k1", induction, metric="class_exclusivity")
    assert not group_is_applicable("sae_pre:k1", induction, metric="detection_score")


def test_sae_disabled_on_race_one_pole():
    spec = spec_for_task("race_one_pole")
    assert family_is_applicable("sae", spec) is False
    assert family_is_applicable("caa", spec) is False
    assert family_is_applicable("gradiend", spec) is True
    assert group_is_applicable("sae:kstar", spec) is False


def test_dump_enabled_methods_do_not_create_gaps():
    spec = spec_for_task(
        "religion",
        results={
            "model": "gpt2-small",
            "task": "religion",
            "suite": "core",
            "config": {"enabled_methods": ["sae"], "ablations": {"pair": True}},
        },
    )
    assert family_is_applicable("caa", spec) is True
    assert family_is_applicable("gradiend", spec) is True
    assert group_is_applicable("caa:two_pole", spec) is True
    assert group_is_applicable("caa:one_pole", spec) is True


def test_actiend_ridge_is_opt_in_and_needs_actiend():
    core = spec_for_task("religion", suite="core")
    assert "actiend" in core["enabled_methods"]
    assert "actiend_ridge" not in core["enabled_methods"]
    spec = spec_for_task("religion", suite="full_plus")
    assert "actiend_ridge" in spec["enabled_methods"]
    assert group_is_applicable("actiend_ridge:two_pole", spec) is True
    assert group_is_applicable("actiend_ridge:one_pole", spec) is True


def test_joint_not_in_core():
    spec = spec_for_task("gender_en", suite="core")
    assert group_is_applicable("sae:kstar", spec) is True
    assert group_is_applicable("sae:joint", spec) is False
    full = spec_for_task("gender_en", suite="full")
    assert group_is_applicable("sae:joint", full) is True
