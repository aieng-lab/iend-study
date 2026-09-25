from pathlib import Path

from slurm._study_array_table import (
    build_jobs,
    load_slurm_profile_defaults,
    profile_groups,
)


MODELS = ("llama-3.1-8b", "qwen3.5-9b-base")


def test_default_keeps_one_cell_per_model_task():
    jobs = build_jobs(models=MODELS, tasks=["gender_en"], suite="core")

    assert len(jobs) == 2
    assert all("methods" not in job for job in jobs)


def test_core_can_expand_one_cell_per_method_family():
    jobs = build_jobs(
        models=MODELS,
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
    )

    assert len(jobs) == 14
    assert [job["index"] for job in jobs] == list(range(14))
    for model in MODELS:
        assert {
            job["methods"] for job in jobs if job["model"] == model
        } == {"gradiend", "actiend", "sae", "caa", "cga", "caga", "agiend"}


def test_method_filter_restricts_expanded_cells():
    jobs = build_jobs(
        models=["llama-3.1-8b"],
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
        methods=["gradiend", "sae"],
    )

    assert [job["methods"] for job in jobs] == ["gradiend", "sae"]


def test_large_models_get_conservative_method_specific_profiles():
    defaults = load_slurm_profile_defaults(Path("configs/slurm_profiles.yaml"))
    jobs = build_jobs(
        models=MODELS,
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
        slurm_profile_defaults=defaults,
    )

    by_model_method = {
        (job["model"], job["methods"]): job["slurm_profile"] for job in jobs
    }
    for model in MODELS:
        assert by_model_method[(model, "gradiend")] == "gpumem-141-1x"
        assert by_model_method[(model, "actiend")] == "gpumem-80-1x"
        assert by_model_method[(model, "cga")] == "gpumem-141-host128-1x"
        assert by_model_method[(model, "caga")] == "gpumem-80-1x"
        assert by_model_method[(model, "agiend")] == "gpumem-141-1x"
        assert by_model_method[(model, "caa")] == "gpumem-48-1x"
    assert by_model_method[("llama-3.1-8b", "sae")] == "gpumem-80-1x"
    assert by_model_method[("qwen3.5-9b-base", "sae")] == "gpumem-80-1x"

    assert profile_groups(jobs) == [
        ("gpumem-141-1x", "0,6,7,13"),
        ("gpumem-80-1x", "1,2,5,8,9,12"),
        ("gpumem-48-1x", "3,10"),
        ("gpumem-141-host128-1x", "4,11"),
    ]


def test_explicit_profile_overrides_every_method_default():
    defaults = load_slurm_profile_defaults(Path("configs/slurm_profiles.yaml"))
    jobs = build_jobs(
        models=["llama-3.1-8b"],
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
        slurm_profile_defaults=defaults,
        slurm_profile_override="gpumem-141-test-1x",
    )

    assert {job["slurm_profile"] for job in jobs} == {"gpumem-141-test-1x"}
    assert profile_groups(jobs) == [("gpumem-141-test-1x", "0,1,2,3,4,5,6")]


def test_every_configured_profile_exists_and_small_models_use_safe_floor():
    defaults = load_slurm_profile_defaults(Path("configs/slurm_profiles.yaml"))
    configured = set((defaults.get("defaults") or {}).values())
    for model_profiles in (defaults.get("models") or {}).values():
        configured.update(
            value for value in (model_profiles or {}).values() if isinstance(value, str)
        )
    assert all(Path("slurm/profiles", f"{profile}.sh").is_file() for profile in configured)

    jobs = build_jobs(
        models=["gpt2-small"],
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
        slurm_profile_defaults=defaults,
    )
    assert {job["slurm_profile"] for job in jobs} == {"gpumem-24-1x"}


def test_core_activation_gradient_methods_have_profiles():
    defaults = load_slurm_profile_defaults(Path("configs/slurm_profiles.yaml"))
    jobs = build_jobs(
        models=["gpt2-small", "pythia-70m-deduped"],
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
        methods=["caga", "agiend"],
        slurm_profile_defaults=defaults,
    )

    assert {
        (job["model"], job["methods"], job["slurm_profile"])
        for job in jobs
    } == {
        ("gpt2-small", "caga", "gpumem-24-1x"),
        ("gpt2-small", "agiend", "gpumem-24-1x"),
        ("pythia-70m-deduped", "caga", "gpumem-24-1x"),
        ("pythia-70m-deduped", "agiend", "gpumem-24-1x"),
    }


def test_27b_assigns_every_core_method_the_safe_profile():
    defaults = load_slurm_profile_defaults(Path("configs/slurm_profiles.yaml"))
    jobs = build_jobs(
        models=["gemma-3-27b-pt"],
        tasks=["gender_en"],
        suite="core",
        array_by_method=True,
        slurm_profile_defaults=defaults,
    )
    assert {job["methods"] for job in jobs} == {
        "gradiend", "actiend", "sae", "caa", "cga", "caga", "agiend"
    }
    by_method = {job["methods"]: job["slurm_profile"] for job in jobs}
    assert {by_method[method] for method in ("gradiend", "cga", "agiend")} == {
        "gpumem-141-3x"
    }
    assert {by_method[method] for method in ("caga", "actiend", "sae", "caa")} == {
        "gpumem-141-host128-1x"
    }
    assert {
        by_method[method] for method in ("actiend", "sae", "caa")
    } == {"gpumem-141-host128-1x"}
