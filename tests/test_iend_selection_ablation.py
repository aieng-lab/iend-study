from study.config import load_study_config
from study.training_profiles import build_training_arguments, shared_training_kwargs


def test_study_passes_encoding_e_without_changing_convergence(tmp_path):
    cfg = load_study_config(
        model="gpt2-small",
        task="gender_en",
        suite="full_plus",
        cli_overrides={
            "training": {
                "selection_metric": "encoding_e",
                "convergent_metric": "correlation",
            }
        },
    )
    shared = shared_training_kwargs(cfg, backend="actiend")
    args = build_training_arguments(
        "actiend",
        experiment_dir=str(tmp_path),
        shared=shared,
        split_mode="tensors",
        one_pole=True,
    )

    assert args.selection_metric == "encoding_e"
    assert args.convergent_metric == "correlation"
