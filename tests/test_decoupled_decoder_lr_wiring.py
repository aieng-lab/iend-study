"""Study-side wiring for the opt-in decoupled decoder learning rate.

The load-bearing guarantee is that an *unconfigured* run is byte-identical to
before: the key must be absent from the training kwargs entirely, not present
as ``None``, so no train-artifact hash or recorded metadata shifts for any
existing config.

ACTIEND carries a configured decoder rate in ``defaults.yaml``; GRADIEND and the
closed-form methods stay shared, so "unconfigured" is tested on those.
"""

from __future__ import annotations

import pytest

from study.config import load_study_config, training_kwargs_from_config
from study.training_profiles import shared_training_kwargs


def _cfg(**training):
    cfg = load_study_config(model="gpt2-small", task="gender_en")
    if training:
        cfg.raw.setdefault("training", {}).update(training)
    return cfg


class TestUnconfiguredIsUnchanged:
    def test_key_is_absent_not_none_when_unset(self):
        """Absent, so existing kwargs dicts and their hashes do not shift."""
        kwargs = training_kwargs_from_config(_cfg(), backend="gradiend")
        assert "learning_rate_decoder" not in kwargs

    def test_shared_training_kwargs_also_omits_it(self):
        kwargs = shared_training_kwargs(_cfg(), backend="gradiend")
        assert "learning_rate_decoder" not in kwargs

    def test_resolver_returns_none(self):
        assert _cfg().learning_rate_decoder(backend="gradiend") is None

    def test_actiend_default_decoder_rate_comes_from_defaults_yaml(self):
        assert _cfg().learning_rate_decoder(backend="actiend") == pytest.approx(1.15e-2)

    def test_shared_learning_rate_is_untouched(self):
        kwargs = shared_training_kwargs(_cfg(), backend="caga")
        assert kwargs["learning_rate"] == pytest.approx(1e-5)


class TestResolutionPriority:
    def test_backend_specific_key_wins_over_shared(self):
        cfg = _cfg(
            learning_rate_decoder=1e-4,
            learning_rate_decoder_actiend=2e-3,
        )
        assert cfg.learning_rate_decoder(backend="actiend") == pytest.approx(2e-3)

    def test_shared_key_applies_when_no_backend_key(self):
        cfg = _cfg(learning_rate_decoder=1e-4)
        assert cfg.learning_rate_decoder(backend="gradiend") == pytest.approx(1e-4)

    def test_cli_beats_a_yaml_backend_key(self):
        """Superseded 2026-08-28: previously named ...but_not_backend_specific.

        The old assertion required an explicit CLI value to lose to a YAML
        backend key. That is the silent-discard bug fixed in StudyConfig:
        resolution is now specificity-then-provenance, so a CLI value outranks a
        YAML default while a *more specific* CLI flag still outranks a shared one
        (see test_cli_backend_key_still_beats_cli_shared).
        """
        cfg = _cfg(learning_rate_decoder=1e-4)
        assert cfg.learning_rate_decoder(5e-3, backend="actiend") == pytest.approx(5e-3)

        cfg_backend = _cfg(learning_rate_decoder_actiend=2e-3)
        assert cfg_backend.learning_rate_decoder(
            5e-3, backend="actiend"
        ) == pytest.approx(5e-3)

    def test_cli_backend_key_still_beats_cli_shared(self):
        """Specificity is preserved: --lr-decoder-actiend beats a shared value."""
        from study.config import load_study_config

        cfg = load_study_config(
            model="gpt2-small",
            task="gender_en",
            cli_overrides={
                "training": {
                    "learning_rate_decoder": 3e-3,
                    "learning_rate_decoder_actiend": 7e-3,
                }
            },
        )
        assert cfg.learning_rate_decoder(backend="actiend") == pytest.approx(7e-3)

    def test_actiend_pre_reuses_the_actiend_key(self):
        """Mirrors learning_rate's own actiend_pre -> actiend aliasing."""
        cfg = _cfg(learning_rate_decoder_actiend=2e-3)
        assert cfg.learning_rate_decoder(backend="actiend_pre") == pytest.approx(2e-3)

    def test_gradiend_key_does_not_leak_into_actiend(self):
        cfg = _cfg(learning_rate_decoder_gradiend=9e-3)
        assert cfg.learning_rate_decoder(backend="actiend") == pytest.approx(1.15e-2)
        assert cfg.learning_rate_decoder(backend="gradiend") == pytest.approx(9e-3)


class TestConfiguredPathReachesTrainingArguments:
    def test_kwargs_carry_the_value(self):
        cfg = _cfg(learning_rate_decoder_actiend=2e-3)
        kwargs = shared_training_kwargs(cfg, backend="actiend")
        assert kwargs["learning_rate_decoder"] == pytest.approx(2e-3)

    def test_build_training_arguments_sets_the_field_and_records_metadata(self):
        from study.training_profiles import build_training_arguments

        cfg = _cfg(learning_rate_decoder_actiend=2e-3)
        shared = shared_training_kwargs(cfg, backend="actiend")
        args = build_training_arguments(
            "actiend", experiment_dir="/tmp/x", shared=shared
        )

        assert args.learning_rate_decoder == pytest.approx(2e-3)
        assert args.learning_rate == pytest.approx(cfg.learning_rate(backend="actiend"))
        assert args.metadata["learning_rate_decoder"] == pytest.approx(2e-3)

    def test_unconfigured_build_leaves_field_none_and_metadata_clean(self):
        from study.training_profiles import build_training_arguments

        shared = shared_training_kwargs(_cfg(), backend="gradiend")
        args = build_training_arguments(
            "gradiend", experiment_dir="/tmp/x", shared=shared
        )

        assert args.learning_rate_decoder is None
        assert "learning_rate_decoder" not in args.metadata

    def test_artifact_hash_changes_when_decoder_rate_changes(self):
        from study.stages.train import _train_artifact_hash

        cfg = _cfg()
        base = shared_training_kwargs(cfg, backend="gradiend")
        decoupled = dict(base)
        decoupled["learning_rate_decoder"] = 1e-2
        common = {
            "backend": "gradiend",
            "cfg": cfg,
            "split_mode": "none",
            "target_classes": ["female", "male"],
        }
        assert _train_artifact_hash(shared=base, **common) != _train_artifact_hash(
            shared=decoupled, **common
        )


class TestCliSurface:
    def test_run_study_exposes_and_forwards_both_decoder_flags(self):
        """run_study builds its parser inside main(), so assert on the source.

        An earlier version of this test looked for a ``build_parser`` factory and
        skipped when it was absent, which meant it asserted nothing at all.
        """
        import run_study

        with open(run_study.__file__, encoding="utf-8") as handle:
            text = handle.read()

        for flag, key in (
            ("--lr-decoder-actiend", "learning_rate_decoder_actiend"),
            ("--lr-decoder-gradiend", "learning_rate_decoder_gradiend"),
        ):
            assert flag in text, f"{flag} is not declared in run_study"
            # Declaring the flag without mapping it into training_cli would
            # accept the argument and silently ignore it.
            assert key in text, f"{flag} is declared but never forwarded as {key}"

    def test_flags_are_read_back_under_their_argparse_dest(self):
        """A declared flag whose dest is never read is silently inert."""
        import run_study

        with open(run_study.__file__, encoding="utf-8") as handle:
            text = handle.read()

        assert "args.lr_decoder_actiend" in text
        assert "args.lr_decoder_gradiend" in text

    def test_array_task_forwards_the_env_var(self, monkeypatch):
        import slurm._array_task as array_task

        source = array_task.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        # The env var must reach run_study as the CLI flag; a silent drop would
        # make a submitted ablation quietly run the default configuration.
        assert "LR_DECODER_ACTIEND" in text
        assert "--lr-decoder-actiend" in text

    def test_step_budget_overrides_reach_array_tasks(self):
        from pathlib import Path

        import run_study
        import slurm._array_task as array_task

        run_text = Path(run_study.__file__).read_text(encoding="utf-8")
        array_text = Path(array_task.__file__).read_text(encoding="utf-8")
        assert "--max-steps" in run_text
        assert 'training_cli["max_steps"]' in run_text
        assert "--eval-steps" in run_text
        assert 'training_cli["eval_steps"]' in run_text
        assert "MAX_STEPS" in array_text and "--max-steps" in array_text
        assert "EVAL_STEPS" in array_text and "--eval-steps" in array_text

    def test_standard_slurm_launchers_forward_gradiend_lr(self):
        """The cross-model sweep must not silently run GRADIEND at default LR."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        expected = {
            "slurm/study_single.sh": "--lr-gradiend",
            "slurm/_tasks_loop.sh": "--lr-gradiend",
            "slurm/study_tasks.sh": "LR_GRADIEND=",
            "slurm/study_array.sh": "LR_GRADIEND=",
            "slurm/_array_task.py": "--lr-gradiend",
        }
        for relative, marker in expected.items():
            text = (root / relative).read_text(encoding="utf-8")
            assert "LR_GRADIEND" in text, f"{relative} drops LR_GRADIEND"
            assert marker in text, f"{relative} does not forward {marker}"

    def test_bulk_arrays_default_to_lower_priority(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        study_array = (root / "slurm/study_array.sh").read_text(encoding="utf-8")
        submit = (root / "slurm/submit_train.sh").read_text(encoding="utf-8")

        assert 'SBATCH_NICE="${SBATCH_NICE:-1000}"' in study_array
        assert '--nice="${SBATCH_NICE}"' in study_array
        assert "NICE_ARGS" in submit
        assert '--nice="${SBATCH_NICE}"' in submit


NEW_LARGE_MODELS = (
    "gemma-3-270m", "gemma-3-4b-pt", "gemma-3-27b-pt", "qwen3.5-27b",
    "qwen3.5-35b-a3b-base", "llama-3.3-70b-instruct", "gpt-oss-20b",
)


class TestAgiendDecoderLr:
    """AGIEND has its own decoder-LR key; the large models set it to the ``auto`` reachability lift."""

    @pytest.mark.parametrize("model", NEW_LARGE_MODELS)
    def test_new_large_models_use_auto_for_agiend_only(self, model):
        cfg = load_study_config(model=model, task="gender_en")
        assert cfg.learning_rate_decoder(backend="agiend") == "auto"
        # the other backends keep their own behaviour: GRADIEND shared, ACTIEND its explicit rate
        assert cfg.learning_rate_decoder(backend="gradiend") is None
        assert cfg.learning_rate_decoder(backend="actiend") == pytest.approx(1.15e-2)
        assert cfg.learning_rate_decoder(backend="caga") is None

    @pytest.mark.parametrize("model", ("gpt2-small", "pythia-70m-deduped", "gemma-2-2b"))
    def test_existing_models_keep_agiend_shared(self, model):
        assert load_study_config(model=model, task="gender_en").learning_rate_decoder(backend="agiend") is None

    def test_reaches_the_training_kwargs_for_agiend_only(self):
        cfg = load_study_config(model="qwen3.5-27b", task="gender_en")
        assert training_kwargs_from_config(cfg, backend="agiend")["learning_rate_decoder"] == "auto"
        assert "learning_rate_decoder" not in training_kwargs_from_config(cfg, backend="gradiend")

    def test_the_sentinel_stays_a_string_and_a_number_is_still_floated(self):
        cfg = _cfg(learning_rate_decoder_agiend="auto")
        assert cfg.learning_rate_decoder(backend="agiend") == "auto"
        cfg = _cfg(learning_rate_decoder_agiend=2e-3)
        assert cfg.learning_rate_decoder(backend="agiend") == pytest.approx(2e-3)

    def test_a_cli_value_beats_the_yaml_auto(self):
        cfg = load_study_config(
            model="qwen3.5-27b", task="gender_en",
            cli_overrides={"training": {"learning_rate_decoder_agiend": 5e-3}},
        )
        assert cfg.learning_rate_decoder(backend="agiend") == pytest.approx(5e-3)

    def test_a_shared_decoder_lr_does_not_leak_into_agiend_when_it_has_its_own_key(self):
        cfg = _cfg(learning_rate_decoder=1e-4, learning_rate_decoder_agiend="auto")
        assert cfg.learning_rate_decoder(backend="agiend") == "auto"
        assert cfg.learning_rate_decoder(backend="gradiend") == pytest.approx(1e-4)
