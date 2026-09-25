"""Study-side wiring for the optimizer choice (SGD reachability arm).

Same load-bearing guarantee as the decoupled decoder LR: an unconfigured run
must produce a byte-identical kwargs dict, so no train-artifact hash shifts for
any existing config. Rationale in ``IEND_THEORY_PLAN.md`` 2.7.
"""

from __future__ import annotations

from study.config import load_study_config, training_kwargs_from_config
from study.training_profiles import shared_training_kwargs


def _cfg(**training):
    cfg = load_study_config(model="gpt2-small", task="gender_en")
    if training:
        cfg.raw.setdefault("training", {}).update(training)
    return cfg


class TestUnconfiguredIsUnchanged:
    def test_optim_absent_not_defaulted(self):
        assert "optim" not in training_kwargs_from_config(_cfg(), backend="actiend")

    def test_sgd_momentum_absent_not_defaulted(self):
        kwargs = training_kwargs_from_config(_cfg(), backend="actiend")
        assert "sgd_momentum" not in kwargs

    def test_shared_training_kwargs_also_omits_both(self):
        kwargs = shared_training_kwargs(_cfg(), backend="actiend")
        assert "optim" not in kwargs and "sgd_momentum" not in kwargs


class TestConfiguredPathIsForwarded:
    def test_optim_reaches_training_kwargs(self):
        kwargs = training_kwargs_from_config(_cfg(optim="sgd"), backend="actiend")
        assert kwargs["optim"] == "sgd"

    def test_momentum_reaches_training_kwargs(self):
        kwargs = training_kwargs_from_config(
            _cfg(optim="sgd", sgd_momentum=0.9), backend="actiend"
        )
        assert kwargs["sgd_momentum"] == 0.9

    def test_artifact_hash_changes_when_optimizer_changes(self):
        """An SGD arm must not silently reuse an Adam checkpoint."""
        from study.stages.train import _train_artifact_hash

        cfg = _cfg()
        base = shared_training_kwargs(cfg, backend="actiend")
        sgd = dict(base, optim="sgd")
        common = {
            "backend": "actiend",
            "cfg": cfg,
            "split_mode": "none",
            "target_classes": ["F", "M"],
        }
        assert _train_artifact_hash(shared=base, **common) != _train_artifact_hash(
            shared=sgd, **common
        )


class TestFailFastIsTheDefault:
    """A job that continues past a fatal error reports COMPLETED for nothing.

    Eleven of twenty-four cells in the 2026-08-28 large-model LR screen hit CUDA
    OOM, produced no results, and exited 0. Opting IN to error visibility is the
    wrong default.
    """

    LAUNCHERS = (
        "slurm/study_array.sh",
        "slurm/study_tasks.sh",
        "slurm/_tasks_loop.sh",
        "slurm/study_single.sh",
    )

    def test_every_launcher_defaults_fail_fast_on(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in self.LAUNCHERS:
            text = (root / rel).read_text(encoding="utf-8")
            assert "FAIL_FAST" in text, f"{rel} does not forward FAIL_FAST"
            assert "FAIL_FAST:-1" in text, f"{rel} still defaults FAIL_FAST off"
            assert "FAIL_FAST:-}" not in text, f"{rel} has an unset default left"

    def test_fail_fast_can_still_be_disabled(self):
        """FAIL_FAST=0 must remain an explicit opt-out."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "slurm/_tasks_loop.sh").read_text(encoding="utf-8")
        assert '"${FAIL_FAST:-1}" == "1"' in text


class TestOnePoleKeepsPrePrune:
    """The workaround that made GRADIEND unable to run `none` at 8B.

    Disabling pre_prune for one-pole GRADIEND also disabled lazy_init (which is
    gated on pre_prune_config), so the encoder was built at full width: 26 GiB
    at 8B instead of ~280 MB. The package now stratifies over the classes
    actually present, so the strip is no longer needed.
    """

    def test_train_stage_no_longer_nulls_pre_prune(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "study/stages/train.py").read_text(encoding="utf-8")
        assert "args.pre_prune_config = None" not in text

    def test_reason_is_recorded_so_it_is_not_reinstated(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "study/stages/train.py").read_text(encoding="utf-8")
        assert "26 GiB at 8B" in text

    def test_one_pole_config_still_carries_a_prune_config(self):
        from study.config import load_study_config, training_kwargs_from_config

        cfg = load_study_config(model="gpt2-small", task="gender_en")
        kwargs = training_kwargs_from_config(cfg, backend="gradiend")
        assert kwargs.get("pre_prune_config") is not None
