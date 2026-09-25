"""``--tune-lr``: the package's LR finder wired into the study's train stage.

The flag is an execution switch (like ``--train-ablations``): it must reach ``_train_once`` from the CLI
and from Slurm, stay out of the config hash, apply to the learned backends only, train at the chosen rate,
record the search, and fail when no learning rate converges.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from study.config import load_study_config
from study.stages import train as train_stage

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tune_lr=None):
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    if tune_lr is not None:
        cfg.raw["cli"] = {"tune_lr": tune_lr}
    return cfg


class TestFlagWiring:
    def test_cli_flag_reaches_the_pipeline_config(self):
        src = (ROOT / "run_study.py").read_text(encoding="utf-8")
        assert '"--tune-lr"' in src and '"tune_lr": bool(args.tune_lr)' in src

    def test_slurm_forwards_the_env_var(self):
        assert "TUNE_LR=${TUNE_LR:-}" in (ROOT / "slurm" / "study_array.sh").read_text(encoding="utf-8")
        assert "--tune-lr" in (ROOT / "slurm" / "_array_task.py").read_text(encoding="utf-8")

    def test_array_task_passes_the_flag(self, monkeypatch, tmp_path):
        spec = importlib.util.spec_from_file_location("array_task", ROOT / "slurm" / "_array_task.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        (tmp_path / "job_table.json").write_text(
            json.dumps({"jobs": [{"model": "qwen3.5-27b", "task": "gender_en", "suite": "core"}]}), encoding="utf-8"
        )
        seen = {}
        monkeypatch.setattr(subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
        monkeypatch.setenv("SUBMIT_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
        monkeypatch.setenv("TUNE_LR", "1")
        mod.main()
        assert "--tune-lr" in seen["cmd"]

    def test_absent_by_default(self, monkeypatch, tmp_path):
        spec = importlib.util.spec_from_file_location("array_task2", ROOT / "slurm" / "_array_task.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        (tmp_path / "job_table.json").write_text(
            json.dumps({"jobs": [{"model": "qwen3.5-27b", "task": "gender_en", "suite": "core"}]}), encoding="utf-8"
        )
        seen = {}
        monkeypatch.setattr(subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
        monkeypatch.setenv("SUBMIT_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
        monkeypatch.delenv("TUNE_LR", raising=False)
        mod.main()
        assert "--tune-lr" not in seen["cmd"]

    def test_does_not_change_the_config_hash(self):
        base = _cfg().config_hash()
        assert _cfg(tune_lr=True).config_hash() == base

    def test_enabled_reads_the_cli_switch(self):
        assert train_stage.tune_lr_enabled(_cfg(tune_lr=True)) is True
        assert train_stage.tune_lr_enabled(_cfg(tune_lr=False)) is False
        assert train_stage.tune_lr_enabled(_cfg()) is False


class TestBackendScope:
    def test_only_the_learned_encoder_decoders_are_tuned(self):
        assert train_stage.TUNABLE_LR_BACKENDS == {"gradiend", "actiend", "agiend"}
        for closed_form in ("cga", "caga", "caa", "sae", "actiend_pre"):
            assert closed_form not in train_stage.TUNABLE_LR_BACKENDS

    def test_train_once_tunes_before_training_and_trains_at_the_chosen_rate(self):
        import inspect

        src = inspect.getsource(train_stage._train_once)
        assert "tune_lr_enabled(cfg) and backend in TUNABLE_LR_BACKENDS" in src
        assert src.index("_tune_lr_for_training(") < src.index("trainer.train(")
        assert "trainer.train(callbacks=[SignalDiversityCallback()], **train_overrides)" in src
        # the chosen rate, not the yaml one, is what gets recorded for the artifact
        assert 'train_overrides.get("learning_rate"' in src


def _result(status, best=None, probes=(), window=None, bracket=None):
    from gradiend.trainer.core.lr_search import LRSearchResult, Probe, RunState

    return LRSearchResult(
        status, best,
        [Probe(lr=lr, state=RunState(state), steps_run=steps, confirmed=conf) for lr, state, steps, conf in probes],
        bracket=bracket, window=window,
    )


class TestTuneStep:
    def _args(self, lr=1e-6, max_steps=500):
        return SimpleNamespace(learning_rate=lr, max_steps=max_steps)

    def test_returns_the_chosen_rate_and_a_summary(self, monkeypatch, tmp_path):
        calls = {}

        def fake(trainer, initial, **kw):
            calls.update(initial=initial, **kw)
            return _result("converged", 5e-6, [(1e-6, "converged", 150, True)], window=(1e-6, 2e-5))

        import gradiend.trainer.core.lr_search as lr_search

        monkeypatch.setattr(lr_search, "tune_learning_rate", fake)
        overrides, summary = train_stage._tune_lr_for_training(
            object(), self._args(), backend="gradiend", experiment_dir=tmp_path
        )
        assert overrides == {"learning_rate": 5e-6}
        assert calls["initial"] == 1e-6 and calls["probe_steps"] == 150 and calls["experiment_dir"] == str(tmp_path)
        assert summary["status"] == "converged" and summary["best_lr"] == 5e-6
        assert summary["window"] == [1e-6, 2e-5] and summary["initial_lr"] == 1e-6
        assert summary["probes"] == [{"lr": 1e-6, "state": "converged", "steps_run": 150, "confirmed": True}]
        json.dumps(summary)  # goes into done.json

    def test_a_probe_is_never_longer_than_the_configured_budget(self, monkeypatch, tmp_path):
        seen = {}
        import gradiend.trainer.core.lr_search as lr_search

        monkeypatch.setattr(
            lr_search, "tune_learning_rate",
            lambda trainer, initial, **kw: seen.update(kw) or _result("converged", initial),
        )
        train_stage._tune_lr_for_training(object(), self._args(1e-5, max_steps=100), backend="actiend", experiment_dir=tmp_path)
        assert seen["probe_steps"] == 100  # ACTIEND's whole run is 100 steps: the probe is the full run

    @pytest.mark.parametrize("status", ["no_convergent_lr", "budget_exhausted"])
    def test_no_convergent_lr_fails_fast_with_the_trace(self, monkeypatch, tmp_path, status):
        import gradiend.trainer.core.lr_search as lr_search
        from gradiend.trainer.core.lr_search import NoConvergentLearningRate

        monkeypatch.setattr(
            lr_search, "tune_learning_rate",
            lambda trainer, initial, **kw: _result(status, None, [(1e-6, "collapsed", 25, None)], bracket=(1e-7, 2e-7)),
        )
        with pytest.raises(NoConvergentLearningRate, match="lr_search.json") as exc:
            train_stage._tune_lr_for_training(object(), self._args(), backend="gradiend", experiment_dir=tmp_path)
        assert exc.value.result.status == status


class TestTunedRateIsNeverLost:
    def test_done_json_records_the_chosen_rate_and_the_search(self):
        import inspect

        src = inspect.getsource(train_stage._train_once)
        assert '"lr_search": out["lr_search"]' in src and '"learning_rate": out["learning_rate"]' in src

    def test_reload_reads_the_trained_rate_back_from_done_json(self, tmp_path):
        (tmp_path / "done.json").write_text(json.dumps({"extras": {"learning_rate": 5e-6}}), encoding="utf-8")
        assert train_stage._reload_learning_rate(tmp_path, fallback=1e-6) == 5e-6

    @pytest.mark.parametrize("content", [None, "not json", json.dumps({"extras": {}}), json.dumps({})])
    def test_reload_falls_back_to_the_configured_rate_when_unrecorded(self, tmp_path, content):
        if content is not None:
            (tmp_path / "done.json").write_text(content, encoding="utf-8")
        assert train_stage._reload_learning_rate(tmp_path, fallback=1e-6) == 1e-6

    def test_the_reload_row_uses_it(self):
        import inspect

        assert "_reload_learning_rate(" in inspect.getsource(train_stage._reload_train_once)


def _write_search(path, **over):
    saved = {
        "status": "converged", "best_lr": 5e-6, "backend": "gradiend", "probe_steps": 150, "initial_lr": 1e-6,
        "window": [1e-6, 2e-5], "bracket": None,
        "probes": [{"lr": 1e-6, "state": "converged", "steps_run": 150, "confirmed": True}],
    }
    saved.update(over)
    (path / "lr_search.json").write_text(json.dumps(saved), encoding="utf-8")


class TestFinishedSearchIsReused:
    KW = dict(backend="gradiend", initial_lr=1e-6, probe_steps=150)

    def test_a_matching_converged_search_is_returned(self, tmp_path):
        _write_search(tmp_path)
        got = train_stage._load_completed_lr_search(tmp_path, **self.KW)
        assert got["best_lr"] == 5e-6 and got["window"] == [1e-6, 2e-5]

    @pytest.mark.parametrize(
        "over",
        [
            {"status": "no_convergent_lr", "best_lr": None},
            {"backend": "actiend"},
            {"initial_lr": 1e-5},
            {"probe_steps": 100},
            {"best_lr": None},
        ],
    )
    def test_anything_that_does_not_match_is_searched_again(self, tmp_path, over):
        _write_search(tmp_path, **over)
        assert train_stage._load_completed_lr_search(tmp_path, **self.KW) is None

    def test_missing_or_corrupt_file(self, tmp_path):
        assert train_stage._load_completed_lr_search(tmp_path, **self.KW) is None
        (tmp_path / "lr_search.json").write_text("{", encoding="utf-8")
        assert train_stage._load_completed_lr_search(tmp_path, **self.KW) is None

    def test_tune_step_skips_the_search_and_says_so(self, monkeypatch, tmp_path):
        import gradiend.trainer.core.lr_search as lr_search

        _write_search(tmp_path)

        def boom(*a, **k):
            raise AssertionError("the finished search must not run again")

        monkeypatch.setattr(lr_search, "tune_learning_rate", boom)
        overrides, summary = train_stage._tune_lr_for_training(
            object(), SimpleNamespace(learning_rate=1e-6, max_steps=500), backend="gradiend", experiment_dir=tmp_path
        )
        assert overrides == {"learning_rate": 5e-6} and summary["reused"] is True

    def test_a_new_search_stores_what_is_needed_to_reuse_it(self, monkeypatch, tmp_path):
        import gradiend.trainer.core.lr_search as lr_search

        seen = {}
        monkeypatch.setattr(
            lr_search, "tune_learning_rate",
            lambda trainer, initial, **kw: seen.update(kw) or _result("converged", initial),
        )
        train_stage._tune_lr_for_training(
            object(), SimpleNamespace(learning_rate=1e-6, max_steps=500), backend="agiend", experiment_dir=tmp_path
        )
        assert seen["metadata"] == {"backend": "agiend", "probe_steps": 150}
