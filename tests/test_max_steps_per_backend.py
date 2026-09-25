"""Per-backend max_steps (ACTIEND=100, GRADIEND=500) — 2026-08-31.

ACTIEND is schedule-robust and converges fast; GRADIEND is schedule-sensitive
and needs the full budget. A single max_steps forced both to the same value.
"""

from __future__ import annotations

from study.config import load_study_config, training_kwargs_from_config


def _cfg(**training):
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="full_plus")
    # cfg.training returns a copy; mutate the backing raw dict.
    cfg.raw.setdefault("training", {}).update(training)
    return cfg


class TestMaxStepsPerBackend:
    def test_actiend_uses_its_own_budget(self):
        cfg = _cfg(max_steps=500, max_steps_actiend=100)
        assert cfg.max_steps(backend="actiend") == 100
        assert cfg.max_steps(backend="gradiend") == 500

    def test_actiend_pre_follows_actiend(self):
        cfg = _cfg(max_steps=500, max_steps_actiend=100)
        assert cfg.max_steps(backend="actiend_pre") == 100

    def test_falls_back_to_shared_when_unset(self):
        cfg = _cfg(max_steps=500)
        # defaults.yaml now ships max_steps_actiend: 100; clear it to test the
        # pure fallback path.
        cfg.raw["training"].pop("max_steps_actiend", None)
        assert cfg.max_steps(backend="actiend") == 500
        assert cfg.max_steps(backend="gradiend") == 500

    def test_training_kwargs_carry_the_resolved_value(self):
        cfg = _cfg(max_steps=500, max_steps_actiend=100)
        assert training_kwargs_from_config(cfg, backend="actiend")["max_steps"] == 100
        assert training_kwargs_from_config(cfg, backend="gradiend")["max_steps"] == 500

    def test_cli_max_steps_still_wins_over_backend_default(self):
        cfg = _cfg(max_steps=250, max_steps_actiend=100)
        # Simulate a CLI --max-steps by marking it a CLI key.
        cfg.cli_training_keys = frozenset({"max_steps"})
        assert cfg.max_steps(backend="actiend") == 250


class TestEvalStepsPerBackend:
    def test_backend_specific_intervals(self):
        cfg = _cfg(eval_steps=50, eval_steps_gradiend=25, eval_steps_actiend=20)
        assert cfg.eval_steps(backend="gradiend") == 25
        assert cfg.eval_steps(backend="actiend") == 20
        assert cfg.eval_steps(backend="actiend_pre") == 20

    def test_training_kwargs_carry_backend_interval(self):
        cfg = _cfg(eval_steps=50, eval_steps_gradiend=25, eval_steps_actiend=20)
        assert training_kwargs_from_config(cfg, backend="gradiend")["eval_steps"] == 25
        assert training_kwargs_from_config(cfg, backend="actiend")["eval_steps"] == 20

    def test_explicit_shared_cli_interval_wins(self):
        cfg = _cfg(eval_steps=10, eval_steps_gradiend=25, eval_steps_actiend=20)
        cfg.cli_training_keys = frozenset({"eval_steps"})
        assert cfg.eval_steps(backend="gradiend") == 10
        assert cfg.eval_steps(backend="actiend") == 10
