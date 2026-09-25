"""The GRADIEND scope A/B arm must actually flip the parameter scope.

`SignalScope.layers()` on a *gradient* signal was a silent no-op until the
package fix of 2026-08-20/21, so pre- and post-fix runs share a `config_hash`
while training over different parameter sets (see
configs/suites/gradiend_scope_ab.yaml). These pin that the two arms differ in
the one way that matters, so the comparison cannot silently become a no-op
again.
"""

from __future__ import annotations

import shutil
import tempfile

from study.config import list_suites, load_study_config, training_kwargs_from_config
from study.training_profiles import build_training_arguments


def _args_for(suite: str):
    # tmp_path/tmpdir are unusable on this Windows checkout (see CLAUDE.md).
    tmp = tempfile.mkdtemp(prefix="scope-ab-")
    cfg = load_study_config(model="gpt2-small", task="race", suite=suite)
    shared = training_kwargs_from_config(cfg)
    try:
        return build_training_arguments(
            "gradiend",
            experiment_dir=tmp,
            shared=dict(shared),
            one_pole=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class TestScopeArms:
    def test_the_ab_suite_is_registered(self):
        assert "gradiend_scope_ab" in list_suites()

    def test_default_suite_restricts_gradiend_to_layers(self):
        args = _args_for("full_plus")
        assert args.signal_scope is not None, (
            "the main-experiment arm must carry SignalScope.layers()"
        )

    def test_ab_suite_leaves_gradiend_on_the_full_parameter_scope(self):
        args = _args_for("gradiend_scope_ab")
        assert args.signal_scope is None, (
            "the A/B arm must fall back to the package's default scope; a "
            "non-None scope here means the comparison is a no-op"
        )

    def test_the_two_arms_hash_differently(self):
        """The whole failure mode was two different scopes sharing one hash."""
        a = load_study_config(model="gpt2-small", task="race", suite="full_plus")
        b = load_study_config(
            model="gpt2-small", task="race", suite="gradiend_scope_ab"
        )
        assert a.config_hash() != b.config_hash()
