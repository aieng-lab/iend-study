"""CAGA/AGIEND wiring: additive activation-gradient signal, existing backends intact.

The activation-gradient signal (dL/dh) is the third signal axis. These backends
must build a ``Signal.activation_gradient`` while leaving ACTIEND (activation
value) and GRADIEND (weight gradient) byte-for-byte unchanged, so enabling them
is inert for every existing run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from study.training_profiles import (
    is_activation_gradient_backend,
    is_agiend_backend,
    is_caga_backend,
    build_training_arguments,
)


def test_backend_classification():
    assert is_caga_backend("caga") and is_activation_gradient_backend("caga")
    assert is_agiend_backend("agiend") and is_activation_gradient_backend("agiend")
    assert not is_activation_gradient_backend("actiend")
    assert not is_activation_gradient_backend("gradiend")
    assert not is_activation_gradient_backend("caa")


@pytest.mark.parametrize(
    "backend,expected_kind",
    [
        ("caga", "activation_gradient"),
        ("agiend", "activation_gradient"),
        ("actiend", "activation"),
        ("gradiend", "gradient"),
    ],
)
def test_signal_kind_per_backend(backend, expected_kind):
    from gradiend.signal_space import signal_kind

    args = build_training_arguments(
        backend=backend,
        experiment_dir=Path(tempfile.mkdtemp()),
        shared={"activation_site": "prediction"},
        split_mode="none",
        metadata={},
        learning_rate=1e-4,
    )
    assert signal_kind(args.signal) == expected_kind


def test_caga_records_signal_kind_meta_and_is_activation_scoped():
    args = build_training_arguments(
        backend="caga",
        experiment_dir=Path(tempfile.mkdtemp()),
        shared={"activation_site": "prediction"},
        split_mode="none",
        metadata={},
        learning_rate=1e-4,
    )
    meta = args.metadata or {}
    assert meta.get("signal_kind") == "activation_gradient"
    assert meta.get("activation_site")  # resolved a site, like ACTIEND


@pytest.mark.parametrize("backend", ["caga", "agiend"])
def test_activation_gradient_backends_keep_package_next_token_default(backend):
    args = build_training_arguments(
        backend=backend,
        experiment_dir=Path(tempfile.mkdtemp()),
        shared={"activation_site": "prediction"},
        split_mode="none",
        metadata={},
        learning_rate=1e-4,
    )
    # The study leaves objective selection to the package.  Its activation-
    # gradient dataset resolves ``auto`` to the package's clm_next_token default.
    assert args.prediction_objective == "auto"
