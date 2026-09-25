"""The retired synthetic IOI task must never reach model loading."""

from __future__ import annotations

import pytest

from study.runner import run_study


def test_retired_ioi_fails_before_study_setup():
    with pytest.raises(RuntimeError, match="retired; use 'ioi_mib'"):
        run_study(model="gpt2-small", task="ioi")
