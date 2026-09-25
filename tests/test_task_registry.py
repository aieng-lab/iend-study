"""Task registry resolves variants via YAML extends / builder."""

from __future__ import annotations

import pytest
from study.registry import TASK_BUILDERS, resolve_builder_id


def test_canonical_builders_registered():
    assert "gender_en" in TASK_BUILDERS
    assert "race" in TASK_BUILDERS
    # Variants must NOT require a hardcoded leaf entry.
    assert "gender_en_pre" not in TASK_BUILDERS
    assert "race_one_pole" not in TASK_BUILDERS


def test_resolve_via_extends():
    assert resolve_builder_id("gender_en_pre") == "gender_en"
    assert resolve_builder_id("race_one_pole") == "race"
    assert resolve_builder_id("religion_one_pole") == "religion"


def test_resolve_exact():
    assert resolve_builder_id("gender_en") == "gender_en"
    assert resolve_builder_id("ioi") == "ioi"
    assert resolve_builder_id("repetition") == "repetition"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_builder_id("definitely_not_a_task_xyz")
