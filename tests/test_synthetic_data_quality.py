"""Publication/study-readiness invariants for study-owned synthetic tasks."""

from __future__ import annotations

import pytest

from study.data.synthetic import (
    INDUCTION_DATA_VERSION,
    KEY_VALUE_DATA_VERSION,
    _data_version_for_task,
    build_induction,
    build_key_value,
)


@pytest.mark.parametrize("builder", [build_induction, build_key_value])
def test_circuit_prompts_are_unique_and_split_disjoint(builder):
    frame = builder(
        rows_per_split={"train": 200, "validation": 50, "test": 50},
        seed=13,
        difficulty="easy",
    )
    assert len(frame) == 300
    assert frame["masked"].is_unique
    prompts = {
        split: set(group["masked"].astype(str))
        for split, group in frame.groupby("split")
    }
    assert prompts["train"].isdisjoint(prompts["validation"])
    assert prompts["train"].isdisjoint(prompts["test"])
    assert prompts["validation"].isdisjoint(prompts["test"])


def test_corrected_circuit_tasks_have_independent_cache_versions():
    assert _data_version_for_task("induction") == INDUCTION_DATA_VERSION
    assert _data_version_for_task("key_value") == KEY_VALUE_DATA_VERSION
    assert _data_version_for_task("ioi") != INDUCTION_DATA_VERSION

