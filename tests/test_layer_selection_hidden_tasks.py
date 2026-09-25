"""The layer-selection appendix must never read hidden/deprecated tasks."""

from __future__ import annotations

from analysis.layer_selection_appendix import _paper_task_results


def test_hidden_tasks_are_excluded(tmp_path):
    for task in ("ioi", "ioi_mib", "race", "race_one_pole", "religion_one_pole",
                 "gender_en", "gender_en_pre", "repetition"):
        d = tmp_path / task
        d.mkdir()
        (d / "results.json").write_text("{}")
    kept = sorted(p.parent.name for p in _paper_task_results(tmp_path))
    assert kept == ["gender_en", "ioi_mib", "race", "repetition"]
