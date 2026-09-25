"""Regression tests for scripts/recover_cga_caga_agiend_causal.py.

A task's causal/ directory accumulates one progress-<jobid>.json per job that
ever ran there, across the project's whole history -- including jobs that ran
before DIRECTION_POLARITY_PROTOCOL_VERSION=2 existed. An old, structurally-
complete checkpoint is exactly as "causal_raw_entry_ok" as a fresh one, so the
loader must prefer the most recently computed entry per method id, not
whichever file happens to sort first by name.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "recover_cga_caga_agiend_causal", ROOT / "scripts" / "recover_cga_caga_agiend_causal.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)  # type: ignore[union-attr]
_load_progress_by_method = _MODULE._load_progress_by_method


def _valid_entry(mid: str, signed_effect: float) -> dict:
    strength_row = {
        "strength": 1.0,
        "signed_effect": signed_effect,
        "lms": 0.09,
        "base_lms": 0.09,
        "lms_ok": True,
        "selection_metric": 0.5,
        "is_random_control": False,
    }
    return {
        "method": mid,
        "backend": mid.split(":")[0],
        "target_class": "M",
        "strengths": [strength_row],
        "selected_strength": 1.0,
        "selected": dict(strength_row),
        "weaken_strengths": [dict(strength_row, strength=-1.0)],
        "weaken_selected_strength": -1.0,
        "weaken_selected": dict(strength_row, strength=-1.0),
        "meta": {},
    }


def test_recovery_prefers_the_newest_progress_file_not_the_first_by_name(tmp_path: Path):
    task_dir = tmp_path / "gender_en"
    causal_dir = task_dir / "causal"
    causal_dir.mkdir(parents=True)

    mid = "agiend:M"
    old_path = causal_dir / "progress-1000000.json"
    new_path = causal_dir / "progress-2000000.json"

    old_path.write_text(
        json.dumps({"by_method": {mid: _valid_entry(mid, signed_effect=-0.04)}}),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps({"by_method": {mid: _valid_entry(mid, signed_effect=0.31)}}),
        encoding="utf-8",
    )

    # Filenames alone would sort progress-1000000.json first (correctly old),
    # but make the mtime ordering the OPPOSITE of filename order, so a loader
    # that trusts filename sort order over mtime is provably wrong here.
    now = time.time()
    os.utime(old_path, (now, now))
    os.utime(new_path, (now + 100, now + 100))

    recovered = _load_progress_by_method(task_dir)

    assert recovered[mid]["selected"]["signed_effect"] == 0.31


def test_recovery_skips_an_incomplete_newer_file_and_keeps_the_complete_older_one(
    tmp_path: Path,
):
    task_dir = tmp_path / "gender_en"
    causal_dir = task_dir / "causal"
    causal_dir.mkdir(parents=True)

    mid = "cga:M"
    old_path = causal_dir / "progress-1000000.json"
    new_path = causal_dir / "progress-2000000.json"

    old_path.write_text(
        json.dumps({"by_method": {mid: _valid_entry(mid, signed_effect=0.5)}}),
        encoding="utf-8",
    )
    # A newer but incomplete entry (e.g. mid-sweep checkpoint) must not win.
    new_path.write_text(
        json.dumps({"by_method": {mid: {"method": mid, "strengths": []}}}),
        encoding="utf-8",
    )

    now = time.time()
    os.utime(old_path, (now, now))
    os.utime(new_path, (now + 100, now + 100))

    recovered = _load_progress_by_method(task_dir)

    assert recovered[mid]["selected"]["signed_effect"] == 0.5
