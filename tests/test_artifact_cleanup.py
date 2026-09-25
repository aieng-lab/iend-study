from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.artifact_cleanup import (
    MARKER_NAME,
    cleanup_finished_cell_checkpoints,
    should_cleanup_checkpoints,
)

_ROOT = Path("runs") / "_test_artifact_cleanup"


def _tree(name: str) -> Path:
    d = _ROOT / name
    if d.exists():
        shutil.rmtree(d)
    for rel, size in [
        ("artifacts/gradiend__pair__F-M/model/model.safetensors", 100),
        ("artifacts/gradiend__pair__F-M/seeds/seed_0/model.safetensors", 100),
        ("artifacts/gradiend__pair__F-M/seeds/seed_0/input_index_map.safetensors", 50),
        ("artifacts/gradiend__pair__F-M/done.json", 7),
        ("artifacts/gradiend__pair__F-M/training.json", 7),
        ("artifacts/gradiend__pair__F-M/training_convergence.png", 7),
        ("artifacts/gradiend__onepole__F/model/model.safetensors", 100),
        ("artifacts/actiend__pair__F-M/model/model.safetensors", 100),
        ("artifacts/actiend_pre__pair__F-M/model/model.safetensors", 100),
        ("artifacts/cga__pair__F-M/model/model.safetensors", 100),
        ("results.json", 7),
    ]:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    return d


def _left(d: Path) -> set[str]:
    return {str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file()}


def test_deletes_only_weights_of_requested_backends():
    d = _tree("only_requested")
    try:
        out = cleanup_finished_cell_checkpoints(d, ["gradiend"])
        left = _left(d)
        assert out["files"] == 4 and out["bytes"] == 350
        assert out["backends"] == ["gradiend"]
        # metadata of the cleaned backend survives
        assert "artifacts/gradiend__pair__F-M/done.json" in left
        assert "artifacts/gradiend__pair__F-M/training.json" in left
        assert "artifacts/gradiend__pair__F-M/training_convergence.png" in left
        assert "results.json" in left
        # other backends' weights untouched (they may belong to a concurrent job)
        assert "artifacts/actiend__pair__F-M/model/model.safetensors" in left
        assert "artifacts/cga__pair__F-M/model/model.safetensors" in left
        assert not any(p.endswith(".safetensors") and "gradiend__" in p for p in left)
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_actiend_does_not_match_actiend_pre():
    d = _tree("actiend_vs_pre")
    try:
        cleanup_finished_cell_checkpoints(d, ["actiend"])
        left = _left(d)
        assert "artifacts/actiend__pair__F-M/model/model.safetensors" not in left
        assert "artifacts/actiend_pre__pair__F-M/model/model.safetensors" in left
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_writes_provenance_marker_and_appends():
    d = _tree("marker")
    try:
        cleanup_finished_cell_checkpoints(d, ["gradiend"])
        cleanup_finished_cell_checkpoints(d, ["cga"])
        marker = json.loads((d / MARKER_NAME).read_text(encoding="utf-8"))
        assert [m["backends"] for m in marker] == [["gradiend"], ["cga"]]
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


def test_missing_artifacts_dir_is_a_noop():
    d = _ROOT / "empty"
    try:
        d.mkdir(parents=True, exist_ok=True)
        out = cleanup_finished_cell_checkpoints(d, ["gradiend"])
        assert out["files"] == 0 and not (d / MARKER_NAME).exists()
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)


_OK_ROWS = [{"method": "gradiend:F-M", "status": "ok"}, {"method": "x|proxy", "status": "partial"}]


def test_gate_requires_flag_causal_no_errors_no_error_rows():
    on = {"cleanup_checkpoints": True}
    enabled = {"gradiend", "causal"}
    assert should_cleanup_checkpoints(on, enabled=enabled, errors=[], method_rows=_OK_ROWS)
    # opt-in: off by default / unset
    assert not should_cleanup_checkpoints({}, enabled=enabled, errors=[], method_rows=_OK_ROWS)
    assert not should_cleanup_checkpoints(None, enabled=enabled, errors=[], method_rows=_OK_ROWS)
    assert not should_cleanup_checkpoints(
        {"cleanup_checkpoints": False}, enabled=enabled, errors=[], method_rows=_OK_ROWS
    )
    # causal skipped -> the cell is not complete, checkpoints still needed
    assert not should_cleanup_checkpoints(on, enabled={"gradiend"}, errors=[], method_rows=_OK_ROWS)
    # any stage error -> keep everything for the resume
    assert not should_cleanup_checkpoints(on, enabled=enabled, errors=["boom"], method_rows=_OK_ROWS)
    # an errored row -> keep
    bad = _OK_ROWS + [{"method": "gradiend:M", "status": "error"}]
    assert not should_cleanup_checkpoints(on, enabled=enabled, errors=[], method_rows=bad)
    # nothing produced -> keep
    assert not should_cleanup_checkpoints(on, enabled=enabled, errors=[], method_rows=[])
