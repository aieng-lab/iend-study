"""REFRESH_CAUSAL must re-enter status=ok results (not ~20s no-op)."""

from __future__ import annotations

import json
from pathlib import Path

from study.runner import _should_skip_existing


def _ok_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return root


def test_skip_existing_skips_ok() -> None:
    d = _ok_dir(Path("runs") / "_test_refresh_causal_skip" / "skip_ok")
    try:
        should_skip, config_changed = _should_skip_existing(d, skip_existing=True)
        assert should_skip is True
        assert config_changed is False
    finally:
        (d / "results.json").unlink(missing_ok=True)


def test_refresh_causal_does_not_skip_ok() -> None:
    d = _ok_dir(Path("runs") / "_test_refresh_causal_skip" / "refresh")
    try:
        should_skip, config_changed = _should_skip_existing(
            d, skip_existing=True, refresh_causal=True
        )
        assert should_skip is False
        assert config_changed is False
    finally:
        (d / "results.json").unlink(missing_ok=True)


def test_skip_existing_false_never_skips() -> None:
    d = _ok_dir(Path("runs") / "_test_refresh_causal_skip" / "noskip")
    try:
        should_skip, config_changed = _should_skip_existing(d, skip_existing=False)
        assert should_skip is False
        assert config_changed is False
    finally:
        (d / "results.json").unlink(missing_ok=True)


def test_explicit_missing_method_reenters_ok_task() -> None:
    d = Path("runs") / "_test_refresh_causal_skip" / "missing_method"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(
        json.dumps({"status": "ok", "methods": [{"method": "gradiend:F", "status": "ok"}]}),
        encoding="utf-8",
    )
    try:
        should_skip, config_changed = _should_skip_existing(
            d, skip_existing=True, requested_methods={"cga"}
        )
        assert should_skip is False
        assert config_changed is False
    finally:
        (d / "results.json").unlink(missing_ok=True)


def test_refresh_encoder_eval_does_not_skip_ok() -> None:
    d = _ok_dir(Path("runs") / "_test_refresh_causal_skip" / "refresh_encoder_eval")
    try:
        should_skip, config_changed = _should_skip_existing(
            d, skip_existing=True, refresh_encoder_eval=True
        )
        assert should_skip is False
        assert config_changed is False
    finally:
        (d / "results.json").unlink(missing_ok=True)


def test_explicit_completed_method_still_skips_ok_task() -> None:
    d = Path("runs") / "_test_refresh_causal_skip" / "completed_method"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(
        json.dumps({"status": "ok", "methods": [{"method": "cga:F", "status": "ok"}]}),
        encoding="utf-8",
    )
    try:
        should_skip, _ = _should_skip_existing(
            d, skip_existing=True, requested_methods={"cga"}
        )
        assert should_skip is True
    finally:
        (d / "results.json").unlink(missing_ok=True)
