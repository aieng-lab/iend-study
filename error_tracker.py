"""Centralized error tracker – appends structured records to a JSONL file.

Usage:
    from error_tracker import track_error

    try:
        ...
    except Exception as exc:
        track_error(exc, context="evaluate_decoder", backend="sae", method_id="sae_F_k32")
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

_DEFAULT_PATH = os.environ.get("GRADIEND_ERROR_LOG", "error_report.jsonl")
_DISABLE_VALUES = {"", "0", "false", "off", "none", "null"}
_MAX_BYTES = int(os.environ.get("GRADIEND_ERROR_LOG_MAX_BYTES", str(2 * 1024 * 1024)))
_lock = Lock()


def _logging_disabled(path_value: str) -> bool:
    return path_value.strip().lower() in _DISABLE_VALUES


def _rotate_if_needed(path: Path) -> None:
    if _MAX_BYTES <= 0:
        return
    if not path.exists():
        return
    if path.stat().st_size <= _MAX_BYTES:
        return
    backup = path.with_suffix(path.suffix + ".1")
    try:
        if backup.exists():
            backup.unlink()
    except OSError:
        # Rotation is best-effort; continue with append fallback.
        return
    try:
        path.replace(backup)
    except OSError:
        # If replace fails we still try to append to avoid dropping reports.
        return


def track_error(
    exc: BaseException,
    *,
    context: str = "",
    report_path: Optional[str] = None,
    print_msg: bool = True,
    **extra: Any,
) -> Dict[str, Any]:
    """Record *exc* to a JSONL error report and optionally print a one-liner.

    Parameters
    ----------
    exc:
        The caught exception.
    context:
        Short label for *where* in the pipeline the error occurred
        (e.g. ``"evaluate_decoder"``, ``"causal"``, ``"sae_train"``).
    report_path:
        Override the JSONL file path (default: ``error_report.jsonl`` or
        ``$GRADIEND_ERROR_LOG``).
    print_msg:
        If True (default), also print a short summary to stdout.
    **extra:
        Arbitrary key/value pairs stored alongside the record
        (``backend``, ``method_id``, ``layer``, …).

    Returns the record dict for programmatic use.
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "error_type": type(exc).__qualname__,
        "error_msg": str(exc),
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
    }
    record.update(extra)

    path_value = report_path if report_path is not None else _DEFAULT_PATH
    if not _logging_disabled(path_value):
        path = Path(path_value)
        with _lock:
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    if print_msg:
        label = f"  {context}: " if context else "  "
        print(f"{label}{type(exc).__name__}: {exc}", flush=True)

    return record
