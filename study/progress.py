"""Tiny crash-visible progress file. Not used for skip-existing (that is results.json)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def write_pipeline_progress(output_dir: Optional[Any], **fields: Any) -> None:
    """Overwrite ``pipeline_progress.json`` under the run dir. Never raises."""
    if output_dir is None:
        return
    try:
        path = Path(output_dir) / "pipeline_progress.json"
        payload = {k: v for k, v in fields.items() if v is not None}
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return
