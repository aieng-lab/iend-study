"""promote_subdir_results must refuse a screen whose rows lost their encoder metrics."""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("promote", ROOT / "scripts" / "promote_subdir_results.py")
promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(promote)


def _row(mid, with_encoder):
    metrics = {"roc_auc_neutral": 0.9} if with_encoder else {"causal_signed_effect": 0.1}
    return {"method": mid, "status": "ok", "metrics": metrics}


def _tree(root: Path, main_rows, screen_rows):
    for name, rows in (("main", main_rows), ("main/screen", screen_rows)):
        d = root / name
        (d / "artifacts" / "gradiend__onepole__A").mkdir(parents=True, exist_ok=True)
        (d / "artifacts" / "gradiend__onepole__A" / "done.json").write_text("{}", encoding="utf-8")
        (d / "results.json").write_text(json.dumps({"methods": rows}), encoding="utf-8")
    return root / "main" / "screen", root / "main"


def test_rows_that_lost_encoder_metrics_block_the_promotion():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = _tree(Path(tmp), [_row("gradiend:A", True)], [_row("gradiend:A", False)])
        ok, why = promote.completeness(src, dst, "gradiend")
        assert not ok and "lost their encoder metrics" in why
        ok, _ = promote.completeness(src, dst, "gradiend", allow_missing_rows=True)
        assert ok


def test_intact_rows_pass():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = _tree(Path(tmp), [_row("gradiend:A", True)], [_row("gradiend:A", True)])
        assert promote.completeness(src, dst, "gradiend")[0]
