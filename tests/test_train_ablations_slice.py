"""``--train-ablations``: pair / one-pole slices as separate (parallel-GPU) jobs."""
import json
from types import SimpleNamespace

import pytest

from study.config import load_study_config
from study.runner import _should_skip_existing
from study.stages.train import _ablations, train_ablations_filter


def _cfg(train_ablations=None, **abl):
    cli = {"train_ablations": train_ablations} if train_ablations is not None else {}
    return SimpleNamespace(raw={"ablations": {"pair": True, "one_pole": True, **abl}, "cli": cli})


def test_no_filter_keeps_both():
    assert _ablations(_cfg()) == {"pair": True, "one_pole": True}


@pytest.mark.parametrize(
    "requested,expected",
    [
        (["pair"], {"pair": True, "one_pole": False}),
        (["one_pole"], {"pair": False, "one_pole": True}),
        (["one-pole"], {"pair": False, "one_pole": True}),
        (["pair,one_pole"], {"pair": True, "one_pole": True}),
    ],
)
def test_filter_narrows(requested, expected):
    assert _ablations(_cfg(requested)) == expected


def test_filter_never_enables_what_the_task_disables():
    # a one-pole-only task asked for "pair" trains nothing, it does not invent pairs
    assert _ablations(_cfg(["pair"], pair=False)) == {"pair": False, "one_pole": False}


def test_unknown_name_fails_fast():
    with pytest.raises(ValueError, match="Unknown --train-ablations"):
        train_ablations_filter(_cfg(["pairs_and_more"]))


def test_filter_does_not_touch_config_or_hash():
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    base = cfg.config_hash()
    cfg.raw["cli"] = {"train_ablations": ["pair"]}
    assert cfg.config_hash() == base
    assert cfg.raw["ablations"].get("one_pole") is not False  # full task definition kept


def test_slice_run_never_takes_whole_task_skip(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"status": "ok", "config_hash": "h", "methods": []}), encoding="utf-8"
    )
    kw = dict(skip_existing=True, config_hash="h")
    assert _should_skip_existing(tmp_path, **kw) == (True, False)
    assert _should_skip_existing(tmp_path, train_slice=True, **kw) == (False, False)


def test_table_builder_drops_tasks_without_the_slice():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm"))
    from _study_array_table import build_jobs

    common = dict(models=["gpt2-small"], suite="core", tasks=["gender_en", "ioi_mib"])
    pair = {j["task"] for j in build_jobs(train_ablations=["pair"], **common)}
    one = {j["task"] for j in build_jobs(train_ablations=["one_pole"], **common)}
    everything = {j["task"] for j in build_jobs(**common)}
    assert pair == {"gender_en"}          # ioi_mib is one-pole only
    assert one == {"gender_en", "ioi_mib"}
    assert everything == {"gender_en", "ioi_mib"}


# --- concurrent slices must not wipe each other's rows ------------------------------------
# Found live 2026-09-24 (gemma-2-2b gradiend_lr1e7): the one-pole job's later write erased the
# pair job's encoder rows, because merge_method_rows wipes a whole family whenever the writer
# carries encoder rows for it.

def _row(mid, **metrics):
    return {"method": mid, "status": "ok", "metrics": metrics or {"roc_auc": 0.9}}


def _merge(prev_rows, new_rows, regimes=None):
    from study.results_merge import merge_method_rows

    return {
        r["method"]
        for r in merge_method_rows(prev_rows, new_rows, enabled={"gradiend", "causal"}, regimes=regimes)
    }


PAIR = [_row("gradiend:negative-positive"), _row("gradiend:negative-positive:negative", causal_signed_effect=0.2)]
ONE = [_row("gradiend:positive"), _row("gradiend:negative"), _row("gradiend:positive", causal_signed_effect=0.1)]


def test_row_regime_classification():
    from study.results_merge import row_regime

    assert row_regime("gradiend:negative-positive") == "pair"
    assert row_regime("gradiend:F-M|proxy=her_his") == "pair"
    assert row_regime("actiend:F-M:F:tok_all") == "pair"
    assert row_regime("gradiend:positive") == "one_pole"
    assert row_regime("actiend:united_states:tok_all") == "one_pole"
    assert row_regime("gradiend") is None


def test_unsliced_writer_still_overwrites_the_whole_family():
    # unchanged historical behaviour: a full run replaces every gradiend row
    assert _merge(PAIR, ONE) == {"gradiend:positive", "gradiend:negative"}


def test_one_pole_slice_keeps_the_pair_slices_rows():
    got = _merge(PAIR, ONE, regimes=frozenset({"one_pole"}))
    assert {"gradiend:negative-positive", "gradiend:negative-positive:negative"} <= got
    assert {"gradiend:positive", "gradiend:negative"} <= got


def test_pair_slice_keeps_the_one_pole_slices_rows():
    got = _merge(ONE, PAIR, regimes=frozenset({"pair"}))
    assert {"gradiend:positive", "gradiend:negative", "gradiend:negative-positive"} <= got


def test_slice_ignores_stale_copies_of_the_other_slices_rows():
    # the slice process loaded these at startup; the fresh on-disk row must win
    fresh = [_row("gradiend:negative-positive", roc_auc=0.99)]
    from study.results_merge import merge_method_rows

    stale_in_writer = [_row("gradiend:negative-positive", roc_auc=0.10), *ONE]
    out = merge_method_rows(
        fresh, stale_in_writer, enabled={"gradiend", "causal"}, regimes=frozenset({"one_pole"})
    )
    pair = next(r for r in out if r["method"] == "gradiend:negative-positive")
    assert pair["metrics"]["roc_auc"] == 0.99


def test_slice_marker_is_read_from_config_and_not_persisted():
    from study.results_merge import merge_study_payload

    prev = {"methods": PAIR, "config": {}, "raw": {}}
    cur = {"methods": ONE, "config": {"train_ablations": ["one_pole"]}, "raw": {}}
    out = merge_study_payload(prev, cur, enabled={"gradiend", "causal"})
    ids = {r["method"] for r in out["methods"]}
    assert "gradiend:negative-positive" in ids and "gradiend:positive" in ids
    assert "train_ablations" not in out["config"]


# --- rebuilding lost rows must not retrain finished artifacts ------------------------------
# Re-entering an existing task always counts as "config changed" (stored vs requested hash are
# computed differently), which turns on rerun_orphaned_artifacts: a finished artifact that no
# result row points at gets RETRAINED. After the slice-merge bug wiped the pair rows, the
# "rebuild rows" reruns therefore retrained finished pair cells for hours (gemma gradiend_lr1e7).

def test_no_rerun_orphans_flag_is_wired_end_to_end():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert '"--no-rerun-orphans"' in (root / "run_study.py").read_text(encoding="utf-8")
    runner = (root / "study" / "runner.py").read_text(encoding="utf-8")
    assert 'cli.get("no_rerun_orphaned_artifacts")' in runner
    assert 'cli["rerun_orphaned_artifacts"] = True' in runner
    assert "--no-rerun-orphans" in (root / "slurm" / "_array_task.py").read_text(encoding="utf-8")
    assert "NO_RERUN_ORPHANS=${NO_RERUN_ORPHANS:-}" in (root / "slurm" / "study_array.sh").read_text(encoding="utf-8")


def test_no_rerun_orphans_does_not_change_the_config_hash():
    cfg = load_study_config(model="gpt2-small", task="gender_en", suite="core")
    base = cfg.config_hash()
    cfg.raw["cli"] = {"no_rerun_orphaned_artifacts": True, "train_ablations": ["pair"]}
    assert cfg.config_hash() == base


def test_array_task_passes_the_flag(monkeypatch, tmp_path):
    import importlib.util
    import json as _json
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "array_task", __import__("pathlib").Path(__file__).resolve().parents[1] / "slurm" / "_array_task.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "job_table.json").write_text(
        _json.dumps({"jobs": [{"model": "gemma-2-2b", "task": "race", "suite": "core"}]}), encoding="utf-8"
    )
    seen = {}
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    monkeypatch.setenv("SUBMIT_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("NO_RERUN_ORPHANS", "1")
    mod.main()
    assert "--no-rerun-orphans" in seen["cmd"]
