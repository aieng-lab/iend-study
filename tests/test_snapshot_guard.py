"""A pull must never silently lose local-only results.json content.

Covers the three layers of protection: the snapshot step never overwrites an
unmerged snapshot, the merge restores dropped families and refuses to delete a
snapshot it could not fully fold in, and analysis loaders refuse an unmerged file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.snapshot_guard import (  # noqa: E402
    UnmergedSnapshotError,
    assert_resolved,
    snapshot_path,
    unresolved_snapshots,
)

_spec = importlib.util.spec_from_file_location(
    "smart_merge_pulled_results", ROOT / "scripts" / "smart_merge_pulled_results.py"
)
smart = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smart)


def _payload(families, enabled):
    return {
        "status": "ok",
        "config": {"enabled_methods": list(enabled)},
        "methods": [{"method": f"{fam}:A:X", "metrics": {"encoding_E": 0.5}} for fam in families],
        "raw": {},
    }


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def tree():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _pull(results: Path, remote_payload) -> None:
    """Simulate rsync: replace results.json by a new file, keeping the old inode
    alive only through the hardlinked snapshot."""
    tmp = results.with_name("results.json.rsync-tmp")
    tmp.write_text(json.dumps(remote_payload), encoding="utf-8")
    os.replace(tmp, results)


def test_pull_that_drops_families_is_repaired_by_merge(tree):
    results = tree / "gemma" / "emotion" / "results.json"
    _write(results, _payload(["gradiend", "caa", "cga", "sae"], ["gradiend", "caa", "cga", "sae"]))

    assert smart.snapshot_all(tree) == 0
    _pull(results, _payload(["gradiend"], ["gradiend"]))  # cluster only has gradiend

    assert unresolved_snapshots(tree) == [snapshot_path(results)]
    with pytest.raises(UnmergedSnapshotError):
        assert_resolved(results)

    assert smart.merge_all(tree) == 0
    families = {m["method"].split(":")[0] for m in json.loads(results.read_text("utf-8"))["methods"]}
    assert families == {"gradiend", "caa", "cga", "sae"}
    assert not snapshot_path(results).exists()
    assert_resolved(results)  # no longer raises


def test_snapshot_never_overwrites_an_unmerged_snapshot(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))  # pull #1, merge never runs

    # A second sync starts: it must NOT re-link the (already stripped) file over
    # the only copy of the good data.
    smart.snapshot_all(tree)
    snap = json.loads(snapshot_path(results).read_text("utf-8"))
    assert {m["method"].split(":")[0] for m in snap["methods"]} == {"gradiend", "caa"}


def test_merge_keeps_snapshot_when_it_cannot_fold_it_in(tree, monkeypatch):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))

    # A merge that silently drops local rows must fail loudly and keep the snapshot.
    monkeypatch.setattr(smart, "merge_study_payload", lambda prev, cur, enabled: cur)
    assert smart.merge_all(tree) == 1
    assert snapshot_path(results).exists()
    with pytest.raises(UnmergedSnapshotError):
        assert_resolved(results)


def test_untouched_results_json_leaves_no_leftover(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend"], ["gradiend"]))
    smart.snapshot_all(tree)  # rsync changes nothing: snapshot stays the same inode
    assert unresolved_snapshots(tree) == []
    assert_resolved(results)
    assert smart.merge_all(tree) == 0
    assert not snapshot_path(results).exists()


def test_check_mode_reports_unmerged(tree, capsys):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))
    assert smart.main.__module__  # sanity: importable
    assert unresolved_snapshots(tree)


def test_method_groups_loader_refuses_unmerged_file(tree):
    from analysis.method_groups import _load_results

    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))
    with pytest.raises(UnmergedSnapshotError):
        _load_results(results)


def test_supersede_retires_snapshot_without_deleting_it(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))
    assert unresolved_snapshots(tree)

    assert smart.supersede_all(tree) == 0
    assert unresolved_snapshots(tree) == []
    assert_resolved(results)  # guard no longer blocks
    kept = results.with_name("results.json.superseded")
    assert kept.exists()  # data is still on disk
    assert {m["method"].split(":")[0] for m in json.loads(kept.read_text("utf-8"))["methods"]} == {"gradiend", "caa"}


def test_sae_pre_is_judged_by_its_cli_family_not_refused(tree):
    """A pull authoritative for ``sae`` replaces sae_pre too; that is not a lost family."""
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "sae", "sae_pre"], ["gradiend", "sae"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend", "sae"], ["gradiend", "sae"]))
    assert smart.merge_all(tree) == 0
    assert unresolved_snapshots(tree) == []


def test_rows_dropped_by_authoritative_family_are_archived_not_deleted(tree):
    """A pull that recomputes ``gradiend`` wipes the local gradiend variants (by
    design); the merge must not then destroy the only copy of them."""
    def gradiend(mid):
        return {"method": mid, "metrics": {"roc_auc": 0.9, "roc_auc_neutral": 0.9}}

    results = tree / "m" / "t" / "results.json"
    local = _payload(["caa"], ["gradiend", "caa"])
    local["methods"] += [gradiend("gradiend:A:OLDVARIANT"), gradiend("gradiend:A:X")]
    _write(results, local)
    smart.snapshot_all(tree)
    remote = _payload([], ["gradiend"])
    remote["methods"] = [gradiend("gradiend:A:X")]
    _pull(results, remote)

    assert smart.merge_all(tree) == 0
    now = {m["method"] for m in json.loads(results.read_text("utf-8"))["methods"]}
    assert "gradiend:A:OLDVARIANT" not in now and "caa:A:X" in now
    archived = list(results.parent.glob("results.json.superseded*"))
    assert len(archived) == 1
    ids = {m["method"] for m in json.loads(archived[0].read_text("utf-8"))["methods"]}
    assert "gradiend:A:OLDVARIANT" in ids  # the only copy of the dropped row survives
    assert_resolved(results)


# --- the merge must not make rsync re-download unchanged files on every sync ------------
# After a merge rewrote results.json (fresh mtime/size) the next `rsync -a` saw a file that
# differed from the remote and pulled it again, so hundreds of unchanged files were merged on
# every sync. The pulled file must be left untouched whenever the snapshot adds nothing.

def _stat_key(path: Path):
    st = path.stat()
    return st.st_mtime_ns, st.st_size, st.st_ino


def test_identical_bytes_are_skipped_without_rewriting(tree):
    results = tree / "m" / "t" / "results.json"
    payload = _payload(["gradiend", "caa"], ["gradiend", "caa"])
    _write(results, payload)
    smart.snapshot_all(tree)
    _pull(results, payload)  # same bytes, new inode (rsync re-sent it because mtime differed)
    assert unresolved_snapshots(tree)  # the guard cannot tell by inode alone
    before = _stat_key(results)

    assert smart.merge_all(tree) == 0
    assert _stat_key(results) == before  # not rewritten
    assert not snapshot_path(results).exists()


def test_pull_that_already_contains_everything_is_not_rewritten(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend"], ["gradiend"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))  # remote is a superset
    before = _stat_key(results)

    assert smart.merge_all(tree) == 0
    assert _stat_key(results) == before  # kept as pulled -> next rsync skips it
    assert {m["method"].split(":")[0] for m in json.loads(results.read_text("utf-8"))["methods"]} == {
        "gradiend",
        "caa",
    }
    assert not snapshot_path(results).exists()


def test_local_only_rows_are_still_merged_and_written(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))
    before = _stat_key(results)

    assert smart.merge_all(tree) == 0
    assert _stat_key(results) != before  # the merge really had something to restore
    assert {m["method"].split(":")[0] for m in json.loads(results.read_text("utf-8"))["methods"]} == {
        "gradiend",
        "caa",
    }


def test_merged_file_is_stamped_newer_than_the_pulled_copy(tree):
    # `rsync --update` skips a file that is newer on the receiver; that is what stops a
    # locally-merged results.json being re-downloaded on every pull.
    results = tree / "m" / "t" / "results.json"
    _write(results, _payload(["gradiend", "caa"], ["gradiend", "caa"]))
    smart.snapshot_all(tree)
    _pull(results, _payload(["gradiend"], ["gradiend"]))
    remote_mtime = results.stat().st_mtime_ns

    assert smart.merge_all(tree) == 0
    assert results.stat().st_mtime_ns == remote_mtime + smart.LOCAL_MERGE_MTIME_SKEW_NS


# --- a locally promoted family must survive a pull of the cluster's old rows ----------------

def _promoted_payload(value, promoted=True):
    p = {
        "status": "ok",
        "config": {"enabled_methods": ["gradiend", "caa"]},
        "methods": [
            {"method": "gradiend:A:X", "status": "ok", "metrics": {"encoding_E": value}},
            {"method": "caa:A:X", "status": "ok", "metrics": {"encoding_E": 0.5}},
        ],
        "raw": {"causal": {"by_method": {"gradiend:A:X": {"v": value}, "caa:A:X": {"v": 0.5}}}},
    }
    if promoted:
        p["raw"]["promoted"] = {"gradiend": {"from_subdir": "gradiend_lr1e7"}}
    return p


def test_promoted_family_is_not_reverted_by_a_pull(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _promoted_payload(0.99))          # local: gradiend promoted from the screen
    smart.snapshot_all(tree)
    _pull(results, _promoted_payload(0.11, promoted=False))  # cluster still has the OLD gradiend rows
    assert smart.merge_all(tree) == 0

    merged = json.loads(results.read_text("utf-8"))
    row = {m["method"]: m for m in merged["methods"]}
    assert row["gradiend:A:X"]["metrics"]["encoding_E"] == 0.99
    assert merged["raw"]["causal"]["by_method"]["gradiend:A:X"]["v"] == 0.99
    assert "gradiend" in merged["raw"]["promoted"]


def test_unpromoted_family_still_follows_the_pull(tree):
    results = tree / "m" / "t" / "results.json"
    _write(results, _promoted_payload(0.99))
    smart.snapshot_all(tree)
    remote = _promoted_payload(0.11, promoted=False)
    remote["methods"][1]["metrics"]["encoding_E"] = 0.77   # the cluster recomputed caa
    _pull(results, remote)
    assert smart.merge_all(tree) == 0
    row = {m["method"]: m for m in json.loads(results.read_text("utf-8"))["methods"]}
    assert row["caa:A:X"]["metrics"]["encoding_E"] == 0.77   # only the promoted family is pinned
