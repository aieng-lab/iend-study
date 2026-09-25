"""Slurm log sync and local pruning in scripts/rsync_analysis.sh.

Job logs live outside the repo and were never synced, so diagnosing why a job
did what it did always needed a manual ssh. They are pulled by mtime and old
local copies are deleted; the deletion is the part worth pinning.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/rsync_analysis.sh").read_text(encoding="utf-8")
# The local-prune block: from its directory-name guard to the next section.
_PRUNE_START = SCRIPT.index('if [[ "${LOCAL_LOG_DIR}" == */slurm-logs')
PRUNE_BLOCK = SCRIPT[_PRUNE_START : SCRIPT.index("# Explicit, out-of-band fetch", _PRUNE_START)]


class TestFetch:
    def test_selects_by_mtime_on_the_cluster(self):
        """The remote dir holds every job ever run; never pull all of it."""
        assert "-mtime -${SLURM_LOG_FETCH_DAYS}" in SCRIPT
        assert "--files-from=-" in SCRIPT

    def test_has_a_size_cap(self):
        assert 'SLURM_LOG_MAX_SIZE:-32m' in SCRIPT

    def test_can_be_disabled(self):
        assert '"${SYNC_SLURM_LOGS:-1}" != "0"' in SCRIPT

    def test_failure_is_non_fatal(self):
        """A missing log dir must not abort a results sync."""
        assert "note: slurm log sync failed (non-fatal)" in SCRIPT


class TestPruneSafety:
    def test_prune_is_guarded_on_the_directory_name(self):
        """A mis-set LOCAL_REPO must not turn this into a delete elsewhere."""
        assert '"${LOCAL_LOG_DIR}" == */slurm-logs' in SCRIPT

    def test_prune_is_scoped_to_slurm_files(self):
        assert "-name 'slurm-*'" in SCRIPT

    def test_prune_is_local_only(self):
        """The cluster's own logs must never be deleted."""
        prune = PRUNE_BLOCK
        assert "-delete" in prune
        assert "ssh" not in prune, "prune block must not reach the cluster"
        assert "${REMOTE}" not in prune

    def test_dry_run_counts_instead_of_deleting(self):
        block = PRUNE_BLOCK
        start = block.index("if ((${#DRY_RUN[@]})); then")
        dry = block[start : block.index("(dry-run) would delete", start)]
        assert "-delete" not in dry, "dry-run must not delete"
        assert "wc -l" in dry

    def test_retention_is_configurable(self):
        assert "SLURM_LOG_RETENTION_DAYS:=2" in SCRIPT


class TestNotCommitted:
    def test_local_log_dir_is_gitignored(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "slurm-logs/" in text
