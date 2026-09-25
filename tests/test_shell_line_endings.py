"""Shell scripts must use LF endings.

Editing a .sh file with Python's Path.write_text on Windows silently converts it
to CRLF, and bash on the cluster then fails with:

    slurm/iend_lr_cross_model.sh: line 18: $'\r': command not found
    set: pipefail: invalid option name

Every launcher broke this way on 2026-08-28. Read/write bytes when patching
shell scripts, or pass newline="".
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = sorted(
    p
    for d in ("slurm", "scripts")
    for p in (ROOT / d).rglob("*")
    if p.suffix in {".sh", ".sbatch"} and p.is_file()
)


def test_shell_scripts_exist():
    assert SHELL, "no shell scripts found - has the layout changed?"


@pytest.mark.parametrize("path", SHELL, ids=lambda p: p.name)
def test_no_carriage_returns(path):
    data = path.read_bytes()
    assert b"\r" not in data, (
        f"{path.relative_to(ROOT)} contains CR bytes; bash on the cluster will "
        "fail on every line. Patch shell scripts via read_bytes/write_bytes."
    )
