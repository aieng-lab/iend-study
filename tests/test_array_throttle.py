"""Every submitted array carries a concurrency cap (sbatch ``--array=<spec>%N``, the ArrayTaskThrottle)."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _snippet() -> str:
    text = (ROOT / "slurm" / "study_array.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.search(r'ARRAY_THROTTLE="\$\{ARRAY_THROTTLE:-3\}".*?\nwith_array_throttle\(\) \{.*?\n\}\n', text, re.S)
    assert match, "throttle block not found in study_array.sh"
    return match.group(0)


def _run(spec, throttle=None):
    env = dict(os.environ)
    env.pop("ARRAY_THROTTLE", None)
    if throttle is not None:
        env["ARRAY_THROTTLE"] = throttle
    return subprocess.run(
        [BASH, "-c", _snippet() + f'\nwith_array_throttle "{spec}"'], capture_output=True, text=True, env=env
    )


def test_default_cap_is_three():
    out = _run("0-4")
    assert out.returncode == 0 and out.stdout == "0-4%3"


@pytest.mark.parametrize("spec", ["0", "0-27", "0,2,5-7", "3-3"])
def test_appends_to_any_plain_spec(spec):
    assert _run(spec).stdout == f"{spec}%3"


def test_custom_cap():
    assert _run("0-9", "5").stdout == "0-9%5"


def test_zero_removes_the_cap():
    assert _run("0-9", "0").stdout == "0-9"


def test_an_existing_cap_is_left_alone():
    assert _run("0-9%2", "5").stdout == "0-9%2"


@pytest.mark.parametrize("bad", ["abc", "-1", "2.5", "3x"])
def test_an_invalid_value_fails_instead_of_silently_uncapping(bad):
    out = _run("0-4", bad)
    assert out.returncode != 0 and "Invalid ARRAY_THROTTLE" in out.stderr and out.stdout == ""


def test_the_submitting_function_uses_it_and_the_other_launcher_is_capped_too():
    study = (ROOT / "slurm" / "study_array.sh").read_text(encoding="utf-8")
    assert 'array_spec="$(with_array_throttle "$2")"' in study and '--array="${array_spec}"' in study
    iend = (ROOT / "slurm" / "iend_lr_cross_model.sh").read_text(encoding="utf-8")
    assert 'ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_THROTTLE}"' in iend
