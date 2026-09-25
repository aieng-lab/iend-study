"""Keep a bare ``pytest`` invocation cheap on memory-constrained machines.

pytest imports *every* collected test file before applying ``-k``/``-m``
filters — so ``pytest -k "sae"`` still pays the full import cost of files
it will end up deselecting. The files below ``import torch`` at module
scope, which is the single biggest import-time RAM cost in this repo
(observed: a broad ``-k`` sweep across the whole ``tests/`` tree exhausted
16GB of RAM on a local dev machine, 2026-08-18).

Set ``GRADIEND_RUN_HEAVY_TESTS=1`` to include them, or run a heavy file
directly by path (``pytest tests/test_neutral_protocol.py``) — direct
invocation bypasses this list entirely since it's collection-glob based,
not a marker.
"""

from __future__ import annotations

import os

_HEAVY_TEST_FILES = [
    "test_actiend_signal_scale_and_claim.py",
    "test_activation_protocol.py",
    "test_caa_neutral_tokens.py",
    "test_cga_eval.py",
    "test_clm_score_patch.py",
    "test_neutral_protocol.py",
]

if os.environ.get("GRADIEND_RUN_HEAVY_TESTS", "").strip().lower() not in {
    "1",
    "true",
    "yes",
}:
    collect_ignore_glob = list(_HEAVY_TEST_FILES)
    print(
        f"tests/conftest.py: skipping {len(_HEAVY_TEST_FILES)} torch-importing "
        "test file(s) (set GRADIEND_RUN_HEAVY_TESTS=1 to include them): "
        + ", ".join(_HEAVY_TEST_FILES),
        flush=True,
    )
