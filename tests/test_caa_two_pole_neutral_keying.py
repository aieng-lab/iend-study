"""Regression guard: CAA neutral scores must be keyed by ``vector_key``, not ``cls``.

``encode_caa_scores._score_side`` populates ``neu_concat_parts`` / ``neu_layer_parts``
keyed by ``vector_key``.  For a two-pole pair the ``vector_key`` is the pair
("F-M") while ``cls`` is a single pole ("F"), so reading the neutrals back by
``cls`` raised ``KeyError: 'F'`` and crashed every two-pole CAA encode (job
4065209, 2026-09-04).  The mean-path always read by ``vector_key``; the concat and
per-layer paths wrongly read by ``cls``.  A full end-to-end test needs a GPU model,
so this is a source-level guard (mirrors the ``GENDER_CLASS_TARGETS`` grep guard)
that the buggy ``[cls]`` indexing of the neutral dicts does not return.
"""

from __future__ import annotations

import re
from pathlib import Path

CAA_EVAL = Path(__file__).resolve().parents[1] / "caa_eval.py"


def test_neutral_dicts_are_not_indexed_by_cls():
    source = CAA_EVAL.read_text(encoding="utf-8")
    offenders = re.findall(r"neu_(?:concat|layer)_parts\[cls\]", source)
    assert not offenders, (
        "CAA neutral dicts are keyed by vector_key; indexing them by cls breaks "
        f"two-pole pairs. Found: {offenders}"
    )


def test_neutral_dicts_are_indexed_by_vector_key():
    source = CAA_EVAL.read_text(encoding="utf-8")
    # Both populate and read sites must use vector_key so one-pole (vector_key==cls)
    # and two-pole (vector_key==pair) stay consistent.
    assert "neu_concat_parts[vector_key]" in source
    assert "neu_layer_parts[vector_key]" in source
