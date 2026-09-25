"""score_class_probs_by_dataset used to silently fall back to a hardcoded
gender {'M': ['he'], 'F': ['she']} word-probe dict for ANY task whenever its
frame lacked row-wise factual/alternative columns (causal_eval.py, fixed
2026-08-21). That's wrong for every non-gender task and either crashed later
with an opaque KeyError (pronoun_number: "no probs_by_dataset panels
containing 'plural'") or, worse, could have silently mis-scored a task whose
classes happened to collide with 'M'/'F'. Fixed by deriving per-class word
targets from the frame's own label/label_class columns instead, and raising
loudly when that isn't possible. See causal_eval.py::score_class_probs_by_dataset
and causal_eval.py::_class_word_targets_from_frame.
"""

import pandas as pd

from causal_eval import _class_word_targets_from_frame


def test_derives_class_word_targets_from_frame_labels():
    df = pd.DataFrame(
        {
            "masked": [f"row {i}" for i in range(6)],
            "label": ["they", "we", "them", "he", "she", "him"],
            "label_class": ["plural", "plural", "plural", "singular", "singular", "singular"],
        }
    )

    targets = _class_word_targets_from_frame(df, dataset_class_col="label_class")

    assert targets == {
        "plural": ["them", "they", "we"],
        "singular": ["he", "him", "she"],
    }


def test_excludes_neutral_group_from_derived_targets():
    df = pd.DataFrame(
        {
            "masked": ["a", "b", "c"],
            "label": ["they", "he", "whatever"],
            "label_class": ["plural", "singular", "neutral"],
        }
    )

    targets = _class_word_targets_from_frame(df, dataset_class_col="label_class")

    assert "neutral" not in targets
    assert set(targets) == {"plural", "singular"}


def test_returns_empty_when_frame_cannot_derive_targets():
    """No 'label'/'factual' column at all -> caller must treat this as
    "cannot score", not silently substitute another task's words."""
    df = pd.DataFrame({"masked": ["a", "b"], "label_class": ["plural", "singular"]})

    assert _class_word_targets_from_frame(df, dataset_class_col="label_class") == {}


def test_no_hardcoded_gender_word_default_remains_in_source():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "causal_eval.py").read_text(
        encoding="utf-8"
    )
    assert "GENDER_CLASS_TARGETS" not in source
    assert '"M": ["he"]' not in source
    assert "targets or GENDER_CLASS_TARGETS" not in source
