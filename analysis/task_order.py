"""Canonical task order for every paper-facing table and figure.

Keep this list presentation-oriented: it is deliberately independent from the
launcher/config order, which groups tasks by their data construction.  Unknown
or experimental tasks sort after the published suite, alphabetically, so adding
an analysis cannot silently reshuffle established columns.
"""

from __future__ import annotations

from typing import Sequence, TypeVar


# Requested paper sequence.  These are the visible task IDs used by the
# current study; ``ioi_mib`` is rendered as ``IOI`` in the LaTeX macros.
PAPER_TASK_ORDER: tuple[str, ...] = (
    "gender_en",
    "emotion",
    "race",
    "religion",
    "ravel_continent",
    "ravel_country",
    "ravel_language",
    "language",
    "pronoun_number",
    "pronoun_person",
    "ioi_mib",
    "key_value",
    "induction",
    "repetition",
    "function_composition",
    # Deprecated hidden checks are retained as known IDs so ad-hoc diagnostic
    # tables remain deterministic without appearing in the published suite.
    "race_one_pole",
    "religion_one_pole",
)

# Legacy IOI results use this ID; make them appear at the IOI position too.
_TASK_RANK = {task: index for index, task in enumerate(PAPER_TASK_ORDER)}
_TASK_RANK["ioi"] = _TASK_RANK["ioi_mib"]

T = TypeVar("T")


def task_sort_key(task: str) -> tuple[int, str]:
    """Stable paper order, with unknown task identifiers sorted last."""
    text = str(task)
    return (_TASK_RANK.get(text, len(_TASK_RANK)), text)


def order_tasks(tasks: Sequence[T], *, task_of=str) -> list[T]:
    """Return *tasks* in canonical paper order without dropping unknown IDs."""
    return sorted(tasks, key=lambda value: task_sort_key(str(task_of(value))))
