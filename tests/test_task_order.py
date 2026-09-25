"""Canonical task ordering used by TASKS=all launchers and paper tables."""

from analysis.task_order import PAPER_TASK_ORDER, order_tasks
from study.config import list_tasks


def test_paper_tables_follow_the_requested_task_sequence():
    assert PAPER_TASK_ORDER[:15] == (
        "gender_en", "emotion", "race", "religion", "ravel_continent",
        "ravel_country", "ravel_language", "language", "pronoun_number",
        "pronoun_person", "ioi_mib", "key_value", "induction", "repetition",
        "function_composition",
    )
    assert order_tasks(["key_value", "emotion", "race", "unknown", "ioi_mib"]) == [
        "emotion", "race", "ioi_mib", "key_value", "unknown",
    ]


def test_list_tasks_orders_multiclass_then_binary_then_one_pole():
    assert list_tasks() == [
        # Multi-class (n=3).
        "language",
        "pronoun_person",
        "race",
        "ravel_continent",
        "ravel_country",
        "ravel_language",
        "religion",
        # Binary (n=2).
        "emotion",
        "gender_en",
        "pronoun_number",
        # One-pole-only.
        "function_composition",
        "induction",
        "ioi_mib",
        "key_value",
        "repetition",
    ]
