"""A skipped causal id must always have a stored entry to restore.

Regression: on Llama-3.1-8B, AGIEND (`language`, `pronoun_person`) and CAGA
one-pole ids were reported "already complete" and skipped, yet never restored
into ``results.json``: ids invalidated for repair were re-added to the skip set
from progress checkpoints while the restore loop deliberately skipped them.
"""

from study.stages.causal import orphaned_skip_ids, stamp_progress_entry


def test_invalidated_id_readded_by_progress_is_orphaned():
    orphans = orphaned_skip_ids(
        {"agiend:en", "agiend:de-en:de"},
        progress_ids={"agiend:en", "agiend:de-en:de"},
        previous_ids={"agiend:en"},
        invalidated_ids={"agiend:en"},
    )
    assert orphans == {"agiend:en"}


def test_id_with_a_restorable_entry_is_kept():
    assert (
        orphaned_skip_ids(
            {"caga:asia"},
            progress_ids=set(),
            previous_ids={"caga:asia"},
            invalidated_ids=set(),
        )
        == set()
    )


def test_id_without_any_entry_is_orphaned():
    assert orphaned_skip_ids(
        {"caga:asia"}, progress_ids=set(), previous_ids=set(), invalidated_ids=set()
    ) == {"caga:asia"}


def test_progress_entry_is_stamped_from_its_file_without_mutating_input():
    entry = {"method": "agiend:en", "meta": {}}
    progress = {
        "direction_polarity_protocol_version": 2,
        "agiend_causal_grid_protocol_version": 2,
        "cga_causal_grid_protocol_version": 2,
    }
    stamped = stamp_progress_entry(entry, progress)
    assert stamped["meta"]["direction_polarity_protocol_version"] == 2
    assert stamped["meta"]["agiend_causal_grid_protocol_version"] == 2
    assert entry["meta"] == {}


def test_existing_entry_stamp_is_not_overwritten():
    stamped = stamp_progress_entry(
        {"meta": {"direction_polarity_protocol_version": 1}},
        {"direction_polarity_protocol_version": 2},
    )
    assert stamped["meta"]["direction_polarity_protocol_version"] == 1
