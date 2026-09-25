from __future__ import annotations

from study.data.synthetic import REPETITION_COUNTS, build_repetition
from study.config import load_study_config
from study.tasks import TaskBundle, subset_task_bundle


def _bundle():
    frame = build_repetition(
        rows_per_split={"train": 10, "validation": 10, "test": 10},
        seed=3,
        repetition_counts=(1, 4),
    )
    yes = frame[["masked", "split", "label"]].rename(columns={"label": "YES"})
    no = frame[["masked", "split", "alternative"]].rename(
        columns={"alternative": "NO"}
    )
    return TaskBundle(
        task_id="repetition",
        classes=["YES", "NO"],
        data_per_class={"YES": yes, "NO": no},
        merged_df=frame,
        neutrals=frame.iloc[:0].copy(),
    )


def test_repetition_generator_balances_levels_and_keeps_explicit_counterfactual():
    frame = _bundle().merged_df
    assert set(frame["repetition_count"]) == {1, 4}
    assert set(frame["label"]) == {"yes"}
    assert set(frame["alternative"]) == {"no"}
    assert frame.groupby(["split", "repetition_count"]).size().min() == 5
    assert frame["masked"].str.endswith("yes [MASK]").all()
    assert frame["masked"].nunique() == len(frame)
    assert frame["context_family"].nunique() > 1


def test_repetition_context_is_matched_across_confidence_levels():
    frame = _bundle().merged_df
    for (_split, _context_id), group in frame.groupby(["split", "context_id"]):
        assert set(group["repetition_count"]) == {1, 4}
        assert group["context_family"].nunique() == 1
        assert group["context_role"].nunique() == 1
        assert group["context_topic"].nunique() == 1
        prefixes = {
            text.rsplit("yes", int(count))[0]
            for text, count in zip(group["masked"], group["repetition_count"])
        }
        assert len(prefixes) == 1


def test_repetition_delimiters_are_balanced_and_fixed_within_context():
    frame = build_repetition(
        rows_per_split={"train": 32},
        seed=0,
        repetition_counts=(1, 4),
        repetition_delimiters=("space", "comma", "semicolon", "pipe"),
    )
    assert frame.groupby(["split", "repetition_count", "repetition_delimiter"]).size().nunique() == 1
    for (_split, _context_id), group in frame.groupby(["split", "context_id"]):
        assert group["repetition_delimiter"].nunique() == 1
        assert set(group["repetition_count"]) == {1, 4}

    examples = {
        row.repetition_delimiter: row.masked
        for row in frame.itertuples()
        if int(row.repetition_count) == 4
    }
    assert examples["space"].endswith("yes yes yes yes [MASK]")
    assert examples["comma"].endswith("yes, yes, yes, yes, [MASK]")
    assert examples["semicolon"].endswith("yes; yes; yes; yes; [MASK]")
    assert examples["pipe"].endswith("yes | yes | yes | yes | [MASK]")


def test_subset_task_bundle_reconstructs_one_pole_alternative_class():
    subset = subset_task_bundle(_bundle(), column="repetition_count", value="4")
    assert set(subset.merged_df["repetition_count"]) == {4}
    assert set(subset.data_per_class) == {"YES", "NO"}
    assert set(subset.data_per_class["YES"]["YES"]) == {"yes"}
    assert set(subset.data_per_class["NO"]["NO"]) == {"no"}


def test_repetition_yaml_preserves_string_class_ids():
    cfg = load_study_config(model="pythia-70m-deduped", task="repetition")
    assert cfg.task["classes"] == ["YES", "NO"]
    assert cfg.training["one_pole_classes"] == ["YES"]


def test_study_repetition_defaults_drop_n1():
    assert 1 not in REPETITION_COUNTS
    assert min(REPETITION_COUNTS) >= 4
