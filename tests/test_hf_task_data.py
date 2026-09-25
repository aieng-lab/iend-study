from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from study.tasks import load_hf_splits
from study.tasks import prior


def test_run_study_fail_fast_reraises_first_task_error(monkeypatch):
    import run_study

    calls = []

    def fail_on_first_task(**kwargs):
        calls.append(kwargs["task"])
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(run_study, "list_tasks", lambda **_kw: ["first", "second"])
    monkeypatch.setattr(run_study, "run_study", fail_on_first_task)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_study.py", "--model", "gpt2-small", "--fail-fast"],
    )

    with pytest.raises(RuntimeError, match="intentional failure"):
        run_study.main()

    assert calls == ["first"]




class _FakeSplit:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self.frame.copy()


def test_load_hf_splits_uses_published_split(monkeypatch):
    ds = {
        "train": _FakeSplit(pd.DataFrame({"value": [1, 2], "split": ["old", "old"]})),
        "val": _FakeSplit(pd.DataFrame({"value": [3], "split": ["old"]})),
    }
    calls = []

    def fake_load_dataset(*args):
        calls.append(args)
        return ds

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    frame = load_hf_splits("org/data", config_name="held-out", max_rows_per_split=1)

    assert calls == [("org/data", "held-out")]
    assert frame.to_dict("records") == [
        {"value": 1, "split": "train"},
        {"value": 3, "split": "validation"},
    ]


def test_pronoun_smoke_keeps_all_classes_and_splits(monkeypatch):
    rows = []
    classes = ["1SG", "1PL", "2SGPL", "3SG", "3PL"]
    for split in ("train", "validation", "test"):
        for cls in classes:
            rows.extend(
                {"masked": f"context {i} [MASK]", "label": cls.lower(), "label_class": cls, "split": split}
                for i in range(50)
            )
    monkeypatch.setattr(prior, "load_hf_splits", lambda _hf_id: pd.DataFrame(rows))

    frame = prior._load_pronoun_data({}, smoke=True)

    assert len(frame) == 3 * 5 * 40
    assert set(frame["label_class"]) == set(classes)
    assert set(frame["split"]) == {"train", "validation", "test"}


def test_uploaded_dataset_configs_replace_runtime_generators():
    expected = {
        "emotion": (
            "aieng-lab/en-sentiment-nrc",
            "aieng-lab/en-sentiment-nrc-neutral",
        ),
        "pronoun_person": (
            "aieng-lab/en-pronouns",
            "aieng-lab/en-pronoun-neutral",
        ),
        "pronoun_number": (
            "aieng-lab/en-pronouns",
            "aieng-lab/en-pronoun-neutral",
        ),
    }
    for task_id, (dataset, neutral) in expected.items():
        with open(f"configs/tasks/{task_id}.yaml", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)["data"]
        assert data["hf_dataset"] == dataset
        assert data["neutral_hf"] == neutral
        assert not ({"local_path", "local_dir", "ensure_via", "force_regenerate"} & data.keys())


def test_labeled_df_for_eval_applies_class_merge_map():
    from study.tasks import TaskBundle, labeled_df_for_eval

    rows = [
        {"masked": "a [MASK]", "label": "he", "label_class": "3SG", "split": "validation"},
        {"masked": "b [MASK]", "label": "they", "label_class": "3PL", "split": "validation"},
        {"masked": "c [MASK]", "label": "I", "label_class": "1SG", "split": "train"},
        {"masked": "d [MASK]", "label": "we", "label_class": "1PL", "split": "train"},
        {"masked": "e [MASK]", "label": "you", "label_class": "2SGPL", "split": "test"},
    ]
    bundle = TaskBundle(
        task_id="pronoun_number",
        classes=["singular", "plural"],
        data_per_class={},
        merged_df=pd.DataFrame(rows),
        neutrals=pd.DataFrame({"text": ["n"]}),
        class_merge_map={
            "singular": ["1SG", "3SG"],
            "plural": ["1PL", "3PL"],
        },
    )
    out = labeled_df_for_eval(bundle)
    assert set(out["label_class"]) == {"singular", "plural"}
    assert "2SGPL" not in set(out["label_class"])
    assert set(out["raw_label_class"]) == {"1SG", "1PL", "3SG", "3PL"}
    assert len(out) == 4


def test_labeled_df_for_eval_noop_without_merge_map():
    from study.tasks import TaskBundle, labeled_df_for_eval

    df = pd.DataFrame(
        [
            {"masked": "a", "label": "M", "label_class": "M", "split": "train"},
            {"masked": "b", "label": "F", "label_class": "F", "split": "train"},
        ]
    )
    bundle = TaskBundle(
        task_id="gender_en",
        classes=["M", "F"],
        data_per_class={},
        merged_df=df,
        neutrals=pd.DataFrame({"text": ["n"]}),
    )
    assert labeled_df_for_eval(bundle) is df


def test_one_pole_expand_and_encode_classes():
    from study.tasks import (
        TaskBundle,
        encode_target_classes,
        expand_one_pole_cf_rows,
        labeled_df_for_eval,
    )

    df = pd.DataFrame(
        [
            {
                "masked": "a b a [MASK]",
                "label": "b",
                "label_class": "MATCH",
                "alternative": "x",
                "alternative_class": "DISTRACTOR",
                "split": "validation",
            },
            {
                "masked": "c d c [MASK]",
                "label": "d",
                "label_class": "MATCH",
                "alternative": "y",
                "alternative_class": "DISTRACTOR",
                "split": "test",
            },
        ]
    )
    bundle = TaskBundle(
        task_id="induction",
        classes=["MATCH", "DISTRACTOR"],
        data_per_class={},
        merged_df=df,
        neutrals=pd.DataFrame({"text": ["n"]}),
    )
    sae_df = labeled_df_for_eval(bundle, expand_one_pole=False)
    assert encode_target_classes(bundle, sae_df) == ["MATCH"]
    assert set(sae_df["label_class"]) == {"MATCH"}

    caa_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    assert set(caa_df["label_class"]) == {"MATCH", "DISTRACTOR"}
    assert len(caa_df) == 4
    assert encode_target_classes(bundle, caa_df) == ["MATCH", "DISTRACTOR"]
    expanded = expand_one_pole_cf_rows(df, classes=bundle.classes)
    assert (expanded["label_class"] == "DISTRACTOR").sum() == 2
    cf = expanded[expanded["label_class"] == "DISTRACTOR"].iloc[0]
    assert cf["label"] == "x"
    assert cf["alternative"] == "b"
    assert cf["alternative_class"] == "MATCH"


def test_ioi_one_pole_sae_vs_caa_classes():
    """IOI: SUBJECT is CF-only; SAE must not require it on label_class."""
    from study.method_ids import label_tokens_from_config
    from study.tasks import TaskBundle, encode_target_classes, labeled_df_for_eval

    df = pd.DataFrame(
        [
            {
                "masked": "When Eric and Jack went, Eric gave a note to [MASK]",
                "label": "Jack",
                "label_class": "IO",
                "alternative": "Eric",
                "alternative_class": "SUBJECT",
                "split": "test",
            }
        ]
    )
    bundle = TaskBundle(
        task_id="ioi",
        classes=["IO", "SUBJECT"],
        data_per_class={},
        merged_df=df,
        neutrals=pd.DataFrame({"text": ["n"]}),
    )
    sae_df = labeled_df_for_eval(bundle, expand_one_pole=False)
    assert encode_target_classes(bundle, sae_df) == ["IO"]
    caa_df = labeled_df_for_eval(bundle, expand_one_pole=True)
    assert encode_target_classes(bundle, caa_df) == ["IO", "SUBJECT"]
    # Do not poison CAA prediction-fill with class ids as tokens.
    assert label_tokens_from_config(["IO", "SUBJECT"], {}) == {}
    assert label_tokens_from_config(["M", "F"], {"target_tokens": {"M": "he", "F": "she"}}) == {
        "M": "he",
        "F": "she",
    }


def test_identify_sae_drops_absent_cf_class(monkeypatch):
    """Passing SUBJECT in target_classes must soft-drop, not raise."""
    import numpy as np
    import sae_eval as se

    labels = ["IO", "IO", "IO", "IO"]
    latents = np.random.randn(4, 8).astype(np.float32)

    # Bypass model/SAE load: inject after encoding by stubbing identify internals.
    calls = {}

    def fake_select_per_class(lat, lab, **kw):
        calls["target_classes"] = list(kw.get("target_classes") or [])
        assert "SUBJECT" not in calls["target_classes"]
        assert calls["target_classes"] == ["IO"]
        return se.SAEFeatureSelection(
            feature_indices=[0],
            scores={0: 1.0},
            mean_by_class={"IO": {0: 1.0}},
            contrast_pair=("IO", "IO"),
            mode="per_class",
            layer=0,
            sae_id="id",
            features_by_class={"IO": [0]},
        )

    monkeypatch.setattr(se, "select_per_class_features", fake_select_per_class)
    monkeypatch.setattr(se, "load_sae", lambda *a, **k: object())
    monkeypatch.setattr(se, "resid_sites", lambda layer: ("mod", "id"))
    monkeypatch.setattr(se, "collect_study_activations", lambda *a, **k: np.zeros((4, 8)))
    monkeypatch.setattr(se, "collect_study_neutral_activations", lambda *a, **k: np.zeros((4, 8)))
    monkeypatch.setattr(se, "encode_sae", lambda *a, **k: latents)
    monkeypatch.setattr(se, "active_sae_release", lambda: "dummy")

    df = pd.DataFrame(
        {
            "masked": [f"t{i}" for i in range(4)],
            "label_class": labels,
            "split": ["validation"] * 4,
        }
    )
    sels, *_ = se.identify_sae_features(
        object(),
        object(),
        df,
        layer=0,
        modes=["per_class", "joint"],
        target_classes=["IO", "SUBJECT"],
        class_a="IO",
        class_b="SUBJECT",
        progress=lambda m: None,
    )
    assert "per_class" in sels
    assert "joint" not in sels  # bipolar soft-skipped
    assert calls["target_classes"] == ["IO"]


def test_claim_classes_one_pole():
    from study.tasks import TaskBundle, claim_classes_for_study

    df = pd.DataFrame(
        [
            {
                "masked": "a b a [MASK]",
                "label": "b",
                "label_class": "MATCH",
                "alternative": "x",
                "alternative_class": "DISTRACTOR",
            }
        ]
    )
    bundle = TaskBundle(
        task_id="induction",
        classes=["MATCH", "DISTRACTOR"],
        data_per_class={},
        merged_df=df,
        neutrals=pd.DataFrame({"text": ["n"]}),
    )
    assert claim_classes_for_study(
        bundle, one_pole=True, one_pole_classes=["MATCH"]
    ) == ["MATCH"]
    assert claim_classes_for_study(bundle, one_pole=True) == ["MATCH"]
    assert claim_classes_for_study(bundle, one_pole=False) == ["MATCH", "DISTRACTOR"]


def test_primary_fair_methods_uses_claim_classes():
    from results_schema import primary_fair_methods, require_claim_classes

    results = {
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "enabled_methods": ["gradiend", "actiend", "sae", "caa"],
        }
    }
    assert require_claim_classes(results) == ["MATCH"]
    methods = primary_fair_methods(results)
    assert "gradiend:MATCH" in methods
    assert "actiend:MATCH" in methods
    assert "sae:MATCH:kstar" in methods
    assert "caa:MATCH:act_prediction" in methods
    assert all("DISTRACTOR" not in m for m in methods)


def test_encoder_ablation_methods_hides_sae_distractor_ghosts():
    from results_schema import encoder_ablation_methods

    results = {
        "config": {
            "target_classes": ["MATCH", "DISTRACTOR"],
            "claim_classes": ["MATCH"],
            "enabled_methods": ["sae"],
        },
        "methods": [
            {
                "method": "sae:MATCH:all_k1",
                "status": "ok",
                "metrics": {"roc_auc_neutral": 1.0},
            },
            {
                "method": "sae:DISTRACTOR:k1",
                "status": "error",
                "error": "no feature",
                "metrics": {},
                "extras": {"causal": {"meta": {"error": "no feature"}}},
            },
        ],
        "raw": {
            "causal": {
                "by_method": {
                    "sae:DISTRACTOR:k1": {"meta": {"error": "no feature"}},
                    "sae:MATCH:k1": {"meta": {}},
                }
            }
        },
    }
    methods = encoder_ablation_methods(results)
    assert "sae:MATCH:all_k1" in methods
    assert "sae:MATCH:k1" in methods
    assert all("DISTRACTOR" not in m for m in methods)


def test_sae_encode_classes_one_pole():
    from sae_eval import sae_encode_classes

    raw = {
        "selections": {
            "per_class": {
                "features_by_class": {"MATCH": [3, 7], "DISTRACTOR": []},
            }
        },
        "selected_layer_by_class": {"MATCH": 5},
        "by_layer": {
            "5": {
                "selections": {
                    "per_class": {"features_by_class": {"MATCH": [3]}},
                }
            }
        },
    }
    assert sae_encode_classes(raw) == ["MATCH"]


def test_language_task_is_three_class_monolingual():
    from study.data.synthetic import SMOKE_ROWS, build_language_with_neutrals

    df, neu = build_language_with_neutrals(rows_per_split=SMOKE_ROWS, seed=0, use_fixture=True)
    assert not df.empty
    assert set(df["label_class"].astype(str)) == {"en", "fr", "de"}
    assert set(df["alternative_class"].astype(str)) <= {"en", "fr", "de"}
    assert (df["label_class"].astype(str) != df["alternative_class"].astype(str)).all()
    assert (df["label"].astype(str) != df["alternative"].astype(str)).any()
    assert df["masked"].astype(str).str.contains(r"\[MASK\]").all()
    for sample in df["masked"].astype(str):
        assert sample.startswith("[")
        assert "\n" not in sample
        assert sample.rstrip().endswith("[MASK]")
        assert sample[1:3] in {"en", "fr", "de"}
        body = sample.split("]", 1)[-1].split("[MASK]", 1)[0]
        assert len(body.split()) >= 10

    assert not neu.empty
    # Deliberately genuinely unmasked, unlike df["masked"] above: this is the
    # external neutral corpus, not a training/eval cloze pair, and
    # gradiend's create_masked_pair_from_text (which derives its own mask
    # position/target) raises if handed text that already contains "[MASK]" --
    # confirmed live 2026-08-22 on gpt2-small/language's fair encoder eval,
    # when this generator wrongly reused the same "[lang] ...[MASK]" cloze
    # template for the neutral rows' "text" column too. The "[lang] " tag
    # prefix is still kept (other code keys off it to identify source
    # language) -- only the cloze-specific "[MASK]"/prefix-only truncation is
    # gone.
    assert "masked" not in neu.columns
    neu_tags = set()
    for sample in neu["text"].astype(str):
        assert sample.startswith("[")
        assert "\n" not in sample
        assert "[MASK]" not in sample
        neu_tags.add(sample[1:3])
        body = sample.split("]", 1)[-1]
        assert len(body.split()) >= 10
    assert neu_tags <= {"es", "it", "nl", "pt"}
    assert neu_tags


def test_language_circuit_bundle_uses_other_language_neutrals():
    from study.data.synthetic import build_circuit_task

    cfg = {
        "id": "language",
        "classes": ["en", "fr", "de"],
        "data": {
            "classes": ["en", "fr", "de"],
            "neutral_langs": ["es", "it", "nl", "pt"],
        },
        "training": {"source": "both"},
        "ablations": {"pair": True, "one_pole": True},
        "neutral": {"max_rows": 8, "val_rows": 4, "train_rows": 8},
        "smoke": {"neutral": {"max_rows": 8, "val_rows": 4, "train_rows": 8}},
    }
    bundle = build_circuit_task("language", cfg, smoke=True)
    assert bundle.classes == ["en", "fr", "de"]
    assert set(bundle.data_per_class) == {"en", "fr", "de"}
    texts = bundle.neutrals["text"].astype(str)
    tags = {t[1:3] for t in texts if t.startswith("[")}
    assert tags <= {"es", "it", "nl", "pt"}
    assert tags


def test_circuit_task_excluded_words_ignore_missing_labels(monkeypatch):
    from study.data import synthetic

    task = {
        "id": "ioi",
        "classes": ["a", "b"],
        "data": {"force_regenerate": False},
        "neutral": {"max_rows": 1, "val_rows": 1, "train_rows": 1},
    }
    frame = pd.DataFrame(
        {
            "masked": ["x [MASK]", "y [MASK]"],
            "label": ["Alice", float("nan")],
            "label_class": ["a", "a"],
            "alternative": ["Bob", float("nan")],
            "split": ["train", "validation"],
        }
    )
    monkeypatch.setattr(synthetic, "ensure_synthetic", lambda *_args, **_kwargs: "unused.csv")
    monkeypatch.setattr(synthetic.pd, "read_csv", lambda _path: frame)
    monkeypatch.setattr(
        synthetic,
        "load_neutral_from_cfg",
        lambda *_args, **_kwargs: pd.DataFrame({"text": ["neutral"]}),
    )

    bundle = synthetic.build_circuit_task("ioi", task)

    assert bundle.excluded_words == ["alice", "bob"]


def test_merged_one_pole_rejects_non_task_alt_classes():
    from study.stages.train import _merged_one_pole_frame
    from study.tasks import TaskBundle

    df = pd.DataFrame(
        {
            "masked": ["a [MASK]", "b [MASK]"],
            "label": ["house", "water"],
            "label_class": ["en", "en"],
            "alternative": ["casa", "agua"],
            "alternative_class": ["OTHER", "es"],
            "split": ["train", "train"],
        }
    )
    bundle = TaskBundle(
        task_id="language",
        classes=["en", "fr"],
        data_per_class={},
        merged_df=df,
        neutrals=pd.DataFrame({"text": ["x"]}),
    )
    assert _merged_one_pole_frame(bundle) is None


def test_sae_rival_readout_maps_one_pole():
    import numpy as np
    import study.sae_engine as poc

    latents = np.arange(12, dtype=float).reshape(4, 3)
    rival = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    others, other_by, val_other_by, other_c = poc._sae_rival_readout_maps(
        "IO",
        ["IO"],
        ["IO", "IO", "IO", "IO"],
        latents,
        rival_test_by_class={"SUBJECT": rival},
        rival_val_by_class={"SUBJECT": rival},
    )
    assert others == ["SUBJECT"]
    assert other_c == "SUBJECT"
    assert np.array_equal(other_by["SUBJECT"], rival)
    assert np.array_equal(val_other_by["SUBJECT"], rival)


def test_encoder_labels_normalize_spaces_and_signed():
    import pandas as pd
    from sae_eval import _labels_from_encoder_df, _labels_for_target_class

    df = pd.DataFrame(
        {
            "source_id": ["United States", "china", "United States", "china"],
            "encoded": [0.1, 0.9, 0.2, 0.8],
        }
    )
    labels, col = _labels_from_encoder_df(
        df, class_a="united_states", class_b="china"
    )
    assert col == "source_id"
    assert labels is not None
    assert set(labels) >= {"united_states", "china"}

    signed = pd.DataFrame({"label": [1.0, -1.0, 1.0, -1.0], "encoded": [1, 2, 3, 4]})
    labels2, col2 = _labels_from_encoder_df(signed, class_a="christian", class_b="muslim")
    assert col2 == "label"
    assert list(labels2).count("christian") == 2

    one = pd.DataFrame({"source_id": ["jewish", "other"], "encoded": [1.0, 0.0]})
    labels3, _ = _labels_for_target_class(one, target_class="jewish")
    assert labels3 is not None
    assert (labels3 == "jewish").any()


def test_alias_legacy_factual_decoder_metrics():
    from causal_eval import _alias_legacy_factual_metrics

    entry = {
        "id": "c1",
        "lms": {"lms": 0.1},
        "christian_factual": 0.7,
        "jewish": 0.2,
        "probs": {},
    }
    out = _alias_legacy_factual_metrics(entry)
    assert out["probs"]["christian"] == 0.7
    assert out["probs_factual"]["christian"] == 0.7
