"""Regression tests for multiclass religion/race/pronoun report gaps."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from caa_eval import _subset_df
from causal_study import iter_trainer_class_groups
from results_schema import resolve_encoder_metrics, tensors_all_alias
from sae_eval import _roc_auc
from study.config import StudyConfig, load_study_config
from study.training_profiles import build_training_arguments


def test_religion_mask_templates_fill_for_actiend_activation():
    """Regression: religion uses [MASK]; ACTIEND must not require [PRONOUN]."""
    from gradiend.trainer.core.signals import Signal
    from gradiend.trainer.text.prediction.dataset import (
        TextActivationTrainingDataset,
        _filled_prediction_from_template,
    )

    class MockTokenizer:
        mask_token = "[MASK]"
        mask_token_id = 103
        pad_token_id = 0
        unk_token_id = 1
        vocab = {
            "[MASK]": 103,
            "from": 20,
            "a": 21,
            "Christian": 26,
            "point": 22,
            "of": 23,
            "view": 24,
        }

        def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
            import torch

            tokens = str(text).split()
            ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
            out = {"input_ids": ids}
            if return_offsets_mapping:
                start = 0
                offsets = []
                for tok in tokens:
                    end = start + len(tok)
                    offsets.append((start, end))
                    start = end + 1
                out["offset_mapping"] = offsets
            if return_tensors == "pt":
                out = {k: torch.tensor([v]) for k, v in out.items()}
            return out

        def tokenize(self, text):
            return str(text).split()

        def convert_tokens_to_ids(self, token):
            if isinstance(token, list):
                return [self.vocab.get(str(t), self.unk_token_id) for t in token]
            return self.vocab.get(str(token), self.unk_token_id)

        def convert_ids_to_tokens(self, token_id):
            for tok, idx in self.vocab.items():
                if idx == int(token_id):
                    return tok
            return None

    tokenizer = MockTokenizer()
    masked = "from a [MASK] point of view"
    item = _filled_prediction_from_template(
        tokenizer,
        template=masked,
        target="Christian",
        max_length=32,
        mask_placeholder="[MASK]",
    )
    assert tokenizer.mask_token_id not in item["input_ids"].tolist()
    assert item["prediction_mask"].sum().item() >= 1
    assert TextActivationTrainingDataset.default_signal(
        Signal.activation()
    ).options["token_selector"] == "prediction"


def test_resolve_mask_placeholder_defaults_to_mask():
    from study.tasks import TaskBundle, resolve_mask_placeholder

    bundle = TaskBundle(
        task_id="religion",
        classes=["christian"],
        data_per_class={},
        merged_df=pd.DataFrame(),
        neutrals=pd.DataFrame(),
    )
    assert resolve_mask_placeholder(bundle) == "[MASK]"
    assert resolve_mask_placeholder(bundle, {"data": {"mask_placeholder": "[PRONOUN]"}}) == "[PRONOUN]"


def test_caa_subset_stratifies_by_label_class():
    rows = []
    for cls in ("christian", "muslim", "jewish"):
        for i in range(100):
            rows.append({"split": "train", "label_class": cls, "i": i})
    df = pd.DataFrame(rows)
    out = _subset_df(df, "train", max_size=60)
    counts = out["label_class"].value_counts().to_dict()
    assert set(counts) == {"christian", "muslim", "jewish"}
    assert all(c >= 20 for c in counts.values())
    assert len(out) <= 60


def test_iter_trainer_class_groups_uses_per_class_map():
    raw_a = {"trainer": object(), "split_mode": "none"}
    raw_b = {"trainer": object(), "split_mode": "none"}
    by_class = {
        "gradiend": {
            "christian": raw_a,
            "muslim": raw_a,
            "jewish": raw_b,
        }
    }
    groups = list(
        iter_trainer_class_groups(
            {"gradiend": raw_a},
            target_classes=["christian", "muslim", "jewish"],
            train_raw_by_class=by_class,
        )
    )
    assert len(groups) == 2
    by_raw = {id(g[1]): set(g[2]) for g in groups}
    assert by_raw[id(raw_a)] == {"christian", "muslim"}
    assert by_raw[id(raw_b)] == {"jewish"}


def test_resolve_encoder_metrics_tensors_all_alias():
    assert tensors_all_alias("actiend:M:tensors") == "actiend:M:all"
    assert tensors_all_alias("actiend:M:all") == "actiend:M:tensors"
    assert tensors_all_alias("caa:M:all_act_mean") is None
    results = {
        "methods": [
            {
                "method": "actiend:christian:all",
                "metrics": {
                    "roc_auc": 0.9,
                    "roc_auc_neutral": 0.9,
                    "balanced_accuracy": 0.8,
                    "cohens_d": 1.2,
                },
            }
        ]
    }
    m, src, inherited = resolve_encoder_metrics(
        results, "actiend:christian:tensors_tok_all_gate_encoder_direction"
    )
    assert inherited is True
    assert src == "actiend:christian:all"
    assert m.get("roc_auc") == 0.9


def test_roc_auc_equal_means_not_subchance():
    # Equal means + anti-rank pattern must not report ~0.28.
    scores = np.array([0.2, 0.8, 0.8, 0.2], dtype=float)
    labels = np.array(["A", "A", "B", "B"])
    assert abs(scores[labels == "A"].mean() - scores[labels != "A"].mean()) < 1e-12
    auc = _roc_auc(scores, labels, class_a="A")
    assert auc is not None and auc >= 0.5


def test_actiend_none_uses_layers_scope():
    with TemporaryDirectory() as td:
        args = build_training_arguments(
            "actiend",
            experiment_dir=td,
            split_mode="none",
            shared={
                "learning_rate": 1e-5,
                "max_steps": 1,
                "source": "alternative",
                "target": "diff",
            },
        )
    assert getattr(args, "signal_scope", None) is not None


def test_one_pole_defaults_to_min_auc_n_o_selection():
    with TemporaryDirectory() as td:
        args = build_training_arguments(
            "actiend",
            experiment_dir=td,
            split_mode="none",
            one_pole=True,
            shared={
                "learning_rate": 1e-5,
                "max_steps": 1,
                "source": "both",
                "target": "diff",
            },
        )
    assert args.convergent_metric == "min_auc_n_o"
    assert abs(float(args.convergent_score_threshold) - 0.9) < 1e-12
    assert args.convergent_mean_by_class_threshold is None


def test_pair_defaults_to_correlation_selection():
    with TemporaryDirectory() as td:
        args = build_training_arguments(
            "gradiend",
            experiment_dir=td,
            split_mode="none",
            one_pole=False,
            shared={
                "learning_rate": 1e-5,
                "max_steps": 1,
                "source": "alternative",
                "target": "diff",
            },
        )
    assert args.convergent_metric == "correlation"


def test_convergent_metric_override_wins():
    with TemporaryDirectory() as td:
        args = build_training_arguments(
            "actiend",
            experiment_dir=td,
            split_mode="none",
            one_pole=True,
            shared={
                "learning_rate": 1e-5,
                "max_steps": 1,
                "source": "both",
                "target": "diff",
                "convergent_metric": "correlation",
            },
        )
    assert args.convergent_metric == "correlation"


def test_hf_trainer_stub_tokenizer_class_binding(monkeypatch):
    """Regression: ``tokenizer = tokenizer`` in a class body is a NameError."""
    from study.stages import sae as sae_stage
    from study.config import StudyConfig

    class _Tok:
        pad_token = None
        eos_token = "<eos>"

    class _Model:
        moved_to = None

        def eval(self):
            return self

        def to(self, _device):
            self.moved_to = str(_device)
            return self

    model_kwargs = {}

    def _load_model(*_args, **kwargs):
        model_kwargs.update(kwargs)
        return _Model()

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        _load_model,
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: _Tok(),
    )
    cfg = StudyConfig(
        model_key="gpt2-small",
        task_id="induction",
        suite_id="core",
        raw={
            "model": {"hf_model": "gpt2"},
            "training": {"torch_dtype": "bfloat16"},
        },
    )
    trainer = sae_stage._hf_trainer_stub(cfg)
    assert trainer.tokenizer.eos_token == "<eos>"
    assert trainer.tokenizer.pad_token == "<eos>"
    assert trainer.get_model().base_model is not None
    assert str(model_kwargs["torch_dtype"]) == "torch.bfloat16"
    assert trainer.get_model().base_model.moved_to == "cuda"


def test_shared_backend_lrs_have_no_task_exceptions():
    """Study default: ACTIEND 5e-5 / GRADIEND 1e-4 on every task."""
    for task in (
        "gender_en",
        "emotion",
        "induction",
        "pronoun_number",
        "pronoun_person",
        "race",
        "religion",
        "ravel_country",
        "ravel_continent",
        "religion_one_pole",
    ):
        cfg = load_study_config(model="gpt2-small", task=task)
        assert abs(cfg.learning_rate(backend="actiend") - 5e-5) < 1e-12, task
        assert abs(cfg.learning_rate(backend="gradiend") - 1e-4) < 1e-12, task
        t = cfg.training
        assert "learning_rate_actiend" not in (cfg.task.get("training") or {})
        assert t.get("learning_rate_actiend") == 5e-5
        assert t.get("learning_rate_gradiend") == 1e-4


def test_pythia_actiend_lr_matches_gpt2_small():
    cfg = load_study_config(model="pythia-70m-deduped", task="language", suite="full")

    assert cfg.learning_rate(backend="gradiend") == 1e-6
    assert cfg.learning_rate(backend="actiend") == 5e-5
    assert "learning_rate_actiend" not in (cfg.model.get("training") or {})


def test_pronoun_number_backend_lrs():
    cfg = StudyConfig(
        model_key="gpt2-small",
        task_id="pronoun_number",
        suite_id="full",
        raw={
            "training": {
                "learning_rate": 1e-3,
                "learning_rate_gradiend": 1e-3,
                "learning_rate_actiend": 1e-5,
            }
        },
    )
    assert abs(cfg.learning_rate(backend="gradiend") - 1e-3) < 1e-12
    assert abs(cfg.learning_rate(backend="actiend") - 1e-5) < 1e-12
