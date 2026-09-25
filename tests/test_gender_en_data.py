from __future__ import annotations

import pandas as pd

import gender_en_data as ged


def test_load_ajibawa_name_filled_uses_load_dataset_helper(monkeypatch):
    frame = pd.DataFrame(
        {
            "masked": ["Janie knows [MASK] is right.", "Bob thinks [MASK] is wrong."],
            "label": ["M", "F"],
            "name": ["Bob", "Janie"],
            "pronoun": ["he", "she"],
            "split": ["train", "validation"],
            "template_id": [1, 2],
        }
    )
    calls = []

    def fake_load(repo_id, *, config_name, max_rows_per_split):
        calls.append((repo_id, config_name, max_rows_per_split))
        return frame

    import study.tasks as tasks

    monkeypatch.setattr(tasks, "load_hf_splits", fake_load)

    out = ged.load_ajibawa_name_filled(max_rows_per_split=10)

    assert calls == [(ged.GENTER_AJIBAWA, "n1", 10)]
    assert set(out["label_class"]) == {"M", "F"}


def test_read_geneutral_uses_load_neutral(monkeypatch):
    calls = []

    def fake_load_neutral(*, hf_id, max_rows, **kwargs):
        calls.append((hf_id, max_rows))
        return pd.DataFrame({"text": ["hello world"]})

    import study.tasks as tasks

    monkeypatch.setattr(tasks, "load_neutral", fake_load_neutral)

    df = ged.read_geneutral(max_size=1)
    assert calls == [(ged.BIASNEUTRAL_AJIBAWA, 1)]
    assert df["text"].tolist() == ["hello world"]
