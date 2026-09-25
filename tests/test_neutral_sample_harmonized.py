"""Every method must score Detection on the SAME neutral texts.

The neutral pool is shuffled once (seed 0) and split; every method then takes the
first ``encoder_eval_max_size`` texts of the split. GRADIEND/AGIEND used to draw a
random ``sample(n=max_size)`` inside the package, and ACTIEND/SAE used the whole
split, so the methods scored different neutral sets.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from neutral_protocol import texts_for_split
from sae_eval import fair_encoder_eval_from_trainer


class _Stop(Exception):
    pass


def _pool(n_val=30, n_test=50):
    rows = [{"text": f"val {i}", "split": "validation"} for i in range(n_val)]
    rows += [{"text": f"test {i}", "split": "test"} for i in range(n_test)]
    return pd.DataFrame(rows)


def test_iend_path_receives_the_same_head_slice_as_texts_for_split():
    pool = _pool()
    seen = {}

    class Trainer:
        config = SimpleNamespace(neutral_data=pool)

        def evaluate_encoder(self, **kw):
            seen[kw["split"]] = kw["neutral_data_df"]["text"].tolist()
            if len(seen) == 2:
                raise _Stop
            return {"encoder_df": None}

    with pytest.raises(_Stop):
        fair_encoder_eval_from_trainer(
            Trainer(), target_classes=["A"], max_size=7, backend="gradiend"
        )
    for split in ("validation", "test"):
        assert seen[split] == texts_for_split(pool, split, max_rows=7)
        assert len(seen[split]) == 7


def test_no_cap_leaves_the_package_default_untouched():
    class Trainer:
        config = SimpleNamespace(neutral_data=_pool())

        def evaluate_encoder(self, **kw):
            assert "neutral_data_df" not in kw
            raise _Stop

    with pytest.raises(_Stop):
        fair_encoder_eval_from_trainer(
            Trainer(), target_classes=["A"], max_size=None, backend="gradiend"
        )


def _record_use_cache(**call_kwargs):
    seen = []

    class Trainer:
        config = SimpleNamespace(neutral_data=_pool())

        def evaluate_encoder(self, **kw):
            seen.append(kw["use_cache"])
            if len(seen) == 2:
                raise _Stop
            return {"encoder_df": None}

    with pytest.raises(_Stop):
        fair_encoder_eval_from_trainer(
            Trainer(), target_classes=["A"], max_size=7, backend="gradiend", **call_kwargs
        )
    return seen


def test_encoder_cache_is_off_by_default_so_a_fresh_fit_never_reads_old_weights_csv():
    assert _record_use_cache() == [False, False]


def test_encoder_cache_can_be_enabled_for_both_splits_on_reload():
    assert _record_use_cache(use_cache=True) == [True, True]


def test_only_the_reload_path_enables_the_encoder_cache():
    import inspect

    from study.stages import train

    src = inspect.getsource(train)
    # Exactly one call site turns it on (the reload of an unchanged checkpoint) ...
    assert src.count("use_cache=True,") == 1
    # ... and it sits in _reload_train_once, not in a fit path.
    reload_src = inspect.getsource(train._reload_train_once)
    assert "use_cache=True," in reload_src
    for fit in (train._train_once, train._fit_cga_once, train._fit_caga_once):
        assert "use_cache=True" not in inspect.getsource(fit)
